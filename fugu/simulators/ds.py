from __future__ import print_function

import numpy as np
import scipy as sp
import multiprocessing
import networkx as nx
import torch
from collections import deque
from multiprocessing import Queue
from multiprocessing import Process, Value, Array

_IF_TYPE = 1
_LEAKY_TYPE = 2
_STOCHASTIC_TYPE = 3
_RECORD_SPIKES = 4
_RECORD_POTENTIAL = 5
_RECORD_PREACTIVATION = 6

#mode = 'cpu'
#num_workers = 3

def check_type(neuron_type, check_type):
    str_type = format(neuron_type, 'b')
    if check_type > len(str_type):
        return False
    else:
        str_check = str_type[-check_type]
        return str_check is '1'

def set_types(types):
    return np.sum([2**(t-1) for t in types])

def check_delay(graph):
    return graph.graph['has_delay']

def remove_delay(graph, verbose = 0):
    max_delay = np.max([graph[edge[0]][edge[1]]['delay'] for edge in graph.edges()])
    iteration = 0
    while max_delay > 1:
        iteration = iteration+1
        for i,edge in enumerate(list(graph.edges())):
            if verbose > 0:
                print(str(i) + ' on iteration ' + str(iteration))
            from_neuron = edge[0]
            to_neuron = edge[1]
            old_delay = graph[from_neuron][to_neuron]['delay']
            if old_delay > 1:
                to_add_neuron = graph.number_of_nodes()
                graph.add_node(to_add_neuron)
                graph.node[to_add_neuron]['potential'] = 0.0
                graph.node[to_add_neuron]['threshold'] = 0.5
                graph.add_edge(from_neuron, to_add_neuron, weight=1.0, delay=1)
                graph.add_edge(to_add_neuron, to_neuron, weight=graph[from_neuron][to_neuron]['weight'], delay=old_delay-1)
                graph.remove_edge(edge[0], edge[1])
        max_delay = np.max([graph[edge[0]][edge[1]]['delay'] for edge in graph.edges()])
    graph.graph['has_delay'] = False
    return graph

def restore_delay(graph):
    done = False
    while not done:
        edges_list = list(graph.edges())
        for edge in edges_list:
            if graph.out_degree(edge[1])==1 and graph[edge[0]][edge[1]]['weight'] > graph.node[edge[1]]['threshold']:
                target_neuron = list(graph.neighbors(edge[1]))[0]
                first_edge = graph[edge[0]][edge[1]].copy()
                next_edge = graph[edge[1]][target_neuron].copy()
                graph.remove_edge(edge[0], edge[1])
                graph.remove_edge(edge[1], target_neuron)
                graph.remove_node(edge[1])
                graph.add_edge(edge[0],
                               target_neuron,
                               weight=next_edge['weight'],
                               delay=first_edge['delay']+next_edge['delay'])
                break
        else:
            done=True
    graph.graph['has_delay'] = True
    return graph



def SpikingPDE_to_ds_graph(graph, neuron_list):
    graph_to_return = graph.copy()
    for neuron in neuron_list:
        node = graph_to_return.node[neuron.name]
        for var in vars(neuron):
            node[var] = vars(neuron)[var]
    graph_to_return.graph['has_delay'] = False
    return graph_to_return

#Not complete/Not needed
def initialize_worker_blocks(graph, mode='cpu', batch_size = 1000, num_workers=1):
    neuron_list = list(graph.nodes())
    neuron_type_dict = dict()
    for neuron in neuron_list:
        types = [_IF_TYPE]
        if 'decay' in graph.nodes[neuron] and graph.nodes[neuron]['decay']>0:
            types.append(_LEAKY_TYPE)
        if 'p' in graph.nodes[neuron] and graph.nodes[neuron]['p']< 1.0:
            types.append(_STOCHASTIC_TYPE)
        neuron_type = set_types(types)
        if neuron_type in neuron_type_dict.keys():
            neuron_type_dict[neuron_type].append(neuron)
        else:
            neuron_type_dict[neuron_type] = [neuron]
    job_blocks = []
    for neuron_type in neuron_type_dict.keys():
        neurons = neuron_type_dict[neuron_type]
        for start_idx in range(0, len(neurons), batch_size):
            stop_idx = np.min([start_idx+batch_size, len(neurons)])
            job_blocks.append((len(job_blocks), neuron_type, neurons[start_idx:stop_idx]))

    worker_blocks = [[]]*num_workers
    for i, job in enumerate(job_blocks):
        worker_blocks[i%num_workers].append(job)

    return worker_blocks

def separate_types(graph, batch_size = 1000):
    neuron_list = list(graph.nodes())
    neuron_type_dict = dict()
    for neuron in neuron_list:
        types = [_IF_TYPE]
        if 'decay' in graph.nodes[neuron] and graph.nodes[neuron]['decay']<1.0:
            types.append(_LEAKY_TYPE)
        if 'p' in graph.nodes[neuron] and graph.nodes[neuron]['p']< 1.0:
            types.append(_STOCHASTIC_TYPE)
        if 'record' in graph.nodes[neuron]:
            to_record = graph.nodes[neuron]['record']
            if 'spikes' in to_record:
                types.append(_RECORD_SPIKES)
            if 'potential' in to_record:
                types.append(_RECORD_POTENTIAL)
            if 'preactivation' in to_record:
                types.append(_RECORD_PREACTIVATION)
        neuron_type = set_types(types)
        if neuron_type in neuron_type_dict.keys():
            neuron_type_dict[neuron_type].append(neuron)
        else:
            neuron_type_dict[neuron_type] = [neuron]
    job_blocks = []
    for neuron_type in neuron_type_dict.keys():
        neurons = neuron_type_dict[neuron_type]
        for start_idx in range(0, len(neurons), batch_size):
            stop_idx = np.min([start_idx+batch_size, len(neurons)])
            job_blocks.append((len(job_blocks), neuron_type, neurons[start_idx:stop_idx]))
    return job_blocks

def initialize_neuron_states(graph, neuron_list, neuron_type, mode='cpu'):
    if mode is 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)
    elif mode is 'cuda':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    states = dict()
    states['type'] = neuron_type
    if check_type(neuron_type,_IF_TYPE):
        threshold = torch.tensor([graph.nodes[neuron]['threshold'] for neuron in neuron_list]).float()
        potential = torch.tensor([graph.nodes[neuron]['potential'] for neuron in neuron_list]).float()
        states['threshold'] = threshold
        states['potential'] = potential
    if check_type(neuron_type, _LEAKY_TYPE):
        decay = torch.tensor([graph.nodes[neuron]['decay'] for neuron in neuron_list]).float()
        states['decay'] = decay
    if check_type(neuron_type, _STOCHASTIC_TYPE):
        p = torch.tensor([graph.nodes[neuron]['p'] for neuron in neuron_list]).float()
        states['p'] = p
    return states

def get_neuron_tensors(graph, batch_size=1000, verbose=0, mode='cpu'):
    neuron_tensors = dict()
    if mode is 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)
    elif mode is 'cuda':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        print("Invalid Mode")
        exit()
    if 'has_delay' in graph.graph and graph.graph['has_delay']:
        if verbose>0:
            print("Removing delays")
        graph = remove_delay(graph, verbose=verbose)
    for i, neuron_group in enumerate(separate_types(graph, batch_size=batch_size)):
        neuron_tensors[i] = initialize_neuron_states(graph, neuron_group[2], neuron_group[1], mode=mode)
        if mode is 'cpu':
            neuron_tensors[i]['index'] = torch.tensor(neuron_group[2], dtype=torch.long)
        elif mode is 'cuda':
            neuron_tensors[i]['index'] = torch.tensor(neuron_group[2], dtype=torch.long)
        weights = torch.zeros((len(graph.nodes()), len(neuron_group[2])))
        for j, neuron in enumerate(neuron_group[2]):
            in_edges = graph.in_edges(neuron)
            for in_edge in in_edges:
                in_neighbor = in_edge[0]
                this_node = in_edge[1]
                weights[in_neighbor, j] = graph[in_neighbor][this_node]['weight']
        neuron_tensors[i]['weight'] = weights
    return neuron_tensors

def step_simulation(neuron_tensors, spikes = None, injection=None, watches=[], verbose=0):
    number_of_neurons = np.sum([neuron_tensors[ten_num]['index'].numel() for ten_num in neuron_tensors])
    new_spikes = torch.zeros((number_of_neurons,))
    for i, ten_num in enumerate(neuron_tensors):
        n_tensor = neuron_tensors[ten_num]
        neuron_type = n_tensor['type']
        #Integrate Injection
        if injection is not None:
            injection = injection.to(device=n_tensor['potential'].device)
            n_tensor['potential'] = n_tensor['potential'] + injection[n_tensor['index']]
        #Integrate Spikes
        if spikes is not None:
            n_tensor['potential'] = n_tensor['potential'] + torch.matmul(spikes,n_tensor['weight'])
        if check_type(neuron_type, _RECORD_PREACTIVATION):
            if 'preactivation_history' not in n_tensor:
                if verbose > 0:
                    print("Creating Preactivation History for tensor " + str(ten_num) + '.')
                    n_tensor['preactivation_history'] = deque()
            n_tensor['preactivation_history'].append((len(n_tensor['preactivation_history']),
                                                          n_tensor['potential'].clone() ))
        #Threshold
        if check_type(neuron_type, _STOCHASTIC_TYPE):
            to_reset = (n_tensor['potential'] > n_tensor['threshold']).byte()
            where_spike = to_reset*torch.bernoulli(n_tensor['p']).byte()
            n_tensor['potential'][to_reset] = 0
        else:
            where_spike = n_tensor['potential'] > n_tensor['threshold']
        new_spikes[torch.masked_select(n_tensor['index'], where_spike)] = 1
        #Send spikes
        #Decay
        if check_type(neuron_type, _LEAKY_TYPE):
            n_tensor['potential'] = n_tensor['potential']*(1-n_tensor['decay'])
            n_tensor['potential'][where_spike] = 0
        else:
            n_tensor['potential'][:] = 0
        if check_type(neuron_type, _RECORD_SPIKES):
            if 'spike_history' not in n_tensor:
                if verbose > 0:
                    print("Creating Spike History for tensor " + str(ten_num) + ".")
                n_tensor['spike_history'] = deque()
            n_tensor['spike_history'].append((len(n_tensor['spike_history']),
                                              torch.masked_select(n_tensor['index'], where_spike)))
        if check_type(neuron_type, _RECORD_POTENTIAL):
            if 'potential_history' not in n_tensor:
                n_tensor['potential_history'] = deque()
            n_tensor['potential_history'].append((len(n_tensor['potential_history']),
                                                  n_tensor['potential'].clone()))
    for watch in watches:
        func = getattr(watch, 'on_end_timestep', None)
        if callable(func):
            watch.on_end_timestep(neuron_tensors, new_spikes)
    return neuron_tensors, new_spikes

def run_simulation(graph, timesteps, injection_dictionary, start_time=0,batch_size=1000,watches=[], verbose = 0, mode='cpu'):
    neuron_tensors = get_neuron_tensors(graph, batch_size=batch_size, mode=mode)
    spikes = None #torch.zeros_like(neuron_tensors['index'])
    injection = None
    if verbose > 0:
        print("Starting Simualation for " + str(timesteps) + ' timesteps.')
    for t in range(start_time, start_time+timesteps):
        if verbose > 0:
            print ("{0:.2f}".format(100*t/timesteps) + '% complete.', end='\r')
        if t in injection_dictionary:
            injection = injection_dictionary[t]
            if injection.numel() < graph.number_of_nodes():
                expanded_injection = torch.zeros(graph.number_of_nodes())
                expanded_injection[0:injection.numel()] = injection
                injection = expanded_injection
        else:
            injection = None
        neuron_tensors, spikes = step_simulation(neuron_tensors, spikes, injection, watches, verbose=verbose)
    if verbose > 0:
        print("")
        print('Done.')
    return neuron_tensors

def get_injection_tensor_from_initial_walkers(initial_dictionary,graph, num_neurons, mode='cpu'):
    if mode is 'cpu':
        torch.set_default_tensor_type(torch.FloatTensor)
    elif mode is 'cuda':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        print("Invalid mode")
        exit()
    target_tensor = torch.zeros(num_neurons,)
    for target in initial_dictionary:
        value = initial_dictionary[target]
        for i, node in enumerate(graph.nodes):
            if 'groups' in graph.nodes[node]:
                groups = graph.nodes[node]['groups']
            else:
                groups = []
            if str(target) in groups and 'counter' in groups:
                target_idx = graph.nodes[node]['name']
                target_tensor[target_idx] = -value
    return target_tensor
