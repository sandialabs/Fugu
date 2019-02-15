import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
default_brick_dimensionality = {
    'input_shape' : [()],
    'output_shape' : [()],
    'D' : 0,
    'layer' : 'output',
    'input_coding' : 'unknown',
    'output_coding' : 'unknown'
}

input_coding_types = ['current',
                      'unary-B',
                      'unary-L',
                      'binary-B',
                      'binary-L',
                     'temporal-B',
                     'temporal-L',
                      'Raster',
                      'Population',
                      'Rate',
                     'Undefined']

class Scaffold:
    supported_backends = ['ds']
    def __init__(self):
        self.circuit = nx.DiGraph()
        self.pos = {}
        self.count = {}
        self.graph = None
        self.is_built = False
    def add_brick(self, brick_function, input_nodes=[-1], dimensionality = None, name=None, output=False):
        if name is None and brick_function.name is not None:
            name = brick_function.name
        elif name is None:
            brick_type = str(type(brick_function))
            if brick_type not in self.count:
                self.count[brick_type] = 0
            name = brick_type+"_" + str(self.count[brick_type])
            self.count[brick_type] = self.count[brick_type]+1
        elif name in self.circuit.nodes:
            raise ValueError("Node name already used.")

        if brick_function.name is None:
            brick_function.name = name

        node_number = self.circuit.number_of_nodes()
        self.circuit.add_node(node_number,
                             brick=brick_function)#,dimensionality=dimensionality)

        #Make sure we're working with a list of inputs
        if type(input_nodes) is not list:
            input_nodes = [input_nodes]

        #Make sure our inputs are integer formatted
        input_nodes = [-2 if node is 'input' else node for node in input_nodes]

        #Replace -1 with last node
        input_nodes = [node_number-1 if node==-1 else node for node in input_nodes]

        #Create tuples for (node, channel) if not already done
        input_nodes = [(node, 0) if type(node) is int else node for node in input_nodes]

        #Process inputs
        for node in [node[0] for node in input_nodes]:
            if node <-1:
                self.circuit.node[node_number]['layer'] = 'input'
            else:
                self.circuit.add_edge(node, node_number)
        self.circuit.nodes[node_number]['input_nodes'] = input_nodes
        if output:
            self.circuit.node[node_number]['layer'] = 'output'

    def resolve_timing(self):
        # Set weights to equal source node timing (T_out + D?)

        # From end to front
        #     Identify predecessor nodes; if 1, then pass, if >1, then compute longest path to each source
        #     If longest path size is different, then add delay node to shorter one to make equivalent.

        # Have to define D as "time from first input within T_in to first output within T_out.  Have to assume T_in(2)= T_out(1)
        nodes = list(self.circuit.nodes())
        edges = list(self.circuit.edges())
        for edge in edges:
            self.circuit.edges[edge[0], edge[1]]['weight']=self.circuit.node[edge[0]]['brick'].dimensionality['D']

        for node in reversed(nodes):

            # loop backwards through nodes
            # Find predecessor node for 'node'
            pred_nodes=list(self.circuit.predecessors(node))
            print(node, list(pred_nodes), len(list(pred_nodes)), len(pred_nodes))

            if len(list(pred_nodes))<2:
                pass
            else:
                distances=[]
                for target_node in list(pred_nodes):
                    distance_guess=0
                    target_paths=nx.all_simple_paths(self.circuit, 0, target_node)
                    for path in map(nx.utils.pairwise, target_paths):
                        path_lengths=self.get_weight(path)
                        if path_lengths>distance_guess: distance_guess=path_lengths

                    distance_guess=distance_guess+self.circuit.edges[target_node, node]['weight']
                    print(target_node, distance_guess)
                    distances.append(distance_guess)

                # Now, we need to add delay nodes to paths less than longest distance

                max_value=max(distances)
                max_index=distances.index(max_value)
                print(list(distances), max_index, max_value)

                for i in range(0, len(pred_nodes)):
                    # Check if this path needs a delay node
                    if(distances[i]<max_value):
                        target_node=pred_nodes[i]
                        print('Adding delay node of length ', max_value-distances[i], 'between ', target_node, ' and ', node)
                        N_delay=self.circuit.nodes[target_node]['N_out']
                        self.circuit.add_node(target_node+0.5, brick=Delay, N_in=N_delay, N_out=N_delay, T_in=1, T_out=1, D=max_value-distances[i], layer='delay')
                        self.pos[target_node+0.5]=np.array([target_node+0.5, self.pos[target_node][1]])
                        self.circuit.remove_edge(target_node, node)
                        self.circuit.add_edge(target_node, target_node+0.5, weight=self.circuit.nodes[target_node]['D'])
                        self.circuit.add_edge(target_node+0.5, node, weight=self.circuit.nodes[target_node+0.5]['D'])

    def get_weight(self, path):
        total_len=0
        for i in list(path):
            total_len=total_len+self.circuit.edges[i]['weight']

        return total_len

    def all_nodes_built(self, verbose=0):
        b = True
        for node in self.circuit.nodes:
            b = b and self.circuit.nodes[node]['brick'].is_built
        if verbose>0:
            print("Nodes built:")
            for node in self.circuit.nodes:
                print(str(node) + ":" + str(self.circuit.nodes[node]['brick'].is_built))
        return b

    def all_in_neighbors_built(self, node):
        in_neighbors = [edge[0] for edge in self.circuit.in_edges(nbunch=node)]
        b = True
        for neighbor in in_neighbors:
            b = b and self.circuit.nodes[neighbor]['brick'].is_built
        return b

    def lay_bricks(self, verbose=0):
        built_graph = nx.DiGraph()
        # Handle Input Nodes
        if verbose>0:
            print("Laying Input Bricks.")
        for node in [node for node in self.circuit.nodes
         if 'layer' in self.circuit.nodes[node]
         and self.circuit.nodes[node]['layer'] is 'input']:
            (built_graph,
             dimensionality,
             complete_node,
             output_lists,
             output_codings) = self.circuit.nodes[node]['brick'].build(built_graph,
                              None,
                              None,
                              None,
                              None)
            self.circuit.nodes[node]['output_lists'] = output_lists
            self.circuit.nodes[node]['output_codings'] = output_codings
            self.circuit.nodes[node]['dimensionality'] = dimensionality
            self.circuit.nodes[node]['complete_node'] = complete_node
            if verbose>0:
                print("Completed: " + str(node))
        while not self.all_nodes_built(verbose=verbose):
            #Over unbuilt, ready edges
            for node in [node for node in self.circuit.nodes
                         if (not self.circuit.nodes[node]['brick'].is_built)
                         and self.all_in_neighbors_built(node)]:
                inputs = {}
                if verbose>0:
                    print('Laying Brick: ' + str(node))
                for input_number in range(0,len(self.circuit.nodes[node]['input_nodes'])):
                    if verbose>0:
                        print("Processing input: " + str(input_number))
                    inputs[input_number] = {'input_node':self.circuit.nodes[node]['input_nodes'][input_number][0],
                                           'input_channel':self.circuit.nodes[node]['input_nodes'][input_number][1]}
                dimensionality = [self.circuit.nodes[inputs[key]['input_node']]['dimensionality']
                                  for key in inputs]
                complete_node = [self.circuit.nodes[inputs[key]['input_node']]['complete_node'][inputs[key]['input_channel']] for key in inputs]
                input_lists = [self.circuit.nodes[inputs[key]['input_node']]['output_lists'][inputs[key]['input_channel']] for key in inputs]
                input_codings = [self.circuit.nodes[inputs[key]['input_node']]['output_codings'][inputs[key]['input_channel']] for key in inputs]
                (built_graph,
                 dimensionality,
                 complete_node,
                 output_lists,
                 output_codings) = self.circuit.nodes[node]['brick'].build(built_graph,
                                                      dimensionality,
                                                      complete_node,
                                                      input_lists,
                                                      input_codings)
                self.circuit.nodes[node]['dimensionality'] = dimensionality
                self.circuit.nodes[node]['output_codings'] = output_codings
                self.circuit.nodes[node]['output_lists'] = output_lists
                self.circuit.nodes[node]['complete_node'] = complete_node
                if verbose>0:
                    print("Complete.")
        self.is_built=True
        self.graph = built_graph
        return built_graph

    def _create_ds_injection(self):
        #find input nodes
        import torch
        input_nodes = [node  for node in self.circuit.nodes if ('layer' in self.circuit.nodes[node] )
                       and (self.circuit.nodes[node]['layer'] is 'input')]
        injection_tensors = {}
        for input_node in input_nodes:
            input_spikes = self.circuit.nodes[input_node]['brick'].get_input_value()
            for t in range(0, input_spikes.shape[-1]):
                if t not in injection_tensors:
                    injection_tensors[t] = torch.zeros((self.graph.number_of_nodes(),)).float()
                for local_idx, neuron in enumerate(self.circuit.nodes[input_node]['output_lists'][0]):
                    tensor_idx = list(self.graph.nodes).index(neuron)
                    injection_tensors[t][tensor_idx] += input_spikes[local_idx,t]
        return injection_tensors

    def evaluate(self, max_runtime=10, backend='ds', record_all=False):
        if backend not in Scaffold.supported_backends:
            raise ValueError("Backend " + str(backend) + " not supported.")
        if backend is 'ds':
            from ds import run_simulation
            injection_values = self._create_ds_injection()
            for node in self.circuit.nodes:
                if 'layer' in self.circuit.nodes[node] and self.circuit.nodes[node]['layer'] is 'output':
                    for o_list in self.circuit.nodes[node]['output_lists']:
                        for neuron in o_list:
                            self.graph.nodes[neuron]['record'] = ['spikes']
            ds_graph = nx.convert_node_labels_to_integers(self.graph)
            ds_graph.graph['has_delay']=True
            if record_all:
                for neuron in ds_graph.nodes:
                    ds_graph.nodes[neuron]['record'] = ['spikes']
            for neuron in ds_graph.nodes:
                if 'potential' not in ds_graph.nodes[neuron]:
                    ds_graph.nodes[neuron]['potential'] = 0.0
            results = run_simulation(ds_graph,
                                     max_runtime,
                                     injection_values)
            spike_result = {}
            for group in results:
                if 'spike_history' in results[group]:
                    spike_history = results[group]['spike_history']
                    for entry in spike_history:
                        if entry not in spike_result:
                            spike_result[entry[0]] = []
                        spike_result[entry[0]].extend(entry[1].tolist())
        return spike_result



class Brick(ABC):
    def __init__(self):
        self.name = "Empty Brick"
        self.supported_codings = []

    @abstractmethod
    def build(self, graph,
                   dimensionality,
                   complete_node,
                   input_lists,
                   input_codings):
        pass
class InputBrick(Brick):

    @abstractmethod
    def get_input_value(self, t=None):
        pass

class Spike_Input(InputBrick):
    def __init__(self, spikes, time_dimension = False,
                 coding='Undefined', name=None):
        self.vector = spikes
        self.coding = coding
        self.time_dimension = time_dimension
        self.is_built = False
        self.name = name
        self.supported_codings = []
        self.dimensionality = {'D' : 0}

    def get_input_value(self, t=None):
        if t is None:
            return self.vector
        else:
            assert type(t_range) is int
            return self.vector[...,t:t+1][...,-1]


    def build(self, graph,
             dimensionality,
             complete_node,
             input_lists,
             input_codings):
        if not self.time_dimension:
            self.vector = np.expand_dims(self.vector,
                                         len(self.vector.shape))

        output_lists = [[]]
        for i, index in enumerate(np.ndindex(self.vector.shape[:-1])):
            neuron_name = self.name+"_" + str(i)
            graph.add_node(neuron_name,
                           index=index,
                           threshold=0.0,
                           decay=0.0,
                           p=1.0)
            output_lists[0].append(neuron_name)
        output_codings = [self.coding]
        complete_node = self.name+"_complete"
        #outputs = [{'neurons':output_lists,
        #            'shape':self.vector.shape[:-1],
        #            'coding':output_codings,
        #            'complete_node':complete_node}]
        graph.add_node(complete_node,
                      index = -1,
                      threshold = 0.0,
                      decay = 0.0,
                      p=1.0,
                      potential = 0.5)
        self.is_built = True
        return (graph, {'output_shape':[self.vector.shape],
                        'output_coding':self.coding,
                        'layer' : input,
                        'D':0},
               [complete_node],
               output_lists,
               output_codings)

class Threshold(Brick):
    def __init__(self, threshold, decay=0.0, p=1.0, name=None, output_coding=None):
        super(Brick, self).__init__()
        self.is_built=False
        self.dimensionality = {'D':0}
        self.name = name
        self.p = p
        self.decay = decay
        self.threshold = threshold
        self.output_coding=output_coding
        self.supported_codings = ['current', 'Undefined']
    def build(self,
             graph,
             dimensionality,
             complete_node,
             input_lists,
             input_codings):
        if len(input_codings)!=1:
            raise ValueError("Only one input is permitted.")
        if input_codings[0] not in self.supported_codings:
            raise ValueError("Input coding not supported. Expected: "
                             + str(self.supported_codings)
                             + " Found: "
                             + str(input_codings[0]))
        graph.add_node(self.name,
                       threshold=self.threshold,
                      decay=self.decay,
                      p=self.p)
        for edge in input_lists[0]:
            graph.add_edge(edge['source'],
                           self.name,
                           weight=edge['weight'],
                           delay=edge['delay'])
        output_lists = [[self.name]]
        if self.output_coding is None:
            output_codings = ['Unary']
        else:
            output_codings = [self.output_coding]
        self.is_built = True
        return (graph,
               self.dimensionality,
                complete_node,
                output_lists,
                output_codings
               )

class Dot(Brick):
    def __init__(self, weights, name=None):
        super(Brick, self).__init__()
        self.is_built = False
        self.dimensionality = {'D':1}
        self.name = name
        self.weights = weights
        self.supported_codings = ['Raster', 'Undefined']
    def build(self,
             graph,
             dimensionality,
             complete_node,
             input_lists,
             input_codings):
        output_list = []
        output_codings = ['current']
        if len(input_codings)>1:
            raise ValueError("Only one input is permitted.")
        if input_codings[0] not in self.supported_codings:
            raise ValueError("Input coding not supported. Expected: "
                             + str(self.supported_codings)
                             + " Found: "
                             + str(input_codings[0]))
        if len(input_lists[0]) != len(self.weights):
            raise ValueError("Input length does not match weights. Expected: "
                            + str(len(self.weights))
                            + " Found: "
                            + str(len(input_lists[0])))
        for i, weight in enumerate(self.weights):
            output_list.append({'source':input_lists[0][i],
                                'weight':weight,
                                'delay':1
                               })
        if type(dimensionality) is list:
            dimensionality = dimensionality[0]
        dimensionality['D'] = dimensionality['D'] + 1
        graph.add_node(self.name+"_complete",
                      threshold=0.5,
                      potential=0.0,
                      decay = 0.0,
                      index = -1,
                      p=1.0)
        graph.add_edge(complete_node[0],
                       self.name+"_complete",
                       weight=1.0,
                       delay=1)
        self.is_built = True
        return (graph,
                dimensionality,
                self.name+"_complete",
                [output_list],
                output_codings)

class Copy(Brick):
    def __init__(self, name=None):
        super(Brick, self).__init__()
        self.is_built=False
        self.dimensionality = {'D':1}
        self.name = name
        self.supported_codings = ['unary-B',
                      'unary-L',
                      'binary-B',
                      'binary-L',
                     'temporal-B',
                     'temporal-L',
                       'Raster',
                     'Undefined']
    def build(self, graph,
              dimensionality,
             complete_node,
             input_lists,
             input_codings):
        num_copies = 2
        if type(dimensionality) is list:
            self.dimensionality = dimensionality[0]
        else:
            self.dimensionality = dimensionality
        self.dimensionality['D'] = 1
        #self.dimensionality['output_coding'] = self.dimensionality['input_coding']
        #input_shape = input_lists #self.dimensionality['input_shape']
        if len(input_lists) > 1:
            print('Copying first input only!')
        #if len(input_shape) == 0:
        #    print("No input into brick")
        #    return -1
        #input_shape = input_shape[0]
        if input_codings[0] not in self.supported_codings:
            print("Unsupported input coding: " + input_codings[0])
            return -1
        output_lists = [ [] for i in range(num_copies)]
        output_codings = []
        for neuron in input_lists[0]:
            for copy_num in range(0,num_copies):
                copy_name = str(neuron)+"_copy" + str(copy_num)
                graph.add_node(copy_name, threshold=0.5, decay=0, p=1.0, index = graph.nodes[neuron]['index'])
                graph.add_edge(neuron, copy_name, weight=1.0, delay=1)
                output_lists[copy_num].append(copy_name)
        for copy_num in range(0,num_copies):
            output_codings.append(input_codings[0])
        graph.add_node(self.name+"_complete",
                       threshold = 0.5,
                       decay = 0,
                       p=1.0,
                       index=-1
                      )
        graph.add_edge(complete_node[0],
                       self.name+"_complete",
                       weight=1.0,
                       delay=1)
        self.is_built = True
        return (graph, self.dimensionality, [self.name+"_complete"]*num_copies , output_lists, output_codings)
