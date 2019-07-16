import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyNN.utility.plotting import Figure, Panel, comparison_plot
from pyNN.connectors import FromListConnector

from warnings import warn

from .backend import Backend


class pynn_Backend(Backend):

    def __init__(self, runtime=10):
        super(Backend, self).__init__()
        self.brick_population_map = {}
        self.node_neuron_map = {}
        self.simulator = 'brian'
        self.spike_value = 1.00
        self.no_decay = 1000000000
        self.min_delay = 10.00
        self.runtime = runtime * self.min_delay
        #self.runtime = 300

    def _generate_pynn_model(self, fugu_scaffold, per_brick_population=False):
        if self.simulator == 'brian':
            # PyNN only has support for brian1 (i.e. brian) which is only python 2.x compatible
            assert sys.version_info <= (3,0)
            import pyNN.brian as pynn_sim
        else:
            raise ValueError("unsupported pyNN backend")

        fugu_circuit = fugu_scaffold.circuit
        fugu_graph = fugu_scaffold.graph

        if per_brick_population:
            warn("Fugu currently only supports generating a single PyNN population for the entire Fugu circuit")

            brick_to_population = {}

            for node, vals in fugu_circuit.nodes.data():
                # generate PyNN population for fugu brick
                brick_to_population[node] = population
                if 'layer' in vals:
                    if val['layer'] == 'input':
                        pass
                    if val['layer'] == 'output':
                        pass
        else:
            pynn_sim.setup(timestep=0.01)
            input_population_size = 0
            main_population_size = 0

            input_nodes = set()

            for node, vals in fugu_circuit.nodes.data():
                if 'layer' in vals: 
                    if vals['layer'] == 'input':
                        output_list = fugu_circuit.nodes[node]['output_lists'][0]
                        #for output_list in fugu_circuit.nodes[node]['output_lists']:
                        for neuron in output_list:
                            self.node_neuron_map[neuron] = input_population_size
                            input_nodes.add(neuron)

                            #if 'p': # probabiity of firing
                            input_population_size += 1

            total_neurons = len(fugu_graph.nodes)
            input_neurons = pynn_sim.Population(input_population_size, 
                                                pynn_sim.SpikeSourceArray(spike_times=[self.min_delay]),
                                                label='input')

            initial_values = [] # this is really dumb

            parameters = []
            parameters.append('v_thresh')
            parameters.append('v_rest')
            parameters.append('v_reset')
            parameters.append('tau_m')
            parameters.append('tau_refrac')
            parameter_values = {key:[] for key in parameters} 
            pynn_index = 0

            for node, vals in fugu_graph.nodes.data():
                if node not in input_nodes:
                    threshold = vals['threshold']
                    parameter_values['v_thresh'].append(self.spike_value * threshold if threshold > 0.0 else 0.05)
                    parameter_values['tau_refrac'].append(self.min_delay)
                    rest = 0.0
                    if 'potential' in vals:
                        rest = vals['potential']
                    parameter_values['v_rest'].append(rest)
                    parameter_values['v_reset'].append(0.0)

                    if 'decay' in vals:
                        parameter_values['tau_m'].append(vals['decay'] if vals['decay'] > 0.0 else self.no_decay)
                    else:
                        parameter_values['tau_m'].append(self.no_decay)

                    initial_values.append(0.0)

                    self.node_neuron_map[node] = pynn_index
                    pynn_index += 1

            main_neurons = pynn_sim.Population(total_neurons - input_population_size, 
                                               pynn_sim.IF_curr_alpha(**parameter_values),
                                               initial_values={'v':initial_values},
                                               label='main')
            """
            print("Neurons:")
            print("name,\ttype,\tid,\tv_rest,\tv_reset,\tv_thresh")
            for node in self.node_neuron_map:
                if node in input_nodes:
                    node_type = 'input'
                else:
                    node_type = 'main'
                node_id = self.node_neuron_map[node]
                neuron = main_neurons[node_id]
                print("{},\t{},\t{},\t{},\t{},\t{}".format(node,
                                                           node_type, 
                                                           node_id, 
                                                           neuron.v_rest, 
                                                           neuron.v_reset, 
                                                           neuron.v_thresh))
            print("<<<")
            """

            input_input = []
            input_to_main = []
            main_connections = []
            for u, v, values in fugu_graph.edges.data():
                weight = 0.0
                delay = self.min_delay
                if 'weight' in values:
                    weight = values['weight'] * 1.0
                if 'delay' in values:
                    delay = values['delay'] + 1.0
                    delay = delay * self.min_delay
                if u in input_nodes: 
                    if v in input_nodes:
                        input_input.append((self.node_neuron_map[u], self.node_neuron_map[v], weight, delay))
                    else:
                        input_to_main.append((self.node_neuron_map[u], self.node_neuron_map[v], weight, delay))
                else:
                    main_connections.append((self.node_neuron_map[u], self.node_neuron_map[v], weight, delay))

            synapse = pynn_sim.StaticSynapse()

            if len(input_input) > 0:
                input_self  = FromListConnector(input_input)
                input_self_synapses = pynn_sim.Projection(input_neurons, input_neurons, input_self, synapse, label='input-input')

            if len(input_to_main) > 0:
                input_connector = FromListConnector(input_to_main)
                input_synapses = pynn_sim.Projection(input_neurons, main_neurons, input_connector, synapse, label='input-main')

            if len(main_connections) > 0:
                main_connector = FromListConnector(main_connections)
                main_synapses = pynn_sim.Projection(main_neurons, main_neurons, main_connector, synapse, label='main-main')

            """
            print("<<<")
            print("Synapses:")
            print("input to main---")
            print("From,\tTo,\tWeight\tDelay")
            for synapse in input_synapses.get(["weight", "delay"], format="list"):
                print(synapse)
            print("main to main---")
            print("From,\tTo,\tWeight\tDelay")
            for synapse in main_synapses.get(["weight", "delay"], format="list"):
                print(synapse)
            """

            main_neurons.record(['spikes','v'])
            input_neurons.record(['spikes'])

            pynn_sim.run(self.runtime)

            main_data = main_neurons.get_data()
            input_data = input_neurons.get_data()

            pynn_sim.end()

            spike_result = pd.DataFrame({'time':[],'neuron_number':[]})

            signals = main_data.segments[0].filter(name='v')[0]
            main_spiketrains = main_data.segments[0].spiketrains
            input_spiketrains = input_data.segments[0].spiketrains

            #print("Internal spike train results")
            # sort spike trains
            spikes = {}
            for node in self.node_neuron_map:
                pynn_index = self.node_neuron_map[node]
                if node not in input_nodes:
                    spiketrain = main_spiketrains[pynn_index]
                else:
                    spiketrain = input_spiketrains[pynn_index]
                if spiketrain.any():
                    #print("Spike train: {}\tNeuron: {}".format(spiketrain, node))
                    for time in np.array(spiketrain):
                        if time not in spikes:
                            spikes[time] = set()
                        spikes[time].add(fugu_graph.nodes[node]['neuron_number'])

            for time in spikes:
                mini_df = pd.DataFrame()
                times = [time for i in spikes[time]]
                mini_df['time'] = times
                mini_df['neuron_number'] = list(spikes[time])
                spike_result = spike_result.append(mini_df, sort=False)

            #print(input_data.segments[0].spiketrains)
            #print(main_data.segments[0].spiketrains)
            #print(signals.annotations)
            #print(len(signals))
            #print(dir(signals))
            #print(signals.array_annotations)
            #plt.plot(signals[2])
            Figure(Panel(signals, ylabel="Membrane potential (mV)", yticks=True))
            plt.show()

            #print(vs)
            #print(len(main_data.segments))
            #print(main_data.segments[0].all_data)
            #print(vs)
            #Figure( Panel(signals, ylabel="Membrane potential (mV)", yticks=True, xticks=True))
            return spike_result

    def stream(self,
               scaffold,
               input_values,
               stepping,
               record,
               backend_args):
        warn("Stepping not supported yet.  Use a batching mode.")
        return None
    
    def batch(self,
             scaffold,
             input_values,
             n_steps,
             record,
             backend_args):
        # generate neurons
        per_brick_population = False
        if 'pynn_backend' in backend_args:
            self.simulator = backend_args['pynn_backend']
        if 'per_brick_population' in backend_args:
            per_brick_population = backend_args['per_brick_population']
        spike_result = self._generate_pynn_model(scaffold, per_brick_population)
        spike_result = spike_result.sort_values('time')
        return spike_result 
