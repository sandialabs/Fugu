import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

from warnings import warn

from .backend import Backend

BRIAN_BACKEND = 0
SPINNAKER_BACKEND = 1

class pynn_Backend(Backend):

    def __init__(self, runtime=10):
        super(Backend, self).__init__()
        self.brick_population_map = {}
        self.node_neuron_map = {}

        # these parameters will have to be set based on the simulator
        self.no_decay = 1000000000 

        self.spike_value = 1.00
        self.min_delay = 10.00

        self.steps = runtime
        self.runtime = self.steps * self.min_delay
        self.tau_syn_E = 5.0
        self.backend = BRIAN_BACKEND 

    def _run_pynn_sim(self, fugu_scaffold, simulator='brian', per_brick_population=False, verbose=False):
        if simulator == 'brian':
            # PyNN only has support for brian1 (i.e. brian) which is only python 2.x compatible
            assert sys.version_info <= (3,0)
            import pyNN.brian as pynn_sim

            from pyNN.connectors import FromListConnector

            self.backend = BRIAN_BACKEND 

            self.spike_value = 1.00
            self.min_delay = 10.00
            self.tau_syn_E = 1.49

            self.runtime = self.steps * self.min_delay

            pynn_sim.setup(timestep=0.01)
        elif simulator == 'spinnaker' or simulator == 'spynnaker':
            assert sys.version_info <= (3,0)
            import pyNN.spiNNaker as pynn_sim
            from pyNN.spiNNaker import FromListConnector

            self.backend = SPINNAKER_BACKEND 

            #self.no_decay = 65535
            self.no_decay = 7.71762
            self.min_delay = 0.50
            self.tau_syn_E = 1.00
            self.spike_value = 1.0

            self.runtime = self.steps * self.min_delay

            pynn_sim.setup(timestep=0.50)
        else:
            raise ValueError("unsupported pyNN backend")

        fugu_circuit = fugu_scaffold.circuit
        fugu_graph = fugu_scaffold.graph

        if per_brick_population:
            warn("Fugu currently only supports generating a single PyNN population for the entire Fugu circuit")
        else:

            # Setup neuron populations
            input_population_size = 0
            main_population_size = 0

            input_nodes = set()

            for node, vals in fugu_circuit.nodes.data():
                if 'layer' in vals: 
                    if vals['layer'] == 'input':
                        output_list = fugu_circuit.nodes[node]['output_lists'][0]
                        for neuron in output_list:
                            self.node_neuron_map[neuron] = input_population_size
                            input_nodes.add(neuron)

                            input_population_size += 1

            total_neurons = len(fugu_graph.nodes)
            input_neurons = pynn_sim.Population(input_population_size, 
                                                pynn_sim.SpikeSourceArray(spike_times=[self.min_delay]),
                                                label='input')

            # Set intial conditions and parameter values for neurons
            initial_values = []

            parameters = []
            parameters.append('v_thresh')
            parameters.append('v_rest')
            parameters.append('v_reset')
            parameters.append('tau_m')
            parameters.append('tau_refrac')
            parameters.append('tau_syn_E')
            parameter_values = {key:[] for key in parameters} 
            pynn_index = 0

            for node, vals in fugu_graph.nodes.data():
                if node not in input_nodes:
                    threshold = vals['threshold']
                    parameter_values['v_thresh'].append(self.spike_value * threshold if threshold > 0.0 else 0.10)
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

                    parameter_values['tau_syn_E'].append(self.tau_syn_E)

                    #if self.backend == SPINNAKER_BACKEND:
                        #initial_values.append((node,0.0))
                    #else:
                    initial_values.append(0.0)

                    self.node_neuron_map[node] = pynn_index
                    pynn_index += 1

            main_neurons = pynn_sim.Population(total_neurons - input_population_size, 
                                               pynn_sim.IF_curr_exp(**parameter_values),
                                               label='main')
            main_neurons.initialize(v=initial_values)

            if verbose:
                print("Neurons:")
                if self.backend == SPINNAKER_BACKEND:
                    for param in parameters:
                        print("{}: {}".format(param, main_neurons.get(param)))
                else:
                    print("name,\ttype,\tid,\tv_rest,\tv_reset,\tv_thresh")
                    for node in self.node_neuron_map:
                        if node in input_nodes:
                            node_type = 'input'
                        else:
                            node_type = 'main'
                        node_id = self.node_neuron_map[node]
                        neuron = main_neurons[node_id]
                        print("{},\t{},\t{},\t{},\t{},\t{}".format(node, node_type, node_id, 
                                                                   neuron.v_rest, neuron.v_reset, neuron.v_thresh))
                    print("<<<")

            # Setup Synapses

            # generate connection lists
            input_input = []
            input_main = []
            main_main = []
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
                        input_main.append((self.node_neuron_map[u], self.node_neuron_map[v], weight, delay))
                else:
                    main_main.append((self.node_neuron_map[u], self.node_neuron_map[v], weight, delay))

            synapse = pynn_sim.StaticSynapse()

            # create projections
            if len(input_input) > 0:
                input_self  = FromListConnector(input_input)
                input_self_synapses = pynn_sim.Projection(input_neurons, input_neurons, input_self, synapse, label='input-input')

            if len(input_main) > 0:
                input_connector = FromListConnector(input_main)
                input_synapses = pynn_sim.Projection(input_neurons, main_neurons, input_connector, synapse, label='input-main')

            if len(main_main) > 0:
                main_connector = FromListConnector(main_main)
                main_synapses = pynn_sim.Projection(main_neurons, main_neurons, main_connector, synapse, label='main-main')

            # run simulation and collect results
            main_neurons.record(['spikes','v'])
            input_neurons.record('spikes')

            if verbose:
                print("Runtime: {}".format(self.runtime))
            pynn_sim.run(self.runtime)

            main_data = main_neurons.get_data()
            input_data = input_neurons.get_data()

            pynn_sim.end()

            if verbose:
                print("<<<")
                print("Synapses:")
                if self.backend != SPINNAKER_BACKEND:
                    print("input to main---")
                    print("From,\tTo,\tWeight\tDelay")
                    for synapse in input_synapses.get(["weight", "delay"], format="list"):
                        print(synapse)
                    print("main to main---")
                    print("From,\tTo,\tWeight\tDelay")
                    for synapse in main_synapses.get(["weight", "delay"], format="list"):
                        print(synapse)


            # process results
            spike_result = pd.DataFrame({'time':[],'neuron_number':[]})

            main_spiketrains = main_data.segments[0].spiketrains
            input_spiketrains = input_data.segments[0].spiketrains
            main_voltage = main_data.segments[0].filter(name='v')[0]

            spikes = {}
            for node in self.node_neuron_map:
                if verbose:
                    print("---results for: {}".format(node))
                pynn_index = self.node_neuron_map[node]
                if node not in input_nodes:
                    spiketrain = main_spiketrains[pynn_index]
                    if verbose:
                        print("voltage  {}".format(main_voltage[pynn_index]))
                else:
                    if verbose:
                        print("this is an input node")
                    spiketrain = input_spiketrains[pynn_index]
                if verbose:
                    print("spiketimes  {}".format(spiketrain))
                if spiketrain.any():
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
        per_brick_population = False
        verbose = False
        simulator = 'brian'
        if 'backend' in backend_args:
            simulator = backend_args['backend']
        if 'per_brick_population' in backend_args:
            per_brick_population = backend_args['per_brick_population']
        if 'verbose' in backend_args:
            verbose = backend_args['verbose'] 
        spike_result = self._run_pynn_sim(scaffold, simulator, per_brick_population, verbose)
        spike_result = spike_result.sort_values('time')
        return spike_result 
