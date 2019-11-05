import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

from timeit import default_timer as timer

from warnings import warn

from .backend import Backend

BRIAN_BACKEND = 0
SPINNAKER_BACKEND = 1

class pynn_Backend(Backend):

    def __init__(self):
        super(Backend, self).__init__()
        self.brick_population_map = {}
        self.node_neuron_map = {}

        # these parameters will have to be set based on the simulator
        self.defaults = {}

        self.backend = BRIAN_BACKEND
        self.collect_metrics = False
        self.store_voltage = False
        self.scale_factor = 1.0
        self.metrics = {}

        self.fugu_circuit = None
        self.fugu_graph = None
        self.verbose = False

        self.number_of_runs = 0

    def GetMetrics(self):
        return self.metrics

    def _process_input_values(self, input_values):
        raw_spike_arrays = {}
        for time_step in sorted(input_values.keys()):
            for neuron in input_values[time_step]:
                if neuron not in raw_spike_arrays:
                    raw_spike_arrays[neuron] = []
                raw_spike_arrays[neuron].append(time_step)
        return raw_spike_arrays

    def _create_projections(self):
        if self.backend == BRIAN_BACKEND:
            from pyNN.connectors import FromListConnector
        elif self.backend == SPINNAKER_BACKEND:
            from pyNN.spiNNaker import FromListConnector

        synapse = self.pynn_sim.StaticSynapse()

        self.input_main_synapses = []
        if len(self.input_to_main_exite) > 0:
            connector = FromListConnector(self.input_to_main_exite)
            self.input_main_synapses.append(
                                       self.pynn_sim.Projection(
                                              self.input_population,
                                              self.main_population,
                                              connector,
                                              synapse,
                                              label="Input-to-Main",
                                              )
                                       )
        if len(self.input_to_main_inhib) > 0:
            connector = FromListConnector(self.input_to_main_inhib)
            self.input_main_synapses.append(
                                       self.pynn_sim.Projection(
                                              self.input_population,
                                              self.main_population,
                                              connector,
                                              synapse,
                                              label="Input-to-Main",
                                              receptor_type='inhibitory',
                                              )
                                       )

        self.main_synapses = []
        if len(self.main_to_main_exite) > 0:
            connector = FromListConnector(self.main_to_main_exite)
            self.main_synapses.append(
                                 self.pynn_sim.Projection(
                                        self.main_population,
                                        self.main_population,
                                        connector,
                                        synapse,
                                        label="Main-to-Main",
                                        )
                                 )
        if len(self.main_to_main_inhib) > 0:
            connector = FromListConnector(self.main_to_main_inhib)
            self.main_synapses.append(
                                 self.pynn_sim.Projection(
                                        self.main_population,
                                        self.main_population,
                                        connector,
                                        synapse,
                                        label="Main-to-Main",
                                        receptor_type='inhibitory',
                                        )
                                 )

    def _run_pynn_sim(self, steps, run_input):
        max_runtime = steps * self.defaults['min_delay']
        if self.verbose:
            print("Max runtime: {}".format(max_runtime))
            print("Old inputs:", self.input_population.get('spike_times'))

        if self.number_of_runs > 1:
            self.pynn_sim.reset()

        input_spikes = [[] for n in self.input_to_pynn]
        processed_spikes = self._process_input_values(run_input)
        if self.verbose:
            print("Run input: {}".format(run_input))
            print("Processed input value: {}".format(processed_spikes))
            print("Input to pynn: {}".format(self.input_to_pynn))
        for neuron in self.input_to_pynn:
            if neuron in processed_spikes:
                spike_array = [spike_time * self.defaults['min_delay'] for spike_time in processed_spikes[neuron]]
                if self.backend == BRIAN_BACKEND:
                    input_spikes[self.input_to_pynn[neuron]] = [val / 10.0 for val in spike_array]
                else:
                    input_spikes[self.input_to_pynn[neuron]] = spike_array

        if self.verbose:
            print("Input spikes: {}".format(input_spikes))

        self.input_population.set(spike_times=input_spikes)

        if self.backend == BRIAN_BACKEND:
            self.main_population.set(**self.parameter_values)
            self.main_population.initialize(v=self.initial_potentials)
            self._create_projections()

        if self.verbose:
            print("New inputs:", self.input_population.get('spike_times'))
            if self.backend == BRIAN_BACKEND:
                print("___Parameter values___:")
                print("min delay: {}".format(self.defaults['min_delay']))
                for param in self.parameter_values:
                    main_params = self.main_population.get(param)
                    print("Parameter: {}, {}".format(param, main_params))
                    #for neuron in self.neuron_to_pynn:
                        #print("{}, {}".format(neuron, main_params[self.neuron_to_pynn[neuron]]))

                print("___Initial potentials___:")
                for neuron in self.neuron_to_pynn:
                    print("{}, {}".format(neuron, self.initial_potentials[self.neuron_to_pynn[neuron]]))

                print("---Input to main connections---")
                for synapse_project in self.input_main_synapses:
                    for edge in synapse_project.connections:
                        print(edge.as_tuple('index', 'weight', 'delay'))
                    print('===')

                print("---Main to main connections---")
                for synapse_project in self.main_synapses:
                    for edge in synapse_project.connections:
                        print(edge.as_tuple('index', 'weight', 'delay'))
                    print('===')

                #for index, spike_time in enumerate(self.input_population.get('spike_times')):
                    #print("New input spike times for {}: {}".format(index, spike_time))

        if self.collect_metrics:
            start = timer()
        self.pynn_sim.run(max_runtime)
        if self.collect_metrics:
            self.metrics['runtime'].append(timer() - start)

    def get_results(self):
        main_data = self.main_population.get_data()
        #input_data = self.input_population.get_data()

        def _process_run(run_index):
            spike_result = pd.DataFrame({'time':[],'neuron_number':[]})

            main_spiketrains = main_data.segments[run_index].spiketrains
            #input_spiketrains = input_data.segments[run_index].spiketrains

            if self.store_voltage:
                if self.report_all:
                    main_voltage = main_data.segments[run_index].filter(name='v')[0]
                else:
                    main_voltage = []
                    data = main_data.segments[run_index].filter(name='v')[0]
                    for neuron in self.output_names:
                        main_voltage.append(data[self.neuron_to_pynn[neuron]])

            spikes = {}
            for neuron in self.neuron_to_pynn:
                if self.report_all or neuron in self.output_names:
                    pynn_index = self.neuron_to_pynn[neuron]
                    spiketrain = main_spiketrains[pynn_index]
                    if self.verbose:
                        print("---results for: {}".format(neuron))
                        if self.store_voltage:
                            if self.report_all:
                                print("voltage  {}".format(main_voltage[pynn_index]))
                        print("spiketimes  {}".format(spiketrain))

                    if spiketrain.any():
                        for time in np.array(spiketrain):
                            if time not in spikes:
                                spikes[time] = set()
                            spikes[time].add(self.fugu_graph.node[neuron]['neuron_number'])

            labels = []
            for neuron in self.neuron_to_pynn:
                labels.append(neuron)
            if self.show_plots and self.store_voltage:
                from pyNN.utility.plotting import Figure, Panel
                import matplotlib.pyplot as plt
                if self.report_all:
                    Figure(
                        Panel(main_voltage,
                              ylabel="Membrane potential (mV)",
                              yticks=True, linewidth=2.0),
                        #title="Neuron: {}".format(neuron),
                        annotations=""
                    )
                else:
                    for i, neuron in enumerate(self.output_names):
                        label = "Neuron: {}".format(neuron)
                        plt.plot(main_voltage[i].times, main_voltage[i], label=label)
                    plt.legend()
                plt.show()

            #for neuron in self.input_to_pynn:
                #if self.report_all or neuron in self.output_names:
                    #input_index = self.input_to_pynn[neuron]
                    #spiketrain = input_spiketrains[input_index]
                    #if self.verbose:
                        #print("---results for: {}".format(neuron))
                        #print("spiketimes  {}".format(spiketrain))
                    #if spiketrain.any():
                        #for time in np.array(spiketrain):
                            #if time not in spikes:
                                #spikes[time] = set()
                            #spikes[time].add(self.fugu_graph.node[neuron]['neuron_number'])

            for time in spikes:
                mini_df = pd.DataFrame()
                times = [time for i in spikes[time]]
                mini_df['time'] = times
                mini_df['neuron_number'] = list(spikes[time])
                spike_result = spike_result.append(mini_df, sort=False)

            return spike_result

        # process results
        results = []
        for i in range(self.number_of_runs):
            results.append(_process_run(i))
        return results

    def embed(self, fugu_scaffold, record, embedding_args={}):
        self.report_all = record == 'all'

        simulator = embedding_args['backend'] if 'backend' in embedding_args else 'brian'

        self.show_plots = embedding_args['show_plots'] if 'show_plots' in embedding_args else False
        self.verbose = embedding_args['verbose'] if 'verbose' in embedding_args else False
        self.collect_metrics = embedding_args['collect_metrics'] if 'collect_metrics' in embedding_args else False
        self.store_voltage = embedding_args['store_voltage'] if 'store_voltage' in embedding_args else False
        self.scale_factor = embedding_args['scale_factor'] if 'scale_factor' in embedding_args else 1.0

        if self.collect_metrics:
            start = timer()
            self.metrics['runtime'] = []
        self.fugu_circuit = fugu_scaffold.circuit
        self.fugu_graph = fugu_scaffold.graph

        # setup simulator parameters
        if simulator == 'brian':
            # PyNN only has support for brian1 (i.e. brian) which is only python 2.x compatible
            assert sys.version_info <= (3,0)

            #import pyNN.brian as pynn_sim
            import pyNN.brian as test_sim
            self.pynn_sim = test_sim
            from pyNN.connectors import FromListConnector
            from pyNN.parameters import Sequence

            self.backend = BRIAN_BACKEND

            self.defaults['min_delay'] = 1.00
            self.defaults['tau_syn_E'] = 1.00
            self.defaults['tau_syn_I'] = 1.00
            self.defaults['i_offset'] = 0.00
            self.defaults['tau_m'] = 100000000
            self.defaults['v_rest'] = 0.0

            self.pynn_sim.setup(timestep=self.defaults['min_delay'])

        elif simulator == 'spinnaker' or simulator == 'spynnaker':
            assert sys.version_info <= (3,0)

            #import pyNN.spiNNaker as pynn_sim
            import pyNN.spiNNaker as test_sim
            self.pynn_sim = test_sim
            from pyNN.spiNNaker import FromListConnector

            self.backend = SPINNAKER_BACKEND

            self.defaults['min_delay'] = 1.00 * self.scale_factor
            self.defaults['cm'] = 1.00
            self.defaults['tau_m'] = 1.00
            self.defaults['i_offset'] = 0.00
            self.defaults['v_rest'] = 0.0

            self.pynn_sim.setup(timestep=1, max_delay=100)
            self.pynn_sim.set_number_of_neurons_per_core(self.pynn_sim.extra_models.IF1_curr_delta, 100)
        else:
            raise ValueError("unsupported pyNN backend")

        self.output_names = set()

        # Create dict for easy lookup of the vertices in a brick
        self.brick_neurons = {}
        for vertex in self.fugu_graph.nodes:
            brick = self.fugu_graph.nodes[vertex]['brick']
            if brick not in self.brick_neurons:
                self.brick_neurons[brick] = []
            self.brick_neurons[brick].append(vertex)

        if not self.report_all:
            for brick in self.fugu_circuit.nodes:
                if 'layer' in self.fugu_circuit.nodes[brick] and self.fugu_circuit.nodes[brick]['layer'] == 'output':
                    for o_list in self.fugu_circuit.nodes[brick]['output_lists']:
                        for neuron in o_list:
                            self.output_names.add(neuron)

        if self.verbose:
            print("Brick_vertices: {}".format(self.brick_neurons))

        # Setup neurons:
        #   Input neurons (each gets their own population)
        #   Main neurons (all go into one population)

        params = []
        params.append('v_thresh')
        params.append('v_rest')
        params.append('tau_m')
        params.append('i_offset')
        if self.backend == BRIAN_BACKEND:
            params.append('v_reset')
            params.append('tau_refrac')
            params.append('tau_syn_E')
            params.append('tau_syn_I')
        else:
            params.append('cm')

        self.parameter_values = {key:[] for key in params}

        self.initial_potentials = []

        # Sets a neuron's properties and initial values
        def add_neuron_params(neuron):
            neuron_props = self.fugu_graph.nodes[neuron]
            if 'threshold' in neuron_props:
                thresh = neuron_props['threshold']
            else:
                thresh = 1.0
                print("Error, threshold not found in neuron_props: {}".format(neuron_props))
            self.parameter_values['v_thresh'].append(thresh if thresh > 0.0 else 0.01)

            self.parameter_values['v_rest'].append(self.defaults['v_rest'])

            if 'potential' in neuron_props:
                self.initial_potentials.append(neuron_props['potential'])
            else:
                self.initial_potentials.append(self.defaults['v_rest'])

            self.parameter_values['i_offset'].append(self.defaults['i_offset'])

            if self.backend == BRIAN_BACKEND:
                self.parameter_values['v_reset'].append(self.defaults['v_rest'])
                self.parameter_values['tau_refrac'].append(self.defaults['min_delay'])
                self.parameter_values['tau_syn_E'].append(self.defaults['tau_syn_E'])
                self.parameter_values['tau_syn_I'].append(self.defaults['tau_syn_I'])

                if 'decay' in neuron_props:
                    self.parameter_values['tau_m'].append(self.defaults['tau_m'] * (1 - neuron_props['decay']))
                else:
                    self.parameter_values['tau_m'].append(self.defaults['tau_m'])
            else:
                self.parameter_values['cm'].append(self.defaults['cm'])
                if 'decay' in neuron_props and neuron_props['decay'] != 0.0:
                    raise ValueError("sPyNNaker backend currently only supports no decay")
                self.parameter_values['tau_m'].append(self.defaults['tau_m'])

        pynn_index = 0
        self.neuron_to_pynn = {} # Map of neuron to pynn_index
        self.input_to_pynn = {} # Map of input neuron to pynn_index
        input_index = 0
        self.output_neurons = set()

        if self.collect_metrics:
            start = timer()

        # Create neurons
        if self.backend == BRIAN_BACKEND:
            #processed_spikes = self._process_input_values(run_inputs[0])
            input_spikes = []

        for brick in self.fugu_circuit.nodes:
            brick = self.fugu_circuit.nodes[brick]
            is_input = brick['layer'] == 'input'
            for neuron in self.brick_neurons[brick['name']]:
                if is_input:
                    self.input_to_pynn[neuron] = input_index
                    input_index += 1
                else:
                    self.neuron_to_pynn[neuron] = pynn_index
                    add_neuron_params(neuron)
                    pynn_index += 1
                if neuron in brick['control_nodes'][0]['complete']:
                    self.output_neurons.add(neuron)

        if self.verbose:
            print("---Neuron to pynn index---")
            for neuron in self.neuron_to_pynn:
                print("{}, {}".format(neuron, self.neuron_to_pynn[neuron]))

        self.input_population = self.pynn_sim.Population(
                                       input_index,
                                       self.pynn_sim.SpikeSourceArray(),
                                       label='Input Population',
                                       )
        if self.backend == BRIAN_BACKEND:
            self.main_population = self.pynn_sim.Population(
                                          pynn_index,
                                          self.pynn_sim.IF_curr_exp(**self.parameter_values),
                                          label='Main Population',
                                          )
        else:
            self.main_population = self.pynn_sim.Population(
                                          pynn_index,
                                          self.pynn_sim.extra_models.IF1_curr_delta(**self.parameter_values),
                                          label='Main Population',
                                          )
        self.main_population.initialize(v=self.initial_potentials)

        if self.verbose:
            print("___Parameter values___:")
            print("min delay: {}".format(self.defaults['min_delay']))
            for param in self.parameter_values:
                main_params = self.main_population.get(param)
                print("Parameter: {}, {}".format(param, main_params))
                print("neuron values ---")
                for neuron in self.neuron_to_pynn:
                    print("{}, {}".format(neuron, self.parameter_values[param][self.neuron_to_pynn[neuron]]))

            print("___Initial potentials___:")
            for neuron in self.neuron_to_pynn:
                print("{}, {}".format(neuron, self.initial_potentials[self.neuron_to_pynn[neuron]]))

        # Setup synpases:
        #   Input-to-main
        #   Main-to-main

        # create edge lists
        self.input_to_main_exite = []
        self.main_to_main_exite = []
        self.input_to_main_inhib = []
        self.main_to_main_inhib = []

        synapse_prop_names = ['weight', 'delay']
        self.input_to_main_props = [{key:[] for key in synapse_prop_names}, {key:[] for key in synapse_prop_names}]
        self.main_to_main_props = [{key:[] for key in synapse_prop_names}, {key:[] for key in synapse_prop_names}]
        for u, v, values in self.fugu_graph.edges.data():
            weight = 0.0
            delay = self.defaults['min_delay']
            is_inhib = False
            if 'weight' in values:
                weight = values['weight']
                if self.backend == SPINNAKER_BACKEND:
                    weight = abs(weight)
                if values['weight'] < 0:
                    is_inhib = True
            if 'delay' in values:
                if self.backend == BRIAN_BACKEND:
                    delay = values['delay'] - 1
                else:
                    delay = values['delay']
                delay = delay * self.defaults['min_delay']
            if u in self.input_to_pynn:
                if is_inhib:
                    self.input_to_main_inhib.append((self.input_to_pynn[u], self.neuron_to_pynn[v], weight, delay))
                    self.input_to_main_props[1]['weight'].append(weight)
                    self.input_to_main_props[1]['delay'].append(delay)
                else:
                    self.input_to_main_exite.append((self.input_to_pynn[u], self.neuron_to_pynn[v], weight, delay))
                    self.input_to_main_props[0]['weight'].append(weight)
                    self.input_to_main_props[0]['delay'].append(delay)
            elif u in self.neuron_to_pynn and v in self.neuron_to_pynn:
                if is_inhib:
                    self.main_to_main_inhib.append((self.neuron_to_pynn[u], self.neuron_to_pynn[v], weight, delay))
                    self.main_to_main_props[1]['weight'].append(weight)
                    self.main_to_main_props[1]['delay'].append(delay)
                else:
                    self.main_to_main_exite.append((self.neuron_to_pynn[u], self.neuron_to_pynn[v], weight, delay))
                    self.main_to_main_props[0]['weight'].append(weight)
                    self.main_to_main_props[0]['delay'].append(delay)

        # create projections
        self._create_projections()

        if self.verbose:
            if self.backend == BRIAN_BACKEND:
                print("---Input to main connections---")
                for synapse_project in self.input_main_synapses:
                    for edge in synapse_project.connections:
                        print(edge.as_tuple('index', 'weight', 'delay'))
                    print('===')

                print("---Main to main connections---")
                for synapse_project in self.main_synapses:
                    for edge in synapse_project.connections:
                        print(edge.as_tuple('index', 'weight', 'delay'))
                    print('===')

            print("---Input to main connections---")
            print("----Excite----")
            for edge in self.input_to_main_exite:
                print(edge)
            print("----Inhib----")
            for edge in self.input_to_main_inhib:
                print(edge)

            print("---Main to main connections---")
            print("----Excite----")
            for edge in self.main_to_main_exite:
                print(edge)
            print("----Inhib----")
            for edge in self.main_to_main_inhib:
                print(edge)

        if self.collect_metrics:
            self.metrics['embed_time'] = timer() - start

        # Run sim
        if self.store_voltage:
            self.main_population.record(['spikes','v'])
        else:
            self.main_population.record(['spikes'])
        self.input_population.record(['spikes'])

    def cleanup(self):
        self.pynn_sim.end()

    def stream(self, scaffold, input_values, stepping, record, backend_args):
        warn("Stepping not supported yet.  Use a batching mode.")
        return None

    def batch(self, input_values, n_steps, backend_args=None):
        self.number_of_runs += 1
        self._run_pynn_sim(n_steps, input_values)

        #return spike_result
