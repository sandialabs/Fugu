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
        self.metrics = {}

    def _process_input_values(self, input_values):
        raw_spike_arrays = {}
        for time_step in sorted(input_values.keys()):
            for neuron in input_values[time_step]:
                if neuron not in raw_spike_arrays:
                    raw_spike_arrays[neuron] = []
                raw_spike_arrays[neuron].append(time_step)
        return raw_spike_arrays

    def GetMetrics(self):
        return self.metrics

    def _run_pynn_sim(self, fugu_scaffold, steps, run_inputs, report_all=False, simulator='brian', verbose=False, show_plots=False):
        if self.collect_metrics:
            self.metrics['runtime'] = []
        fugu_circuit = fugu_scaffold.circuit
        fugu_graph = fugu_scaffold.graph

        # setup simulator parameters
        if simulator == 'brian':
            # PyNN only has support for brian1 (i.e. brian) which is only python 2.x compatible
            assert sys.version_info <= (3,0)

            import pyNN.brian as pynn_sim
            from pyNN.connectors import FromListConnector
            from pyNN.parameters import Sequence

            self.backend = BRIAN_BACKEND 

            self.defaults['min_delay'] = 1.00
            self.defaults['tau_syn_E'] = 1.00
            self.defaults['tau_syn_I'] = 1.00
            self.defaults['i_offset'] = 0.00
            self.defaults['tau_m'] = 100000000
            self.defaults['v_rest'] = 0.0 

            max_runtime = steps * self.defaults['min_delay'] 

            pynn_sim.setup(timestep=self.defaults['min_delay'])

        elif simulator == 'spinnaker' or simulator == 'spynnaker':
            assert sys.version_info <= (3,0)

            import pyNN.spiNNaker as pynn_sim
            from pyNN.spiNNaker import FromListConnector

            self.backend = SPINNAKER_BACKEND 

            self.defaults['min_delay'] = 1.00 * self.scale_factor
            self.defaults['cm'] = 1.00
            self.defaults['tau_m'] = 1.00
            self.defaults['i_offset'] = 0.00
            self.defaults['v_rest'] = 0.0 

            max_runtime = steps * self.defaults['min_delay'] 

            #pynn_sim.setup(timestep=self.defaults['min_delay'])
            pynn_sim.setup(timestep=1, max_delay=100)
            pynn_sim.set_number_of_neurons_per_core(pynn_sim.extra_models.IF1_curr_delta, 100)
        else:
            raise ValueError("unsupported pyNN backend")

        fugu_circuit = fugu_scaffold.circuit
        fugu_graph = fugu_scaffold.graph
        output_names = set()

        # Create dict for easy lookup of the vertices in a brick
        brick_neurons = {}
        for vertex in fugu_graph.nodes:
            brick = fugu_graph.nodes[vertex]['brick']
            if brick not in brick_neurons:
                brick_neurons[brick] = []
            brick_neurons[brick].append(vertex)

        if not report_all:
            for brick in fugu_circuit.nodes:
                if 'layer' in fugu_circuit.nodes[brick] and fugu_circuit.nodes[brick]['layer'] == 'output':
                    for o_list in fugu_circuit.nodes[brick]['output_lists']:
                        for neuron in o_list:
                            output_names.add(neuron)

        if verbose:
            print("Brick_vertices: {}".format(brick_neurons))

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

        parameter_values = {key:[] for key in params} 

        initial_potential = []

        # Sets a neuron's properties and initial values
        def add_neuron_params(neuron):
            neuron_props = fugu_graph.nodes[neuron]
            if 'threshold' in neuron_props:
                thresh = neuron_props['threshold']
            else:
                thresh = 1.0
                print("Error, threshold not found in neuron_props: {}".format(neuron_props))
            parameter_values['v_thresh'].append(thresh if thresh > 0.0 else 0.01)

            parameter_values['v_rest'].append(self.defaults['v_rest'])

            initial_potential.append(neuron_props['potential'] if 'potential' in neuron_props else self.defaults['v_rest'])

            parameter_values['i_offset'].append(self.defaults['i_offset'])

            if self.backend == BRIAN_BACKEND:
                parameter_values['v_reset'].append(self.defaults['v_rest'])
                parameter_values['tau_refrac'].append(self.defaults['min_delay'])
                parameter_values['tau_syn_E'].append(self.defaults['tau_syn_E'])
                parameter_values['tau_syn_I'].append(self.defaults['tau_syn_I'])
                if 'decay' in neuron_props:
                    parameter_values['tau_m'].append(neuron_props['decay'] if neuron_props['decay'] > 0.0 else self.defaults['tau_m'])
                else:
                    parameter_values['tau_m'].append(self.defaults['tau_m'])
            else:
                parameter_values['cm'].append(self.defaults['cm'])
                if 'decay' in neuron_props and neuron_props['decay'] != 0.0:
                    raise ValueError("sPyNNaker backend currently only supports no decay")
                parameter_values['tau_m'].append(self.defaults['tau_m'])

        pynn_index = 0
        neuron_to_pynn = {} # Map of neuron to pynn_index
        input_to_pynn = {} # Map of input neuron to pynn_index
        input_index = 0
        output_neurons = set()

        if self.collect_metrics:
            start = timer()

        # Create neurons
        if self.backend == BRIAN_BACKEND:
            processed_spikes = self._process_input_values(run_inputs[0])
            input_spikes = []

        for brick in fugu_circuit.nodes:
            brick = fugu_circuit.nodes[brick]
            is_input = brick['layer'] == 'input'
            for neuron in brick_neurons[brick['name']]:
                if is_input:
                    if self.backend == BRIAN_BACKEND:
                        if neuron in processed_spikes:
                            input_spikes.append([spike_time * self.defaults['min_delay'] for spike_time in processed_spikes[neuron]])
                        else:
                            input_spikes.append([])

                    input_to_pynn[neuron] = input_index
                    input_index += 1
                else:
                    neuron_to_pynn[neuron] = pynn_index
                    add_neuron_params(neuron)
                    pynn_index += 1
                if neuron in brick['control_nodes'][0]['complete']:
                    output_neurons.add(neuron)

        if verbose:
            print("---Neuron to pynn index---")
            for neuron in neuron_to_pynn:
                print("{}, {}".format(neuron, neuron_to_pynn[neuron]))

        if verbose:
            print("___Parameter values___:")
            for param in parameter_values:
                print("Parameter: {}".format(param))
                for neuron in neuron_to_pynn:
                    print("{}, {}".format(neuron, parameter_values[param][neuron_to_pynn[neuron]]))

            print("___Initial potentials___:")
            for neuron in neuron_to_pynn:
                print("{}, {}".format(neuron, initial_potential[neuron_to_pynn[neuron]]))


        if self.backend == BRIAN_BACKEND:
            input_population = pynn_sim.Population(input_index, pynn_sim.SpikeSourceArray(spike_times=input_spikes), label='Input Population')
            main_population = pynn_sim.Population(pynn_index, pynn_sim.IF_curr_exp(**parameter_values), label='Main Population')
        else:
            input_population = pynn_sim.Population(input_index, pynn_sim.SpikeSourceArray(), label='Input Population')
            main_population = pynn_sim.Population(pynn_index, pynn_sim.extra_models.IF1_curr_delta(**parameter_values), label='Main Population')
        main_population.initialize(v=initial_potential)

        # Setup synpases:
        #   Input-to-main
        #   Main-to-main

        # create edge lists
        input_to_main_exite = []
        main_to_main_exite = []
        input_to_main_inhib = []
        main_to_main_inhib = []
        for u, v, values in fugu_graph.edges.data():
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
            if u in input_to_pynn:
                if is_inhib:
                    input_to_main_inhib.append((input_to_pynn[u], neuron_to_pynn[v], weight, delay))
                else:
                    input_to_main_exite.append((input_to_pynn[u], neuron_to_pynn[v], weight, delay))
            elif u in neuron_to_pynn and v in neuron_to_pynn:
                if is_inhib:
                    main_to_main_inhib.append((neuron_to_pynn[u], neuron_to_pynn[v], weight, delay))
                else:
                    main_to_main_exite.append((neuron_to_pynn[u], neuron_to_pynn[v], weight, delay))

        # create projections
        synapse = pynn_sim.StaticSynapse()

        input_main_synapses = []
        if len(input_to_main_exite) > 0:
            connector = FromListConnector(input_to_main_exite)
            input_main_synapses.append(pynn_sim.Projection(input_population, main_population, connector, synapse, label="Input-to-Main"))
        if len(input_to_main_inhib) > 0:
            connector = FromListConnector(input_to_main_inhib)
            input_main_synapses.append(pynn_sim.Projection(input_population, main_population, connector, synapse, label="Input-to-Main", receptor_type='inhibitory'))

        main_synapses = []
        if len(main_to_main_exite) > 0:
            connector = FromListConnector(main_to_main_exite)
            main_synapses.append(pynn_sim.Projection(main_population, main_population, connector, synapse, label="Main-to-Main"))
        if len(main_to_main_inhib) > 0:
            connector = FromListConnector(main_to_main_inhib)
            main_synapses.append(pynn_sim.Projection(main_population, main_population, connector, synapse, label="Main-to-Main", receptor_type='inhibitory'))

        if verbose:
            print("---Input to main connections---")
            print("----Excite----")
            for edge in input_to_main_exite:
                print(edge)
            print("----Inhib----")
            for edge in input_to_main_inhib:
                print(edge)

            print("---Main to main connections---")
            print("----Excite----")
            for edge in main_to_main_exite:
                print(edge)
            print("----Inhib----")
            for edge in main_to_main_inhib:
                print(edge)

        if self.collect_metrics:
            self.metrics['embed_time'] = timer() - start

        # Run sim
        if self.store_voltage:
            main_population.record(['spikes','v'])
        else:
            main_population.record(['spikes'])
        input_population.record(['spikes'])

        if verbose:
            print("Runtime: {}".format(max_runtime))

        def check_complete_status(t):
            spike_data = main_population.get_data().segments[0].spiketrains
            to_remove = []
            for neuron in output_neurons:
                if len(spike_data[neuron_to_pynn[neuron]]) > 1:
                    to_remove.append(neuron)

            for neuron in to_remove:
                output_neurons.remove(neuron)

            if len(output_neurons) > 0:
                return t + self.defaults['min_delay'] 
            else:
                return max_runtime

        #pynn_sim.run(max_runtime, callbacks=[check_complete_status])
        for run_id, run_input in enumerate(run_inputs):
            if self.backend == SPINNAKER_BACKEND:
                input_spikes = [[] for n in input_to_pynn]
                processed_spikes = self._process_input_values(run_input)
                for neuron in input_to_pynn:
                    if neuron in processed_spikes:
                        spike_array = [spike_time * self.defaults['min_delay'] for spike_time in processed_spikes[neuron]]
                        input_spikes[input_to_pynn[neuron]] = spike_array
                input_population.set(spike_times=input_spikes)

            if verbose:
                for index, spike_time in enumerate(input_population.get('spike_times')):
                    print("Set spike times for {}: {}".format(index, spike_time))

            if run_id > 0:
                pynn_sim.reset()

            if self.collect_metrics:
                start = timer()
            pynn_sim.run(max_runtime)
            if self.collect_metrics:
                self.metrics['runtime'].append(timer() - start)

        main_data = main_population.get_data()
        input_data = input_population.get_data()

        pynn_sim.end()

        # process results
        spike_results = []
        for index, run in enumerate(run_inputs):
            if verbose:
                print("Spike results for run index {}".format(index))
            spike_result = pd.DataFrame({'time':[],'neuron_number':[]})

            main_spiketrains = main_data.segments[index].spiketrains
            input_spiketrains = input_data.segments[index].spiketrains

            if self.store_voltage:
                main_voltage = main_data.segments[index].filter(name='v')[0]

            spikes = {}
            for neuron in neuron_to_pynn:
                if report_all or neuron in output_names:
                    pynn_index = neuron_to_pynn[neuron]
                    spiketrain = main_spiketrains[pynn_index]
                    if verbose:
                        print("---results for: {}".format(neuron))
                        if self.store_voltage:
                            print("voltage  {}".format(main_voltage[pynn_index]))
                        print("spiketimes  {}".format(spiketrain))

                    if spiketrain.any():
                        for time in np.array(spiketrain):
                            if time not in spikes:
                                spikes[time] = set()
                            spikes[time].add(fugu_graph.node[neuron]['neuron_number'])

            labels = []
            for neuron in neuron_to_pynn:
                labels.append(neuron)
            if show_plots and self.store_voltage:
                from pyNN.utility.plotting import Figure, Panel
                import matplotlib.pyplot as plt
                Figure(
                    Panel(main_voltage,
                          ylabel="Membrane potential (mV)",
                          yticks=True, linewidth=2.0),
                    #title="Neuron: {}".format(neuron), 
                    annotations=""
                )
                plt.show()

            for neuron in input_to_pynn:
                if report_all or neuron in output_names:
                    input_index = input_to_pynn[neuron]
                    spiketrain = input_spiketrains[input_index]
                    if verbose:
                        print("---results for: {}".format(neuron))
                        print("spiketimes  {}".format(spiketrain))
                    if spiketrain.any():
                        for time in np.array(spiketrain):
                            if time not in spikes:
                                spikes[time] = set()
                            spikes[time].add(fugu_graph.node[neuron]['neuron_number'])

            for time in spikes:
                mini_df = pd.DataFrame()
                times = [time for i in spikes[time]]
                mini_df['time'] = times
                mini_df['neuron_number'] = list(spikes[time])
                spike_result = spike_result.append(mini_df, sort=False)
            spike_results.append(spike_result)

        return spike_results

    def stream(self, scaffold, input_values, stepping, record, backend_args):
        warn("Stepping not supported yet.  Use a batching mode.")
        return None
    
    def batch(self, scaffold, input_values, n_steps, record, backend_args):
        record_all = record == 'all' 

        simulator = backend_args['backend'] if 'backend' in backend_args else 'brian'
        verbose = backend_args['verbose'] if 'verbose' in backend_args else False
        show_plots = backend_args['show_plots'] if 'show_plots' in backend_args else False

        self.collect_metrics = backend_args['collect_metrics'] if 'collect_metrics' in backend_args else False
        self.store_voltage = backend_args['store_voltage'] if 'store_voltage' in backend_args else False
        self.scale_factor = backend_args['scale_factor'] if 'scale_factor' in backend_args else 1.0 

        #@TEMPORARY: will figure out a way to do this at a higher level (at the Fugu level)
        run_inputs = backend_args['run_inputs'] if 'run_inputs' in backend_args else [input_values]
        spike_result = self._run_pynn_sim(scaffold, n_steps, run_inputs, 
                                            report_all = record_all,
                                            simulator = simulator, 
                                            verbose = verbose, 
                                            show_plots = show_plots)

        return spike_result if len(spike_result) > 1 else spike_result[0]
