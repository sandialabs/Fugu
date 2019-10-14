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

    def __init__(self, runtime=10):
        super(Backend, self).__init__()
        self.brick_population_map = {}
        self.node_neuron_map = {}

        # these parameters will have to be set based on the simulator
        self.defaults = {}
        self.steps = runtime
        self.runtime = self.steps * 0.01

        self.backend = BRIAN_BACKEND 
        self.collect_metrics = False
        self.metrics = {}

    def GetMetrics(self):
        return self.metrics

    def _run_pynn_sim(self, fugu_scaffold, simulator='brian', verbose=False, show_plots=False):
        fugu_circuit = fugu_scaffold.circuit
        fugu_graph = fugu_scaffold.graph

        # setup simulator parameters
        if simulator == 'brian':
            # PyNN only has support for brian1 (i.e. brian) which is only python 2.x compatible
            assert sys.version_info <= (3,0)
            import pyNN.brian as pynn_sim

            from pyNN.connectors import FromListConnector

            self.backend = BRIAN_BACKEND 

            self.defaults['min_delay'] = 1.00
            self.defaults['tau_syn_E'] = 1.00
            self.defaults['tau_syn_I'] = 1.00
            self.defaults['i_offset'] = 0.00
            self.defaults['tau_m'] = 100000000
            self.defaults['v_rest'] = 0.0 

            self.runtime = self.steps * self.defaults['min_delay'] 

            pynn_sim.setup(timestep=self.defaults['min_delay'])

        elif simulator == 'spinnaker' or simulator == 'spynnaker':
            assert sys.version_info <= (3,0)

            import pyNN.spiNNaker as pynn_sim
            from pyNN.spiNNaker import FromListConnector

            self.backend = SPINNAKER_BACKEND 

            self.defaults['min_delay'] = 1.00
            self.defaults['cm'] = 1.00
            self.defaults['tau_m'] = 1.00
            self.defaults['i_offset'] = 0.00
            self.defaults['v_rest'] = 0.0 

            self.runtime = self.steps * self.defaults['min_delay'] 

            pynn_sim.setup(timestep=self.defaults['min_delay'])
        else:
            raise ValueError("unsupported pyNN backend")

        fugu_circuit = fugu_scaffold.circuit
        fugu_graph = fugu_scaffold.graph

        # Create dict for easy lookup of the vertices in a brick
        brick_neurons = {}
        for vertex in fugu_graph.nodes:
            brick = fugu_graph.nodes[vertex]['brick']
            if brick not in brick_neurons:
                brick_neurons[brick] = []
            brick_neurons[brick].append(vertex)

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
        input_spikes = []
        input_index = 0

        if self.collect_metrics:
            start = timer()

        # Create neurons
        for brick in fugu_circuit.nodes:
            brick = fugu_circuit.nodes[brick]
            if 'layer' in brick and brick['layer'] == 'input':
                spike_arrays = brick['brick'].vector
                if verbose:
                    print("Brick's spike arrays: {}".format(spike_arrays))

                for neuron in brick_neurons[brick['name']]:
                    index = fugu_graph.nodes[neuron]['index']
                    if index != -1:
                        spike_array = []
                        for i, spike in enumerate(spike_arrays[index]):
                            if verbose:
                                print("i, spike: {} {}".format(i, spike))
                            if spike == 1:
                                spike_array.append(i * self.defaults['min_delay'])
                        if verbose:
                            print("Spike array for {}: {}".format(neuron, spike_array))
                        if len(spike_array) > 0:
                            input_to_pynn[neuron] = input_index
                            input_spikes.append(spike_array)
                            input_index += 1
                    else:
                        neuron_to_pynn[neuron] = pynn_index
                        add_neuron_params(neuron)
                        pynn_index += 1
            else:
                for neuron in brick_neurons[brick['name']]:
                    neuron_to_pynn[neuron] = pynn_index
                    add_neuron_params(neuron)
                    pynn_index += 1
        if verbose:
            print("---Neuron to pynn index---")
            for neuron in neuron_to_pynn:
                print("{}, {}".format(neuron, neuron_to_pynn[neuron]))

        if verbose:
            print("___Parameter values___:")
            #print(parameter_values)
            for param in parameter_values:
                print("Parameter: {}".format(param))
                for neuron in neuron_to_pynn:
                    print("{}, {}".format(neuron, parameter_values[param][neuron_to_pynn[neuron]]))

            print("___Initial potentials___:")
            for neuron in neuron_to_pynn:
                print("{}, {}".format(neuron, initial_potential[neuron_to_pynn[neuron]]))

        input_population = pynn_sim.Population(input_index, pynn_sim.SpikeSourceArray(spike_times=input_spikes), label='Input Population')

        if self.backend == BRIAN_BACKEND:
            main_population = pynn_sim.Population(pynn_index, pynn_sim.IF_curr_exp(**parameter_values), label='Main Population')
        else:
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
        connector = FromListConnector(main_to_main_exite)
        main_synapses.append(pynn_sim.Projection(main_population, main_population, connector, synapse, label="Main-to-Main"))
        connector = FromListConnector(main_to_main_inhib)
        main_synapses.append(pynn_sim.Projection(main_population, main_population, connector, synapse, label="Main-to-Main", receptor_type='inhibitory'))

        if verbose:
            print("---Input to main connections---")
            print("----Excite----")
            for neuron in input_to_main_exite:
                for edge in input_to_main_exite[neuron]:
                    print(edge)
            print("----Inhib----")
            for neuron in input_to_main_inhib:
                for edge in input_to_main_inhib[neuron]:
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
        main_population.record(['spikes','v'])
        input_population.record(['spikes'])

        if verbose:
            print("Runtime: {}".format(self.runtime))

        if self.collect_metrics:
            start = timer()
        pynn_sim.run(self.runtime)
        if self.collect_metrics:
            self.metrics['runtime'] = timer() - start 

        main_data = main_population.get_data()
        input_data = input_population.get_data()

        pynn_sim.end()

        # process results
        spike_result = pd.DataFrame({'time':[],'neuron_number':[]})

        main_spiketrains = main_data.segments[0].spiketrains
        main_voltage = main_data.segments[0].filter(name='v')[0]
        input_spiketrains = input_data.segments[0].spiketrains

        spikes = {}
        for neuron in neuron_to_pynn:
            pynn_index = neuron_to_pynn[neuron]
            spiketrain = main_spiketrains[pynn_index]
            if verbose:
                print("---results for: {}".format(neuron))
                #print("voltage  {}".format(main_voltage[pynn_index]))
                print("spiketimes  {}".format(spiketrain))

            if spiketrain.any():
                for time in np.array(spiketrain):
                    if time not in spikes:
                        spikes[time] = set()
                    spikes[time].add(fugu_graph.node[neuron]['neuron_number'])

        labels = []
        for neuron in neuron_to_pynn:
            labels.append(neuron)
        if show_plots:
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
        simulator = backend_args['backend'] if 'backend' in backend_args else 'brian'
        verbose = backend_args['verbose'] if 'verbose' in backend_args else False
        show_plots = backend_args['show_plots'] if 'show_plots' in backend_args else False
        self.collect_metrics = backend_args['collect_metrics'] if 'collect_metrics' in backend_args else False
        spike_result = self._run_pynn_sim(scaffold, simulator, verbose, show_plots)
        spike_result = spike_result.sort_values('time')
        return spike_result 
