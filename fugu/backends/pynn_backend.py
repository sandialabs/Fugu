import sys

import numpy as np
import pandas as pd

from timeit import default_timer as timer
from warnings import warn

from .backend import Backend
from ..utils.misc import CalculateSpikeTimes

BRIAN_BACKEND = 0
SPINNAKER_BACKEND = 1

BACKEND_NAMES = {}
BACKEND_NAMES[BRIAN_BACKEND] = "brian"
BACKEND_NAMES[SPINNAKER_BACKEND] = "spinnaker"


class pynn_Backend(Backend):

    def __init__(self):
        super(Backend, self).__init__()
        self._initialize_members()

    def _initialize_members(self):
        # maps
        self.neuron_type_map = {}
        self.neuron_index_map = {}

        self.input_neurons = set()
        self.main_neurons = set()
        self.output_neurons = set()

        self.input_neuron_types = {}
        self.input_neuron_type_names = []
        self.main_neuron_types = {}
        self.main_neuron_type_names = []

        # Neuron populations and properties
        # Keyed by neuron type name
        self.main_indicies = {}

        self.input_populations = {}
        self.main_populations = {}
        self.property_values = {}
        self.initial_potentials = {}

        # Neuron projections
        self.synapse_receptors = ['excitatory', 'inhibitory']
        '''
        key structure:
        - synapse receptor
            - source population
                - target population
        '''
        self.input_main_edge_lists = {syn_type: {} for syn_type in self.synapse_receptors}
        self.main_edge_lists = {syn_type: {} for syn_type in self.synapse_receptors}
        self.input_main_edge_properties = {syn_type: {} for syn_type in self.synapse_receptors}
        self.main_edge_properties = {syn_type: {} for syn_type in self.synapse_receptors}

        # I really hate this, this needs to be better
        self.input_main_edge_index_map = {}
        self.input_main_edge_counts = {}
        self.main_edge_index_map = {}
        self.main_edge_counts = {}

        self.input_main_synapses = {}
        self.main_synapses = {}

        # these parameters will have to be set based on the simulator
        self.defaults = {}

        self.backend = BRIAN_BACKEND
        self.collect_metrics = False
        self.return_potentials = False
        self.scale_factor = 1.0
        self.metrics = {}

        self.fugu_scaffold = None
        self.verbose = False

    def print_network_info(self):
        print("Input population counts:---")
        for input_type in self.input_indicies:
            print(input_type, self.input_indicies[input_type])
        print("Main population counts:---")
        for main_type in self.main_indicies:
            print(main_type, self.main_indicies[main_type])
        print("---Neuron to pynn index---")
        print("Neuron name, type, index:")
        for neuron in self.neuron_index_map:
            print("{}, {}, {}".format(self.neuron_type_map[neuron], neuron, self.neuron_index_map[neuron]))

        print("___Parameter values___:")
        print("min delay: {}".format(self.defaults['min_delay']))
        for neuron_type in self.main_populations:
            print(">>>Neuron type: {}<<<".format(neuron_type))
            for prop in self.property_values[neuron_type]:
                main_props = self.main_populations[neuron_type].get(prop)
                print("Parameter: {}, {}".format(prop, main_props))
                print("Parameter count: {}".format(len(self.property_values[neuron_type][prop])))
                print("neuron values ---")
                for neuron in self.neuron_index_map:
                    if self.neuron_type_map[neuron] == neuron_type:
                        print("{}, {}".format(
                                         neuron,
                                         self.property_values[neuron_type][prop][self.neuron_index_map[neuron]],
                                         ))

            print("___Initial potentials___:")
            for neuron in self.neuron_index_map:
                if self.neuron_type_map[neuron] == neuron_type:
                    print("{}, {}".format(
                                     neuron,
                                     self.initial_potentials[neuron_type][self.neuron_index_map[neuron]],
                                     ))

        print("---Input to main connections (Edge List)---")
        for receptor in self.input_main_synapses:
            for source in self.input_main_synapses[receptor]:
                for target in self.input_main_synapses[receptor][source]:
                    print("Synapse: ({})-{}->({})".format(source, receptor, target))
                    if self.backend == BRIAN_BACKEND or self.backend == SPINNAKER_BACKEND:
                        for edge in self.input_main_edge_lists[receptor][source][target]:
                            print(edge)
                    else:
                        for edge in self.input_main_synapses[receptor][source][target]:
                            print(edge)
        print('===')

        print("---Main to main connections (Edge List)---")
        for receptor in self.main_synapses:
            for source in self.main_synapses[receptor]:
                for target in self.main_synapses[receptor][source]:
                    print("Synapse: ({})-{}->({})".format(source, receptor, target))
                    if self.backend == BRIAN_BACKEND or self.backend == SPINNAKER_BACKEND:
                        for edge in self.main_edge_lists[receptor][source][target]:
                            print(edge)
                    else:
                        for edge in self.main_synapses[receptor][source][target]:
                            print(edge)

        print('===')
        print("---Input to main properties---")
        for receptor in self.input_main_edge_properties:
            for source_type in self.input_main_edge_properties[receptor]:
                for target_type in self.input_main_edge_properties[receptor][source_type]:
                    print("expected values:")
                    for prop in self.input_main_edge_properties[receptor][source_type][target_type]:
                        print("Property: {}".format(prop))
                        for value in self.input_main_edge_properties[receptor][source_type][target_type][prop]:
                            print(value)
                    print("actual values:")
                    for prop in self.input_main_edge_properties[receptor][source_type][target_type]:
                        print("Property: {}".format(prop))
                        for value in self.input_main_synapses[receptor][source_type][target_type].get(
                                                                                                   prop,
                                                                                                   format='list'
                                                                                                   ):
                            print(value)
        print("---Main to main properties---")
        for receptor in self.main_edge_properties:
            for source_type in self.main_edge_properties[receptor]:
                for target_type in self.main_edge_properties[receptor][source_type]:
                    print("expected values:")
                    for prop in self.main_edge_properties[receptor][source_type][target_type]:
                        print("Property: {}".format(prop))
                        for value in self.main_edge_properties[receptor][source_type][target_type][prop]:
                            print(value)
                    print("actual values:")
                    for prop in self.main_edge_properties[receptor][source_type][target_type]:
                        print("Property: {}".format(prop))
                        for value in self.main_synapses[receptor][source_type][target_type].get(prop, format='list'):
                            print(value)

    def _create_projection(self, edge_list, source_population, target_population, label, receptor_type):
        connector = self.FromListConnector(edge_list)
        synapse = self.pynn_sim.StaticSynapse()
        return self.pynn_sim.Projection(
                               source_population,
                               target_population,
                               connector,
                               synapse,
                               label=label,
                               receptor_type=receptor_type,
                               )

    def _create_projections(self):
        # go through each list
        for receptor in self.input_main_edge_lists:
            self.input_main_synapses[receptor] = {}
            for source_type in self.input_main_edge_lists[receptor]:
                source_population = self.input_populations[source_type]
                self.input_main_synapses[receptor][source_type] = {}
                for target_type in self.input_main_edge_lists[receptor][source_type]:
                    target_population = self.main_populations[target_type]
                    edge_list = self.input_main_edge_lists[receptor][source_type][target_type]
                    label = "input-to-main_{}-{}".format(source_type, target_type)
                    self.input_main_synapses[receptor][source_type][target_type] = self._create_projection(
                                                                                          edge_list,
                                                                                          source_population,
                                                                                          target_population,
                                                                                          label=label,
                                                                                          receptor_type=receptor,
                                                                                          )
                    if self.backend != SPINNAKER_BACKEND:
                        self.input_main_synapses[receptor][source_type][target_type].set(
                                **self.input_main_edge_properties[receptor][source_type][target_type]
                                )
        for receptor in self.main_edge_lists:
            self.main_synapses[receptor] = {}
            for source_type in self.main_edge_lists[receptor]:
                source_population = self.main_populations[source_type]
                self.main_synapses[receptor][source_type] = {}
                for target_type in self.main_edge_lists[receptor][source_type]:
                    target_population = self.main_populations[target_type]
                    edge_list = self.main_edge_lists[receptor][source_type][target_type]
                    label = "main-to-main_{}-{}".format(source_type, target_type)
                    self.main_synapses[receptor][source_type][target_type] = self._create_projection(
                                                                                    edge_list,
                                                                                    source_population,
                                                                                    target_population,
                                                                                    label=label,
                                                                                    receptor_type=receptor,
                                                                                    )
                    if self.backend != SPINNAKER_BACKEND:
                        self.main_synapses[receptor][source_type][target_type].set(
                                **self.main_edge_properties[receptor][source_type][target_type]
                                )

    def _create_pynn_network(self):
        # create populations
        for input_type in self.input_neuron_type_names:
            if self.input_indicies[input_type] > 0:
                self.input_populations[input_type] = self.pynn_sim.Population(
                                                                     self.input_indicies[input_type],
                                                                     self.input_neuron_types[input_type](),
                                                                     label="Input-{}".format(input_type),
                                                                     )

        for main_type in self.main_neuron_type_names:
            if self.main_indicies[main_type] > 0:
                self.main_populations[main_type] = self.pynn_sim.Population(
                                                                   self.main_indicies[main_type],
                                                                   self.main_neuron_types[main_type](
                                                                          **(self.property_values[main_type])
                                                                          ),
                                                                   label="Main-{}".format(main_type),
                                                                   )
                self.main_populations[main_type].initialize(v=self.initial_potentials[main_type])

        if self.return_potentials:
            for neuron_type in self.main_populations:
                self.main_populations[neuron_type].record(['spikes', 'v'])
        else:
            for neuron_type in self.main_populations:
                self.main_populations[neuron_type].record(['spikes'])
        for neuron_type in self.input_populations:
            self.input_populations[neuron_type].record(['spikes'])

        # create projections
        self._create_projections()

    def compile(self, scaffold, compile_args={}):
        # creates neuron populations and synapses
        simulator = compile_args['backend'] if 'backend' in compile_args else 'brian'

        self.report_all = compile_args['record'] == 'all' if 'record' in compile_args else False
        self.show_plots = compile_args['show_plots'] if 'show_plots' in compile_args else False
        self.verbose = compile_args['verbose'] if 'verbose' in compile_args else False
        self.collect_metrics = compile_args['collect_metrics'] if 'collect_metrics' in compile_args else False
        self.return_potentials = compile_args['return_potentials'] if 'return_potentials' in compile_args else False
        self.scale_factor = compile_args['scale_factor'] if 'scale_factor' in compile_args else 1.0

        if self.collect_metrics:
            start = timer()
            self.metrics['runtime'] = []
        self.fugu_scaffold = scaffold

        # setup simulator parameters
        if simulator == 'brian':
            # PyNN only has support for brian1 (i.e. brian) which is only python 2.x compatible
            assert sys.version_info <= (3, 0)

            import pyNN.brian as test_sim
            self.pynn_sim = test_sim
            from pyNN.connectors import FromListConnector
            self.FromListConnector = FromListConnector

            self.backend = BRIAN_BACKEND

            self.defaults['min_delay'] = 1.00
            self.defaults['tau_syn_E'] = 1.00
            self.defaults['tau_syn_I'] = 1.00
            self.defaults['i_offset'] = 0.00
            self.defaults['tau_m'] = 100000000
            self.defaults['v_rest'] = 0.0

            self.input_neuron_type_names.append('SpikeSourceArray')
            self.input_neuron_types[self.input_neuron_type_names[0]] = self.pynn_sim.SpikeSourceArray
            self.main_neuron_type_names.append('IF_curr_exp')
            self.main_neuron_types[self.main_neuron_type_names[0]] = self.pynn_sim.IF_curr_exp

            self.pynn_sim.setup(timestep=1)

        elif simulator == 'spinnaker' or simulator == 'spynnaker':
            # Currently we only support spinnaker running on python 2
            assert sys.version_info <= (3, 0)

            import pyNN.spiNNaker as test_sim
            self.pynn_sim = test_sim
            from pyNN.spiNNaker import FromListConnector
            self.FromListConnector = FromListConnector

            self.backend = SPINNAKER_BACKEND

            self.defaults['min_delay'] = 1.00 * self.scale_factor
            self.defaults['cm'] = 1.00
            self.defaults['tau_m'] = 1.00
            self.defaults['i_offset'] = 0.00
            self.defaults['v_rest'] = 0.0

            self.input_neuron_type_names.append('SpikeSourceArray')
            self.input_neuron_types[self.input_neuron_type_names[0]] = self.pynn_sim.SpikeSourceArray
            self.main_neuron_type_names.append('IF0_curr_delta')
            self.main_neuron_type_names.append('IF1_curr_delta')
            self.main_neuron_types[self.main_neuron_type_names[0]] = self.pynn_sim.extra_models.IF0_curr_delta
            self.main_neuron_types[self.main_neuron_type_names[1]] = self.pynn_sim.extra_models.IF1_curr_delta

            self.pynn_sim.setup(timestep=1)
            for main_type in self.main_neuron_types:
                self.pynn_sim.set_number_of_neurons_per_core(self.main_neuron_types[main_type], 100)
        else:
            raise ValueError("unsupported pyNN backend")

        # Create dict for easy lookup of the vertices in a brick
        self.brick_neurons = {}
        for vertex in self.fugu_scaffold.graph.nodes:
            brick = self.fugu_scaffold.graph.nodes[vertex]['brick']
            if brick not in self.brick_neurons:
                self.brick_neurons[brick] = []
            self.brick_neurons[brick].append(vertex)

        if not self.report_all:
            for brick in self.fugu_scaffold.circuit.nodes:
                if 'layer' in self.fugu_scaffold.circuit.nodes[brick]:
                    if self.fugu_scaffold.circuit.nodes[brick]['layer'] == 'output':
                        for o_list in self.fugu_scaffold.circuit.nodes[brick]['output_lists']:
                            for neuron in o_list:
                                self.output_neurons.add(neuron)

        if self.verbose:
            print("Brick_vertices: {}".format(self.brick_neurons))

        # Setup neurons:
        #   Input neurons (each gets their own population)
        # Main neurons (all go into one population)

        props = []
        props.append('v_thresh')
        props.append('v_rest')
        props.append('tau_m')
        props.append('i_offset')
        if self.backend == BRIAN_BACKEND:
            props.append('v_reset')
            props.append('tau_refrac')
            props.append('tau_syn_E')
            props.append('tau_syn_I')
        else:
            props.append('cm')

        self.input_indicies = {input_type: 0 for input_type in self.input_neuron_type_names}

        for main_type in self.main_neuron_type_names:
            self.main_indicies[main_type] = 0
            self.property_values[main_type] = {prop: [] for prop in props}
            self.initial_potentials[main_type] = []

        # Sets a neuron's properties and initial values
        def add_neuron_props(neuron):
            neuron_props = self.fugu_scaffold.graph.nodes[neuron]

            if self.backend == BRIAN_BACKEND:
                neuron_type = self.main_neuron_type_names[0]
            else:
                neuron_type = self.main_neuron_type_names[1]
                if 'decay' in neuron_props:
                    if neuron_props['decay'] == 1.0:
                        neuron_type = self.main_neuron_type_names[0]
                    elif neuron_props['decay'] == 0.0:
                        neuron_type = self.main_neuron_type_names[1]
                    else:
                        raise ValueError("sPyNNaker backend currently only supports no decay or instant decay")

            self.neuron_type_map[neuron] = neuron_type

            if self.backend == BRIAN_BACKEND:
                self.property_values[neuron_type]['v_reset'].append(self.defaults['v_rest'])
                self.property_values[neuron_type]['tau_refrac'].append(self.defaults['min_delay'])
                self.property_values[neuron_type]['tau_syn_E'].append(self.defaults['tau_syn_E'])
                self.property_values[neuron_type]['tau_syn_I'].append(self.defaults['tau_syn_I'])

                if 'decay' in neuron_props:
                    decay = neuron_props['decay']
                    if decay >= 1:
                        self.property_values[neuron_type]['tau_m'].append(1)
                        if decay > 1:
                            print("Fugu warning: decay value is truncated to 1")
                    else:
                        self.property_values[neuron_type]['tau_m'].append(self.defaults['tau_m'] * (1 - decay))
                else:
                    self.property_values[neuron_type]['tau_m'].append(self.defaults['tau_m'])
            else:
                self.property_values[neuron_type]['tau_m'].append(self.defaults['tau_m'])
                self.property_values[neuron_type]['cm'].append(self.defaults['cm'])

            if 'threshold' in neuron_props:
                thresh = neuron_props['threshold']
            else:
                thresh = 1.0
                print("Error, threshold not found in \"{}\"'s props: {}".format(neuron, neuron_props))
            self.property_values[neuron_type]['v_thresh'].append(thresh if thresh > 0.0 else 0.01)

            self.property_values[neuron_type]['v_rest'].append(self.defaults['v_rest'])

            if 'potential' in neuron_props:
                self.initial_potentials[neuron_type].append(neuron_props['potential'])
            else:
                self.initial_potentials[neuron_type].append(self.defaults['v_rest'])

            self.property_values[neuron_type]['i_offset'].append(self.defaults['i_offset'])

            self.neuron_index_map[neuron] = self.main_indicies[neuron_type]
            self.main_indicies[neuron_type] += 1
            self.main_neurons.add(neuron)

        # Create neurons
        for brick in self.fugu_scaffold.circuit.nodes:
            brick = self.fugu_scaffold.circuit.nodes[brick]
            is_input = brick['layer'] == 'input' if 'layer' in brick else False
            for neuron in self.brick_neurons[brick['tag']]:
                if is_input:
                    input_type = self.input_neuron_type_names[0]
                    self.neuron_type_map[neuron] = input_type
                    self.neuron_index_map[neuron] = self.input_indicies[input_type]
                    self.input_indicies[input_type] += 1
                    self.input_neurons.add(neuron)
                else:
                    add_neuron_props(neuron)
                if neuron in brick['control_nodes'][0]['complete'] or self.report_all:
                    self.output_neurons.add(neuron)

        # Setup synpases:
        synapse_prop_names = ['weight', 'delay']
        for u, v, values in self.fugu_scaffold.graph.edges.data():
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
            u_type = self.neuron_type_map[u]
            v_type = self.neuron_type_map[v]
            if is_inhib:
                syn_receptor = self.synapse_receptors[1]
            else:
                syn_receptor = self.synapse_receptors[0]
            u_index = self.neuron_index_map[u]
            v_index = self.neuron_index_map[v]
            synapse = (self.neuron_index_map[u], self.neuron_index_map[v], weight, delay)

            if u in self.input_neurons:
                if u_type not in self.input_main_edge_lists[syn_receptor]:
                    self.input_main_edge_lists[syn_receptor] = {u_type: {}}
                    self.input_main_edge_properties[syn_receptor] = {u_type: {}}
                    self.input_main_edge_counts[syn_receptor] = {u_type: {}}
                if v_type not in self.input_main_edge_lists[syn_receptor][u_type]:
                    self.input_main_edge_lists[syn_receptor][u_type][v_type] = []
                    self.input_main_edge_properties[syn_receptor][u_type][v_type] = {'weight': [], 'delay': []}
                    self.input_main_edge_counts[syn_receptor][u_type][v_type] = 0

                self.input_main_edge_lists[syn_receptor][u_type][v_type].append(synapse)
            else:
                if u_type not in self.main_edge_lists[syn_receptor]:
                    self.main_edge_lists[syn_receptor] = {u_type: {}}
                    self.main_edge_properties[syn_receptor] = {u_type: {}}
                    self.main_edge_counts[syn_receptor] = {u_type: {}}
                if v_type not in self.main_edge_lists[syn_receptor][u_type]:
                    self.main_edge_lists[syn_receptor][u_type][v_type] = []
                    self.main_edge_properties[syn_receptor][u_type][v_type] = {'weight': [], 'delay': []}
                    self.main_edge_counts[syn_receptor][u_type][v_type] = 0

                self.main_edge_lists[syn_receptor][u_type][v_type].append(synapse)

        # Create mappings to edge indicies to edges
        for receptor in self.input_main_edge_lists:
            for source in self.input_main_edge_lists[receptor]:
                for target in self.input_main_edge_lists[receptor][source]:
                    edge_list = self.input_main_edge_lists[receptor][source][target]
                    edge_list.sort()
                    count = self.input_main_edge_counts[receptor][source][target]
                    for u, v, weight, delay in edge_list:
                        self.input_main_edge_properties[receptor][source][target]['weight'].append(weight)
                        self.input_main_edge_properties[receptor][source][target]['delay'].append(delay)
                        self.input_main_edge_index_map[(u, v)] = (count, syn_receptor)
                        count += 1
                    self.input_main_edge_counts[receptor][source][target] = count
        for receptor in self.main_edge_lists:
            for source in self.main_edge_lists[receptor]:
                for target in self.main_edge_lists[receptor][source]:
                    edge_list = self.main_edge_lists[receptor][source][target]
                    edge_list.sort()
                    count = self.main_edge_counts[receptor][source][target]
                    for u, v, weight, delay in edge_list:
                        self.main_edge_properties[receptor][source][target]['weight'].append(weight)
                        self.main_edge_properties[receptor][source][target]['delay'].append(delay)
                        self.main_edge_index_map[(u, v)] = (count, syn_receptor)
                        count += 1
                    self.main_edge_counts[receptor][source][target] = count

        if self.collect_metrics:
            start = timer()
        self._create_pynn_network()
        self.set_input_spikes()
        if self.collect_metrics:
            self.metrics['embed_time'] = timer() - start

        if self.verbose and self.backend == BRIAN_BACKEND:
            print("Finished compiling")
            self.print_network_info()

    def run(self, n_steps=10, return_potentials=False):
        # Runs circuit for n_steps then returns data

        max_runtime = n_steps * self.defaults['min_delay']
        if self.collect_metrics:
            start = timer()
        self.pynn_sim.run(max_runtime)
        if self.collect_metrics:
            self.metrics['runtime'].append(timer() - start)

        main_data = {}
        main_spiketrains = {}
        input_data = {}
        input_spiketrains = {}
        for neuron_type in self.main_populations:
            data = self.main_populations[neuron_type].get_data()
            main_data[neuron_type] = data
            main_spiketrains[neuron_type] = data.segments[-1].spiketrains

        for neuron_type in self.input_populations:
            data = self.input_populations[neuron_type].get_data()
            input_data[neuron_type] = data
            input_spiketrains[neuron_type] = data.segments[-1].spiketrains

        if self.return_potentials:
            main_voltage = {}
            potentials = pd.DataFrame({'neuron_number': [], 'potential': []})
            for neuron in self.main_neurons:
                if neuron in self.output_neurons:
                    neuron_type = self.neuron_type_map[neuron]
                    data = main_data[neuron_type].segments[-1].analogsignals
                    pynn_index = self.neuron_index_map[neuron]
                    voltage = data[0][pynn_index]
                    if neuron_type not in main_voltage:
                        main_voltage[neuron_type] = {}
                    main_voltage[neuron_type][pynn_index] = voltage
                    potentials = potentials.append(
                                              {
                                                'neuron_number': pynn_index,
                                                'potential': voltage,
                                                },
                                              ignore_index=True,
                                              )

        spikes = {}
        for neuron in self.main_neurons:
            if neuron in self.output_neurons:
                pynn_index = self.neuron_index_map[neuron]
                neuron_type = self.neuron_type_map[neuron]
                spiketrain = main_spiketrains[neuron_type][pynn_index]
                if self.verbose:
                    print("---results for: {}".format(neuron))
                    if self.return_potentials:
                        print("voltage  {}".format(main_voltage[neuron_type][pynn_index]))
                    print("spiketimes  {}".format(spiketrain))

                if spiketrain.any() or len(np.array(spiketrain)) > 0:
                    neuron_number = self.fugu_scaffold.graph.node[neuron]['neuron_number']
                    for time in np.array(spiketrain):
                        if time not in spikes:
                            spikes[time] = set()
                        spikes[time].add(neuron_number)

        if self.show_plots and self.return_potentials:
            from pyNN.utility.plotting import Figure, Panel
            import matplotlib.pyplot as plt
            for neuron_type in main_data:
                labels = []
                for neuron in self.neuron_index_map:
                    if self.neuron_type_map[neuron] == neuron_type:
                        labels.append(neuron)
                segment = main_data[neuron_type].segments[-1]
                vm = segment.analogsignals[0]
                plt.plot(vm.times, vm)
            plt.legend()
            plt.show()

        for neuron in self.input_neurons:
            if self.report_all or neuron in self.output_neurons:
                input_index = self.neuron_index_map[neuron]
                input_type = self.neuron_type_map[neuron]
                spiketrain = input_spiketrains[input_type][input_index]
                if self.verbose:
                    print("---results for: {}".format(neuron))
                    print("spiketimes  {}".format(spiketrain))
                if spiketrain.any() or len(np.array(spiketrain)) > 0:
                    for time in np.array(spiketrain):
                        if time not in spikes:
                            spikes[time] = set()
                        spikes[time].add(self.fugu_scaffold.graph.node[neuron]['neuron_number'])

        spike_result = pd.DataFrame({'time': [], 'neuron_number': []})
        for time in spikes:
            mini_df = pd.DataFrame()
            times = [time for i in spikes[time]]
            mini_df['time'] = times
            mini_df['neuron_number'] = list(spikes[time])
            spike_result = spike_result.append(mini_df, sort=False)

        if self.return_potentials:
            return spike_result, potentials
        else:
            return spike_result

    def cleanup(self):
        # Deletes/frees neurons and synapses
        if self.verbose:
            print("Cleaning pynn sim")
        self.pynn_sim.end()
        self._initialize_members()

    def reset(self):
        # Reset time-step and reset neuron/synapse properties
        if self.verbose:
            print(">>>Resetting<<<")
        self.pynn_sim.reset()

    def set_properties(self, properties={}):
        # Set properties for specific neurons and synapses
        # properties = dictionary of property for bricks
        if self.verbose:
            print("Before set_properties")
            print(properties)
            self.print_network_info()

        for brick in properties:
            if brick != 'compile_args':
                brick_tag = self.fugu_scaffold.name_to_tag[brick]
                brick_id = self.fugu_scaffold.brick_to_number[brick_tag]
                changes = self.fugu_scaffold.circuit.nodes[brick_id]['brick'].set_properties(properties[brick])
                if changes:
                    neuron_props, synapse_props = changes
                    if self.verbose:
                        print("Neuron changes: {}".format(neuron_props))
                        print("Synapse changes: {}".format(synapse_props))
                    if neuron_props:
                        for neuron in neuron_props:
                            neuron_type = self.neuron_type_map[neuron]
                            neuron_index = self.neuron_index_map[neuron]
                            properties = neuron_props[neuron]
                            for prop in properties:
                                prop_value = properties[prop]
                                if prop == 'potential':
                                    self.initial_potential[neuron_type][neuron_index] = prop_value
                                elif prop == 'threshold':
                                    prop_value = prop_value if prop_value > 0.0 else 0.01
                                    self.property_values[neuron_type]['v_thresh'][neuron_index] = prop_value
                                elif prop == 'current_offset':
                                    self.property_values[neuron_type]['i_offset'][neuron_index] = prop_value

                    if synapse_props:
                        for synapse in synapse_props:
                            if type(synapse) is tuple:
                                pre, post = [self.neuron_index_map[p] for p in synapse]
                                is_input = pre in self.input_neurons
                                pre_type, post_type = [self.neuron_type_map[p] for p in synapse]
                                if is_input:
                                    edge_index, receptor = self.input_main_edge_index_map[(pre, post)]
                                else:
                                    edge_index, receptor = self.main_edge_index_map[(pre, post)]
                                if self.verbose:
                                    print(synapse, pre_type, post_type, edge_index, receptor, synapse_props[synapse])

                                properties = synapse_props[synapse]
                                for prop in properties:
                                    value = properties[prop]
                                    if is_input:
                                        self.input_main_edge_properties[receptor][pre_type][post_type][prop][edge_index] = value
                                    else:
                                        self.main_edge_properties[receptor][pre_type][post_type][prop][edge_index] = value
                            else:
                                pre = synapse
                                is_input = pre in self.input_neurons
                                pre_index = self.neuron_index_map[pre]
                                pre_type = self.neuron_type_map[pre]
                                properties = synapse_props[pre]
                                for fugu_edge in self.fugu_scaffold.graph.edges:
                                    if fugu_edge[0] == pre:
                                        post_index = self.neuron_index_map[fugu_edge[1]]
                                        post_type = self.neuron_type_map[fugu_edge[1]]
                                        pynn_edge = (pre_index, post_index)
                                        if is_input:
                                            edge_index, receptor = self.input_main_edge_index_map[pynn_edge]
                                        else:
                                            edge_index, receptor = self.main_edge_index_map[pynn_edge]
                                        for prop in properties:
                                            value = properties[prop]
                                            if is_input:
                                                self.input_main_edge_properties[receptor][pre_type][post_type][prop][edge_index] = value
                                            else:
                                                self.main_edge_properties[receptor][pre_type][post_type][prop][edge_index] = value

        for neuron_type in self.main_populations:
            self.main_populations[neuron_type].set(**self.property_values[neuron_type])
            self.main_populations[neuron_type].initialize(v=self.initial_potentials[neuron_type])
        self.set_input_spikes()

        if self.backend != SPINNAKER_BACKEND:
            for receptor in self.input_main_synapses:
                for source_type in self.input_main_synapses[receptor]:
                    for target_type in self.input_main_synapses[receptor][source_type]:
                        self.input_main_synapses[receptor][source_type][target_type].set(
                               **self.input_main_edge_properties[receptor][source_type][target_type]
                               )
            for receptor in self.main_synapses:
                for source_type in self.main_synapses[receptor]:
                    for target_type in self.main_synapses[receptor][source_type]:
                        self.main_synapses[receptor][source_type][target_type].set(
                               **self.main_edge_properties[receptor][source_type][target_type]
                               )

        if self.verbose:
            print("After set_properties")
            self.print_network_info()

    def set_input_spikes(self):
        initial_spikes = CalculateSpikeTimes(self.fugu_scaffold.circuit)
        processed_spikes = {}
        for time_step in sorted(initial_spikes.keys()):
            for neuron in initial_spikes[time_step]:
                if neuron not in processed_spikes:
                    processed_spikes[neuron] = []
                processed_spikes[neuron].append(time_step)

        input_spikes = {source: [[] for n in self.input_neurons] for source in self.input_neuron_type_names}

        if self.verbose:
            print("Processed input value: {}".format(processed_spikes))
        for neuron in self.input_neurons:
            if neuron in processed_spikes:
                spike_array = [spike_time * self.defaults['min_delay'] for spike_time in processed_spikes[neuron]]
                neuron_type = self.neuron_type_map[neuron]
                if self.backend == BRIAN_BACKEND:
                    input_spikes[neuron_type][self.neuron_index_map[neuron]] = [val / 10.0 for val in spike_array]
                else:
                    input_spikes[neuron_type][self.neuron_index_map[neuron]] = spike_array

        if self.verbose:
            print("Input spikes: {}".format(input_spikes))

        for neuron_type in input_spikes:
            if len(input_spikes[neuron_type]) > 0:
                self.input_populations[neuron_type].set(spike_times=input_spikes[neuron_type])
