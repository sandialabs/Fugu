#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:37:44 2019

@author: smusuva
"""
import networkx as nx
import numpy as np
from warnings import warn

from ..backends.backend import Backend, snn_Backend, ds_Backend
from ..backends.pynn_backend import pynn_Backend
from ..utils.export_utils import results_df_from_dict


class Scaffold:
    """Class to handle a scaffold of bricks"""

    supported_backends = ['ds', 'snn', 'ds_legacy', 'snn_legacy', 'pynn']

    def __init__(self):
        self.circuit = nx.DiGraph()
        self.pos = {}
        self.count = {}
        self.graph = None
        self.is_built = False
        self.metrics = None

    def add_brick(self, brick_function, input_nodes=[-1], metadata=None, name=None, output=False):
        """
        Add a brick to the scaffold.

        Arguments:
            + brick_function - object of type brick
            + input_nodes - list of node numbers (Default: [-1])
            + dimesionality -  dictionary of shapes and parameters of the brick (Default: None)
            + name - string of the brick's name (Default: none)
            + output - bool flag to indicate if a brick is an output brick (Default: False)

        Returns:
            + None

        Exceptions:
            + Raises ValueError if node name is already used.
            """

        if name is None and brick_function.name is not None:
            name = brick_function.name
        elif name is None:
            brick_type = str(type(brick_function))
            if brick_type not in self.count:
                self.count[brick_type] = 0
            name = "{}_{}".format(brick_type, self.count[brick_type])
            self.count[brick_type] = self.count[brick_type]+1
        elif name in self.circuit.nodes:
            raise ValueError("Node name already used.")
        if name in [self.circuit.nodes[node]['name'] for node in self.circuit.nodes]:
            raise ValueError("Node name already used.")

        if brick_function.name is None:
            brick_function.name = name

        node_number = self.circuit.number_of_nodes()
        self.circuit.add_node(
                       node_number,
                       name=name,
                       brick=brick_function,
                       )  # ,metadata=metadata)

        # Make sure we're working with a list of inputs
        if type(input_nodes) is not list:
            input_nodes = [input_nodes]

        # Make sure our inputs are integer formatted
        node_names = {self.circuit.nodes[node]['name']: node for node in self.circuit.nodes}

        processed_input_nodes = []
        for node in input_nodes:
            processed_node = node

            if processed_node == 'input':
                processed_node = -2

            if type(processed_node) is str:
                processed_node = node_names[processed_node]

            if type(processed_node) is tuple and type(processed_node[0]) is str:
                processed_node = (node_names[processed_node[0]], processed_node[1])

            # Replace -1 with last node
            if processed_node == -1:
                processed_node = node_number - 1

            # Create tuples for (node, channel) if not already done
            if type(processed_node) is int:
                processed_node = (processed_node, 0)
            processed_input_nodes.append(processed_node)

        # Process inputs
        for node in [node[0] for node in processed_input_nodes]:
            if node < -1:
                self.circuit.nodes[node_number]['layer'] = 'input'
            else:
                self.circuit.add_edge(node, node_number)
        self.circuit.nodes[node_number]['input_nodes'] = processed_input_nodes
        if output:
            self.circuit.nodes[node_number]['layer'] = 'output'

    def resolve_timing(self):
        # Set weights to equal source node timing (T_out + D?)

        # From end to front
        #     Identify predecessor nodes; if 1, then pass, if >1, then compute longest path to each source
        #     If longest path size is different, then add delay node to shorter one to make equivalent.

        # Have to define D as time from first input within T_in to first output within T_out.
        # Have to assume T_in(2)= T_out(1)
        nodes = list(self.circuit.nodes())
        edges = list(self.circuit.edges())
        for edge in edges:
            self.circuit.edges[edge[0], edge[1]]['weight'] = self.circuit.nodes[edge[0]]['brick'].metadata['D']

        for node in reversed(nodes):

            # loop backwards through nodes
            # Find predecessor node for 'node'
            pred_nodes = list(self.circuit.predecessors(node))
            print(node, list(pred_nodes), len(list(pred_nodes)), len(pred_nodes))

            if len(list(pred_nodes)) < 2:
                pass
            else:
                distances = []
                for target_node in list(pred_nodes):
                    distance_guess = 0
                    target_paths = nx.all_simple_paths(self.circuit, 0, target_node)
                    for path in map(nx.utils.pairwise, target_paths):
                        path_lengths = self.get_weight(path)
                        if path_lengths > distance_guess:
                            distance_guess = path_lengths

                    distance_guess = distance_guess+self.circuit.edges[target_node, node]['weight']
                    print(target_node, distance_guess)
                    distances.append(distance_guess)

                # Now, we need to add delay nodes to paths less than longest distance

                max_value = max(distances)
                max_index = distances.index(max_value)
                print(list(distances), max_index, max_value)

                for i in range(0, len(pred_nodes)):
                    # Check if this path needs a delay node
                    if(distances[i] < max_value):
                        target_node = pred_nodes[i]
                        print(
                          'Adding delay node of length {} between {} and {}'.format(
                                                                               max_value-distances[i],
                                                                               target_node,
                                                                               node,
                                                                               )
                          )
                        N_delay = self.circuit.nodes[target_node]['N_out']
                        self.circuit.add_node(
                                       target_node+0.5,
                                       brick=Delay,
                                       N_in=N_delay,
                                       N_out=N_delay,
                                       T_in=1,
                                       T_out=1,
                                       D=max_value-distances[i],
                                       layer='delay',
                                       )
                        self.pos[target_node + 0.5] = np.array([target_node + 0.5, self.pos[target_node][1]])
                        self.circuit.remove_edge(target_node, node)
                        self.circuit.add_edge(
                                       target_node,
                                       target_node + 0.5,
                                       weight=self.circuit.nodes[target_node]['D'],
                                       )
                        self.circuit.add_edge(
                                       target_node + 0.5,
                                       node,
                                       weight=self.circuit.nodes[target_node + 0.5]['D'],
                                       )

    def get_weight(self, path):
        total_len = 0
        for i in list(path):
            total_len = total_len + self.circuit.edges[i]['weight']

        return total_len

    def all_nodes_built(self, verbose=0):
        """
        Check if all nodes are built.

        Arguments:
            + verbose - int to indicate level of verbosity (Default: 0 to indicate no messages)
        Returns:
            + bool with True if all nodes are built, Fase otherwise
            """

        b = True
        for node in self.circuit.nodes:
            b = b and self.circuit.nodes[node]['brick'].is_built
        if verbose > 0:
            print("Nodes built:")
            for node in self.circuit.nodes:
                print("{}:{}".format(node, self.circuit.nodes[node]['brick'].is_built))
        return b

    def all_in_neighbors_built(self, node):
        """
        Check if all neighbors of a node are built.

        Arguments:
            + node - node whose neighbors are checked

        Returns:
            + bool - indicates if all neighbors are built.
            """

        in_neighbors = [edge[0] for edge in self.circuit.in_edges(nbunch=node)]
        b = True
        for neighbor in in_neighbors:
            b = b and self.circuit.nodes[neighbor]['brick'].is_built
        return b

    def _assign_brick_names(self, built_graph, name, field='brick'):
        new_nodes = [new_node for new_node, node_value in built_graph.nodes(data=True) if field not in node_value]
        for new_node in new_nodes:
            built_graph.nodes[new_node]['brick'] = name
        return built_graph

    def lay_bricks(self, verbose=0):
        """
        Build a computational graph that can be used by the backend.

        Arguments:
            + verbose - int value to specify level of verbosity (Default: 0 to indicate None)

        Returns:
            networkX diGraph
        """
        built_graph = nx.DiGraph()
        # Handle Input Nodes
        if verbose > 0:
            print("Laying Input Bricks.")
        for node in [node for node in self.circuit.nodes
                     if 'layer' in self.circuit.nodes[node]
                        and self.circuit.nodes[node]['layer'] == 'input']:
            (built_graph,
             metadata,
             control_nodes,
             output_lists,
             output_codings) = self.circuit.nodes[node]['brick'].build(built_graph, None, None, None, None)
            self._assign_brick_names(built_graph,  self.circuit.nodes[node]['name'])
            self.circuit.nodes[node]['output_lists'] = output_lists
            self.circuit.nodes[node]['output_codings'] = output_codings
            self.circuit.nodes[node]['metadata'] = metadata
            self.circuit.nodes[node]['control_nodes'] = control_nodes
            if verbose > 0:
                print("Completed: {}".format(node))
        while not self.all_nodes_built(verbose=verbose):
            # Over unbuilt, ready edges
            for node in [node for node in self.circuit.nodes
                         if (not self.circuit.nodes[node]['brick'].is_built)
                         and self.all_in_neighbors_built(node)]:
                inputs = {}
                if verbose > 0:
                    print('Laying Brick: '.format(node))
                for input_number in range(0, len(self.circuit.nodes[node]['input_nodes'])):
                    if verbose > 0:
                        print("Processing input: {}".format(input_number))
                    inputs[input_number] = {
                                             'input_node': self.circuit.nodes[node]['input_nodes'][input_number][0],
                                             'input_channel': self.circuit.nodes[node]['input_nodes'][input_number][1],
                                             }
                metadata = []
                control_nodes = []
                input_lists = []
                input_codings = []
                for key in inputs:
                    circuit_node = self.circuit.nodes[inputs[key]['input_node']]
                    input_channel = inputs[key]['input_channel']
                    metadata.append(circuit_node['metadata'])
                    control_nodes.append(circuit_node['control_nodes'][input_channel])
                    input_lists.append(circuit_node['output_lists'][input_channel])
                    input_codings.append(circuit_node['output_codings'][input_channel])

                (built_graph,
                 metadata,
                 control_nodes,
                 output_lists,
                 output_codings) = self.circuit.nodes[node]['brick'].build(
                                                                       built_graph,
                                                                       metadata,
                                                                       control_nodes,
                                                                       input_lists,
                                                                       input_codings,
                                                                       )
                self._assign_brick_names(built_graph,  self.circuit.nodes[node]['name'])
                self.circuit.nodes[node]['metadata'] = metadata
                self.circuit.nodes[node]['output_codings'] = output_codings
                self.circuit.nodes[node]['output_lists'] = output_lists
                self.circuit.nodes[node]['control_nodes'] = control_nodes
                if verbose > 0:
                    print("Complete.")
        for i, n in enumerate(built_graph.nodes):
            built_graph.nodes[n]['neuron_number'] = i
        self.is_built = True
        self.graph = built_graph
        return built_graph

    def _create_ds_injection(self):
        warn("This function is depreciated and will be removed when 'ds_legacy' is removed.")
        # find input nodes
        import torch
        input_nodes = [node for node in self.circuit.nodes if ('layer' in self.circuit.nodes[node])
                       and (self.circuit.nodes[node]['layer'] == 'input')]
        injection_tensors = {}
        for input_node in input_nodes:
            input_spikes = self.circuit.nodes[input_node]['brick'].get_input_value()
            for t in range(0, input_spikes.shape[-1]):
                if t not in injection_tensors:
                    injection_tensors[t] = torch.zeros((self.graph.number_of_nodes(),)).float()
                for local_idx, neuron in enumerate(self.circuit.nodes[input_node]['output_lists'][0]):
                    tensor_idx = list(self.graph.nodes).index(neuron)
                    injection_tensors[t][tensor_idx] += input_spikes[local_idx, t]
        return injection_tensors

    def evaluate(self, max_runtime=10, backend='ds', record='output', record_all=False, backend_args={}):
        """
        Run the computational graph through the backend.

        Arguments:
            + max_runtime - int value to specify number of time steps (Default: 10)
            + backend - string value of the backend simulator or device name (Default: 'ds')
            + record_all - bool value to indicate if all neurons spikes are to be recorded (Default: False)

        Returns:
            + dictionary of time step and spiking neurons. (if record_all is True, all spiking neurons are shown
            else only the output neurons)

        Exceptions:
            + ValueError if backend is not in list of supported backends
        """

        if type(backend) is str:
            if backend not in Scaffold.supported_backends:
                raise ValueError("Backend " + str(backend) + " not supported.")
            if backend == 'ds_legacy':
                warn("'ds_legacy' option uses old instantiation of backend object.  Use 'ds' in the future.")
                from fugu.backends.ds import run_simulation
                injection_values = self._create_ds_injection()
                for node in self.circuit.nodes:
                    if 'layer' in self.circuit.nodes[node] and self.circuit.nodes[node]['layer'] == 'output':
                        for o_list in self.circuit.nodes[node]['output_lists']:
                            for neuron in o_list:
                                self.graph.nodes[neuron]['record'] = ['spikes']
                ds_graph = nx.convert_node_labels_to_integers(self.graph)
                ds_graph.graph['has_delay'] = True
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
                            if entry[0] not in spike_result:
                                spike_result[entry[0]] = []
                            spike_result[entry[0]].extend(entry[1].tolist())
                return results_df_from_dict(spike_result, 'time', 'neuron_number')
            if backend == 'snn_legacy':
                warn("'snn_legacy' option uses old instantiation of backend object.  Use 'snn' in the future.")
                from fugu.backends.sushi_chef import serve_fugu_to_snn
                spike_result = serve_fugu_to_snn(
                                 self.circuit,
                                 self.graph,
                                 n_steps=max_runtime,
                                 record_all=record_all,
                                 ds_format=True,
                                 )
                return results_df_from_dict(spike_result, 'time', 'neuron_number')
            if backend == 'ds':
                backend = ds_Backend()
            if backend == 'snn':
                backend = snn_Backend()
                backend_args['ds_format'] = True
            if backend == 'pynn':
                backend = pynn_Backend()
                self.metrics = backend.GetMetrics()
        if not issubclass(type(backend), Backend):
            raise ValueError("Invalid backend option.")
        results = backend.serve(
                    self,
                    n_steps=max_runtime,
                    record=record,
                    batch=True,
                    summary_steps=None,
                    record_all=record_all,
                    backend_args=backend_args,
                    )
        return results

    def summary(self, verbose=0):

        """Display a summary of the scaffold."""

        print("Scaffold is built: {}".format(self.is_built))
        print("-------------------------------------------------------")
        print("List of Bricks:")
        print("\r\n")
        for i, node in enumerate(self.circuit.nodes):
            print("Brick No.: {}".format(i))
            print("Brick Name: {}".format(self.circuit.nodes[node]['name']))
            print(self.circuit.nodes[node])
            print("Brick is built: {}".format(self.circuit.nodes[node]['brick'].is_built))
            print("\r\n")
        print("-------------------------------------------------------")
        print("\r\n")
        print("-------------------------------------------------------")
        print("List of Brick Edges:")
        print("\r\n")
        for i, edge in enumerate(self.circuit.edges):
            print("Edge: {}".format(edge))
            print(self.circuit.edges[edge])

        if verbose > 0:
            print("-------------------------------------------------------")
            print("\r\n")

            if self.graph is not None:
                print("List of Neurons:")
                print("\r\n")
                print("Neuron Number | Neuron Name | Neuron Properties")
                for i, neuron in enumerate(sorted(self.graph.nodes)):
                    print(str(i) + " | " + str(neuron) + " | " + str(self.graph.nodes[neuron]))
                print("\r\n")
                print("-------------------------------------------------------")
                print("List of Synapses:")
                print("\r\n")
                print("Synapse Between | Synapse Properties" if verbose > 1 else "Syanpse Between")
                for i, synapse in enumerate(self.graph.edges):
                    print(str(synapse) + " | " + str(self.graph.edges[synapse]) if verbose > 1 else str(synapse))
