import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from utils import results_df_from_dict
import pandas as pd
from warnings import warn

default_brick_metadata = {
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
    """Class to handle a scaffold of bricks"""

    supported_backends = ['ds', 'snn', 'ds_legacy', 'snn_legacy']

    def __init__(self):
        self.circuit = nx.DiGraph()
        self.pos = {}
        self.count = {}
        self.graph = None
        self.is_built = False

    def add_brick(self, brick_function, input_nodes=[-1], metadata = None, name=None, output=False):
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
        self.circuit.add_node(node_number, name=name,
                             brick=brick_function)#,metadata=metadata)

        #Make sure we're working with a list of inputs
        if type(input_nodes) is not list:
            input_nodes = [input_nodes]

        #Make sure our inputs are integer formatted
        input_nodes = [-2 if node == 'input' else node for node in input_nodes]
        node_names = {self.circuit.nodes[node]['name']:node for node in self.circuit.nodes}
        input_nodes = [node_names[node] if type(node) is str else node for node in input_nodes]
        input_nodes = [(node_names[node[0]], node[1]) if type(node) is tuple and type(node[0]) is str else node for node in input_nodes]


        #Replace -1 with last node
        input_nodes = [node_number-1 if node==-1 else node for node in input_nodes]

        #Create tuples for (node, channel) if not already done
        input_nodes = [(node, 0) if type(node) is int else node for node in input_nodes]

        #Process inputs
        for node in [node[0] for node in input_nodes]:
            if node < -1:
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
            self.circuit.edges[edge[0], edge[1]]['weight']=self.circuit.node[edge[0]]['brick'].metadata['D']

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
                        print('Adding delay node of length {} between {} and {}'.format(max_value-distances[i],
                                                                                        target_node,
                                                                                        node))
                        N_delay=self.circuit.nodes[target_node]['N_out']
                        self.circuit.add_node(target_node+0.5,
                                              brick=Delay,
                                              N_in=N_delay,
                                              N_out=N_delay,
                                              T_in=1,
                                              T_out=1,
                                              D=max_value-distances[i],
                                              layer='delay')
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
        if verbose>0:
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
    
    def _assign_brick_names(self,built_graph, name, field_to_check='brick'):
        new_nodes = [new_node for new_node, node_value in built_graph.nodes(data=True) if field_to_check not in node_value]
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
             output_codings) = self.circuit.nodes[node]['brick'].build(built_graph,
                              None,
                              None,
                              None,
                              None)
            self._assign_brick_names(built_graph,  self.circuit.nodes[node]['name'])
            self.circuit.nodes[node]['output_lists'] = output_lists
            self.circuit.nodes[node]['output_codings'] = output_codings
            self.circuit.nodes[node]['metadata'] = metadata
            self.circuit.nodes[node]['control_nodes'] = control_nodes
            if verbose > 0:
                print("Completed: {}".format(node))
        while not self.all_nodes_built(verbose=verbose):
            #Over unbuilt, ready edges
            for node in [node for node in self.circuit.nodes
                         if (not self.circuit.nodes[node]['brick'].is_built)
                         and self.all_in_neighbors_built(node)]:
                inputs = {}
                if verbose > 0:
                    print('Laying Brick: '.format(node))
                for input_number in range(0,len(self.circuit.nodes[node]['input_nodes'])):
                    if verbose > 0:
                        print("Processing input: {}".format(input_number))
                    inputs[input_number] = {'input_node':self.circuit.nodes[node]['input_nodes'][input_number][0],
                                           'input_channel':self.circuit.nodes[node]['input_nodes'][input_number][1]}
                metadata = [self.circuit.nodes[inputs[key]['input_node']]['metadata']
                                  for key in inputs]
                control_nodes = [self.circuit.nodes[inputs[key]['input_node']]['control_nodes'][inputs[key]['input_channel']] for key in inputs]
                input_lists = [self.circuit.nodes[inputs[key]['input_node']]['output_lists'][inputs[key]['input_channel']] for key in inputs]
                input_codings = [self.circuit.nodes[inputs[key]['input_node']]['output_codings'][inputs[key]['input_channel']] for key in inputs]
                (built_graph,
                 metadata,
                 control_nodes,
                 output_lists,
                 output_codings) = self.circuit.nodes[node]['brick'].build(built_graph,
                                                      metadata,
                                                      control_nodes,
                                                      input_lists,
                                                      input_codings)
                self._assign_brick_names(built_graph,  self.circuit.nodes[node]['name'])
                self.circuit.nodes[node]['metadata'] = metadata
                self.circuit.nodes[node]['output_codings'] = output_codings
                self.circuit.nodes[node]['output_lists'] = output_lists
                self.circuit.nodes[node]['control_nodes'] = control_nodes
                if verbose > 0:
                    print("Complete.")
        for i, n in enumerate(built_graph.nodes):
           built_graph.nodes[n]['neuron_number'] = i
        self.is_built=True
        self.graph = built_graph
        return built_graph
    def _create_ds_injection(self):
        warn("This function should only be called with 'ds_legacy' backend.  'ds_legacy' is deprecated and will be removed.  This method will be removed with it.")
        #find input nodes
        import torch
        input_nodes = [node  for node in self.circuit.nodes if ('layer' in self.circuit.nodes[node] )
                       and (self.circuit.nodes[node]['layer'] == 'input')]
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
                from ds import run_simulation
                injection_values = self._create_ds_injection()
                for node in self.circuit.nodes:
                    if 'layer' in self.circuit.nodes[node] and self.circuit.nodes[node]['layer'] == 'output':
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
                            if entry[0] not in spike_result:
                                spike_result[entry[0]] = []
                            spike_result[entry[0]].extend(entry[1].tolist())
                return results_df_from_dict(spike_result,'time','neuron_number')
            if backend == 'snn_legacy':
                warn("'snn_legacy' option uses old instantiation of backend object.  Use 'snn' in the future.")
                from backends.sushi_chef import serve_fugu_to_snn
                spike_result = serve_fugu_to_snn(self.circuit, self.graph, n_steps=max_runtime, record_all=record_all, ds_format=True)
                return results_df_from_dict(spike_result,'time','neuron_number')
            if backend == 'ds':
                backend = ds_Backend()
            if backend == 'snn':
                backend = snn_Backend()
        if not issubclass(type(backend),Backend):
            raise ValueError("Invalid backend option.")
        results = backend.serve(self, 
                      n_steps=max_runtime, 
                      record=record, 
                      batch=True,
                      summary_steps=None,
                      record_all=record_all,
                      backend_args=backend_args)
        return results
        
            
    def summary(self, verbose=0):

        """Display a summary of the scaffold."""

        print("Scaffold is built: {}".format(self.is_built))
        print("-------------------------------------------------------")
        print("List of Bricks:")
        print("\r\n")
        for i,node in enumerate(self.circuit.nodes):
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
                for i, neuron in enumerate(self.graph.nodes):
                    print("Neuron Number | Neuron Name | Neuron Properties")
                    print(str(i) + " | " + str(neuron) + " | " + str(self.graph.nodes[neuron]))
                print("\r\n")
                print("-------------------------------------------------------")
                print("List of Synapses:")
                print("\r\n")
                for i, synapse in enumerate(self.graph.edges):
                    print("Synapse Between | Synapse Properties" if verbose > 1 else "Syanpse Between")
                    print(str(synapse) + " | " + str(self.graph.edges[synapse]) if verbose > 1 else str(synapse))

class Backend(ABC):
    """Abstract Base Class definition of a Backend"""
    def __init__(self):
        self.features = {'supports_stepping':False,
                         'supports_streaming_input':False,
                         'supports_additive_leak':False,
                         'supports_hebbian_learning':False}
        
    def serve(self,
            scaffold,
            n_steps=None,
            max_steps=None,
            record='output',
            record_all=False,
            summary_steps=None,
            batch=True,
            backend_args={}):
        if max_steps is not None:
            raise ValueError("Self-Halt is not yet implemented")
        if max_steps is not None and n_steps is not None:
            raise ValueError("Cannot specify both n_steps and max_steps")
        if record_all:
            record = 'all'
           
        input_nodes = [node  for node in scaffold.circuit.nodes if ('layer' in scaffold.circuit.nodes[node] )
                       and (scaffold.circuit.nodes[node]['layer'] == 'input')]
        if batch:
            input_values = {}
            for timestep in range(0,n_steps):
                input_values[timestep]=deque()
            for input_node in input_nodes:
                for timestep, spike_list in enumerate(scaffold.circuit.nodes[input_node]['brick']):
                    input_values[timestep].extend(spike_list)
            results = self.batch(scaffold, input_values, n_steps, record, backend_args)
        else:
            if not self.features['supports_stepping']:
                raise ValueError("Backend does not support stepping. Use a batch mode.")
            results = pd.DataFrame()
            halt = False
            step_number = 0
            while step_number < n_steps and not halt:
                input_values = deque()
                for input_node in input_nodes:
                    input_values.extend(next(scaffold.circuit.nodes[input_node]['brick'],[]))
                result, halt = self.stream(scaffold, input_values,  record, backend_args)
                results = results.append(result)
        return results
    
    @abstractmethod
    def stream(self,
               scaffold,
               input_values,
               stepping,
               record,
               backend_args):
        pass
    
    @abstractmethod
    def batch(self,
             scaffold,
             input_values,
             n_steps,
             record,
             backend_args):
        pass

class snn_Backend(Backend):
    """Backend for Srideep's Noteworthy Network (SNN)"""
    def __init__(self):
        super(Backend, self).__init__()

    def _serve_fugu_to_snn(self, fugu_circuit, fugu_graph, n_steps=1, record_all=False, ds_format=False):
        '''Reads in a built fugu graph and converts it to a spiking neural network 
        and runs it for n steps'''
        import backends.SpikingNeuralNetwork as snn
        nn = snn.NeuralNetwork()
        neuron_dict = {}
        
        
        ''' Add Neurons '''
        #Add in input and output neurons. Use the fugu_circuit information to identify input and output layers
        #For input nodes, create input neurons and identity and associate the appropriate inputs to it
        #for output neurons, obtain neuron parameters from fugu_graph and create LIFNeurons
        #Add neurons to spiking neural network
        for node, vals in fugu_circuit.nodes.data():
            if 'layer' in vals: 
                if vals['layer'] == 'input':
                    
                    input_spike_lists = [input_spikes for input_spikes in fugu_circuit.nodes[node]['brick']]
                    input_values = {}
                    for neuron in fugu_circuit.nodes[node]['output_lists'][0]:
                        input_values[neuron] = []
                    for timestep, spike_list in enumerate(input_spike_lists):
                        for neuron in spike_list:
                            input_values[neuron].append(1)
                        for neuron in input_values:
                            if(len(input_values[neuron]) < timestep+1):
                                input_values[neuron].append(0)
                            
                    
                    for neuron in fugu_circuit.nodes[node]['output_lists'][0]:
                        rc = True if record_all else vals.get('record', False)
                        neuron_dict[neuron] = snn.InputNeuron(neuron, record=rc)
                        neuron_dict[neuron].connect_to_input(input_values[neuron])
                        nn.add_neuron(neuron_dict[neuron])
                if vals['layer'] == 'output':
                    for olist in fugu_circuit.nodes[node]['output_lists']:
                        for neuron in olist:
                            params = fugu_graph.nodes[neuron]
                            th = params.get('threshold', 0.0)
                            rv = params.get('reset_voltage', 0.0)
                            lk = params.get('leakage_constant', 1.0)
                            vol =params.get('voltage', 0.0)
                            if 'potential' in params:
                                vol = params['potential']
                            if 'decay' in params:
                                lk = 1.0 - params['decay']
                            neuron_dict[neuron] = snn.LIFNeuron(neuron, threshold=th, reset_voltage=rv, leakage_constant=lk, voltage=vol, record=True)
                            nn.add_neuron(neuron_dict[neuron])
        #add other neurons from fugu_graph to spiking neural network
        #parse through the fugu_graph and if a neuron is not present in spiking neural network, add to it.                    
        for neuron, params in fugu_graph.nodes.data():
            if neuron not in neuron_dict.keys():
                th = params.get('threshold', 0.0)
                rv = params.get('reset_voltage', 0.0)
                lk = params.get('leakage_constant', 1.0)
                vol = params.get('voltage', 0.0)
                if 'potential' in params:
                    vol = params['potential']
                if 'decay' in params:
                    lk = 1.0 - params['decay']
                rc = True if record_all else params.get('record', False)
                neuron_dict[neuron] = snn.LIFNeuron(neuron, threshold=th, reset_voltage=rv, leakage_constant=lk, voltage=vol, record=rc)
                nn.add_neuron(neuron_dict[neuron])
        
        ''' Add Synapses '''
        #add synapses from fugu_graph edge information        
        for n1, n2, params in fugu_graph.edges.data():
            delay = int(params.get('delay', 1))
            wt = params.get('weight', 1.0)
            syn = snn.Synapse(neuron_dict[n1], neuron_dict[n2], delay=delay, weight=wt)
            nn.add_synapse(syn)
        
        del neuron_dict
        
        ''' Run the Simulator '''
        df = nn.run(n_steps=n_steps, debug_mode=False)
        res = {}
        #if ds format is required, convert neuron names to numbers and return dictionary
        #else return dataframe
        if ds_format:
            numerical_cols = {}
            for c in df.columns:
                numerical_cols[c] = fugu_graph.nodes[c]['neuron_number']
            df = df.rename(index=int, columns=numerical_cols) 
            
            for r in df.index:
                l = []
                for c in df.columns:
                    if df.loc[r][c] == 1:
                        l.append(c)
                    res[r] = l
    
            return res
        else:
            return df
    
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
        spike_result = self._serve_fugu_to_snn(scaffold.circuit, scaffold.graph, n_steps=n_steps, record_all=True if record=='all' else False, ds_format=True)
        return results_df_from_dict(spike_result,'time','neuron_number')
    
class ds_Backend(Backend):
    """Backend for the ds simulator"""
    def __init__(self):
        super(Backend,self).__init__()
        
    def _create_ds_injection(self,scaffold,input_values):
        #find input nodes
        import torch

        injection_tensors = {}
        for t in range(len(input_values)):
            injection_tensors[t] = torch.zeros((scaffold.graph.number_of_nodes(),)).float()
            spiking_neurons = [scaffold.graph.nodes[neuron]['neuron_number'] for neuron in input_values[t]]
            injection_tensors[t][spiking_neurons] = 1
        return injection_tensors    
    
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
        from ds import run_simulation
        injection_values = self._create_ds_injection(scaffold,input_values)
        if record =='output':
            for node in scaffold.circuit.nodes:
                if 'layer' in scaffold.circuit.nodes[node] and scaffold.circuit.nodes[node]['layer'] == 'output':
                    for o_list in scaffold.circuit.nodes[node]['output_lists']:
                        for neuron in o_list:
                            scaffold.graph.nodes[neuron]['record'] = ['spikes']
        ds_graph = nx.convert_node_labels_to_integers(scaffold.graph)
        ds_graph.graph['has_delay']=True
        if record == 'all':
            for neuron in ds_graph.nodes:
                ds_graph.nodes[neuron]['record'] = ['spikes']
        for neuron in ds_graph.nodes:
            if 'potential' not in ds_graph.nodes[neuron]:
                ds_graph.nodes[neuron]['potential'] = 0.0
        results = run_simulation(ds_graph,
                                 n_steps,
                                 injection_values)
        spike_result = pd.DataFrame({'time':[],'neuron_number':[]})
        for group in results:
            if 'spike_history' in results[group]:
                spike_history = results[group]['spike_history']
                for entry in spike_history:
                    mini_df = pd.DataFrame()
                    neurons = entry[1].tolist()
                    times = [entry[0]]*len(neurons)
                    mini_df['time'] = times
                    mini_df['neuron_number'] = neurons
                    spike_result = spike_result.append(mini_df)
        return spike_result


class Brick(ABC):
    """Abstract Base Class definition of a Brick class"""

    def __init__(self):
        self.name = "Empty Brick"
        self.is_built = False
        self.supported_codings = []

    @abstractmethod
    def build(self, graph,
                   metadata,
                   control_nodes,
                   input_lists,
                   input_codings):
        """
        Build the computational graph of the brick. Method must be defined in any class inheriting from Brick.

        Arguments:
            + graph - networkx graph
            + metadata - A dictionary of shapes and parameters
            + control_nodes - list of dictionary of auxillary nodes.  Acceptable keys include: 'complete' - A list of neurons that fire when the brick is done, 'begin' - A list of neurons that fire when the brick begins computation (used for temporal processing)
            + input_lists - list of lists of nodes for input neurons
            + input_codings - list of input coding types (as strings)
        """
        pass

class InputBrick(Brick):
    """Abstract Base class for handling inputs inherited from Brick"""
    
    def __init__(self):
        self.streaming = False
    
    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def get_input_value(self, t=None):
        """
        Abstract method to get input values. InputBricks must implement this method

        Arguments:
            + t - type of input (Default: None)
        """
        pass
    
class Vector_Input(InputBrick):
    """Class to handle a vector of spiking input. Inherits from InputBrick"""

    def __init__(self, spikes, time_dimension = False,
                 coding='Undefined', batchable = True, name=None):
        '''
        Construtor for this brick.
        Arguments:
		    + spikes - A numpy array of which neurons should spike at which times
			+ time_dimension - Time dimesion is included as dimension -1
			+ coding - Coding type to be represented.
            + batchable - True if input should represent static data; currently True is the only supported mode.
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
        '''
        super(InputBrick, self).__init__()
        self.vector = np.array(spikes)
        self.coding = coding
        self.time_dimension = time_dimension
        self.is_built = False
        self.batchable = batchable
        self.name = name
        self.index_map = None
        self.supported_codings = []
        self.metadata = {'D' : 0}
        self.current_time=0
        
    def __iter__(self):
        self.current_time=0
        return self
    
    def __next__(self):
        if self.vector.shape[-1] > self.current_time:
            self.current_time += 1
            this_time_vector = self.vector[...,self.current_time-1]
            local_idxs = np.array(np.where(this_time_vector))
            num_spikes = len(local_idxs[0])
            global_idxs = deque()
            for spike in range(num_spikes):
                idx_to_build = deque()
                for dimension in range(len(local_idxs)):
                    idx_to_build.append(local_idxs[dimension][spike])
                global_idxs.append(tuple(idx_to_build))
            spiking_neurons = [self.name+"_"+str(idx) for idx in global_idxs]
            return spiking_neurons
        else:
            raise StopIteration


    def get_input_value(self, t=None):
        warn("get_input_value is deprecated and will be removed from later versions.  Please ensure that your backend is up-to-date.")
        if t is None:
            return self.vector
        else:
            assert type(t) is int
            return self.vector[...,t:t+1][...,-1]

    def build(self, graph,
             metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build spike input brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - list of dictionary of auxillary nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if not self.time_dimension:
            self.vector = np.expand_dims(self.vector,
                                         len(self.vector.shape))

        output_lists = [[]]

        self.index_map = np.ndindex(self.vector.shape[:-1])
        for i, index in enumerate(self.index_map):
            #neuron_name = self.name+"_" + str(i)
            neuron_name = self.name+"_"+str(index)

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
               [{'complete':complete_node}],
               output_lists,
               output_codings)
        
class Spike_Input(Vector_Input):
    def __init__(self,input_spikes,*args,**kwargs):
        super().__init__(input_spikes, *args, **kwargs)
        warn("Spike_Input is deprecated.  Use Vector_Input instead.")

class PRN(Brick):
    """Psuedo-random neuron brick.  Generates spikes randomly (a uniform random [0,1] draw is compared against a threshold).  Implemented in-backend"""
    
    def __init__(self, probability=0.5, steps = None, shape=(1,), name=None, output_coding='Undefined'):
        '''
        Constructor for this brick.
        Arguments:
            + probability - Probability of a spike at any timestep
            + steps - Number of timesteps to produce spikes. None provides un-ending output.
            + shape - shape of the neurons in the brick
            + output_coding - Desired output coding for the brick
        '''
        super(Brick, self).__init__()
        self.is_built = False
        self.metadata = {}
        self.probability = probability
        self.name = name
        self.shape = shape
        self.steps = steps
        self.output_coding = output_coding
        self.supported_codings = input_coding_types
        
    def build(self,
              graph,
              metadata,
              control_nodes,
              input_lists,
              input_codings):
        if len(input_lists) == 0:
            raise ValueError("PRN brick requires at least 1 input.")
        #Driver Neuron
        driver_neuron = self.name+'_driver'
        graph.add_node(driver_neuron, threshold=0.7, decay=1.0)
        graph.add_edge(driver_neuron, driver_neuron, weight=1.0, delay=1)
        #PRNs
        output_list = []
        for neuron_index in np.ndindex(self.shape):
            output_neuron = self.name+'_' + str(neuron_index)
            graph.add_node(output_neuron, threshold=0.7, decay=1.0, p=self.probability)
            output_list.append(output_neuron)
            graph.add_edge(driver_neuron, output_neuron, weight=1.0, delay=1)
        complete_neuron = self.name+'_complete'
        complete_threshold = self.steps - 1.1 if self.steps  is not None else 1.0
        graph.add_node(complete_neuron, threshold=complete_threshold, decay = 0.0)
        if self.steps is not None:
            graph.add_edge(driver_neuron, complete_neuron, weight=1.0, delay = 1)
            graph.add_edge(complete_neuron, driver_neuron, weight=-10.0, delay=1)
        for input_control_nodes in control_nodes:
            graph.add_edge(input_control_nodes['complete'], driver_neuron, weight=1.0, delay=1)
        self.is_built = True
        return (graph,
                self.metadata,
                [{'complete':complete_neuron}],
                [output_list],
                [self.output_coding])

class Threshold(Brick):
    """Class to handle Threshold Brick. Inherits from Brick"""

    def __init__(self, threshold, decay=0.0, p=1.0, name=None, output_coding=None):
        '''
        Construtor for this brick.
        Arguments:
            + threshold - Threshold value.  For input coding 'current', float.  For 'temporal-L', int.
			+ decay - Decay value for threshold neuron ('current' input only)
			+ p - Probability of firing when exceeding threshold ('current' input only)
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
			+ output_coding - Force a return of this output coding.  Default is 'unary-L'
        '''
        super(Brick, self).__init__()
        self.is_built=False
        self.metadata = {}
        self.name = name
        self.p = p
        self.decay = decay
        self.threshold = threshold
        self.output_coding=output_coding
        self.supported_codings = ['current', 'Undefined', 'temporal-L']

    def build(self,
             graph,
             metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build Threshold brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - list of dictionary of auxillary nodes.
              Expected keys: 'complete' - A neurons that fire when the brick is done
                             'begin' - A neurons that first when the brick begins processing (for temporal coded inputs)
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_codings)!=1:
            raise ValueError("Only one input is permitted.")
        if input_codings[0] not in self.supported_codings:
            raise ValueError("Input coding not supported. Expected: {} ,Found: {}".format(self.supported_codings, 
                                                                                          input_codings[0]))
        if input_codings[0] == 'current' or input_codings[0] == 'Undefined':
            graph.add_node(self.name,
                           threshold=self.threshold,
                          decay=self.decay,
                          p=self.p)
            for edge in input_lists[0]:
                graph.add_edge(edge['source'],
                               self.name,
                               weight=edge['weight'],
                               delay=edge['delay'])
            new_complete_node = control_nodes[0]['complete']
            self.metadata['D'] = 0
            output_lists = [[self.name]]
        elif input_codings[0] == 'temporal-L':
            self.metadata['D'] = None
            new_complete_node = self.name+'_complete'
            graph.add_node(new_complete_node,
                           index = -1,
                           threshold = len(input_lists[0])-.00001,
                           decay = 0.0,
                           p=1.0)
            output_neurons = []
            #Find 'begin' neuron -- We need to fix this
            #Tentatively is fixed!
            begin_neuron = control_nodes[0]['begin']
            #for input_neuron in [input_n for input_n in input_lists[0] if graph.nodes[input_n]['index']==-2]:
            #    begin_neuron = input_neuron

            for input_neuron in [input_n for input_n in input_lists[0] if graph.nodes[input_n]['index'] is not -2]:
                threshold_neuron_name = "{}_{}".format(self.name, graph.nodes[input_neuron]['index'])
                graph.add_node(threshold_neuron_name,
                           index = graph.nodes[input_neuron]['index'],
                           threshold=1.0,
                           decay = 0.0,
                           p=self.p)
                output_neurons.append(threshold_neuron_name)
                assert(type(self.threshold) is int)
                graph.add_edge(begin_neuron,
                           threshold_neuron_name,
                           weight=2.0,
                           delay=self.threshold+1)
                graph.add_edge(input_neuron,
                               threshold_neuron_name,
                               weight = -3.0,
                               delay = 1)
                graph.add_edge(input_neuron,
                               new_complete_node,
                               weight=1.0,
                               delay=1)
                output_lists = [output_neurons]
            graph.add_edge(begin_neuron,
                            new_complete_node,
                            weight = len(input_lists[0]),
                            delay = self.threshold+1)
            graph.add_edge(new_complete_node,
                            new_complete_node,
                            weight = -10,
                            delay = 1)
        else:
            raise ValueError("Invalid coding")
        if self.output_coding is None:
            output_codings = ['unary-L']
        else:
            output_codings = [self.output_coding]
        self.is_built = True
        return (graph,
               self.metadata,
                [{'complete':new_complete_node}],
                output_lists,
                output_codings
               )

class Dot(Brick):
    """Class to handle the Dot brick. Inherits from Brick"""

    def __init__(self, weights, name=None):
        '''
        Construtor for this brick.
        Arguments:
            + weights - Vector against which the input is dotted.
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
        '''
        super(Brick, self).__init__()
        self.is_built = False
        self.metadata = {'D':1}
        self.name = name
        self.weights = weights
        self.supported_codings = ['Raster', 'Undefined']

    def build(self,
             graph,
             metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build Dot brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - list of dictionary of auxillary nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  ('Raster', 'Undefined' supported)

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list dictionary of control nodes ('complete')
            + list of output edges
            + list of coding formats of output ('current')
        """

        output_list = []
        output_codings = ['current']
        if len(input_codings)>1:
            raise ValueError("Only one input is permitted.")
        if input_codings[0] not in self.supported_codings:
            raise ValueError("Input coding not supported. Expected: {}, Found: {}".format(self.supported_codings, 
                                                                                          input_codings[0]))
        if len(input_lists[0]) != len(self.weights):
            raise ValueError("Input length does not match weights. Expected: {}, Found: {}".format(len(self.weights),
                                                                                                   len(input_lists[0])))
        for i, weight in enumerate(self.weights):
            output_list.append({'source':input_lists[0][i],
                                'weight':weight,
                                'delay':1
                               })
        if type(metadata) is list:
            metadata = metadata[0]
        metadata['D'] = metadata['D'] + 1
        graph.add_node(self.name+"_complete",
                      threshold=0.5,
                      potential=0.0,
                      decay = 0.0,
                      index = -1,
                      p=1.0)
        graph.add_edge(control_nodes[0]['complete'],
                       self.name+"_complete",
                       weight=1.0,
                       delay=1)
        self.is_built = True
        return (graph,
                metadata,
                [{'complete':self.name+"_complete"}],
                [output_list],
                output_codings)

class Copy(Brick):
    """Class to handle Copy Brick. Inherits from Brick
    """

    def __init__(self, name=None):
        '''
        Construtor for this brick.
        Arguments:
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
        '''
        super(Brick, self).__init__()
        self.is_built=False
        self.metadata = {'D':1}
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
              metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build Copy brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - list of dictionaries of auxillary nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of dictionaries of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """
        num_copies = 2
        if type(metadata) is list:
            self.metadata = metadata[0]
        else:
            self.metadata = metadata
        self.metadata['D'] = 1
        #self.metadata['output_coding'] = self.metadata['input_coding']
        #input_shape = input_lists #self.metadata['input_shape']
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
                copy_name = "{}_copy{}".format(neuron, copy_num)
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
        graph.add_edge(control_nodes[0]['complete'],
                       self.name+"_complete",
                       weight=1.0,
                       delay=1)
        self.is_built = True
        return (graph, self.metadata, [{'complete':self.name+"_complete"}]*num_copies , output_lists, output_codings)
    
class Concatenate(Brick):
    ''' Brick that concatenates multiple inputs into a single vector.  All codings are supported except 'current'; first coding is used if not specified.
    Arguments:
		+ name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
    '''
    def __init__(self, name=None, coding=None):   
        super(Brick, self).__init__()
        self.is_built = False
        self.metadata = {'D':0}
        self.name = name
        self.supported_codings = input_coding_types
        if coding is not None:
            self.coding = coding
        else:
            self.coding = None
    def build(self,
             graph,
             metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build concatenate brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All codings are allowed except 'current'.

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output (1 output)
            + list of coding formats of output (Coding matches input coding)
        """
        #Keep the same coding as input 0 for the output
        #This is an arbitrary decision at this point.
        #Generally, your brick will impart some coding, but that isn't the case here.
        if self.coding is None:
            output_codings = [input_codings[0]]
        else:
            output_codings = [self.coding]

        
        new_complete_node_name = self.name + '_complete'
        graph.add_node(new_complete_node_name,
                      index = -1,
                      threshold = 1.0,
                      decay =0.0,
                      p=1.0,
                      potential=0.0)
        for idx in range(len(input_lists)):
            graph.add_edge(control_nodes[idx]['complete'], new_complete_node_name,weight=(1/len(input_lists))+0.000001,delay=1)


        output_lists = [[]]
        for input_brick in input_lists:
            for input_neuron in input_brick:
                relay_neuron_name = "{}_relay_{}".format(self.name, input_neuron)
                graph.add_node(relay_neuron_name,
                               index = (len(output_lists[0]),),
                               threshold = 0.0,
                               decay = 0.0,
                               p=1.0,
                               potential=0.0)
                graph.add_edge(input_neuron, relay_neuron_name, weight=1.0, delay=1)
                output_lists[0].append(relay_neuron_name)
        
        self.is_built=True


        return (graph,
               self.metadata,
                [{'complete':new_complete_node_name}],
                output_lists,
                output_codings
               )

class AND_OR(Brick):
    ''' Brick for performing a logical AND/OR.  Operation is performed entry-wise, matching based on index.  All codings are supported.
    Arguments:
        + mode - Either 'And' or 'Or'; determines the operation
		+ name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
    '''
    def __init__(self, mode='AND',name=None):   #A change here
        super(Brick, self).__init__()
        #The brick hasn't been built yet.
        self.is_built = False
        #Leave for compatibility, D represents the depth of the circuit.  Needs to be updated.
        self.metadata = {'D':1}
        #We just store the name passed at construction.
        self.name = name
        #For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = input_coding_types
        self.mode = mode  #A change here
    def build(self,
             graph,
             metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build AND_OR brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All codings are allowed.

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output (1 output)
            + list of coding formats of output (Coding matches input coding)
        """
		#Expect two inputs
        if len(input_codings)!=2:
            raise ValueError('Only two inputs supported.')
        #Only two supported modes, AND and OR
        if self.mode != 'AND' and self.mode != 'OR':
            raise ValueError('Unsupported mode.')
        #Keep the same coding as input 0 for the output
        #This is an arbitrary decision at this point.
        #Generally, your brick will impart some coding, but that isn't the case here.
        output_codings = [input_codings[0]]

        #All bricks should provide a neuron that spikes when the brick has completed processing.
        #We just put in a basic relay neuron that will spike when it recieves any spike from its
        #single input, which is the complete_node from the first input.
        #All nodes we add to the graph should have basic neuron parameters (threshold, decay)
        #Reasonable defaults will be filled-in, but these defaults may depend on the execution platform.
        #Additionally, nodes should have a field called 'index' which is a local index used to reference the
        #position of the node.  This can be used by downstream bricks.  A simple example might be
        #a 3-bit binary representation will add 3 nodes to the graph with indices 0,1,2
        #We do have to do some work to establish best practices here.
        new_complete_node_name = self.name + '_complete'
        graph.add_node(new_complete_node_name,
                      index = -1,
                      threshold = 0.0,
                      decay =0.0,
                      p=1.0,
                      potential=0.0)
        graph.add_edge(control_nodes[0]['complete'], new_complete_node_name,weight=1.0,delay=1)


        output_lists = [[]]
        threshold_value = 1.0 if self.mode == 'AND' else 0.5
        #We also, obviously, need to build the computational portion of our graph
        for operand0 in input_lists[0]:
            for idx_num, operand1 in enumerate(input_lists[1]):
                #If indices match, we'll do an AND on them
                if graph.nodes[operand0]['index'] == graph.nodes[operand1]['index']:
                    #Remember all of our output neurons need to be marked
                    and_node_name = "{}_{}_{}".format(self.name, operand0, operand1)
                    output_lists[0].append(and_node_name)
                    graph.add_node(and_node_name,
                       index=0,
                       threshold=threshold_value,
                       decay=1.0,
                       p=1.0,
                       potential=0.0
                      )
                    graph.add_edge(operand0,
                                  and_node_name,
                                  weight=0.75,
                                  delay=1.0)
                    graph.add_edge(operand1,
                                  and_node_name,
                                  weight=0.75,
                                  delay=1.0)
        self.is_built=True


        return (graph,
               self.metadata,
                [{'complete':new_complete_node_name}],
                output_lists,
                output_codings
               )

class Shortest_Path(Brick):
    '''This brick provides a single-source shortest path determination. Expects a single input where the index corresponds to the node number on the graph.

    '''
    def __init__(self, target_graph, target_node=None, name=None, return_path=False, output_coding = 'temporal-L'):
        '''
        Construtor for this brick.
        Arguments:
            + target_graph - NetworkX.Digraph object representing the graph to be searched
			+ target_node - Node in the graph that is the target of the paths
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
			+ output_coding - Output coding type, default is 'temporal-L'
        '''
        super(Brick, self).__init__()
        #The brick hasn't been built yet.
        self.is_built = False
        #We just store the name passed at construction.
        self.name = name
        #For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = input_coding_types
        #Right now, we'll convert node labels to integers in the order of
        #graph.nodes() However, in the fugure, this should be improved to be
        #more flexible.
        for i,node in enumerate(target_graph.nodes()):
            if node is target_node:
                self.target_node = i
        self.target_graph = target_graph
        self.output_codings = [output_coding]
        self.metadata = {'D':None}
        self.return_path = return_path

    def build(self,
             graph,
             metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build Parity brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All coding types supported

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(input_coding,
                                                                                           self.supported_codings))

        #All bricks should provide a neuron that spikes when the brick has completed processing.
        #We just put in a basic relay neuron that will spike when it recieves any spike from its
        #single input, which is the complete_node from the first input.
        #All nodes we add to the graph should have basic neuron parameters (threshold, decay)
        #Reasonable defaults will be filled-in, but these defaults may depend on the execution platform.
        #Additionally, nodes should have a field called 'index' which is a local index used to reference the
        #position of the node.  This can be used by downstream bricks.  A simple example might be
        #a 3-bit binary representation will add 3 nodes to the graph with indices 0,1,2
        #We do have to do some work to establish best practices here.
        #new_complete_node_name = self.name + '_complete'
        #graph.add_node(new_complete_node_name,
        #              index = -1,
        #              threshold = 0.0,
        #              decay =0.0,
        #              p=1.0,
        #              potential=0.0)
        #complete_node = [new_complete_node_name]
        new_begin_node_name = self.name+'_begin'
        graph.add_node(new_begin_node_name,
                      threshold = 0.5,
                      decay = 0.0,
                      potential=0.0)
        graph.add_edge(control_nodes[0]['complete'],
                      self.name+'_begin',
                      weight = 1.0,
                      delay = 1)

        for node in self.target_graph.nodes:
            node_name = self.name + str(node)
            graph.add_node(node_name,
                           index = (node,),
                           threshold=1.0,
                           decay=0.0,
                           potential=0.0)
            graph.add_edge(node_name, node_name, weight=-1000, delay=1)
            if node==self.target_node:
                complete_node_list = [node_name]

        edge_reference_names = []
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)
            for neighbor in self.target_graph.neighbors(node):
                delay = self.target_graph.edges[node,neighbor]['weight']
                neighbor_name = self.name + str(neighbor)
                if self.return_path:
                    if node == self.target_node:
                        reference_name = "{}-{}-{}".format(self.name, node, neighbor)
                        edge_reference_names.append(reference_name)

                        graph.add_node(reference_name, threshold=1.0, decay=0.0,potential=0.0)
                        graph.add_edge(node_name, reference_name, weight=-1000, delay=delay - 1.0)
                        graph.add_edge(reference_name, neighbor_name, weight=-1000, delay=1.0)
                        graph.add_edge(neighbor_name, reference_name, weight=-1000, delay=1.0) 
                    else:
                        reference_name = "{}-{}-{}".format(self.name, node, neighbor)
                        edge_reference_names.append(reference_name)

                        graph.add_node(reference_name, threshold=1.0, decay=0.0,potential=0.0)
                        graph.add_edge(node_name, reference_name, weight=1.1, delay=delay - 1.0)
                        graph.add_edge(reference_name, neighbor_name, weight=1.1, delay=1.0)
                        
                        graph.add_edge(neighbor_name, reference_name, weight=-1000, delay=1.0) 
                else:
                    if node == self.target_node:
                        graph.add_edge(node_name, neighbor_name, weight=-1000, delay=delay)
                    else:
                        delay = self.target_graph.edges[node,neighbor]['weight']
                        graph.add_edge(node_name, neighbor_name, weight=1.5, delay=delay)

        for input_neuron in input_lists[0]:
            index = graph.nodes[input_neuron]['index']
            if type(index) is tuple:
                index = index[0]
            if type(index) is not int:
                raise TypeError("Neuron index should be Tuple or Int.")
            graph.add_edge(input_neuron,
                          self.name+str(index),
                         weight = 2.0,
                         delay = 1)

        self.is_built=True

        #Remember, bricks can have more than one output, so we need a list of list of output neurons
        output_lists = [[self.name+str(self.target_node)], edge_reference_names]

        return (graph,
               self.metadata,
                [{'complete':complete_node_list[0], 'begin':new_begin_node_name}],
                output_lists,
                self.output_codings
               )

class Breadth_First_Search(Brick):
    '''This brick performs a BFS traversal. Expects a single input where the index corresponds to the node number on the graph.

    '''
    def __init__(self, target_graph, target_node=None, name=None, output_coding = 'temporal-L'):
        '''
        Construtor for this brick.
        Arguments:
            + target_graph - NetworkX.Digraph object representing the graph to be searched
			+ target_node - Node in the graph that is the target of the paths
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
			+ output_coding - Output coding type, default is 'temporal-L'
        '''
        super(Brick, self).__init__()
        #The brick hasn't been built yet.
        self.is_built = False
        #We just store the name passed at construction.
        self.name = name
        #For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = input_coding_types
        #Right now, we'll convert node labels to integers in the order of
        #graph.nodes() However, in the fugure, this should be improved to be
        #more flexible.
        self.target_node = target_node
        self.target_graph = target_graph
        self.output_codings = [output_coding]
        self.metadata = {'D':None}

        # mappings of the original graph to the embedded graph and vice-versa
        # used primarily to interpret what the spikes mean
        self.neuron_vertex_map = {}
        self.vertex_neuron_map = {}
        self.edge_synapse_map = {}
        self.synapse_edge_map = {}

    def get_neuron_vertex_map(self):
        return self.neuron_vertex_map

    def get_vertex_neuron_map(self):
        return self.vertex_neuron_map

    def get_synapse_edge_map(self):
        return self.synapse_edge_map

    def get_edge_synapse_map(self):
        return self.edge_synapse_map

    def build(self,
             graph,
             metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build Parity brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All coding types supported

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(input_coding,
                                                                                           self.supported_codings))

        #All bricks should provide a neuron that spikes when the brick has completed processing.
        #We just put in a basic relay neuron that will spike when it recieves any spike from its
        #single input, which is the complete_node from the first input.
        #All nodes we add to the graph should have basic neuron parameters (threshold, decay)
        #Reasonable defaults will be filled-in, but these defaults may depend on the execution platform.
        #Additionally, nodes should have a field called 'index' which is a local index used to reference the
        #position of the node.  This can be used by downstream bricks.  A simple example might be
        #a 3-bit binary representation will add 3 nodes to the graph with indices 0,1,2
        #We do have to do some work to establish best practices here.
        new_begin_node_name = self.name+'_begin'
        graph.add_node(new_begin_node_name,
                      threshold = 0.5,
                      decay = 0.0,
                      potential=0.0)
        graph.add_edge(control_nodes[0]['complete'],
                      self.name+'_begin',
                      weight = 1.0,
                      delay = 1)

        complete_name = self.name + '_complete'
        graph.add_node(complete_name,
                index = len(self.target_graph.nodes),
                threshold = 1.0 if self.target_node else 1.0 * len(self.target_graph.nodes),
                decay = 0.0,
                potential = 0.0)
        complete_node_list = [complete_name]

        target_node_list = []
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)

            self.neuron_vertex_map[node] = node
            self.vertex_neuron_map[node] = node

            graph.add_node(node_name,
                           index = (node,),
                           threshold=1.0,
                           decay=0.0,
                           potential=0.0)
            graph.add_edge(node_name, node_name, weight=-1000, delay=1)
            if self.target_node:
                if node==self.target_node:
                    target_node_list.append(node_name)
                    graph.add_edge(node_name, complete_name, weight=1.0, delay=1.0)
            else:
                graph.add_edge(node_name, complete_name, weight=1.0, delay=1.0)

        edge_reference_names = []
        reference_index = len(self.target_graph.nodes) + 1
        for node in self.target_graph.nodes:
            node_name = self.name + str(node)
            neighbors = list(self.target_graph.neighbors(node))
            for neighbor in neighbors:
                neighbor_name = self.name + str(neighbor)
                reference_name = "{}-{}-{}".format(self.name, node, neighbor)

                self.synapse_edge_map[reference_index] = (node, neighbor)
                self.edge_synapse_map[(node,neighbor)] = reference_index

                edge_reference_names.append(reference_name)

                graph.add_node(reference_name, index=reference_index, threshold=1.0, decay=0.0,potential=0.0, from_vertex=node, to_vertex=neighbor )
                graph.add_edge(neighbor_name, reference_name, weight=-1000, delay=1.0) 

                if self.target_node and node == self.target_node:
                    weight = -1000
                else:
                    weight = 1.1
                graph.add_edge(node_name, reference_name, weight=weight, delay=1.0)
                graph.add_edge(reference_name, neighbor_name, weight=weight, delay=1.0)
                reference_index += 1

        for input_neuron in input_lists[0]:
            index = graph.nodes[input_neuron]['index']
            if type(index) is tuple:
                index = index[0]
            if type(index) is not int:
                raise TypeError("Neuron index should be Tuple or Int.")
            graph.add_edge(input_neuron,
                          self.name+str(index),
                         weight = 2.0,
                         delay = 1)

        self.is_built=True

        #Remember, bricks can have more than one output, so we need a list of list of output neurons
        output_lists = [complete_node_list, edge_reference_names, target_node_list]

        return (graph,
               self.metadata,
                [{'complete':complete_node_list[0], 'begin':new_begin_node_name}],
                output_lists,
                self.output_codings
               )

'''
class AND(AND_OR):
    Performs logical AND.

    Input neurons are matched based on index.

    All codings are supported.
    def __init__(self, name=None):
        super(AND_OR, self).__init__(mode='AND', name=name)
'''

class ParityCheck(Brick):
    '''Brick to compute the parity of a 4 bit input.
    The output spikes after 2 time steps if the input has odd parity
    '''
    #author: Srideep Musuvathy
    #email: smusuva@sandia.gov
    #last updated: April 8, 2019'''

    def __init__(self, name=None):
        '''
        Construtor for this brick.
        Arguments:
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
        '''
        super().__init__()
        self.is_built = False
        self.metadata = {'D': 1}
        self.name = name
        self.supported_codings = ['binary-B', 'binary-L', 'Raster']

    def build(self,
              graph,
              metadata,
              control_nodes,
              input_lists,
              input_codings):
        """
        Build Parity brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_codings) != 1:
            raise ValueError('Parity check takes in 1 input')

        output_codings = [input_codings[0]]

        new_complete_node_name = self.name + '_complete'

        graph.add_node(new_complete_node_name,
                      index = -1,
                      threshold = 0.0,
                      decay =0.0,
                      p=1.0,
                      potential=0.0)
        graph.add_edge(control_nodes[0]['complete'], new_complete_node_name, weight=1.0, delay=2)
        complete_node = new_complete_node_name

        #add 4 hidden nodes with thresholds <=1, >=1, <=3, >=3.
        #since the thresholds only compute >=, the <=1, <=3 computations are performed by negating
        #the threshold weights and the inputs (via the weights on incomming edges)

        #first hidden node and connect edges from input layer
        graph.add_node('h_00',
                      index=0,
                      threshold=-1.1,
                      decay=1.0,
                      p=1.0,
                      potential=0.0)
        graph.add_edge(input_lists[0][0],
                      'h_00',
                      weight=-1.0,
                      delay=1)
        graph.add_edge(input_lists[0][1],
                      'h_00',
                      weight=-1.0,
                      delay=1)
        graph.add_edge(input_lists[0][2],
                      'h_00',
                      weight=-1.0,
                      delay=1)
        graph.add_edge(input_lists[0][3],
                      'h_00',
                      weight=-1.0,
                      delay=1)

        #second hidden node and edges from input layer
        graph.add_node('h_01',
                      index=1,
                      threshold=0.9,
                      decay=1.0,
                      p=1.0,
                      potential=0.0)
        graph.add_edge(input_lists[0][0],
                      'h_01',
                      weight=1.0,
                      delay=1)
        graph.add_edge(input_lists[0][1],
                      'h_01',
                      weight=1.0,
                      delay=1)
        graph.add_edge(input_lists[0][2],
                      'h_01',
                      weight=1.0,
                      delay=1)
        graph.add_edge(input_lists[0][3],
                      'h_01',
                      weight=1.0,
                      delay=1)

        #third hidden node and edges from input layer
        graph.add_node('h_02',
                      index=2,
                      threshold=-3.1,
                      decay=1.0,
                      p=1.0,
                      potential=0.0)
        graph.add_edge(input_lists[0][0],
                      'h_02',
                      weight=-1.0,
                      delay=1)
        graph.add_edge(input_lists[0][1],
                      'h_02',
                      weight=-1.0,
                      delay=1)
        graph.add_edge(input_lists[0][2],
                      'h_02',
                      weight=-1.0,
                      delay=1)
        graph.add_edge(input_lists[0][3],
                      'h_02',
                      weight=-1.0,
                      delay=1)

        #fourth hidden node and edges from input layer
        graph.add_node('h_03',
                      index=3,
                      threshold=2.9,
                      decay=1.0,
                      p=1.0,
                      potential=0.0)
        graph.add_edge(input_lists[0][0],
                      'h_03',
                      weight=1.0,
                      delay=1)
        graph.add_edge(input_lists[0][1],
                      'h_03',
                      weight=1.0,
                      delay=1)
        graph.add_edge(input_lists[0][2],
                      'h_03',
                      weight=1.0,
                      delay=1)
        graph.add_edge(input_lists[0][3],
                      'h_03',
                      weight=1.0,
                      delay=1)

        #output_node and edges from hidden nodes
        graph.add_node('parity',
                      index=4,
                      threshold=2.9,
                      decay=1.0,
                      p=1.0,
                      potential=0.0)
        graph.add_edge('h_00',
                      'parity',
                      weight=1.0,
                      delay=1)
        graph.add_edge('h_01',
                      'parity',
                      weight=1.0,
                      delay=1)
        graph.add_edge('h_02',
                      'parity',
                      weight=1.0,
                      delay=1)
        graph.add_edge('h_03',
                      'parity',
                      weight=1.0,
                      delay=1)

        self.is_built=True

        output_lists = [['parity']]

        return (graph, self.metadata, [{'complete':complete_node}], output_lists,
                output_codings)

class LongestIncreasingSubsequence(Brick):
    '''This brick performs a BFS traversal. Expects a single input where the index corresponds to the node number on the graph.

    '''
    def __init__(self, sequence, name=None, output_coding = 'temporal-L'):
        '''
        Construtor for this brick.
        Arguments:
            + target_graph - NetworkX.Digraph object representing the graph to be searched
			+ target_node - Node in the graph that is the target of the paths
		    + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
			+ output_coding - Output coding type, default is 'temporal-L'
        '''
        super(Brick, self).__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types

        self.output_codings = [output_coding]
        self.metadata = {'D':None}

        self.sequence = sequence 

    def build(self,
             graph,
             metadata,
             control_nodes,
             input_lists,
             input_codings):
        """
        Build LIS brick.

        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + metadata - dictionary to define the shapes and parameters of the brick
            + control_nodes - dictionary of lists of auxillary networkx nodes.  Excpected keys: 'complete' - A list of neurons that fire when the brick is done
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats.  All coding types supported

        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + dictionary of control nodes ('complete')
            + list of output
            + list of coding formats of output
        """

        if len(input_lists) != 1:
            raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(input_coding,
                                                                                           self.supported_codings))

        new_begin_node_name = self.name+'_begin'
        graph.add_node(new_begin_node_name,
                      threshold = 0.1,
                      decay = 0.0,
                      potential=0.0)
        graph.add_edge(control_nodes[0]['complete'],
                      self.name+'_begin',
                      weight = 1.0,
                      delay = 0.01)

        complete_name = self.name + '_complete'
        graph.add_node(complete_name,
                index = len(self.sequence),
                threshold = 0.1,
                decay = 0.0,
                potential = 0.0)
        complete_node_list = [complete_name]

        levels = [[] for i in range(len(self.sequence))]
        for i, x_i in enumerate(self.sequence):
            column_a = []
            column_b = []
            x_name = f"x_{i}"
            L0_A_name = f"L_1-x_{i}-A"

            # create x_i neuron
            graph.add_node(x_name,
                            threshold = 0.0,
                            decay = 0.0,
                            potential = 0.0)

            # create column
            graph.add_node(L0_A_name,
                            threshold = 0.9,
                            decay = 0.0,
                            potential = 0.0)

            graph.add_edge(x_name, L0_A_name, weight = 1.0, delay = 0.0)
            graph.add_edge(L0_A_name, L0_A_name, weight = -2.0, delay = 0.0)

            levels[0].append(L0_A_name)

            for j in range(i):
                L_B_name = f"L_{j + 1}-x_{i}-B"
                L_A_name = f"L_{j + 2}-x_{i}-A"
                graph.add_node(L_B_name,
                               threshold = 0.9,
                               decay = 0.0,
                               potential = 0.0)
                graph.add_node(L_A_name,
                               threshold = 1.9,
                               decay = 0.0,
                               potential = 0.0)

                # Alarms
                graph.add_edge(x_name, L_B_name, weight = -1.0, delay = j + 1.01)
                graph.add_edge(x_name, L_A_name, weight = 1.0, delay = 0.0)

                graph.add_edge(L_B_name, L_A_name, weight = 1.0, delay = 0.0)
                graph.add_edge(L_A_name, L_A_name, weight = -2.0, delay = 0.0)
                
                levels[j].append(L_B_name)
                levels[j+1].append(L_A_name)

        for level in levels:
            if len(level) > 1:
                graph.add_edge(level[0], level[1], weight = 1.0, delay = 0.0)
                graph.add_edge(level[0], level[2], weight = 1.0, delay = 0.0)
                for i in range(1, len(level) - 2,2):
                    graph.add_edge(level[i], level[i + 2], weight = 1.0, delay = 0.0)
                    graph.add_edge(level[i], level[i + 3], weight = 1.0, delay = 0.0)

        for input_neuron in input_lists[0]:
            index = graph.nodes[input_neuron]['index']
            if type(index) is tuple:
                index = index[0]
            if type(index) is not int:
                raise TypeError("Neuron index should be Tuple or Int.")
            for i, value in enumerate(self.sequence):
                graph.add_edge(input_neuron,
                              f"x_{i}",
                              weight = 1.0,
                              delay = value)

        self.is_built=True

        output_Ls = []
        for l in levels[-1]:
            if 'A' in l:
                output_Ls.append(l)

        #Remember, bricks can have more than one output, so we need a list of list of output neurons
        #output_lists = [complete_node_list, output_Ls]
        output_lists = [complete_node_list]

        return (graph,
               self.metadata,
                [{'complete':complete_node_list[0], 'begin':new_begin_node_name}],
                output_lists,
                self.output_codings
               )
