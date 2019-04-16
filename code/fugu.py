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
    """Class to handle a scaffold of bricks"""
    
    supported_backends = ['ds']
    
    def __init__(self):
        self.circuit = nx.DiGraph()
        self.pos = {}
        self.count = {}
        self.graph = None
        self.is_built = False
        
    def add_brick(self, brick_function, input_nodes=[-1], dimensionality = None, name=None, output=False):
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
            name = brick_type+"_" + str(self.count[brick_type])
            self.count[brick_type] = self.count[brick_type]+1
        elif name in self.circuit.nodes:
            raise ValueError("Node name already used.")
        if name in [self.circuit.nodes[node]['name'] for node in self.circuit.nodes]:
            raise ValueError("Node name already used.")

        if brick_function.name is None:
            brick_function.name = name

        node_number = self.circuit.number_of_nodes()
        self.circuit.add_node(node_number, name=name,
                             brick=brick_function)#,dimensionality=dimensionality)

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
                print(str(node) + ":" + str(self.circuit.nodes[node]['brick'].is_built))
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
            if verbose > 0:
                print("Completed: " + str(node))
        while not self.all_nodes_built(verbose=verbose):
            #Over unbuilt, ready edges
            for node in [node for node in self.circuit.nodes
                         if (not self.circuit.nodes[node]['brick'].is_built)
                         and self.all_in_neighbors_built(node)]:
                inputs = {}
                if verbose > 0:
                    print('Laying Brick: ' + str(node))
                for input_number in range(0,len(self.circuit.nodes[node]['input_nodes'])):
                    if verbose > 0:
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
                if verbose > 0:
                    print("Complete.")
        self.is_built=True
        self.graph = built_graph
        return built_graph

    def _create_ds_injection(self):
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

    def evaluate(self, max_runtime=10, backend='ds', record_all=False):
        """
        Run the computational graph through the backend.
        
        Arguments:
            + max_runtime - int value to specify number of time steps (Default: 10)
            + backend - string value of the backend simulator or device name (Default: 'ds')
            + record_all - bool value to indicate if all neurons spikesa re to be recorded (Default: False)
            
        Returns:
            + dictionary of time step and spiking neurons. (if record_all is True, all spiking neurons are shown
            else only the output neurons)
            
        Exceptions:
            + ValueError if backend is not in list of supported backends
        """
        
        if backend not in Scaffold.supported_backends:
            raise ValueError("Backend " + str(backend) + " not supported.")
        if backend == 'ds':
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
        return spike_result
    
    def summary(self):
        """Display a summary of the scaffold."""
        
        print("Scaffold is built: " + str(self.is_built))
        print("-------------------------------------------------------")
        print("List of Bricks:")
        print("\r\n")
        for i,node in enumerate(self.circuit.nodes):
            print("Brick No.: " + str(i))
            print("Brick Name: " + str(self.circuit.nodes[node]['name']))
            print(self.circuit.nodes[node])
            print("Brick is built: " + str(self.circuit.nodes[node]['brick'].is_built))
            print("\r\n")
        print("-------------------------------------------------------")
        print("\r\n")
        print("-------------------------------------------------------")
        print("List of Brick Edges:")
        print("\r\n")
        for i, edge in enumerate(self.circuit.edges):
            print("Edge: " + str(edge))
            print(self.circuit.edges[edge])
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
                print("Synapse Between | Synapse Properties")
                print(str(synapse) + " | " + str(self.graph.edges[synapse]))


class Brick(ABC):
    """Abstract Base Class definition of a Brick class"""
    
    def __init__(self):
        self.name = "Empty Brick"
        self.is_built = False
        self.supported_codings = []

    @abstractmethod
    def build(self, graph,
                   dimensionality,
                   complete_node,
                   input_lists,
                   input_codings):
        """
        Build the computational graph of the brick. Method must be defined in any class inheriting from Brick.
        
        Arguments:
            + graph - networkx graph
            + dimensionality - A dictionary of shapes and parameters
            + complete_node - list of nodes that indicate the end of computation
            + input_lists - list of lists of nodes for input neurons
            + input_codings - list of input coding types (as strings)
        """
        pass
    
class InputBrick(Brick):
    """Abstract Base class for handling inputs inherited from Brick"""

    @abstractmethod
    def get_input_value(self, t=None):
        """
        Abstract method to get input values. InputBriacks must implement this method
        
        Arguments:
            + t - type of input (Default: None)
        """
        pass

class Spike_Input(InputBrick):
    """Class to handle Spike Input. Inherits from InputBrick"""
    
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
            assert type(t) is int
            return self.vector[...,t:t+1][...,-1]

    def build(self, graph,
             dimensionality,
             complete_node,
             input_lists,
             input_codings):
        """
        Build spike input brick.
        
        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + dimensionality - dictionary to define the shapes and parameters of the brick
            + complete_node - list of networkx nodes to indicate completion of the computation
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats
            
        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of complete nodes
            + list of output
            + list of coding formats of output
        """
        
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
    """Class to handle Threshold Brick. Inherits from Brick"""
    
    def __init__(self, threshold, decay=0.0, p=1.0, name=None, output_coding=None):
        super(Brick, self).__init__()
        self.is_built=False
        self.dimensionality = {}
        self.name = name
        self.p = p
        self.decay = decay
        self.threshold = threshold
        self.output_coding=output_coding
        self.supported_codings = ['current', 'Undefined', 'temporal-L']
        
    def build(self,
             graph,
             dimensionality,
             complete_node,
             input_lists,
             input_codings):
        """
        Build Threshold brick.
        
        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + dimensionality - dictionary to define the shapes and parameters of the brick
            + complete_node - list of networkx nodes to indicate completion of the computation
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats
            
        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of complete nodes
            + list of output
            + list of coding formats of output
        """
        
        if len(input_codings)!=1:
            raise ValueError("Only one input is permitted.")
        if input_codings[0] not in self.supported_codings:
            raise ValueError("Input coding not supported. Expected: "
                             + str(self.supported_codings)
                             + " Found: "
                             + str(input_codings[0]))
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
            new_complete_node = complete_node
            self.dimensionality['D'] = 0
            output_lists = [[self.name]]
        elif input_codings[0] == 'temporal-L':
            self.dimensionality['D'] = None
            graph.add_node(self.name+'_complete',
                           index = -1,
                           threshold = len(input_lists[0])-1.0001,
                           decay = 1.0,
                           p=1.0)
            new_complete_node = [self.name+'_complete']
            output_neurons = []
            #Find 'begin' neuron -- We need to fix this
            for input_neuron in [input_n for input_n in input_lists[0] if graph.nodes[input_n]['index']==-2]:
                begin_neuron = input_neuron
                
            for input_neuron in [input_n for input_n in input_lists[0] if graph.nodes[input_n]['index'] is not -2]:
                threshold_neuron_name = self.name+'_'+str(graph.nodes[input_neuron]['index'])
                graph.add_node(threshold_neuron_name,
                           index = graph.nodes[input_neuron]['index'],
                           threshold=1.0,
                           decay = 0.0,
                           p=self.p)
                output_neurons.append(threshold_neuron_name)
                graph.add_edge(begin_neuron, 
                           threshold_neuron_name,
                           weight=2.0,
                           delay=self.threshold)
                graph.add_edge(input_neuron,
                               threshold_neuron_name,
                               weight = -3.0,
                               delay = 1)
                graph.add_edge(input_neuron,
                               self.name+'_complete',
                               weight=1.0,
                               delay=1)
                output_lists = [output_neurons]
        else:
            raise ValueError("Invalid coding")
        if self.output_coding is None:
            output_codings = ['Unary']
        else:
            output_codings = [self.output_coding]
        self.is_built = True
        return (graph,
               self.dimensionality,
                new_complete_node,
                output_lists,
                output_codings
               )

class Dot(Brick):
    """Class to handle the Dot brick. Inherits from Brick"""
    
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
        """
        Build Dot brick.
        
        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + dimensionality - dictionary to define the shapes and parameters of the brick
            + complete_node - list of networkx nodes to indicate completion of the computation
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats
            
        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of complete nodes
            + list of output
            + list of coding formats of output
        """
        
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
    """Class to handle Copy Brick. Inherits from Brick"""
    
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
        """
        Build Copy brick.
        
        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + dimensionality - dictionary to define the shapes and parameters of the brick
            + complete_node - list of networkx nodes to indicate completion of the computation
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats
            
        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of complete nodes
            + list of output
            + list of coding formats of output
        """
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
    
class ParityCheck(Brick):
    '''Brick to compute the parity of a 4 bit input. 
    The output spikes after 2 time steps if the input has odd parity
    
    author: Srideep Musuvathy
    email: smusuva@sandia.gov
    last updated: April 8, 2019'''
    
    def __init__(self, name=None):
        super().__init__()
        self.is_built = False
        self.dimensionality = {'D': 1}
        self.name = name
        self.supported_codings = ['binary-B', 'binary-L', 'Raster']
    
    def build(self,
              graph,
              dimensionality,
              complete_node,
              input_lists,
              input_codings):
        """
        Build Parity brick.
        
        Arguments:
            + graph - networkx graph to define connections of the computational graph
            + dimensionality - dictionary to define the shapes and parameters of the brick
            + complete_node - list of networkx nodes to indicate completion of the computation
            + input_lists - list of nodes that will contain input
            + input_coding - list of input coding formats
            
        Returns:
            + graph of a computational elements and connections
            + dictionary of output parameters (shape, coding, layers, depth, etc)
            + list of complete nodes
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
        graph.add_edge(complete_node[0], new_complete_node_name, weight=1.0, delay=2)
        complete_node = [new_complete_node_name]
        
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
        
        return (graph, self.dimensionality, complete_node, output_lists, 
                output_codings)
