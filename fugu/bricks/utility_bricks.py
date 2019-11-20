#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:55 2019

@author: smusuva
"""
from .bricks import Brick
from .bricks import input_coding_types #added by Darb on 11/20/2019

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

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
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
        return (graph, metadata, [{'complete':self.name+"_complete"}], [output_list], output_codings)

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
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
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
    ''' 
    Brick that concatenates multiple inputs into a single vector.  All codings are supported except 'current'; first coding is used if not specified.
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
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
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
            graph.add_edge(control_nodes[idx]['complete'], new_complete_node_name,weight=(1/len(input_lists))+0.000001, delay=1)


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


        return (graph, self.metadata, [{'complete':new_complete_node_name}], output_lists, output_codings)

class AND_OR(Brick):
    ''' 
    Brick for performing a logical AND/OR.  Operation is performed entry-wise, matching based on index.  All codings are supported.
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
    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
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


        return (graph, self.metadata, [{'complete':new_complete_node_name}], output_lists, output_codings)

'''
class AND(AND_OR):
    Performs logical AND.

    Input neurons are matched based on index.

    All codings are supported.
    def __init__(self, name=None):
        super(AND_OR, self).__init__(mode='AND', name=name)
'''

class ParityCheck(Brick):
    '''
    Brick to compute the parity of a 4 bit input.
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

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
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

        return (graph, self.metadata, [{'complete':complete_node}], output_lists, output_codings)

class TemporalAdder(Brick):
    '''
    Brick that "adds" spike times together:

    More specifically, consider you have three neurons u, v, and w that first spike at times t_u, t_v, and t_w.
    Assuming v spikes before w (so t_v < t_w), we want t_u = t_w + t_v, i.e. u fires t_v timesteps after w fires.
    '''
    def __init__(self, number_of_elements, design = 'default', name = None, output_coding = 'temporal-L'):
        '''
        Construtor for this brick.
        Arguments:
            + number_of_elements - number of signals you want to add together
            + name - Name of the brick.  If not specified, a default will be used.  Name should be unique.
            + output_coding - Output coding type, default is 'temporal-L'
        '''
        super(Brick, self).__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types

        self.output_codings = [output_coding]
        self.metadata = {'D':None}

        self.num_elements = number_of_elements
        self.design = design

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Adder brick.

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

        if len(input_lists) < 2:
            if len(input_lists) > 0:
                if len(input_lists[0]) < 2:
                    pass
                    #raise ValueError('Incorrect Number of Inputs.')
            else:
                raise ValueError('Incorrect Number of Inputs.')
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(input_coding,
                                                                                           self.supported_codings))

        begin_node_name = self.name+'_begin'
        graph.add_node(begin_node_name,
                       threshold = 0.1,
                       decay = 0.0,
                       potential=0.0)

        complete_name = self.name + '_complete'
        graph.add_node(complete_name,
                threshold = 0.1,
                decay = 0.0,
                potential = 0.0)
        complete_node_list = [complete_name]

        output_name = "Sum"
        graph.add_node(output_name,
                       threshold = 0.00,
                       decay = 0.0,
                       potential = -0.01)

        graph.add_edge(output_name,
                       complete_name,
                       weight = 1.0,
                       delay = 2.0)
        graph.add_edge(output_name, output_name, weight = -5.0, delay = 2.0)

        #for input_signal in input_lists[0]:
            #graph.add_edge(input_signal,
                           #begin_node_name,
                           #weight = 1.0,
                           #delay = 1.0)
        
        increment_timer_name = "T_I"
        decrement_timer_name = "T_D"
        graph.add_node(increment_timer_name,
                        threshold = self.num_elements - 0.01,
                        decay = 0.0,
                        potential = 0.0)
        graph.add_edge(increment_timer_name, increment_timer_name, weight = self.num_elements, delay = 2.0)
        graph.add_edge(increment_timer_name, output_name, weight = 1.0, delay = 2.0)
        graph.add_node(decrement_timer_name,
                        threshold = 0.99,
                        decay = 0.0,
                        potential = 1.0)
        graph.add_edge(decrement_timer_name, decrement_timer_name, weight = 1.0, delay = 2.0)
        graph.add_edge(decrement_timer_name, output_name, weight = -1.0, delay = 2.0)

        graph.add_edge(output_name, increment_timer_name, weight = -1 * self.num_elements, delay = 2.0)
        graph.add_edge(output_name, decrement_timer_name, weight = -1 * self.num_elements, delay = 2.0)

        for input_list in input_lists:
            for input_signal in input_list:
                graph.add_edge(input_signal, increment_timer_name, weight = 1.0, delay = 2.0)
                graph.add_edge(input_signal, decrement_timer_name, weight = -2.0, delay = 2.0)

        self.is_built=True

        #Remember, bricks can have more than one output, so we need a list of list of output neurons
        output_lists = [[output_name]]

        return (graph, self.metadata,
                [{'complete':complete_node_list[0], 'begin':begin_node_name}],
                output_lists, self.output_codings)

class Register(Brick):
    '''
    Brick that stores the binary encoding of an non-negative integer.
    Brick also allows the value stored to be incremented (but not decremented).
    '''

    def __init__(self, max_size, initial_value=0, name=None, output_coding='Undefined'):
        super(Brick, self).__init__()
        self.is_built = False
        self.name = name
        self.supported_codings = input_coding_types

        self.output_codings = [output_coding]
        self.metadata = {'D':None}

        self.max_size = max_size
        self.initial_value = initial_value

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        """
        Build Register brick.

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
        num_inputs = 0
        input_names = []
        for input_list in input_lists:
            num_inputs += len(input_list)
            for name in input_list:
                input_names.append(name)

        if num_inputs < 2:
            raise ValueError("Too few inputs: requires at least two, one for recall and others for incrementing")
        for input_coding in input_codings:
            if input_coding not in self.supported_codings:
                raise ValueError("Unsupported Input Coding. Found: {}. Allowed: {}".format(input_coding, self.supported_codings))

        begin_node_name = self.name+'_begin'
        graph.add_node(begin_node_name,
                       threshold = 0.1,
                       decay = 0.0,
                       potential=0.0)

        complete_name = self.name + '_complete'
        graph.add_node(complete_name,
                threshold = 0.1,
                decay = 0.0,
                potential = 0.0)
        complete_node_list = [complete_name]

        recall_name = "recall"
        graph.add_node(recall_name, threshold = 0.99, decay = 0.0, potential = 0.0)
        reset_name = "reset"
        graph.add_node(reset_name, threshold = 0.99, decay = 0.0, potential = 0.0)

        graph.add_edge(recall_name, reset_name, weight = 1.0, delay = self.max_size)

        add_name = "add_one"
        graph.add_node(add_name, threshold = 0.99, decay = 0.0, potential = 0.0)
        #add_counter = "add_counter"
        #graph.add_node(add_counter, threshold = 0.00, decay = 0.0, potential = 0.00)

        #graph.add_edge(add_name, add_name, weight = 1.0, delay = self.max_size)
        #graph.add_edge(add_name, add_counter, weight = 1.0, delay = 1.0)
        #graph.add_edge(add_counter, add_name, weight = -1.0, delay = 1.0)
        
        graph.add_edge(input_names[0], recall_name, weight = 1.0, delay = 1.0)
        for input_name in input_names[1:]:
            #graph.add_edge(input_name, add_counter, weight = -0.99, delay = 1.0)
            graph.add_edge(input_name, add_name, weight = 1.0, delay = self.max_size)

        register_name_base  = "slot_{}"
        output_name_base = "output_{}"
        outputs = []

        # determine initial states
        initial_value = self.initial_value
        bit_string = [0.0 for i in range(self.max_size)]
        for i in range(self.max_size - 1, -1, -1):
            power_of_2 = 2 ** i
            if power_of_2 <= initial_value:
                initial_value -= power_of_2
                bit_string[i] = 1.0

        for i in range(self.max_size):
            register_name = register_name_base.format(i)
            output_name = output_name_base.format(i)
            outputs.append(output_name)

            graph.add_node(register_name, threshold = 1.99, decay = 0.0, potential = bit_string[i])
            graph.add_node(output_name, threshold = 1.99, decay = 1.0, potential = 0.0)
            
            graph.add_edge(register_name, output_name, weight = 1.0, delay = 1.0)
            graph.add_edge(recall_name, register_name, weight = 1.0, delay = 1.0)
            graph.add_edge(recall_name, output_name, weight = 1.0, delay = 2.0)

        for i in range(1, self.max_size):
            prev_name = register_name_base.format(i - 1)
            curr_name = register_name_base.format(i)
            graph.add_edge(prev_name, curr_name, weight = 1.0, delay = 1.0)

        graph.add_edge(add_name, register_name_base.format(0), weight = 1.0, delay = self.max_size)

        self.is_built=True

        #Remember, bricks can have more than one output, so we need a list of list of output neurons
        output_lists = [outputs]

        return (graph, self.metadata,
                [{'complete':complete_node_list[0], 'begin':begin_node_name}],
                output_lists, self.output_codings)
