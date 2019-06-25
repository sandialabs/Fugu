# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 07:50:46 2019

@author: jbaimon
"""
from math import floor, ceil, log
import random
import networkx as nx
import numpy as np
import fugu
from fugu import Scaffold, Brick, Spike_Input, InputBrick


     

    
class random_source(InputBrick):
    def __init__(self, N=None, T=1, time_dimension=False, coding='Undefined', name=None):
        #
        # N is the size of the random binary number, in bits
        
        self.is_built = False
        self.dimensionality={'D':0}
        self.name=name
        self.supported_codings = []
        self.N = N
        self.T = T
        self.time_dimension = time_dimension
        self.coding = coding
        
        if N is None:
            raise ValueError('Need to define size of random souce N')
    
    def get_input_value(self, t=None):
        pass

   
    def build(self, graph, dimensionality, complete_node, input_lists, input_codings):
        
        # Expect 0 inputs
        if input_codings:
            raise ValueError('Random source does not have any inputs!')
        
#        print(input_codings)
        #if len(input_codings)!=-2:
            #raise ValueError('Random source does not have any inputs!')
        
        # Output is random binary source over T time steps for N neurons
        output_codings=['binary-b']
        
        output_lists=[[]]
        
        # Define complete node
        new_complete_node_name = self.name + '_complete'
        graph.add_node(new_complete_node_name, index=-1, threshold = -1.0, decay = 0.0, p=1.0, potential = 0.0)
        # one time-step computation
        

        for i in range(0, self.N):
            neuron_name = self.name+"_" + str(i)
            graph.add_node(neuron_name, index=i, threshold=-1.0, decay=0.0, p=0.5)
            output_lists[0].append(neuron_name)
            graph.add_edge(new_complete_node_name, neuron_name, weight=-5.0, delay=self.T)
        
        graph.add_edge(new_complete_node_name, new_complete_node_name, weight=-5.0, delay=1)
        complete_node=[new_complete_node_name]
                    
        

        self.is_built = True
        return(graph, {'output_shape':self.N, 'output_coding':self.coding, 'layer' : input, 'D':0}, complete_node, output_lists, output_codings)

class binary2unary(Brick):
    def __init__(self, name=None):
        super(Brick, self).__init__()
        self.is_built = False
        self.dimensionality={'D':1}
        self.name=name
        self.supported_codings = fugu.input_coding_types
    def build(self, graph, dimensionality, complete_node, input_lists, input_codings):
        # Expect 1 inputs
        if len(input_codings)!=1:
            raise ValueError('Expected only one input')
        
        output_codings=['binary-B']
        if(input_codings[0]=='binary-B'):
            # Circuit for big Endian coding
            output_codings=['unary-B']
        
        if(input_codings[0]=='binary-L'):
            output_codings=['unary-L']
            
        # Define complete node
        new_complete_node_name = self.name + '_complete'
        graph.add_node(new_complete_node_name, index=-1, threshold = 0.0, decay = 0.0, p=1.0, potential = 0.0)
        # one time-step computation
        
        graph.add_edge(complete_node[0], new_complete_node_name, weight=1.0, delay=1)
        complete_node=[new_complete_node_name]
        
        # Define circuit
        
        # Define size of input layer
        #N_size=ceil(log(N,2))
        
        N_size=len(input_lists[0])
        N=2**N_size
        # N_size should be size of the input layer
        
        output_lists=[[]]
        for i in range(0, N):
            binary_i=bin(i)[2:].zfill(N_size)
            sum_bin=sum(int(binary_i_1) for binary_i_1 in binary_i)
            node_name=self.name+'_'+str(i)
            graph.add_node(node_name, index=i, threshold=sum_bin, decay=1.0, p=1.0, potential=0.0)
            output_lists[0].append(node_name)
            for j in range(0, N_size):
                graph.add_edge(input_lists[0][j], output_lists[0][i], weight=int(2*(int(binary_i[j])-.5)), delay=1.0)
            
        self.is_built = True
        return (graph, self.dimensionality, complete_node, output_lists, output_codings)
        
        
class grid_ring(Brick):
    def __init__(self, name=None, ringSize=7):
        super(Brick, self).__init__()
        self.is_built = False
        self.ringSize=ringSize
        self.dimensionality={'D':1}
        self.name=name
        self.supported_codings = fugu.input_coding_types
        
    def build(self, graph, dimensionality, complete_node, input_lists, input_codings):
        
        # Grid RW will produce a 2D grid-cell like module of a given size with a number of random walkers starting from a predefined position
        if (len(input_codings)!=1):
            raise ValueError('Expected only one input')
            
        # Define complete node
        new_complete_node_name = self.name + '_complete'
        graph.add_node(new_complete_node_name, index=-1, threshold = 0.0, decay = 0.0, p=1.0, potential = 0.0)
        # one time-step computation
        graph.add_edge(complete_node[0], new_complete_node_name, weight=1.0, delay=1)
        complete_node=[new_complete_node_name]
        
        # Set up grid circut
        output_lists=[[]]
        nodes_list=[]
        output_codings='raster'
        # Determine population sizes
        #
        # The Ring is the 2D ring (really a prime-number sized torus) that the random walk moves over
        
        ringSize=self.ringSize
        num_ring=ringSize**2
        
        # There are 6 control rings.  These are smaller (repeated use after 3-5 distance) sets of control nodes that push activity in one of the 6 directions
        num_control_ring_size=(ringSize%3+3)**2
        
        # Track what neuron index each control ring population starts
        control_ring_counts=[]
        for i in range(0, 6):
            control_ring_counts.append(num_control_ring_size*i)
        
        control_dim=ringSize%3+3
        
        # Add neurons for all the ring population and control ring population
        for i in range(0, num_ring+6*num_control_ring_size):
            node_name=self.name+'_'+str(i)
            graph.add_node(node_name, index=i, threshold=1.0, decay=1.0, p=1.0, potential=0.0)
            if(i<num_ring):
                output_lists[0].append(node_name)
                
            nodes_list.append(node_name)
            # Neurons 0 ... numRing-1 are ring neurons
            # Neurons numRing... numRing+num_control are control 
        

        k=0
        for i in range(0, num_ring):
            y_shift=floor(i/ringSize)
            
#            control_node=y_shift*control_dim+x_shift
            # For each ring neuron, compute which control ring neurons it projects to, and which it receives projections from...
            target_y=[y_shift, y_shift, y_shift, (y_shift-1)%ringSize, (y_shift-1)%ringSize, (y_shift+1)%ringSize, (y_shift+1)%ringSize]
            targets=[i, i-1, i+1, i-ringSize, i-ringSize-1, i+ringSize, i+ringSize+1]
            control_targets=[]
            target_x=[]
            # Correct target populations to account for going around rings
            for j in range(0, 7):
                targets[j]=targets[j]%num_ring
                if(floor(targets[j]/ringSize)<target_y[j]):
                    targets[j]=targets[j]+ringSize
                elif(floor(targets[j]/ringSize)>target_y[j]):
                    targets[j]=targets[j]-ringSize
                
                target_x.append((targets[j]-target_y[j]*ringSize))
                control_targets.append(control_dim*(target_y[j]%3)+(target_x[j]%3))
                if(control_dim==4):
                    # Check for one edge layer
                    if(target_y[j]==ringSize-1):
                        # Need to fix
                        control_targets[j]=control_targets[j]+12
                    if(target_x[j]%ringSize==ringSize-1):
                        control_targets[j]=control_targets[j]+3
                if(control_dim==5):
                    if(target_y[j]==ringSize-1):
                        # Need to fix
                        control_targets[j]=control_targets[j]+15
                    elif(target_y[j]==ringSize-2):
                        control_targets[j]=control_targets[j]+12
                    if(target_x[j]==ringSize-1):
                        control_targets[j]=control_targets[j]+4
                    elif(target_x[j]==ringSize-2):
                        control_targets[j]=control_targets[j]+3
            
            # Add connections to neighboring ring neurons
            graph.add_edge(nodes_list[i], nodes_list[targets[0]], weight=1.1, delay=2)
            
            # Add connections to appropriate downstream control_ring neurons
            for j in range(1,7):
                graph.add_edge(nodes_list[i], nodes_list[targets[j]], weight=.6, delay=2)
                        
            
            # Add connections from appropriate upstream control_ring neurons
            for j in range(0,6):
                graph.add_edge(nodes_list[i], nodes_list[num_ring+control_ring_counts[j]+control_targets[0]], weight=.6, delay=1)
                graph.add_edge(nodes_list[num_ring+control_ring_counts[j]+control_targets[0]], nodes_list[i], weight=-0.5, delay=1)
                graph.add_edge(nodes_list[num_ring+control_ring_counts[j]+control_targets[j+1]], nodes_list[i], weight=.6, delay=1)              
        
        k=num_ring        
        # Add connections from input neurons to control_ring neurons
        
        
        for i in range(0, 6):
            # Loop through each output set
            for j in range(0, num_control_ring_size):
                graph.add_edge(input_lists[0][i], nodes_list[k], weight=.5, delay=1)
                
                k=k+1
       
        self.is_built = True
        return (graph, self.dimensionality, complete_node, output_lists, output_codings)
                
        # NOTE:  Probably need a connection from the upstream brick's control neuron to origin neurons
        
        # NOTE: Need to somehow account for time in this model
