# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:36:48 2019

@author: jbaimon
"""

import numpy as np
import networkx as nx
from math import log, ceil

class time2binary:
    def __init__(self):
        pass
    
    def resolveParams(N_in, T_in):
        T_out=1
        # Convert T_in to a binary code
        N_out=max(log(T_in))*N_in
        D=T_in+1
        
    
    def buildBrick(self, N_in, N_out, T_in, T_out, D):
        # Builds a circuit which converts an input of size N_inxTin to a binary code output
        g=nx.DiGraph()
        k=0
        # Build input layer; which exists in standalone form, but will be replaced in implementation
        g.add_node(N_in, layer='inputs')
        in_start=0
        in_end=N_in-1
        # Build internal counter layer; which exists to facilitate conversion
        g.add_node(T_in, layer='internal', tau=0)
        cycle_start=N_in
        cycle_end=N_in+T_in-1
        # Build intermediate binary layer; which will be ported into output layer.
        g.add_node(N_out, layer='internal', tau=0)  
        int_start=N_in+T_in
        int_end=N_in+T_in+N_out-1
        # Build output binary layer; which will provide outputs at single time-step
        g.add_node(N_out, layer='outputs', tau=-1)
        out_start=int_end+1
        out_end=out_start+N_out-1
        bin_size=int(ceil(log(T_in)))
        
        # Build a list of connections representing binary encoding from ring to binary
        # Counter layer probably needs to be activated by something
        ring2binary=[]
        for i in range(1, T_in+1):
            binary_i=bin(i)[2:].zfill(int(bin_size))
            ring2binary.append(list(binary_i))
            if(i<T_in):
                g.add_edge(i, i+1, weight=1.0)
    
        for i in range(0, N_in):
            for k in range(0, int(bin_size)+1):
                # Input from input neuron to each neuron in intermediate bin
                g.add_edge(i, int_start+i*bin_size+k, weight=0.5)
                # Input from intermediate bin neuron to corresponding output neuron
                g.add_edge(int_start+i*bin_size+k, out_start+i*bin_size+k, weight=0.5)
                # Provide input to final output layer from last ring neuron
                g.add_edge(cycle_end, out_start+i*bin_size+k, weight=0.5)
                
            for j in range(0, T_in):
                # Map ring neurons to appropriate binary outputs
                k=0
                for target in ring2binary[j]:
                    if target=='1':
                        g.add_edge(cycle_start+j, int_start+i*bin_size+k, weight=0.5)
                    k=k+1
        
        
        g.adjacency_list()
        return g
    
class binary2time:
    def __init__(self):
        pass
    
    def resolveParams(N_in, bin_size):
        # somehow we need to communicate the # binary bits alongside of a binary code
        N_out=N_in/bin_size
        T_out=2^bin_size
        D=T_out+1
        
    
    def buildBrick(self, N_in, N_out, T_in, T_out, D):
        # Builds a circuit which converts a binary input to an N_outxTout to a time coded output
        g=nx.DiGraph()
        k=0
        # Build input layer; which exists in standalone form, but will be replaced in implementation
        g.add_node(N_in, layer='inputs')
        in_start=0
        in_end=N_in-1
        
        bin_size=int(N_in/N_out)
        # Build internal counter layer; which exists to facilitate conversion
        g.add_node(T_out*(bin_size+1), layer='internal')
        cycle_start=N_in
        cycle_end=N_in+T_out*(bin_size+1)-1
        # Build intermedia layer; which will drive output nodes.
        g.add_node(N_out*3, layer='internal')  
        int_start=cycle_end+1
        int_end=int_start+N_out*3-1
        
        g.add_node(N_out, layer='outputs')
        out_start=int_end+1
        out_end=out_start+N_out-1
        # Build a population of binary coded neurons that are sequentially active.
        ring2binary=[]
        ring_pos=0
        i_ct=1
        for i in range(cycle_start, cycle_end+1, bin_size+1):
            binary_i=bin(i_ct)[2:].zfill(int(bin_size))
            ring2binary.append(list(binary_i))
            #if(i<T_in):
            g.add_edge(i, i+bin_size+1, weight=1.0)
            k=1
            for target in ring2binary[i_ct-1]:
                if target=='1':
                    g.add_edge(i, i+k, weight=1.0)
                
                for int_target in range(int_start, int_end, 3):
                    g.add_edge(i+k, int_target, weight=-1.0)
                    g.add_edge(i+k, int_target+1, weight=1.0)
                    
                k=k+1
            i_ct=i_ct+1
        
        for i in range(0, N_in, bin_size):
            for k in range(0, bin_size):
                #for int_target in range(int_start, int_end+1, 3):
                g.add_edge(i+k, int_start+3*i/bin_size, weight=1.0)
                g.add_edge(i+k, int_start+1+3*i/bin_size, weight=-1.0)
                g.add_edge(i+k, int_start+2+3*i/bin_size, weight=1.0)

        k=out_start                    
        for i in range(int_start, int_end+1, 3):
            g.add_edge(i, k, weight=-1.0)
            g.add_edge(i+1, k, weight=-1.0)
            g.add_edge(i+2, k, weight=1.0)
            k=k+1
        
        g.adjacency_list()
        return g