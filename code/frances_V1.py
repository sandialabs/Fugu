#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon April 22 23:18:56 2019

@author: fschanc

simple brick = essentially a simple cell.  "center" is how many pixels in each direction you go (zero will give you a one-pixel center), 
    "surround" is how many steps outside of center (0 gives you no surround)
hR_DS brick = essentially a half-Reichardt model (convergence of two spatially offset inputs, one with a delay)
fR_DS brick = based on the full-Reichardt model (two anti-symmetric half-Reichardts inhibit each other)
figure 1 is a cartoon of visual input, figure 2 is the spiking output of the models.  
    Neurons 0 - 10 are input neurons 
    Neurons 12-21,23-32 are half-Reichardt models - there should be 10x2=20, two for each spatial location (one in each direction)
    Neurons 54-63, 85-94 are full-Reichardt models (yes I know I didn't have to make the half-Reichardts all over again) again 5 pairs of detectors, one for each direction
    Neurons 96-105 are simple cells tuned to only respond to single-pixel stimuli
    Neurons 107-116 are stimple cells tuned to respond to edges
    Neurons 118-127 are half-Reichardts hooked up to simple cells (pixel detectors)
    Neurons 
hR_DS and fR_DS should respond to moving stimuli for one pixel step per time step
disadvantage to hRds is that they also respond to flashing stimuli if the stimuli are bigger than 1 pixel
neither will respond to stimuli moving faster than 1 pixel/time step
    
"""


import networkx as nx
import numpy as np
import fugu
from fugu import Scaffold, Brick, Spike_Input, Copy 
import matplotlib.pyplot as plt
from IPython import get_ipython

 

### simple cell brick ##################################
class Simple(Brick):
    def __init__(self, ctr, ctr_weight, surr, surr_weight, max_cols, name=None):   #A change here
        super(Brick, self).__init__()
        #The brick hasn't been built yet.
        self.is_built = False
        self.ctr = ctr
        self.ctr_weight = ctr_weight
        self.surr = surr
        self.surr_weight = surr_weight
        self.max_cols = max_cols
        #Leave for compatibility, D represents the depth of the circuit.  Needs to be updated.
        self.dimensionality = {'D':1}  
        #We just store the name passed at construction.
        self.name = name
        #For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = fugu.input_coding_types
    def build(self,
             graph,
             dimensionality,
             control_nodes,
             input_lists,
             input_codings):
        #check that input data dimensions make sense
        #Keep the same output coding as input?
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
        #We also, obviously, need to build the computational portion of our graph
        for idx_num, operand0 in enumerate(input_lists[0]):
            simple_node_name = self.name + '_' + operand0  #str(operand0)
            output_lists[0].append(simple_node_name)
            graph.add_node(simple_node_name,
                           index=0,
                           threshold=0.5,
                           decay=1.0,
                           p=1.0,
                           potential=0.0
                          )
            for i in range(idx_num-self.ctr,idx_num+self.ctr+1):
                if (i>=0 and i<self.max_cols):
                    graph.add_edge(input_lists[0][i],
                                  simple_node_name,
                                  weight=self.ctr_weight,
                                  delay=1.0)    
            if self.surr > 0:        
                for i in range(idx_num-self.ctr-self.surr,idx_num-self.ctr):
                    if (i>=0 and i<self.max_cols):
                        graph.add_edge(input_lists[0][i],
                                      simple_node_name,
                                      weight=-self.surr_weight,
                                      delay=1.0)      
                for i in range(idx_num+self.ctr+1,idx_num+self.ctr+self.surr+1):
                    if (i>=0 and i<self.max_cols):
                        graph.add_edge(input_lists[0][i],
                                      simple_node_name,
                                      weight=-self.surr_weight,
                                      delay=1.0)                      

        self.is_built=True
            
        return (graph,
               self.dimensionality,
                control_nodes,
                output_lists,
                output_codings
               ) 
        
 
### half-Reichardt brick ##################################
        # this is the dumbest way possible to make a direction-selective detector
class hR_DS(Brick):
    def __init__(self, dir_step, max_cols, name=None):   #A change here
        super(Brick, self).__init__()
        #The brick hasn't been built yet.
        self.is_built = False
        self.dir_step = dir_step
        self.max_cols = max_cols
        #Leave for compatibility, D represents the depth of the circuit.  Needs to be updated.
        self.dimensionality = {'D':1}  
        #We just store the name passed at construction.
        self.name = name
        #For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = fugu.input_coding_types
    def build(self,
             graph,
             dimensionality,
             control_nodes,
             input_lists,
             input_codings):
        #check that input data dimensions make sense
        #Keep the same output coding as input?
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
        #We also, obviously, need to build the computational portion of our graph
        for idx_num, operand0 in enumerate(input_lists[0]):
            ds_node_name = self.name + '_' + operand0  #str(operand0)
            output_lists[0].append(ds_node_name)
            graph.add_node(ds_node_name,
                           index=0,
                           threshold=0.7,
                           decay=1.0,
                           p=1.0,
                           potential=0.0
                          )
            graph.add_edge(operand0,
                          ds_node_name,
                          weight=0.5,
                          delay=1.0)    
            if (idx_num-self.dir_step >= 0 and idx_num-self.dir_step < self.max_cols):
                graph.add_edge(input_lists[0][idx_num-self.dir_step],
                          ds_node_name,
                          weight=0.5,
                          delay=2.0)
        self.is_built=True
            
        return (graph,
               self.dimensionality,
                control_nodes,
                output_lists,
                output_codings
               )        
        
        
        
        
### full-Reichardt brick ##################################
        # less dumb way of making a Reichardt detector - will still be very velocity sensitive
class fR_DS(Brick):
    def __init__(self, dir_step, max_cols, name=None):   #A change here
        super(Brick, self).__init__()
        #The brick hasn't been built yet.
        self.is_built = False
        self.dir_step = dir_step
        self.max_cols = max_cols
        #Leave for compatibility, D represents the depth of the circuit.  Needs to be updated.
        self.dimensionality = {'D':1}  
        #We just store the name passed at construction.
        self.name = name
        #For this example, we'll let any input coding work even though the answer might not make sense.
        self.supported_codings = fugu.input_coding_types
    def build(self,
             graph,
             dimensionality,
             control_nodes,
             input_lists,
             input_codings):
        #check that input data dimensions make sense
        #Keep the same output coding as input?
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
        #We also, obviously, need to build the computational portion of our graph
        for idx_num, operand0 in enumerate(input_lists[0]):
            pre1_node_name = 'half-_' + self.name + '_' + operand0  #str(operand0)
        #    output_lists[0].append(pre1_node_name)  # we essentially do half-Reichardts en route to full Reichardts, but that isn't the point
            graph.add_node(pre1_node_name,
                           index=0,
                           threshold=0.7,
                           decay=1.0,
                           p=1.0,
                           potential=0.0
                          )
            graph.add_edge(operand0,
                          pre1_node_name,
                          weight=0.5,
                          delay=1.0)    
            if (idx_num-self.dir_step >= 0 and idx_num-self.dir_step < self.max_cols):
                graph.add_edge(input_lists[0][idx_num-self.dir_step],
                          pre1_node_name,
                          weight=0.5,
                          delay=2.0)
        for idx_num, operand0 in enumerate(input_lists[0]):         
            pre2_node_name = 'half+_' + self.name + '_' + operand0  #str(operand0)
        #    output_lists[0].append(pre2_node_name)
            graph.add_node(pre2_node_name,
                           index=0,
                           threshold=0.7,
                           decay=1.0,
                           p=1.0,
                           potential=0.0
                          )
            graph.add_edge(operand0,
                          pre2_node_name,
                          weight=0.5,
                          delay=1.0)    
            if (idx_num+self.dir_step >= 0 and idx_num+self.dir_step < self.max_cols):
                graph.add_edge(input_lists[0][idx_num+self.dir_step],
                          pre2_node_name,
                          weight=0.5,
                          delay=2.0)    
        for idx_num, operand0 in enumerate(input_lists[0]):
            ds1_node_name = 'ds' + self.name + '_' + operand0  #str(operand0)
            output_lists[0].append(ds1_node_name)
            graph.add_node(ds1_node_name,
                           index=0,
                           threshold=0.5,
                           decay=1.0,
                           p=1.0,
                           potential=0.0
                          )
            pre1_node_name = 'half-_' + self.name + '_' + operand0  #str(operand0)
            graph.add_edge(pre1_node_name,
                          ds1_node_name,
                          weight=1.0,
                          delay=1.0)    
            if (idx_num-self.dir_step >= 0 and idx_num-self.dir_step < self.max_cols):
                pre2_node_name = 'half+_' + self.name + '_' + input_lists[0][idx_num-self.dir_step]   ### MY PROBLEM IS ADDING THIS EDGE...
                graph.add_edge(pre2_node_name,
                          ds1_node_name,
                          weight=-1.0,
                          delay=1.0)    
            
        self.is_built=True
            
        return (graph,
               self.dimensionality,
                control_nodes,
                output_lists,
                output_codings
               )


        
## ideally it would be time + 2D but let's do time+1D for now       
numrows = 1;
numcols = 10
numframes = 50
Spiking_Input = np.zeros((numcols,numframes))  ##shoud number of frames always be the first dimension?
stim_index = 0
for i in range(min(5,numframes)):
    Spiking_Input[stim_index,i]=1
    stim_index+=1
    if stim_index >= numcols:
        stim_index = 0
if numframes>5:
    stim_index = numcols-2
    for i in range(5,9):
        Spiking_Input[stim_index,i]=1
        stim_index-=1
        if stim_index < 0:
            stim_index = numcols-1
if numframes>9:
    i = 9;
    while i < min(22,numframes):
        Spiking_Input[range(0,numcols),i]=1
        i += 1
        if i < numframes:
            Spiking_Input[range(0,numcols),i]=1
        i += 1
        if i < numframes:
            Spiking_Input[range(0,numcols),i]=1            
        i += 1
        if i < numframes:
            Spiking_Input[range(0,numcols),i]=0
        i += 1
        if i < numframes:
            Spiking_Input[range(0,numcols),i]=0
        i += 1
        if i < numframes:
            Spiking_Input[range(0,numcols),i]=0      
if numframes > 22: 
    stim_index=0
    for i in range(22,min(30,numframes)):
        Spiking_Input[stim_index,i]=1
        stim_index+=2
        if stim_index >= numcols:
            stim_index = 0
stim_index = 0
stimsize = 3
if numframes >= 30:
    for i in range(min(30,numframes),numframes):
        if stim_index+stimsize < numcols:
            Spiking_Input[range(stim_index,stim_index+stimsize),i]=1
        if stim_index+stimsize >= numcols:
            Spiking_Input[range(stim_index,numcols),i]=1
            print(stim_index+stimsize-numcols)
            Spiking_Input[range(0,stim_index+stimsize-numcols),i]=1
        stim_index+=1
        if stim_index>=numcols:
            stim_index=0
            
scaffold = Scaffold()
scaffold.add_brick(Spike_Input(Spiking_Input, time_dimension='True', coding='Raster', name='InputSequence'), 'input' )
scaffold.add_brick(hR_DS(name='hRds1',dir_step=1,max_cols=numcols), (0,0), output=True)
scaffold.add_brick(hR_DS(name='hRds2',dir_step=-1,max_cols=numcols), (0,0), output=True)
scaffold.add_brick(fR_DS(name='fRds1',dir_step=1,max_cols=numcols), (0,0), output=True)
scaffold.add_brick(fR_DS(name='fRds2',dir_step=-1,max_cols=numcols), (0,0), output=True)
scaffold.add_brick(Simple(name='simple1', ctr=0, ctr_weight=1.0, surr=1, surr_weight=0.5, max_cols=numcols), (0,0), output=True)
scaffold.add_brick(Simple(name='simple2', ctr=0, ctr_weight=1.0, surr=1, surr_weight=0.4, max_cols=numcols), (0,0), output=True)
scaffold.add_brick(hR_DS(name='hRds3',dir_step=1,max_cols=numcols), (5,0), output=True)



#print('\n\n-------------------------------------------------------\nNodes:')
#for node in scaffold.circuit.nodes:
#    print("node",node,": ",[scaffold.circuit.nodes[node]['brick'].name],)
#print("\n")   
 
scaffold.lay_bricks()
 
result = scaffold.evaluate(max_runtime=100,backend='ds', record_all=False)

scaffold.summary()

print('binary input image sequence (transposed to each row is a time step)...: ')
print('\n\n-------------------------------------------------------\nInputs:')
#print(Spiking_Input.transpose()) 

print('\n\n-------------------------------------------------------\nResults:')
print(result) 

# image raster
tdata=[]
pdata=[]
for i in range(0,numframes):
    ii = np.where(Spiking_Input[range(0,numcols),i]==1)[0]
    if len(ii) > 0:
        for j in range(0,len(ii)):
            tdata.append(i)
            pdata.append(ii[j])
get_ipython().run_line_magic('matplotlib', 'qt')   
plt.figure(1)
plt.plot(tdata,pdata,'bs')
plt.xlabel('time step')
plt.xlim([-0.5,numframes])
plt.xticks(np.arange(0,numframes+1,1))
plt.ylabel('stimulus')
plt.ylim([-0.5,numcols-0.5])


## neuron raster
tndata=[]
ndata=[]
for idx, item in enumerate(result):
    if len(result[idx]) > 0:
        for i in range(0,len(result[idx])):
            tndata.append(idx)
            ndata.append(result[idx][i])  
plt.figure(2)        
plt.plot(tndata,ndata,'r|')
plt.xlabel('time step')
plt.xlim([0,numframes])
plt.xticks(np.arange(0,numframes+1,1))
plt.ylabel('neuron index')
plt.ylim([10,94])

plt.figure(3)        
plt.plot(tndata,ndata,'r|')
plt.xlabel('time step')
plt.xlim([0,numframes])
plt.xticks(np.arange(0,numframes+1,1))
plt.ylabel('neuron index')
plt.ylim([95,len(scaffold.graph.nodes)])
