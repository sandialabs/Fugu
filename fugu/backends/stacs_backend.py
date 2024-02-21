#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import deque
from warnings import warn

from .backend import Backend
from ..utils.export_utils import results_df_from_dict
from ..utils.misc import CalculateSpikeTimes

import os
import subprocess
import yaml
import numpy as np
import networkx as nx
import pandas as pd

class stacs_Backend(Backend):
    def _create_dirs(self):
        # Create necessary directory structure
        if not os.path.exists(self.netwkdir):
            os.makedirs(self.netwkdir)
        netfiles = self.netwkdir + '/files'
        if not os.path.exists(netfiles):
            os.makedirs(netfiles)
        recordir = self.netwkdir + '/record'
        if not os.path.exists(recordir):
            os.makedirs(recordir)

    def _export_graph(self):
        # Export Fugu Circuit to files
        fugufiles = self.netwkdir + '/files'
        # Some helper data structures for the conversion
        neuronindex = {}
        neuronmap = {}
        neuronname = {}
        for i, neuron in enumerate(self.fugu_graph.nodes):
            neuronindex[neuron] = i
            neuronname[str(neuron)] = i
            neuronmap[i] = neuron

        neuron_default = {'v':0.0,'v_thresh':0.0,'v_reset':0.0,'v_bias':0.0,'v_leak':1.0,'p_spike':1.0,'I_syn':0.0}
        # Input Neurons require both control nodes and output lists
        input_neurons = []
        input_spikes = {}

        for brick in self.fugu_circuit.nodes:
            if 'layer' in self.fugu_circuit.nodes[brick] and self.fugu_circuit.nodes[brick]['layer'] == 'input':
                # Control Nodes
                #print(self.fugu_circuit.nodes[brick]['control_nodes'][0]['begin'])
                neuron = self.fugu_circuit.nodes[brick]['control_nodes'][0]['begin']
                input_neurons.append(neuronindex[neuron])
                input_spikes[neuronindex[neuron]] = list([1])
                # Need to update the default potential for synchronization of timing
                # (instead of spiking due to initial potential, spike when inputs arrive)
                self.fugu_graph.nodes[self.fugu_circuit.nodes[brick]['control_nodes'][0]['begin']]['potential'] = 0.0
                # Output Lists
                #print(self.fugu_circuit.nodes[brick]['output_lists'])
                for l, lists in enumerate(self.fugu_circuit.nodes[brick]['output_lists']):
                    for n, neuron in enumerate(lists):
                        input_neurons.append(neuronindex[neuron])
                        input_spikes[neuronindex[neuron]] = (self.fugu_circuit.nodes[brick]['brick'].vector[n]).tolist()

        input_neurons.sort() # This gets them in neuron order

        event_list = []
        spike_input = {}
        for neuron in input_neurons:
            evtlist = []
            ts = 0
            for spike in input_spikes[neuron]:
                if spike == 1:
                    evtlist.append(ts)
                ts += 1
            event_list.append(evtlist)
        spike_input["spike_list"] = event_list
        with open(fugufiles + "/fugu_input.yml","w") as file:
            yaml.dump(spike_input,file)

        with open(fugufiles + '/fugu_input.csv','w') as fp_inp:
            for neuron in range(self.fugu_graph.number_of_nodes()):
                if neuron in input_neurons:
                    fp_inp.write('0\n')
                else:
                    fp_inp.write('\n')
        self.n_inputs = len(event_list)
        
        # Convert graph to sparse-csv files for stacs to build with
        # This essentially all gets converted into a flat population
        with open(fugufiles + '/fugu_v.csv','w') as fp_v, open(fugufiles + '/fugu_v_thresh.csv','w') as fp_vt, \
                open(fugufiles + '/fugu_v_reset.csv','w') as fp_vr, open(fugufiles + '/fugu_v_bias.csv','w') as fp_vb, \
                open(fugufiles + '/fugu_v_leak.csv','w') as fp_vl, open(fugufiles + '/fugu_p_spike.csv','w') as fp_ps,  \
                open(fugufiles + '/fugu_I_syn.csv','w') as fp_is, open(fugufiles + '/fugu_vtxname.csv','w') as fp_vn:
                    for neuron in self.fugu_graph.nodes:
                        if 'potential' in self.fugu_graph.nodes[neuron]:
                            fp_v.write(str(self.fugu_graph.nodes[neuron]['potential'])+'\n')
                        else:
                            fp_v.write(str(neuron_default['v'])+'\n')
                        if 'threshold' in self.fugu_graph.nodes[neuron]:
                            fp_vt.write(str(self.fugu_graph.nodes[neuron]['threshold'])+'\n')
                        else:
                            fp_vt.write(str(neuron_default['v_thresh'])+'\n')
                        if 'reset_voltage' in self.fugu_graph.nodes[neuron]:
                            fp_vr.write(str(graph.nodes[neuron]['reset_voltage'])+'\n')
                        else:
                            fp_vr.write(str(neuron_default['v_reset'])+'\n')
                        if 'bias' in self.fugu_graph.nodes[neuron]:
                            fp_vb.write(str(self.fugu_graph.nodes[neuron]['bias'])+'\n')
                        else:
                            fp_vb.write(str(neuron_default['v_bias'])+'\n')
                        if 'decay' in self.fugu_graph.nodes[neuron]:
                            fp_vl.write(str(self.fugu_graph.nodes[neuron]['decay'])+'\n')
                        else:
                            fp_vl.write(str(neuron_default['v_leak'])+'\n')
                        if 'p' in self.fugu_graph.nodes[neuron]:
                            fp_ps.write(str(self.fugu_graph.nodes[neuron]['p'])+'\n')
                        else:
                            fp_ps.write(str(neuron_default['p_spike'])+'\n')
                        if 'current' in self.fugu_graph.nodes[neuron]: # current doesn't actually exist
                            fp_is.write(str(self.fugu_graph.nodes[neuron]['current'])+'\n')
                        else:
                            fp_is.write(str(neuron_default['I_syn'])+'\n')
                        fp_vn.write(str(neuron)+'\n')

        # Connection weights and delays
        with open(fugufiles + '/fugu_weight.csv','w') as fp_wgt, open(fugufiles + '/fugu_delay.csv','w') as fp_del:
            for i, neuron in enumerate(self.fugu_graph.nodes):
                # Might need some sorting first?
                for edge in self.fugu_graph.in_edges(neuron):
                    fp_wgt.write(str(neuronindex[edge[0]]) + ':' + str(self.fugu_graph.edges[edge]['weight']) + ',')
                    fp_del.write(str(neuronindex[edge[0]]) + ':' + str(self.fugu_graph.edges[edge]['delay']) + ',')
                fp_wgt.write('\n')
                fp_del.write('\n')

        self.n_neurons = len(neuronindex)

    def _create_yaml(self):
        # Create default fugunet yaml files
        model_yml = """type: record
events:
- spike
- clamp
---
type: stream
modname: spike_input
modtype: 5
param:
- name: n
  value: 0
port:
- name: input
  value: files/fugu_input.yml
---
type: vertex
modname: fugu_neuron
modtype: 80
param: null
state:
- name: v
  init: file
  filetype: csv-dense
  filename: files/fugu_v.csv
- name: v_thresh
  init: file
  filetype: csv-dense
  filename: files/fugu_v_thresh.csv
- name: v_reset
  init: file
  filetype: csv-dense
  filename: files/fugu_v_reset.csv
- name: v_bias
  init: file
  filetype: csv-dense
  filename: files/fugu_v_bias.csv
- name: v_leak
  init: file
  filetype: csv-dense
  filename: files/fugu_v_leak.csv
- name: p_spike
  init: file
  filetype: csv-dense
  filename: files/fugu_p_spike.csv
- name: I_syn
  init: file
  filetype: csv-dense
  filename: files/fugu_I_syn.csv
- name: I_clamp
  init: constant
  value: 0.0
---
type: edge
modname: fugu_synapse
modtype: 81
param: null
state:
- name: delay
  init: file
  rep: tick
  filetype: csv-sparse
  filename: files/fugu_delay.csv
- name: weight
  init: file
  filetype: csv-sparse
  filename: files/fugu_weight.csv
---
type: edge
modname: synclamp
modtype: 15
state:
- name: delay
  init: constant
  rep: tick
  value: 1.0
"""
        modelname = self.netwkdir + '/' + self.filebase + '.model' 
        modelconfig = []
        with open(modelname,"w") as file:
            file.write(model_yml)
        # Modify the model configuration
        with open(modelname, "r") as file:
            modelymls = yaml.safe_load_all(file)
            for modelyml in modelymls:
                modelconfig.append(modelyml)
        # 0 is the records file, 1 is the fugu neurons population
        modelconfig[1]['param'][0]['value'] = self.n_inputs
        with open(modelname,"w") as file:
            yaml.dump_all(modelconfig,file,sort_keys=False) 

        graph_yml = """stream:
- modname: spike_input
  coord:
  - 0.0
  - 0.0
  - 0.0
vertex:
- modname: fugu_neuron
  order: 12
  shape: point
  coord:
  - 0.0
  - 0.0
  - 0.0
edge:
- source: spike_input
  target:
  - fugu_neuron
  modname: synclamp
  cutoff: 0.0
  connect:
  - type: file
    filetype: csv-sparse
    filename: files/fugu_input.csv
- source: fugu_neuron
  target:
  - fugu_neuron
  modname: fugu_synapse
  cutoff: 0.0
  connect:
  - type: file
    filetype: csv-sparse
    filename: files/fugu_delay.csv
"""
        graphname = self.netwkdir + '/' + self.filebase + '.graph' 
        with open(graphname,"w") as file:
            file.write(graph_yml)
        # Modify the graph configuration
        with open(graphname,"r") as file:
            graphconfig = yaml.safe_load(file)
        # Update the number of neurons in the population
        graphconfig['vertex'][0]['order'] = self.n_neurons
        with open(graphname,"w") as file:
            yaml.dump(graphconfig,file,sort_keys=False)

        sim_yml = """runmode: simulate
randseed: 1421
plastic: false
episodic: false
loadbal: false
selfconn: true
rpcport: /stacs/rpc
rpcpause: false
netwkdir: ./fugunet
netparts: 1
netfiles: 1
filebase: fugunet
fileload: ''
filesave: .out
recordir: record
groupdir: group
tstep: 1.0
teventq: 20.0
tdisplay: 1000.0
trecord: 10000.0
tbalance: 60000.0
tsave: 60000.0
tmax: 60000.0
"""

        simname = self.netwkdir + '/' + self.filebase + '.yml' 
        with open(simname,"w") as file:
            file.write(sim_yml)
        # Modify the simulation configuration
        with open(simname,"r") as file:
            simconfig = yaml.safe_load(file)
        simconfig['netwkdir'] = self.netwkdir
        simconfig['filebase'] = self.filebase
        simconfig['netfiles'] = self.npdat
        simconfig['netparts'] = self.npnet
        self.t_rec = simconfig['trecord']
        with open(simname,"w") as file:
            yaml.dump(simconfig,file,sort_keys=False)
        

    def _build_network(self):
        # Build the STACS network snapshot
        runlist = self.runcmd.split()
        runlist.append('build')
        if (self.debug_mode):        
            subprocess.run(runlist)
        else:
            subprocess.run(runlist, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def compile(self, scaffold, compile_args={}):
        # creates neuron populations and synapses
        self.fugu_circuit = scaffold.circuit
        self.fugu_graph = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        self.name_to_tag = scaffold.name_to_tag
        if 'record' in compile_args:
            self.record = compile_args['record']
        else:
            self.record = False
        if 'debug_mode' in compile_args:
            self.debug_mode = compile_args['debug_mode']
        else:
            self.debug_mode = False
        if 'stacsbin' in compile_args:
            self.stacsbin = compile_args['stacsbin']
        else:
            self.stacsbin = ''
        if 'netwkdir' in compile_args:
            self.netwkdir = compile_args['netwkdir']
        else:
            self.netwkdir = './fugunet'
        if 'filebase' in compile_args:
            self.filebase = compile_args['filebase']
        else:
            self.filebase = 'fugunet'
        if 'format' in compile_args:
            self.format = compile_args['format']
        else:
            self.format = 'dataframe'
        if 'npdat' in compile_args:
            self.npdat = compile_args['npdat']
        else:
            self.npdat = 1
        if 'npnet' in compile_args:
            self.npnet = compile_args['npnet']
        else:
            self.npnet = 1
        if 'nprun' in compile_args:
            self.nprun = compile_args['nprun']
        else:
            self.nprun = 1

        # Some run commands
        if self.stacsbin == '':
            charmrun = 'charmrun'
            stacsrun = 'stacs'
        else:
            charmrun = self.stacsbin + '/charmrun'
            stacsrun = self.stacsbin + '/stacs'
        charm_pe = '+p' + str(self.nprun)
        runconf = self.netwkdir + '/' + self.filebase + '.yml'
        self.runcmd = charmrun + ' ' + charm_pe + ' ' + stacsrun + ' ' + runconf
        if (self.debug_mode):
            print('Run command: ' + self.runcmd)

        # metadata
        self.n_inputs = 0
        self.n_neurons = 0
        self.t_rec = 1000
        self.t_max = 1000

        self._create_dirs()
        self._export_graph()
        self._create_yaml()
        self._build_network()

    def _collect_output(self):
        # Count the number of vertex models
        vtxmods = {}
        # This is done by looping through the .state files
        for datidx in range(0,self.npdat) :
            fname = self.netwkdir + '/' + self.filebase + '.state.' + str(datidx)
            with open(fname, 'r') as fstate:
                for line in fstate:
                    vtxmods[line.split(None, 1)[0]] = vtxmods.get(line.split(None, 1)[0], 0) + 1

        # Bookkeeping and counting
        maxidx = 0
        preidx = {}
        vtxidx = {}
        vertex_modnames = []
        population_prefix = []
        for key, value in vtxmods.items():
            vertex_modnames.append(key)
            preidx[key] = maxidx
            population_prefix.append(maxidx)
            vtxidx[key] = 0
            maxidx += value
        population_prefix.append(maxidx)

        # Reindexing
        index = 0
        vtxmap = np.zeros(maxidx)
        for datidx in range(0,self.npdat) :
            fname = self.netwkdir + '/' + self.filebase + '.state.' + str(datidx)
            with open(fname, 'r') as fstate:
                for line in fstate:
                    vertex = line.split(None, 1)[0]
                    vtxmap[index] = preidx[vertex] + vtxidx[vertex]
                    index += 1
                    vtxidx[vertex] += 1
                
        # Some metadata for plotting the spike raster
        TICKS_PER_MS = 1000000
        # rec times
        trecs = list(range(int(self.t_rec), int(self.t_max), int(self.t_rec)))

        # Check if files exist
        if (not trecs) :
            trecs.append(int(self.t_max))
        if (trecs[-1] < int(self.t_max)) :
            trecs.append(int(self.t_max))
        # check filenames
        for r in range(0,self.npdat) :
            for t in trecs :
                fname = self.netwkdir + '/record/' + self.filebase + '.evtlog.' + str(t) + '.' + str(r)
                if not (os.path.isfile(fname)) :
                    print ('file %s does not exist' % fname)
        
        # Read event data
        evtcount = 0
        evtlist = [[] for _ in range(len(vtxmap))]
        spike_times = []
        spike_neurons = []
        for t in trecs :
            for r in range(0,self.npdat) :
                # open file for reading
                fname = self.netwkdir + '/record/' + self.filebase + '.evtlog.' + str(t) + '.' + str(r)
                if (os.path.isfile(fname)) :
                    with open(fname, 'r') as frec:
                        for line in frec :
                            words = line.split()
                            evtype = int(words[0])
                            tstamp = (int(words[1], 16) // 100000)/10.0
                            idx = int(words[2])
                            # reindex
                            idx = int(vtxmap[idx])
                            # spikes only
                            if evtype == 0:
                                evtlist[idx].append(tstamp)
                                evtcount += 1
                                spike_times.append(tstamp)
                                spike_neurons.append(idx-1.0) # 0th neuron is the spike input
        if (self.format == 'eventlist') :
          return evtlist
        else: # 'dataframe' default format
          spikes_df = pd.DataFrame.from_dict({'time': spike_times, 'neuron_number': spike_neurons})
          return spikes_df.sort_values(by=['time'])

    def run(self, n_steps=10, return_potentials=False):
        """
        Runs the Simulator
        * runs circuit for n_steps then returns data
        * if not None raise error
        Returns:
        bool: True if ds format is required and converst neuron names to numbers and returns dictionary, False returns dataframe

        """
        self.t_max = n_steps

        simname = self.netwkdir + '/' + self.filebase + '.yml' 
        # Modify the simulation configuration
        with open(simname,"r") as file:
            simconfig = yaml.safe_load(file)
        simconfig['tmax'] = self.t_max
        with open(simname,"w") as file:
            yaml.dump(simconfig,file,sort_keys=False)
        
        runlist = self.runcmd.split()
        runlist.append('simulate')
        if (self.debug_mode):
            subprocess.run(runlist)
        else:
            subprocess.run(runlist, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        spikes_out = self._collect_output();
        
        return spikes_out

    def cleanup(self):
        # Deletes/frees neurons and synapses
        pass

    def reset(self):
        # resets time-step to 0 and resets neuron/synapse properties
        self._build_network()

    def set_properties(self, properties={}):
        """Set properties for specific neurons and synapses
        Args:
            properties: dictionary of properties for bricks
        """
        pass

    def set_input_spikes(self):
        """
        Get new initial spike times
        """
        pass
