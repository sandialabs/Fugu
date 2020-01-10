#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:18:44 2018

@author: smusuva
"""
from __future__ import print_function

from collections import Iterable, defaultdict
import pandas as pd
from ..neuron.neuron import Neuron, LIFNeuron
from ..synapse.synapse import Synapse


class NeuralNetwork:
    def __init__(self):
        # self.nrns = set()
        self.nrns = {}
        self.synps = {}
        self._nrn_count = 0

    def add_neuron(self, new_neuron=None):
        '''Add a neuron to the network. If a string is passed, a default LIFNeuron is created with that name.'''
        if not new_neuron:
            self._nrn_count += 1
            neuron = LIFNeuron(str(self._nrn_count))
        elif type(new_neuron) == str:
            self._nrn_count += 1
            neuron = LIFNeuron(new_neuron)
        elif isinstance(new_neuron, Neuron):
            self._nrn_count += 1
            neuron = new_neuron
        else:
            raise TypeError("{0} must be of type Neuron or str".format(new_neuron))
        self.nrns[neuron.name] = neuron

    def add_multiple_neurons(self, neuron_iterable=None):
        '''Add Neurons from an iterable such as a list'''
        if not neuron_iterable:
            self.add_neuron()

        if not isinstance(neuron_iterable, Iterable):
            raise TypeError("{0} is not iterable".format(neuron_iterable))

        for n in neuron_iterable:
            self.add_neuron(n)

    def list_neurons(self):
        print("Neurons: {", end='')
        for n in self.nrns:
            print("{},".format(self.nrns[n].name), end=" ")
        print("\b\b}")

    def add_synapse(self, new_synapse=None):
        '''Add synapse to a network. If a tuple is provided, a new Synapse object is created and added'''
        if not new_synapse:
            raise TypeError("Needs synapse object with pre and post neurons")
        elif type(new_synapse) == tuple and len(new_synapse) >= 2 and len(new_synapse) < 5:
            tmpsyn = Synapse(*new_synapse)
        elif type(new_synapse) == Synapse:
            tmpsyn = new_synapse
        elif isinstance(new_synapse, Synapse):
            tmpsyn = new_synapse
        else:
            raise TypeError("Must provide Synapse Object")

        if tmpsyn.get_key() not in self.synps:
            self.synps[tmpsyn.get_key()] = tmpsyn
            self.update_network(tmpsyn)
        else:
            print(
              "Warning! Not Added! "
              "{0} already defined in network. "
              "(Use <synapse>.set_params() to update synapse)".format(
                                                                 tmpsyn,
                                                                 )
              )

    def add_multiple_synapses(self, synapse_iterable=None):
        '''Add synapses from an iterable containing synapses'''
        if not isinstance(synapse_iterable, Iterable):
            raise TypeError("{0} is not iterable".format(synapse_iterable))

        for s in synapse_iterable:
            self.add_synapse(s)

    # NOT SURE IF THIS IS NEEDED! ###
    # def build_network(self):
    #    for s in self.synps:
    #        if s._post in self._inmap.keys():
    #            self._inmap[s._post].add(s)
    #        else:
    #            self._inmap[s._post] = {s}
    #

    def update_input_neuron(self, neuron_name, input_values):
        self.nrns[neuron_name].connect_to_input(input_values)

    # Will be called automatically if a synapse is added
    def update_network(self, new_synapse):
        '''build the connection map from the Synapses and Neuron information contained in them'''
        new_synapse._post.presyn.add(new_synapse)

    def step(self):
        '''Evolve the network over one time step'''
        for n in self.nrns:
            self.nrns[n].update_state()

        for s in self.synps:
            self.synps[s].update_state()

    def run(self, n_steps=1, debug_mode=False, record_potentials=False):
        '''Iterate the network evolution for n_steps number of times and return results as a pandas dataFrame'''

        tempdct = defaultdict(list)
        nrn_list = []
        for n in self.nrns:
            nrn_list.append(self.nrns[n].name)

        for t in range(0, n_steps):
            self.step()
            tempdct[t] = []
            for n in self.nrns:
                if self.nrns[n].spike and self.nrns[n].record:
                    if debug_mode:
                        tempdct[t].append((1, self.nrns[n].voltage))
                    else:
                        tempdct[t].append(1)
                else:
                    if debug_mode:
                        tempdct[t].append((0, self.nrns[n].voltage))
                    else:
                        tempdct[t].append(0)

        df = pd.DataFrame.from_dict(tempdct, orient='index', columns=nrn_list)
        df.columns.rename('Neurons', inplace=True)
        df.index.rename('Time', inplace=True)

        if not debug_mode:
            drop_list = [self.nrns[n].name for n in self.nrns if not self.nrns[n].record]
            df = df.drop(drop_list, axis=1)

        if record_potentials:
            final_potentials = pd.DataFrame({'potential': [], 'neuron_number': []})
            neuron_number = 0
            for neuron in self.nrns:
                final_potentials = final_potentials.append(
                                                            {
                                                              'potential': self.nrns[neuron].voltage,
                                                              'neuron_number': neuron_number,
                                                              },
                                                            ignore_index=True,
                                                            )
                neuron_number += 1
            return df, final_potentials
        return df
