#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
isort:skip_file
"""

# fmt: off
from .backend import Backend
import sys


class spinnaker2_Backend(Backend):

    def _build_network(self):
        from spinnaker2 import snn, hardware

        C = self.fugu_circuit
        G = self.fugu_graph
        self.network = snn.Network()

        # Add input neurons, as identified by circuit information.
        spikeTimes = {}
        for _, vals in C.nodes.data():
            if vals.get('layer') != 'input': continue
            for timestep, neurons in enumerate(vals['brick']):  # A brick reports its spikes via the enumeration interface. A brick can also contain other properties.
                for n in neurons:
                    node = G.nodes[n]
                    if not 'spikes' in node: node['spikes'] = []
                    node['spikes'].append(timestep)
            for neurons in vals['output_lists']:
                for n in neurons:
                    node = G.nodes[n]
                    if not 'spikes' in node: node['spikes'] = []
                    index = len(spikeTimes)  # For the purpose of making connections, every neuron needs an integer index within its population.
                    node['index'] = index
                    spikeTimes[index] = node['spikes']  # Could be empty, but also could be updated to contain spikes later.
        source = snn.Population(len(spikeTimes), 'spike_list', spikeTimes)
        self.network.add(source)

        # Tag output neurons based on circuit information.
        for _, vals in C.nodes.data():
            if vals.get('layer') != 'output': continue
            for neurons in vals['output_lists']:
                for n in neurons:
                    node = G.nodes[n]
                    if 'spikes' in node: continue  # This is an input, so no need to record.
                    if not 'outputs' in node: node['outputs'] = {'spike': {}}

        # Determine voltage scaling
        # Synaptic weights are integers in [-15,15] and we want to use all available precision.
        # Since neurons are coupled only by spike events, they can have independent scales.
        self.defaultScale = 15
        for n1, n2, edge in G.edges.data():
            weight = abs(edge.get('weight', 1.0))
            if weight != 1.0:
                node = G.nodes[n2]  # The receiving side of the connection.
                maxWeight = node.get('maxWeight', 1.0)
                if weight > maxWeight: node['maxWeight'] = weight

        # Add all other neurons.
        #Vinits   = []
        Vspikes  = []
        Vresets  = []
        Vretains = []
        Vbiases  = []
        #Ps       = []
        self.recordSpikes   = False
        self.recordVoltages = False
        for n, node in G.nodes.data():
            if 'spikes' in node: continue  # Input, so don't treat as regular neuron.

            node['index'] = len(Vspikes)  # Unlike neuron_number, this is our position within a specific population.
            scale = self.defaultScale / node.get('maxWeight', 1.0)
            if scale != self.defaultScale: node['scale'] = scale
            outputs = node.get('outputs', {})
            if 'spike' in outputs: self.recordSpikes   = True
            if 'V'     in outputs: self.recordVoltages = True

            Vinit   =       node.get('voltage',       0.0)
            Vspike  =       node.get('threshold',     1.0)
            Vreset  =       node.get('reset_voltage', 0.0)
            Vretain = 1.0 - node.get('decay',         0.0)
            Vbias   =       node.get('bias',          0.0)
            P       =       node.get('p',             1.0)
            if 'potential'        in node: Vinit   = node['potential']
            if 'leakage_constant' in node: Vretain = node['leakage_constant']

            #Vinits  .append (Vinit * scale)
            Vspikes .append (Vspike * scale)
            Vresets .append (Vreset * scale)
            Vretains.append (Vretain)
            Vbiases .append (Vbias * scale)
            #Ps      .append (P)

        params = {'threshold':   Vspikes,
                  'alpha_decay': Vretains,
                  'i_offset':    Vbiases,
                  'v_reset':     Vresets,
                  'reset':       'reset_to_v_reset'}
        record = []
        if self.recordSpikes:   record.append('spikes')
        if self.recordVoltages: record.append('v')
        self.lif = lif = snn.Population(len(Vspikes), 'lif', params, record=record)  # TODO: use spike sink instead of recording
        self.network.add(lif)

        # TODO: create a spike sink population

        # Create synapses
        maxDelay = 8
        relay = snn.Population(0, 'lif', {'alpha_decay': 0})
        connections = {(source, lif):   [],
                       (source, relay): [],
                       (lif, lif):      [],
                       (lif, relay):    [],
                       (relay, lif):    [],
                       (relay, relay):  []}
        for n1, n2, edge in G.edges.data():
            delay  = round(edge.get('delay',  1))
            weight =       edge.get('weight', 1.0)
            node1 = G.nodes[n1]
            node2 = G.nodes[n2]
            i1 = node1['index']
            i2 = node2['index']
            scale = node2.get('scale', self.defaultScale)
            weight *= scale

            pre = source if 'spikes' in node1 else lif
            while delay > maxDelay:
                # allocate a relay neuron
                i3 = relay.size
                relay.size += 1
                # make a connection to it
                s = [i1, i3, 1, maxDelay-1]  # s2 always adds 1 to delay value
                connections[(pre, relay)].append(s)
                # update source neuron and remaining delay
                pre = relay
                i1 = i3
                delay -= maxDelay

            s = [i1, i2, weight, delay-1]
            connections[(pre, lif)].append(s)

        if relay.size: self.network.add(relay)
        for pre in (source, lif, relay):
            for post in (lif, relay):
                c = connections[(pre, post)]
                if c: self.network.add(snn.Projection(pre, post, c))

    def compile(self, scaffold, compile_args={}):
        # creates neuron populations and synapses
        self.fugu_circuit    = scaffold.circuit
        self.fugu_graph      = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        self.recordInGraph   = 'recordInGraph' in compile_args
        self._build_network()

    def run(self, n_steps=10, return_potentials=False):
        from spinnaker2 import snn, hardware

        if return_potentials and not self.recordVoltages:
            self.recordVoltages = True
            self.lif.record.append('v')
        hw = hardware.SpiNNaker2Chip(eth_ip='192.168.1.52')  # TODO: make IP address a parameter
        hw.run(self.network, n_steps)

        # collect outputs
        G = self.fugu_graph
        if self.recordSpikes:   spikes   = self.lif.get_spikes()
        if self.recordVoltages: voltages = self.lif.get_voltages()
        if self.recordInGraph:
            for n, node in G.nodes.data():
                if not 'outputs' in node: continue
                index = node['index']
                for v, vdict in node['outputs'].items():
                    if v == 'spike':
                        if 'spikes' in node:  # This is an input that also got selected as an output.
                            # All this does is report back to the user exactly those spikes that were specified as input.
                            # This is mainly for convenience while visualizing the run.
                            vdict['data'] = node['spikes']
                        else:
                            vdict['data'] = spikes[index]
                    elif v == 'V':
                        vdict['data'] = data = []
                        scale = node.get('scale', self.defaultScale)
                        for V in voltages[index]: data.append(V / scale)
            return True

        spikeTimes = []
        spikeNeurons = []
        potentialValues = []
        potentialNeurons = []
        for _, vals in self.fugu_circuit.nodes.data():
            if vals.get('layer') != 'output': continue
            for neurons in vals['output_lists']:
                for n in neurons:
                    node = G.nodes[n]
                    neuron_number = node['neuron_number']
                    index         = node['index']
                    if 'spikes' in node:
                        for s in node['spikes']:
                            spikeTimes  .append(s)
                            spikeNeurons.append(neuron_number)
                        continue
                    if 'spike' in node['outputs']:
                        for s in spikes[index]:
                            spikeTimes  .append(s)
                            spikeNeurons.append(neuron_number)
                    if return_potentials:
                        scale = node.get('scale', self.defaultScale)
                        potentialValues .append(voltages[index][-1] / scale)
                        potentialNeurons.append(neuron_number)
        spikes = pd.DataFrame({'time': spikeTimes, 'neuron_number': spikeNeurons}, copy=False)
        spikes.sort_values('time', inplace=True)  # put in spike time order
        if not return_potentials: return spikes
        potential = pd.DataFrame({'neuron_number': potentialNeurons, 'potential': potentialValues}, copy=False)
        return spikes, potentials

    def cleanup(self):
        # Deletes/frees neurons and synapses
        pass

    def reset(self):
        # resets time-step to 0 and resets neuron/synapse properties
        self._build_network()

    def set_properties(self, properties={}):
        for brick in properties:
            if brick == 'compile_args': continue
            brick_id = self.brick_to_number[brick]
            changes = self.fugu_circuit.nodes[brick_id]['brick'].set_properties(properties[brick])
        self._build_network()

    def set_input_spikes(self):
        # Clean out old spike structures.
        for n, node in self.fugu_graph.nodes.data():
            if 'spikes' in node: del node['spikes']  # Allow list to be built from scratch.
        self._build_network()
