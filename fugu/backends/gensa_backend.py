"""
isort:skip_file
"""

# fmt: off
import os
import subprocess
import pandas as pd

from .backend import Backend, PortDataIterator
from ..utils.OutputParser import OutputParser


class gensa_Backend(Backend):

    def writeGraphFile(self):
        G = self.fugu_graph
        self.out = open('model.graph', 'w')  # If we are called from fuguWrapper.py, this blows away the original graph file.

        # Tag output neurons based on circuit information.
        if self.record != 'all':
            for cn, vals in self.fugu_circuit.nodes.data():
                if vals.get('layer') != 'output': continue
                for n in PortDataIterator(vals):
                    node = G.nodes[n]
                    if not 'outputs' in node: node['outputs'] = {}
                    if not 'spike' in node['outputs']: node['outputs']['spike'] = {}
        if self.traceV:
            for n, node in G.nodes.data():
                if not 'outputs' in node: node['outputs'] = {}
                if not 'V' in node['outputs']: node['outputs']['V'] = {}
                # 'V' will be removed from input neurons below.

        # Add input neurons, as identified by circuit information.
        for cn, vals in self.fugu_circuit.nodes.data():
            if vals.get('layer') != 'input': continue
            # Distribute spike times to individual neurons. Store in G.
            for timestep, neurons in enumerate(vals['brick']):
                for n in neurons:
                    node = G.nodes[n]
                    if '$pikes' not in node: node['$pikes'] = []
                    spikes = node['$pikes']
                    spikes.append(timestep)
            # Write input neurons to file
            for n in PortDataIterator(vals):
                node  = G.nodes[n]
                index = node['neuron_number']
                self.out.write('{}\n'.format(index))
                if 'outputs' in node and 'V' in node['outputs']:
                    # can't trace V on input neurons
                    del node['outputs']['V']
                    if not node['outputs']: del node['outputs']
                self.writeSynapsesAndOutputs(G, n)
                # Embed input pattern
                if not '$pikes' in node: continue
                self.out.write(' t')  # spike time list
                between = False
                for t in node['$pikes']:
                    if between: self.out.write(',')
                    self.out.write('{}'.format(t))
                    between = True
                self.out.write('\n')

        # Add all other neurons.
        for n, node in G.nodes.data():
            if '$pikes' in node: continue  # skip input neurons, because they've already been done
            index  = node['neuron_number']
            Vinit  = node.get('voltage',       0.0)
            Vspike = node.get('threshold',     1.0)
            Vreset = node.get('reset_voltage', 0.0)
            decay  = node.get('decay',         0.0)
            P      = node.get('p',             1.0)
            if 'potential'        in node: Vinit =       node['potential']
            if 'leakage_constant' in node: decay = 1.0 - node['leakage_constant']
            self.out.write('{},{},{},{},{}\n'.format(index,Vspike,Vreset,decay,P))
            self.writeSynapsesAndOutputs(G, n)

        self.out.close()

    def writeSynapsesAndOutputs(self, G, n):
        for n1, n2, edge in G.out_edges(n, data=True):  # n1 is the same as n
            index  = G.nodes[n2]['neuron_number']
            weight =     edge.get('weight', 1.0)
            delay  = int(edge.get('delay',  1))
            self.out.write(' {},{},{}\n'.format(index,weight,delay))
        node = G.nodes[n]
        if not 'outputs' in node: return
        outputs = node['outputs']
        gotSpike = False
        for o, vals in outputs.items():
            self.out.write(' o\n')
            if o == 'spike':
                gotSpike = True  # since 'spike' is the default, we don't store it, but do remember that we handled it.
            else:
                self.out.write('  p{}\n'.format(o))  # record non-default probe
        if self.record == 'all' and not gotSpike:
            self.out.write(' o\n')
        # In all cases, we depend on the default column naming scheme in SST GNA: index.probe

    def compile(self, scaffold, compile_args={}):
        self.fugu_circuit    = scaffold.circuit
        self.fugu_graph      = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        self.record          = compile_args.get('record',        False)
        self.recordInGraph   = compile_args.get('recordInGraph', False)
        self.sst             = compile_args.get('sst',          'sst')
        self.gna             = compile_args.get('gna',          'gna.py')
        if not 'gna' in compile_args: print('WARNING: Path to gna.py should be provided in compile_args. This simulation will probably fail without it.')
        # Wait to write graph file until run(), because we need to know the value of return_potentials first.

    def run(self, n_steps=10, return_potentials=False):
        self.traceV = return_potentials
        self.writeGraphFile()
        subprocess.run([self.sst, self.gna, '--', '--steps', str(n_steps), '--dt', '1', '--neurons', 'model.graph'])  # call SST GNA
        # SST saves output in a file called 'out'.
        op = OutputParser()
        op.parse('out')  # Reads entire file into memory at once, then closes file.
        # If we are called from fuguWrapper.py, then the original 'out' will get overwritten.

        if self.recordInGraph:
            for n, node in self.fugu_graph.nodes.data():
                if not 'outputs' in node: continue
                outputs = node['outputs']
                if '$pikes' in node:  # This is an input that also got selected as an output.
                    # All this does is report back to the user exactly those spikes that were specified as input.
                    # This is mainly for convenience while visualizing the run.
                    if not 'spike' in outputs: continue
                    vdict = outputs['spike']
                    vdict['data'] = node['$pikes']
                    continue
                index = node['neuron_number']
                for v, vdict in node['outputs'].items():
                    if v == 'spike':
                        c = op.getColumn(str(index))
                        if not c: continue
                        vdict['data'] = data = []
                        for i, s in enumerate(c.values):
                            if s: data.append(i + c.startRow)
                    elif v == 'V':
                        c = op.getColumn('{}.V'.format(index))
                        vdict['data'] = c.values;
                        vdict['startRow'] = c.startRow
            return True

        spikeTimes = []
        spikeNeurons = []
        potentialValues = []
        potentialNeurons = []
        for cn, vals in self.fugu_circuit.nodes.data():
            if vals.get('layer') != 'output': continue
            for n in PortDataIterator(vals):
                node = self.fugu_graph.nodes[n]
                index = node['neuron_number']
                if '$pikes' in node:  # Neuron is both an input and an output.
                    # Simply copy the input pattern to the output.
                    for s in node['$pikes']:
                        spikeTimes  .append(s)
                        spikeNeurons.append(index)
                    continue
                # General case: transfer outputs from file
                outputs = node['outputs']
                c = op.getColumn(str(index))  # This column might not be present, if this neuron did not spike.
                if c:
                    for i, s in enumerate(c.values):
                        if s:
                            spikeTimes  .append(i + c.startRow)
                            spikeNeurons.append(index)
                if return_potentials:
                    c = op.getColumn('{}.V'.format(index))  # This column should always be present if return_potentials is true.
                    potentialValues .append(c.values[-1])
                    potentialNeurons.append(neuron_number)
        spikes = pd.DataFrame({'time':spikeTimes, 'neuron_number':spikeNeurons}, copy=False)
        spikes.sort_values('time', inplace=True)  # put in spike time order
        if not return_potentials: return spikes
        potential = pd.DataFrame({'neuron_number':potentialNeurons, 'potential':potentialValues}, copy=False)
        return spikes, potentials

    def cleanup(self):
        pass

    def reset(self):
        pass

    def set_properties(self, properties={}):
        for brick in properties:
            if brick != 'compile_args':
                brick_id = self.brick_to_number[brick]
                self.fugu_circuit.nodes[brick_id]['brick'].set_properties(properties[brick])
        # must call run() for changes to take effect

    def set_input_spikes(self):
        # Clean out old spike structures.
        for n, node in self.fugu_graph.nodes.data():
            if '$pikes' in node:
                del node['$pikes']  # Allow list to be built from scratch.
        # When run() is called, network will be rebuilt with new spike times.
