# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.


import math
import pandas as pd
import numpy as np

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.proc.io.dataloader import SpikeDataloader
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from .backend import Backend


class InputIterator:
    """ Supporting class for lava_Backend. Feeds the inputs to Dataloader process.
        Note that the output of this iterator is a tuple (input values, ground truth).
        Ground truth can be a scalar or vector. Since we don't care, we output the scalar 0.
    """

    def __init__(self):
        self.inputs = []  # map from position to spike list

    def __getitem__(self, i):
        result = np.zeros((len(self.inputs), 1))
        for j, n in enumerate(self.inputs):
            if i in n: result[j,0] = 1
        return result, 0

    def __len__(self):
        return self.length   # will be set by backend code below


class lava_Backend(Backend):
    def __init__(self):
        super(Backend, self).__init__()

    def _allocate(self, v, dv, vth, count=1):
        """ Lava LIF has single fixed {du, dv, vth} for entire population.
            We don't do anything with du, but every (dv, vth) combination requires
            a separate population (or "process" in Lava terminology). We also
            need to assemble a list of initial voltages for each population.
        """
        key = (dv, vth)
        if not key in self.process:
            self.process[key] = {'v':[], 'dv':dv, 'vth':vth, 'index':len(self.process)}
            self.anchor = key
        pd = self.process[key]  # 'pd' for 'process data'
        start = len(pd['v'])
        pd['v'].extend([v] * count)
        return pd, start  # The caller can assume that exactly 'count' neurons were allocated.

    def _build_network(self):
        G = self.fugu_graph
        self.process = {}  # mapping from (dv,vth) tuples to dictionaries of LIF process configuration parameters
        self.inputIterator = InputIterator()
        self.inputIterator.length = self.duration

        # Special keys in graph node:
        # $pikes -- list of times when this node spikes. Defined iff this is an input.
        # outputs -- Dictionary of output configurations. Key is name of variable to trace.
        #            Value is a dictionary of config values. Within the dictionary, 'o'
        #            comes from N2A fuguWrapper, and can support multiple output files.
        #            For in-graph recording, 'data' is a list containing the recorded data.
        #            For spikes, 'data' is a list of spike times. For V or other variables,
        #            'data' contains a trace of values for every time step in the simulation.
        #            Other sub-keys are reserved for use by the backend.
        # maxDelay -- Longest delay on any synapse going out from this node. Defined iff > 1.
        # neuronPD -- Process data for the main neuron.
        # neuronIndex -- Position of this neuron in 'neuronProcess'.
        # inputIndex -- Position of this neuron in 'inputIterator'. Defined iff this is an input node.
        # delayIndex -- Position of first neuron in 'delayProcess' associated with this node.
        #               Defined iff maxDelay > 1.

        # Analyze connections.
        # Find max delay depth for each neuron.
        # This will determine the chain of extra neurons we need to add to buffer a spike.
        for n1, n2, edge in G.edges.data():
            node = G.nodes[n1]
            delay = round(edge.get('delay', 1))
            maxDelay = node.get('maxDelay', 1)
            if delay > maxDelay: node['maxDelay'] = delay

        # Tag output neurons based on circuit information.
        if self.record != 'all':
            for cn, vals in self.fugu_circuit.nodes.data():
                if vals.get('layer') != 'output': continue
                for list in vals['output_lists']:
                    for n in list:
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
            # Stash all the spikes in their respective nodes.
            for timestep, neurons in enumerate(vals['brick']):
                for n in neurons:
                    node = G.nodes[n]
                    if '$pikes' not in node: node['$pikes'] = []
                    spikes = node['$pikes']
                    # Loihi-1 does not report spikes in cycle 0, so we don't see immediate effects of input in that cycle.
                    # Thus, we need to shift all timing forward by 1 to compensate.
                    # Presumably, Lava will primarily go to Loihi hardware, so assume the same limitation.
                    spikes.append(timestep + 1)
            # Set up InputIterator
            for list in vals['output_lists']:
                for n in list:
                    node = G.nodes[n]
                    if 'outputs' in node and 'V' in node['outputs']:
                        # Reverse our own decision to add V, since it can't be traced on input neurons.
                        # User is responsible to avoid adding other things which can't be traced.
                        del node['outputs']['V']
                        if not node['outputs']: del node['outputs']
                    if not '$pikes' in node: continue  # It's conceivable that some input neurons never issue a spike in some configurations.
                    node['inputIndex'] = len(self.inputIterator.inputs)
                    self.inputIterator.inputs.append(node['$pikes'])

        # Add all other neurons ...
        delayPD = None
        Pwarning = False
        for n, node in G.nodes.data():
            # Set up delay chains. (Applies to all neurons, including inputs.)
            maxDelay = node.get('maxDelay', 1)
            if maxDelay > 1:
                delayPD, start = self._allocate(0, 1, 1, maxDelay-1)
                node['delayIndex'] = start

            # Set up regular neurons.
            if '$pikes' in node: continue  # This was created as an input node, so can't be a regular neuron.

            Vreset = node.get('reset_voltage', 0.0)  # Offset from zero, since zero is always the reset voltage.
            Vinit  = node.get('voltage',       0.0) - Vreset
            Vspike = node.get('threshold',     1.0) - Vreset
            Vdecay = node.get('decay',         0.0)
            P      = node.get('p',             1.0)
            if 'potential'        in node: Vinit  =       node['potential']
            if 'leakage_constant' in node: Vdecay = 1.0 - node['leakage_constant']

            if P != 1 and not Pwarning:
                print('WARNING: Probabilistic firing not supported by Lava') # But we plow on anyway.
                Pwarning = True

            pd, start = self._allocate(Vinit, Vdecay, Vspike)
            node['neuronPD'] = pd
            node['neuronIndex'] = start

            # Add outputs to process config based on graph information.
            # TODO: encourage Lava to support probing of variable on a specific neuron.
            if 'outputs' in node:
                if 'outputs' not in pd: pd['outputs'] = {}
                pdo = pd['outputs']
                for v in node['outputs']:
                    if v not in pdo: pdo[v] = None

        # Create Lava process objects
        for pd in self.process.values():
            v   = pd['v']
            dv  = pd['dv']
            vth = pd['vth']
            count = len(v)
            index = pd['index']
            #print("process", index, vth, dv, v)
            pd['process'] = process = LIF(shape=(count,), v=v, vth=vth, du=1, dv=dv, name=f'lif{index}')
            pdo = pd.get('outputs', {})
            for v in pdo:
                if   v == 'spike':
                    pdo[v] = m = Monitor()
                    m.probe(process.s_out, self.duration)
                elif v == 'V':
                    pdo[v] = m = Monitor()
                    m.probe(process.v, self.duration)
                elif v == 'I':
                    pdo[v] = m = Monitor()
                    m.probe(process.u, self.duration)
        inputProcess = SpikeDataloader(dataset=self.inputIterator)

        # Create weight matrices
        processSize = len(self.process)
        self.inputIterator.W = [None] * processSize
        inputSize = len(self.inputIterator.inputs)
        for pd in self.process.values():
            index = pd['index']
            outputSize = len(pd['v'])
            self.inputIterator.W[index] = np.zeros((outputSize, inputSize))
        for pd1 in self.process.values():
            size1 = len(pd1['v'])
            pd1['W'] = W = [None] * processSize
            for pd2 in self.process.values():
                index2 = pd2['index']
                size2 = len(pd2['v'])
                W[index2] = np.zeros((size2, size1))

        # Connect delay chains
        if delayPD:  # For efficiency, stage some variables that will be used below. If delayPD is not defined, these values won't be used.
            # Notice that W connects the delay process to itself.
            delayIndex = delayPD['index']
            W = delayPD['W'][delayIndex]
        for n, node in G.nodes.data():
            maxDelay = node.get('maxDelay', 1)
            if maxDelay == 1: continue
            #print("have delay", n, maxDelay)
            start = node['delayIndex']
            if '$pikes' in node:  # from input
                sourceIndex = node['inputIndex']
                self.inputIterator.W[delayIndex][start,sourceIndex] = 1
            else:  # from regular neuron
                sourceIndex = node['neuronIndex']
                pd1 = node['neuronPD']
                pd1.W[delayIndex][start,sourceIndex] = 1
            if maxDelay == 2: continue
            for i in range(start, start+maxDelay-2): W[i+1,i] = 1

        # Add synapses
        for n1, n2, edge in G.edges.data():
            node1 = G.nodes[n1]
            node2 = G.nodes[n2]
            weight = edge.get('weight', 1)
            delay  = edge.get('delay',  1)
            processIndex = node2['neuronPD']['index']
            targetIndex  = node2['neuronIndex']
            if delay == 1:  # go directly from source neuron to target neuron
                if '$pikes' in node1:  # from input
                    sourceIndex = node1['inputIndex']
                    W = self.inputIterator.W[processIndex]
                else:  # from regular neuron
                    sourceIndex = node1['neuronIndex']
                    pd1 = node1['neuronPD']
                    W = pd1['W'][processIndex]
            else:  # delay > 1; go from delay neuron to target neuron
                sourceIndex = node1['delayIndex'] + delay - 2;
                W = delayPD['W'][processIndex]
            W[targetIndex,sourceIndex] = weight

        # Connect processes
        for pd2 in self.process.values():
            process2 = pd2['process']
            index2 = pd2['index']
            W = self.inputIterator.W[index2]
            if np.count_nonzero(W):
                #print("connecting inputs", process2.name, W.shape)
                c = Dense(weights=W)
                inputProcess.s_out.connect(c.s_in)
                c.a_out.connect(process2.a_in)
            self.inputIterator.W[index2] = None  # possibly release memory
        for pd1 in self.process.values():
            process1 = pd1['process']
            W1 = pd1['W']
            for pd2 in self.process.values():
                process2 = pd2['process']
                index2 = pd2['index']
                W = W1[index2]
                if np.count_nonzero(W):
                    #print("connecting", process1.name, process2.name, W.shape)
                    c = Dense(weights=W)
                    process1.s_out.connect(c.s_in)
                    c.a_out.connect(process2.a_in)
                W1[index2] = None

    def compile(self, scaffold, compile_args={}):
        self.fugu_circuit    = scaffold.circuit
        self.fugu_graph      = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        self.record          = compile_args.get('record', False)
        self.recordInGraph   = 'recordInGraph' in compile_args
        # Wait to build the network until run() because we need to know the value of return_potentials first.

    def run(self, n_steps=10, return_potentials=False):
        self.traceV = return_potentials
        self.duration = n_steps + 1
        self._build_network()

        runCondition = RunSteps(num_steps=self.duration)
        runConfig = Loihi1SimCfg()
        process = self.process[self.anchor]['process']  # Presumably any LIF process can be used to run the graph. TODO: What if there are disconnected components?
        process.run(condition=runCondition, run_cfg=runConfig)

        # collect outputs
        for pd in self.process.values():
            if not 'outputs' in pd: continue
            pdo = pd['outputs']
            index = pd['index']
            for o, m in pdo.items():  # dictionary from variable to monitor object
                if   o == 'spike': key = 's_out'
                elif o == 'V':     key = 'v'
                elif o == 'I':     key = 'u'
                else:              key = o
                pdo[o] = m.get_data()[f'lif{index}'][key]  # just the time series data
                #print("got", pdo[o])
        process.stop()

        if self.recordInGraph:
            for n, node in self.fugu_graph.nodes.data():
                if not 'outputs' in node: continue
                if '$pikes' in node:  # This is an input that also got selected as an output.
                    # All this does is report back to the user exactly those spikes that were specified as input.
                    # This is mainly for convenience while visualizing the run.
                    vdict = node['outputs']['spike']
                    vdict['data'] = data = []
                    for s in node['$pikes']: data.append(s-1)
                    continue
                neuronIndex = node['neuronIndex']
                pdo = node['neuronPD']['outputs']
                for v, vdict in node['outputs'].items():
                    if v == 'spike':
                        vdict['data'] = data = []
                        for i, r in enumerate(pdo[v]):  # rows (time steps) of dataset
                            if r[neuronIndex]: data.append(i-1)
                    elif v == 'V':
                        vdict['data'] = data = []
                        Vreset = node.get('reset_voltage', 0.0)
                        for r in pdo[v]: data.append(r[neuronIndex] + Vreset)
                        #del data[0]  # TODO: what is the timing relationship between Monitor readouts and actual state?
                    else:
                        vdict['data'] = data = []
                        for r in pdo[v]: data.append(r[neuronIndex])
                        #del data[0]  # ditto
            return True

        spikeTimes = []
        spikeNeurons = []
        potentialValues = []
        potentialNeurons = []
        for n, node in self.fugu_graph.nodes.data():
            if not 'outputs' in node: continue
            neuron_number = node['neuron_number']
            if '$pikes' in node:
                for s in node['$pikes']:
                    spikeTimes  .append(s-1)
                    spikeNeurons.append(neuron_number)
                continue
            outputs = node['outputs']
            neuronIndex = node['neuronIndex']
            pdo = node['neuronPD']['outputs']
            if 'spike' in outputs:
                for i, r in enumerate(pdo['spike']):
                    if r[neuronIndex]:
                        spikeTimes  .append(i-1)
                        spikeNeurons.append(neuron_number)
            if return_potentials:
                Vreset = node.get('reset_voltage', 0.0)
                potentialValues .append(pdo['V'][-1][neuronIndex] + Vreset)
                potentialNeurons.append(neuron_number)
        spikes = pd.DataFrame({'time':spikeTimes, 'neuron_number':spikeNeurons}, copy=False)
        spikes.sort_values('time', inplace=True)  # put in spike time order
        if not return_potentials: return spikes
        potential = pd.DataFrame({'neuron_number':potentialNeurons, 'potential':potentialValues}, copy=False)
        return spikes, potentials

    def cleanup(self):
        pass

    def reset(self):
        pass  # since run() must called anyway, there is nothing to do here

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
