# Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.

"""
isort:skip_file
"""

# fmt: off
import pandas as pd
import numpy as np

from .backend import Backend
from .lava_interfaces import Loihi2HWInterface, Loihi2SimInterface, calculateBitLength


def warnIfValueExceedsPrecision(value, precision, value_name):
    if value < precision: print(f"WARNING: {value_name} value exceeds available precision")


def warnIfFeatureNotAvailable(feature):
    # warns user that a feature is not supported but we'll still try to run anyways
    print(f"WARNING: {feature} is not supported by Lava") 


class FeatureNotAvailableException(Exception):
    pass


class NonIntegerDelayValueException(Exception):
    pass


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

    def __shape__(self):
        return self.shape()

    def shape(self):
        return (len(self.inputs), self.length)


class lava_Backend(Backend):
    def __init__(self):
        super(Backend, self).__init__()

    def _allocate(self, v, dv, vth, b, count=1):
        """ Lava LIF has single fixed {du, dv, vth} for entire population.
            We don't do anything with du, but every (dv, vth) combination requires
            a separate population (or "process" in Lava terminology). We also
            need to assemble a list of initial voltages for each population.
        """
        key = (dv, vth)
        if not key in self.process:
            self.process[key] = {'v':[], 'dv':dv, 'vth':vth, 'b':[], 'index':len(self.process), 'max_delay':0}
            self.anchor = key
        p_data = self.process[key]  # 'p_data' for 'process data'
        start = len(p_data['v'])
        p_data['v'].extend([v] * count)
        p_data['b'].extend([b] * count)
        p_data['count'] = len(p_data['v'])
        return p_data, start  # The caller can assume that exactly 'count' neurons were allocated.

    def _build_network(self):
        G = self.fugu_graph
        self.process = {}  # mapping from (dv,vth) tuples to dictionaries of LIF process configuration parameters
        self.inputIterator = InputIterator()
        self.inputIterator.length = self.duration

        # Special keys in graph node:
        # out_spikes -- list of times when this node spikes. Defined iff this is an input.
        # outputs -- Dictionary of output configurations. Key is name of variable to trace.
        #            Value is a dictionary of config values. Within the dictionary, 'o'
        #            comes from N2A fuguWrapper, and can support multiple output files.
        #            For in-graph recording, 'data' is a list containing the recorded data.
        #            For spikes, 'data' is a list of spike times. For V or other variables,
        #            'data' contains a trace of values for every time step in the simulation.
        #            Other sub-keys are reserved for use by the backend.
        # maxDelay -- Longest delay on any synapse going out from this node. Defined iff > 1.
        # neuronPD -- Process data for the main neuron.
        # neuronIndex -- Position of this neuron in process.
        # inputIndex -- Position of this neuron in 'inputIterator'. Defined iff this is an input node.
        # delayIndex -- Position of first neuron in 'delayProcess' associated with this node.
        #               Defined iff maxDelay > 1.

        # Analyze connections.
        # Find max weight over all synapses.
        #   Fugu's Loihi backend scales weight on a per-target-neuron basis, but it looks
        #   like Lava treats an entire weight matrix as having the same scaling, so we just
        #   determine a single scale for all weights in the network.

        print("Analyzing connections")
        maxWeight = 1
        delaySum  = 0
        for n1, n2, edge in G.edges.data():
            node = G.nodes[n1]
            delay = edge.get('delay', 1)
            if not isinstance(delay, int):
                raise NonIntegerDelayValueException()
            edge['delay'] = delay

            weight             = abs(edge.get('weight', 1.0))
            max_outgoing_delay = node.get('maxOutDelay', 1)
            if delay  > max_outgoing_delay:  node['maxOutDelay'] = delay
            if weight > maxWeight:           maxWeight        = weight

            delaySum += max(1, delay) 
        
        maxDelayValue = 63 # 63 is the largest possible delay value on Loihi

        # Replace edges with large delays with relay neurons
        print("Replacing edges with large delays with chains of relay neurons")
        edges_to_expand = []
        for n1, n2, edge in G.edges.data():
            delay = edge['delay']
            if delay > maxDelayValue:
                edges_to_expand.append((n1, n2))
        
        for n1, n2 in edges_to_expand:
            edge_data = G.edges[(n1, n2)]
            delay = edge_data['delay']
            weight = edge_data['weight']

            relay_count = 0
            current_node = n1
            next_node = f"{n1}-{n2}-relay-node:{relay_count}"
            while delay > maxDelayValue:
                G.add_node(
                            next_node,
                            threshold=0.5,
                            decay=1,
                            )
                G.add_edge(
                            current_node,
                            next_node,
                            weight=1.0,
                            delay=maxDelayValue,
                            )
                delay -= maxDelayValue
                current_node = next_node
                relay_count += 1
                next_node = f"{n1}-{n2}-relay-node:{relay_count}"
            
            G.add_edge(
                        current_node,
                        n2,
                        weight=weight,
                        delay=delay,
                        )
        G.remove_edges_from(edges_to_expand)

        # Find max bias and max threshold -- See above. A similar comment applies on how fine-grained bias can be.
        print("Find max bias")
        maxBias      = 0
        maxThreshold = 0
        for n, node in G.nodes.data():
            Vreset =     node.get('reset_voltage', 0.0)  # Offset from zero, since zero is always the reset voltage.
            Vspike = abs(node.get('threshold',     1.0) - Vreset)
            Vbias  = abs(node.get('bias',          0.0))
            if Vbias  > maxBias:      maxBias      = Vbias
            if Vspike > maxThreshold: maxThreshold = Vspike


        # Determine scale for voltage
        # See loihi_backend.py for reasoning behind this section.
        # All these calculations assume the quirks of Loihi-1.
        # May need something else for Loihi-2.
        print("Determining voltage scale")
        self.scale_factor = 1 << 20
        minScale   = 1 << 6

        #   Compensate for threshold
        if maxThreshold:
            bits = calculateBitLength(abs(maxThreshold * self.scale_factor)) # The power of the MSB needed to represent Vspike. The number of bits required is actually (bits+1).
            bit_limit = self.loihi2Interface.threshold_bit_limit 
            
            excess = bits - bit_limit
            if excess > 0:
                self.scale_factor >>= excess
                print(f"Scaling scale factor based on excess {excess}")
                warnIfValueExceedsPrecision(self.scale_factor, minScale, "Threshold")

        #   Compensate for weight magnitude
        #   As long as firing threshold can be represented, we don't care by how much it might
        #   be exceeded. Thus we don't worry about the sum of weights, only individual weights.
        print("Compensate for weight magnitude")
        if maxWeight:
            bits = calculateBitLength(abs(maxWeight * self.scale_factor))
            bit_limit = 20 # hightest possible bit position for weight
            
            excess = bits - bit_limit
            if excess > 0:
                self.scale_factor >>= excess
                print(f"Scaling scale factor based on excess {excess}")
                warnIfValueExceedsPrecision(self.scale_factor, minScale, "Weight")

        #   Compensate for large bias values
        print("Compensate for large bias")
        if maxBias:
            # bias is signed 13-bit (12 significant bits), with up to 7 bits of shift
            bits = calculateBitLength(abs(maxBias * self.scale_factor))
            biasExp = max(0, bits - 11)  # 11 is highest allowable power in mantissa
            bit_limit = biasExp
            
            excess = bits - bit_limit
            if excess > 0:
                self.scale_factor >>= excess
                print(f"Scaling scale factor based on excess {excess}")
                warnIfValueExceedsPrecision(self.scale_factor, minScale, "Bias")

        print(f"Scale factor: {self.scale_factor}")

        # Tag output neurons based on circuit information.
        print("Tag output neurons")
        for cn, vals in self.fugu_circuit.nodes.data():
            if self.record != 'all' and vals.get('layer') != 'output': continue
            if self.record == 'all':
                for n, node in G.nodes.data():
                    if not 'outputs' in node:          node['outputs'] = {}
                    if not 'spike' in node['outputs']: node['outputs']['spike'] = {}
            else:
                for list in vals['output_lists']:
                    for n in list:
                        node = G.nodes[n]
                        if not 'outputs' in node:          node['outputs'] = {}
                        if not 'spike' in node['outputs']: node['outputs']['spike'] = {}
        if self.traceV:
            for n, node in G.nodes.data():
                if 'outputs' in node:
                    if not 'V' in node['outputs']: node['outputs']['V'] = {}
                    if not 'I' in node['outputs']: node['outputs']['I'] = {}
                    #print(f"{n}, {node}, Adding V?")
                # 'V' will be removed from input neurons below.

        # Add input neurons, as identified by circuit information.
        print("Add input neurons")
        for cn, vals in self.fugu_circuit.nodes.data():
            if vals.get('layer') != 'input': continue
            # Stash all the spikes in their respective nodes.
            for timestep, neurons in enumerate(vals['brick']):
                for n in neurons:
                    node = G.nodes[n]
                    if 'out_spikes' not in node: node['out_spikes'] = []
                    spikes = node['out_spikes']
                    # Loihi-1 does not report spikes in cycle 0, so we don't see immediate effects of input in that cycle.
                    # The Fugu Loihi backend shifts timing later by 1 to compensate, then removes the shift in post-processing.
                    # Not sure how Lava will work on Loihi hardware. It may prove necessary to include that compensation here too.
                    # Unfortunately, if there are also spontaneously-firing neurons, it is impossible to align all the timing,
                    # so it is best to avoid the shift unless absolutely necessary.
                    spikes.append(timestep)
            # Set up InputIterator
            for list in vals['output_lists']:
                for n in list:
                    node = G.nodes[n]
                    #print(f"Input node: {n} {node}")
                    if 'outputs' in node and 'V' in node['outputs']:
                        # Reverse our own decision to add V, since it can't be traced on input neurons.
                        # User is responsible to avoid adding other things which can't be traced.
                        del node['outputs']['V']
                        del node['outputs']['I']
                        if not node['outputs']: del node['outputs']
                    if not 'out_spikes' in node: continue  # It's conceivable that some input neurons never issue a spike in some configurations.
                    node['inputIndex'] = len(self.inputIterator.inputs)
                    self.inputIterator.inputs.append(node['out_spikes'])
                    #print(f"Adding {node['out_spikes']} to ")

        # Add all other neurons ...
        print("Add neurons")
        Pwarning = False
        for n, node in G.nodes.data():
            # Set up regular neurons.
            if 'out_spikes' in node: continue  # This was created as an input node, so can't be a regular neuron.

            Vreset = node.get('reset_voltage', 0.0)  # Offset from zero, since zero is always the reset voltage.
            Vinit  = node.get('voltage',       0.0)
            Vspike = node.get('threshold',     1.0)
            Vdecay = node.get('decay',         0.0)
            Vbias  = node.get('bias',          0.0)
            P      = node.get('p',             1.0)
            if 'potential'        in node: Vinit  =       node['potential']
            if 'leakage_constant' in node: Vdecay = 1.0 - node['leakage_constant']

            Vinit  = int(self.scale_factor * (Vinit  - Vreset))
            Vspike = int(self.scale_factor * (Vspike - Vreset))
            Vbias  = int(self.scale_factor *  Vbias)
            Vdecay = int(Vdecay)

            if P != 1 and not Pwarning:
                warnIfFeatureNotAvailable("Probabalistic firing") # But we plow on anyway.
                Pwarning = True

            p_data, start = self._allocate(Vinit, Vdecay, Vspike, Vbias)
            node['neuronPD']    = p_data
            node['neuronIndex'] = start
            #print(f"Key for {n}: ({Vdecay}, {Vspike})")
            #print(f"\tp_data: {p_data}")
            #print(f"\tstart: {start}")

            # Add outputs to process config based on graph information.
            # TODO: encourage Lava to support probing of variable on a specific neuron.
            if 'outputs' in node:
                if 'outputs' not in p_data: p_data['outputs'] = {}
                p_datao = p_data['outputs']
                #print(f"Node outputs for {n}: {node['outputs']}")
                for v in node['outputs']:
                    if v not in p_datao: p_datao[v] = None
                #print(f"p_datao: {p_datao}")

        # Create Lava process objects
        print("Create lava process objects")
        output_processes = []
        for p_data in self.process.values():
            v           = p_data['v']
            dv          = p_data['dv']
            vth         = p_data['vth']
            b           = p_data['b']
            count       = len(v)
            index       = p_data['index']
            is_delay_PD = p_data.get('is_delay', False)

            process_name = f"lif{index}"

            p_data['process'] = process = self.loihi2Interface.get_lif_process(
                                                                    count=count,
                                                                    initial_voltages=v,
                                                                    spike_threshold=vth,
                                                                    decay_constant=dv,
                                                                    bias_mants=b,
                                                                    name=process_name,
                                                                    )
            p_data['process_name'] = process_name
            p_datao = p_data.get('outputs', {})
            #print(f"{process_name}: {p_datao}")
            for v in p_datao:
                if   v == 'spike': # and not self.traceV:
                    output_processes.append(p_data)
                elif v == 'V':
                    probe = self.loihi2Interface.setup_probe(process.v)
                    p_datao[v] = probe
                    if is_delay_PD:
                        self.probe_list[f"delay v {process_name}"] = probe
                    else:
                        self.probe_list[f"v {process_name}"] = probe
                elif v == 'I':
                    probe = self.loihi2Interface.setup_probe(process.u)
                    p_datao[v] = probe
                    self.probe_list[f"i {process_name}"] = probe

        print("Setting up input process")
        self.loihi2Interface.setup_input_process(self.inputIterator, self.scale_factor)
        print("Setting up output process")
        self.loihi2Interface.setup_output_process(output_processes)

        # Create weight matrices
        print("Create weight matrices")
        processSize = len(self.process)
        input_to_lif_weight_matrices = [None] * processSize
        input_to_lif_delay_matrices = [None] * processSize
        inputSize = len(self.inputIterator.inputs)
        for p_data in self.process.values():
            index = p_data['index']
            outputSize = len(p_data['v'])
            input_to_lif_weight_matrices[index] = np.zeros((outputSize, inputSize), dtype=int)
            input_to_lif_delay_matrices[index] = np.zeros((outputSize, inputSize), dtype=int)
        for p_data1 in self.process.values():
            size1 = len(p_data1['v'])
            p_data1['W'] = W = [None] * processSize
            p_data1['D'] = D = [None] * processSize
            for p_data2 in self.process.values():
                index2 = p_data2['index']
                size2 = len(p_data2['v'])
                W[index2] = np.zeros((size2, size1), dtype=int)
                D[index2] = np.zeros((size2, size1), dtype=int)

        # Add synapses
        print("Populate synapse data matrices")
        self.weightScale = self.scale_factor
        #print("source:SourcePD:source index, target:targetPD:target_index, scaled weight, actual weight, scaled delay, actual delay")
        for n1, n2, edge in G.edges.data():
            node1 = G.nodes[n1]
            node2 = G.nodes[n2]
            weight = edge.get('weight', 1) * self.weightScale

            delay  = edge.get('delay',  1)

            if delay > node2['neuronPD']['max_delay']:
                node2['neuronPD']['max_delay'] = delay

            processIndex = node2['neuronPD']['index']
            targetIndex  = node2['neuronIndex']
            if 'out_spikes' in node1:  # from input
                sourceIndex = node1['inputIndex']
                sourcePD = 'Input'
                W = input_to_lif_weight_matrices[processIndex]
                D = input_to_lif_delay_matrices[processIndex]
            else:  # from regular neuron
                sourceIndex = node1['neuronIndex']
                sourcePD = node1['neuronPD']['index']
                p_data1 = node1['neuronPD']
                W = p_data1['W'][processIndex]
                D = p_data1['D'][processIndex]
            W[targetIndex,sourceIndex] = round(weight)
            D[targetIndex,sourceIndex] = delay - 1
            #print(f"{n1}:{sourcePD}:{sourceIndex} {n2}:{node2['neuronPD']['index']}:{targetIndex} {W[targetIndex, sourceIndex]} {edge.get('weight', 1)} {D[targetIndex, sourceIndex]} {edge.get('delay', 1)}")


        # Connect processes
        print("Connecting input to processes")
        # Connect inputs
        for p_data2 in self.process.values():
            process2 = p_data2['process']
            self.loihi2Interface.connect_input_to_lif(
                                    input_weights=input_to_lif_weight_matrices,
                                    input_delays=input_to_lif_delay_matrices,
                                    lif_data=p_data2,
                                    lif_process=process2,
                                    )

        # Connect everything else
        print("Connecting everything else")
        for p_data1 in self.process.values():
            process1 = p_data1['process']
            for p_data2 in self.process.values():
                process2 = p_data2['process']
                self.loihi2Interface.connect_lif_to_lif(
                                        source_lif_data=p_data1,
                                        source_lif_process=process1,
                                        dest_lif_data=p_data2,
                                        dest_lif_process=process2,
                                        )

        print("Connecting lifs to passthrough layer (which connects to output)")
        for p_data in output_processes:
            process = p_data['process']
            weights = self.loihi2Interface.get_pass_through_weights(p_data, self.weightScale)
            #print(f"Pass through weights for: {p_data['process_name']}\n{weights}")
            self.loihi2Interface.connect_lif_to_output(weights, process)

    def compile(self, scaffold, compile_args={}):
        self.fugu_circuit    = scaffold.circuit
        self.fugu_graph      = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        self.record          = compile_args.get('record', False)
        self.recordInGraph   = 'recordInGraph' in compile_args
        self.lavaConfig      = compile_args.get('config', 'sim2')
        self.traceV          = compile_args.get('return_potentials', False)
        self.enableProfiler  = compile_args.get('enable_profiler', False)
        # Wait to build the network until run() because we need to know the value of return_potentials first.

    def run(self, n_steps=10, return_potentials=False):
        from lava.magma.core.run_conditions import RunSteps
        from lava.utils.profiler import Profiler

        self.duration = n_steps + 1

        self.hw_run = False
        if 'hw' in self.lavaConfig:
            self.hw_run = True

        self.probe_list = {}

        if self.lavaConfig == "hw2":
            self.loihi2Interface = Loihi2HWInterface(duration=self.duration)
        elif self.lavaConfig == "hw1":
            pass
        elif self.lavaConfig == "sim2":
            self.loihi2Interface = Loihi2SimInterface(duration=self.duration)
        else:  # sim1 and all others
            pass

        print("Building network")
        self._build_network()

        # @TODO: Get rid of this switch
        print("Setting up run config")
        if self.lavaConfig == "hw2":
            probe_list = [self.probe_list[tag] for tag in self.probe_list]
            if self.loihi2Interface.inputUProbe is not None:
                probe_list.append(self.loihi2Interface.inputUProbe)
            if self.loihi2Interface.inputVProbe is not None:
                probe_list.append(self.loihi2Interface.inputVProbe)
            if self.loihi2Interface.outputUProbe is not None:
                probe_list.append(self.loihi2Interface.outputUProbe)
            if self.loihi2Interface.outputVProbe is not None:
                probe_list.append(self.loihi2Interface.outputVProbe)
            runConfig = self.loihi2Interface.get_config(probe_list)
        elif self.lavaConfig == "hw1":
            from lava.magma.core.run_configs import Loihi1HwCfg
            runConfig = Loihi1HwCfg()
        elif self.lavaConfig == "sim2":
            runConfig = self.loihi2Interface.get_config([self.probe_list[tag] for tag in self.probe_list])
        else:  # sim1 and all others
            from lava.magma.core.run_configs import Loihi1SimCfg
            runConfig = Loihi1SimCfg()

        #print("------")
        #for process_key in self.process:
            #print(f"process key: {process_key}")
            #print(f"\t:{self.process[process_key]}")
            #print(f"\t:{self.process[process_key]['process'].du}")
            #print(f"\t:{self.process[process_key]['process'].dv}")
            #print(f"\t:{self.process[process_key]['process'].v}")
            #print(f"\t:{self.process[process_key]['process'].u}")
            #print(f"\t:{self.process[process_key]['process'].vth}")
            #print(f"\t:{self.process[process_key]['process'].bias_mant}")
            #print(f"\t:{self.process[process_key]['process'].bias_exp}")
        #print("------")

        process = self.process[self.anchor]['process']  # Presumably any LIF process can be used to run the graph. TODO: What if there are disconnected components?
        print("Running network")

        # Setup profiler if enabled:
        if self.enableProfiler:
            profiler = Profiler.init(runConfig)
            profiler.energy_probe(num_steps=self.duration)

        runCondition = RunSteps(num_steps=self.duration)
        process.run(condition=runCondition, run_cfg=runConfig)

        # collect outputs
        print("Processing output")
        for p_data in self.process.values():
            if not 'outputs' in p_data: continue
            p_datao = p_data['outputs']
            index = p_data['index']
            #print(f"index: {index}\noutputs: {p_data['outputs']}")
            for o, p in p_datao.items():  # dictionary from variable to monitor object
                if o == 'spike':
                    p_datao[o] = self.loihi2Interface.get_spike_output(p_data)
                elif not self.hw_run:
                    if o == 'V':     key = 'v'
                    elif o == 'I':     key = 'u'
                    else:              key = o
                    p_datao[o] = p.get_data()[f'lif{index}'][key]  # just the time series data
                #print("got", p_datao[o])
                #print(f"o: {o} \n{p_datao[o]}")

        if not self.hw_run:
            self.loihi2Interface.print_output_probe_data()

        #print(f"SPIKE OUTPUT: {self.loihi2Interface.get_spike_output()}")
        process.stop()

        if self.hw_run and len(self.probe_list) > 0:
            print("Hardware state probe results:")
            for tag in self.probe_list:
                probe = self.probe_list[tag]
                num_output_neurons = int(len(probe.time_series) / self.duration)
                print(f"Probe data for: {tag}")
                print(probe.time_series.reshape(num_output_neurons, self.duration))
            self.loihi2Interface.print_output_probe_data()

        # Process profiler data
        if self.enableProfiler:
            print(">>> Profiler data:")
            profiler.statement
            print()
            profiler.power_breakdown()
            print()
            profiler.energy_breakdown()
            print()
            print(f"Total execution time: {np.round(np.sum(profiler.execution_time), 6)} s")
            print(f"Total power: {np.round(profiler.power, 6)} W") 
            print(f"Total energy: {np.round(profiler.energy, 6)} J")
            print(f"Static energy: {np.round(profiler.static_energy, 6)} J")

        #print("Probe data for input process lif")
        #num_neurons = int(len(self.loihi2Interface.inputStateProbe.time_series) / self.duration)
        #print(self.loihi2Interface.inputStateProbe.time_series.reshape(num_neurons, self.duration))


        timeOffset = 0
        #if self.inputIterator.inputs and not self.hw_run: timeOffset = 1  # because SpikeDataloader does not step data until after spike phase.

        if self.recordInGraph:
            for n, node in self.fugu_graph.nodes.data():
                if not 'outputs' in node: continue
                if 'out_spikes' in node:  # This is an input that also got selected as an output.
                    # All this does is report back to the user exactly those spikes that were specified as input.
                    # This is mainly for convenience while visualizing the run.
                    vdict = node['outputs']['spike']
                    vdict['data'] = data = []
                    for s in node['out_spikes']: data.append(s+timeOffset)
                    continue
                neuronIndex = node['neuronIndex']
                p_datao = node['neuronPD']['outputs']
                for v, vdict in node['outputs'].items():
                    if v == 'spike':
                        vdict['data'] = data = []
                        for i, r in enumerate(p_datao[v]):  # rows (time steps) of dataset
                            if r[neuronIndex]: data.append(i)
                    elif v == 'V':
                        vdict['data'] = data = []
                        Vreset = node.get('reset_voltage', 0.0)
                        for r in p_datao[v]: data.append(r[neuronIndex] / self.scale_factor + Vreset)
                        #del data[0]  # TODO: what is the timing relationship between Monitor readouts and actual state?
                    elif v == 'I':
                        vdict['data'] = data = []
                        for r in p_datao[v]: data.append(r[neuronIndex] / self.scale_factor)
                        #del data[0]  # ditto
                    else:
                        vdict['data'] = data = []
                        for r in p_datao[v]: data.append(r[neuronIndex])
                        #del data[0]  # ditto
            return True

        spikeTimes = []
        spikeNeurons = []
        spikeNames = []
        potentialValues = []
        potentialNeurons = []
        for n, node in self.fugu_graph.nodes.data():
            if not 'outputs' in node: continue
            neuron_number = node['neuron_number']
            if 'out_spikes' in node:
                for s in node['out_spikes']:
                    spikeTimes  .append(s+timeOffset)
                    spikeNeurons.append(neuron_number)
                    spikeNames  .append(n)
                continue
            outputs = node['outputs']
            neuronIndex = node['neuronIndex']
            p_datao = node['neuronPD']['outputs']
            if 'spike' in outputs:
                for i, s in enumerate(p_datao['spike'][neuronIndex]):
                    if s:
                        spikeTimes  .append(i - 2)
                        spikeNeurons.append(neuron_number)
                        spikeNames  .append(n)
        spikes = pd.DataFrame({'time':spikeTimes, 'neuron_number':spikeNeurons, 'neuron_name':spikeNames}, copy=False)
        spikes.sort_values(by=['time', 'neuron_name'], inplace=True)  # put in spike time order
        #print(spikes)
        if not self.traceV or self.hw_run: return spikes
        potentials = pd.DataFrame({'neuron_number':potentialNeurons, 'potential':potentialValues}, copy=False)
        #print(potentials)
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
            if 'out_spikes' in node:
                del node['out_spikes']  # Allow list to be built from scratch.
        # When run() is called, network will be rebuilt with new spike times.
