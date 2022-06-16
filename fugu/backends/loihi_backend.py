import math
from .backend import Backend
import nxsdk.api.n2a as nx
import pandas as pd


class loihi_Backend(Backend):
    def __init__(self):
        super(Backend, self).__init__()
        self.nextCoreID = 0

    def setCoreID(self, p):
        key = (
            p.vMinExp,
            p.vMaxExp,
            p.noiseExpAtCompartment,
            p.noiseMantAtCompartment,
            p.noiseExpAtRefractoryDelay,
            p.noiseMantAtRefractoryDelay,
            p.randomizeVoltage,
            p.randomizeCurrent,
            p.compartmentCurrentDecay ==
            0,  # 4096 and 0 are incompatible. Force the extreme end into a separate compartment. Choice of which end based on opposite of default.
            p.compartmentVoltageDecay == 4096)
        if not key in self.cores:
            self.cores[key] = {
                'count': self.maxNeuronsPerCore
            }  # forces allocation of new core ID
        core = self.cores[key]
        if core['count'] < self.maxNeuronsPerCore:
            core['count'] += 1
        else:
            core['count'] = 1
            core['id'] = self.nextCoreID
            self.nextCoreID += 1
        p.logicalCoreId = core['id']

    def _build_network(self):
        self.net = nx.NxNet()
        self.cores = {}
        self.nextCoreID = 0
        G = self.fugu_graph

        # Determine if spike threshold can be shared, or if it varies between neurons.
        useDiscreteVTh = 0
        Vspike = None
        for n, node in G.nodes.data():
            if node.get(
                    'p', 1
            ) != 1:  # We know we will use different thresholds for probabilistic neurons.
                useDiscreteVTh = 1
                break
            temp = node.get('threshold', 1)
            if not Vspike:
                Vspike = temp
                continue
            if temp != Vspike:
                useDiscreteVTh = 1
                break

        """
        Analyze connections.
        Determine best delay buffer depth.
        """
        delayBuffer = 8
        avgDelay = 0
        for n1, n2, edge in G.edges.data():
            avgDelay += max(1, round(edge.get('delay', 1)))
            weight = abs(edge.get('weight', 1.0))
            if weight != 1.0:
                node = G.nodes[n2]  # The receiving side of the connection.
                maxWeight = node.get('maxWeight', 1.0)
                if weight > maxWeight: node['maxWeight'] = weight
        if avgDelay:
            avgDelay /= len(G.edges)
            delayBuffer = min(64, max(8, 1 << round(math.log2(avgDelay))))
        numDelayBits = round(math.log2(delayBuffer))
        self.maxNeuronsPerCore = min(self.maxNeuronsPerCore,
                                     round(8192 / delayBuffer))

        # Add input neurons, as identified by circuit information.
        for cn, vals in self.fugu_circuit.nodes.data():
            if vals.get('layer') != 'input': continue
            for timestep, neurons in enumerate(vals['brick']):
                for n in neurons:
                    node = G.nodes[n]
                    if '$pikes' not in node: node['$pikes'] = []
                    spikes = node['$pikes']
                    spikes.append(
                        timestep + 1
                    )  # Loihi does not report spikes in cycle 0, so we won't see immediate effects of input in that cycle. Shift all timing forward by 1 to compensate.
            for list in vals['output_lists']:
                for n in list:
                    node = G.nodes[n]
                    if not '$pikes' in node: continue
                    spikeGen = self.net.createSpikeGenProcess(1)
                    spikeGen.addSpikes(0, node['$pikes'])
                    node[
                        'cx'] = spikeGen  # nx neurons are stored directly in the graph

        # Add all other neurons ...

        """
        Note:
        * General note about fixed point: bit positions are zero-based and start from least-significant bit.
        * Position and power are different things. The value in a particular bit position may have
        * a power different than the position number. The difference is usually the exponent, an amount of shift
        * to position bits within the word. It is important to understand the distinction between
        * position, power, exponent and shift when reasoning about fixed-point numbers.

        * The main register for voltage is 24 bits wide. It is a signed value, so it has 23 significant bits.
        * The 23rd bit is at position 22 (zero-based). However, the maximum bit position for weight is 20
        * (MSB in position 7, plus max exponent 7, plus built-in shift of 6). To allow full range for weight,
        * we assign power 1 to bit position 20. This leaves 2 extra bits of headroom, so V can vary between (-8,8).
        * This is sufficient for summing any amount of weight, since threshold should not exceed (-1,1).
        """
        self.defaultScale = 1 << 20
        minScale = 1 << 6

        """
        The following are prototypes that are used repeatedly when building the network.
        Here we set common parameter values that don't change. Later, we set other
        parameters to more specific values.
        """

        # Regular neuron
        compProto = nx.CompartmentPrototype()
        compProto.numDendriticAccumulators = delayBuffer
        compProto.useDiscreteVTh = useDiscreteVTh

        # Regular connections
        connProto = nx.ConnectionPrototype(numDelayBits=numDelayBits)

        """
        Poisson spike generator for probabilistic neurons
        The configuration below tries to neutralize Loihi's quirky adjustments to the RNG
        by adding and upshifting so we get numbers in [0,255]. However, it turns out
        that the actual range is [1,256]. This is convenient, if a bit surprising,
        because it allows us to cover the full range of probability values exactly.
        """
        poissonCompProto = nx.CompartmentPrototype()
        poissonCompProto.useDiscreteVTh = 1  # TODO: check if all probabilities are the same
        poissonCompProto.functionalState = nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE  # Enable spontaneous spiking.
        poissonCompProto.compartmentVoltageDecay = 4096
        poissonCompProto.enableNoise = 1
        poissonCompProto.randomizeVoltage = 1
        poissonCompProto.noiseMantAtCompartment = 2  # Neutralizes the -128 bias in the RNG. See Loihi mathematical model white paper.
        poissonCompProto.noiseExpAtCompartment = 13  # +7 neutralizes the RNG down-shift. +6 up-shifts it to align with threshold.

        # Receives Poisson spikes and passes them on to the main compartment
        relayCompProto = nx.CompartmentPrototype()
        relayCompProto.vThMant = 100
        relayCompProto.stackOut = nx.COMPARTMENT_OUTPUT_MODE.PUSH

        # For connection from Poisson compartment to relay compartment
        relayConnProto = nx.ConnectionPrototype(weight=200)

        for n, node in G.nodes.data():
            if 'cx' in node:
                continue  # This was created as an input node, so can't be a regular neuron.

            Vreset = node.get(
                'reset_voltage', 0.0
            )  # Not an actual Loihi parameter. Rather, this is an offset from zero, since zero is always the reset voltage.
            Vinit = node.get('voltage', 0.0) - Vreset
            Vspike = node.get('threshold', 0.0) - Vreset
            Vdecay = node.get('decay', 0.0)
            P = node.get('p', 1.0)
            if 'potential' in node: Vinit = node['potential']
            if 'leakage_constant' in node:
                Vdecay = 1.0 - node['leakage_constant']

            # Determine scale for V
            scale = self.defaultScale

            #   Compensate for threshold
            if Vspike != 0:
                bits = math.floor(
                    math.log2(abs(Vspike * scale))
                )  # The power of the MSB needed to represent Vspike. The number of bits required is actually (bits+1).
                excess = bits - 22  # 22 is power of MSB of largest possible theshold. See notes on fixed-point above.
                if excess > 0:
                    scale >>= excess
                    node['scale'] = scale
                    if scale < minScale:
                        print(
                            "WARNING: threshold exceeds available precision: ",
                            n)

            """
            maxWeight
            * Compensate for weight magnitude
            As long as firing threshold can be represented, we don't care by how much it might
            be exceeded. Thus we don't worry about the sum of weights, only individual weights.
            """
            maxWeight = node.get('maxWeight', 1.0)
            if maxWeight > 1.0:
                bits = math.floor(math.log2(maxWeight * scale))
                excess = bits - 20  # 20 is highest possible bit position for weight
                if excess > 0:
                    scale >>= excess
                    node['scale'] = scale
                    if scale < minScale:
                        print(
                            "WARNING: at least one incoming weight exceeds available precision: ",
                            n)

            """
            bias 
            * Compensate for large bias values
            """
            bias = round(node.get('bias', 0.0) * scale)
            if bias:
                # bias is signed 13-bit (12 significant bits), with up to 7 bits of shift
                bit = math.floor(math.log2(abs(bias)))
                biasExp = max(0, bit -
                              11)  # 11 is highest allowable power in mantissa
                excess = biasExp - 7
                if excess > 0:
                    # Give up precision to support large bias.
                    scale >>= excess
                    bias >>= excess
                    biasExp = 7
                    node['scale'] = scale
                    if scale < minScale:
                        print("WARNING: bias exceeds available precision: ", n)
                if biasExp > 0: bias = bias >> biasExp
                """
                if biasExp > 0 
                * Set neuron to evaluate internal dynamics.
                * This option makes it act as if it just fired a spike.
                """
                # (How is this different than FIRED_LAST_TIME_STEP?)

                functionalState = nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE
            else:
                biasExp = 0
                functionalState = nx.COMPARTMENT_FUNCTIONAL_STATE.INACTIVE  # default start state

            Vinit = round(Vinit * scale)
            Vdecay = round(Vdecay * 4096)
            Vspike = round(Vspike * scale)
            vThMant = max(
                0, (Vspike >> 6) - 1
            )  # has fixed exponent of 2^6; Reduce by 1 to force equivalent of >= comparison.

            compProto.compartmentVoltage = Vinit  # nxnet doesn't do anything with this. With nxnet, we can't configure an initial voltage.
            compProto.biasMant = bias
            compProto.biasExp = biasExp
            compProto.vThMant = vThMant
            compProto.functionalState = functionalState
            compProto.compartmentVoltageDecay = Vdecay
            self.setCoreID(compProto)

            if P == 1:  # Deterministic firing
                compProto.compartmentJoinOperation = nx.COMPARTMENT_JOIN_OPERATION.SKIP  # default
                compProto.stackIn = 0  #nx.COMPARTMENT_INPUT_MODE.UNASSIGNED  # default
                node['cx'] = self.net.createCompartment(compProto)
            else:
                """
                Probabilistic firing
                Create an arrangement of 3 neurons which implements the defined Fugu behavior.
                The approach is to AND a spike from the main neuron with a Poisson spike from an
                auxiliary neuron. Unfortunately, the aux neuron must be on a separate
                core, so we must also create a relay neuron that participates in the
                multi-compartment neuron.
                Main output neuron is cx
                """

                # Main output neuron
                compProto.compartmentJoinOperation = nx.COMPARTMENT_JOIN_OPERATION.AND
                compProto.stackIn = nx.COMPARTMENT_INPUT_MODE.POP_A
                cx = self.net.createCompartment(compProto)
                node['cx'] = cx

                """
                Poisson spike generator
                The threshold is determined by the spike probability.
                """
                poissonCompProto.vThMant = round((1 - P) * 256)
                self.setCoreID(poissonCompProto)
                cxp = self.net.createCompartment(poissonCompProto)

                # Relay neuron
                self.setCoreID(relayCompProto)
                cxr = self.net.createCompartment(relayCompProto)
                cxr.logicalCoreId = cx.logicalCoreId
                cx.inputCompartmentId0 = cxr.nodeId
                cxp.connect(cxr, prototype=relayConnProto)


            if 'outputs' in node:
            """
            Add probes to neuron based on graph information.
            """
                for v, vdict in node['outputs'].items():
                    if '$pikes' in node:
                        continue  # This is an input, so can't be probed.
                    if v == 'spike':
                        vdict['probe'] = node['cx'].probe(
                            [nx.ProbeParameter.SPIKE])[0]
                    elif v == 'V':
                        vdict['probe'] = node['cx'].probe(
                            [nx.ProbeParameter.COMPARTMENT_VOLTAGE])[0]
                    elif v == 'I':
                        vdict['probe'] = node['cx'].probe(
                            [nx.ProbeParameter.COMPARTMENT_CURRENT])[0]

        """Add probes to output neurons based on circuit information."""
        self.outputs = []
        for node, vals in self.fugu_circuit.nodes.data():
            if vals.get('layer') != 'output': continue
            for l in vals['output_lists']:
                self.outputs.extend(l)
        for n in self.outputs:
            node = G.nodes[n]
            if '$pikes' in node:
                continue  # This is an input, so can't be probed.
            if not 'outputs' in node: node['outputs'] = {'spike': {}}
            vdict = node['outputs']['spike']
            if not 'probe' in vdict:
                vdict['probe'] = node['cx'].probe([nx.ProbeParameter.SPIKE])[0]
            if self.return_potentials:
                if not 'V' in node['outputs']: node['outputs']['V'] = {}
                vdict = node['outputs']['V']
                if not 'probe' in v:
                    vdict['probe'] = node['cx'].probe(
                        [nx.ProbeParameter.COMPARTMENT_VOLTAGE])[0]

        # Add synapses
        maxDelay = delayBuffer - 2
        delayCompProto = nx.CompartmentPrototype(
            vThMant=100, numDendriticAccumulators=delayBuffer)
        delayConnProto = nx.ConnectionPrototype(weight=200,
                                                delay=maxDelay,
                                                numDelayBits=numDelayBits)
        for n1, n2, edge in G.edges.data():
            neuron1 = G.nodes[n1]['cx']
            node2 = G.nodes[n2]
            neuron2 = node2['cx']

            scale = node2.get('scale', self.defaultScale)
            delay = max(
                0,
                round(edge.get('delay', 1)) - 1
            )  # Loihi always adds 1 cycle to requested delay, while Fugu expresses exact delay.
            weight = round(edge.get('weight', 1.0) * scale)

            # Determine weight exponent
            # 8 bits of available precision
            # There is a built-in up-shift by 2^6.
            weightExponent = 0
            if weight:
                bit = math.floor(math.log2(abs(weight)))
                weightExponent = max(
                    -8, bit - 13
                )  # Put MSB at position 13. This is position 7 for the weight mantissa, plus up-shift by 6 when accumulated weight is applied to voltage.
            if weightExponent >= 0: weight >>= weightExponent
            else: weight <<= -weightExponent
            weight >>= 6  # Offset the internal up-shift, so now MSB is at position 7.

            # Insert relay neurons for long delays
            while delay > maxDelay:
                self.setCoreID(delayCompProto)
                next = self.net.createCompartment(delayCompProto)
                neuron1.connect(next, prototype=delayConnProto)
                neuron1 = next
                delay -= maxDelay + 1  # Include one delay cycle for relay neuron.

            connProto.weight = weight
            connProto.weightExponent = weightExponent
            connProto.delay = delay
            # Default sign mode is MIXED.
            # By default, SDK compiler determines precision and compression mode.
            neuron1.connect(neuron2, prototype=connProto)

    def compile(self, scaffold, compile_args={}):
        self.fugu_circuit = scaffold.circuit
        self.fugu_graph = scaffold.graph
        self.brick_to_number = scaffold.brick_to_number
        self.recordInGraph = 'recordInGraph' in compile_args
        self.maxNeuronsPerCore = compile_args.get('maxNeuronsPerCore', 1024)
        # Wait to build the network until run() because we need to know the value of return_potentials first.

    def run(self, n_steps=10, return_potentials=False):
        self.return_potentials = return_potentials
        self._build_network()
        self.net.run(n_steps + 1)  # start counting at cycle 1
        self.net.disconnect()

        # collect outputs

        if self.recordInGraph:
            for n, node in self.fugu_graph.nodes.data():
                if not 'outputs' in node: continue
                if '$pikes' in node:
                    """
                    This is an input that also got selected as an output.
                    All this does is report back to the user exactly those spikes that were specified as input.
                    This is mainly for convenience while visualizing the run.
                    """
                    outputs = node['outputs']
                    if not 'spike' in outputs: outputs['spike'] = {}
                    vdict = outputs['spike']
                    data = []
                    vdict['data'] = data
                    for s in node['$pikes']:
                        data.append(s - 1)
                    continue
                for v, vdict in node['outputs'].items():
                    if v == 'spike':
                        data = []
                        vdict['data'] = data
                        for i, s in enumerate(vdict['probe'].data):
                            if s: data.append(i)
                    elif v == 'V':
                        data = []
                        vdict['data'] = data
                        scale = node.get('scale', self.defaultScale)
                        Vreset = node.get('reset_voltage', 0.0)
                        for V in vdict['probe'].data:
                            data.append(V / scale + Vreset)
                    elif v == 'I':
                        data = []
                        vdict['data'] = data
                        scale = node.get('scale', self.defaultScale)
                        for I in vdict['probe'].data:
                            data.append(I / scale)
                    else:
                        vdict['data'] = vdict['probe'].data
            return True

        spikeTimes = []
        spikeNeurons = []
        potentialValues = []
        potentialNeurons = []
        for n in self.outputs:
            node = self.fugu_graph.nodes[n]
            neuron_number = node['neuron_number']
            if '$pikes' in node:
                for s in node['$pikes']:
                    spikeTimes.append(s - 1)
                    spikeNeurons.append(neuron_number)
                continue
            outputs = node['outputs']
            for i, s in enumerate(outputs['spike']['probe'].data):
                if s:
                    spikeTimes.append(i)
                    spikeNeurons.append(neuron_number)
            if return_potentials:
                scale = node.get('scale', self.defaultScale)
                Vreset = node.get('reset_voltage', 0.0)
                potentialValues.append(outputs['V']['probe'].data[-1] / scale +
                                       Vreset)
                potentialNeurons.append(neuron_number)
        spikes = pd.DataFrame(
            {
                'time': spikeTimes,
                'neuron_number': spikeNeurons
            }, copy=False)
        spikes.sort_values('time', inplace=True)  # put in spike time order
        if not return_potentials: return spikes
        potential = pd.DataFrame(
            {
                'neuron_number': potentialNeurons,
                'potential': potentialValues
            },
            copy=False)
        return spikes, potentials

    def cleanup(self):
        # Deletes/frees neurons and synapses
        pass

    def reset(self):
        # resets time-step to 0 and resets neuron/synapse properties
        pass  # since run() must called anyway, there is nothing to do here

    def set_properties(self, properties={}):
        """
        Set properties for specific neurons and synapses
        Args:
            properties (dict): dictionary of properties for bricks
        """
        for brick in properties:
            if brick != 'compile_args':
                brick_id = self.brick_to_number[brick]
                self.fugu_circuit.nodes[brick_id]['brick'].set_properties(
                    properties[brick])
        # must call run() for changes to take effect

    def set_input_spikes(self):
        # Clean out old spike structures.
        for n, node in self.fugu_graph.nodes.data():
            if '$pikes' in node:
                del node['$pikes']  # Allow list to be built from scratch.
                del node[
                    'cx']  # In case neuron is no longer used, don't keep old compartment around.
        # When run() is called, network will be rebuilt with new spike times.
