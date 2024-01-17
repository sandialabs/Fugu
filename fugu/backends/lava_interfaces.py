import abc
import sys

from abc import abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {'__slots__': ()})

import math
import numpy as np


def calculateBitLength(value):
    return math.floor(math.log2(abs(value)))


def calcWeightExponent(weights):
    largest_weight_magnitude = abs(weights).max()
    bit = calculateBitLength(largest_weight_magnitude)
    weightExponent = max(-8, bit - 13)  # Put MSB at position 13. This is position 7 for the weight mantissa, plus up-shift by 6 when accumulated weight is applied to voltage.
    return weightExponent


def calcHardwareWeights(weights, weightExponent):
    if weightExponent >= 0: scaled_weights = (weights / (2 ** weightExponent)) / 64
    else:                   scaled_weights = (weights * (2 ** (-weightExponent))) / 64
    integer_weights = scaled_weights.astype('int')
    return integer_weights


class LoihiInterface(ABC):
    def __init__(self, duration):
        self.threshold_bit_limit = 22 # 22 is power of MSB of largest possible theshold BUT hardware scales up threshold values by 6 bits. See notes on fixed-point in Loihi_backend.
        self.duration = duration

        from lava.proc.lif.process import LIF
        self.LIF = LIF
        from lava.proc.sparse.process import Sparse, DelaySparse
        self.Sparse = Sparse
        self.DelaySparse = DelaySparse
        from lava.proc.dense.process import Dense, DelayDense
        self.Dense = Dense
        self.DelayDense = DelayDense
        from lava.proc.io.sink import RingBuffer as Sink
        self.Sink = Sink

        from lava.utils.weightutils import SignMode
        self.SignMode = SignMode

        from scipy.sparse import csr_matrix
        self.csr_matrix = csr_matrix

    @abstractmethod
    def setup_input_process(self, input_iterator, scale_factor):
        pass

    @abstractmethod
    def setup_output_process(self, output_processes):
        pass

    @abstractmethod
    def get_config(self, callback_functions=[]):
        pass

    @abstractmethod
    def get_lif_process(self, count, initial_voltages, spike_threshold, decay_constant, bias_mants, name):
        pass

    @abstractmethod
    def connect_input_to_lif(self, input_weights, input_delays, input_process, lif_data, lif_process):
        pass

    @abstractmethod
    def connect_lif_to_output(self, output_weights, lif_process):
        pass

    @abstractmethod
    def connect_lif_to_lif(self, source_lif_data, source_lif_process, dest_lif_data, dest_lif_process):
        pass

    @abstractmethod
    def setup_probe(self, var):
        pass

    @abstractmethod
    def get_spike_output(self, process_data):
        pass

    @abstractmethod
    def print_output_probe_data(self):
        pass


class Loihi2SimInterface(LoihiInterface):

    def __init__(self, duration):
        super().__init__(duration)
        from lava.proc.monitor.process import Monitor
        self.Monitor = Monitor
        from lava.proc.io.dataloader import SpikeDataloader
        self.SpikeDataloader = SpikeDataloader

        from lava.proc.lif.models import PyLifModelBitAcc 
        from lava.proc.sparse.models import PySparseModelBitAcc, PyDelaySparseModelBitAcc
        self.use_bit_acc = False 
        self.proc_model_map = {}
        if self.use_bit_acc:
            self.proc_model_map[self.LIF] = PyLifModelBitAcc
            self.proc_model_map[self.Sparse] = PySparseModelBitAcc
            self.proc_model_map[self.DelaySparse] = PyDelaySparseModelBitAcc
            self.threshold_bit_limit = 16

        self.process_output_map = {}

        self.current_decay_scale_factor = 4095 if self.use_bit_acc else 1
        self.voltage_decay_scale_factor = 4096 if self.use_bit_acc else 1 

    def setup_input_process(self, input_iterator, scale_factor):
        self.input_process = self.SpikeDataloader(dataset=input_iterator)

    def setup_output_process(self, output_processes):
        count = 0
        for process in output_processes:
            self.process_output_map[process['index']] = count
            count += process['count']

        self.pass_through_lif = self.LIF(shape=(count,), v=0, u=0, vth=1, du=self.current_decay_scale_factor, dv=self.voltage_decay_scale_factor, bias_mant=0, name=f"lif_passthrough")
        self.sink_output = self.Sink(shape=(count,), buffer=self.duration)

        self.pass_through_lif.s_out.connect(self.sink_output.a_in)
        self.outputSMonitor = self.Monitor()
        self.outputUMonitor = self.Monitor()
        self.outputVMonitor = self.Monitor()
        self.outputSMonitor.probe(self.pass_through_lif.s_out, self.duration)
        self.outputUMonitor.probe(self.pass_through_lif.u, self.duration)
        self.outputVMonitor.probe(self.pass_through_lif.v, self.duration)

    def get_config(self, callback_functions=[]):
        from lava.magma.core.run_configs import Loihi2SimCfg
        return Loihi2SimCfg(exception_proc_model_map=self.proc_model_map)

    def get_lif_process(self, count, initial_voltages, spike_threshold, decay_constant, bias_mants, name):
        return self.LIF(
                    shape=(count,),
                    v=initial_voltages,
                    u=0,
                    vth=spike_threshold >> 6 if self.use_bit_acc else spike_threshold, #This is annoying sin
                    du=self.current_decay_scale_factor,
                    dv=int(decay_constant * self.voltage_decay_scale_factor),
                    bias_mant=bias_mants,
                    name=name,
                    )

    def connect_input_to_lif(self, input_weights, input_delays, lif_data, lif_process):
        lif_index = lif_data['index']
        weights = input_weights[lif_index].astype('int')
        delays = input_delays[lif_index].astype('int')
        if np.count_nonzero(weights):
            if self.use_bit_acc:
                exponent = calcWeightExponent(weights)
                weights = calcHardwareWeights(weights, exponent)
                c = self.DelaySparse(weights=self.csr_matrix(weights), delays=self.csr_matrix(delays), weight_exp=exponent, sign_mode=self.SignMode.MIXED)
            else:
                c = self.DelaySparse(weights=self.csr_matrix(weights), delays=self.csr_matrix(delays), sign_mode=self.SignMode.MIXED)
            self.input_process.s_out.connect(c.s_in)
            c.a_out.connect(lif_process.a_in)
        input_weights[lif_index] = None

    def connect_lif_to_output(self, output_weights, lif_process):
        if np.count_nonzero(output_weights):
            if self.use_bit_acc:
                exponent = calcWeightExponent(output_weights)
                output_weights = calcHardwareWeights(output_weights, exponent)
                dummy_connection = self.Sparse(weights=self.csr_matrix(output_weights), sign_mode=self.SignMode.EXCITATORY, weight_exp=exponent)
            else:
                dummy_connection = self.Sparse(weights=self.csr_matrix(output_weights), sign_mode=self.SignMode.EXCITATORY)
            lif_process.s_out.connect(dummy_connection.s_in)
            dummy_connection.a_out.connect(self.pass_through_lif.a_in)

    def connect_lif_to_lif(self, source_lif_data, source_lif_process, dest_lif_data, dest_lif_process):
        dest_lif_index = dest_lif_data['index']
        W = source_lif_data['W'][dest_lif_index].astype('int')
        D = source_lif_data['D'][dest_lif_index].astype('int')
        if np.count_nonzero(W):
            if self.use_bit_acc:
                exponent = calcWeightExponent(W)
                W = calcHardwareWeights(W, exponent)
                c = self.DelaySparse(weights=self.csr_matrix(W), delays=self.csr_matrix(D), weight_exp=exponent, sign_mode=self.SignMode.MIXED)
            else:
                c = self.DelaySparse(weights=self.csr_matrix(W), delays=self.csr_matrix(D), sign_mode=self.SignMode.MIXED)
            source_lif_process.s_out.connect(c.s_in)
            c.a_out.connect(dest_lif_process.a_in)
        source_lif_data['W'][dest_lif_index] = None

    def get_pass_through_weights(self, lif_data, weight_scale_factor):
        W = np.eye(
                N=self.pass_through_lif.proc_params['shape'][0],
                M=lif_data['count'],
                k=-self.process_output_map[lif_data['index']],
                )
        W = W * weight_scale_factor 
        return W

    def setup_probe(self, var):
        m = self.Monitor()
        m.probe(var, self.duration)
        return m

    def get_spike_output(self, process_data):
        process_index = process_data['index']
        process_count = process_data['count']
        start = self.process_output_map[process_index]
        spike_output = self.sink_output.data.get()
        spike_output = spike_output[start:start + process_count]
        return spike_output

    def print_output_probe_data(self):
        print("~~~~~ Output probe data:")
        print(f"Spike: {self.outputSMonitor.get_data()}")
        print(f"U: {self.outputUMonitor.get_data()}")
        print(f"V: {self.outputVMonitor.get_data()}")


class Loihi2HWInterface(LoihiInterface):


    def __init__(self, duration):
        super().__init__(duration)
        from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
        self.PyToNxAdapter = PyToNxAdapter
        self.NxToPyAdapter = NxToPyAdapter
        from lava.proc.io.source import RingBuffer as SpikeGenerator
        self.SpikeGenerator = SpikeGenerator

        from lava.utils.loihi2_state_probes import StateProbe
        self.StateProbe = StateProbe

        # For some reason, threshold is scaled up by 6 bits (so by a value of 64) when it gets put onto hardware so we have fewer bits to work with. Otherwise we will run into overflow/garbage values.
        self.threshold_bit_limit = 16 

        from lava.proc.lif.ncmodels import NcL2ModelLif
        self.proc_model_map = {}
        self.proc_model_map[self.LIF] = NcL2ModelLif

        self.process_output_map = {}

        self.current_decay_scale_factor = 4095
        self.voltage_decay_scale_factor = 4096

        self.use_sparse = False
        self.connection_model = self.Sparse if self.use_sparse else self.Dense
        self.delay_connection_model = self.DelaySparse if self.use_sparse else self.DelayDense
    
    def format_matrix(self, m):
        if self.use_sparse:
            return self.csr_matrix(m)
        else:
            return m

    def setup_input_process(self, input_iterator, scale_factor):
        input_count = input_iterator.shape()[0]
        spike_trains = np.zeros(shape=input_iterator.shape(), dtype=int)
        for index, step in enumerate(input_iterator.inputs):
            spike_trains[index][step] = 1

        sg_input = self.SpikeGenerator(data=spike_trains)
        py_nx_adapt = self.PyToNxAdapter(shape=(input_count,))

        self.input_process= self.LIF(
                shape=(input_count,),
                v=0,
                u=0,
                vth=0,
                du=self.current_decay_scale_factor,
                dv=self.voltage_decay_scale_factor,
                bias_mant=0,
                name=f"lif_input",
                )
        weights = np.eye(input_count) * scale_factor
        exponent = calcWeightExponent(weights)
        weights = calcHardwareWeights(weights, exponent)

        formatted_weights = self.format_matrix(weights)

        dummy_connection = self.connection_model(weights=formatted_weights, sign_mode=self.SignMode.EXCITATORY)

        sg_input.s_out.connect(py_nx_adapt.inp)
        py_nx_adapt.out.connect(dummy_connection.s_in)
        dummy_connection.a_out.connect(self.input_process.a_in)
        self.inputUProbe = None
        self.inputVProbe = None
        #self.inputUProbe = StateProbe(self.input_process.u)
        #self.inputVProbe = StateProbe(self.input_process.v)

    def setup_output_process(self, output_processes):
        self.spike_output = None
        count = 0
        for process in output_processes:
            self.process_output_map[process['index']] = count
            count += process['count']

        #self.process_output_map['INPUT'] = count
        #count += self.input_process.proc_params['shape'][0]

        self.pass_through_lif = self.LIF(
                                    shape=(count,),
                                    v=0,
                                    u=0,
                                    vth=0,
                                    du=self.current_decay_scale_factor,
                                    dv=self.voltage_decay_scale_factor,
                                    bias_mant=0,
                                    name=f"lif_passthrough",
                                    )
        sink_nx2py = self.NxToPyAdapter(shape=(count,))
        self.sink_output = self.Sink(shape=(count,), buffer=self.duration)

        self.pass_through_lif.s_out.connect(sink_nx2py.inp)
        sink_nx2py.out.connect(self.sink_output.a_in)

        self.outputUProbe = None
        self.outputVProbe = None
        #self.outputUProbe = StateProbe(self.pass_through_lif.u)
        #self.outputVProbe = StateProbe(self.pass_through_lif.v)

        #weights = np.eye(
                    #N=self.pass_through_lif.proc_params['shape'][0],
                    #M=self.input_process.proc_params['shape'][0],
                    #k=-self.process_output_map['INPUT'],
                    #) * 16384
        #exponent = calcWeightExponent(weights)
        #weights = calcHardwareWeights(weights, exponent)
        #dummy_connection = Sparse(weights=weights, weight_exp=exponent, sign_mode=SignMode.EXCITATORY)
        #self.input_process.s_out.connect(dummy_connection.s_in)
        #dummy_connection.a_out.connect(self.pass_through_lif.a_in)

    def get_config(self, callback_functions=[]):
        from lava.magma.core.run_configs import Loihi2HwCfg
        return Loihi2HwCfg(
                callback_fxs=callback_functions,
                exception_proc_model_map=self.proc_model_map)

    def get_lif_process(self, count, initial_voltages, spike_threshold, decay_constant, bias_mants, name):
        integer_initial_voltages = [round(voltage) for voltage in initial_voltages]
        integer_bias_mants = [round(bias) for bias in bias_mants]
        scaled_threshold = round(spike_threshold) >> 6 # This is to account for another extra 6 bits of scaling that is going on
        scaled_decay = int(self.voltage_decay_scale_factor * decay_constant)
        return self.LIF(
                shape=(count,),
                v=integer_initial_voltages,
                u=0,
                vth=scaled_threshold,
                du=self.current_decay_scale_factor,
                dv=scaled_decay,
                bias_mant=integer_bias_mants,
                name=name,
                )

    def connect_input_to_lif(self, input_weights, input_delays, lif_data, lif_process):
        lif_index = lif_data['index']
        max_delay = lif_data['max_delay']
        W = input_weights[lif_index].astype('int')
        D = input_delays[lif_index].astype('int')
        #print(f"Connecting input to {lif_data['process_name']}")
        #print(f"Original weights:\n{W}")
        if np.count_nonzero(W):
            exponent = calcWeightExponent(W)
            weights = calcHardwareWeights(W, exponent)
            formatted_weights = self.format_matrix(weights)
            formatted_delays = self.format_matrix(D)
            #print(f"Hardware weights:\n{weights}")
            #print(f"Delay:\n{D}")
            
            connection_name = f"input_to_{lif_data['process_name']}"
            c = self.delay_connection_model(
                    weights=formatted_weights,
                    delays=formatted_delays,
                    max_delay=max_delay,
                    weight_exp=exponent,
                    sign_mode=self.SignMode.MIXED,
                    name=connection_name,
                    )
            self.input_process.s_out.connect(c.s_in)
            c.a_out.connect(lif_process.a_in)
        input_weights[lif_index] = None

    def connect_lif_to_output(self, output_weights, lif_process):
        #print(f"Connecting: {lif_process.proc_params['name']} to pass_through")
        if np.count_nonzero(output_weights):
            #print(f"Original weights:\n{output_weights}")
            exponent = calcWeightExponent(output_weights)
            weights = calcHardwareWeights(output_weights, exponent)
            formatted_weights = self.format_matrix(weights)
            #print(f"Hardware weights:\n{weights}")
            dummy_connection = self.connection_model(
                    weights=formatted_weights,
                    weight_exp=exponent,
                    sign_mode=self.SignMode.EXCITATORY,
                    name=f"{lif_process.proc_params['name']}_pass_through",
                    )
            lif_process.s_out.connect(dummy_connection.s_in)
            dummy_connection.a_out.connect(self.pass_through_lif.a_in)

    def connect_lif_to_lif(self, source_lif_data, source_lif_process, dest_lif_data, dest_lif_process):
        #print(f"Connecting: {source_lif_data['process_name']} to {dest_lif_data['process_name']}")
        dest_lif_index = dest_lif_data['index']
        W = source_lif_data['W'][dest_lif_index].astype('int')
        D = source_lif_data['D'][dest_lif_index].astype('int')
        max_delay = dest_lif_data['max_delay']
        if np.count_nonzero(W):
            #print(f"Original weights:\n{W}")
            exponent = calcWeightExponent(W)
            weights = calcHardwareWeights(W, exponent)
            formatted_weights = self.format_matrix(weights)
            formatted_delays = self.format_matrix(D)
            #print(f"Hardware wegiths:\n{weights}")
            #print(f"Delay:\n{D}")

            c = self.delay_connection_model(
                    weights=formatted_weights,
                    delays=formatted_delays,
                    max_delay=max_delay,
                    weight_exp=exponent,
                    sign_mode=self.SignMode.MIXED,
                    name=f"{source_lif_process.proc_params['name']}_{dest_lif_process.proc_params['name']}",
                    )
            source_lif_process.s_out.connect(c.s_in)
            c.a_out.connect(dest_lif_process.a_in)

        source_lif_data['W'][dest_lif_index] = None

    def get_pass_through_weights(self, lif_data, weight_scale_factor):
        W = np.eye(
                N=self.pass_through_lif.proc_params['shape'][0],
                M=lif_data['count'],
                k=-self.process_output_map[lif_data['index']],
                )
        W = W * weight_scale_factor 
        return W

    def setup_probe(self, var):
        p = self.StateProbe(var)
        return p

    def get_spike_output(self, process_data):
        process_index = process_data['index']
        process_count = process_data['count']
        start = self.process_output_map[process_index]
        if self.spike_output is None:
            self.spike_output = self.sink_output.data.get()
            #print(self.spike_output)
        spike_output = self.spike_output[start:start + process_count]
        return spike_output

    def print_output_probe_data(self):
        print("Input probe data:")
        if self.inputUProbe is not None:
            num_input_neurons = int(len(self.inputUProbe.time_series) / self.duration)
            print("U:")
            print(self.inputUProbe.time_series.reshape(num_input_neurons, self.duration))
        if self.inputVProbe is not None:
            num_input_neurons = int(len(self.inputVProbe.time_series) / self.duration)
            print("V:")
            print(self.inputVProbe.time_series.reshape(num_input_neurons, self.duration))

        print("Output probe data:")
        if self.outputUProbe is not None:
            num_output_neurons = int(len(self.outputUProbe.time_series) / self.duration)
            print("U:")
            print(self.outputUProbe.time_series.reshape(num_output_neurons, self.duration))
        if self.outputVProbe is not None:
            num_output_neurons = int(len(self.outputVProbe.time_series) / self.duration)
            print("V:")
            print(self.outputVProbe.time_series.reshape(num_output_neurons, self.duration))
