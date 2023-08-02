import numpy as np
import pytest

from fugu.backends import snn_Backend
from fugu.bricks.convolution_bricks import convolution_1d, convolution_2d
from fugu.bricks.input_bricks import BaseP_Input
from fugu.scaffold import Scaffold


class Test_Convolution1D:
    def setup_method(self):
        self.basep = 2
        self.bits = 2
        self.pvector = [1, 1]
        self.filters = [2, 3]
        self.mode = "full"

    # Convolution answer: np.convolve([1,1],[2,3]) = [2,5,3]

    @pytest.mark.parametrize("basep", [2, 3, 4, 5, 6, 7, 8, 9])
    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 7, 8])
    @pytest.mark.parametrize("nSpikes", [0, 1, 2, 3])
    def test_full_mode_spikes(self, basep, bits, nSpikes):
        subt = np.zeros(3)
        subt[:nSpikes] = 0.1
        subt = np.reshape(subt, (3,))
        thresholds = (
            np.array([2, 5, 3]) - subt
        )  # convolution answer is [2,5,3]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_1d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("basep", [2, 3, 4, 5, 6, 7, 8, 9])
    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 7, 8])
    @pytest.mark.parametrize("nSpikes", [0, 1, 2])
    def test_same_mode_spikes(self, basep, bits, nSpikes):
        self.mode = "same"
        subt = np.zeros(2)
        subt[:nSpikes] = 0.1
        subt = np.reshape(subt, (2,))
        thresholds = (
            np.array([2, 5]) - subt
        )  # convolution answer is [2,5]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_1d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("basep", [2, 3, 4, 5, 6, 7, 8, 9])
    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 7, 8])
    @pytest.mark.parametrize("nSpikes", [0, 1])
    def test_valid_mode_spikes(self, basep, bits, nSpikes):
        self.mode = "valid"
        subt = np.zeros(1)
        subt[:nSpikes] = 0.1
        subt = np.reshape(subt, (1,))
        thresholds = (
            np.array([5]) - subt
        )  # convolution answer is [2,5]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_1d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("basep", [2])
    @pytest.mark.parametrize("bits", [2])
    @pytest.mark.parametrize("thresholds", np.arange(1.9, 5, 1))
    def test_scalar_threshold(self, basep, bits, thresholds):
        ans_thresholds = np.array(
            [2, 5, 3]
        )  # 2d convolution answer is [2,5,3]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire
        nSpikes = len(ans_thresholds[ans_thresholds > thresholds])

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_1d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    def test_thresholds_shape(self, mode):
        self.mode = mode
        with pytest.raises(ValueError):
            thresholds = np.array([[4, 10]])
            self.run_convolution_1d(thresholds)

    def get_num_output_neurons(self, thresholds):
        Am = len(self.pvector)
        Bm = len(self.filters)
        Gm = Am + Bm - 1

        if not hasattr(thresholds, "__len__") and (not isinstance(thresholds, str)):
            if self.mode == "full":
                thresholds_size = Gm

            if self.mode == "valid":
                lmins = np.minimum((Am), (Bm))
                lb = np.array([lmins - 1])
                ub = np.array([Gm - lmins])
                thresholds_size = ub[0] - lb[0] + 1

            if self.mode == "same":
                apos = np.array([Am])
                gpos = np.array([Gm])

                lb = np.floor(0.5 * (gpos - apos))
                ub = np.floor(0.5 * (gpos + apos) - 1)
                thresholds_size = ub[0] - lb[0] + 1
        else:
            thresholds_size = len(thresholds)

        return thresholds_size

    def output_spike_positions(self, basep, bits, pvector, filters, thresholds):
        thresholds_size = self.get_num_output_neurons(thresholds)
        offset = 4  # begin/complete nodes for input and output nodes
        input_basep_len = len(pvector) * basep * bits
        output_ini_position = offset + input_basep_len
        output_end_position = offset + input_basep_len + thresholds_size
        return [output_ini_position, output_end_position]

    def output_mask(self, output_positions, result):
        ini = output_positions[0]
        end = output_positions[1]
        return (result["neuron_number"] >= ini) & (result["neuron_number"] <= end)

    def run_convolution_1d(self, thresholds):
        scaffold = Scaffold()
        scaffold.add_brick(
            BaseP_Input(
                np.array([self.pvector]),
                p=self.basep,
                bits=self.bits,
                collapse_binary=False,
                name="Input0",
                time_dimension=False,
            ),
            "input",
        )
        scaffold.add_brick(
            convolution_1d(
                self.pvector,
                self.filters,
                thresholds,
                self.basep,
                self.bits,
                name="convolution_",
                mode=self.mode,
            ),
            [(0, 0)],
            output=True,
        )
        scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result


class Test_Convolution2D:
    def setup_method(self):
        self.basep = 2
        self.bits = 2
        self.pvector = [[1, 1], [1, 1]]
        self.filters = [[1, 2], [3, 4]]
        self.pshape = np.array(self.pvector).shape
        self.filters_shape = np.array(self.filters).shape
        self.mode = "full"

    @pytest.mark.parametrize("basep", [2, 3, 4, 5, 6, 7, 8, 9])
    @pytest.mark.parametrize("bits", [2, 3, 4, 7, 8])
    @pytest.mark.parametrize("nSpikes", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_full_mode_spikes(self, basep, bits, nSpikes):
        subt = np.zeros(9)
        subt[:nSpikes] = 0.1
        subt = np.reshape(subt, (3, 3))
        thresholds = (
            np.array([[1, 3, 2], [4, 10, 6], [3, 7, 4]]) - subt
        )  # 2d convolution answer is [[1,3,2][,4,10,6],[3,7,4]]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_2d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("basep", [2])
    @pytest.mark.parametrize("bits", [2])
    @pytest.mark.parametrize("thresholds", np.arange(0.9, 11, 1))
    def test_scalar_threshold(self, basep, bits, thresholds):
        ans_thresholds = np.array(
            [[1, 3, 2], [4, 10, 6], [3, 7, 4]]
        )  # 2d convolution answer is [[1,3,2][,4,10,6],[3,7,4]]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire
        nSpikes = len(ans_thresholds[ans_thresholds > thresholds])

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_2d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("basep", [2])
    @pytest.mark.parametrize("bits", [2])
    @pytest.mark.parametrize("nSpikes", [0, 1, 2, 3, 4])
    def test_same_mode_spikes(self, basep, bits, nSpikes):
        self.mode = "same"
        subt = np.zeros(4)
        subt[:nSpikes] = 0.1
        subt = np.reshape(subt, (2, 2))
        thresholds = (
            np.array([[1, 3], [4, 10]]) - subt
        )  # 2d convolution answer is [[1,3],[4,10]]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_2d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("basep", [2, 3])
    @pytest.mark.parametrize("bits", [2, 3])
    @pytest.mark.parametrize("nSpikes", [0, 1])
    def test_valid_mode_spikes(self, basep, bits, nSpikes):
        self.mode = "valid"
        subt = np.zeros(1)
        subt[:nSpikes] = 0.1
        subt = np.reshape(subt, (1, 1))
        thresholds = (
            np.array([[10]]) - subt
        )  # 2d convolution answer is [[10]]. Spikes fire when less than threshold. Thus subtract 0.1 so that spikes fire

        self.basep = basep
        self.bits = bits
        result = self.run_convolution_2d(thresholds)

        # get output positions in result
        output_positions = self.output_spike_positions(
            self.basep, self.bits, self.pvector, self.filters, thresholds
        )
        output_mask = self.output_mask(output_positions, result)

        # Check calculations
        if nSpikes == 0:
            expected_spikes = list(np.array([]))
        else:
            expected_spikes = list(np.ones((nSpikes,)))

        calculated_spikes = list(result[output_mask].to_numpy()[:, 0])
        assert expected_spikes == calculated_spikes

    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    def test_thresholds_shape(self, mode):
        self.mode = mode
        with pytest.raises(ValueError):
            thresholds = np.array([[4, 10]])
            self.run_convolution_2d(thresholds)

    def get_num_output_neurons(self, thresholds):
        Am, An = self.pshape
        Bm, Bn = self.filters_shape
        Gm, Gn = Am + Bm - 1, An + Bn - 1

        if not hasattr(thresholds, "__len__") and (not isinstance(thresholds, str)):
            if self.mode == "full":
                thresholds_size = (Gm) * (Gn)

            if self.mode == "valid":
                lmins = np.minimum((Am, An), (Bm, Bn))
                lb = lmins - 1
                ub = np.array([Gm, Gn]) - lmins
                thresholds_size = (ub[0] - lb[0] + 1) * (ub[1] - lb[1] + 1)

            if self.mode == "same":
                apos = np.array([Am, An])
                gpos = np.array([Gm, Gn])

                lb = np.floor(0.5 * (gpos - apos))
                ub = np.floor(0.5 * (gpos + apos) - 1)
                thresholds_size = (ub[0] - lb[0] + 1) * (ub[1] - lb[1] + 1)
        else:
            thresholds_size = np.size(thresholds)

        return thresholds_size

    def output_spike_positions(self, basep, bits, pvector, filters, thresholds):
        thresholds_size = self.get_num_output_neurons(thresholds)
        offset = 4  # begin/complete nodes for input and output nodes
        input_basep_len = np.size(pvector) * basep * bits
        output_ini_position = offset + input_basep_len
        output_end_position = offset + input_basep_len + thresholds_size
        return [output_ini_position, output_end_position]

    def output_mask(self, output_positions, result):
        ini = output_positions[0]
        end = output_positions[1]
        return (result["neuron_number"] >= ini) & (result["neuron_number"] <= end)

    def run_convolution_2d(self, thresholds):
        scaffold = Scaffold()
        scaffold.add_brick(
            BaseP_Input(
                np.array([self.pvector]),
                p=self.basep,
                bits=self.bits,
                collapse_binary=False,
                name="Input0",
                time_dimension=False,
            ),
            "input",
        )
        scaffold.add_brick(
            convolution_2d(
                self.pvector,
                self.filters,
                thresholds,
                self.basep,
                self.bits,
                name="convolution_",
                mode=self.mode,
            ),
            [(0, 0)],
            output=True,
        )
        scaffold.lay_bricks()
        scaffold.summary(verbose=1)
        backend = snn_Backend()
        backend_args = {}
        backend.compile(scaffold, backend_args)
        result = backend.run(5)
        return result
