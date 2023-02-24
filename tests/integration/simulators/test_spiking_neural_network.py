import pandas as pd
import pytest

from fugu.simulators.SpikingNeuralNetwork.neuralnetwork import NeuralNetwork
from fugu.simulators.SpikingNeuralNetwork.neuron import InputNeuron, LIFNeuron
from fugu.simulators.SpikingNeuralNetwork.synapse import Synapse


@pytest.fixture
def a_neural_network():
    def _inner(record):
        nn = NeuralNetwork()
        in_neuron_a = LIFNeuron("a")
        in_neuron_b = LIFNeuron("b")
        layer_neuron_1 = LIFNeuron("layer-1-1", record=record)
        layer_neuron_2 = LIFNeuron("layer-1-2", record=record)
        layer_neuron_3 = LIFNeuron("layer-1-3", record=record)
        out_neuron = LIFNeuron("out", record=record)

        synapse_a_1 = Synapse(in_neuron_a, layer_neuron_1)
        synapse_a_2 = Synapse(in_neuron_a, layer_neuron_2)
        synapse_a_3 = Synapse(in_neuron_a, layer_neuron_3)
        synapse_b_1 = Synapse(in_neuron_b, layer_neuron_1)
        synapse_b_2 = Synapse(in_neuron_b, layer_neuron_2)
        synapse_b_3 = Synapse(in_neuron_b, layer_neuron_3)
        synapse_1_o = Synapse(layer_neuron_1, out_neuron)
        synapse_2_o = Synapse(layer_neuron_2, out_neuron)
        synapse_3_o = Synapse(layer_neuron_3, out_neuron)
        synapse_2_1 = Synapse(layer_neuron_2, layer_neuron_1)
        synapse_2_2 = Synapse(layer_neuron_2, layer_neuron_2)
        synapse_2_3 = Synapse(layer_neuron_2, layer_neuron_3)

        nn.add_multiple_neurons(
            [
                in_neuron_a,
                in_neuron_b,
                layer_neuron_1,
                layer_neuron_2,
                layer_neuron_3,
                out_neuron,
            ]
        )

        nn.add_multiple_synapses(
            [
                synapse_a_1,
                synapse_a_2,
                synapse_a_3,
                synapse_b_1,
                synapse_b_2,
                synapse_b_3,
                synapse_1_o,
                synapse_2_o,
                synapse_3_o,
                synapse_2_1,
                synapse_2_2,
                synapse_2_3,
            ]
        )

        return nn

    return _inner


def test_run(a_neural_network):
    nn = a_neural_network(False)
    df = nn.run()
    assert type(df) == type(pd.DataFrame())
    assert df.shape == (1, 0)

    df = nn.run(n_steps=100)
    assert type(df) == type(pd.DataFrame())
    assert df.shape == (100, 0)


def test_run_with_record(a_neural_network):
    nn = a_neural_network(True)
    df = nn.run()
    assert type(df) == type(pd.DataFrame())
    assert df.shape == (1, 4)

    for _, row in df.iterrows():
        assert row["layer-1-1"] == 0
        assert row["layer-1-2"] == 0
        assert row["layer-1-3"] == 0
        assert row["out"] == 0

    df = nn.run(n_steps=100)
    assert type(df) == type(pd.DataFrame())
    assert df.shape == (100, 4)

    for _, row in df.iterrows():
        assert row["layer-1-1"] == 0
        assert row["layer-1-2"] == 0
        assert row["layer-1-3"] == 0
        assert row["out"] == 0


# TODO what does a Spiking Neural Network look like?
#      are they designed like ML and Deep Learning NN? inputs -> layer 1 -> ... -> layer n -> outputs
#      can they be designed after a general graph?
