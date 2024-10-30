import pandas as pd
import pytest

from fugu.simulators.SpikingNeuralNetwork.neuralnetwork import NeuralNetwork
from fugu.simulators.SpikingNeuralNetwork.neuron import InputNeuron, LIFNeuron
from fugu.simulators.SpikingNeuralNetwork.synapse import LearningSynapse


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

        synapse_a_1 = LearningSynapse(in_neuron_a, layer_neuron_1)
        synapse_a_2 = LearningSynapse(in_neuron_a, layer_neuron_2)
        synapse_a_3 = LearningSynapse(in_neuron_a, layer_neuron_3)
        synapse_b_1 = LearningSynapse(in_neuron_b, layer_neuron_1)
        synapse_b_2 = LearningSynapse(in_neuron_b, layer_neuron_2)
        synapse_b_3 = LearningSynapse(in_neuron_b, layer_neuron_3)
        synapse_1_o = LearningSynapse(layer_neuron_1, out_neuron)
        synapse_2_o = LearningSynapse(layer_neuron_2, out_neuron)
        synapse_3_o = LearningSynapse(layer_neuron_3, out_neuron)
        synapse_2_1 = LearningSynapse(layer_neuron_2, layer_neuron_1)
        synapse_2_2 = LearningSynapse(layer_neuron_2, layer_neuron_2)
        synapse_2_3 = LearningSynapse(layer_neuron_2, layer_neuron_3)

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


@pytest.fixture
def spiking_neural_network():
    nn = NeuralNetwork()
    a = LIFNeuron("a", voltage=0.1, record=True)
    b = LIFNeuron("b", record=True)
    synapse = LearningSynapse(a, b)
    nn.add_multiple_neurons([a, b])
    nn.add_synapse(synapse)
    return nn


@pytest.fixture
def spiking_neural_network_w_reset():
    nn = NeuralNetwork()
    a = LIFNeuron("a", voltage=0.1, reset_voltage=0.1, record=True)
    b = LIFNeuron("b", record=True)
    synapse = LearningSynapse(a, b)
    nn.add_multiple_neurons([a, b])
    nn.add_synapse(synapse)
    return nn


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


def test_run_with_spikes(spiking_neural_network):
    df = spiking_neural_network.run(n_steps=2)
    assert type(df) == type(pd.DataFrame())
    assert df.shape == (2, 2)
    assert df.size == 4
    assert df.loc[0]["a"] == 1
    assert df.loc[0]["b"] == 0
    assert df.loc[1]["a"] == 0
    assert df.loc[1]["b"] == 1


def test_run_with_spikes_debug(spiking_neural_network):
    df = spiking_neural_network.run(n_steps=2, debug_mode=True)
    assert type(df) == type(pd.DataFrame())
    assert df.shape == (2, 2)
    assert df.size == 4
    assert df.loc[0]["a"] == (1, 0.0)
    assert df.loc[0]["b"] == (0, 0.0)
    assert df.loc[1]["a"] == (0, 0.0)
    assert df.loc[1]["b"] == (1, 0.0)


def test_run_with_spikes_reset_debug(spiking_neural_network_w_reset):
    df = spiking_neural_network_w_reset.run(n_steps=2, debug_mode=True)
    assert type(df) == type(pd.DataFrame())
    assert df.shape == (2, 2)
    assert df.size == 4
    assert df.loc[0]["a"] == (1, 0.1)
    assert df.loc[0]["b"] == (0, 0.0)
    assert df.loc[1]["a"] == (1, 0.1)
    assert df.loc[1]["b"] == (1, 0.0)


def test_run_with_record_potentials(a_neural_network, spiking_neural_network, spiking_neural_network_w_reset):
    df, fp = a_neural_network(False).run(record_potentials=True)
    assert type(df) == type(pd.DataFrame())
    assert type(fp) == type(pd.DataFrame())
    for i in range(5):
        assert fp.loc[i]["potential"] == 0.0
        assert fp.loc[i]["neuron_number"] == float(i)

    df, fp = spiking_neural_network.run(record_potentials=True)
    assert type(df) == type(pd.DataFrame())
    assert type(fp) == type(pd.DataFrame())
    for i in range(2):
        assert fp.loc[i]["potential"] == 0.0
        assert fp.loc[i]["neuron_number"] == float(i)

    df, fp = spiking_neural_network_w_reset.run(record_potentials=True)
    assert type(df) == type(pd.DataFrame())
    assert type(fp) == type(pd.DataFrame())
    assert fp.loc[0]["potential"] == 0.1
    assert fp.loc[0]["neuron_number"] == 0.0
    assert fp.loc[1]["potential"] == 0.0
    assert fp.loc[1]["neuron_number"] == 1.0
