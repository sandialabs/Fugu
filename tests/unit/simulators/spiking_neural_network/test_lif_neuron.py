import pytest
from fugu.simulators.SpikingNeuralNetwork.neuron import LIFNeuron


@pytest.fixture
def default_neuron():
    return LIFNeuron()


def test_constructor_defaults(default_neuron):
    assert default_neuron.name == None
    # from parent abstract class
    assert default_neuron.spike == False
    assert default_neuron.spike_hist == []

    assert default_neuron.threshold == 0.0
    assert default_neuron.reset_voltage == 0.0
    assert default_neuron.leakage_constant == 1.0
    # and their _ counterparts
    assert default_neuron._T == 0.0
    assert default_neuron._R == 0.0
    assert default_neuron._m == 1.0

    assert default_neuron.voltage == 0.0
    assert default_neuron.v == 0.0

    assert default_neuron.presyn == set()
    assert default_neuron.record == False

    assert default_neuron.prob == 1.0


@pytest.mark.parametrize(
    "p, expected_error",
    [(1.1, ValueError), (5.0, ValueError), (-0.1, ValueError), (-10, ValueError)],
)
def test_invalid_spiking_probablity(p, expected_error):
    with pytest.raises(expected_error):
        LIFNeuron(p=p)


def test_update_state_on_default_neuron(default_neuron):
    reference_spike_hist = []

    for _ in range(100):
        assert default_neuron.update_state() == None

        reference_spike_hist.append(False)

        assert default_neuron.spike == False
        assert default_neuron.v == 0.0
        assert default_neuron.spike_hist == reference_spike_hist

def test_show_state_of_default_neuron(capsys, default_neuron):
    assert default_neuron.show_state() == None

    out, _ = capsys.readouterr()
    assert out == "Neuron None: 0.0 volts, spike = False\n"
