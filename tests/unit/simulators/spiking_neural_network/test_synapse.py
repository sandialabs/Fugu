from collections import deque

import numpy as np
import pytest

from fugu.simulators.SpikingNeuralNetwork.neuron import InputNeuron, LIFNeuron
from fugu.simulators.SpikingNeuralNetwork.synapse import Synapse


@pytest.fixture
def lif_neuron():
    def _inner(name=None):
        return LIFNeuron(name=name)

    return _inner


@pytest.fixture
def input_neuron():
    def _inner(name=None):
        return InputNeuron(name=name)

    return _inner


@pytest.fixture
def default_synapse_w_lif_neurons():
    n1 = LIFNeuron("n1")
    n2 = LIFNeuron("n2")
    return Synapse(n1, n2)


@pytest.fixture
def default_synapse_w_input_neurons():
    n1 = InputNeuron("n1")
    n2 = InputNeuron("n2")
    return Synapse(n1, n2)


@pytest.mark.parametrize(
    "pre_neuron, post_neuron",
    [
        (None, lif_neuron),
        (input_neuron, None),
        (None, None),
        (lif_neuron, object),
        ([], input_neuron),
        (lif_neuron, set()),
        (1, 2),
        (input_neuron, "fail"),
    ],
)
def test_constructor_exceptions(pre_neuron, post_neuron):
    with pytest.raises(TypeError):
        Synapse(pre_neuron, post_neuron)


@pytest.mark.parametrize("delay", ["fail", [], set, object])
def test_contructor_delay_type_check(lif_neuron, delay):
    with pytest.raises(TypeError):
        Synapse(lif_neuron("n1"), lif_neuron("n2"), delay=delay)


@pytest.mark.parametrize(
    "delay",
    [
        0,
    ],
)
def test_constructor_delay_value_check(lif_neuron, delay):
    with pytest.raises(ValueError):
        Synapse(lif_neuron("n1"), lif_neuron("n2"), delay=delay)


def test_constructor_defaults(
    default_synapse_w_lif_neurons, default_synapse_w_input_neurons
):
    assert default_synapse_w_lif_neurons.delay == 1
    assert default_synapse_w_lif_neurons._d == 1
    assert default_synapse_w_lif_neurons.weight == 1.0
    assert default_synapse_w_lif_neurons._w == 1.0
    assert default_synapse_w_lif_neurons._hist == deque(np.zeros(1))
    assert default_synapse_w_lif_neurons.name == "s_n1_n2"

    assert default_synapse_w_input_neurons.delay == 1
    assert default_synapse_w_input_neurons._d == 1
    assert default_synapse_w_input_neurons.weight == 1.0
    assert default_synapse_w_input_neurons._w == 1.0
    assert default_synapse_w_input_neurons._hist == deque(np.zeros(1))
    assert default_synapse_w_input_neurons.name == "s_n1_n2"


def test_neuron_getters():
    pre_neuron = LIFNeuron("pre")
    post_neuron = LIFNeuron("post")
    synapse = Synapse(pre_neuron, post_neuron)

    assert synapse.pre_neuron == pre_neuron
    assert synapse.post_neuron == post_neuron


def test_synapse_key():
    pre_neuron = LIFNeuron("pre")
    post_neuron = LIFNeuron("post")
    synapse = Synapse(pre_neuron, post_neuron)

    assert synapse.get_key() == (pre_neuron, post_neuron)


def test_weight_setter(default_synapse_w_lif_neurons, default_synapse_w_input_neurons):
    assert default_synapse_w_lif_neurons.weight == 1.0
    default_synapse_w_lif_neurons.weight = 2.0
    assert default_synapse_w_lif_neurons.weight == 2.0

    assert default_synapse_w_input_neurons.weight == 1.0
    default_synapse_w_input_neurons.weight = 3.0
    assert default_synapse_w_input_neurons.weight == 3.0


@pytest.mark.parametrize("delay", ["fail", [], set, object])
def test_delay_setter_type_check(default_synapse_w_lif_neurons, delay):
    assert default_synapse_w_lif_neurons.delay == 1
    with pytest.raises(TypeError):
        default_synapse_w_lif_neurons.delay = delay


@pytest.mark.parametrize(
    "delay",
    [
        0,
    ],
)
def test_delay_setter_value_check(default_synapse_w_lif_neurons, delay):
    assert default_synapse_w_lif_neurons.delay == 1
    with pytest.raises(ValueError):
        default_synapse_w_lif_neurons.delay = delay


def test_delay_setter(default_synapse_w_input_neurons):
    assert default_synapse_w_input_neurons.delay == 1
    default_synapse_w_input_neurons.delay = 2
    assert default_synapse_w_input_neurons.delay == 2


@pytest.mark.parametrize("delay", ["fail", [], set, object])
def test_set_params_type_check(default_synapse_w_lif_neurons, delay):
    with pytest.raises(TypeError):
        default_synapse_w_lif_neurons.set_params(new_delay=delay)


@pytest.mark.parametrize(
    "delay",
    [
        0,
        -1,
    ],
)
def test_set_params_value_check(default_synapse_w_lif_neurons, delay):
    with pytest.raises(ValueError):
        default_synapse_w_lif_neurons.set_params(new_delay=delay)


@pytest.mark.parametrize("delay", [2, 3, 4, 100])
def test_set_params(
    default_synapse_w_lif_neurons, default_synapse_w_input_neurons, delay
):
    assert default_synapse_w_lif_neurons.delay == 1
    default_synapse_w_lif_neurons.set_params(new_delay=delay)
    assert default_synapse_w_lif_neurons.delay == delay

    assert default_synapse_w_input_neurons.delay == 1
    default_synapse_w_input_neurons.set_params(new_delay=delay)
    assert default_synapse_w_input_neurons.delay == delay


def test_show_params(capsys, default_synapse_w_lif_neurons):
    assert default_synapse_w_lif_neurons.show_params() == None
    out, _ = capsys.readouterr()
    assert (
        out
        == "Synapse LIFNeuron n1(0.0, 0.0, 1.0) -> LIFNeuron n2(0.0, 0.0, 1.0):\n delay  : 1\n weight : 1.0\n"
    )

    assert default_synapse_w_lif_neurons.set_params(new_delay=2, new_weight=2.0) == None
    assert default_synapse_w_lif_neurons.show_params() == None
    out, _ = capsys.readouterr()
    assert (
        out
        == "Synapse LIFNeuron n1(0.0, 0.0, 1.0) -> LIFNeuron n2(0.0, 0.0, 1.0):\n delay  : 2\n weight : 2.0\n"
    )


def test_named__str__(capsys, default_synapse_w_lif_neurons):
    print(default_synapse_w_lif_neurons)
    out, _ = capsys.readouterr()
    assert out == "Synapse s_n1_n2(1, 1.0)\n"


def test_named__repr__(capsys, default_synapse_w_lif_neurons):
    print(repr(default_synapse_w_lif_neurons))
    out, _ = capsys.readouterr()
    assert out == "s_n1_n2\n"


# TODO input validation for weight
# TODO test update_state method
