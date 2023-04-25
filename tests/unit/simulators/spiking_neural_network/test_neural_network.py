import pytest

from fugu.simulators.SpikingNeuralNetwork.neuralnetwork import NeuralNetwork
from fugu.simulators.SpikingNeuralNetwork.neuron import LIFNeuron
from fugu.simulators.SpikingNeuralNetwork.synapse import Synapse


@pytest.fixture
def blank_network():
    return NeuralNetwork()


@pytest.fixture
def neuron_a():
    return LIFNeuron(name="a")


@pytest.fixture
def neurons_and_synapses():
    """
    Fixture for a set of LIFNeurons, connected via synapses as follows:
        a --> b --> c --> d

    Returned dictionary can be described as:
        neurons[0] = a
        neurons[1] = b
        neurons[2] = c
        neurons[3] = d

        synapses[0] = a --> b
        synapses[1] = b --> c
        synapses[2] = c --> d
    """
    neuron_a = LIFNeuron("a")
    neuron_b = LIFNeuron("b")
    neuron_c = LIFNeuron("c")
    neuron_d = LIFNeuron("d")
    neurons = [neuron_a, neuron_b, neuron_c, neuron_d]

    synapse_a_b = Synapse(neuron_a, neuron_b)
    synapse_b_c = Synapse(neuron_b, neuron_c)
    synapse_c_d = Synapse(neuron_c, neuron_d)
    synapses = [synapse_a_b, synapse_b_c, synapse_c_d]

    return {"neurons": neurons, "synapses": synapses}


def test_constructor(blank_network):
    assert blank_network.nrns == {}
    assert blank_network.synps == {}
    assert blank_network._nrn_count == 0


@pytest.mark.parametrize(
    "neuron",
    [
        1,
    ],
)
def test_add_neuron_type_int_check(blank_network, neuron):
    with pytest.raises(TypeError):
        blank_network.add_neuron(neuron)


@pytest.mark.xfail()
@pytest.mark.parametrize(
    "neuron",
    [
        {},
        [],
        set(),
    ],
)
def test_add_neuron_type_check(blank_network, neuron):
    with pytest.raises(TypeError):
        blank_network.add_neuron(neuron)


def test_add_neuron(blank_network, neuron_a):
    assert blank_network.add_neuron() == None
    assert blank_network._nrn_count == 1
    assert len(blank_network.nrns) == 1

    assert blank_network.add_neuron(neuron_a) == None
    assert blank_network._nrn_count == 2
    assert len(blank_network.nrns) == 2
    assert blank_network.nrns["a"] == neuron_a

    assert blank_network.add_neuron("b") == None
    assert blank_network._nrn_count == 3
    assert len(blank_network.nrns) == 3


@pytest.mark.parametrize(
    "neurons",
    [
        1,
        2.0,
    ],
)
def test_add_multiple_neurons_check(blank_network, neurons):
    with pytest.raises(TypeError):
        blank_network.add_multiple_neurons(neurons)


@pytest.mark.xfail
def test_add_multiple_neurons_w_none(blank_network):
    assert len(blank_network.nrns) == 0
    assert blank_network.add_multiple_neurons() == None
    assert len(blank_network.nrns) == 1


def test_add_multiple_neurons(blank_network):
    assert len(blank_network.nrns) == 0
    assert blank_network.add_multiple_neurons("abcd") == None
    assert len(blank_network.nrns) == 4


def test_list_neurons_empty(capsys, blank_network):
    assert blank_network.list_neurons() == None
    out, _ = capsys.readouterr()
    assert out == "Neurons: {\b\b}\n"


def test_list_neurons(capsys, blank_network):
    assert blank_network.add_multiple_neurons("abcd") == None
    assert blank_network.list_neurons() == None
    out, _ = capsys.readouterr()
    assert out == "Neurons: {a, b, c, d, \b\b}\n"


def test_add_synapse_none_check(blank_network):
    with pytest.raises(TypeError):
        blank_network.add_synapse()


def test_add_synapse_type_check(blank_network):
    with pytest.raises(TypeError):
        blank_network.add_synapse({})


def test_add_synapse_subclass(blank_network):
    """
    This particular test case covers the `isinstance` check in the
    `add_synapse` method for when the previous `type` check fails in the event
    that the `new_synapse` is a subclass of Synapse, and not an instance of it
    itself.
    """

    class SubSynapse(Synapse):
        pass

    synapse = SubSynapse(LIFNeuron("a"), LIFNeuron("b"))
    assert blank_network.add_synapse(synapse) == None


def test_add_synapse_class_type_check(blank_network, neuron_a):
    with pytest.raises(TypeError):
        blank_network.add_synapse(neuron_a)


@pytest.mark.parametrize(
    "new_synapse",
    [
        (LIFNeuron("0"), LIFNeuron("1")),
        (LIFNeuron("0"), LIFNeuron("1"), 1, 1.0),
    ],
)
def test_add_synapse_tuple(blank_network, new_synapse):
    blank_network.add_synapse(new_synapse)


@pytest.mark.parametrize(
    "new_synapse",
    [
        (LIFNeuron("0")),
        (LIFNeuron("0"), LIFNeuron("1"), 1, 1.0, None),
    ],
)
def test_add_synapse_tuple_error(blank_network, new_synapse):
    with pytest.raises(TypeError):
        blank_network.add_synapse(new_synapse)


def test_add_synapse(capsys, blank_network, neurons_and_synapses):
    neurons = neurons_and_synapses["neurons"]
    synapses = neurons_and_synapses["synapses"]

    assert blank_network.add_multiple_neurons(neurons) == None
    assert blank_network.synps == {}

    assert blank_network.add_synapse(synapses[0]) == None
    assert blank_network.synps[(neurons[0], neurons[1])] == synapses[0]
    assert blank_network.add_synapse(synapses[1]) == None
    assert blank_network.synps[(neurons[1], neurons[2])] == synapses[1]
    assert blank_network.add_synapse(synapses[2]) == None
    assert blank_network.synps[(neurons[2], neurons[3])] == synapses[2]
    assert len(blank_network.synps) == 3

    assert blank_network.add_synapse(synapses[0]) == None
    out, _ = capsys.readouterr()
    assert (
        out
        == "Warning! Not Added! Synapse s_a_b(1, 1.0) already defined in network. (Use <synapse>.set_params() to update synapse)\n"
    )


@pytest.mark.parametrize(
    "synapses",
    [
        1,
        2.0,
    ],
)
def test_add_multiple_synapses_check(blank_network, synapses):
    with pytest.raises(TypeError):
        blank_network.add_multiple_synapses(synapses)


def test_add_multiple_synapse(blank_network, neurons_and_synapses):
    neurons = neurons_and_synapses["neurons"]
    synapses = neurons_and_synapses["synapses"]

    assert blank_network.add_multiple_synapses(synapses) == None
    assert blank_network.synps[(neurons[0], neurons[1])] == synapses[0]
    assert blank_network.synps[(neurons[1], neurons[2])] == synapses[1]
    assert blank_network.synps[(neurons[2], neurons[3])] == synapses[2]
    assert len(blank_network.synps) == 3


def test_step(blank_network, neurons_and_synapses):
    synapses = neurons_and_synapses["synapses"]

    assert blank_network.step() == None
    assert blank_network.add_multiple_synapses(synapses) == None
    assert blank_network.step() == None


# TODO should add_multiple_neurons method accept {} (even though it's an instance of Iterable)
# TODO list_neurons: why the "\b\b"?
# TODO remove NOT SURE IF THIS IS NEEDED section - after legacy test suite is ported over
# TODO test update_input_neuron method
