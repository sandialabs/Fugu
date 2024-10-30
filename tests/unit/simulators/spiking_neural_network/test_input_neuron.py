import pytest

from fugu.simulators.SpikingNeuralNetwork.neuron import InputNeuron

base_neuron = InputNeuron

@pytest.fixture
def default_neuron():
    return base_neuron()


@pytest.fixture
def custom_voltage_neuron():
    def _inner(v):
        return base_neuron(voltage=v)

    return _inner


@pytest.fixture
def named_neuron():
    return base_neuron(name="Testing")


def test_constructor_defaults(default_neuron):
    assert default_neuron.name == None
    # from parent abstract class
    assert default_neuron.spike == False
    assert default_neuron.spike_hist == []

    assert default_neuron.threshold == 0.1
    assert default_neuron.voltage == 0.0
    assert default_neuron.record == False
    # and their _ counterparts
    assert default_neuron._T == 0.1

    assert default_neuron._it == None


def test_constructor_custom_voltage(custom_voltage_neuron):
    neuron = custom_voltage_neuron(0.1)
    assert neuron.voltage == 0.1


@pytest.mark.parametrize(
    "in_stream",
    [[0, 1, 2], list(), (0, 1), set(), sorted([4, 1, 3, 2])],
)
def test_connect_to_input(default_neuron, in_stream):
    assert default_neuron.connect_to_input(in_stream) == None


def test_connect_to_input_exceptions(default_neuron):
    in_stream = 1234
    with pytest.raises(TypeError):
        default_neuron.connect_to_input(in_stream)


def test_threshold_setter(default_neuron):
    assert default_neuron.threshold == 0.1
    default_neuron.threshold = 0.7
    assert default_neuron.threshold == 0.7


def test__str__(capsys, default_neuron):
    print(default_neuron)
    out, _ = capsys.readouterr()
    val_str = "InputNeuron"
    assert out == f"{val_str} None\n"


def test__repr__(capsys, default_neuron):
    print(repr(default_neuron))
    out, _ = capsys.readouterr()
    val_str = "InputNeuron"
    assert out == f"{val_str} None\n"


@pytest.mark.parametrize("in_stream", [[0.1 + 0.2j]])
def test_update_state_exception(default_neuron, in_stream):
    default_neuron.connect_to_input(in_stream)
    with pytest.raises(TypeError):
        default_neuron.update_state()


def test_update_state_on_default_neuron(default_neuron):
    with pytest.raises(TypeError):
        default_neuron.update_state()

    default_neuron.connect_to_input([-0.01, 0.2])

    default_neuron.update_state()
    assert default_neuron.spike == False
    assert default_neuron.voltage == 0

    default_neuron.update_state()
    assert default_neuron.spike == True
    assert default_neuron.voltage == 0

    default_neuron.update_state()
    assert default_neuron.spike == False
    assert default_neuron.voltage == 0


def test_named__str__(capsys, named_neuron):
    print(named_neuron)
    out, _ = capsys.readouterr()
    val_str = "InputNeuron"
    assert out == f"{val_str} Testing\n"


def test_named__repr__(capsys, named_neuron):
    print(repr(named_neuron))
    out, _ = capsys.readouterr()
    val_str = "InputNeuron"
    assert out == f"{val_str} Testing\n"
