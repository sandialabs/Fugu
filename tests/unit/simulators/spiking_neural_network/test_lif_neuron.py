import pytest

from fugu.simulators.SpikingNeuralNetwork.neuron import LIFNeuron


@pytest.fixture
def default_neuron():
    return LIFNeuron()


@pytest.fixture
def custom_voltage_neuron():
    def _inner(v):
        return LIFNeuron(voltage=v)

    return _inner


@pytest.fixture
def dull_neuron():
    return LIFNeuron(voltage=0.1, p=0.0)


@pytest.fixture
def named_neuron():
    return LIFNeuron(name="Testing")


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
    assert default_neuron._b == 0.0
    assert default_neuron.voltage == 0.0
    assert default_neuron.v == 0.0

    assert default_neuron.presyn == set()
    assert default_neuron.record == False

    assert default_neuron.prob == 1.0


def test_constructor():
    neuron = LIFNeuron(
        name="neuron",
        threshold=1,
        reset_voltage=1,
        leakage_constant=1,
        voltage=1,
        bias=1,
        p=1,
    )
    assert neuron.name == "neuron"
    # from parent abstract class
    assert neuron.spike == False
    assert neuron.spike_hist == []

    assert neuron.threshold == 1.0
    assert neuron.reset_voltage == 1.0
    assert neuron.leakage_constant == 1.0
    # and their _ counterparts
    assert neuron._T == 1.0
    assert neuron._R == 1.0
    assert neuron._m == 1.0
    assert neuron._b == 1.0
    assert neuron.voltage == 1.0
    assert neuron.v == 1.0

    assert neuron.presyn == set()
    assert neuron.record == False

    assert neuron.prob == 1.0


@pytest.mark.parametrize("param", [True, {}, "test", list(), set()])
def test_constructor_type_errors(param):
    if type(param) is not str and param is not None:
        with pytest.raises(TypeError):
            LIFNeuron(name=param)

    with pytest.raises(TypeError):
        LIFNeuron(threshold=param)

    with pytest.raises(TypeError):
        LIFNeuron(reset_voltage=param)

    with pytest.raises(TypeError):
        LIFNeuron(leakage_constant=param)

    with pytest.raises(TypeError):
        LIFNeuron(voltage=param)

    with pytest.raises(TypeError):
        LIFNeuron(bias=param)

    if type(param) is not bool:
        with pytest.raises(TypeError):
            LIFNeuron(record=param)


@pytest.mark.parametrize("m", [-1, -0.1, 1.1, 10])
def test_warning_on_unrealistic_leakage_constant(m):
    with pytest.raises(UserWarning):
        LIFNeuron(leakage_constant=m)


@pytest.mark.parametrize(
    "p, expected_error",
    [
        (1.1, ValueError),
        (5.0, ValueError),
        (-0.1, ValueError),
        (-10.0, ValueError),
        (-10, ValueError),
        (True, TypeError),
        ({}, TypeError),
    ],
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


def test_update_state_on_dull_neuron(dull_neuron):
    reference_spike_hist = []

    for _ in range(100):
        assert dull_neuron.update_state() == None

        reference_spike_hist.append(False)

        assert dull_neuron.spike == False
        assert dull_neuron.v == 0.1
        assert dull_neuron.spike_hist == reference_spike_hist


def test_update_state_on_custom_voltage_neuron(capsys, custom_voltage_neuron):
    neuron = custom_voltage_neuron(0.1)

    # sneaking in a quick test for show_state method
    assert neuron.show_state() == None
    out, _ = capsys.readouterr()
    assert out == "Neuron None: 0.1 volts, spike = False\n"

    neuron.update_state()
    assert neuron.spike == True
    assert neuron.v == 0.0
    assert neuron.spike_hist == [True]

    # ditto, again...
    assert neuron.show_state() == None
    out, _ = capsys.readouterr()
    assert out == "Neuron None: 0.0 volts, spike = True\n"

    neuron.update_state()

    assert neuron.spike == False
    assert neuron.v == 0.0
    assert neuron.spike_hist == [True, False]

    # ditto, one last time
    assert neuron.show_state() == None
    out, _ = capsys.readouterr()
    assert out == "Neuron None: 0.0 volts, spike = False\n"


def test_show_state_of_default_neuron(capsys, default_neuron):
    assert default_neuron.show_state() == None
    out, _ = capsys.readouterr()
    assert out == "Neuron None: 0.0 volts, spike = False\n"


def test_show_state_of_custom_voltage_neuron(capsys, custom_voltage_neuron):
    neuron = custom_voltage_neuron(0.2)
    assert neuron.show_state() == None
    out, _ = capsys.readouterr()
    assert out == "Neuron None: 0.2 volts, spike = False\n"


def test_show_params_on_default_neuron(capsys, default_neuron):
    assert default_neuron.show_params() == None
    out, _ = capsys.readouterr()
    assert (
        out
        == "Neuron 'None':\nThreshold\t  :0.0 volts,\nReset voltage\t  :0.0 volts,\nLeakage Constant :1.0\nBias :0.0\n\n"
    )


def test_show_params(capsys):
    neuron = LIFNeuron(threshold=0.7, reset_voltage=0.2, leakage_constant=0.9)
    assert neuron.show_params() == None
    out, _ = capsys.readouterr()
    assert (
        out
        == "Neuron 'None':\nThreshold\t  :0.7 volts,\nReset voltage\t  :0.2 volts,\nLeakage Constant :0.9\nBias :0.0\n\n"
    )


def test_threshold_setter(default_neuron):
    assert default_neuron.threshold == 0.0
    default_neuron.threshold = 0.7
    assert default_neuron.threshold == 0.7


def test_reset_voltage_setter(default_neuron):
    assert default_neuron.reset_voltage == 0.0
    default_neuron.reset_voltage = 0.2
    assert default_neuron.reset_voltage == 0.2


def test_leakage_constant_setter(default_neuron):
    assert default_neuron.leakage_constant == 1.0
    default_neuron.leakage_constant = 0.9
    assert default_neuron.leakage_constant == 0.9


def test__str__(capsys, default_neuron):
    print(default_neuron)
    out, _ = capsys.readouterr()
    assert out == "LIFNeuron None(0.0, 0.0, 1.0)\n"


def test__repr__(capsys, default_neuron):
    print(repr(default_neuron))
    out, _ = capsys.readouterr()
    assert out == "LIFNeuron None\n"


def test_named__str__(capsys, named_neuron):
    print(named_neuron)
    out, _ = capsys.readouterr()
    assert out == "LIFNeuron Testing(0.0, 0.0, 1.0)\n"


def test_named__repr__(capsys, named_neuron):
    print(repr(named_neuron))
    out, _ = capsys.readouterr()
    assert out == "LIFNeuron Testing\n"


# TODO add test(s) for show_presynapses method
