import pytest

from fugu.utils.validation import int_to_float, validate_type


@pytest.mark.parametrize("n, f", [(1, 1.0), (-10, -10.0), (500, 500.0), (0, 0.0)])
def test_int_to_float(n, f):
    out = int_to_float(n)
    assert type(out) is float
    assert out == f


@pytest.mark.parametrize(
    "n, t", [("a", str), (None, type(None)), ([], list), (set(), set), ({}, dict)]
)
def test_int_to_float_passthrough(n, t):
    out = int_to_float(n)
    assert type(out) is t


@pytest.mark.parametrize(
    "i, t", [(0, str), ("test", object), (1.0, int), (1, type(None)), ([], float)]
)
def test_validate_type_raises(i, t):
    with pytest.raises(TypeError):
        validate_type(i, t)


@pytest.mark.parametrize(
    "i, t",
    [(1, int), (1.0, float), ("test", str), ([], list), (None, type(None)), ({}, dict)],
)
def test_validate_type(i, t):
    validate_type(i, t)


@pytest.mark.parametrize(
    "i, ts",
    [
        (1, [int, float]),
        (1.0, [int, float]),
        ("test", [str, int, []]),
        ([], [object, dict, list]),
        (None, [object, type(None), dict]),
        ({}, [list, dict]),
    ],
)
def test_validate_types(i, ts):
    validate_type(i, ts)
