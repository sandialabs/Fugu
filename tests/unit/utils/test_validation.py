"""
isort:skip_file
"""

from collections.abc import Iterable

import pytest

from fugu.utils.validation import int_to_float, validate_instance, validate_type


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
    "p, t", [(0, str), ("test", object), (1.0, int), (1, type(None)), ([], float)]
)
def test_validate_type_raises(p, t):
    with pytest.raises(TypeError):
        validate_type(p, t)


@pytest.mark.parametrize(
    "p, t",
    [(1, int), (1.0, float), ("test", str), ([], list), (None, type(None)), ({}, dict)],
)
def test_validate_type(p, t):
    validate_type(p, t)


@pytest.mark.parametrize(
    "p, ts",
    [
        (1, [int, float]),
        (1.0, [int, float]),
        ("test", [str, int, []]),
        ([], [object, dict, list]),
        (None, [object, type(None), dict]),
        ({}, [list, dict]),
    ],
)
def test_validate_types(p, ts):
    validate_type(p, ts)


@pytest.mark.parametrize(
    "p, instances",
    [
        (1, [int, Iterable]),
        ("test", [Iterable, int]),
        (1.0, [Iterable, float]),
        ({}, [Iterable, str]),
    ],
)
def test_validate_instances(p, instances):
    validate_instance(p, instances)


@pytest.mark.parametrize(
    "p, instances",
    [
        (1.0, [int, Iterable]),
        (1.0, [str, Iterable, dict]),
        (1.0, [[], Iterable, dict]),
    ],
)
def test_validate_instances_raises(p, instances):
    with pytest.raises(TypeError):
        validate_instance(p, instances)
