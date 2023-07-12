"""
isort:skip_file
"""

from collections.abc import Iterable

import pytest
import numpy as np

from fugu.utils.validation import int_to_float, validate_instance, validate_type


@pytest.mark.parametrize(
    "n, f",
    [
        (1, 1.0),
        (-10, -10.0),
        (500, 500.0),
        (0, 0.0),
        (np.int32(1), 1.0),
        (np.int32(-9), -9.0),
        (np.int32(499), 499.0),
        (np.int32(0), 0.0),
        (np.int32(7.0), 7.0),
    ],
)
def test_int_to_float(n, f):
    out = int_to_float(n)
    assert type(out) is float
    assert out == f


@pytest.mark.parametrize(
    "n, t",
    [
        ("a", str),
        (None, type(None)),
        ([], list),
        (set(), set),
        ({}, dict),
        (7.0, float),
        (np.float64(7.0), np.float64),
        (np.float64(7), np.float64),
        (np.bool_(True), np.bool_),
        (np.bool_(False), np.bool_),
        (np.str_("testing"), np.str_),
    ],
)
def test_int_to_float_passthrough(n, t):
    out = int_to_float(n)
    assert type(out) is t


@pytest.mark.parametrize(
    "p, t",
    [
        (0, str),
        ("test", object),
        (1.0, int),
        (1, type(None)),
        ([], float),
        (1.0, np.int32),
        (1.0, np.float64),
    ],
)
def test_validate_type_raises(p, t):
    with pytest.raises(TypeError):
        validate_type(p, t)


@pytest.mark.parametrize(
    "p, t",
    [
        (1, int),
        (1.0, float),
        ("test", str),
        ([], list),
        (None, type(None)),
        ({}, dict),
        (np.int32(7), np.int32),
        (np.float64(7.0), np.float64),
        (np.bool_(True), np.bool_),
        (np.bool_(False), np.bool_),
        (np.str_("testing"), np.str_),
    ],
)
def test_validate_type(p, t):
    validate_type(p, t)


@pytest.mark.parametrize(
    "p, ts",
    [
        (1, [int, float, np.int32]),
        (1.0, [int, float, np.float64]),
        ("test", [str, int, []]),
        ([], [object, dict, list]),
        (None, [object, type(None), dict]),
        ({}, [list, dict]),
        (np.int32(7), [int, np.int32]),
        (np.float64(7.0), [float, np.float64]),
        (np.bool_(True), [bool, np.bool_]),
        (np.bool_(False), [bool, np.bool_]),
        (np.str_("testing"), [str, np.str_]),
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
        (np.int32(7), [Iterable, np.int32]),
        (np.float64(7.0), [Iterable, np.float64]),
        (np.str_("testing"), [Iterable, np.str_]),
        (np.bool_(True), [Iterable, np.bool_]),
        (np.bool_(False), [Iterable, np.bool_]),
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
