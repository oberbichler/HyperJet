import pytest
import hyperjet as hj
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from copy import copy
import pickle

if __name__ == "__main__":
    import os
    import sys

    print(f"pid: {os.getpid()}")
    pytest.main(sys.argv)


# initialization


def test_init():
    u = hj.SScalar(f=1.2)
    assert_equal(len(u), 0)
    assert_allclose(u.f, 1.2)
    assert_allclose(u.d("x"), 0)
    assert_allclose(u.d("y"), 0)
    assert_allclose(u.d("z"), 0)

    u = hj.SScalar(f=1.2, d={"x": 1, "y": 2})
    assert_equal(len(u), 2)
    assert_allclose(u.f, 1.2)
    assert_allclose(u.d("x"), 1)
    assert_allclose(u.d("y"), 2)
    assert_allclose(u.d("z"), 0)


def test_mul():
    u = hj.SScalar(f=1, d={"x": 1})
    v = hj.SScalar(f=1, d={"y": 1})
    r = u + v**2
    assert_equal(len(r), 2)
    assert_allclose(r.f, 2)
    assert_allclose(r.d("x"), 1)
    assert_allclose(r.d("y"), 2)
    assert_allclose(r.d("z"), 0)


def test_self_multiplication():
    r = hj.SScalar(f=3, d={"x": 1, "y": 6})
    r *= r

    u = hj.SScalar(f=3, d={"x": 1, "y": 6})
    e = u * hj.SScalar(f=3, d={"x": 1, "y": 6})

    assert_allclose(r.f, e.f)
    assert_allclose(r.d("x"), e.d("x"))
    assert_allclose(r.d("y"), e.d("y"))

    assert_allclose(r.f, 9)
    assert_allclose(r.d("x"), 6)
    assert_allclose(r.d("y"), 36)


def test_self_division():
    r = hj.SScalar(f=3, d={"x": 1, "y": 6})
    r /= r

    assert_allclose(r.f, 1)
    assert_allclose(r.d("x"), 0, atol=1e-15)
    assert_allclose(r.d("y"), 0, atol=1e-15)


# The binding surface. The mathematics is characterized in
# test/src/test_sscalar.cpp; these tests pin what the bindings add on top.


def test_construction():
    assert_equal(len(hj.SScalar()), 0)
    assert_allclose(hj.SScalar().f, 0)

    assert_allclose(hj.SScalar(f=1.5).f, 1.5)
    assert_equal(len(hj.SScalar(f=1.5)), 0)

    assert_allclose(hj.SScalar.constant(1.5).f, 1.5)
    assert_equal(len(hj.SScalar.constant(1.5)), 0)

    v = hj.SScalar.variable("a", 1.5)

    assert_allclose(v.f, 1.5)
    assert_equal(len(v), 1)
    assert_allclose(v.d("a"), 1)

    # an unknown name has a zero derivative rather than raising
    assert_allclose(v.d("b"), 0)


def test_properties():
    u = hj.SScalar(f=3, d={"x": 1, "y": 6})

    assert_allclose(u.f, 3)
    assert_equal(u.size, 2)
    assert_equal(len(u), 2)


def test_repr():
    # The derivatives live in an unordered map, so the order of the terms is
    # unspecified. Only the value comes first and every term appears once.
    text = repr(hj.SScalar(f=3, d={"x": 1, "y": 6}))

    assert text.startswith("3")
    assert "+1*dx" in text
    assert "+6*dy" in text

    # negative derivatives are printed without a plus sign
    assert_equal(repr(hj.SScalar(f=1, d={"x": -2})), "1 -2*dx")


def test_eval():
    u = hj.SScalar(f=3, d={"x": 1, "y": 6})

    assert_allclose(u.eval({"x": 0.5, "y": -0.25}), 2.0)

    # names missing from the displacement contribute nothing, unknown ones are
    # ignored
    assert_allclose(u.eval({"x": 0.5}), 3.5)
    assert_allclose(u.eval({}), 3.0)
    assert_allclose(u.eval({"w": 100.0}), 3.0)


def test_operators():
    u = hj.SScalar(f=3, d={"x": 1})
    v = hj.SScalar(f=4, d={"y": 2})

    for op, f, dx, dy in [
        (lambda: u + v, 7, 1, 2),
        (lambda: u - v, -1, 1, -2),
        (lambda: u * v, 12, 4, 6),
        (lambda: u / v, 0.75, 0.25, -0.375),
        (lambda: -u, -3, -1, 0),
    ]:
        r = op()
        assert_allclose(r.f, f)
        assert_allclose(r.d("x"), dx)
        assert_allclose(r.d("y"), dy)

    for op, f, dx in [
        (lambda: u + 2, 5, 1),
        (lambda: 2 + u, 5, 1),
        (lambda: u - 2, 1, 1),
        (lambda: 2 - u, -1, -1),
        (lambda: u * 2, 6, 2),
        (lambda: 2 * u, 6, 2),
        (lambda: u / 2, 1.5, 0.5),
        (lambda: 2 / u, 2 / 3, -2 / 9),
        (lambda: u**2, 9, 6),
        (lambda: abs(-u), 3, 1),
    ]:
        r = op()
        assert_allclose(r.f, f)
        assert_allclose(r.d("x"), dx)


def test_in_place_operators():
    def fresh():
        return hj.SScalar(f=3, d={"x": 1}), hj.SScalar(f=4, d={"y": 2})

    for op, f, dx, dy in [
        (lambda a, b: a.__iadd__(b), 7, 1, 2),
        (lambda a, b: a.__isub__(b), -1, 1, -2),
        (lambda a, b: a.__imul__(b), 12, 4, 6),
        (lambda a, b: a.__itruediv__(b), 0.75, 0.25, -0.375),
    ]:
        a, b = fresh()
        r = op(a, b)
        assert_allclose(r.f, f)
        assert_allclose(r.d("x"), dx)
        assert_allclose(r.d("y"), dy)


def test_comparison_uses_the_value_only():
    a = hj.SScalar(f=1, d={"x": 100})
    b = hj.SScalar(f=2, d={"y": -100})
    c = hj.SScalar(f=1)

    assert a < b and a <= b and b > a and b >= a and a != b
    assert not (a == b)

    # equal values compare equal no matter what the derivatives say
    assert a == c and a <= c and a >= c

    assert a < 2 and a == 1 and 2 > a and 1 == a


def test_numpy_aliases_agree():
    u = hj.SScalar(f=0.3, d={"x": 1})
    v = hj.SScalar(f=0.4, d={"y": 1})

    for alias, name in [
        ("arccos", "acos"),
        ("arcsin", "asin"),
        ("arctan", "atan"),
        ("arccosh", "acosh"),
        ("arcsinh", "asinh"),
        ("arctanh", "atanh"),
    ]:
        w = hj.SScalar(f=1.5, d={"x": 1}) if name == "acosh" else u
        assert_allclose(getattr(w, alias)().f, getattr(w, name)().f)
        assert_allclose(getattr(w, alias)().d("x"), getattr(w, name)().d("x"))

    assert_allclose(u.arctan2(v).f, u.atan2(v).f)


def test_no_serialization():
    # Unlike the DDScalar bindings, SScalar has neither pickle nor copy support.
    u = hj.SScalar(f=3, d={"x": 1})

    with pytest.raises(TypeError):
        copy(u)

    with pytest.raises(TypeError):
        pickle.dumps(u)
