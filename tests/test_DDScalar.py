from typing import Type
import pytest
import hyperjet as hj
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal, assert_allclose
from math import sqrt, cos, sin, tan, acos, asin, atan, pi
from copy import copy, deepcopy

if __name__ == '__main__':
    import os
    import sys
    print(f'pid: {os.getpid()}')
    pytest.main(sys.argv)


@pytest.fixture
def u():
    a = 3
    b = 5
    return hj.DD2Scalar([75, b**2, 2 * a * b, 0, 2 * b, 2 * a])


@pytest.fixture
def v():
    a = 3
    b = 5
    return hj.DD2Scalar([225, 2 * a * b**2, 2 * a**2 * b, 2 * b**2, 4 * a * b, 2 * a**2])


@pytest.fixture
def x():
    a = 3
    b = 5
    return hj.DDScalar([75, b**2, 2 * a * b, 0, 2 * b, 2 * a])


@pytest.fixture
def y():
    a = 3
    b = 5
    return hj.DDScalar([225, 2 * a * b**2, 2 * a**2 * b, 2 * b**2, 4 * a * b, 2 * a**2])


@pytest.fixture
def sample_trig():
    u = hj.DD2Scalar([1 / 2, 1 / 6, 1 / 5, 0, 1 / 15, 1 / 25])
    v = hj.DD2Scalar([-1 / 4, -1 / 6, -1 / 10, -1 / 18, -1 / 15, -1 / 50])

    return u, v


def check(jet, f, g, h):
    assert_allclose(jet.f, f)
    assert_array_almost_equal(jet.g, g)
    assert_array_almost_equal(jet.h, h)


# init


def test_init_by_value():
    # static size
    u = hj.DD2Scalar(f=1, size=2)
    assert_equal(u.size, 2)
    assert_allclose(u.data, [1, 0, 0, 0, 0, 0])

    # dynamic size
    u = hj.DDScalar(f=1, size=2)
    assert_equal(u.size, 2)
    assert_allclose(u.data, [1, 0, 0, 0, 0, 0])

    # static size with invalid size argument throws
    with pytest.raises(RuntimeError):
        hj.DD2Scalar(f=1, size=3)

    # dynamic size without argument has size=0
    u = hj.DDScalar(f=1)
    assert_equal(u.size, 0)
    assert_allclose(u.data, [1])


def test_init_by_data():
    # static size
    u = hj.DD2Scalar([1, 2, 3, 4, 5, 6])
    assert_equal(u.size, 2)
    assert_allclose(u.data, [1, 2, 3, 4, 5, 6])

    # dynamic size
    u = hj.DDScalar([1, 2, 3, 4, 5, 6])
    assert_equal(u.size, 2)
    assert_allclose(u.data, [1, 2, 3, 4, 5, 6])

    # static size with invalid data length throws
    with pytest.raises(TypeError):
        hj.DD2Scalar([1, 2, 3, 4, 5])

    # dynamic size with invalid data length throws
    with pytest.raises(RuntimeError):
        hj.DDScalar([1, 2, 3, 4, 5])


def test_empty():
    # static size
    u = hj.DD2Scalar.empty()
    assert(u.size == 2)

    # static size with same size argument
    u = hj.DD2Scalar.empty(size=2)
    assert(u.size == 2)

    # static size with different size argument throws
    with pytest.raises(RuntimeError):
        hj.DD2Scalar.empty(size=3)

    # dynamic size without size argument -> scalar
    u = hj.DDScalar.empty()
    assert(u.size == 0)

    # dynamic size with size argument
    u = hj.DDScalar.empty(size=2)
    assert(u.size == 2)


def test_zero():
    # static size
    u = hj.DD2Scalar.zero()
    assert(u.size == 2)

    # static size with same size argument
    u = hj.DD2Scalar.zero(size=2)
    assert(u.size == 2)

    # static size with different size argument throws
    with pytest.raises(RuntimeError):
        hj.DD2Scalar.zero(size=3)

    # dynamic size without size argument -> scalar
    u = hj.DDScalar.zero()
    assert(u.size == 0)

    # dynamic size with size argument
    u = hj.DDScalar.zero(size=2)
    assert(u.size == 2)


def test_constant():
    # static size
    u = hj.DD2Scalar.constant(f=3.5)
    assert(u.f == 3.5)
    assert(u.size == 2)

    # static size with same size argument
    u = hj.DD2Scalar.constant(f=3.5, size=2)
    assert(u.f == 3.5)
    assert(u.size == 2)

    # static size with different size argument throws
    with pytest.raises(RuntimeError):
        hj.DD2Scalar.constant(f=3.5, size=3)

    # dynamic size without size argument -> scalar
    u = hj.DDScalar.constant(f=3.5)
    assert(u.f == 3.5)
    assert(u.size == 0)

    # dynamic size with size argument
    u = hj.DDScalar.constant(f=3.5, size=2)
    assert(u.f == 3.5)
    assert(u.size == 2)


def test_variable():
    # static size
    u = hj.DD2Scalar.variable(i=1, f=3.5)
    assert(u.f == 3.5)
    assert(u.size == 2)

    # static size with same size argument
    u = hj.DD2Scalar.variable(i=1, f=3.5, size=2)
    assert(u.f == 3.5)
    assert(u.size == 2)

    # static size with different size argument throws
    with pytest.raises(RuntimeError):
        hj.DD2Scalar.variable(i=1, f=3.5, size=3)

    # dynamic size without size argument throws
    with pytest.raises(TypeError):
        hj.DDScalar.variable(i=1, f=3.5)

    # dynamic size with size argument
    u = hj.DDScalar.variable(i=1, f=3.5, size=2)
    assert(u.f == 3.5)
    assert(u.size == 2)


def test_variables():
    # static size
    u = hj.DD2Scalar.variables([3, 5])
    assert_equal(len(u), 2)
    assert_allclose(u[0].is_dynamic, False)
    assert_allclose(u[1].is_dynamic, False)
    assert_allclose(u[0].data, [3, 1, 0, 0, 0, 0])
    assert_allclose(u[1].data, [5, 0, 1, 0, 0, 0])

    # dynamic size
    u = hj.DDScalar.variables([3, 5])
    assert_equal(len(u), 2)
    assert_allclose(u[0].is_dynamic, True)
    assert_allclose(u[1].is_dynamic, True)
    assert_allclose(u[0].data, [3, 1, 0, 0, 0, 0])
    assert_allclose(u[1].data, [5, 0, 1, 0, 0, 0])


# serialization


def test_copy(u):
    v = copy(u)
    assert_equal(v.data, u.data)

    v = deepcopy(u)
    assert_equal(v.data, u.data)


# get / set


def test_is_dynamic(u, x):
    assert_equal(u.is_dynamic, False)
    assert_equal(x.is_dynamic, True)


def test_size(u, x):
    assert_equal(u.size, 2)
    assert_equal(x.size, 2)


def test_resize(u, x):
    # static throws
    with pytest.raises(AttributeError):
        u.resize(3)

    x.resize(3)
    assert_equal(x.size, 3)


# operations


class UnaryOperation:
    def __init__(self, u, expected):
        self.u = u
        self.expected_u = np.array(u.data)
        self.expected_r = np.array(expected)

    def check(self):
        assert_allclose(self.u.data, self.expected_u, atol=1e-15)
        assert_allclose(self.r.data, self.expected_r, atol=1e-15)


class Sqrt(UnaryOperation):
    def __init__(self, u):
        super().__init__(u, [5 * np.sqrt(3), 5 * np.sqrt(3) / 6, np.sqrt(3), -5 * np.sqrt(3) / 36, np.sqrt(3) / 6, 0])
        self.r = np.sqrt(u)


class Cos(UnaryOperation):
    def __init__(self, u):
        super().__init__(u, [np.cos(75), -25 * np.sin(75), -30 * np.sin(75), -625 * np.cos(75), -750 * np.cos(75) - 10 * np.sin(75), -900 * np.cos(75) - 6 * np.sin(75)])
        self.r = np.cos(u)


class Sin(UnaryOperation):
    def __init__(self, u):
        super().__init__(u, [np.sin(75), 25 * np.cos(75), 30 * np.cos(75), -625 * np.sin(75), 10 * np.cos(75) - 750 * np.sin(75), 6 * np.cos(75) - 900 * np.sin(75)])
        self.r = np.sin(u)


class Tan(UnaryOperation):
    def __init__(self, u):
        super().__init__(u, [np.tan(75), 25 * np.tan(75)**2 + 25, 30 * np.tan(75)**2 + 30, 1250 * (np.tan(75) ** 2 + 1) * np.tan(75), 10 * (150 * np.tan(75) + 1) * (np.tan(75)**2 + 1), 6 * (300 * np.tan(75) + 1) * (np.tan(75)**2 + 1)])
        self.r = np.tan(u)


class BinaryOperation:
    def __init__(self, u, v, expected):
        self.u = u
        self.v = v
        self.expected_u = np.array(u.data)
        self.expected_v = np.array(v.data)
        self.expected_r = np.array(expected)

    def check(self):
        assert_allclose(self.u.data, self.expected_u, atol=1e-15)
        assert_allclose(self.v.data, self.expected_v, atol=1e-15)
        assert_allclose(self.r.data, self.expected_r, atol=1e-15)


class Add(BinaryOperation):
    def __init__(self, u, v):
        super().__init__(u, v, [300, 175, 120, 50, 70, 24])
        self.r = u + v


class Sub(BinaryOperation):
    def __init__(self, u, v):
        super().__init__(u, v, [-150, -125, -60, -50, -50, -12])
        self.r = u - v


class Mul(BinaryOperation):
    def __init__(self, u, v):
        super().__init__(u, v, [16875, 16875, 13500, 11250, 13500, 8100])
        self.r = u * v


class Div(BinaryOperation):
    def __init__(self, u, v):
        super().__init__(u, v, [1 / 3, -1 / 9, 0, 2 / 27, 0, 0])
        self.r = u / v


class BinaryIOperation:
    def __init__(self, u, v, expected):
        self.u = u
        self.v = v
        self.expected_u = np.array(expected)
        self.expected_v = np.array(v.data)

    def check(self):
        assert_allclose(self.u.data, self.expected_u, atol=1e-15)
        assert_allclose(self.v.data, self.expected_v, atol=1e-15)


class IAdd(BinaryIOperation):
    def __init__(self, u, v):
        super().__init__(u, v, [300, 175, 120, 50, 70, 24])
        u += v


class ISub(BinaryIOperation):
    def __init__(self, u, v):
        super().__init__(u, v, [-150, -125, -60, -50, -50, -12])
        u -= v


class IMul(BinaryIOperation):
    def __init__(self, u, v):
        super().__init__(u, v, [16875, 16875, 13500, 11250, 13500, 8100])
        u *= v


class IDiv(BinaryIOperation):
    def __init__(self, u, v):
        super().__init__(u, v, [1 / 3, -1 / 9, 0, 2 / 27, 0, 0])
        u /= v


u_operations = [Sqrt]


@pytest.mark.parametrize('operation', u_operations, ids=[o.__name__ for o in u_operations])
def test_unary_operation_static(u, operation):
    # operation with same static size
    r = operation(u)

    r.check()


@pytest.mark.parametrize('operation', u_operations, ids=[o.__name__ for o in u_operations])
def test_binary_operation_dynamic(x, operation):
    # operation with same dynamic size
    r = operation(x)

    r.check()


b_operations = [Add, IAdd, Sub, ISub, Mul, IMul, Div, IDiv]


@pytest.mark.parametrize('operation', b_operations, ids=[o.__name__ for o in b_operations])
def test_binary_operation_static(u, v, operation):
    # operation with same static size
    r = operation(u, v)

    r.check()

    # operation with different static size throws
    with pytest.raises(TypeError):
        operation(u, hj.DD3Scalar())

    # operation with dynamic size throws
    with pytest.raises(TypeError):
        operation(u, hj.DDScalar(size=2))


@pytest.mark.parametrize('operation', b_operations, ids=[o.__name__ for o in b_operations])
def test_binary_operation_dynamic(x, y, operation):
    # operation with same dynamic size
    r = operation(x, y)

    r.check()

    # operation with different dynamic size throws
    with pytest.raises(RuntimeError):
        x + hj.DDScalar(size=3)

    # operation with static size throws
    with pytest.raises(TypeError):
        x + hj.DD2Scalar()
