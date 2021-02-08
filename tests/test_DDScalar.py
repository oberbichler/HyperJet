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
    def __init__(self, name, u, expected):
        self.name = name
        self.u = np.array(u, dtype=float)
        self.expected_u = np.array(u, dtype=float)
        self.expected_r = np.array(expected, dtype=float)

    def check(self, d):
        self.actual_u = d(self.u)
        self.actual_r = self.compute(self.actual_u)

        assert_allclose(self.actual_u.data, self.expected_u, atol=1e-15)
        assert_allclose(self.actual_r.data, self.expected_r, atol=1e-15)


class Sqrt(UnaryOperation):
    def __init__(self, name, u, expected):
        super().__init__(name, u, expected)

    def compute(self, u):
        return np.sqrt(u)


class Cos(UnaryOperation):
    def __init__(self, name, u, expected):
        super().__init__(name, u, expected)

    def compute(self, u):
        return np.cos(u)


class Sin(UnaryOperation):
    def __init__(self, name, u, expected):
        super().__init__(name, u, expected)

    def compute(self, u):
        return np.sin(u)


class Tan(UnaryOperation):
    def __init__(self, name, u, expected):
        super().__init__(name, u, expected)

    def compute(self, u):
        return np.tan(u)


class ACos(UnaryOperation):
    def __init__(self, name, u, expected):
        super().__init__(name, u, expected)

    def compute(self, u):
        return np.arccos(u)


class ASin(UnaryOperation):
    def __init__(self, name, u, expected):
        super().__init__(name, u, expected)

    def compute(self, u):
        return np.arcsin(u)


class ATan(UnaryOperation):
    def __init__(self, name, u, expected):
        super().__init__(name, u, expected)

    def compute(self, u):
        return np.arctan(u)


class Pow(UnaryOperation):
    def __init__(self, name, u, c, expected):
        super().__init__(name, u, expected)
        self.c = c

    def compute(self, u):
        return np.power(u, 3)


def isnumber(value):
    if type(value) == float:
        return True
    if type(value) == int:
        return True
    return False


class BinaryOperation:
    def __init__(self, name, u, v, expected):
        self.name = name
        self.u = u
        self.v = v
        self.expected_u = np.array(u)
        self.expected_v = np.array(v)
        self.expected_r = np.array(expected)

    def check(self, d):
        self.actual_u = self.u if isnumber(self.u) else d(self.u)
        self.actual_v = self.v if isnumber(self.v) else d(self.v)
        self.actual_r = self.compute(self.actual_u, self.actual_v)

        assert_allclose(self.actual_u if isnumber(self.u) else self.actual_u.data, self.expected_u, atol=1e-15)
        assert_allclose(self.actual_v if isnumber(self.v) else self.actual_v.data, self.expected_v, atol=1e-15)
        assert_allclose(self.actual_r.data, self.expected_r, atol=1e-15)


class Add(BinaryOperation):
    def __init__(self, name, u, v, expected):
        super().__init__(name, u, v, expected)

    def compute(self, u, v):
        return u + v


class Sub(BinaryOperation):
    def __init__(self, name, u, v, expected):
        super().__init__(name, u, v, expected)

    def compute(self, u, v):
        return u - v


class Mul(BinaryOperation):
    def __init__(self, name, u, v, expected):
        super().__init__(name, u, v, expected)

    def compute(self, u, v):
        return u * v


class Div(BinaryOperation):
    def __init__(self, name, u, v, expected):
        super().__init__(name, u, v, expected)

    def compute(self, u, v):
        return u / v


class IBinaryOperation:
    def __init__(self, name, u, v, expected):
        self.name = name
        self.u = u
        self.v = v
        self.expected_u = np.array(expected)
        self.expected_v = np.array(v)

    def check(self, d):
        self.actual_u = d(self.u)
        self.actual_v = d(self.v)
        self.actual_r = self.compute(self.actual_u, self.actual_v)

        assert_allclose(self.actual_u.data, self.expected_u, atol=1e-15)
        assert_allclose(self.actual_v.data, self.expected_v, atol=1e-15)


class IAdd(IBinaryOperation):
    def __init__(self, name, u, v, expected):
        super().__init__(name, u, v, expected)

    def compute(self, u, v):
        u += v


class ISub(IBinaryOperation):
    def __init__(self, name, u, v, expected):
        super().__init__(name, u, v, expected)

    def compute(self, u, v):
        u -= v


class IMul(IBinaryOperation):
    def __init__(self, name, u, v, expected):
        super().__init__(name, u, v, expected)

    def compute(self, u, v):
        u *= v


class IDiv(IBinaryOperation):
    def __init__(self, name, u, v, expected):
        super().__init__(name, u, v, expected)

    def compute(self, u, v):
        u /= v


u_operations = [
    Sqrt(
        name='sqrt',
        u=[75, 25, 30, 0, 10, 6],
        expected=[5 * np.sqrt(3), 5 * np.sqrt(3) / 6, np.sqrt(3), -5 * np.sqrt(3) / 36, np.sqrt(3) / 6, 0],
    ),
    Cos(
        name='cos',
        u=[75, 25, 30, 0, 10, 6],
        expected=[np.cos(75), -25 * np.sin(75), -30 * np.sin(75), -625 * np.cos(75), -750 * np.cos(75) - 10 * np.sin(75), -900 * np.cos(75) - 6 * np.sin(75)],
    ),
    Sin(
        name='sin',
        u=[75, 25, 30, 0, 10, 6],
        expected=[np.sin(75), 25 * np.cos(75), 30 * np.cos(75), -625 * np.sin(75), 10 * np.cos(75) - 750 * np.sin(75), 6 * np.cos(75) - 900 * np.sin(75)],
    ),
    Tan(
        name='tan',
        u=[75, 25, 30, 0, 10, 6],
        expected=[np.tan(75), 25 * np.tan(75)**2 + 25, 30 * np.tan(75)**2 + 30, 1250 * (np.tan(75)**2 + 1) * np.tan(75), 10 * (150 * np.tan(75) + 1) * (np.tan(75)**2 + 1), 6 * (300 * np.tan(75) + 1) * (np.tan(75)**2 + 1)],
    ),
    ACos(
        name='acos',
        u=[0.6, 0.2, -0.12, 0., -0.04, 0.048],
        expected=[0.927295218001612, -0.25, 0.15, -0.046875, 0.078125, -0.076875],
    ),
    ASin(
        name='asin',
        u=[0.6, 0.2, -0.12, 0., -0.04, 0.048],
        expected=[0.643501108793284, 0.25, -0.15, 0.046875, -0.078125, 0.076875],
    ),
    ATan(
        name='atan',
        u=[0.6, 0.2, -0.12, 0., -0.04, 0.048],
        expected=[0.540419500270584, 0.147058823529412, -0.0882352941176471, -0.0259515570934256, -0.0138408304498270, 0.0259515570934256],
    ),
    Pow(
        name='pow',
        u=[75, 25, 30, 0, 10, 6],
        c=3,
        expected=[421875, 421875, 506250, 281250, 506250, 506250],
    ),
]


@pytest.mark.parametrize('operation', u_operations, ids=[o.name for o in u_operations])
def test_unary_operation_static(operation):
    # operation with same static size
    operation.check(hj.DD2Scalar)


@pytest.mark.parametrize('operation', u_operations, ids=[o.name for o in u_operations])
def test_unary_operation_dynamic(operation):
    # operation with same dynamic size
    operation.check(hj.DDScalar)


b_operations = [
    Add(
        name='add dd+dd',
        u=[75, 25, 30, 0, 10, 6],
        v=[225, 150, 90, 50, 60, 18],
        expected=[300, 175, 120, 50, 70, 24],
    ),
    Add(
        name='add dd+s',
        u=[75, 25, 30, 0, 10, 6],
        v=3,
        expected=[78, 25, 30, 0, 10, 6],
    ),
    Add(
        name='add s+dd',
        u=3,
        v=[75, 25, 30, 0, 10, 6],
        expected=[78, 25, 30, 0, 10, 6],
    ),
    Sub(
        name='sub dd-dd',
        u=[75, 25, 30, 0, 10, 6],
        v=[225, 150, 90, 50, 60, 18],
        expected=[-150, -125, -60, -50, -50, -12],
    ),
    Sub(
        name='sub s-dd',
        u=3,
        v=[75, 25, 30, 0, 10, 6],
        expected=[-72, -25, -30, 0, -10, -6],
    ),
    Sub(
        name='sub dd-s',
        u=[75, 25, 30, 0, 10, 6],
        v=3,
        expected=[72, 25, 30, 0, 10, 6],
    ),
    Mul(
        name='mul dd*dd',
        u=[75, 25, 30, 0, 10, 6],
        v=[225, 150, 90, 50, 60, 18],
        expected=[16875, 16875, 13500, 11250, 13500, 8100],
    ),
    Mul(
        name='mul s*dd',
        u=3,
        v=[75, 25, 30, 0, 10, 6],
        expected=[225, 75, 90, 0, 30, 18],
    ),
    Mul(
        name='mul dd*s',
        u=[75, 25, 30, 0, 10, 6],
        v=3,
        expected=[225, 75, 90, 0, 30, 18],
    ),
    Div(
        name='div dd/dd',
        u=[75, 25, 30, 0, 10, 6],
        v=[225, 150, 90, 50, 60, 18],
        expected=[1 / 3, -1 / 9, 0, 2 / 27, 0, 0],
    ),
    Div(
        name='div s/dd',
        u=3,
        v=[75, 25, 30, 0, 10, 6],
        expected=[1 / 25, -1 / 75, -2 / 125, 2 / 225, 2 / 375, 6 / 625],
    ),
    Div(
        name='div dd/s',
        u=[75, 25, 30, 0, 10, 6],
        v=3,
        expected=[25, 25 / 3, 10, 0, 10 / 3, 2],
    ),
]


@pytest.mark.parametrize('operation', b_operations, ids=[o.name for o in b_operations])
def test_binary_operation_static(operation):
    # operation with same static size
    operation.check(hj.DD2Scalar)

    if isnumber(operation.actual_u) or isnumber(operation.actual_v):
        return

    # operation with different static size throws
    with pytest.raises(TypeError):
        operation.compute(operation.actual_u, hj.DD3Scalar())

    # operation with dynamic size throws
    with pytest.raises(TypeError):
        operation.compute(operation.actual_u, hj.DDScalar(size=2))


@pytest.mark.parametrize('operation', b_operations, ids=[o.name for o in b_operations])
def test_binary_operation_dynamic(operation):
    # operation with same dynamic size
    operation.check(hj.DDScalar)

    if isnumber(operation.actual_u) or isnumber(operation.actual_v):
        return

    # operation with different dynamic size throws
    with pytest.raises(RuntimeError):
        operation.compute(operation.actual_u, hj.DDScalar(size=3))

    # operation with static size throws
    with pytest.raises(TypeError):
        operation.compute(operation.actual_u, hj.DD2Scalar())


i_operations = [
    IAdd(
        name='iadd',
        u=[75, 25, 30, 0, 10, 6],
        v=[225, 150, 90, 50, 60, 18],
        expected=[300, 175, 120, 50, 70, 24],
    ),
    ISub(
        name='isub',
        u=[75, 25, 30, 0, 10, 6],
        v=[225, 150, 90, 50, 60, 18],
        expected=[-150, -125, -60, -50, -50, -12],
    ),
    IMul(
        name='imul',
        u=[75, 25, 30, 0, 10, 6],
        v=[225, 150, 90, 50, 60, 18],
        expected=[16875, 16875, 13500, 11250, 13500, 8100],
    ),
    IDiv(
        name='idiv',
        u=[75, 25, 30, 0, 10, 6],
        v=[225, 150, 90, 50, 60, 18],
        expected=[1 / 3, -1 / 9, 0, 2 / 27, 0, 0],
    ),
]


@pytest.mark.parametrize('operation', i_operations, ids=[o.name for o in i_operations])
def test_incemental_operation_static(operation):
    # operation with same static size
    operation.check(hj.DD2Scalar)

    if isnumber(operation.actual_u) or isnumber(operation.actual_v):
        return

    # operation with different static size throws
    with pytest.raises(TypeError):
        operation.compute(operation.actual_u, hj.DD3Scalar())

    # operation with dynamic size throws
    with pytest.raises(TypeError):
        operation.compute(operation.actual_u, hj.DDScalar(size=2))


@pytest.mark.parametrize('operation', i_operations, ids=[o.name for o in i_operations])
def test_incemental_operation_dynamic(operation):
    # operation with same dynamic size
    operation.check(hj.DDScalar)

    if isnumber(operation.actual_u) or isnumber(operation.actual_v):
        return

    # operation with different dynamic size throws
    with pytest.raises(RuntimeError):
        operation.compute(operation.actual_u, hj.DDScalar(size=3))

    # operation with static size throws
    with pytest.raises(TypeError):
        operation.compute(operation.actual_u, hj.DD2Scalar())
