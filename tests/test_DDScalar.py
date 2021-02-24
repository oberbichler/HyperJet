import pytest
import hyperjet as hj
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from copy import copy

if __name__ == '__main__':
    import os
    import sys
    print(f'pid: {os.getpid()}')
    pytest.main(sys.argv)


# initialization


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


# test data


class VariableSet:
    def __init__(self, dtype):
        self.dtype = dtype

    @property
    def u1(self):
        return self.dtype([1.80000000000000, 1.20000000000000, -0.360000000000000, 0.400000000000000, -0.240000000000000, 0.144000000000000])

    @property
    def u2(self):
        return self.dtype([1.41421356237310, -0.353553390593274, 0.353553390593274, -0.0883883476483184, 0.0883883476483184, -0.0883883476483184])

    @property
    def u3(self):
        return self.dtype([-0.227202094693087, -1.16861715705383, 0.350585147116150, -0.0623680359932327, 0.135572126503353, -0.110788667374236])

    @property
    def u4(self):
        return self.dtype([0.973847630878195, -0.272642513631704, 0.0817927540895113, -1.49322142634184, 0.475230679265721, -0.158927754597619])

    @property
    def u5(self):
        return self.dtype([-4.28626167462806, 23.2464469720624, -6.97393409161873, -231.388035688946, 67.0917660094774, -18.7327429845195])

    @property
    def u6(self):
        return self.dtype([2.94217428809568, 3.72896781158072, -1.11869034347422, 5.47972024538469, -2.01681285477348, 0.828781925126886])

    @property
    def u7(self):
        return self.dtype([3.10747317631727, 3.53060914571482, -1.05918274371444, 5.65163108913514, -2.04855024131202, 0.826401621136496])

    @property
    def u8(self):
        return self.dtype([0.946806012846268, 0.124270048845782, -0.0372810146537347, -0.240959761098066, 0.0598609234448416, -0.0105020741027055])

    @property
    def u9(self):
        return self.dtype([1, 2, 3, 4, 5, 6])


static_set = VariableSet(hj.DD2Scalar)
dynamic_set = VariableSet(hj.DDScalar)


def check(act, exp):
    assert_allclose(act.data, exp, atol=1e-16)


# properties


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_init_by_array(ctx):
    u = ctx.dtype(f=1, g=[2, 3], hm=[[4, 5], [5, 6]])
    assert_equal(u.size, 2)
    assert_allclose(u.data, [1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_values(ctx):
    # static size
    u = ctx.dtype([1, 2, 3, 4, 5, 6])

    assert_equal(u.f, 1)

    assert_equal(u.g[0], 2)
    assert_equal(u.g[1], 3)

    assert_equal(u.h(0, 0), 4)
    assert_equal(u.h(0, 1), 5)
    assert_equal(u.h(1, 0), 5)
    assert_equal(u.h(1, 1), 6)

    u.f = 6
    u.g[0] = 5
    u.g[1] = 4
    u.set_h(0, 0, 3)
    u.set_h(0, 1, 2)
    u.set_h(1, 0, 2)
    u.set_h(1, 1, 1)

    assert_equal(u.f, 6)

    assert_equal(u.g[0], 5)
    assert_equal(u.g[1], 4)

    assert_equal(u.h(0, 0), 3)
    assert_equal(u.h(0, 1), 2)
    assert_equal(u.h(1, 0), 2)
    assert_equal(u.h(1, 1), 1)


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_ndarray(ctx):
    # static size
    u = ctx.dtype([1, 2, 3, 4, 5, 6])

    assert_allclose(u.g, [2, 3])

    assert_equal(np.triu(u.hm()), [[4, 5], [0, 6]])

    u.g[:] = [5, 4]
    assert_allclose(u.g, [5, 4])

    u.set_hm([[3, 2], [0, 1]])
    assert_equal(np.triu(u.hm()), [[3, 2], [0, 1]])

    assert_allclose(u.data, [1, 5, 4, 3, 2, 1])

    u.data[:] = [3, 4, 5, 6, 7, 8]
    assert_allclose(u.data, [3, 4, 5, 6, 7, 8])


def test_is_dynamic():
    assert_equal(static_set.u1.is_dynamic, False)
    assert_equal(dynamic_set.u1.is_dynamic, True)


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_size(ctx):
    u = ctx.u1
    assert_equal(u.size, 2)


# resizing


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_resize(ctx):
    u = ctx.u9

    if u.is_dynamic:
        r = u.pad_right(5)
        assert_equal(r.size, 5)
    else:
        with pytest.raises(AttributeError):
            u.pad_right(5)


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_pad_right(ctx):
    u = ctx.u9

    if u.is_dynamic:
        r = u.pad_right(new_size=5)
        check(r, [1, 2, 3, 0, 0, 0, 4, 5, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        with pytest.raises(AttributeError):
            u.pad_right(new_size=5)


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_pad_left(ctx):
    u = ctx.u9

    if u.is_dynamic:
        r = u.pad_left(new_size=5)
        check(r, [1, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6])
    else:
        with pytest.raises(AttributeError):
            u.pad_left(new_size=5)


# serialization


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_copy(ctx):
    u = ctx.u1
    v = copy(u)
    assert_equal(v.data, u.data)


# arithmetic operations


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_negative(ctx):
    r = np.negative(ctx.u1)
    check(r, [-1.8, -1.2, 0.36, -0.4, 0.24, -0.144])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_add(ctx):
    r = ctx.u1 + ctx.u2
    check(r, [3.21421356237310, 0.846446609406726, -0.00644660940672624, 0.311611652351682, -0.151611652351682, 0.0556116523516816])

    r = ctx.u1 + 3
    check(r, [4.80000000000000, 1.20000000000000, -0.360000000000000, 0.400000000000000, -0.240000000000000, 0.144000000000000])

    r = 3 + ctx.u1
    check(r, [4.80000000000000, 1.20000000000000, -0.360000000000000, 0.400000000000000, -0.240000000000000, 0.144000000000000])

    r = ctx.u1
    r += ctx.u2
    check(r, [3.21421356237310, 0.846446609406726, -0.00644660940672624, 0.311611652351682, -0.151611652351682, 0.0556116523516816])

    r = ctx.u1
    r += 3
    check(r, [4.80000000000000, 1.20000000000000, -0.360000000000000, 0.400000000000000, -0.240000000000000, 0.144000000000000])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_sub(ctx):
    r = ctx.u1 - ctx.u2
    check(r, [0.385786437626905, 1.55355339059327, -0.713553390593274, 0.488388347648318, -0.328388347648318, 0.232388347648318])

    r = ctx.u1 - 3
    check(r, [-1.20000000000000, 1.20000000000000, -0.360000000000000, 0.400000000000000, -0.240000000000000, 0.144000000000000])

    r = 3 - ctx.u1
    check(r, [1.20000000000000, -1.20000000000000, 0.360000000000000, -0.400000000000000, 0.240000000000000, -0.144000000000000])

    r = ctx.u1
    r -= ctx.u2
    check(r, [0.385786437626905, 1.55355339059327, -0.713553390593274, 0.488388347648318, -0.328388347648318, 0.232388347648318])

    r = ctx.u1
    r -= 3
    check(r, [-1.20000000000000, 1.20000000000000, -0.360000000000000, 0.400000000000000, -0.240000000000000, 0.144000000000000])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_mul(ctx):
    r = ctx.u1 * ctx.u2
    check(r, [2.54558441227157, 1.06066017177982, 0.127279220613579, -0.441941738241592, 0.371231060122937, -0.210010714012405])

    r = ctx.u1 * 3
    check(r, [5.40000000000000, 3.60000000000000, -1.08000000000000, 1.20000000000000, -0.720000000000000, 0.432000000000000])

    r = 3 * ctx.u1
    check(r, [5.40000000000000, 3.60000000000000, -1.08000000000000, 1.20000000000000, -0.720000000000000, 0.432000000000000])

    r = ctx.u1
    r *= ctx.u2
    check(r, [2.54558441227157, 1.06066017177982, 0.127279220613579, -0.441941738241592, 0.371231060122937, -0.210010714012405])

    r = ctx.u1
    r *= 3
    check(r, [5.40000000000000, 3.60000000000000, -1.08000000000000, 1.20000000000000, -0.720000000000000, 0.432000000000000])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_div(ctx):
    r = ctx.u1 / ctx.u2
    check(r, [1.27279220613579, 1.16672618895780, -0.572756492761104, 0.945755319837007, -0.684125810797985, 0.467751135754901])

    r = ctx.u1 / 3
    check(r, [0.600000000000000, 0.400000000000000, -0.120000000000000, 0.133333333333333, -0.0800000000000000, 0.0480000000000000])

    r = 3 / ctx.u1
    check(r, [1.66666666666667, -1.11111111111111, 0.333333333333333, 1.11111111111111, -0.222222222222222, 0])

    r = ctx.u1
    r /= ctx.u2
    check(r, [1.27279220613579, 1.16672618895780, -0.572756492761104, 0.945755319837007, -0.684125810797985, 0.467751135754901])

    r = ctx.u1
    r /= 3
    check(r, [0.600000000000000, 0.400000000000000, -0.120000000000000, 0.133333333333333, -0.0800000000000000, 0.0480000000000000])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_reciprocal(ctx):
    r = np.reciprocal(ctx.u1)
    check(r, [0.555555555555556, -0.370370370370370, 0.111111111111111, 0.370370370370370, -0.0740740740740741, 0])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_sqrt(ctx):
    r = np.sqrt(ctx.u1)
    check(r, [1.34164078649987, 0.447213595499958, -0.134164078649987, 0, -0.0447213595499958, 0.0402492235949962])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_cbrt(ctx):
    r = np.cbrt(ctx.u1)
    check(r, [1.21644039911468, 0.270320088692151, -0.0810960266076453, -0.0300355654102390, -0.0180213392461434, 0.0216256070953721])


# trigonometric functions


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_cosh(ctx):
    r = np.cosh(ctx.u1)
    check(r, [3.10747317631727, 3.53060914571482, -1.05918274371444, 5.65163108913514, -2.04855024131202, 0.826401621136496])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_sinh(ctx):
    r = np.sinh(ctx.u1)
    check(r, [2.94217428809568, 3.72896781158072, -1.11869034347422, 5.47972024538469, -2.01681285477348, 0.828781925126886])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_tanh(ctx):
    r = np.tanh(ctx.u1)
    check(r, [0.946806012846268, 0.124270048845782, -0.0372810146537347, -0.240959761098066, 0.0598609234448416, -0.0105020741027055])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_acosh(ctx):
    r = np.arccosh(ctx.u6)
    check(r, [1.74207758739717, 1.34764847402516, -0.404294542207549, 0.0492485216214053, -0.149539403888938, 0.125720729608191])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_asinh(ctx):
    r = np.arcsinh(ctx.u7)
    check(r, [1.85189546405173, 1.08154501033879, -0.324463503101638, 0.617782395949683, -0.293489219818784, 0.152939466565963])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_atanh(ctx):
    r = np.arctanh(ctx.u8)
    check(r, [1.8, 1.2, -0.36, 0.4, -0.24, 0.144])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_exp(ctx):
    r = np.exp(ctx.u1)
    check(r, [6.04964746441295, 7.25957695729554, -2.17787308718866, 11.1313513345198, -4.06536309608550, 1.65518354626338])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_log(ctx):
    r = np.log(ctx.u1)
    check(r, [0.587786664902119, 0.666666666666667, -0.200000000000000, -0.222222222222222, 0, 0.0400000000000000])

    r = ctx.u1.log(2)
    check(r, [0.847996906554950, 0.961796693925976, -0.288539008177793, -0.320598897975325, 0, 0.0577078016355585])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_log2(ctx):
    r = np.log2(ctx.u1)
    check(r, [0.847996906554950, 0.961796693925976, -0.288539008177793, -0.320598897975325, 0, 0.0577078016355585])


@pytest.mark.parametrize('ctx', [static_set, dynamic_set], ids=['static', 'dynamic'])
def test_log10(ctx):
    r = np.log10(ctx.u1)
    check(r, [0.255272505103306, 0.289529654602168, -0.0868588963806504, -0.0965098848673893, 0, 0.0173717792761301])
