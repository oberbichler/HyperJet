import pytest
import hyperjet as hj
import numpy as np
from numpy.testing import assert_equal, assert_allclose

if __name__ == "__main__":
    import os
    import sys

    print(f"pid: {os.getpid()}")
    pytest.main(sys.argv)


# Statically sized scalars back a NumPy dtype, so arrays of them hold their
# data contiguously instead of pointers to Python objects. The dynamic variants
# hold a std::vector and cannot: they stay object arrays.


static_types = [hj.D3Scalar, hj.DD3Scalar]
dynamic_types = [hj.DScalar, hj.DDScalar]


def values(dtype, offset=0.0):
    n = 4 if dtype.order == 1 else 10
    return [float(i) + 1.0 + offset for i in range(n)]


@pytest.mark.parametrize("dtype", static_types)
def test_static_types_have_a_dtype(dtype):
    d = np.dtype(dtype.dtype)

    assert_equal(d.itemsize, len(values(dtype)) * 8)
    assert_equal(np.zeros(3, dtype=d).nbytes, 3 * d.itemsize)


@pytest.mark.parametrize("dtype", dynamic_types)
def test_dynamic_types_have_none(dtype):
    assert not hasattr(dtype, "dtype")

    u = dtype(values(dtype))

    assert_equal(np.array([u, u]).dtype, np.dtype(object))


@pytest.mark.parametrize("dtype", static_types)
def test_arrays_use_the_dtype(dtype):
    u = dtype(values(dtype))

    a = np.array([u, u, u])

    assert_equal(a.dtype, np.dtype(dtype.dtype))
    assert_equal(a.nbytes, 3 * len(values(dtype)) * 8)


@pytest.mark.parametrize("dtype", static_types)
def test_elements_survive_a_roundtrip(dtype):
    u = dtype(values(dtype))

    a = np.array([u])

    assert_equal(type(a[0]), dtype)
    assert_allclose(a[0].data, u.data)

    a[0] = dtype(values(dtype, offset=100.0))

    assert_allclose(a[0].data, values(dtype, offset=100.0))


@pytest.mark.parametrize("dtype", static_types)
def test_conversion_back_to_object(dtype):
    u = dtype(values(dtype))

    a = np.array([u, u])
    o = np.array(a, dtype=object)

    assert_equal(o.dtype, np.dtype(object))
    assert_allclose(o[0].data, u.data)


# The reference has to be the scalar operation, not the same ufunc applied to
# scalars -- that would go through the very loop under test and pass no matter
# what the loop computes.
@pytest.mark.parametrize("dtype", static_types)
@pytest.mark.parametrize(
    "ufunc, reference",
    [
        (np.add, lambda a, b: a + b),
        (np.subtract, lambda a, b: a - b),
        (np.multiply, lambda a, b: a * b),
        (np.divide, lambda a, b: a / b),
        (np.arctan2, lambda a, b: a.atan2(b)),
        (np.hypot, lambda a, b: hj.hypot(a, b)),
    ],
)
def test_binary_ufuncs_match_the_scalars(dtype, ufunc, reference):
    u = dtype(values(dtype))
    v = dtype(values(dtype, offset=0.5))

    r = ufunc(np.array([u, v]), np.array([v, u]))

    assert_allclose(r[0].data, reference(u, v).data)
    assert_allclose(r[1].data, reference(v, u).data)


@pytest.mark.parametrize("dtype", static_types)
@pytest.mark.parametrize(
    "ufunc, reference",
    [
        (np.negative, lambda a: -a),
        (np.positive, lambda a: a),
        (np.absolute, lambda a: a.abs()),
        (np.reciprocal, lambda a: a.reciprocal()),
        (np.sqrt, lambda a: a.sqrt()),
        (np.cbrt, lambda a: a.cbrt()),
        (np.sin, lambda a: a.sin()),
        (np.cos, lambda a: a.cos()),
        (np.tan, lambda a: a.tan()),
        (np.arctan, lambda a: a.atan()),
        (np.sinh, lambda a: a.sinh()),
        (np.cosh, lambda a: a.cosh()),
        (np.tanh, lambda a: a.tanh()),
        (np.arcsinh, lambda a: a.asinh()),
        (np.arccosh, lambda a: a.acosh()),
        (np.exp, lambda a: a.exp()),
        (np.log, lambda a: a.log()),
        (np.log2, lambda a: a.log2()),
        (np.log10, lambda a: a.log10()),
    ],
)
def test_unary_ufuncs_match_the_scalars(dtype, ufunc, reference):
    u = dtype(values(dtype))

    r = ufunc(np.array([u, u]))

    assert_allclose(r[0].data, reference(u).data)


@pytest.mark.parametrize("dtype", static_types)
def test_matmul_and_vecdot(dtype):
    u = dtype(values(dtype))
    v = dtype(values(dtype, offset=0.5))

    a = np.array([u, v])
    expected = u * u + v * v

    assert_allclose((a @ a).data, expected.data)
    assert_allclose(np.vecdot(a, a).data, expected.data)
    assert_allclose(np.matmul(a, a).data, expected.data)


@pytest.mark.parametrize("dtype", static_types)
def test_reductions(dtype):
    u = dtype(values(dtype))
    v = dtype(values(dtype, offset=0.5))

    a = np.array([u, v])

    assert_allclose(np.sum(a).data, (u + v).data)
    assert_allclose(np.add.reduce(a).data, (u + v).data)


# np.dot rejects the dtype: it is not a ufunc but an __array_function__
# dispatcher, which looks at the array type rather than the dtype, and then
# refuses anything that is not a native or an old-style dtype. `@` is the
# replacement and works for object arrays just as well.
@pytest.mark.parametrize("dtype", static_types)
def test_dot_is_not_supported(dtype):
    u = dtype(values(dtype))

    a = np.array([u, u])

    with pytest.raises(TypeError):
        np.dot(a, a)

    assert_allclose((a @ a).data, np.dot(np.array(a, dtype=object), a).data)
