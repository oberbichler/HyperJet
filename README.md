![HyperJet](https://github.com/oberbichler/HyperJet/raw/main/docs/HyperJet.png?raw=true)

<p align="center"><b>HyperJet — Algorithmic Differentiation with Hyper-Dual Numbers for C++ and Python</b></p>

---

A header-only C++23 library for algorithmic differentiation with hyper-dual numbers. Supports first- and second-order derivatives with both dense indexed variables (`DDScalar`) and sparse named variables (`SScalar`). Includes an extensive Python interface via [pybind11](https://github.com/pybind/pybind11).

[![PyPI](https://img.shields.io/pypi/v/hyperjet)](https://pypi.org/project/hyperjet)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyperjet)
![C++ Standard](https://img.shields.io/badge/C%2B%2B-23-blue)
[![Test Python](https://github.com/oberbichler/HyperJet/actions/workflows/test-python.yml/badge.svg)](https://github.com/oberbichler/HyperJet/actions/workflows/test-python.yml)
[![Test C++](https://github.com/oberbichler/HyperJet/actions/workflows/test-cpp.yml/badge.svg)](https://github.com/oberbichler/HyperJet/actions/workflows/test-cpp.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/oberbichler/HyperJet/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/165487832.svg)](https://zenodo.org/badge/latestdoi/165487832)

## Installation

```
pip install hyperjet
```

**Requirements:** Python ≥ 3.11, NumPy ≥ 2.0

## Quickstart

Import the module:

```python
import hyperjet as hj
```

Create a set of hyper-dual variables, e.g. `x=3` and `y=6`:

```python
x, y = hj.variables([3, 6])
```

`x` and `y` are second-order hyper-dual numbers, indicated by the `hj` postfix:

```python
x
>>> 3hj
```

Get the value as a plain `float`:

```python
x.f
>>> 3
```

Get the first-order derivatives (gradient) as a NumPy array:

```python
x.g  # [dx/dx, dx/dy]
>>> array([1., 0.])
```

Get the second-order derivatives (Hessian matrix):

```python
x.hm()  # [[d²x/dx², d²x/(dx·dy)],
         #  [d²x/(dx·dy), d²x/dy²]]
>>> array([[0., 0.],
           [0., 0.]])
```

For a single variable these derivatives are trivial. Now do a computation:

```python
f = (x * y) / (x - y)
f
>>> -6hj
```

The result is again a hyper-dual number carrying all derivatives:

```python
f.g  # [df/dx, df/dy]
>>> array([-4.,  1.])

f.hm()  # [[d²f/dx², d²f/(dx·dy)],
         #  [d²f/(dx·dy), d²f/dy²]]
>>> array([[-2.66666667,  1.33333333],
           [ 1.33333333, -0.66666667]])
```

## Types

HyperJet provides two families of scalar types:

### `DDScalar` — Dense dual numbers with indexed variables

Stores derivatives in dense arrays. Supports first-order (gradient only) and second-order (gradient + Hessian) derivatives. Available in **static** variants with a compile-time-fixed number of variables, and a **dynamic** variant for arbitrary sizes.

| Python type | Order | Variables | C++ type |
|-------------|-------|-----------|----------|
| `DScalar` | 1 | dynamic | `DDScalar<1, double>` |
| `DDScalar` | 2 | dynamic | `DDScalar<2, double>` |
| `D3Scalar` | 1 | 3 (static) | `DDScalar<1, double, 3>` |
| `DD3Scalar` | 2 | 3 (static) | `DDScalar<2, double, 3>` |

Static variants (`D0Scalar`–`D16Scalar`, `DD0Scalar`–`DD16Scalar`) avoid heap allocation and enable better compiler optimization. The dynamic variants (`DScalar`, `DDScalar`) accept any number of variables at runtime.

The convenience function `hj.variables(values, order=2)` automatically selects the appropriate static type when the number of variables is ≤ 16, and falls back to the dynamic variant otherwise.

To change the number of variables of a dynamic scalar, use `pad_left(new_size)` or `pad_right(new_size)`. They insert the new variables before or after the existing ones and remap the gradient and Hessian accordingly. To start from scratch instead, create a new instance with `empty(size)` or `zero(size)`.

First-order types store only a value and a gradient. The Hessian accessors (`h`, `set_h`, `hm`, `set_hm`) therefore exist on second-order types only — in C++ they are constrained via `requires (order() == 2)`, and in Python they are absent from first-order classes.

### `SScalar` — Sparse dual numbers with named variables

Stores first-order derivatives in a sparse map keyed by variable name (string). Useful when variables are identified by name rather than index, or when only a small subset of derivatives is non-zero.

```python
x = hj.SScalar.variable("x", 3.0)
y = hj.SScalar.variable("y", 6.0)

f = (x * y) / (x - y)
f.f        # value
>>> -6.0
f.d("x")   # df/dx
>>> -4.0
f.d("y")   # df/dy
>>> 1.0
```

`SScalar` is **first order only** — there is no Hessian. Where `DDScalar` offers `h`, `hm` and the second-order factories, `SScalar` has a value and a gradient keyed by name. The variable set does not have to be known in advance: an operation takes the union of the names of its operands, and `d` returns zero for a name the scalar has never seen.

```python
u = hj.SScalar.variable("x", 2.0)
v = hj.SScalar.variable("y", 3.0)

len(u * v)          # the product knows both names
>>> 2
(u * v).d("z")      # an unknown name is zero, not an error
>>> 0.0
(u - u).size        # names survive even where the derivative cancels
>>> 1
```

Two further differences from `DDScalar` worth knowing:

- **No serialization.** `copy`, `deepcopy` and `pickle` raise `TypeError`; the `DDScalar` types support all three.
- **The order of terms in `repr` is unspecified**, because the derivatives live in an unordered map. Only the value comes first.

`eval(d)` contracts the gradient with a displacement given per name. Names missing from the displacement contribute nothing, and names the scalar does not know are ignored:

```python
u = hj.SScalar(f=3, d={"x": 1, "y": 6})

u.eval({"x": 0.5, "y": -0.25})
>>> 2.0
u.eval({"x": 0.5})          # y contributes nothing
>>> 3.5
u.eval({"w": 100.0})        # w is unknown and ignored
>>> 3.0
```

## Validation

Arguments coming from Python are validated. Out-of-range Hessian indices raise `IndexError`; inconsistent sizes raise `RuntimeError` — `eval` with the wrong number of values, `set_hm` with a mismatched shape, a negative size, a variable index outside the gradient, or padding below the current size.

In C++ the element accessors `g(i)`, `h(i)` and `h(i, j)` treat their indices as a precondition: like `std::vector::operator[]` they are only checked via `assert` in debug builds, so they stay free of branches in hot loops. Everything that creates, resizes or evaluates a scalar validates its arguments and throws `std::runtime_error`. Define `HYPERJET_NO_EXCEPTIONS` to fall back to `assert` throughout.

## NumPy Integration

HyperJet scalars work with NumPy for vector and matrix operations.

Compute the normalized cross product of `u = [1, 2, 2]` and `v = [4, 1, -1]` with full second-order derivatives:

```python
import numpy as np

variables = hj.DDScalar.variables([1, 2,  2,
                                   4, 1, -1])

u = np.array(variables[:3])  # [1hj, 2hj,  2hj]
v = np.array(variables[3:])  # [4hj, 1hj, -1hj]

normal = np.cross(u, v)
normal /= np.linalg.norm(normal)
normal
>>> array([-0.331042hj, 0.744845hj, -0.579324hj], dtype=object)
```

The result is a three-dimensional NumPy array of hyper-dual numbers. Extract value and derivatives from any component:

```python
normal[0].f
>>> -0.3310423554409472

normal[0].g
>>> array([ 0.00453483, -0.01020336,  0.00793595,  0.07255723, -0.16325376, 0.12697515])

normal[0].hm()
>>> array([[ 0.00434846, -0.01091775,  0.00647611, -0.0029818 , -0.01143025, -0.02335746],
           [-0.01091775,  0.02711578, -0.01655522,  0.00444165,  0.03081974,  0.04858632],
           [ 0.00647611, -0.01655522,  0.0093492 , -0.00295074, -0.02510461, -0.03690759],
           [-0.0029818 ,  0.00444165, -0.00295074, -0.02956956,  0.03025289, -0.01546811],
           [-0.01143025,  0.03081974, -0.02510461,  0.03025289,  0.01355789, -0.02868433],
           [-0.02335746,  0.04858632, -0.03690759, -0.01546811, -0.02868433,  0.03641839]])
```

### A real dtype instead of `dtype=object`

Statically sized scalars back a NumPy dtype, so an array of them stores its data contiguously rather than as pointers to Python objects. NumPy picks it up on its own:

```python
a = np.array(hj.variables([1.0, 2.0, 3.0]))
a.dtype
>>> DD3ScalarDType
a.nbytes        # 3 x 80 bytes, contiguous
>>> 240
```

Arithmetic and the mathematical functions then run as compiled loops instead of a Python object loop. Measured over 10 000 elements of `DD3Scalar`:

| | `dtype=object` | dtype | |
|---|---|---|---|
| `a + b` | 349.5 ns | 2.5 ns | 138× |
| `a * b` | 354.3 ns | 3.9 ns | 92× |
| `np.sqrt(a)` | 349.2 ns | 2.8 ns | 123× |
| `np.sin(a)` | 366.4 ns | 7.0 ns | 52× |
| `np.sum(a)` | 353.1 ns | 3.8 ns | 94× |
| `a @ b` | 673.0 ns | 2.7 ns | 246× |

The **dynamic** variants (`DScalar`, `DDScalar`) hold a `std::vector`, so they have no fixed element size and stay object arrays.

Two functions do not work on these arrays: `np.dot` and `np.linalg.norm`. `np.dot` is not a ufunc but an `__array_function__` dispatcher — it looks at the array type rather than at the dtype, and then rejects anything that is not a native or an old-style dtype. Use `@`, `np.matmul` or `np.vecdot` instead; they compute the same thing and work on object arrays too:

```python
a @ a                       # instead of np.dot(a, a)
np.sqrt(np.vecdot(a, a))    # instead of np.linalg.norm(a)
```

`np.array(a, dtype=object)` converts back at any time, which restores the object behaviour including `np.dot`.

## C++ Usage

HyperJet is a single header-only library. Add `include/` to your include path and use C++23:

```cpp
#include <hyperjet/hyperjet.h>
#include <iostream>

int main() {
    using namespace hyperjet;

    // Create second-order variables with 2 variables: x=3, y=6
    auto [x, y] = DDScalar<2, double, 2>::variables(std::array{3.0, 6.0});

    auto f = (x * y) / (x - y);

    std::cout << "f   = " << f.f() << std::endl;       // -6
    std::cout << "df/dx = " << f.g(0) << std::endl;    // -4
    std::cout << "df/dy = " << f.g(1) << std::endl;    //  1
    std::cout << "d²f/dx² = " << f.h(0, 0) << std::endl;
}
```

### CMake Integration

HyperJet uses [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) for dependency management:

```cmake
CPMAddPackage("gh:oberbichler/HyperJet@2.0.0")
target_link_libraries(my_target hyperjet)
```

The library requires a C++23-capable compiler (GCC ≥ 14, Clang ≥ 18, MSVC ≥ 19.38).

## Supported Functions

Both `DDScalar` and `SScalar` support:

| Category | Functions |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `pow`, `abs`, `reciprocal` |
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Exponential | `exp`, `log`, `log2`, `log10` |
| Other | `sqrt`, `cbrt`, `hypot` |

## Utility Functions

Extract values and derivatives from arrays of hyper-dual numbers:

```python
variables = hj.variables([1.0, 2.0, 3.0])
results = [v ** 2 for v in variables]

hj.f(results)   # array of values
hj.d(results)   # array of gradients
hj.dd(results)  # array of Hessians
```

## Reference

If you use HyperJet, please refer to the official GitHub repository:

```bibtex
@misc{HyperJet,
  author = "Thomas Oberbichler",
  title = "HyperJet",
  howpublished = "\url{http://github.com/oberbichler/HyperJet}",
}
```
