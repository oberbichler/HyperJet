## Installation

It is recommended to install `HyperJet` by using `pip`:

```
pip install git+https://github.com/oberbichler/HyperJet
```

## Compute derivatives

Consider the following function _f(x, y)_ in Python:

```python
def f(x, y):
    return (x * y) / (x - y)
```

It is very simple to compute the value of this function for _x=3_ and _y=6_:

```python
x, y = 3, 6                             # x and y are simple numbers

result = f(x, y)    # == -6
```

After importing `HyperJet` into your Python-script

```python
import hyperjet as hj
```

it becomes also very easy to compute the derivative of the result with respect to _x_ and _y_. You just need to express your variables with `Jet`s:

```python
x, y = hj.Jet.variables([3, 6])         # x and y are Jets

result = f(x, y)                        # the result is also a Jet

result.f            # == -6             (Function value)
result.g            # == [-4,  1]       (Gradient)
```

By using a `HyperJet` you can also compute the second derivative:

```python
x, y = hj.HyperJet.variables([3, 6])    # x and y are HyperJets

result = f(x, y)                        # the result is also a HyperJet

result.f            # == -6                             (Function value)
result.g            # == [-4,  1]                       (Gradient)
result.h            # == [[-2.66666667,  1.33333333],   (Hessian)
                    #     [ 1.33333333, -0.66666667]]
```


## Linear Algebra with NumPy

After importing `numpy`

```python
import numpy as np
```

you can use the `numpy` functions to do some linear algebra:

```python
x, y = 3, 4

vector = np.array([x, y])           # is a vector of numbers

length = np.linalg.norm(vector)     # == 5

unit_vector = vector / length       # == [0.6, 0.8]
```

The `HyperJet` library works together with `numpy`. You just have to change the inputs to compute the derivatives:

```python
x, y = hj.HyperJet.variables([3, 4])    # x and y are HyperJets

vector = np.array([x, y])           # is a vector of HyperJets

length = np.linalg.norm(vector)     # is a HyperJet

length.h                            # == [[ 0.128, -0.096],
                                    #     [-0.096,  0.072]]

unit_vector = vector / length       # is a vector of HyperJets

unit_vector[0].h                    # == [[-0.04608,  0.00256],
                                    #     [ 0.00256,  0.02208]]
```