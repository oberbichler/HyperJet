# HyperJet
Automatic differentiation with dual numbers

## Citation

```
@misc{HyperJet,
  author = "Thomas Oberbichler",
  title = "HyperJet",
  howpublished = "\url{http://github.com/oberbichler/HyperJet}",
}
```

## Installation

```
pip install git+https://github.com/oberbichler/HyperJet
```

## How to use it

Import the module.

```python
>>> import hyperjet as hj
```

Create a new `HyperJet` by specifying the size.

```python
>>> a = hj.HyperJet(size=2)
```

You can access the value, the gradiant and the hessian of the `HyperJet` by attributes. By default they are set to zero.

```python
>>> a.f
0.0
>>> a.g
array([0., 0.])
>>> a.g
array([[0., 0.],
       [0., 0.]])
```

You can set the data directly.

```python
>>> a.f = 3
>>> a.g = [1, 0] # or a.g[0] = 1
```

It is also possible to specify the data in the constructor.

```python
>>> b = hj.HyperJet(f=9, g=[0, 1])
```

Doing computations with `HyperJet`s is the same as with `float`s.

```python
>>> a * b
HyperJet<27.0000>
```

Since the result is a `HyperJet` it also stores der derivatives.
