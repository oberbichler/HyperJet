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


def test_init():
    u = hj.SScalar(f=1.2)
    assert_equal(len(u), 0)
    assert_allclose(u.f, 1.2)
    assert_allclose(u.d('x'), 0)
    assert_allclose(u.d('y'), 0)
    assert_allclose(u.d('z'), 0)
    
    u = hj.SScalar(f=1.2, d={'x': 1, 'y': 2})
    assert_equal(len(u), 2)
    assert_allclose(u.f, 1.2)
    assert_allclose(u.d('x'), 1)
    assert_allclose(u.d('y'), 2)
    assert_allclose(u.d('z'), 0)


def test_mul():
    u = hj.SScalar(f=1, d={'x': 1})
    v = hj.SScalar(f=1, d={'y': 1})
    r = u + v**2
    assert_equal(len(r), 2)
    assert_allclose(r.f, 2)
    assert_allclose(r.d('x'), 1)
    assert_allclose(r.d('y'), 2)
    assert_allclose(r.d('z'), 0)
