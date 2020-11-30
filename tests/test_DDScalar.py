import pytest
import hyperjet as hj
import numpy as np
from numpy.testing import assert_almost_equal

if __name__ == '__main__':
    import sys
    import os
    print(f'pid: {os.getpid()}')
    pytest.main(sys.argv)

def test_init():
    dd1 = hj.DD3Scalar(1.0)

    assert_almost_equal(dd1.f, 1.0)

    dd1.f = 5

    assert_almost_equal(dd1.f, 5.0)
