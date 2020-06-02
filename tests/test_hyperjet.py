import hyperjet as hj
import numpy as np
import pytest

from numpy.testing import assert_equal, assert_array_almost_equal, assert_almost_equal
from math import sqrt, cos, sin, tan, acos, asin, atan, pi

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)


@pytest.fixture
def sample():
    a = 3
    b = 5

    u = hj.HyperJet(75, [b ** 2, 2 * a * b], [[0, 2 * b], [2 * b, 2 * a]])
    v = hj.HyperJet(
        225,
        [2 * a * b ** 2, 2 * a ** 2 * b],
        [[2 * b ** 2, 4 * a * b], [4 * a * b, 2 * a ** 2]],
    )

    return u, v


@pytest.fixture
def sample_trig():
    u = hj.HyperJet(1 / 2, [1 / 6, 1 / 5], [[0, 1 / 15], [1 / 15, 1 / 25]])
    v = hj.HyperJet(-1 / 4, [-1 / 6, -1 / 10], [[-1 / 18, -1 / 15], [-1 / 15, -1 / 50]])

    return u, v


def check(jet, f, g, h):
    assert_almost_equal(jet.f, f)
    assert_array_almost_equal(jet.g, g)
    assert_array_almost_equal(jet.h, h)


def test_getter_and_setter():
    u = hj.HyperJet.empty(size=2)

    u.f = 1
    u.g = [2, 3]
    u.h = [[4, 5], [6, 7]]

    assert_equal(len(u), 2)
    check(u, 1, [2, 3], [[4, 5], [6, 7]])

    with pytest.raises(RuntimeError) as ex:
        u.g = [1, 2, 3]

    assert_equal("Invalid shape!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        u.h = [[1, 2, 3], [1, 2, 3]]

    assert_equal("Invalid shape!", str(ex.value))


def test_add(sample):
    u, v = sample
    check(u + v, 300, [175, 120], [[50, 70], [70, 24]])
    check(3 + u, 78, [25, 30], [[0, 10], [10, 6]])
    check(u + 3, 78, [25, 30], [[0, 10], [10, 6]])


def test_sub(sample):
    u, v = sample
    check(u - v, -150, [-125, -60], [[-50, -50], [-50, -12]])
    check(3 - u, -72, [-25, -30], [[0, -10], [-10, -6]])
    check(u - 3, 72, [25, 30], [[0, 10], [10, 6]])


def test_mul(sample):
    u, v = sample
    check(u * v, 16875, [16875, 13500], [[11250, 13500], [13500, 8100]])
    check(3 * u, 225, [75, 90], [[0, 30], [30, 18]])
    check(u * 3, 225, [75, 90], [[0, 30], [30, 18]])


def test_div(sample):
    u, v = sample
    check(u / v, 1 / 3, [-1 / 9, 0], [[2 / 27, 0], [0, 0]])
    check(3 / u, 1 / 25, [-1 / 75, -2 / 125], [[2 / 225, 2 / 375], [2 / 375, 6 / 625]])
    check(u / 3, 25, [25 / 3, 10], [[0, 10 / 3], [10 / 3, 2]])


def test_sqrt(sample):
    u, v = sample
    check(
        u.sqrt(),
        5 * sqrt(3),
        [5 * sqrt(3) / 6, sqrt(3)],
        [[-5 * sqrt(3) / 36, sqrt(3) / 6], [sqrt(3) / 6, 0]],
    )
    check(v.sqrt(), 15, [5, 3], [[0, 1], [1, 0]])


def test_cos(sample):
    u, v = sample
    check(
        u.cos(),
        cos(75),
        [-25 * sin(75), -30 * sin(75)],
        [
            [-625 * cos(75), -750 * cos(75) - 10 * sin(75)],
            [-750 * cos(75) - 10 * sin(75), -900 * cos(75) - 6 * sin(75)],
        ],
    )
    check(
        v.cos(),
        cos(225),
        [-150 * sin(225), -90 * sin(225)],
        [
            [-22500 * cos(225) - 50 * sin(225), -13500 * cos(225) - 60 * sin(225)],
            [-13500 * cos(225) - 60 * sin(225), -8100 * cos(225) - 18 * sin(225)],
        ],
    )


def test_sin(sample):
    u, v = sample
    check(
        u.sin(),
        sin(75),
        [25 * cos(75), 30 * cos(75)],
        [
            [-625 * sin(75), 10 * cos(75) - 750 * sin(75)],
            [10 * cos(75) - 750 * sin(75), 6 * cos(75) - 900 * sin(75)],
        ],
    )
    check(
        v.sin(),
        sin(225),
        [150 * cos(225), 90 * cos(225)],
        [
            [50 * cos(225) - 22500 * sin(225), 60 * cos(225) - 13500 * sin(225)],
            [60 * cos(225) - 13500 * sin(225), 18 * cos(225) - 8100 * sin(225)],
        ],
    )


def test_tan(sample):
    u, v = sample
    check(
        u.tan(),
        tan(75),
        [25 * tan(75) ** 2 + 25, 30 * tan(75) ** 2 + 30],
        [
            [
                1250 * (tan(75) ** 2 + 1) * tan(75),
                10 * (150 * tan(75) + 1) * (tan(75) ** 2 + 1),
            ],
            [
                10 * (150 * tan(75) + 1) * (tan(75) ** 2 + 1),
                6 * (300 * tan(75) + 1) * (tan(75) ** 2 + 1),
            ],
        ],
    )
    check(
        v.tan(),
        tan(225),
        [150 + 150 * tan(225) ** 2, 90 + 90 * tan(225) ** 2],
        [
            [
                50 * (1 + tan(225) ** 2) * (900 * tan(225) + 1),
                60 * (1 + tan(225) ** 2) * (450 * tan(225) + 1),
            ],
            [
                60 * (1 + tan(225) ** 2) * (450 * tan(225) + 1),
                18 * (1 + tan(225) ** 2) * (900 * tan(225) + 1),
            ],
        ],
    )


def test_acos(sample_trig):
    u, v = sample_trig
    check(
        u.acos(),
        pi / 3,
        [-sqrt(3) / 9, -2 * sqrt(3) / 15],
        [[-sqrt(3) / 81, -8 * sqrt(3) / 135], [-8 * sqrt(3) / 135, -2 * sqrt(3) / 45]],
    )
    check(
        v.acos(),
        acos(-1 / 4),
        [2 * sqrt(15) / 45, 2 * sqrt(15) / 75],
        [
            [34 * sqrt(15) / 2025, 64 * sqrt(15) / 3375],
            [64 * sqrt(15) / 3375, 34 * sqrt(15) / 5625],
        ],
    )


def test_asin(sample_trig):
    u, v = sample_trig
    check(
        u.atan(),
        atan(1 / 2),
        [2 / 15, 4 / 25],
        [[-4 / 225, 4 / 125], [4 / 125, 4 / 625]],
    )
    check(
        v.atan(),
        -atan(1 / 4),
        [-8 / 51, -8 / 85],
        [[-104 / 2601, -16 / 289], [-16 / 289, -104 / 7225]],
    )


def test_atan(sample):
    u, v = sample
    check(
        u.atan(),
        atan(75),
        [25 / 5626, 15 / 2813],
        [[-46875 / 15825938, -14060 / 7912969], [-14060 / 7912969, -25311 / 7912969]],
    )
    check(
        v.atan(),
        atan(225),
        [75 / 25313, 45 / 25313],
        [
            [-1898425 / 640747969, -759360 / 640747969],
            [-759360 / 640747969, -683433 / 640747969],
        ],
    )


def test_atan2(sample):
    u, v = sample
    check(hj.HyperJet.atan2(u, v), atan(1 / 3), [-1 / 10, 0], [[3 / 50, 0], [0, 0]])
    check(hj.HyperJet.atan2(v, u), atan(3), [1 / 10, 0], [[-3 / 50, 0], [0, 0]])


def test_pow(sample):
    u, v = sample
    check(u ** 3, 421875, [421875, 506250], [[281250, 506250], [506250, 506250]])
    check(
        v ** (-1 / 3),
        15 ** (1 / 3) / 15,
        [-2 * 15 ** (1 / 3) / 135, -2 * 15 ** (1 / 3) / 225],
        [
            [2 * 15 ** (1 / 3) / 243, 4 * 15 ** (1 / 3) / 2025],
            [4 * 15 ** (1 / 3) / 2025, 2 * 15 ** (1 / 3) / 675],
        ],
    )


def test_equal_than():
    a = hj.HyperJet(3, [])

    assert_equal(a == 1, False)
    assert_equal(a == 3, True)

    assert_equal(1 == a, False)
    assert_equal(3 == a, True)


def test_inequal_than():
    a = hj.HyperJet(3, [])

    assert_equal(a != 1, True)
    assert_equal(a != 3, False)

    assert_equal(1 != a, True)
    assert_equal(3 != a, False)


def test_less_than():
    a = hj.HyperJet(3, [])

    assert_equal(a < 1, False)
    assert_equal(a < 3, False)
    assert_equal(a < 5, True)

    assert_equal(1 < a, True)
    assert_equal(3 < a, False)
    assert_equal(5 < a, False)


def test_greater_than():
    a = hj.HyperJet(3, [])

    assert_equal(a > 1, True)
    assert_equal(a > 3, False)
    assert_equal(a > 5, False)

    assert_equal(1 > a, False)
    assert_equal(3 > a, False)
    assert_equal(5 > a, True)


def test_less_or_equal_than():
    a = hj.HyperJet(3, [])

    assert_equal(a <= 1, False)
    assert_equal(a <= 3, True)
    assert_equal(a <= 5, True)

    assert_equal(1 <= a, True)
    assert_equal(3 <= a, True)
    assert_equal(5 <= a, False)


def test_greater_or_equal_than():
    a = hj.HyperJet(3, [])

    assert_equal(a >= 1, True)
    assert_equal(a >= 3, True)
    assert_equal(a >= 5, False)

    assert_equal(1 >= a, False)
    assert_equal(3 >= a, True)
    assert_equal(5 >= a, True)


def test_enlarge():
    a = hj.HyperJet(7, [1, 2], [[1, 2], [3, 4]])

    b = a.enlarge(right=1)

    assert_equal(len(b), 3)
    assert_equal(b.f, 7)
    assert_array_almost_equal(b.g, [1, 2, 0])
    assert_array_almost_equal(b.h, [[1, 2, 0], [3, 4, 0], [0, 0, 0]])

    b = a.enlarge(left=1)

    assert_equal(len(b), 3)
    assert_equal(b.f, 7)
    assert_array_almost_equal(b.g, [0, 1, 2])
    assert_array_almost_equal(b.h, [[0, 0, 0], [0, 1, 2], [0, 3, 4]])

    b = a.enlarge(right=1, left=1)

    assert_equal(len(b), 4)
    assert_equal(b.f, 7)
    assert_array_almost_equal(b.g, [0, 1, 2, 0])
    assert_array_almost_equal(
        b.h, [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
    )


def test_pickle():
    import pickle

    a = hj.HyperJet(1, [2, 3], [[4, 5], [6, 7]])

    data = pickle.dumps(a)

    b = pickle.loads(data)

    check(a, b.f, b.g, b.h)

    a.f = 2
    a.g[:] = [4, 6]
    a.h[:] = [[8, 10], [12, 14]]

    check(a, 2, [4, 6], [[8, 10], [12, 14]])

    check(b, 1, [2, 3], [[4, 5], [6, 7]])


def test_copy():
    from copy import copy, deepcopy

    for op in [copy, deepcopy]:
        a = hj.HyperJet(1, [2, 3], [[4, 5], [6, 7]])
        b = op(a)

        check(a, b.f, b.g, b.h)

        a.f = 2
        a.g[:] = [4, 6]
        a.h[:] = [[8, 10], [12, 14]]

        check(a, 2, [4, 6], [[8, 10], [12, 14]])

        check(b, 1, [2, 3], [[4, 5], [6, 7]])


def test_variable():
    a = hj.HyperJet.variable(f=3, size=5, index=3)

    check(a, 3, [0, 0, 0, 1, 0], np.zeros((5, 5)))


def test_variables():
    a, b, c = hj.HyperJet.variables([1, 2, 3])

    check(a, 1, [1, 0, 0], np.zeros((3, 3)))
    check(b, 2, [0, 1, 0], np.zeros((3, 3)))
    check(c, 3, [0, 0, 1], np.zeros((3, 3)))


def test_variables_with_offset():
    a, b, c = hj.HyperJet.variables(size=6, values=[1, 2, 3], offset=1)

    check(a, 1, [0, 1, 0, 0, 0, 0], np.zeros((6, 6)))
    check(b, 2, [0, 0, 1, 0, 0, 0], np.zeros((6, 6)))
    check(c, 3, [0, 0, 0, 1, 0, 0], np.zeros((6, 6)))


def test_cast_throws():
    with pytest.raises(TypeError) as ex:
        float(hj.HyperJet(f=3))

    assert_equal(
        "float() argument must be a string or a number, not 'hyperjet.HyperJet'",
        str(ex.value),
    )


def test_throws_when_dimensions_dont_match():
    with pytest.raises(RuntimeError) as ex:
        hj.HyperJet(1, [1, 2], [[1, 2, 3], [4, 5, 6]])

    assert_equal("Invalid shape!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        hj.HyperJet(1, [1, 2, 3], [[1, 2], [3, 4]])

    assert_equal("Invalid shape!", str(ex.value))

    lhs = hj.HyperJet(2, [1, 2])
    rhs = hj.HyperJet(3, [1, 2, 3])

    with pytest.raises(RuntimeError) as ex:
        lhs + rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs - rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs * rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs / rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs += rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs -= rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs *= rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs /= rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        hj.HyperJet.atan2(lhs, rhs)

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs == rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs != rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs < rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs > rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs <= rhs

    assert_equal("Caught an unknown exception!", str(ex.value))

    with pytest.raises(RuntimeError) as ex:
        lhs >= rhs

    assert_equal("Caught an unknown exception!", str(ex.value))


def test_repr():
    a = hj.HyperJet(f=4)
    assert str(a) == "4hj"


def test_backward():
    f = hj.HyperJet(f=284578770, g=[1767570, 428099], h=[[0, 2659], [2659, 322]])

    xs = [
        hj.HyperJet(f=161, g=[44, 7, 49], h=[[6, 1, 14], [1, 0, 0], [14, 0, 0]]),
        hj.HyperJet(f=1329, g=[378, 3, 1325], h=[[54, 0, 378], [0, 0, 1], [378, 1, 882]]),
    ]

    r = f.backward(xs)

    assert_almost_equal(r.g, [239594502, 13657287, 653842105])
    assert_almost_equal(r.h, [[168180390, 9519420, 552110800], [9519420, 114576, 26761147], [552110800, 26761147, 1288165718]])


def test_backward_to():
    f = hj.HyperJet(f=284578770, g=[1767570, 428099], h=[[0, 2659], [2659, 322]])

    xs = [
        hj.HyperJet(f=161, g=[44, 7, 49], h=[[6, 1, 14], [1, 0, 0], [14, 0, 0]]),
        hj.HyperJet(f=1329, g=[378, 3, 1325], h=[[54, 0, 378], [0, 0, 1], [378, 1, 882]]),
    ]

    g = np.zeros(3)
    h = np.zeros((3, 3))

    f.backward_to(xs, g, h, False)

    assert_almost_equal(g, [239594502, 13657287, 653842105])
    assert_almost_equal(h, [[168180390, 9519420, 552110800], [0, 114576, 26761147], [0, 0, 1288165718]])


def test_explode_f():
    value = hj.HyperJet(f=284578770, g=[1767570, 428099], h=[[0, 2659], [2659, 322]])

    g = np.zeros(0)
    h = np.zeros((0, 0))

    f = hj.explode(value, g, h)

    assert_almost_equal(f, 284578770)
    assert_almost_equal(g, np.zeros(0))
    assert_almost_equal(h, np.zeros((0, 0)))


def test_explode_f_and_g():
    value = hj.HyperJet(f=284578770, g=[1767570, 428099], h=[[0, 2659], [2659, 322]])

    g = np.zeros(2)
    h = np.zeros((0, 0))

    f = hj.explode(value, g, h)

    assert_almost_equal(f, 284578770)
    assert_almost_equal(g, [1767570, 428099])
    assert_almost_equal(h, np.zeros((0, 0)))


def test_explode_f_g_and_h():
    value = hj.HyperJet(f=284578770, g=[1767570, 428099], h=[[0, 2659], [2659, 322]])

    g = np.zeros(2)
    h = np.zeros((2, 2))

    f = hj.explode(value, g, h)

    assert_almost_equal(f, 284578770)
    assert_almost_equal(g, [1767570, 428099])
    assert_almost_equal(h, [[0, 2659], [2659, 322]])
