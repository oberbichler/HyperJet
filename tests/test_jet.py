import unittest
from HyperJet import Jet
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from math import sqrt, cos, sin, tan, acos, asin, atan, pi

class TestJet(unittest.TestCase):

    def sample(self):
        a = 3
        b = 5

        u = Jet(75, [b**2, 2*a*b])
        v = Jet(225, [2*a*b**2, 2*a**2*b])

        return u, v

    def sample_trig(self):
        u = Jet(1/2, [1/6, 1/5])
        v = Jet(-1/4, [-1/6, -1/10])

        return u, v

    def check(self, jet, f, g, h=None):
        assert_almost_equal(jet.f, f)
        assert_array_almost_equal(jet.g, g)

    def test_hyperjet_getter_and_setter(self):
        u = Jet(2)

        u.f = 1
        u.g = [2, 3]

        self.assertEqual(len(u), 2)
        self.check(u, 1, [2, 3])

        def wrong_shape_for_g():
            u.g = [1, 2, 3]

        self.assertRaises(RuntimeError, wrong_shape_for_g)


    def test_jet_add(self):
        u, v = self.sample()
        self.check(u + v, 300, [175, 120], [[50, 70], [70, 24]])
        self.check(3 + u, 78, [25, 30], [[0, 10], [10, 6]])
        self.check(u + 3, 78, [25, 30], [[0, 10], [10, 6]])

    def test_jet_sub(self):
        u, v = self.sample()
        self.check(u - v, -150, [-125, -60], [[-50, -50], [-50, -12]])
        self.check(3 - u, -72, [-25, -30], [[0, -10], [-10, -6]])
        self.check(u - 3, 72, [25, 30], [[0, 10], [10, 6]])

    def test_jet_mul(self):
        u, v = self.sample()
        self.check(u * v, 16875, [16875, 13500], [[11250, 13500], [13500, 8100]])
        self.check(3 * u, 225, [75, 90], [[0, 30], [30, 18]])
        self.check(u * 3, 225, [75, 90], [[0, 30], [30, 18]])

    def test_jet_div(self):
        u, v = self.sample()
        self.check(u / v, 1/3, [-1/9, 0], [[2/27, 0], [0, 0]])
        self.check(3 / u, 1/25, [-1/75, -2/125], [[2/225, 2/375], [2/375, 6/625]])
        self.check(u / 3, 25, [25/3, 10], [[0, 10/3], [10/3, 2]])

    def test_jet_sqrt(self):
        u, v = self.sample()
        self.check(u.sqrt(), 5*sqrt(3), [5*sqrt(3)/6, sqrt(3)], [[-5*sqrt(3)/36, sqrt(3)/6], [sqrt(3)/6, 0]])
        self.check(v.sqrt(), 15, [5, 3], [[0, 1], [1, 0]])

    def test_jet_cos(self):
        u, v = self.sample()
        self.check(u.cos(), cos(75), [-25*sin(75), -30*sin(75)], [[-625*cos(75), -750*cos(75) - 10*sin(75)], [-750*cos(75) - 10*sin(75), -900*cos(75) - 6*sin(75)]])
        self.check(v.cos(), cos(225), [-150*sin(225), -90*sin(225)], [[-22500*cos(225) - 50*sin(225), -13500*cos(225) - 60*sin(225)], [-13500*cos(225) - 60*sin(225), -8100*cos(225) - 18*sin(225)]])

    def test_jet_sin(self):
        u, v = self.sample()
        self.check(u.sin(), sin(75), [25*cos(75), 30*cos(75)], [[-625*sin(75), 10*cos(75) - 750*sin(75)], [10*cos(75) - 750*sin(75), 6*cos(75) - 900*sin(75)]])
        self.check(v.sin(), sin(225), [150*cos(225), 90*cos(225)], [[50*cos(225) - 22500*sin(225), 60*cos(225) - 13500*sin(225)], [60*cos(225) - 13500*sin(225), 18*cos(225) - 8100*sin(225)]])

    def test_jet_tan(self):
        u, v = self.sample()
        self.check(u.tan(), tan(75), [25*tan(75)**2 + 25, 30*tan(75)**2 + 30], [[1250*(tan(75)**2 + 1)*tan(75), 10*(150*tan(75) + 1)*(tan(75)**2 + 1)], [10*(150*tan(75) + 1)*(tan(75)**2 + 1), 6*(300*tan(75) + 1)*(tan(75)**2 + 1)]])
        self.check(v.tan(), tan(225), [150 + 150*tan(225)**2, 90 + 90*tan(225)**2], [[50*(1 + tan(225)**2)*(900*tan(225) + 1), 60*(1 + tan(225)**2)*(450*tan(225) + 1)], [60*(1 + tan(225)**2)*(450*tan(225) + 1), 18*(1 + tan(225)**2)*(900*tan(225) + 1)]])

    def test_jet_acos(self):
        u, v = self.sample_trig()
        self.check(u.acos(), pi/3, [-sqrt(3)/9, -2*sqrt(3)/15], [[-sqrt(3)/81, -8*sqrt(3)/135], [-8*sqrt(3)/135, -2*sqrt(3)/45]])
        self.check(v.acos(), acos(-1/4), [2*sqrt(15)/45, 2*sqrt(15)/75], [[34*sqrt(15)/2025, 64*sqrt(15)/3375], [64*sqrt(15)/3375, 34*sqrt(15)/5625]])

    def test_jet_asin(self):
        u, v = self.sample_trig()
        self.check(u.atan(), atan(1/2), [2/15, 4/25], [[-4/225, 4/125], [4/125, 4/625]])
        self.check(v.atan(), -atan(1/4), [-8/51, -8/85], [[-104/2601, -16/289], [-16/289, -104/7225]])

    def test_jet_atan(self):
        u, v = self.sample()
        self.check(u.atan(), atan(75), [25/5626, 15/2813], [[-46875/15825938, -14060/7912969], [-14060/7912969, -25311/7912969]])
        self.check(v.atan(), atan(225), [75/25313, 45/25313], [[-1898425/640747969, -759360/640747969], [-759360/640747969, -683433/640747969]])

    def test_jet_atan2(self):
        u, v = self.sample()
        self.check(Jet.atan2(u, v), atan(1/3), [-1/10, 0], [[3/50, 0], [0, 0]])
        self.check(Jet.atan2(v, u), atan(3), [1/10, 0], [[-3/50, 0], [0, 0]])

    def test_jet_pow(self):
        u, v = self.sample()
        self.check(u**3, 421875, [421875, 506250], [[281250, 506250], [506250, 506250]])
        self.check(v**(-1/3), 15**(1/3)/15, [-2*15**(1/3)/135, -2*15**(1/3)/225], [[2*15**(1/3)/243, 4*15**(1/3)/2025], [4*15**(1/3)/2025, 2*15**(1/3)/675]])

    def test_jet_equal_than(self):
        a = Jet(3, [])

        self.assertEqual(a == 1, False)
        self.assertEqual(a == 3, True)

        self.assertEqual(1 == a, False)
        self.assertEqual(3 == a, True)

    def test_jet_inequal_than(self):
        a = Jet(3, [])

        self.assertEqual(a != 1, True)
        self.assertEqual(a != 3, False)

        self.assertEqual(1 != a, True)
        self.assertEqual(3 != a, False)

    def test_jet_less_than(self):
        a = Jet(3, [])

        self.assertEqual(a < 1, False)
        self.assertEqual(a < 3, False)
        self.assertEqual(a < 5, True)

        self.assertEqual(1 < a, True)
        self.assertEqual(3 < a, False)
        self.assertEqual(5 < a, False)

    def test_jet_greater_than(self):
        a = Jet(3, [])

        self.assertEqual(a > 1, True)
        self.assertEqual(a > 3, False)
        self.assertEqual(a > 5, False)

        self.assertEqual(1 > a, False)
        self.assertEqual(3 > a, False)
        self.assertEqual(5 > a, True)

    def test_jet_less_or_equal_than(self):
        a = Jet(3, [])

        self.assertEqual(a <= 1, False)
        self.assertEqual(a <= 3, True)
        self.assertEqual(a <= 5, True)

        self.assertEqual(1 <= a, True)
        self.assertEqual(3 <= a, True)
        self.assertEqual(5 <= a, False)

    def test_jet_greater_or_equal_than(self):
        a = Jet(3, [])

        self.assertEqual(a >= 1, True)
        self.assertEqual(a >= 3, True)
        self.assertEqual(a >= 5, False)

        self.assertEqual(1 >= a, False)
        self.assertEqual(3 >= a, True)
        self.assertEqual(5 >= a, True)

    def test_jet_enlarge(self):
        a = Jet(7, [1, 2])

        b = a.enlarge(1, False)

        self.assertEqual(len(b), 3)
        self.assertEqual(b.f, 7)
        assert_array_almost_equal(b.g, [1, 2, 0])

        b = a.enlarge(1, True)

        self.assertEqual(len(b), 3)
        self.assertEqual(b.f, 7)
        assert_array_almost_equal(b.g, [0, 1, 2])

        b = a.enlarge(right=1)

        self.assertEqual(len(b), 3)
        self.assertEqual(b.f, 7)
        assert_array_almost_equal(b.g, [1, 2, 0])

        b = a.enlarge(left=1)

        self.assertEqual(len(b), 3)
        self.assertEqual(b.f, 7)
        assert_array_almost_equal(b.g, [0, 1, 2])

        b = a.enlarge(right=1, left=1)

        self.assertEqual(len(b), 4)
        self.assertEqual(b.f, 7)
        assert_array_almost_equal(b.g, [0, 1, 2, 0])

    def test_jet_pickle(self):
        import pickle

        a = Jet(1, [2, 3])

        data = pickle.dumps(a)

        b = pickle.loads(data)

        self.assertEqual(a.f, b.f)
        assert_array_almost_equal(a.g, b.g)

        a.f = 2
        a.g[:] = [4, 6]

        self.check(a, 2, [4, 6])

        self.check(b, 1, [2, 3])

    def test_jet_copy(self):
        from copy import copy, deepcopy

        for op in [copy, deepcopy]:
            a = Jet(1, [2, 3])
            b = op(a)

            self.check(a, b.f, b.g)

            a.f = 2
            a.g[:] = [4, 6]

            self.check(a, 2, [4, 6])

            self.check(b, 1, [2, 3])


if __name__ == '__main__':
    unittest.main()