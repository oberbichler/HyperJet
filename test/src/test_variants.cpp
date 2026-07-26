#include <doctest/doctest.h>

#include <Eigen/Core>

#include <hyperjet/hyperjet.h>

#include <array>  // array
#include <vector> // vector

// Coverage across all four combinations of order and sizing.
//
// test.cpp pins DDScalar<2, double, 3> in numeric detail. These cases run the
// same machinery for every variant, so the first-order and dynamic
// instantiations get compiled and checked as well -- a template that is never
// instantiated is never compiled, and neither warnings nor sanitizers can see
// into it.
//
// The expected values were derived symbolically from quadratic Taylor
// polynomials, independently of the formulas in the header. One set of numbers
// serves both orders: the first-order part of every operation depends only on
// the first-order parts of its inputs, so truncating a second-order
// expectation to [f, g...] gives the first-order one.

namespace variants {

using hyperjet::DDScalar;
using hyperjet::Dynamic;
using hyperjet::index;
using hyperjet::length;

constexpr index Size = 2;
constexpr double S = 1.75;

using D1 = DDScalar<1, double, Size>;
using D2 = DDScalar<2, double, Size>;
using X1 = DDScalar<1, double, Dynamic>;
using X2 = DDScalar<2, double, Dynamic>;

//                          f     g0     g1     h00    h01    h11
const std::vector<double> A{1.5, 0.5, -0.25, 0.75, -0.5, 0.25};
const std::vector<double> B{2.5, -0.75, 0.5, 0.25, 0.75, -0.25};
const std::vector<double> C{3.5, 0.25, 0.75, -0.5, 0.25, 0.5};

const std::vector<double> NegA{-1.5, -0.5, 0.25, -0.75, 0.5, -0.25};

const std::vector<double> AddAB{4, -0.25, 0.25, 1, 0.25, 0};
const std::vector<double> AddAS{3.25, 0.5, -0.25, 0.75, -0.5, 0.25};

const std::vector<double> SubAB{-1, 1.25, -0.75, 0.5, -1.25, 0.5};
const std::vector<double> SubAS{-0.25, 0.5, -0.25, 0.75, -0.5, 0.25};
const std::vector<double> SubSA{0.25, -0.5, 0.25, -0.75, 0.5, -0.25};

const std::vector<double> MulAB{3.75, 0.125, 0.125, 1.5, 0.3125, 0};
const std::vector<double> MulAS{2.625, 0.875, -0.4375, 1.3125, -0.875, 0.4375};

const std::vector<double> DivAB{
    0.59999999999999998,  0.38, -0.22, 0.46800000000000003,
    -0.52200000000000002, 0.248};
const std::vector<double> DivAS{0.8571428571428571,   0.2857142857142857,
                                -0.14285714285714285, 0.42857142857142855,
                                -0.2857142857142857,  0.14285714285714285};
const std::vector<double> DivSA{1.1666666666666667,  -0.3888888888888889,
                                0.19444444444444445, -0.32407407407407407,
                                0.25925925925925924, -0.12962962962962962};

const std::vector<double> SqrtA{1.2247448713915889,   0.20412414523193151,
                                -0.10206207261596575, 0.27216552697590868,
                                -0.1871137997959372,  0.093556899897968601};
const std::vector<double> ExpA{4.4816890703380645,  2.2408445351690323,
                               -1.1204222675845161, 4.4816890703380645,
                               -2.8010556689612907, 1.4005278344806453};

const std::vector<double> Atan2AB{0.54041950027058416,  0.27941176470588236,
                                  -0.16176470588235295, 0.25043252595155707,
                                  -0.32958477508650519, 0.15095155709342561};

const std::vector<double> HypotABC{4.5552167895721496,  -0.054882129994845173,
                                   0.76834981992783247, 0.1914262245000925,
                                   0.37971160421132338, 0.39177906050537065};

// Builds a scalar from data laid out for second order. For first-order types
// the Hessian part is dropped.
template <typename T> T make(const std::vector<double> &data) {
  const index n = T::data_length_from_size(Size);

  typename T::Data d;

  if constexpr (T::is_dynamic()) {
    d.assign(data.begin(), data.begin() + n);
  } else {
    std::copy(data.begin(), data.begin() + n, d.begin());
  }

  return T::create(d);
}

template <typename T>
void check(const T &actual, const std::vector<double> &expected) {
  const index n = T::data_length_from_size(Size);

  REQUIRE(actual.size() == Size);
  REQUIRE(actual.data_length() == n);

  for (index i = 0; i < n; i++) {
    CHECK(actual.data()[i] == doctest::Approx(expected[i]));
  }
}

TEST_CASE_TEMPLATE("variants: values", T, D1, X1, D2, X2) {
  auto a = make<T>(A);

  CHECK(a.size() == Size);
  CHECK(a.data_length() == T::data_length_from_size(Size));
  CHECK(a.f() == doctest::Approx(A[0]));
  CHECK(a.g(0) == doctest::Approx(A[1]));
  CHECK(a.g(1) == doctest::Approx(A[2]));

  if constexpr (T::order() == 2) {
    CHECK(a.h(0, 0) == doctest::Approx(A[3]));
    CHECK(a.h(0, 1) == doctest::Approx(A[4]));
    CHECK(a.h(1, 0) == doctest::Approx(A[4]));
    CHECK(a.h(1, 1) == doctest::Approx(A[5]));

    // the linear index form is not exposed to Python, so this is its only
    // coverage
    CHECK(a.h(0) == doctest::Approx(A[3]));
    CHECK(a.h(1) == doctest::Approx(A[4]));
    CHECK(a.h(2) == doctest::Approx(A[5]));
  }

  a.set_f(9.0);
  a.set_g(0, 8.0);

  CHECK(a.f() == doctest::Approx(9.0));
  CHECK(a.g(0) == doctest::Approx(8.0));

  if constexpr (T::order() == 2) {
    a.set_h(0, 1, 7.0);
    CHECK(a.h(1, 0) == doctest::Approx(7.0));
  }
}

TEST_CASE_TEMPLATE("variants: factories", T, D1, X1, D2, X2) {
  const index n = T::data_length_from_size(Size);

  const auto e = T::empty(Size);

  CHECK(e.size() == Size);
  CHECK(e.data_length() == n);

  const auto z = T::zero(Size);

  CHECK(z.size() == Size);

  for (index i = 0; i < n; i++) {
    CHECK(z.data()[i] == doctest::Approx(0.0));
  }

  const auto c = T::constant(S, Size);

  CHECK(c.f() == doctest::Approx(S));

  for (index i = 1; i < n; i++) {
    CHECK(c.data()[i] == doctest::Approx(0.0));
  }

  const auto v = T::variable(1, S, Size);

  CHECK(v.f() == doctest::Approx(S));
  CHECK(v.g(0) == doctest::Approx(0.0));
  CHECK(v.g(1) == doctest::Approx(1.0));

  const auto vars = T::variables(std::vector<double>{2.0, 3.0});

  REQUIRE(length(vars) == Size);
  CHECK(vars[0].f() == doctest::Approx(2.0));
  CHECK(vars[0].g(0) == doctest::Approx(1.0));
  CHECK(vars[0].g(1) == doctest::Approx(0.0));
  CHECK(vars[1].f() == doctest::Approx(3.0));
  CHECK(vars[1].g(0) == doctest::Approx(0.0));
  CHECK(vars[1].g(1) == doctest::Approx(1.0));
}

TEST_CASE_TEMPLATE("variants: neg", T, D1, X1, D2, X2) {
  check(-make<T>(A), NegA);
}

TEST_CASE_TEMPLATE("variants: add", T, D1, X1, D2, X2) {
  const auto a = make<T>(A);

  check(a + make<T>(B), AddAB);
  check(a + S, AddAS);
  check(S + a, AddAS);
}

TEST_CASE_TEMPLATE("variants: sub", T, D1, X1, D2, X2) {
  const auto a = make<T>(A);

  check(a - make<T>(B), SubAB);
  check(a - S, SubAS);
  check(S - a, SubSA);
}

TEST_CASE_TEMPLATE("variants: mul", T, D1, X1, D2, X2) {
  const auto a = make<T>(A);

  check(a * make<T>(B), MulAB);
  check(a * S, MulAS);
  check(S * a, MulAS);
}

TEST_CASE_TEMPLATE("variants: div", T, D1, X1, D2, X2) {
  const auto a = make<T>(A);

  check(a / make<T>(B), DivAB);
  check(a / S, DivAS);
  check(S / a, DivSA);
}

TEST_CASE_TEMPLATE("variants: in place", T, D1, X1, D2, X2) {
  const auto a = make<T>(A);
  const auto b = make<T>(B);

  {
    auto r = a;
    r += b;
    check(r, AddAB);
  }
  {
    auto r = a;
    r -= b;
    check(r, SubAB);
  }
  {
    auto r = a;
    r *= b;
    check(r, MulAB);
  }
  {
    auto r = a;
    r /= b;
    check(r, DivAB);
  }
  {
    auto r = a;
    r += S;
    check(r, AddAS);
  }
  {
    auto r = a;
    r -= S;
    check(r, SubAS);
  }
  {
    auto r = a;
    r *= S;
    check(r, MulAS);
  }
  {
    auto r = a;
    r /= S;
    check(r, DivAS);
  }
}

TEST_CASE_TEMPLATE("variants: unary functions", T, D1, X1, D2, X2) {
  const auto a = make<T>(A);

  check(sqrt(a), SqrtA);
  check(exp(a), ExpA);
}

TEST_CASE_TEMPLATE("variants: binary functions", T, D1, X1, D2, X2) {
  check(atan2(make<T>(A), make<T>(B)), Atan2AB);
}

TEST_CASE_TEMPLATE("variants: ternary functions", T, D1, X1, D2, X2) {
  check(hypot(make<T>(A), make<T>(B), make<T>(C)), HypotABC);
}

TEST_CASE_TEMPLATE("variants: eval", T, D1, X1, D2, X2) {
  const auto a = make<T>(A);

  // eval is the truncated Taylor polynomial evaluated at the displacement
  const double expected = T::order() == 1 ? 1.75 : 1.375;

  if constexpr (T::is_dynamic()) {
    CHECK(a.eval(std::vector<double>{2.0, 3.0}) == doctest::Approx(expected));
  } else {
    CHECK(a.eval(std::array<double, 2>{2.0, 3.0}) == doctest::Approx(expected));
  }
}

TEST_CASE_TEMPLATE("variants: padding", T, D1, X1, D2, X2) {
  if constexpr (T::is_dynamic()) {
    const auto a = make<T>(A);

    const auto r = a.pad_right(3);

    REQUIRE(r.size() == 3);
    CHECK(r.f() == doctest::Approx(A[0]));
    CHECK(r.g(0) == doctest::Approx(A[1]));
    CHECK(r.g(1) == doctest::Approx(A[2]));
    CHECK(r.g(2) == doctest::Approx(0.0));

    const auto l = a.pad_left(3);

    REQUIRE(l.size() == 3);
    CHECK(l.f() == doctest::Approx(A[0]));
    CHECK(l.g(0) == doctest::Approx(0.0));
    CHECK(l.g(1) == doctest::Approx(A[1]));
    CHECK(l.g(2) == doctest::Approx(A[2]));

    if constexpr (T::order() == 2) {
      CHECK(r.h(0, 0) == doctest::Approx(A[3]));
      CHECK(r.h(0, 1) == doctest::Approx(A[4]));
      CHECK(r.h(1, 1) == doctest::Approx(A[5]));
      CHECK(r.h(0, 2) == doctest::Approx(0.0));
      CHECK(r.h(1, 2) == doctest::Approx(0.0));
      CHECK(r.h(2, 2) == doctest::Approx(0.0));

      CHECK(l.h(1, 1) == doctest::Approx(A[3]));
      CHECK(l.h(1, 2) == doctest::Approx(A[4]));
      CHECK(l.h(2, 2) == doctest::Approx(A[5]));
      CHECK(l.h(0, 0) == doctest::Approx(0.0));
      CHECK(l.h(0, 1) == doctest::Approx(0.0));
      CHECK(l.h(0, 2) == doctest::Approx(0.0));
    }
  }
}

} // namespace variants
