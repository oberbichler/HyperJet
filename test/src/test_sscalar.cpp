#include <doctest/doctest.h>

#include <hyperjet/hyperjet.h>

#include <sstream> // stringstream
#include <string>  // string

// Characterization of SScalar.
//
// test.cpp covers the operators and a handful of functions. This pins the rest,
// so a change to the storage -- interning the names and keeping the derivatives
// dense, for instance -- can be verified rather than hoped for.
//
// The expected values were derived symbolically, so they share no formulas with
// the header. First order means value phi(f) and derivative phi'(f) * d_i, and
// they were generated that way.

namespace {

using S = hyperjet::SScalar<double>;

const S s1(3.0, {{"x", 1.0}, {"y", 6.0}, {"z", 4.0}});
const S s2(4.0, {{"x", 7.0}, {"y", 1.0}});

// acos, asin and atanh need |f| <= 1
const S s3(0.3, {{"x", 0.1}, {"y", 0.8}, {"z", 0.2}});

void check(const S &r, const double f, const double dx, const double dy,
           const double dz) {
  CHECK(r.f() == doctest::Approx(f));
  CHECK(r.d("x") == doctest::Approx(dx));
  CHECK(r.d("y") == doctest::Approx(dy));
  CHECK(r.d("z") == doctest::Approx(dz));
}

TEST_CASE("SScalar construction") {
  CHECK(S().f() == doctest::Approx(0.0));
  CHECK(S().size() == 0);

  CHECK(S(1.5).f() == doctest::Approx(1.5));
  CHECK(S(1.5).size() == 0);

  CHECK(S::constant(1.5).f() == doctest::Approx(1.5));
  CHECK(S::constant(1.5).size() == 0);

  const auto v = S::variable("a", 1.5);

  CHECK(v.f() == doctest::Approx(1.5));
  CHECK(v.size() == 1);
  CHECK(v.d("a") == doctest::Approx(1.0));

  // an unknown name has a zero derivative rather than being an error
  CHECK(v.d("b") == doctest::Approx(0.0));
}

TEST_CASE("SScalar arithmetic functions") {
  check(s1.reciprocal(), 0.33333333333333331, -0.1111111111111111,
        -0.66666666666666663, -0.44444444444444442);

  using std::cbrt;
  check(cbrt(s1), 1.4422495703074083, 0.1602499522563787, 0.96149971353827224,
        0.64099980902551479);
}

TEST_CASE("SScalar trigonometric functions") {
  using std::acos;
  using std::asin;
  using std::atan;
  using std::atan2;

  check(acos(s3), 1.2661036727794992, -0.10482848367219183,
        -0.83862786937753464, -0.20965696734438366);
  check(asin(s3), 0.30469265401539752, 0.10482848367219183, 0.83862786937753464,
        0.20965696734438366);
  check(atan(s1), 1.2490457723982544, 0.10000000000000001, 0.59999999999999998,
        0.40000000000000002);
  check(atan2(s1, s2), 0.64350110879328437, -0.68000000000000005,
        0.83999999999999997, 0.64000000000000001);
}

TEST_CASE("SScalar hyperbolic functions") {
  using std::acosh;
  using std::asinh;
  using std::atanh;
  using std::cosh;
  using std::sinh;
  using std::tanh;

  check(cosh(s1), 10.067661995777765, 10.017874927409903, 60.107249564459408,
        40.07149970963961);
  check(sinh(s1), 10.017874927409903, 10.067661995777765, 60.405971974666592,
        40.270647983111061);
  check(tanh(s1), 0.99505475368673046, 0.0098660371654401904,
        0.05919622299264115, 0.039464148661760762);
  check(acosh(s1), 1.7627471740390861, 0.35355339059327379, 2.1213203435596424,
        1.4142135623730951);
  check(asinh(s1), 1.8184464592320668, 0.31622776601683794, 1.8973665961010275,
        1.2649110640673518);
  check(atanh(s3), 0.3095196042031117, 0.10989010989010989, 0.87912087912087911,
        0.21978021978021978);
}

TEST_CASE("SScalar exponents and logarithms") {
  using std::exp;
  using std::log;
  using std::log10;
  using std::log2;

  check(exp(s1), 20.085536923187668, 20.085536923187668, 120.51322153912601,
        80.342147692750672);
  check(log(s1), 1.0986122886681098, 0.33333333333333331, 2.0,
        1.3333333333333333);
  check(log2(s1), 1.5849625007211561, 0.48089834696298778, 2.8853900817779268,
        1.9235933878519511);
  check(log10(s1), 0.47712125471966244, 0.14476482730108395,
        0.86858896380650363, 0.57905930920433579);

  // log with an explicit base agrees with log2
  check(s1.log(2.0), 1.5849625007211561, 0.48089834696298778,
        2.8853900817779268, 1.9235933878519511);
}

TEST_CASE("SScalar abs") {
  using std::abs;

  // a positive value passes through, a negative one flips value and derivatives
  check(abs(s1), 3.0, 1.0, 6.0, 4.0);
  check(abs(-s1), 3.0, 1.0, 6.0, 4.0);
}

TEST_CASE("SScalar comparison uses the value only") {
  const S a(1.0, {{"x", 100.0}});
  const S b(2.0, {{"y", -100.0}});

  CHECK(a < b);
  CHECK(a <= b);
  CHECK(b > a);
  CHECK(b >= a);
  CHECK(a != b);
  CHECK_FALSE(a == b);

  // equal values compare equal no matter what the derivatives say
  const S c(1.0);

  CHECK(a == c);
  CHECK(a <= c);
  CHECK(a >= c);

  CHECK(a < 2.0);
  CHECK(a == 1.0);
  CHECK(2.0 > a);
  CHECK(1.0 == a);
}

TEST_CASE("SScalar eval") {
  // f plus the derivatives contracted with the displacement
  CHECK(s1.eval({{"x", 0.5}, {"y", -0.25}, {"z", 2.0}}) ==
        doctest::Approx(10.0));

  // names missing from the displacement contribute nothing
  CHECK(s1.eval({{"x", 0.5}}) == doctest::Approx(3.5));
  CHECK(s1.eval({}) == doctest::Approx(3.0));

  // names the scalar does not know are ignored
  CHECK(s1.eval({{"x", 0.5}, {"w", 100.0}}) == doctest::Approx(3.5));
}

TEST_CASE("SScalar size counts the stored derivatives") {
  CHECK(s1.size() == 3);
  CHECK(s2.size() == 2);
  CHECK(S::constant(1.0).size() == 0);

  // an operation takes the union of the names, even where a derivative
  // cancels to zero
  const auto sum = s1 + s2;

  CHECK(sum.size() == 3);

  // deliberate: the names have to survive even when the derivative cancels
  // NOLINTNEXTLINE(misc-redundant-expression)
  const auto diff = s1 - s1;

  CHECK(diff.size() == 3);
  CHECK(diff.d("x") == doctest::Approx(0.0));
}

TEST_CASE("SScalar printing") {
  // The derivatives live in an unordered_map, so the order of the terms is
  // unspecified. Only the value comes first, and every term appears once.
  std::stringstream out;
  out << s2;
  const std::string text = out.str();

  CHECK(text.starts_with("4"));
  CHECK(text.find("*dx") != std::string::npos);
  CHECK(text.find("*dy") != std::string::npos);
  CHECK(text.find("+7") != std::string::npos);
  CHECK(text.find("+1") != std::string::npos);

  // negative derivatives are printed without a plus sign
  std::stringstream negative;
  negative << S(1.0, {{"x", -2.0}});

  CHECK(negative.str() == "1 -2*dx");
}

} // namespace
