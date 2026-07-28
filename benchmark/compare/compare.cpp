// Derivatives of one non-trivial function, computed every way a caller might
// reasonably choose -- first order, then first and second order together.
//
// The micro-benchmarks in ../src measure single operations against nothing.
// This measures the thing a caller actually wants -- value, gradient and full
// Hessian of a real function -- against the alternatives someone choosing a
// library would consider.
//
// The function is the strain energy of a cable with unit-spaced nodes and
// transverse displacements x_i:
//
//   E(x) = sum_i ( sqrt(1 + (x_{i+1} - x_i)^2) - 1 )^2
//
// Each term is the squared elongation of one segment. It is nonlinear, its
// Hessian is not constant, and it is defined for any n, which is what makes it
// usable across all four contenders and all sizes.
//
// The Hessian is banded, but every contender here computes all n^2 entries
// regardless -- dense forward mode does not exploit structure -- so the shape
// does not favour any of them.
//
// Note that the two orders are separate comparisons. ceres::Jet is first order
// only, which is not a limitation of Ceres but what a solver needs: it wants
// Jacobians. Nesting Jet inside Jet to force it into the second-order table
// would benchmark a configuration no Ceres user writes.

#include <hyperjet/hyperjet.h>

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include <ceres/jet.h>

#include <benchmark/benchmark.h>

#include <cmath>   // cbrt, sqrt, abs
#include <cstdio>  // printf
#include <cstdlib> // abort
#include <string>  // string
#include <type_traits>
#include <vector>

// The compiler decides per instantiation whether to inline term() below, and it
// decided differently for different contenders: HyperJet's body is larger, so
// clang left it out of line and every call returned a whole scalar -- 200 bytes
// at n=24 -- through the stack. That cost 1.5x and had nothing to do with the
// libraries, only with where an inlining threshold happened to fall. Forcing it
// uniformly measures the arithmetic instead of the threshold.
#if defined _MSC_VER
#define COMPARE_INLINE __forceinline
#else
#define COMPARE_INLINE __attribute__((always_inline))
#endif

namespace hj = hyperjet;

namespace {

// The function, written once. Each contender supplies an accessor, so all four
// differentiate the same expression rather than four transcriptions of it.
template <typename TAt> auto energy(const int n, TAt at) {
  using std::sqrt;

  // The accessor's plain scalar type. Every intermediate is spelled out as this
  // type on purpose: both Eigen's AutoDiffScalar and autodiff's dual are
  // expression-template types, so `auto d = at(i + 1) - at(i)` would keep an
  // expression that refers to operands which are gone by the time it is read,
  // and returning `auto` from term() would do the same. HyperJet has no
  // expression templates and does not care either way.
  //
  // Starting the accumulator from a literal zero is no help: not every
  // contender's scalar is constructible from a double.
  using Scalar = std::decay_t<decltype(at(0))>;

  const auto term = [&](const int i) COMPARE_INLINE -> Scalar {
    const Scalar d = at(i + 1) - at(i);
    const Scalar s = sqrt(1.0 + d * d) - 1.0;

    return s * s;
  };

  Scalar e = term(0);

  for (int i = 1; i + 1 < n; i++) {
    e = e + term(i);
  }

  return e;
}

std::vector<double> point(const int n) {
  std::vector<double> x(static_cast<std::size_t>(n));

  for (int i = 0; i < n; i++) {
    // an arbitrary but non-degenerate configuration; a straight cable would
    // make every segment's elongation zero and flatter the nonlinear terms
    x[static_cast<std::size_t>(i)] = 0.1 * std::sin(0.7 * (i + 1));
  }

  return x;
}

// The result all four have to agree on: value, gradient, dense Hessian.
//
// The caller allocates it once and every contender fills it in place. Filling a
// fresh one inside the timing loop would put three vector allocations into
// every measurement -- at n=3 that is the same order as the whole computation,
// so it would measure std::vector rather than the differentiation.
struct Result {
  double f{};
  std::vector<double> g;
  std::vector<double> h; // row-major, n by n

  void resize(const int n) {
    g.resize(static_cast<std::size_t>(n));
    h.resize(static_cast<std::size_t>(n * n));
  }

  // For the first-order contenders. deviation() walks whatever is there, so
  // leaving h empty simply takes it out of the comparison.
  void resize_gradient(const int n) {
    g.resize(static_cast<std::size_t>(n));
    h.clear();
  }
};

// --- HyperJet, static size

template <int TSize>
void hyperjet_static(const std::vector<double> &x, Result &r) {
  using S = hj::DDScalar<2, double, TSize>;

  std::array<double, TSize> values{};

  for (int i = 0; i < TSize; i++) {
    values[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)];
  }

  const auto v = S::variables(values);
  const auto e = energy(TSize, [&](const int i) { return v[i]; });

  r.f = e.f();

  for (int i = 0; i < TSize; i++) {
    r.g[static_cast<std::size_t>(i)] = e.g(i);

    for (int j = 0; j < TSize; j++) {
      r.h[static_cast<std::size_t>(i * TSize + j)] = e.h(i, j);
    }
  }
}

// --- HyperJet, dynamic size

void hyperjet_dynamic(const std::vector<double> &x, Result &r) {
  using S = hj::DDScalar<2, double>;

  const auto n = static_cast<int>(x.size());
  const auto v = S::variables(x);
  const auto e = energy(n, [&](const int i) { return v[i]; });

  r.f = e.f();

  for (int i = 0; i < n; i++) {
    r.g[static_cast<std::size_t>(i)] = e.g(i);

    for (int j = 0; j < n; j++) {
      r.h[static_cast<std::size_t>(i * n + j)] = e.h(i, j);
    }
  }
}

// --- Eigen AutoDiffScalar, nested for second order
//
// Eigen ships first-order forward mode. Second order comes from nesting: the
// outer scalar's derivatives are themselves first-order AD scalars, so the
// outer gradient carries the Hessian rows. Seeding that takes O(n^2) writes
// before the function is even called.

template <int TSize>
void eigen_nested(const std::vector<double> &x, Result &r) {
  using Inner = Eigen::AutoDiffScalar<Eigen::Matrix<double, TSize, 1>>;
  using Outer = Eigen::AutoDiffScalar<Eigen::Matrix<Inner, TSize, 1>>;

  Eigen::Matrix<Outer, TSize, 1> v;

  for (int i = 0; i < TSize; i++) {
    v(i).value().value() = x[static_cast<std::size_t>(i)];
    v(i).value().derivatives() = Eigen::Matrix<double, TSize, 1>::Unit(i);

    for (int j = 0; j < TSize; j++) {
      v(i).derivatives()(j).value() = i == j ? 1.0 : 0.0;
      v(i).derivatives()(j).derivatives().setZero();
    }
  }

  const auto e = energy(TSize, [&](const int i) { return v(i); });

  r.f = e.value().value();

  for (int i = 0; i < TSize; i++) {
    r.g[static_cast<std::size_t>(i)] = e.value().derivatives()(i);

    for (int j = 0; j < TSize; j++) {
      r.h[static_cast<std::size_t>(i * TSize + j)] =
          e.derivatives()(i).derivatives()(j);
    }
  }
}

// --- autodiff, dual2nd
//
// autodiff's hessian() seeds one pair (i, j) at a time and re-evaluates the
// function for each, so it calls the function n(n+1)/2 times rather than once.
// That is the library's design, not a misuse of it: dual2nd carries a single
// second-order direction.

void autodiff_dual2nd(const std::vector<double> &x, Result &r) {
  const auto n = static_cast<int>(x.size());

  autodiff::VectorXdual2nd v(n);

  for (int i = 0; i < n; i++) {
    v(i) = x[static_cast<std::size_t>(i)];
  }

  const auto fn = [n](const auto &values) {
    return energy(n, [&](const int i) { return values(i); });
  };

  autodiff::dual2nd u;
  Eigen::VectorXd g;
  const Eigen::MatrixXd h =
      autodiff::hessian(fn, autodiff::wrt(v), autodiff::at(v), u, g);

  r.f = static_cast<double>(u);

  for (int i = 0; i < n; i++) {
    r.g[static_cast<std::size_t>(i)] = g(i);

    for (int j = 0; j < n; j++) {
      r.h[static_cast<std::size_t>(i * n + j)] = h(i, j);
    }
  }
}

// --- Central finite differences
//
// The baseline. No library, no types -- and no exact answer either, which is
// the point of including it: the deviation is reported alongside the timing.

void finite_differences(const std::vector<double> &x, Result &r) {
  const auto n = static_cast<int>(x.size());

  // cbrt(eps) is the usual choice for second derivatives: it balances
  // truncation against cancellation
  const double h = std::cbrt(std::numeric_limits<double>::epsilon());

  const auto eval = [](const std::vector<double> &values) {
    const auto m = static_cast<int>(values.size());

    return energy(
        m, [&](const int i) { return values[static_cast<std::size_t>(i)]; });
  };

  const auto shifted = [&](const int i, const double di, const int j,
                           const double dj) {
    auto values = x;
    values[static_cast<std::size_t>(i)] += di;
    values[static_cast<std::size_t>(j)] += dj;

    return eval(values);
  };

  r.f = eval(x);

  for (int i = 0; i < n; i++) {
    const auto fp = shifted(i, h, i, 0.0);
    const auto fm = shifted(i, -h, i, 0.0);

    r.g[static_cast<std::size_t>(i)] = (fp - fm) / (2.0 * h);
    r.h[static_cast<std::size_t>(i * n + i)] = (fp - 2.0 * r.f + fm) / (h * h);

    for (int j = i + 1; j < n; j++) {
      const auto pp = shifted(i, h, j, h);
      const auto pm = shifted(i, h, j, -h);
      const auto mp = shifted(i, -h, j, h);
      const auto mm = shifted(i, -h, j, -h);

      const auto value = (pp - pm - mp + mm) / (4.0 * h * h);

      r.h[static_cast<std::size_t>(i * n + j)] = value;
      r.h[static_cast<std::size_t>(j * n + i)] = value;
    }
  }
}

// === First order: value and gradient only ===
//
// A separate comparison, because this is the regime Ceres and Eigen were built
// for and where most callers actually live.

// --- HyperJet, static size

template <int TSize>
void hyperjet_static_g(const std::vector<double> &x, Result &r) {
  using S = hj::DDScalar<1, double, TSize>;

  std::array<double, TSize> values{};

  for (int i = 0; i < TSize; i++) {
    values[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)];
  }

  const auto v = S::variables(values);
  const auto e = energy(TSize, [&](const int i) { return v[i]; });

  r.f = e.f();

  for (int i = 0; i < TSize; i++) {
    r.g[static_cast<std::size_t>(i)] = e.g(i);
  }
}

// --- HyperJet, dynamic size

void hyperjet_dynamic_g(const std::vector<double> &x, Result &r) {
  using S = hj::DDScalar<1, double>;

  const auto n = static_cast<int>(x.size());
  const auto v = S::variables(x);
  const auto e = energy(n, [&](const int i) { return v[i]; });

  r.f = e.f();

  for (int i = 0; i < n; i++) {
    r.g[static_cast<std::size_t>(i)] = e.g(i);
  }
}

// --- Ceres Jet
//
// First order by design. The gradient lives in a fixed-size Eigen vector, so
// there is no dynamic counterpart to measure.

template <int TSize> void ceres_jet(const std::vector<double> &x, Result &r) {
  using J = ceres::Jet<double, TSize>;

  std::array<J, TSize> v;

  for (int i = 0; i < TSize; i++) {
    auto &jet = v[static_cast<std::size_t>(i)];

    jet.a = x[static_cast<std::size_t>(i)];
    jet.v.setZero();
    jet.v[i] = 1.0;
  }

  const auto e = energy(TSize, [&](const int i) { return v[i]; });

  r.f = e.a;

  for (int i = 0; i < TSize; i++) {
    r.g[static_cast<std::size_t>(i)] = e.v[i];
  }
}

// --- Eigen AutoDiffScalar, single level

template <int TSize>
void eigen_first_order(const std::vector<double> &x, Result &r) {
  using AD = Eigen::AutoDiffScalar<Eigen::Matrix<double, TSize, 1>>;

  Eigen::Matrix<AD, TSize, 1> v;

  for (int i = 0; i < TSize; i++) {
    v(i).value() = x[static_cast<std::size_t>(i)];
    v(i).derivatives() = Eigen::Matrix<double, TSize, 1>::Unit(i);
  }

  const auto e = energy(TSize, [&](const int i) { return v(i); });

  r.f = e.value();

  for (int i = 0; i < TSize; i++) {
    r.g[static_cast<std::size_t>(i)] = e.derivatives()(i);
  }
}

// --- autodiff, dual

void autodiff_dual(const std::vector<double> &x, Result &r) {
  const auto n = static_cast<int>(x.size());

  autodiff::VectorXdual v(n);

  for (int i = 0; i < n; i++) {
    v(i) = x[static_cast<std::size_t>(i)];
  }

  const auto fn = [n](const auto &values) {
    return energy(n, [&](const int i) { return values(i); });
  };

  autodiff::dual u;
  const Eigen::VectorXd g =
      autodiff::gradient(fn, autodiff::wrt(v), autodiff::at(v), u);

  r.f = static_cast<double>(u);

  for (int i = 0; i < n; i++) {
    r.g[static_cast<std::size_t>(i)] = g(i);
  }
}

// --- Central finite differences, gradient only

void finite_differences_g(const std::vector<double> &x, Result &r) {
  const auto n = static_cast<int>(x.size());

  // cbrt(eps) suits second derivatives; a gradient alone does better with
  // sqrt(eps), and giving finite differences its best shot is the fair way to
  // include it
  const double h = std::sqrt(std::numeric_limits<double>::epsilon());

  const auto eval = [](const std::vector<double> &values) {
    const auto m = static_cast<int>(values.size());

    return energy(
        m, [&](const int i) { return values[static_cast<std::size_t>(i)]; });
  };

  r.f = eval(x);

  for (int i = 0; i < n; i++) {
    auto plus = x;
    auto minus = x;

    plus[static_cast<std::size_t>(i)] += h;
    minus[static_cast<std::size_t>(i)] -= h;

    r.g[static_cast<std::size_t>(i)] = (eval(plus) - eval(minus)) / (2.0 * h);
  }
}

// --- Verification
//
// Timing a contender that computes the wrong thing is worthless, so every
// contender is checked against HyperJet before any measurement runs.

double deviation(const Result &a, const Result &b) {
  auto worst = std::abs(a.f - b.f);

  for (std::size_t i = 0; i < a.g.size(); i++) {
    worst = std::max(worst, std::abs(a.g[i] - b.g[i]));
  }

  for (std::size_t i = 0; i < a.h.size(); i++) {
    worst = std::max(worst, std::abs(a.h[i] - b.h[i]));
  }

  return worst;
}

template <int TSize> void verify() {
  const auto x = point(TSize);

  Result reference;
  reference.resize(TSize);
  hyperjet_static<TSize>(x, reference);

  const auto run = [&](auto fn) {
    Result r;
    r.resize(TSize);
    fn(x, r);

    return deviation(reference, r);
  };

  const struct {
    const char *name;
    double worst;
    double tolerance;
  } checks[] = {
      {"HyperJet dynamic", run(hyperjet_dynamic), 0.0},
      {"Eigen nested", run(eigen_nested<TSize>), 1e-15},
      {"autodiff dual2nd", run(autodiff_dual2nd), 1e-15},
      // finite differences are the odd one out: no tolerance this tight can
      // hold, and the number itself is the interesting result
      {"finite differences", run(finite_differences), 1e-4},
  };

  for (const auto &check : checks) {
    std::printf("  n=%-3d %-20s worst deviation %.3e %s\n", TSize, check.name,
                check.worst,
                check.worst <= check.tolerance ? "" : "<-- FAILED");

    if (check.worst > check.tolerance) {
      std::printf("\nA contender disagrees with HyperJet; timings would be "
                  "meaningless.\n");
      std::abort();
    }
  }
}

template <int TSize> void verify_first_order() {
  const auto x = point(TSize);

  Result reference;
  reference.resize_gradient(TSize);
  hyperjet_static_g<TSize>(x, reference);

  const auto run = [&](auto fn) {
    Result r;
    r.resize_gradient(TSize);
    fn(x, r);

    return deviation(reference, r);
  };

  const struct {
    const char *name;
    double worst;
    double tolerance;
  } checks[] = {
      {"HyperJet dynamic", run(hyperjet_dynamic_g), 0.0},
      {"Ceres Jet", run(ceres_jet<TSize>), 1e-16},
      {"Eigen AutoDiffScalar", run(eigen_first_order<TSize>), 1e-16},
      {"autodiff dual", run(autodiff_dual), 1e-16},
      {"finite differences", run(finite_differences_g), 1e-7},
  };

  for (const auto &check : checks) {
    std::printf("  n=%-3d %-20s worst deviation %.3e %s\n", TSize, check.name,
                check.worst,
                check.worst <= check.tolerance ? "" : "<-- FAILED");

    if (check.worst > check.tolerance) {
      std::printf("\nA contender disagrees with HyperJet; timings would be "
                  "meaningless.\n");
      std::abort();
    }
  }
}

// --- Benchmarks

template <int TSize> void hyperjet_static_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize(TSize);

  for (auto _ : state) {
    // The input is loop-invariant and the computation a pure function of it,
    // so without this the whole thing can be hoisted out of the loop and the
    // measurement becomes fiction.
    benchmark::DoNotOptimize(x);
    hyperjet_static<TSize>(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void hyperjet_dynamic_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize(TSize);

  for (auto _ : state) {
    // The input is loop-invariant and the computation a pure function of it,
    // so without this the whole thing can be hoisted out of the loop and the
    // measurement becomes fiction.
    benchmark::DoNotOptimize(x);
    hyperjet_dynamic(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void eigen_nested_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize(TSize);

  for (auto _ : state) {
    // The input is loop-invariant and the computation a pure function of it,
    // so without this the whole thing can be hoisted out of the loop and the
    // measurement becomes fiction.
    benchmark::DoNotOptimize(x);
    eigen_nested<TSize>(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void autodiff_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize(TSize);

  for (auto _ : state) {
    // The input is loop-invariant and the computation a pure function of it,
    // so without this the whole thing can be hoisted out of the loop and the
    // measurement becomes fiction.
    benchmark::DoNotOptimize(x);
    autodiff_dual2nd(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void finite_differences_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize(TSize);

  for (auto _ : state) {
    // The input is loop-invariant and the computation a pure function of it,
    // so without this the whole thing can be hoisted out of the loop and the
    // measurement becomes fiction.
    benchmark::DoNotOptimize(x);
    finite_differences(x, r);
    benchmark::DoNotOptimize(r);
  }
}

// first order

template <int TSize> void hyperjet_static_g_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize_gradient(TSize);

  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    hyperjet_static_g<TSize>(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void hyperjet_dynamic_g_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize_gradient(TSize);

  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    hyperjet_dynamic_g(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void ceres_jet_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize_gradient(TSize);

  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    ceres_jet<TSize>(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void eigen_first_order_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize_gradient(TSize);

  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    eigen_first_order<TSize>(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void autodiff_dual_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize_gradient(TSize);

  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    autodiff_dual(x, r);
    benchmark::DoNotOptimize(r);
  }
}

template <int TSize> void finite_differences_g_bm(benchmark::State &state) {
  auto x = point(TSize);

  Result r;
  r.resize_gradient(TSize);

  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    finite_differences_g(x, r);
    benchmark::DoNotOptimize(r);
  }
}

} // namespace

#define COMPARE_FIRST_ORDER(size)                                              \
  BENCHMARK_TEMPLATE(hyperjet_static_g_bm, size);                              \
  BENCHMARK_TEMPLATE(hyperjet_dynamic_g_bm, size);                             \
  BENCHMARK_TEMPLATE(ceres_jet_bm, size);                                      \
  BENCHMARK_TEMPLATE(eigen_first_order_bm, size);                              \
  BENCHMARK_TEMPLATE(autodiff_dual_bm, size);                                  \
  BENCHMARK_TEMPLATE(finite_differences_g_bm, size);

COMPARE_FIRST_ORDER(3)
COMPARE_FIRST_ORDER(6)
COMPARE_FIRST_ORDER(12)
COMPARE_FIRST_ORDER(24)

#define COMPARE(size)                                                          \
  BENCHMARK_TEMPLATE(hyperjet_static_bm, size);                                \
  BENCHMARK_TEMPLATE(hyperjet_dynamic_bm, size);                               \
  BENCHMARK_TEMPLATE(eigen_nested_bm, size);                                   \
  BENCHMARK_TEMPLATE(autodiff_bm, size);                                       \
  BENCHMARK_TEMPLATE(finite_differences_bm, size);

COMPARE(3)
COMPARE(6)
COMPARE(12)
COMPARE(24)

int main(int argc, char **argv) {
  std::printf("Verifying first order:\n");

  verify_first_order<3>();
  verify_first_order<6>();
  verify_first_order<12>();
  verify_first_order<24>();

  std::printf("\nVerifying first and second order:\n");

  verify<3>();
  verify<6>();
  verify<12>();
  verify<24>();

  std::printf("\n");

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
