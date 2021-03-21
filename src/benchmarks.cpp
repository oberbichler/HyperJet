#include <hyperjet.h>

#include <benchmark/benchmark.h>

namespace hj = hyperjet;

//                                    f    g0   g1   g2   h00  h01  h02  h11  h12  h22
const hj::DDScalar<2, double, 3> dd1 {3.0, 1.0, 6.0, 4.0, 0.0, 5.0, 9.0, 2.0, 7.0, 8.0};
const hj::DDScalar<2, double, 3> dd2 {4.0, 7.0, 1.0, 0.0, 6.0, 8.0, 2.0, 9.0, 5.0, 3.0};
const hj::DDScalar<2, double, 3> dd3 {0.3, 0.1, 0.8, 0.2, 0.5, 0.7, 0.9, 0.4, 0.6, 0.0};

#define HJ_BENCHMARK(method) \
BENCHMARK_TEMPLATE(method, 1); \
BENCHMARK_TEMPLATE(method, 2); \
BENCHMARK_TEMPLATE(method, 3); \
BENCHMARK_TEMPLATE(method, 4); \
BENCHMARK_TEMPLATE(method, 5); \
BENCHMARK_TEMPLATE(method, 6); \
BENCHMARK_TEMPLATE(method, 7); \
BENCHMARK_TEMPLATE(method, 8); \
BENCHMARK_TEMPLATE(method, -1)->DenseRange(1, 8); \

// add

template <hj::index TSize>
static void add_dd_s(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1 + 3.5;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(add_dd_s);

template <hj::index TSize>
static void add_s_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = 3.5 + dd1;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(add_s_dd);

template <hj::index TSize>
static void add_dd_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    const auto dd2 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1 + dd2;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(add_dd_dd);

// sub

template <hj::index TSize>
static void sub_dd_s(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1 - 3.5;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(sub_dd_s);

template <hj::index TSize>
static void sub_s_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = 3.5 - dd1;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(sub_s_dd);

template <hj::index TSize>
static void sub_dd_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    const auto dd2 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1 - dd2;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(sub_dd_dd);

// mul

template <hj::index TSize>
static void mul_dd_s(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1 * 3.5;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(mul_dd_s);

template <hj::index TSize>
static void mul_s_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = 3.5 * dd1;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(mul_s_dd);

template <hj::index TSize>
static void mul_dd_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    const auto dd2 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1 * dd2;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(mul_dd_dd);

// div

template <hj::index TSize>
static void div_dd_s(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1 / 3.5;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(div_dd_s);

template <hj::index TSize>
static void div_s_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = 3.5 / dd1;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(div_s_dd);

template <hj::index TSize>
static void div_dd_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    const auto dd2 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1 / dd2;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(div_dd_dd);

// iadd

template <hj::index TSize>
static void iadd_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    const auto dd2 = hj::DDScalar<2, double, TSize>::empty(s);
    auto r = dd1;
    for (auto _ : state) {
        r += dd2;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(iadd_dd);

// isub

template <hj::index TSize>
static void isub_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    const auto dd2 = hj::DDScalar<2, double, TSize>::empty(s);
    auto r = dd1;
    for (auto _ : state) {
        r -= dd2;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(isub_dd);

// imul

template <hj::index TSize>
static void imul_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    const auto dd2 = hj::DDScalar<2, double, TSize>::empty(s);
    auto r = dd1;
    for (auto _ : state) {
        r *= dd2;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(imul_dd);

// idiv

template <hj::index TSize>
static void idiv_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    const auto dd2 = hj::DDScalar<2, double, TSize>::empty(s);
    auto r = dd1;
    for (auto _ : state) {
        r /= dd2;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(idiv_dd);

// abs

template <hj::index TSize>
static void abs_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::abs;
    for (auto _ : state) {
        const auto r = abs(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(abs_dd);

// neg

template <hj::index TSize>
static void neg_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = -dd1;
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(neg_dd);

// pow

template <hj::index TSize>
static void pow_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::pow;
    for (auto _ : state) {
        const auto r = pow(dd1, 3.5);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(pow_dd);

// sqrt

template <hj::index TSize>
static void sqrt_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::sqrt;
    for (auto _ : state) {
        const auto r = sqrt(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(sqrt_dd);

// cbrt

template <hj::index TSize>
static void cbrt_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::cbrt;
    for (auto _ : state) {
        const auto r = cbrt(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(cbrt_dd);

// reciprocal

template <hj::index TSize>
static void reciprocal_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1.reciprocal();
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(reciprocal_dd);

// cos

template <hj::index TSize>
static void cos_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::cos;
    for (auto _ : state) {
        const auto r = cos(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(cos_dd);

// sin

template <hj::index TSize>
static void sin_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::sin;
    for (auto _ : state) {
        const auto r = sin(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(sin_dd);

// tan

template <hj::index TSize>
static void tan_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::tan;
    for (auto _ : state) {
        const auto r = tan(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(tan_dd);

// acos

template <hj::index TSize>
static void acos_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::tan;
    using std::acos;
    const auto a = acos(dd1);
    for (auto _ : state) {
        const auto r = acos(a);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(acos_dd);

// asin

template <hj::index TSize>
static void asin_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::tan;
    using std::asin;
    const auto a = asin(dd1);
    for (auto _ : state) {
        const auto r = asin(a);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(asin_dd);

// atan

template <hj::index TSize>
static void atan_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::tan;
    using std::atan;
    const auto a = atan(dd1);
    for (auto _ : state) {
        const auto r = atan(a);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(atan_dd);

// atan2

template <hj::index TSize>
static void atan2_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::cos;
    using std::sin;
    using std::atan2;
    const auto a = cos(dd1);
    const auto b = sin(dd1);
    for (auto _ : state) {
        const auto r = atan2(b, a);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(atan2_dd);

// cosh

template <hj::index TSize>
static void cosh_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::cosh;
    for (auto _ : state) {
        const auto r = cosh(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(cosh_dd);

// sinh

template <hj::index TSize>
static void sinh_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::sinh;
    for (auto _ : state) {
        const auto r = sinh(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(sinh_dd);

// tanh

template <hj::index TSize>
static void tanh_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::tanh;
    for (auto _ : state) {
        const auto r = tanh(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(tanh_dd);

// cosh

template <hj::index TSize>
static void acosh_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::acosh;
    using std::cosh;
    const auto a = cosh(dd1);
    for (auto _ : state) {
        const auto r = acosh(a);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(acosh_dd);

// sinh

template <hj::index TSize>
static void asinh_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::asinh;
    using std::sinh;
    const auto a = sinh(dd1);
    for (auto _ : state) {
        const auto r = asinh(a);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(asinh_dd);

// tanh

template <hj::index TSize>
static void atanh_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::atanh;
    using std::tanh;
    const auto a = tanh(dd1);
    for (auto _ : state) {
        const auto r = atanh(a);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(atanh_dd);

// exp

template <hj::index TSize>
static void exp_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::exp;
    for (auto _ : state) {
        const auto r = exp(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(exp_dd);

// log

template <hj::index TSize>
static void log_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::log;
    for (auto _ : state) {
        const auto r = log(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(log_dd);

// log2

template <hj::index TSize>
static void log2_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::log2;
    for (auto _ : state) {
        const auto r = log2(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(log2_dd);

// log10

template <hj::index TSize>
static void log10_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    using std::log10;
    for (auto _ : state) {
        const auto r = log10(dd1);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(log10_dd);

// logn

template <hj::index TSize>
static void logn_dd(benchmark::State& state) {
    const hj::index s = TSize < 0 ? state.range(0) : TSize;
    const auto dd1 = hj::DDScalar<2, double, TSize>::empty(s);
    for (auto _ : state) {
        const auto r = dd1.log(3.5);
        benchmark::DoNotOptimize(r);
    }
}
HJ_BENCHMARK(logn_dd);

//

BENCHMARK_MAIN();