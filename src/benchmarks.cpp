#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry> // cross

#include <hyperjet.h>

using namespace hyperjet;

//                             f    g0   g1   g2   h00  h01  h02  h11  h12  h22
const DDScalar<double, 3> dd1 {3.0, 1.0, 6.0, 4.0, 0.0, 5.0, 9.0, 2.0, 7.0, 8.0};
const DDScalar<double, 3> dd2 {4.0, 7.0, 1.0, 0.0, 6.0, 8.0, 2.0, 9.0, 5.0, 3.0};
const DDScalar<double, 3> dd3 {0.3, 0.1, 0.8, 0.2, 0.5, 0.7, 0.9, 0.4, 0.6, 0.0};

const Eigen::Matrix<DDScalar<double, 3>, 1, 3> ddv1 {dd1, dd2, dd3};
const Eigen::Matrix<DDScalar<double, 3>, 1, 3> ddv2 {dd3, dd1, dd2};

TEMPLATE_TEST_CASE("Benchmarks", "", (DDScalar<double, 3>), (DDScalar<double, 15>), (DDScalar<double, Dynamic>)) {
    const TestType dd1(TestType::is_dynamic() ? 15 : TestType::static_size());
    const TestType dd2(TestType::is_dynamic() ? 15 : TestType::static_size());
    const TestType dd3(TestType::is_dynamic() ? 15 : TestType::static_size());

    // --- neg

    BENCHMARK("-DDScalar") {
        return -dd1;
    };

    // --- add

    BENCHMARK("DDScalar + DDScalar") {
        return dd1 + dd2;
    };

    BENCHMARK("DDScalar + Scalar") {
        return dd1 + 3.5;
    };

    BENCHMARK("Scalar + DDScalar") {
        return 3.5 + dd1;
    };

    BENCHMARK("DDScalar += DDScalar") {
        auto r = dd1;
        r += dd2;
        return r;
    };

    BENCHMARK("DDScalar += Scalar") {
        auto r = dd1;
        r += 3.5;
        return r;
    };

    // --- sub

    BENCHMARK("DDScalar - DDScalar") {
        return dd1 - dd2;
    };

    BENCHMARK("DDScalar - Scalar") {
        return dd1 - 3.5;
    };

    BENCHMARK("Scalar - DDScalar") {
        return 3.5 - dd1;
    };

    BENCHMARK("DDScalar -= DDScalar") {
        auto r = dd1;
        r -= dd2;
        return r;
    };

    BENCHMARK("DDScalar -= Scalar") {
        auto r = dd1;
        r -= 3.5;
        return r;
    };

    // --- mul

    BENCHMARK("DDScalar * DDScalar") {
        return dd1 * dd2;
    };

    BENCHMARK("Scalar * DDScalar") {
        return 3.5 * dd1;
    };

    BENCHMARK("DDScalar * Scalar") {
        return dd1 * 3.5;
    };

    BENCHMARK("DDScalar *= DDScalar") {
        auto r = dd1;
        r *= dd2;
        return r;
    };

    BENCHMARK("DDScalar *= Scalar") {
        auto r = dd1;
        r *= 3.5;
        return r;
    };

    // --- div

    BENCHMARK("DDScalar / DDScalar") {
        return dd1 / dd2;
    };

    BENCHMARK("DDScalar / Scalar") {
        return dd1 / 3.5;
    };

    BENCHMARK("Scalar / DDScalar") {
        return 3.5 / dd1;
    };

    BENCHMARK("DDScalar /= DDScalar") {
        auto r = dd1;
        r /= dd2;
        return r;
    };

    BENCHMARK("DDScalar /= Scalar") {
        auto r = dd1;
        r /= 3.5;
        return r;
    };

    // --- pow

    BENCHMARK("DDScalar^Scalar") {
        using std::pow;
        return pow(dd1, 3.5);
    };

    // --- sqrt

    BENCHMARK("Sqrt(DDScalar)") {
        using std::sqrt;
        return sqrt(dd1);
    };

    // --- cbrt

    BENCHMARK("Cbrt(DDScalar)") {
        using std::cbrt;
        return cbrt(dd1);
    };

    // --- trig

    BENCHMARK("Cos(DDScalar)") {
        using std::cos;
        return cos(dd1);
    };

    BENCHMARK("Sin(DDScalar)") {
        using std::sin;
        return sin(dd1);
    };

    BENCHMARK("Tan(DDScalar)") {
        using std::tan;
        return tan(dd1);
    };

    BENCHMARK("Acos(DDScalar)") {
        using std::acos;
        return acos(dd1);
    };

    BENCHMARK("Asin(DDScalar)") {
        using std::asin;
        return asin(dd1);
    };

    BENCHMARK("Atan(DDScalar)") {
        using std::atan;
        return atan(dd1);
    };

    BENCHMARK("Atan2(DDScalar, DDScalar)") {
        using std::atan2;
        return atan2(dd1, dd2);
    };

    BENCHMARK("Cosh(DDScalar)") {
        using std::cosh;
        return cosh(dd1);
    };

    BENCHMARK("Sinh(DDScalar)") {
        using std::sinh;
        return sinh(dd1);
    };

    BENCHMARK("Tanh(DDScalar)") {
        using std::tanh;
        return tanh(dd1);
    };

    BENCHMARK("Acosh(DDScalar)") {
        using std::acosh;
        return acosh(dd1);
    };

    BENCHMARK("Asinh(DDScalar)") {
        using std::asinh;
        return asinh(dd1);
    };

    BENCHMARK("Atanh(DDScalar)") {
        using std::atanh;
        return atanh(dd1);
    };

    // ---

    BENCHMARK("Exp(DDScalar)") {
        using std::exp;
        return exp(dd1);
    };

    BENCHMARK("Log(DDScalar)") {
        using std::log;
        return log(dd1);
    };

    BENCHMARK("Log2(DDScalar)") {
        using std::log2;
        return log2(dd1);
    };

    BENCHMARK("Log10(DDScalar)") {
        using std::log10;
        return log10(dd1);
    };

    // // --- dot

    // BENCHMARK("DD3 . DD3") {
    //     return ddv1.dot(ddv2);
    // };

    // // --- cross

    // BENCHMARK("DD3 x DD3") {
    //     return ddv1.cross(ddv2);
    // };

    // // --- norm

    // BENCHMARK("| DD3 |") {
    //     return ddv1.norm();
    // };

    // // --- scale

    // BENCHMARK("Scalar * DD3") {
    //     return 3.5 * ddv1;
    // };
}