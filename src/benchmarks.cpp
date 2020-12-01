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

TEST_CASE("Benchmarks") {
    // --- neg

    BENCHMARK("-DD") {
        return -dd1;
    };

    // --- add

    BENCHMARK("DD + DD") {
        return dd1 + dd2;
    };

    BENCHMARK("DD + S") {
        return dd1 + 3.5;
    };

    BENCHMARK("S + DD") {
        return 3.5 + dd1;
    };

    BENCHMARK("DD += DD") {
        auto r = dd1;
        r += dd2;
        return r;
    };

    BENCHMARK("DD += S") {
        auto r = dd1;
        r += 3.5;
        return r;
    };

    // --- sub

    BENCHMARK("DD - DD") {
        return dd1 - dd2;
    };

    BENCHMARK("DD - S") {
        return dd1 - 3.5;
    };

    BENCHMARK("S - DD") {
        return 3.5 - dd1;
    };

    BENCHMARK("DD -= DD") {
        auto r = dd1;
        r -= dd2;
        return r;
    };

    BENCHMARK("DD -= S") {
        auto r = dd1;
        r -= 3.5;
        return r;
    };

    // --- mul

    BENCHMARK("DD * DD") {
        return dd1 * dd2;
    };

    BENCHMARK("S * DD") {
        return 3.5 * dd1;
    };

    BENCHMARK("DD * S") {
        return dd1 * 3.5;
    };

    // --- div

    BENCHMARK("DD / DD") {
        return dd1 / dd2;
    };

    BENCHMARK("DD / S") {
        return dd1 / 3.5;
    };

    BENCHMARK("S / DD") {
        return 3.5 / dd1;
    };

    // --- pow

    BENCHMARK("DD^S") {
        using std::pow;
        return pow(dd1, 3.5);
    };

    // --- sqrt

    BENCHMARK("Sqrt(DD)") {
        using std::sqrt;
        return sqrt(dd1);
    };

    // --- trig

    BENCHMARK("Cos(DD)") {
        using std::cos;
        return cos(dd1);
    };

    BENCHMARK("Sin(DD)") {
        using std::sin;
        return sin(dd1);
    };

    BENCHMARK("Tan(DD)") {
        using std::tan;
        return tan(dd1);
    };

    BENCHMARK("Acos(DD)") {
        using std::acos;
        return acos(dd1);
    };

    BENCHMARK("Asin(DD)") {
        using std::asin;
        return asin(dd1);
    };

    BENCHMARK("Atan(DD)") {
        using std::atan;
        return atan(dd1);
    };

    BENCHMARK("Atan2(DD, DD)") {
        using std::atan2;
        return atan2(dd1, dd2);
    };

    // --- dot

    BENCHMARK("DD3 . DD3") {
        return ddv1.dot(ddv2);
    };

    // --- cross

    BENCHMARK("DD3 x DD3") {
        return ddv1.cross(ddv2);
    };

    // --- norm

    BENCHMARK("| DD3 |") {
        return ddv1.norm();
    };

    // --- scale

    BENCHMARK("S * DD3") {
        return 3.5 * ddv1;
    };
}