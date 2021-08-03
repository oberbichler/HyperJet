#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry> // cross

#include <hyperjet.h>

#include <sstream> // stringstream

using namespace Catch::literals;
using namespace hyperjet;

//                               f    g0   g1   g2   h00  h01  h02  h11  h12  h22
const DDScalar<2, double, 3> dd1{3.0, 1.0, 6.0, 4.0, 0.0, 5.0, 9.0, 2.0, 7.0, 8.0};
const DDScalar<2, double, 3> dd2{4.0, 7.0, 1.0, 0.0, 6.0, 8.0, 2.0, 9.0, 5.0, 3.0};
const DDScalar<2, double, 3> dd3{0.3, 0.1, 0.8, 0.2, 0.5, 0.7, 0.9, 0.4, 0.6, 0.0};

const Eigen::Matrix<DDScalar<2, double, 3>, 1, 3> ddv1{dd1, dd2, dd3};
const Eigen::Matrix<DDScalar<2, double, 3>, 1, 3> ddv2{dd3, dd1, dd2};

TEST_CASE("<< DD")
{
    std::stringstream output;

    output << dd3;

    REQUIRE(output.str() == "0.3hj");
}

TEST_CASE("Neg")
{
    const auto r1 = -dd1;

    REQUIRE(r1.f() == -3_a);
    REQUIRE(r1.g(0) == -1_a);
    REQUIRE(r1.g(1) == -6_a);
    REQUIRE(r1.g(2) == -4_a);
    REQUIRE(r1.h(0, 0) == -0_a);
    REQUIRE(r1.h(0, 1) == -5_a);
    REQUIRE(r1.h(0, 2) == -9_a);
    REQUIRE(r1.h(1, 1) == -2_a);
    REQUIRE(r1.h(1, 2) == -7_a);
    REQUIRE(r1.h(2, 2) == -8_a);
}

TEST_CASE("Add")
{
    const auto r1 = dd1 + dd2;

    REQUIRE(r1.f() == 7_a);
    REQUIRE(r1.g(0) == 8_a);
    REQUIRE(r1.g(1) == 7_a);
    REQUIRE(r1.g(2) == 4_a);
    REQUIRE(r1.h(0, 0) == 6_a);
    REQUIRE(r1.h(0, 1) == 13_a);
    REQUIRE(r1.h(0, 2) == 11_a);
    REQUIRE(r1.h(1, 1) == 11_a);
    REQUIRE(r1.h(1, 2) == 12_a);
    REQUIRE(r1.h(2, 2) == 11_a);

    const auto r2 = dd1 + 3.5;

    REQUIRE(r2.f() == 6.5_a);
    REQUIRE(r2.g(0) == 1.0_a);
    REQUIRE(r2.g(1) == 6.0_a);
    REQUIRE(r2.g(2) == 4.0_a);
    REQUIRE(r2.h(0, 0) == 0.0_a);
    REQUIRE(r2.h(0, 1) == 5.0_a);
    REQUIRE(r2.h(0, 2) == 9.0_a);
    REQUIRE(r2.h(1, 1) == 2.0_a);
    REQUIRE(r2.h(1, 2) == 7.0_a);
    REQUIRE(r2.h(2, 2) == 8.0_a);

    const auto r3 = 3.5 + dd1;

    REQUIRE(r3.f() == 6.5_a);
    REQUIRE(r3.g(0) == 1.0_a);
    REQUIRE(r3.g(1) == 6.0_a);
    REQUIRE(r3.g(2) == 4.0_a);
    REQUIRE(r3.h(0, 0) == 0.0_a);
    REQUIRE(r3.h(0, 1) == 5.0_a);
    REQUIRE(r3.h(0, 2) == 9.0_a);
    REQUIRE(r3.h(1, 1) == 2.0_a);
    REQUIRE(r3.h(1, 2) == 7.0_a);
    REQUIRE(r3.h(2, 2) == 8.0_a);
}

TEST_CASE("IncAdd")
{
    auto r1 = dd1;
    r1 += dd2;

    REQUIRE(r1.f() == 7_a);
    REQUIRE(r1.g(0) == 8_a);
    REQUIRE(r1.g(1) == 7_a);
    REQUIRE(r1.g(2) == 4_a);
    REQUIRE(r1.h(0, 0) == 6_a);
    REQUIRE(r1.h(0, 1) == 13_a);
    REQUIRE(r1.h(0, 2) == 11_a);
    REQUIRE(r1.h(1, 1) == 11_a);
    REQUIRE(r1.h(1, 2) == 12_a);
    REQUIRE(r1.h(2, 2) == 11_a);

    auto r2 = dd1;
    r2 += 3.5;

    REQUIRE(r2.f() == 6.5_a);
    REQUIRE(r2.g(0) == 1.0_a);
    REQUIRE(r2.g(1) == 6.0_a);
    REQUIRE(r2.g(2) == 4.0_a);
    REQUIRE(r2.h(0, 0) == 0.0_a);
    REQUIRE(r2.h(0, 1) == 5.0_a);
    REQUIRE(r2.h(0, 2) == 9.0_a);
    REQUIRE(r2.h(1, 1) == 2.0_a);
    REQUIRE(r2.h(1, 2) == 7.0_a);
    REQUIRE(r2.h(2, 2) == 8.0_a);
}

TEST_CASE("Sub")
{
    const auto r1 = dd1 - dd2;

    REQUIRE(r1.f() == -1.0_a);
    REQUIRE(r1.g(0) == -6.0_a);
    REQUIRE(r1.g(1) == 5.0_a);
    REQUIRE(r1.g(2) == 4.0_a);
    REQUIRE(r1.h(0, 0) == -6.0_a);
    REQUIRE(r1.h(0, 1) == -3.0_a);
    REQUIRE(r1.h(0, 2) == 7.0_a);
    REQUIRE(r1.h(1, 1) == -7.0_a);
    REQUIRE(r1.h(1, 2) == 2.0_a);
    REQUIRE(r1.h(2, 2) == 5.0_a);

    const auto r2 = dd1 - 3.5;

    REQUIRE(r2.f() == -0.5_a);
    REQUIRE(r2.g(0) == 1.0_a);
    REQUIRE(r2.g(1) == 6.0_a);
    REQUIRE(r2.g(2) == 4.0_a);
    REQUIRE(r2.h(0, 0) == 0.0_a);
    REQUIRE(r2.h(0, 1) == 5.0_a);
    REQUIRE(r2.h(0, 2) == 9.0_a);
    REQUIRE(r2.h(1, 1) == 2.0_a);
    REQUIRE(r2.h(1, 2) == 7.0_a);
    REQUIRE(r2.h(2, 2) == 8.0_a);

    const auto r3 = 3.5 - dd1;

    REQUIRE(r3.f() == 0.5_a);
    REQUIRE(r3.g(0) == -1.0_a);
    REQUIRE(r3.g(1) == -6.0_a);
    REQUIRE(r3.g(2) == -4.0_a);
    REQUIRE(r3.h(0, 0) == -0.0_a);
    REQUIRE(r3.h(0, 1) == -5.0_a);
    REQUIRE(r3.h(0, 2) == -9.0_a);
    REQUIRE(r3.h(1, 1) == -2.0_a);
    REQUIRE(r3.h(1, 2) == -7.0_a);
    REQUIRE(r3.h(2, 2) == -8.0_a);
}

TEST_CASE("IncSub")
{
    auto r1 = dd1;
    r1 -= dd2;

    REQUIRE(r1.f() == -1.0_a);
    REQUIRE(r1.g(0) == -6.0_a);
    REQUIRE(r1.g(1) == 5.0_a);
    REQUIRE(r1.g(2) == 4.0_a);
    REQUIRE(r1.h(0, 0) == -6.0_a);
    REQUIRE(r1.h(0, 1) == -3.0_a);
    REQUIRE(r1.h(0, 2) == 7.0_a);
    REQUIRE(r1.h(1, 1) == -7.0_a);
    REQUIRE(r1.h(1, 2) == 2.0_a);
    REQUIRE(r1.h(2, 2) == 5.0_a);

    auto r2 = dd1;
    r2 -= 3.5;

    REQUIRE(r2.f() == -0.5_a);
    REQUIRE(r2.g(0) == 1.0_a);
    REQUIRE(r2.g(1) == 6.0_a);
    REQUIRE(r2.g(2) == 4.0_a);
    REQUIRE(r2.h(0, 0) == 0.0_a);
    REQUIRE(r2.h(0, 1) == 5.0_a);
    REQUIRE(r2.h(0, 2) == 9.0_a);
    REQUIRE(r2.h(1, 1) == 2.0_a);
    REQUIRE(r2.h(1, 2) == 7.0_a);
    REQUIRE(r2.h(2, 2) == 8.0_a);
}

TEST_CASE("Mul")
{
    const auto r1 = dd1 * dd2;

    REQUIRE(r1.f() == 12.0_a);
    REQUIRE(r1.g(0) == 25.0_a);
    REQUIRE(r1.g(1) == 27.0_a);
    REQUIRE(r1.g(2) == 16.0_a);
    REQUIRE(r1.h(0, 0) == 32.0_a);
    REQUIRE(r1.h(0, 1) == 87.0_a);
    REQUIRE(r1.h(0, 2) == 70.0_a);
    REQUIRE(r1.h(1, 1) == 47.0_a);
    REQUIRE(r1.h(1, 2) == 47.0_a);
    REQUIRE(r1.h(2, 2) == 41.0_a);
}

TEST_CASE("IncMul")
{
    auto r1 = dd1;
    r1 *= dd2;

    REQUIRE(r1.f() == 12.0_a);
    REQUIRE(r1.g(0) == 25.0_a);
    REQUIRE(r1.g(1) == 27.0_a);
    REQUIRE(r1.g(2) == 16.0_a);
    REQUIRE(r1.h(0, 0) == 32.0_a);
    REQUIRE(r1.h(0, 1) == 87.0_a);
    REQUIRE(r1.h(0, 2) == 70.0_a);
    REQUIRE(r1.h(1, 1) == 47.0_a);
    REQUIRE(r1.h(1, 2) == 47.0_a);
    REQUIRE(r1.h(2, 2) == 41.0_a);
}

TEST_CASE("Div")
{
    const auto r1 = dd1 / dd2;

    REQUIRE(r1.f() == 0.75000_a);
    REQUIRE(r1.g(0) == -1.06250_a);
    REQUIRE(r1.g(1) == 1.31250_a);
    REQUIRE(r1.g(2) == 1.00000_a);
    REQUIRE(r1.h(0, 0) == 2.59375_a);
    REQUIRE(r1.h(0, 1) == -2.28125_a);
    REQUIRE(r1.h(0, 2) == 0.12500_a);
    REQUIRE(r1.h(1, 1) == -1.84375_a);
    REQUIRE(r1.h(1, 2) == 0.56250_a);
    REQUIRE(r1.h(2, 2) == 1.43750_a);
}

TEST_CASE("IncDiv")
{
    auto r1 = dd1;
    r1 /= dd2;

    REQUIRE(r1.f() == 0.75000_a);
    REQUIRE(r1.g(0) == -1.06250_a);
    REQUIRE(r1.g(1) == 1.31250_a);
    REQUIRE(r1.g(2) == 1.00000_a);
    REQUIRE(r1.h(0, 0) == 2.59375_a);
    REQUIRE(r1.h(0, 1) == -2.28125_a);
    REQUIRE(r1.h(0, 2) == 0.12500_a);
    REQUIRE(r1.h(1, 1) == -1.84375_a);
    REQUIRE(r1.h(1, 2) == 0.56250_a);
    REQUIRE(r1.h(2, 2) == 1.43750_a);
}

TEST_CASE("Pow")
{
    using std::pow;

    const auto r1 = pow(dd1, 3.5);

    REQUIRE(r1.f() == 46.765371804359690_a);
    REQUIRE(r1.g(0) == 54.559600438419636_a);
    REQUIRE(r1.g(1) == 327.357602630517800_a);
    REQUIRE(r1.g(2) == 218.238401753678540_a);
    REQUIRE(r1.h(0, 0) == 45.466333698683030_a);
    REQUIRE(r1.h(0, 1) == 545.596004384196400_a);
    REQUIRE(r1.h(0, 2) == 672.901738740508800_a);
    REQUIRE(r1.h(1, 1) == 1745.907214029428400_a);
    REQUIRE(r1.h(1, 2) == 1473.109211837330100_a);
    REQUIRE(r1.h(2, 2) == 1163.938142686285600_a);
}

TEST_CASE("Sqrt")
{
    using std::sqrt;

    const auto r1 = sqrt(dd1);

    REQUIRE(r1.f() == 1.732050807568877200_a);
    REQUIRE(r1.g(0) == 0.288675134594812900_a);
    REQUIRE(r1.g(1) == 1.732050807568877400_a);
    REQUIRE(r1.g(2) == 1.154700538379251700_a);
    REQUIRE(r1.h(0, 0) == -0.048112522432468816_a);
    REQUIRE(r1.h(0, 1) == 1.154700538379251700_a);
    REQUIRE(r1.h(0, 2) == 2.405626121623441000_a);
    REQUIRE(r1.h(1, 1) == -1.154700538379251700_a);
    REQUIRE(r1.h(1, 2) == 0.866025403784438700_a);
    REQUIRE(r1.h(2, 2) == 1.539600717839002300_a);
}

TEST_CASE("Cos")
{
    using std::cos;

    const auto r1 = cos(dd1);

    REQUIRE(r1.f() == -0.9899924966004454_a);
    REQUIRE(r1.g(0) == -0.1411200080598672_a);
    REQUIRE(r1.g(1) == -0.8467200483592032_a);
    REQUIRE(r1.g(2) == -0.5644800322394689_a);
    REQUIRE(r1.h(0, 0) == 0.9899924966004454_a);
    REQUIRE(r1.h(0, 1) == 5.2343549393033370_a);
    REQUIRE(r1.h(0, 2) == 2.6898899138629770_a);
    REQUIRE(r1.h(1, 1) == 35.3574898614963000_a);
    REQUIRE(r1.h(1, 2) == 22.7719798619916200_a);
    REQUIRE(r1.h(2, 2) == 14.7109198811281900_a);
}

TEST_CASE("Sin")
{
    using std::sin;

    const auto r1 = dd1.sin();

    REQUIRE(r1.f() == 0.1411200080598672_a);
    REQUIRE(r1.g(0) == -0.9899924966004454_a);
    REQUIRE(r1.g(1) == -5.9399549796026730_a);
    REQUIRE(r1.g(2) == -3.9599699864017817_a);
    REQUIRE(r1.h(0, 0) == -0.1411200080598672_a);
    REQUIRE(r1.h(0, 1) == -5.7966825313614300_a);
    REQUIRE(r1.h(0, 2) == -9.4744125016434780_a);
    REQUIRE(r1.h(1, 1) == -7.0603052833561100_a);
    REQUIRE(r1.h(1, 2) == -10.3168276696399310_a);
    REQUIRE(r1.h(2, 2) == -10.1778601017614400_a);
}

TEST_CASE("Tan")
{
    using std::tan;

    const auto r1 = tan(dd1);

    REQUIRE(r1.f() == -0.14254654307427780_a);
    REQUIRE(r1.g(0) == 1.02031951694242680_a);
    REQUIRE(r1.g(1) == 6.12191710165456100_a);
    REQUIRE(r1.g(2) == 4.08127806776970700_a);
    REQUIRE(r1.h(0, 0) == -0.29088603994271994_a);
    REQUIRE(r1.h(0, 1) == 3.35628134505581470_a);
    REQUIRE(r1.h(0, 2) == 8.01933149271096100_a);
    REQUIRE(r1.h(1, 1) == -8.43125840405306400_a);
    REQUIRE(r1.h(1, 2) == 0.16097165997170980_a);
    REQUIRE(r1.h(2, 2) == 3.50837949645589560_a);
}

TEST_CASE("Acos")
{
    using std::acos;

    const auto r1 = acos(dd3);

    REQUIRE(r1.f() == 1.266103672779499200_a);
    REQUIRE(r1.g(0) == -0.104828483672191830_a);
    REQUIRE(r1.g(1) == -0.838627869377534600_a);
    REQUIRE(r1.g(2) == -0.209656967344383660_a);
    REQUIRE(r1.h(0, 0) == -0.527598302438064400_a);
    REQUIRE(r1.h(0, 1) == -0.761446458322184600_a);
    REQUIRE(r1.h(0, 2) == -0.950368121203936900_a);
    REQUIRE(r1.h(1, 1) == -0.640490515623501800_a);
    REQUIRE(r1.h(1, 2) == -0.684265047266834600_a);
    REQUIRE(r1.h(2, 2) == -0.013823536308420903_a);
}

TEST_CASE("Asin")
{
    using std::asin;

    const auto r1 = asin(dd3);

    REQUIRE(r1.f() == 0.304692654015397500_a);
    REQUIRE(r1.g(0) == 0.104828483672191830_a);
    REQUIRE(r1.g(1) == 0.838627869377534600_a);
    REQUIRE(r1.g(2) == 0.209656967344383660_a);
    REQUIRE(r1.h(0, 0) == 0.527598302438064400_a);
    REQUIRE(r1.h(0, 1) == 0.761446458322184600_a);
    REQUIRE(r1.h(0, 2) == 0.950368121203936900_a);
    REQUIRE(r1.h(1, 1) == 0.640490515623501700_a);
    REQUIRE(r1.h(1, 2) == 0.684265047266834600_a);
    REQUIRE(r1.h(2, 2) == 0.013823536308420899_a);
}

TEST_CASE("Atan")
{
    using std::atan;

    const auto r1 = atan(dd3);

    REQUIRE(r1.f() == 0.291456794477867100_a);
    REQUIRE(r1.g(0) == 0.091743119266055050_a);
    REQUIRE(r1.g(1) == 0.733944954128440400_a);
    REQUIRE(r1.g(2) == 0.183486238532110100_a);
    REQUIRE(r1.h(0, 0) == 0.453665516370675800_a);
    REQUIRE(r1.h(0, 1) == 0.601801195185590400_a);
    REQUIRE(r1.h(0, 2) == 0.815587913475296700_a);
    REQUIRE(r1.h(1, 1) == 0.043767359649861170_a);
    REQUIRE(r1.h(1, 2) == 0.469657436242740500_a);
    REQUIRE(r1.h(2, 2) == -0.020200319838397436_a);
}

TEST_CASE("Atan2")
{
    using std::atan2;

    const auto r1 = atan2(dd1, dd2);

    REQUIRE(r1.f() == 0.6435011087932844_a);
    REQUIRE(r1.g(0) == -0.6800000000000000_a);
    REQUIRE(r1.g(1) == 0.8400000000000000_a);
    REQUIRE(r1.g(2) == 0.6400000000000000_a);
    REQUIRE(r1.h(0, 0) == 0.9664000000000000_a);
    REQUIRE(r1.h(0, 1) == -0.6032000000000000_a);
    REQUIRE(r1.h(0, 2) == 0.7328000000000000_a);
    REQUIRE(r1.h(1, 1) == -2.2384000000000000_a);
    REQUIRE(r1.h(1, 2) == -0.4464000000000000_a);
    REQUIRE(r1.h(2, 2) == 0.3056000000000000_a);
}

TEST_CASE("Dot")
{
    const auto r = ddv1.dot(ddv2);

    REQUIRE(r.f() == 14.1_a);
    REQUIRE(r.g(0) == 28.1_a);
    REQUIRE(r.g(1) == 34.7_a);
    REQUIRE(r.g(2) == 18.6_a);
    REQUIRE(r.h(0, 0) == 38.9_a);
    REQUIRE(r.h(0, 1) == 102.9_a);
    REQUIRE(r.h(0, 2) == 81.6_a);
    REQUIRE(r.h(1, 1) == 64.3_a);
    REQUIRE(r.h(1, 2) == 59.4_a);
    REQUIRE(r.h(2, 2) == 45.9_a);
}

TEST_CASE("Cross")
{
    const auto r = ddv1.cross(ddv2);

    const auto rx = r.x();
    const auto ry = r.y();
    const auto rz = r.z();

    REQUIRE(rx.f() == 15.1_a);
    REQUIRE(rx.g(0) == 55.4_a);
    REQUIRE(rx.g(1) == 3.8_a);
    REQUIRE(rx.g(2) == -1.8_a);
    REQUIRE(rx.h(0, 0) == 144.3_a);
    REQUIRE(rx.h(0, 1) == 73.0_a);
    REQUIRE(rx.h(0, 2) == 10.0_a);
    REQUIRE(rx.h(1, 1) == 62.6_a);
    REQUIRE(rx.h(1, 2) == 31.7_a);
    REQUIRE(rx.h(2, 2) == 20.0_a);

    REQUIRE(ry.f() == -11.91_a);
    REQUIRE(ry.g(0) == -24.94_a);
    REQUIRE(ry.g(1) == -26.52_a);
    REQUIRE(ry.g(2) == -15.88_a);
    REQUIRE(ry.h(0, 0) == -31.68_a);
    REQUIRE(ry.h(0, 1) == -86.42_a);
    REQUIRE(ry.h(0, 2) == -69.42_a);
    REQUIRE(ry.h(1, 1) == -45.48_a);
    REQUIRE(ry.h(1, 2) == -46.32_a);
    REQUIRE(ry.h(2, 2) == -40.92_a);

    REQUIRE(rz.f() == 7.8_a);
    REQUIRE(rz.g(0) == 3.5_a);
    REQUIRE(rz.g(1) == 32.5_a);
    REQUIRE(rz.g(2) == 23.2_a);
    REQUIRE(rz.h(0, 0) == -3.2_a);
    REQUIRE(rz.h(0, 1) == 31.1_a);
    REQUIRE(rz.h(0, 2) == 56.4_a);
    REQUIRE(rz.h(1, 1) == 78.1_a);
    REQUIRE(rz.h(1, 2) == 85.9_a);
    REQUIRE(rz.h(2, 2) == 79.1_a);
}

TEST_CASE("Norm")
{
    const auto r = ddv1.norm();

    REQUIRE(r.f() == 5.0089919145472770_a);
    REQUIRE(r.g(0) == 6.1948592709606230_a);
    REQUIRE(r.g(1) == 4.4400151526317835_a);
    REQUIRE(r.g(2) == 2.4076700872634587_a);
    REQUIRE(r.h(0, 0) == 7.1438962616547620_a);
    REQUIRE(r.h(0, 1) == 6.5451754620124020_a);
    REQUIRE(r.h(0, 2) == 4.8662132130242100_a);
    REQUIRE(r.h(1, 1) == 11.9876946237449200_a);
    REQUIRE(r.h(1, 2) == 10.9103606598557140_a);
    REQUIRE(r.h(2, 2) == 9.2320222391647280_a);
}


const SScalar<double> s1(3.0, {{"x", 1.0}, {"y", 6.0}, {"z", 4.0}});
const SScalar<double> s2(4.0, {{"x", 7.0}, {"y", 1.0}});
const SScalar<double> s3(0.3, {{"x", 0.1}, {"y", 0.8}, {"z", 0.2}});

TEST_CASE("init", "[SScalar]")
{
    using Dual = SScalar<double>;

    const auto x = Dual(1.5, {{"x", 2.0}, {"y", 1.0}});

    REQUIRE(x.size() == 2);

    REQUIRE(x.f() == 1.5_a);

    REQUIRE(x.d("x") == 2.0_a);
    REQUIRE(x.d("y") == 1.0_a);
    REQUIRE(x.d("z") == 0.0_a);
}

TEST_CASE("constant", "[SScalar]")
{
    using Dual = SScalar<double>;

    const auto x = Dual::constant(1.5);

    REQUIRE(x.size() == 0);

    REQUIRE(x.f() == 1.5_a);

    REQUIRE(x.d("x") == 0.0_a);
    REQUIRE(x.d("y") == 0.0_a);
}

TEST_CASE("variable", "[SScalar]")
{
    using Dual = SScalar<double>;

    const auto x = Dual::variable("x", 1.5);

    REQUIRE(x.size() == 1);

    REQUIRE(x.f() == 1.5_a);

    REQUIRE(x.d("x") == 1.0_a);
    REQUIRE(x.d("y") == 0.0_a);
}

TEST_CASE("Neg", "[SScalar]")
{
    const auto r1 = -s1;

    REQUIRE(r1.f() == -3_a);
    REQUIRE(r1.d("x") == -1_a);
    REQUIRE(r1.d("y") == -6_a);
    REQUIRE(r1.d("z") == -4_a);
}

TEST_CASE("Add", "[SScalar]")
{
    const auto r1 = s1 + s2;

    REQUIRE(r1.f() == 7_a);
    REQUIRE(r1.d("x") == 8_a);
    REQUIRE(r1.d("y") == 7_a);
    REQUIRE(r1.d("z") == 4_a);

    const auto r2 = s1 + 3.5;

    REQUIRE(r2.f() == 6.5_a);
    REQUIRE(r2.d("x") == 1.0_a);
    REQUIRE(r2.d("y") == 6.0_a);
    REQUIRE(r2.d("z") == 4.0_a);

    const auto r3 = 3.5 + s1;

    REQUIRE(r3.f() == 6.5_a);
    REQUIRE(r3.d("x") == 1.0_a);
    REQUIRE(r3.d("y") == 6.0_a);
    REQUIRE(r3.d("z") == 4.0_a);
}

TEST_CASE("IncAdd", "[SScalar]")
{
    auto r1 = s1;
    r1 += s2;

    REQUIRE(r1.f() == 7_a);
    REQUIRE(r1.d("x") == 8_a);
    REQUIRE(r1.d("y") == 7_a);
    REQUIRE(r1.d("z") == 4_a);

    auto r2 = s1;
    r2 += 3.5;

    REQUIRE(r2.f() == 6.5_a);
    REQUIRE(r2.d("x") == 1.0_a);
    REQUIRE(r2.d("y") == 6.0_a);
    REQUIRE(r2.d("z") == 4.0_a);
}

TEST_CASE("Sub", "[SScalar]")
{
    const auto r1 = s1 - s2;

    REQUIRE(r1.f() == -1.0_a);
    REQUIRE(r1.d("x") == -6.0_a);
    REQUIRE(r1.d("y") == 5.0_a);
    REQUIRE(r1.d("z") == 4.0_a);

    const auto r2 = s1 - 3.5;

    REQUIRE(r2.f() == -0.5_a);
    REQUIRE(r2.d("x") == 1.0_a);
    REQUIRE(r2.d("y") == 6.0_a);
    REQUIRE(r2.d("z") == 4.0_a);

    const auto r3 = 3.5 - s1;

    REQUIRE(r3.f() == 0.5_a);
    REQUIRE(r3.d("x") == -1.0_a);
    REQUIRE(r3.d("y") == -6.0_a);
    REQUIRE(r3.d("z") == -4.0_a);
}

TEST_CASE("IncSub", "[SScalar]")
{
    auto r1 = s1;
    r1 -= s2;

    REQUIRE(r1.f() == -1.0_a);
    REQUIRE(r1.d("x") == -6.0_a);
    REQUIRE(r1.d("y") == 5.0_a);
    REQUIRE(r1.d("z") == 4.0_a);

    auto r2 = s1;
    r2 -= 3.5;

    REQUIRE(r2.f() == -0.5_a);
    REQUIRE(r2.d("x") == 1.0_a);
    REQUIRE(r2.d("y") == 6.0_a);
    REQUIRE(r2.d("z") == 4.0_a);
}

TEST_CASE("Mul", "[SScalar]")
{
    const auto r1 = s1 * s2;

    REQUIRE(r1.f() == 12.0_a);
    REQUIRE(r1.d("x") == 25.0_a);
    REQUIRE(r1.d("y") == 27.0_a);
    REQUIRE(r1.d("z") == 16.0_a);
}

TEST_CASE("IncMul", "[SScalar]")
{
    auto r1 = s1;
    r1 *= s2;

    REQUIRE(r1.f() == 12.0_a);
    REQUIRE(r1.d("x") == 25.0_a);
    REQUIRE(r1.d("y") == 27.0_a);
    REQUIRE(r1.d("z") == 16.0_a);
}

TEST_CASE("Div", "[SScalar]")
{
    const auto r1 = s1 / s2;

    REQUIRE(r1.f() == 0.75000_a);
    REQUIRE(r1.d("x") == -1.06250_a);
    REQUIRE(r1.d("y") == 1.31250_a);
    REQUIRE(r1.d("z") == 1.00000_a);
}

TEST_CASE("IncDiv", "[SScalar]")
{
    auto r1 = s1;
    r1 /= s2;

    REQUIRE(r1.f() == 0.75000_a);
    REQUIRE(r1.d("x") == -1.06250_a);
    REQUIRE(r1.d("y") == 1.31250_a);
    REQUIRE(r1.d("z") == 1.00000_a);
}
