#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>

#include <Eigen/Core>
#include <Eigen/Geometry> // cross

#include <hyperjet/hyperjet.h>

#include <sstream> // stringstream

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

    REQUIRE(r1.f() == doctest::Approx(-3));
    REQUIRE(r1.g(0) == doctest::Approx(-1));
    REQUIRE(r1.g(1) == doctest::Approx(-6));
    REQUIRE(r1.g(2) == doctest::Approx(-4));
    REQUIRE(r1.h(0, 0) == doctest::Approx(-0));
    REQUIRE(r1.h(0, 1) == doctest::Approx(-5));
    REQUIRE(r1.h(0, 2) == doctest::Approx(-9));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-2));
    REQUIRE(r1.h(1, 2) == doctest::Approx(-7));
    REQUIRE(r1.h(2, 2) == doctest::Approx(-8));
}

TEST_CASE("Add")
{
    const auto r1 = dd1 + dd2;

    REQUIRE(r1.f() == doctest::Approx(7));
    REQUIRE(r1.g(0) == doctest::Approx(8));
    REQUIRE(r1.g(1) == doctest::Approx(7));
    REQUIRE(r1.g(2) == doctest::Approx(4));
    REQUIRE(r1.h(0, 0) == doctest::Approx(6));
    REQUIRE(r1.h(0, 1) == doctest::Approx(13));
    REQUIRE(r1.h(0, 2) == doctest::Approx(11));
    REQUIRE(r1.h(1, 1) == doctest::Approx(11));
    REQUIRE(r1.h(1, 2) == doctest::Approx(12));
    REQUIRE(r1.h(2, 2) == doctest::Approx(11));

    const auto r2 = dd1 + 3.5;

    REQUIRE(r2.f() == doctest::Approx(6.5));
    REQUIRE(r2.g(0) == doctest::Approx(1.0));
    REQUIRE(r2.g(1) == doctest::Approx(6.0));
    REQUIRE(r2.g(2) == doctest::Approx(4.0));
    REQUIRE(r2.h(0, 0) == doctest::Approx(0.0));
    REQUIRE(r2.h(0, 1) == doctest::Approx(5.0));
    REQUIRE(r2.h(0, 2) == doctest::Approx(9.0));
    REQUIRE(r2.h(1, 1) == doctest::Approx(2.0));
    REQUIRE(r2.h(1, 2) == doctest::Approx(7.0));
    REQUIRE(r2.h(2, 2) == doctest::Approx(8.0));

    const auto r3 = 3.5 + dd1;

    REQUIRE(r3.f() == doctest::Approx(6.5));
    REQUIRE(r3.g(0) == doctest::Approx(1.0));
    REQUIRE(r3.g(1) == doctest::Approx(6.0));
    REQUIRE(r3.g(2) == doctest::Approx(4.0));
    REQUIRE(r3.h(0, 0) == doctest::Approx(0.0));
    REQUIRE(r3.h(0, 1) == doctest::Approx(5.0));
    REQUIRE(r3.h(0, 2) == doctest::Approx(9.0));
    REQUIRE(r3.h(1, 1) == doctest::Approx(2.0));
    REQUIRE(r3.h(1, 2) == doctest::Approx(7.0));
    REQUIRE(r3.h(2, 2) == doctest::Approx(8.0));
}

TEST_CASE("IncAdd")
{
    auto r1 = dd1;
    r1 += dd2;

    REQUIRE(r1.f() == doctest::Approx(7));
    REQUIRE(r1.g(0) == doctest::Approx(8));
    REQUIRE(r1.g(1) == doctest::Approx(7));
    REQUIRE(r1.g(2) == doctest::Approx(4));
    REQUIRE(r1.h(0, 0) == doctest::Approx(6));
    REQUIRE(r1.h(0, 1) == doctest::Approx(13));
    REQUIRE(r1.h(0, 2) == doctest::Approx(11));
    REQUIRE(r1.h(1, 1) == doctest::Approx(11));
    REQUIRE(r1.h(1, 2) == doctest::Approx(12));
    REQUIRE(r1.h(2, 2) == doctest::Approx(11));

    auto r2 = dd1;
    r2 += 3.5;

    REQUIRE(r2.f() == doctest::Approx(6.5));
    REQUIRE(r2.g(0) == doctest::Approx(1.0));
    REQUIRE(r2.g(1) == doctest::Approx(6.0));
    REQUIRE(r2.g(2) == doctest::Approx(4.0));
    REQUIRE(r2.h(0, 0) == doctest::Approx(0.0));
    REQUIRE(r2.h(0, 1) == doctest::Approx(5.0));
    REQUIRE(r2.h(0, 2) == doctest::Approx(9.0));
    REQUIRE(r2.h(1, 1) == doctest::Approx(2.0));
    REQUIRE(r2.h(1, 2) == doctest::Approx(7.0));
    REQUIRE(r2.h(2, 2) == doctest::Approx(8.0));
}

TEST_CASE("Sub")
{
    const auto r1 = dd1 - dd2;

    REQUIRE(r1.f() == doctest::Approx(-1.0));
    REQUIRE(r1.g(0) == doctest::Approx(-6.0));
    REQUIRE(r1.g(1) == doctest::Approx(5.0));
    REQUIRE(r1.g(2) == doctest::Approx(4.0));
    REQUIRE(r1.h(0, 0) == doctest::Approx(-6.0));
    REQUIRE(r1.h(0, 1) == doctest::Approx(-3.0));
    REQUIRE(r1.h(0, 2) == doctest::Approx(7.0));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-7.0));
    REQUIRE(r1.h(1, 2) == doctest::Approx(2.0));
    REQUIRE(r1.h(2, 2) == doctest::Approx(5.0));

    const auto r2 = dd1 - 3.5;

    REQUIRE(r2.f() == doctest::Approx(-0.5));
    REQUIRE(r2.g(0) == doctest::Approx(1.0));
    REQUIRE(r2.g(1) == doctest::Approx(6.0));
    REQUIRE(r2.g(2) == doctest::Approx(4.0));
    REQUIRE(r2.h(0, 0) == doctest::Approx(0.0));
    REQUIRE(r2.h(0, 1) == doctest::Approx(5.0));
    REQUIRE(r2.h(0, 2) == doctest::Approx(9.0));
    REQUIRE(r2.h(1, 1) == doctest::Approx(2.0));
    REQUIRE(r2.h(1, 2) == doctest::Approx(7.0));
    REQUIRE(r2.h(2, 2) == doctest::Approx(8.0));

    const auto r3 = 3.5 - dd1;

    REQUIRE(r3.f() == doctest::Approx(0.5));
    REQUIRE(r3.g(0) == doctest::Approx(-1.0));
    REQUIRE(r3.g(1) == doctest::Approx(-6.0));
    REQUIRE(r3.g(2) == doctest::Approx(-4.0));
    REQUIRE(r3.h(0, 0) == doctest::Approx(-0.0));
    REQUIRE(r3.h(0, 1) == doctest::Approx(-5.0));
    REQUIRE(r3.h(0, 2) == doctest::Approx(-9.0));
    REQUIRE(r3.h(1, 1) == doctest::Approx(-2.0));
    REQUIRE(r3.h(1, 2) == doctest::Approx(-7.0));
    REQUIRE(r3.h(2, 2) == doctest::Approx(-8.0));
}

TEST_CASE("IncSub")
{
    auto r1 = dd1;
    r1 -= dd2;

    REQUIRE(r1.f() == doctest::Approx(-1.0));
    REQUIRE(r1.g(0) == doctest::Approx(-6.0));
    REQUIRE(r1.g(1) == doctest::Approx(5.0));
    REQUIRE(r1.g(2) == doctest::Approx(4.0));
    REQUIRE(r1.h(0, 0) == doctest::Approx(-6.0));
    REQUIRE(r1.h(0, 1) == doctest::Approx(-3.0));
    REQUIRE(r1.h(0, 2) == doctest::Approx(7.0));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-7.0));
    REQUIRE(r1.h(1, 2) == doctest::Approx(2.0));
    REQUIRE(r1.h(2, 2) == doctest::Approx(5.0));

    auto r2 = dd1;
    r2 -= 3.5;

    REQUIRE(r2.f() == doctest::Approx(-0.5));
    REQUIRE(r2.g(0) == doctest::Approx(1.0));
    REQUIRE(r2.g(1) == doctest::Approx(6.0));
    REQUIRE(r2.g(2) == doctest::Approx(4.0));
    REQUIRE(r2.h(0, 0) == doctest::Approx(0.0));
    REQUIRE(r2.h(0, 1) == doctest::Approx(5.0));
    REQUIRE(r2.h(0, 2) == doctest::Approx(9.0));
    REQUIRE(r2.h(1, 1) == doctest::Approx(2.0));
    REQUIRE(r2.h(1, 2) == doctest::Approx(7.0));
    REQUIRE(r2.h(2, 2) == doctest::Approx(8.0));
}

TEST_CASE("Mul")
{
    const auto r1 = dd1 * dd2;

    REQUIRE(r1.f() == doctest::Approx(12.0));
    REQUIRE(r1.g(0) == doctest::Approx(25.0));
    REQUIRE(r1.g(1) == doctest::Approx(27.0));
    REQUIRE(r1.g(2) == doctest::Approx(16.0));
    REQUIRE(r1.h(0, 0) == doctest::Approx(32.0));
    REQUIRE(r1.h(0, 1) == doctest::Approx(87.0));
    REQUIRE(r1.h(0, 2) == doctest::Approx(70.0));
    REQUIRE(r1.h(1, 1) == doctest::Approx(47.0));
    REQUIRE(r1.h(1, 2) == doctest::Approx(47.0));
    REQUIRE(r1.h(2, 2) == doctest::Approx(41.0));
}

TEST_CASE("IncMul")
{
    auto r1 = dd1;
    r1 *= dd2;

    REQUIRE(r1.f() == doctest::Approx(12.0));
    REQUIRE(r1.g(0) == doctest::Approx(25.0));
    REQUIRE(r1.g(1) == doctest::Approx(27.0));
    REQUIRE(r1.g(2) == doctest::Approx(16.0));
    REQUIRE(r1.h(0, 0) == doctest::Approx(32.0));
    REQUIRE(r1.h(0, 1) == doctest::Approx(87.0));
    REQUIRE(r1.h(0, 2) == doctest::Approx(70.0));
    REQUIRE(r1.h(1, 1) == doctest::Approx(47.0));
    REQUIRE(r1.h(1, 2) == doctest::Approx(47.0));
    REQUIRE(r1.h(2, 2) == doctest::Approx(41.0));
}

TEST_CASE("Div")
{
    const auto r1 = dd1 / dd2;

    REQUIRE(r1.f() == doctest::Approx(0.75000));
    REQUIRE(r1.g(0) == doctest::Approx(-1.06250));
    REQUIRE(r1.g(1) == doctest::Approx(1.31250));
    REQUIRE(r1.g(2) == doctest::Approx(1.00000));
    REQUIRE(r1.h(0, 0) == doctest::Approx(2.59375));
    REQUIRE(r1.h(0, 1) == doctest::Approx(-2.28125));
    REQUIRE(r1.h(0, 2) == doctest::Approx(0.12500));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-1.84375));
    REQUIRE(r1.h(1, 2) == doctest::Approx(0.56250));
    REQUIRE(r1.h(2, 2) == doctest::Approx(1.43750));
}

TEST_CASE("IncDiv")
{
    auto r1 = dd1;
    r1 /= dd2;

    REQUIRE(r1.f() == doctest::Approx(0.75000));
    REQUIRE(r1.g(0) == doctest::Approx(-1.06250));
    REQUIRE(r1.g(1) == doctest::Approx(1.31250));
    REQUIRE(r1.g(2) == doctest::Approx(1.00000));
    REQUIRE(r1.h(0, 0) == doctest::Approx(2.59375));
    REQUIRE(r1.h(0, 1) == doctest::Approx(-2.28125));
    REQUIRE(r1.h(0, 2) == doctest::Approx(0.12500));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-1.84375));
    REQUIRE(r1.h(1, 2) == doctest::Approx(0.56250));
    REQUIRE(r1.h(2, 2) == doctest::Approx(1.43750));
}

TEST_CASE("Pow")
{
    using std::pow;

    const auto r1 = pow(dd1, 3.5);

    REQUIRE(r1.f() == doctest::Approx(46.765371804359690));
    REQUIRE(r1.g(0) == doctest::Approx(54.559600438419636));
    REQUIRE(r1.g(1) == doctest::Approx(327.357602630517800));
    REQUIRE(r1.g(2) == doctest::Approx(218.238401753678540));
    REQUIRE(r1.h(0, 0) == doctest::Approx(45.466333698683030));
    REQUIRE(r1.h(0, 1) == doctest::Approx(545.596004384196400));
    REQUIRE(r1.h(0, 2) == doctest::Approx(672.901738740508800));
    REQUIRE(r1.h(1, 1) == doctest::Approx(1745.907214029428400));
    REQUIRE(r1.h(1, 2) == doctest::Approx(1473.109211837330100));
    REQUIRE(r1.h(2, 2) == doctest::Approx(1163.938142686285600));
}

TEST_CASE("Sqrt")
{
    using std::sqrt;

    const auto r1 = sqrt(dd1);

    REQUIRE(r1.f() == doctest::Approx(1.732050807568877200));
    REQUIRE(r1.g(0) == doctest::Approx(0.288675134594812900));
    REQUIRE(r1.g(1) == doctest::Approx(1.732050807568877400));
    REQUIRE(r1.g(2) == doctest::Approx(1.154700538379251700));
    REQUIRE(r1.h(0, 0) == doctest::Approx(-0.048112522432468816));
    REQUIRE(r1.h(0, 1) == doctest::Approx(1.154700538379251700));
    REQUIRE(r1.h(0, 2) == doctest::Approx(2.405626121623441000));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-1.154700538379251700));
    REQUIRE(r1.h(1, 2) == doctest::Approx(0.866025403784438700));
    REQUIRE(r1.h(2, 2) == doctest::Approx(1.539600717839002300));
}

TEST_CASE("Cos")
{
    using std::cos;

    const auto r1 = cos(dd1);

    REQUIRE(r1.f() == doctest::Approx(-0.9899924966004454));
    REQUIRE(r1.g(0) == doctest::Approx(-0.1411200080598672));
    REQUIRE(r1.g(1) == doctest::Approx(-0.8467200483592032));
    REQUIRE(r1.g(2) == doctest::Approx(-0.5644800322394689));
    REQUIRE(r1.h(0, 0) == doctest::Approx(0.9899924966004454));
    REQUIRE(r1.h(0, 1) == doctest::Approx(5.2343549393033370));
    REQUIRE(r1.h(0, 2) == doctest::Approx(2.6898899138629770));
    REQUIRE(r1.h(1, 1) == doctest::Approx(35.3574898614963000));
    REQUIRE(r1.h(1, 2) == doctest::Approx(22.7719798619916200));
    REQUIRE(r1.h(2, 2) == doctest::Approx(14.7109198811281900));
}

TEST_CASE("Sin")
{
    using std::sin;

    const auto r1 = dd1.sin();

    REQUIRE(r1.f() == doctest::Approx(0.1411200080598672));
    REQUIRE(r1.g(0) == doctest::Approx(-0.9899924966004454));
    REQUIRE(r1.g(1) == doctest::Approx(-5.9399549796026730));
    REQUIRE(r1.g(2) == doctest::Approx(-3.9599699864017817));
    REQUIRE(r1.h(0, 0) == doctest::Approx(-0.1411200080598672));
    REQUIRE(r1.h(0, 1) == doctest::Approx(-5.7966825313614300));
    REQUIRE(r1.h(0, 2) == doctest::Approx(-9.4744125016434780));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-7.0603052833561100));
    REQUIRE(r1.h(1, 2) == doctest::Approx(-10.3168276696399310));
    REQUIRE(r1.h(2, 2) == doctest::Approx(-10.1778601017614400));
}

TEST_CASE("Tan")
{
    using std::tan;

    const auto r1 = tan(dd1);

    REQUIRE(r1.f() == doctest::Approx(-0.14254654307427780));
    REQUIRE(r1.g(0) == doctest::Approx(1.02031951694242680));
    REQUIRE(r1.g(1) == doctest::Approx(6.12191710165456100));
    REQUIRE(r1.g(2) == doctest::Approx(4.08127806776970700));
    REQUIRE(r1.h(0, 0) == doctest::Approx(-0.29088603994271994));
    REQUIRE(r1.h(0, 1) == doctest::Approx(3.35628134505581470));
    REQUIRE(r1.h(0, 2) == doctest::Approx(8.01933149271096100));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-8.43125840405306400));
    REQUIRE(r1.h(1, 2) == doctest::Approx(0.16097165997170980));
    REQUIRE(r1.h(2, 2) == doctest::Approx(3.50837949645589560));
}

TEST_CASE("Acos")
{
    using std::acos;

    const auto r1 = acos(dd3);

    REQUIRE(r1.f() == doctest::Approx(1.266103672779499200));
    REQUIRE(r1.g(0) == doctest::Approx(-0.104828483672191830));
    REQUIRE(r1.g(1) == doctest::Approx(-0.838627869377534600));
    REQUIRE(r1.g(2) == doctest::Approx(-0.209656967344383660));
    REQUIRE(r1.h(0, 0) == doctest::Approx(-0.527598302438064400));
    REQUIRE(r1.h(0, 1) == doctest::Approx(-0.761446458322184600));
    REQUIRE(r1.h(0, 2) == doctest::Approx(-0.950368121203936900));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-0.640490515623501800));
    REQUIRE(r1.h(1, 2) == doctest::Approx(-0.684265047266834600));
    REQUIRE(r1.h(2, 2) == doctest::Approx(-0.013823536308420903));
}

TEST_CASE("Asin")
{
    using std::asin;

    const auto r1 = asin(dd3);

    REQUIRE(r1.f() == doctest::Approx(0.304692654015397500));
    REQUIRE(r1.g(0) == doctest::Approx(0.104828483672191830));
    REQUIRE(r1.g(1) == doctest::Approx(0.838627869377534600));
    REQUIRE(r1.g(2) == doctest::Approx(0.209656967344383660));
    REQUIRE(r1.h(0, 0) == doctest::Approx(0.527598302438064400));
    REQUIRE(r1.h(0, 1) == doctest::Approx(0.761446458322184600));
    REQUIRE(r1.h(0, 2) == doctest::Approx(0.950368121203936900));
    REQUIRE(r1.h(1, 1) == doctest::Approx(0.640490515623501700));
    REQUIRE(r1.h(1, 2) == doctest::Approx(0.684265047266834600));
    REQUIRE(r1.h(2, 2) == doctest::Approx(0.013823536308420899));
}

TEST_CASE("Atan")
{
    using std::atan;

    const auto r1 = atan(dd3);

    REQUIRE(r1.f() == doctest::Approx(0.291456794477867100));
    REQUIRE(r1.g(0) == doctest::Approx(0.091743119266055050));
    REQUIRE(r1.g(1) == doctest::Approx(0.733944954128440400));
    REQUIRE(r1.g(2) == doctest::Approx(0.183486238532110100));
    REQUIRE(r1.h(0, 0) == doctest::Approx(0.453665516370675800));
    REQUIRE(r1.h(0, 1) == doctest::Approx(0.601801195185590400));
    REQUIRE(r1.h(0, 2) == doctest::Approx(0.815587913475296700));
    REQUIRE(r1.h(1, 1) == doctest::Approx(0.043767359649861170));
    REQUIRE(r1.h(1, 2) == doctest::Approx(0.469657436242740500));
    REQUIRE(r1.h(2, 2) == doctest::Approx(-0.020200319838397436));
}

TEST_CASE("Atan2")
{
    using std::atan2;

    const auto r1 = atan2(dd1, dd2);

    REQUIRE(r1.f() == doctest::Approx(0.6435011087932844));
    REQUIRE(r1.g(0) == doctest::Approx(-0.6800000000000000));
    REQUIRE(r1.g(1) == doctest::Approx(0.8400000000000000));
    REQUIRE(r1.g(2) == doctest::Approx(0.6400000000000000));
    REQUIRE(r1.h(0, 0) == doctest::Approx(0.9664000000000000));
    REQUIRE(r1.h(0, 1) == doctest::Approx(-0.6032000000000000));
    REQUIRE(r1.h(0, 2) == doctest::Approx(0.7328000000000000));
    REQUIRE(r1.h(1, 1) == doctest::Approx(-2.2384000000000000));
    REQUIRE(r1.h(1, 2) == doctest::Approx(-0.4464000000000000));
    REQUIRE(r1.h(2, 2) == doctest::Approx(0.3056000000000000));
}

TEST_CASE("Dot")
{
    const auto r = ddv1.dot(ddv2);

    REQUIRE(r.f() == doctest::Approx(14.1));
    REQUIRE(r.g(0) == doctest::Approx(28.1));
    REQUIRE(r.g(1) == doctest::Approx(34.7));
    REQUIRE(r.g(2) == doctest::Approx(18.6));
    REQUIRE(r.h(0, 0) == doctest::Approx(38.9));
    REQUIRE(r.h(0, 1) == doctest::Approx(102.9));
    REQUIRE(r.h(0, 2) == doctest::Approx(81.6));
    REQUIRE(r.h(1, 1) == doctest::Approx(64.3));
    REQUIRE(r.h(1, 2) == doctest::Approx(59.4));
    REQUIRE(r.h(2, 2) == doctest::Approx(45.9));
}

TEST_CASE("Cross")
{
    const auto r = ddv1.cross(ddv2);

    const auto rx = r.x();
    const auto ry = r.y();
    const auto rz = r.z();

    REQUIRE(rx.f() == doctest::Approx(15.1));
    REQUIRE(rx.g(0) == doctest::Approx(55.4));
    REQUIRE(rx.g(1) == doctest::Approx(3.8));
    REQUIRE(rx.g(2) == doctest::Approx(-1.8));
    REQUIRE(rx.h(0, 0) == doctest::Approx(144.3));
    REQUIRE(rx.h(0, 1) == doctest::Approx(73.0));
    REQUIRE(rx.h(0, 2) == doctest::Approx(10.0));
    REQUIRE(rx.h(1, 1) == doctest::Approx(62.6));
    REQUIRE(rx.h(1, 2) == doctest::Approx(31.7));
    REQUIRE(rx.h(2, 2) == doctest::Approx(20.0));

    REQUIRE(ry.f() == doctest::Approx(-11.91));
    REQUIRE(ry.g(0) == doctest::Approx(-24.94));
    REQUIRE(ry.g(1) == doctest::Approx(-26.52));
    REQUIRE(ry.g(2) == doctest::Approx(-15.88));
    REQUIRE(ry.h(0, 0) == doctest::Approx(-31.68));
    REQUIRE(ry.h(0, 1) == doctest::Approx(-86.42));
    REQUIRE(ry.h(0, 2) == doctest::Approx(-69.42));
    REQUIRE(ry.h(1, 1) == doctest::Approx(-45.48));
    REQUIRE(ry.h(1, 2) == doctest::Approx(-46.32));
    REQUIRE(ry.h(2, 2) == doctest::Approx(-40.92));

    REQUIRE(rz.f() == doctest::Approx(7.8));
    REQUIRE(rz.g(0) == doctest::Approx(3.5));
    REQUIRE(rz.g(1) == doctest::Approx(32.5));
    REQUIRE(rz.g(2) == doctest::Approx(23.2));
    REQUIRE(rz.h(0, 0) == doctest::Approx(-3.2));
    REQUIRE(rz.h(0, 1) == doctest::Approx(31.1));
    REQUIRE(rz.h(0, 2) == doctest::Approx(56.4));
    REQUIRE(rz.h(1, 1) == doctest::Approx(78.1));
    REQUIRE(rz.h(1, 2) == doctest::Approx(85.9));
    REQUIRE(rz.h(2, 2) == doctest::Approx(79.1));
}

TEST_CASE("Norm")
{
    const auto r = ddv1.norm();

    REQUIRE(r.f() == doctest::Approx(5.0089919145472770));
    REQUIRE(r.g(0) == doctest::Approx(6.1948592709606230));
    REQUIRE(r.g(1) == doctest::Approx(4.4400151526317835));
    REQUIRE(r.g(2) == doctest::Approx(2.4076700872634587));
    REQUIRE(r.h(0, 0) == doctest::Approx(7.1438962616547620));
    REQUIRE(r.h(0, 1) == doctest::Approx(6.5451754620124020));
    REQUIRE(r.h(0, 2) == doctest::Approx(4.8662132130242100));
    REQUIRE(r.h(1, 1) == doctest::Approx(11.9876946237449200));
    REQUIRE(r.h(1, 2) == doctest::Approx(10.9103606598557140));
    REQUIRE(r.h(2, 2) == doctest::Approx(9.2320222391647280));
}

const SScalar<double> s1(3.0, {{"x", 1.0}, {"y", 6.0}, {"z", 4.0}});
const SScalar<double> s2(4.0, {{"x", 7.0}, {"y", 1.0}});
const SScalar<double> s3(0.3, {{"x", 0.1}, {"y", 0.8}, {"z", 0.2}});

TEST_CASE("SScalar init")
{
    using Dual = SScalar<double>;

    const auto x = Dual(1.5, {{"x", 2.0}, {"y", 1.0}});

    REQUIRE(x.size() == 2);

    REQUIRE(x.f() == doctest::Approx(1.5));

    REQUIRE(x.d("x") == doctest::Approx(2.0));
    REQUIRE(x.d("y") == doctest::Approx(1.0));
    REQUIRE(x.d("z") == doctest::Approx(0.0));
}

TEST_CASE("SScalar constant")
{
    using Dual = SScalar<double>;

    const auto x = Dual::constant(1.5);

    REQUIRE(x.size() == 0);

    REQUIRE(x.f() == doctest::Approx(1.5));

    REQUIRE(x.d("x") == doctest::Approx(0.0));
    REQUIRE(x.d("y") == doctest::Approx(0.0));
}

TEST_CASE("SScalar variable")
{
    using Dual = SScalar<double>;

    const auto x = Dual::variable("x", 1.5);

    REQUIRE(x.size() == 1);

    REQUIRE(x.f() == doctest::Approx(1.5));

    REQUIRE(x.d("x") == doctest::Approx(1.0));
    REQUIRE(x.d("y") == doctest::Approx(0.0));
}

TEST_CASE("SScalar Neg")
{
    const auto r1 = -s1;

    REQUIRE(r1.f() == doctest::Approx(-3));
    REQUIRE(r1.d("x") == doctest::Approx(-1));
    REQUIRE(r1.d("y") == doctest::Approx(-6));
    REQUIRE(r1.d("z") == doctest::Approx(-4));
}

TEST_CASE("SScalar Add")
{
    const auto r1 = s1 + s2;

    REQUIRE(r1.f() == doctest::Approx(7));
    REQUIRE(r1.d("x") == doctest::Approx(8));
    REQUIRE(r1.d("y") == doctest::Approx(7));
    REQUIRE(r1.d("z") == doctest::Approx(4));

    const auto r2 = s1 + 3.5;

    REQUIRE(r2.f() == doctest::Approx(6.5));
    REQUIRE(r2.d("x") == doctest::Approx(1.0));
    REQUIRE(r2.d("y") == doctest::Approx(6.0));
    REQUIRE(r2.d("z") == doctest::Approx(4.0));

    const auto r3 = 3.5 + s1;

    REQUIRE(r3.f() == doctest::Approx(6.5));
    REQUIRE(r3.d("x") == doctest::Approx(1.0));
    REQUIRE(r3.d("y") == doctest::Approx(6.0));
    REQUIRE(r3.d("z") == doctest::Approx(4.0));
}

TEST_CASE("SScalar IncAdd")
{
    auto r1 = s1;
    r1 += s2;

    REQUIRE(r1.f() == doctest::Approx(7));
    REQUIRE(r1.d("x") == doctest::Approx(8));
    REQUIRE(r1.d("y") == doctest::Approx(7));
    REQUIRE(r1.d("z") == doctest::Approx(4));

    auto r2 = s1;
    r2 += 3.5;

    REQUIRE(r2.f() == doctest::Approx(6.5));
    REQUIRE(r2.d("x") == doctest::Approx(1.0));
    REQUIRE(r2.d("y") == doctest::Approx(6.0));
    REQUIRE(r2.d("z") == doctest::Approx(4.0));
}

TEST_CASE("SScalar Sub")
{
    const auto r1 = s1 - s2;

    REQUIRE(r1.f() == doctest::Approx(-1.0));
    REQUIRE(r1.d("x") == doctest::Approx(-6.0));
    REQUIRE(r1.d("y") == doctest::Approx(5.0));
    REQUIRE(r1.d("z") == doctest::Approx(4.0));

    const auto r2 = s1 - 3.5;

    REQUIRE(r2.f() == doctest::Approx(-0.5));
    REQUIRE(r2.d("x") == doctest::Approx(1.0));
    REQUIRE(r2.d("y") == doctest::Approx(6.0));
    REQUIRE(r2.d("z") == doctest::Approx(4.0));

    const auto r3 = 3.5 - s1;

    REQUIRE(r3.f() == doctest::Approx(0.5));
    REQUIRE(r3.d("x") == doctest::Approx(-1.0));
    REQUIRE(r3.d("y") == doctest::Approx(-6.0));
    REQUIRE(r3.d("z") == doctest::Approx(-4.0));
}

TEST_CASE("SScalar IncSub")
{
    auto r1 = s1;
    r1 -= s2;

    REQUIRE(r1.f() == doctest::Approx(-1.0));
    REQUIRE(r1.d("x") == doctest::Approx(-6.0));
    REQUIRE(r1.d("y") == doctest::Approx(5.0));
    REQUIRE(r1.d("z") == doctest::Approx(4.0));

    auto r2 = s1;
    r2 -= 3.5;

    REQUIRE(r2.f() == doctest::Approx(-0.5));
    REQUIRE(r2.d("x") == doctest::Approx(1.0));
    REQUIRE(r2.d("y") == doctest::Approx(6.0));
    REQUIRE(r2.d("z") == doctest::Approx(4.0));
}

TEST_CASE("SScalar Mul")
{
    const auto r1 = s1 * s2;

    REQUIRE(r1.f() == doctest::Approx(12.0));
    REQUIRE(r1.d("x") == doctest::Approx(25.0));
    REQUIRE(r1.d("y") == doctest::Approx(27.0));
    REQUIRE(r1.d("z") == doctest::Approx(16.0));
}

TEST_CASE("SScalar IncMul")
{
    auto r1 = s1;
    r1 *= s2;

    REQUIRE(r1.f() == doctest::Approx(12.0));
    REQUIRE(r1.d("x") == doctest::Approx(25.0));
    REQUIRE(r1.d("y") == doctest::Approx(27.0));
    REQUIRE(r1.d("z") == doctest::Approx(16.0));
}

TEST_CASE("SScalar Div")
{
    const auto r1 = s1 / s2;

    REQUIRE(r1.f() == doctest::Approx(0.75000));
    REQUIRE(r1.d("x") == doctest::Approx(-1.06250));
    REQUIRE(r1.d("y") == doctest::Approx(1.31250));
    REQUIRE(r1.d("z") == doctest::Approx(1.00000));
}

TEST_CASE("SScalar IncDiv")
{
    auto r1 = s1;
    r1 /= s2;

    REQUIRE(r1.f() == doctest::Approx(0.75000));
    REQUIRE(r1.d("x") == doctest::Approx(-1.06250));
    REQUIRE(r1.d("y") == doctest::Approx(1.31250));
    REQUIRE(r1.d("z") == doctest::Approx(1.00000));
}

TEST_CASE("SScalar Pow")
{
    using std::pow;

    const auto r1 = pow(s1, 3.5);

    REQUIRE(r1.f() == doctest::Approx(46.765371804359690));
    REQUIRE(r1.d("x") == doctest::Approx(54.559600438419636));
    REQUIRE(r1.d("y") == doctest::Approx(327.357602630517800));
    REQUIRE(r1.d("z") == doctest::Approx(218.238401753678540));
}

TEST_CASE("SScalar Sqrt")
{
    using std::sqrt;

    const auto r1 = sqrt(s1);

    REQUIRE(r1.f() == doctest::Approx(1.732050807568877200));
    REQUIRE(r1.d("x") == doctest::Approx(0.288675134594812900));
    REQUIRE(r1.d("y") == doctest::Approx(1.732050807568877400));
    REQUIRE(r1.d("z") == doctest::Approx(1.154700538379251700));
}

TEST_CASE("SScalar Cos")
{
    using std::cos;

    const auto r1 = cos(s1);

    REQUIRE(r1.f() == doctest::Approx(-0.9899924966004454));
    REQUIRE(r1.d("x") == doctest::Approx(-0.1411200080598672));
    REQUIRE(r1.d("y") == doctest::Approx(-0.8467200483592032));
    REQUIRE(r1.d("z") == doctest::Approx(-0.5644800322394689));
}

TEST_CASE("SScalar Sin")
{
    using std::sin;

    const auto r1 = s1.sin();

    REQUIRE(r1.f() == doctest::Approx(0.1411200080598672));
    REQUIRE(r1.d("x") == doctest::Approx(-0.9899924966004454));
    REQUIRE(r1.d("y") == doctest::Approx(-5.9399549796026730));
    REQUIRE(r1.d("z") == doctest::Approx(-3.9599699864017817));
}

TEST_CASE("SScalar Tan")
{
    using std::tan;

    const auto r1 = tan(s1);

    REQUIRE(r1.f() == doctest::Approx(-0.14254654307427780));
    REQUIRE(r1.d("x") == doctest::Approx(1.02031951694242680));
    REQUIRE(r1.d("y") == doctest::Approx(6.12191710165456100));
    REQUIRE(r1.d("z") == doctest::Approx(4.08127806776970700));
}