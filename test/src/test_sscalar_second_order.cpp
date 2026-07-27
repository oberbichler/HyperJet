#include <doctest/doctest.h>

#include <hyperjet/hyperjet.h>

#include <array>      // array
#include <functional> // function
#include <string>     // string
#include <vector>     // vector

// Second order for SScalar: a Hessian keyed by pairs of names.
//
// The inputs are built from variables rather than handed in as data, so the
// operands carry a non-zero Hessian of their own and the da * H term is
// exercised, not just the outer product of the gradient.
//
//   x = 0.5, y = 0.4, z = 0.3
//   u = x*y + z*z          the operand under test
//   t = x + y*z            the second operand
//   w = u + 1.5            for acosh, which needs a value above one
//
// The expectations were derived symbolically from those definitions, so they
// share no formulas with the header.

namespace {

using S = hyperjet::SScalar<2, double>;

S vx() { return S::variable("x", 0.5); }
S vy() { return S::variable("y", 0.4); }
S vz() { return S::variable("z", 0.3); }

S u() { return vx() * vy() + vz() * vz(); }
S t() { return vx() + vy() * vz(); }
S w() { return u() + 1.5; }

// f, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz
using Expected = std::array<double, 10>;

const std::vector<std::pair<std::string, Expected>> expected = {
    {"neg",
     {-0.28999999999999998, -0.40000000000000002, -0.5, -0.59999999999999998, 0,
      -1, 0, 0, 0, -2}},
    {"add",
     {0.91000000000000003, 1.3999999999999999, 0.80000000000000004, 1, 0, 1, 0,
      0, 1, 2}},
    {"sub",
     {-0.33000000000000002, -0.59999999999999998, 0.20000000000000001,
      0.20000000000000001, 0, 1, 0, 0, -1, 2}},
    {"mul",
     {0.17979999999999999, 0.53800000000000003, 0.39700000000000002,
      0.48799999999999999, 0.80000000000000004, 1.24, 0.76000000000000001,
      0.29999999999999999, 0.67000000000000004, 1.72}},
    {"div",
     {0.46774193548387094, -0.10926118626430802, 0.58012486992715917,
      0.66597294484911551, 0.35245543956228392, 0.73008626766473095,
      -1.0036588231345036, -0.56141116444563799, -1.4509415595314021,
      2.3664865227753347}},
    {"add_s",
     {2.04, 0.40000000000000002, 0.5, 0.59999999999999998, 0, 1, 0, 0, 0, 2}},
    {"sub_s",
     {-1.46, 0.40000000000000002, 0.5, 0.59999999999999998, 0, 1, 0, 0, 0, 2}},
    {"s_sub",
     {1.46, -0.40000000000000002, -0.5, -0.59999999999999998, 0, -1, 0, 0, 0,
      -2}},
    {"mul_s",
     {0.50749999999999995, 0.69999999999999996, 0.875, 1.05, 0, 1.75, 0, 0, 0,
      3.5}},
    {"div_s",
     {0.1657142857142857, 0.22857142857142856, 0.2857142857142857,
      0.34285714285714286, 0, 0.5714285714285714, 0, 0, 0, 1.1428571428571428}},
    {"s_div",
     {6.0344827586206895, -8.3234244946492275, -10.404280618311534,
      -12.485136741973841, 22.961171019722006, 7.8929025380294391,
      34.44175652958301, 35.876829718315633, 43.052195661978764,
      10.045512321128378}},
    {"pow2",
     {0.084099999999999994, 0.23200000000000001, 0.28999999999999998,
      0.34799999999999998, 0.32000000000000001, 0.97999999999999998,
      0.47999999999999998, 0.5, 0.59999999999999998, 1.8799999999999999}},
    {"sqrt",
     {0.53851648071345037, 0.37139067635410372, 0.46423834544262965,
      0.55708601453115558, -0.25613150093386466, 0.60831231471792857,
      -0.38419725140079697, -0.40020547020916347, -0.48024656425099621,
      1.2806575046693232}},
    {"cbrt",
     {0.66191059480262293, 0.30432671025407954, 0.38040838781759939,
      0.45649006538111925, -0.27984065310719958, 0.41101595925119933,
      -0.41976097966079934, -0.43725102047999931, -0.5247012245759991,
      0.89199208177919853}},
    {"reciprocal",
     {3.4482758620689653, -4.756242568370987, -5.9453032104637336,
      -7.1343638525564801, 13.12066915412686, 4.5102300217311084,
      19.681003731190291, 20.50104555332322, 24.601254663987863,
      5.7402927549305014}},
    {"cos",
     {0.95824387551269719, -0.11438089004193422, -0.14297611255241777,
      -0.17157133506290131, -0.15331902008203155, -0.47760100020737495,
      -0.22997853012304731, -0.2395609688781743, -0.28747316265380918,
      -0.91687224539424206}},
    {"sin",
     {0.28595222510483553, 0.38329755020507889, 0.47912193775634859,
      0.57494632530761836, -0.045752356016773688, 0.90105343049173003,
      -0.068628534025160531, -0.071488056276208883, -0.085785667531450657,
      1.8135449499876535}},
    {"tan",
     {0.29841278656943165, 0.43562007647525325, 0.54452509559406659,
      0.65343011471287993, 0.1039956807252554, 1.2190447920947025,
      0.15599352108788309, 0.16249325113321156, 0.19499190135985386,
      2.4120906640080908}},
    {"acos",
     {1.2765694890459141, -0.41796119462696196, -0.52245149328370244,
      -0.62694179194044286, -0.05293536256876033, -1.1110721897783553,
      -0.079403043853140495, -0.082711504013688014, -0.099253804816425623,
      -2.2089105389145205}},
    {"asin",
     {0.29422683774898251, 0.41796119462696196, 0.52245149328370244,
      0.62694179194044286, 0.05293536256876033, 1.1110721897783553,
      0.079403043853140495, 0.082711504013688014, 0.099253804816425623,
      2.2089105389145205}},
    {"atan",
     {0.28225742198149112, 0.36896965224610273, 0.46121206530762843,
      0.55345447836915418, -0.078960390481593798, 0.82372364251326469,
      -0.1184405857223907, -0.12337561012749032, -0.14805073215298839,
      1.6671873826469277}},
    {"atan2",
     {0.4375097721174529, -0.089647812166488788, 0.47598719316969051,
      0.54642475987193173, 0.28166825935767664, 0.63894738880302149,
      -0.77766754862643217, -0.67257955577019235, -1.4337954984743198,
      1.6623639331229234}},
    {"cosh",
     {1.0423455278020186, 0.11763278408938138, 0.14704098011172673,
      0.17644917613407207, 0.16677528444832299, 0.50255106578385722,
      0.25016292667248446, 0.26058638195050465, 0.31270365834060559,
      0.96340831045563358}},
    {"sinh",
     {0.29408196022345345, 0.41693821112080748, 0.52117276390100931,
      0.62540731668121119, 0.047053113635752548, 1.1011619198467093,
      0.070579670453628826, 0.073520490055863363, 0.088224588067036036,
      2.1905605612844807}},
    {"tanh",
     {0.28213481266963414, 0.36815997899194819, 0.46019997373993521,
      0.55223996848792223, -0.083096597364279795, 0.81652920077452074,
      -0.12464489604641968, -0.12983843338168716, -0.15580612005802461,
      1.6538325508901115}},
    {"acosh",
     {1.1862021872204589, 0.26942900300331829, 0.33678625375414789,
      0.40414350450497744, -0.087523781203382736, 0.56416778100406728,
      -0.13128567180507411, -0.13675590813028554, -0.16410708975634264,
      1.1502165073089803}},
    {"asinh",
     {0.28608171457448239, 0.38417165551149279, 0.48021456938936602,
      0.57625748326723925, -0.041106827819696677, 0.90904560400411116,
      -0.061660241729545012, -0.064229418468276053, -0.077075302161931272,
      1.8283679149631464}},
    {"atanh",
     {0.29856626366017835, 0.43672890053499291, 0.54591112566874112,
      0.65509335080248932, 0.11062463688625215, 1.2301030474452974,
      0.16593695532937824, 0.17285099513476901, 0.2074211941617228,
      2.4325499356690319}},
    {"exp",
     {1.3364274880254721, 0.5345709952101888, 0.66821374401273603,
      0.80185649281528326, 0.21382839808407553, 1.6037129856305665,
      0.3207425971261133, 0.33410687200636802, 0.40092824640764163,
      3.1539688717401142}},
    {"log",
     {-1.2378743560016174, 1.3793103448275863, 1.7241379310344827,
      2.0689655172413794, -1.9024970273483948, 1.070154577883472,
      -2.853745541022592, -2.9726516052318668, -3.56718192627824,
      2.615933412604043}},
    {"logb",
     {-1.7858751946471525, 1.9899241943296047, 2.4874052429120059,
      2.9848862914944072, -2.7447230266615237, 1.5439067024971072,
      -4.1170845399922857, -4.2886297291586306, -5.1463556749903567,
      3.7739941616595951}},
    {"log2",
     {-1.7858751946471525, 1.9899241943296047, 2.4874052429120059,
      2.9848862914944072, -2.7447230266615237, 1.5439067024971072,
      -4.1170845399922857, -4.2886297291586306, -5.1463556749903567,
      3.7739941616595951}},
    {"log10",
     {-0.53760200210104392, 0.59902687159069212, 0.74878358948836521,
      0.89854030738603829, -0.82624396081474782, 0.46476222795829564,
      -1.2393659412221218, -1.2910061887730435, -1.5492074265276521,
      1.1360854461202783}},
    {"hypot2",
     {0.68447059834590407, 1.0752835867291044, 0.48358541740126842,
      0.61653488260826372, 0.005500905546697776, 0.39447791992727999,
      -0.033529329046538825, 0.15507626526742271, 1.0838343140584952,
      1.0517394615150635}},
};

const Expected &look_up(const std::string &name) {
  for (const auto &[key, value] : expected) {
    if (key == name) {
      return value;
    }
  }

  FAIL("no expectation for ", name);

  return expected.front().second;
}

void check(const std::string &name, const S &r) {
  INFO("case: ", name);

  const Expected &e = look_up(name);

  CHECK(r.f() == doctest::Approx(e[0]));
  CHECK(r.d("x") == doctest::Approx(e[1]));
  CHECK(r.d("y") == doctest::Approx(e[2]));
  CHECK(r.d("z") == doctest::Approx(e[3]));
  CHECK(r.dd("x", "x") == doctest::Approx(e[4]));
  CHECK(r.dd("x", "y") == doctest::Approx(e[5]));
  CHECK(r.dd("x", "z") == doctest::Approx(e[6]));
  CHECK(r.dd("y", "y") == doctest::Approx(e[7]));
  CHECK(r.dd("y", "z") == doctest::Approx(e[8]));
  CHECK(r.dd("z", "z") == doctest::Approx(e[9]));

  // the Hessian is symmetric
  CHECK(r.dd("y", "x") == doctest::Approx(r.dd("x", "y")));
  CHECK(r.dd("z", "y") == doctest::Approx(r.dd("y", "z")));
}

TEST_CASE("SScalar second order") {
  const std::vector<std::pair<std::string, std::function<S()>>> cases = {
      {"neg", [] { return -u(); }},
      {"add", [] { return u() + t(); }},
      {"sub", [] { return u() - t(); }},
      {"mul", [] { return u() * t(); }},
      {"div", [] { return u() / t(); }},
      {"add_s", [] { return u() + 1.75; }},
      {"sub_s", [] { return u() - 1.75; }},
      {"s_sub", [] { return 1.75 - u(); }},
      {"mul_s", [] { return u() * 1.75; }},
      {"div_s", [] { return u() / 1.75; }},
      {"s_div", [] { return 1.75 / u(); }},
      {"pow2", [] { return u().pow(2.0); }},
      {"sqrt", [] { return u().sqrt(); }},
      {"cbrt", [] { return u().cbrt(); }},
      {"reciprocal", [] { return u().reciprocal(); }},
      {"cos", [] { return u().cos(); }},
      {"sin", [] { return u().sin(); }},
      {"tan", [] { return u().tan(); }},
      {"acos", [] { return u().acos(); }},
      {"asin", [] { return u().asin(); }},
      {"atan", [] { return u().atan(); }},
      {"atan2", [] { return u().atan2(t()); }},
      {"cosh", [] { return u().cosh(); }},
      {"sinh", [] { return u().sinh(); }},
      {"tanh", [] { return u().tanh(); }},
      {"acosh", [] { return w().acosh(); }},
      {"asinh", [] { return u().asinh(); }},
      {"atanh", [] { return u().atanh(); }},
      {"exp", [] { return u().exp(); }},
      {"log", [] { return u().log(); }},
      {"logb", [] { return u().log(2.0); }},
      {"log2", [] { return u().log2(); }},
      {"log10", [] { return u().log10(); }},
      {"hypot2", [] { return S::hypot(u(), t()); }},
  };

  for (const auto &[name, fn] : cases) {
    check(name, fn());
  }
}

TEST_CASE("SScalar second order: a variable has no curvature") {
  const auto v = vx();

  CHECK(v.dd("x", "x") == doctest::Approx(0.0));
  CHECK(v.dd("x", "y") == doctest::Approx(0.0));

  // and a value built from a gradient alone has none either
  const S g(3.0, {{"x", 1.0}, {"y", 2.0}});

  CHECK(g.dd("x", "y") == doctest::Approx(0.0));

  // an unknown name is zero rather than an error
  CHECK(u().dd("q", "q") == doctest::Approx(0.0));
  CHECK(u().dd("x", "q") == doctest::Approx(0.0));
}

// Projection onto an explicit list of names -- the bridge to code that wants
// arrays. The values come from d() and dd(), pinned symbolically above; what
// is new here is the layout and the zero-for-unknown rule.

TEST_CASE("SScalar g projects onto the given order") {
  const auto r = vx() * vy() + vz();

  // the caller's order, not the storage order
  const auto g = r.g({"z", "y", "x"});

  CHECK(hyperjet::length(g) == 3);
  CHECK(g[0] == doctest::Approx(r.d("z")));
  CHECK(g[1] == doctest::Approx(r.d("y")));
  CHECK(g[2] == doctest::Approx(r.d("x")));

  // a name the value does not carry reads as zero
  const auto padded = r.g({"w", "x"});

  CHECK(padded[0] == doctest::Approx(0.0));
  CHECK(padded[1] == doctest::Approx(r.d("x")));

  // an empty list is empty, not an error
  CHECK(hyperjet::length(r.g({})) == 0);
}

TEST_CASE("SScalar hm is row-major and symmetric") {
  using std::sqrt;

  const auto r = sqrt(vx() * vy() + vz() * vz());
  const std::vector<std::string> names{"x", "y", "z"};

  const auto h = r.hm(names);

  CHECK(hyperjet::length(h) == 9);

  for (hyperjet::index i = 0; i < 3; i++) {
    for (hyperjet::index j = 0; j < 3; j++) {
      CHECK(h[i * 3 + j] == doctest::Approx(r.dd(names[i], names[j])));
      CHECK(h[i * 3 + j] == doctest::Approx(h[j * 3 + i]));
    }
  }
}

TEST_CASE("SScalar hm zeroes rows and columns of unknown names") {
  const auto r = vx() * vy();
  const std::vector<std::string> names{"x", "w", "y"};

  const auto h = r.hm(names);

  const auto entry = [&](const hyperjet::index i, const hyperjet::index j) {
    return h[i * 3 + j];
  };

  // w is unknown, so its row and column are zero throughout
  for (hyperjet::index i = 0; i < 3; i++) {
    CHECK(entry(1, i) == doctest::Approx(0.0));
    CHECK(entry(i, 1) == doctest::Approx(0.0));
  }

  // the remaining entries keep the positions the caller chose
  CHECK(entry(0, 2) == doctest::Approx(r.dd("x", "y")));
  CHECK(entry(2, 0) == doctest::Approx(r.dd("x", "y")));
}

TEST_CASE("SScalar projects a local value onto a larger set") {
  // Two values carrying different names project onto one shared layout, which
  // is what an assembly step needs.
  const std::vector<std::string> global{"x", "y", "z"};

  const auto a = vx() * vy();
  const auto b = vy() * vz();

  CHECK(a.g(global)[2] == doctest::Approx(0.0)); // a does not know z
  CHECK(b.g(global)[0] == doctest::Approx(0.0)); // b does not know x

  CHECK(hyperjet::length(a.hm(global)) == 9);
  CHECK(hyperjet::length(b.hm(global)) == 9);
}

TEST_CASE("SScalar variables builds a set in the given order") {
  const auto v = S::variables({{"a", 1.0}, {"b", 2.0}, {"c", 3.0}});

  CHECK(hyperjet::length(v) == 3);

  CHECK(v[0].f() == doctest::Approx(1.0));
  CHECK(v[1].f() == doctest::Approx(2.0));
  CHECK(v[2].f() == doctest::Approx(3.0));

  CHECK(v[0].d("a") == doctest::Approx(1.0));
  CHECK(v[0].d("b") == doctest::Approx(0.0));

  // each carries only its own name
  CHECK(v[1].names() == std::vector<std::string>{"b"});
}

} // namespace
