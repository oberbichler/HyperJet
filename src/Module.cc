#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <hyperjet/hyperjet.h>

PYBIND11_MODULE(hyperjet, m) {
    m.doc() = "HyperJet by Thomas Oberbichler";
    m.attr("__author__") = "Thomas Oberbichler";
    m.attr("__copyright__") = "Copyright (c) 2019, Thomas Oberbichler";
    m.attr("__version__") = HYPERJET_VERSION;
    m.attr("__email__") = "thomas.oberbichler@gmail.com";
    m.attr("__status__") = "Development";

    namespace py = pybind11;
    using namespace pybind11::literals;

    #if defined(EIGEN_USE_BLAS)
    m.attr("USE_BLAS") = true;
    #else
    m.attr("USE_BLAS") = false;
    #endif // EIGEN_USE_BLAS

    m.def("explode", &hyperjet::explode<double>, "value"_a, "g"_a, "h"_a);
    m.def("explode", &hyperjet::explode<hyperjet::Jet<double>>, "value"_a,
        "g"_a, "h"_a);
    m.def("explode", &hyperjet::explode<hyperjet::HyperJet<double>>, "value"_a,
        "g"_a, "h"_a);

    m.def("f", [](const double value) { return value; }, "value"_a);
    m.def("f", [](const hyperjet::Jet<double>& value) {
        return value.f(); }, "value"_a);
    m.def("f", [](const hyperjet::HyperJet<double>& value) {
        return value.f(); }, "value"_a);

    hyperjet::Jet<double>::register_python(m);

    hyperjet::HyperJet<double>::register_python(m);
}
