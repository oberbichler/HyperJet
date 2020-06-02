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
    m.attr("__copyright__") = "Copyright (c) 2019-2020, Thomas Oberbichler";
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

    namespace hj = hyperjet;

    hj::Jet<double, hj::Dynamic>::register_python(m, "Jet");
    hj::HyperJet<double, hj::Dynamic>::register_python(m, "HyperJet");

    m.def("explode", py::overload_cast<const double, Eigen::Ref<Eigen::Matrix<double, 1, hj::Dynamic>>, Eigen::Ref<Eigen::Matrix<double, hj::Dynamic, hj::Dynamic>>>(&hj::explode<double, hj::Dynamic>), "value"_a, "g"_a, "h"_a);
    m.def("explode", py::overload_cast<const hj::Jet<double, hj::Dynamic>&, Eigen::Ref<Eigen::Matrix<double, 1, hj::Dynamic>>, Eigen::Ref<Eigen::Matrix<double, hj::Dynamic, hj::Dynamic>>>(&hj::explode<double, hj::Dynamic>), "value"_a, "g"_a, "h"_a);
    m.def("explode", py::overload_cast<const hj::HyperJet<double, hj::Dynamic>&, Eigen::Ref<Eigen::Matrix<double, 1, hj::Dynamic>>, Eigen::Ref<Eigen::Matrix<double, hj::Dynamic, hj::Dynamic>>>(&hj::explode<double, hj::Dynamic>), "value"_a, "g"_a, "h"_a);
}
