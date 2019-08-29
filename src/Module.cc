#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <hyperjet/HyperJet.h>
#include <hyperjet/Jet.h>

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

    m.def("assign", [](const double value,
        Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> h) {
        return value;
    });

    m.def("assign", [](const hyperjet::Jet<double>& jet,
        Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> h) {
        if (g.size() > 0) {
            g = jet.g();
        }
        return jet.f();
    });

    m.def("assign", [](const hyperjet::HyperJet<double>& hyper_jet,
        Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> h) {
        if (g.size() > 0) {
            g = hyper_jet.g();
        }
        if (h.size() > 0) {
            h = hyper_jet.h();
        }
        return hyper_jet.f();
    });

    m.def("f", [](const double value) { return value; }, "value"_a);

    hyperjet::Jet<double>::register_python(m);
                
    hyperjet::HyperJet<double>::register_python(m);
}
