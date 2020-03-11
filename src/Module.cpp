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

    hyperjet::Jet<double, -1>::register_python(m, "JetXd");
    hyperjet::Jet<double, 1>::register_python(m, "Jet1d");
    hyperjet::Jet<double, 2>::register_python(m, "Jet2d");
    hyperjet::Jet<double, 3>::register_python(m, "Jet3d");
    hyperjet::Jet<double, 4>::register_python(m, "Jet4d");
    hyperjet::Jet<double, 5>::register_python(m, "Jet5d");
    hyperjet::Jet<double, 6>::register_python(m, "Jet6d");
    hyperjet::Jet<double, 7>::register_python(m, "Jet7d");
    hyperjet::Jet<double, 8>::register_python(m, "Jet8d");
    hyperjet::Jet<double, 9>::register_python(m, "Jet9d");
    hyperjet::Jet<double, 10>::register_python(m, "Jet10d");
    hyperjet::Jet<double, 11>::register_python(m, "Jet11d");
    hyperjet::Jet<double, 12>::register_python(m, "Jet12d");
    hyperjet::Jet<double, 13>::register_python(m, "Jet13d");
    hyperjet::Jet<double, 14>::register_python(m, "Jet14d");
    hyperjet::Jet<double, 15>::register_python(m, "Jet15d");
    hyperjet::Jet<double, 16>::register_python(m, "Jet16d");

    hyperjet::HyperJet<double, -1>::register_python(m, "HyperJetXd");
    hyperjet::HyperJet<double, 1>::register_python(m, "HyperJet1d");
    hyperjet::HyperJet<double, 2>::register_python(m, "HyperJet2d");
    hyperjet::HyperJet<double, 3>::register_python(m, "HyperJet3d");
    hyperjet::HyperJet<double, 4>::register_python(m, "HyperJet4d");
    hyperjet::HyperJet<double, 5>::register_python(m, "HyperJet5d");
    hyperjet::HyperJet<double, 6>::register_python(m, "HyperJet6d");
    hyperjet::HyperJet<double, 7>::register_python(m, "HyperJet7d");
    hyperjet::HyperJet<double, 8>::register_python(m, "HyperJet8d");
    hyperjet::HyperJet<double, 9>::register_python(m, "HyperJet9d");
    hyperjet::HyperJet<double, 10>::register_python(m, "HyperJet10d");
    hyperjet::HyperJet<double, 11>::register_python(m, "HyperJet11d");
    hyperjet::HyperJet<double, 12>::register_python(m, "HyperJet12d");
    hyperjet::HyperJet<double, 13>::register_python(m, "HyperJet13d");
    hyperjet::HyperJet<double, 14>::register_python(m, "HyperJet14d");
    hyperjet::HyperJet<double, 15>::register_python(m, "HyperJet15d");
    hyperjet::HyperJet<double, 16>::register_python(m, "HyperJet16d");
}
