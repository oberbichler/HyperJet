#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <hyperjet.h>

template <typename TScalar, hyperjet::index TSize>
void register_ddscalar(pybind11::module& m, const std::string& name)
{
    using namespace pybind11::literals;

    namespace py = pybind11;

    using Type = hyperjet::DDScalar<TScalar, TSize>;

    py::class_<Type>(m, name.c_str())
        .def(py::init(&Type::constant))
        .def_property("f", py::overload_cast<>(&Type::f, py::const_), &Type::set_f);
}

PYBIND11_MODULE(hyperjet, m)
{
    m.doc() = "HyperJet by Thomas Oberbichler";
    m.attr("__author__") = "Thomas Oberbichler";
    m.attr("__copyright__") = "Copyright (c) 2019-2020, Thomas Oberbichler";
    m.attr("__version__") = HYPERJET_VERSION;
    m.attr("__email__") = "thomas.oberbichler@gmail.com";
    m.attr("__status__") = "Development";

    register_ddscalar<double, 1>(m, "DD1Scalar");
    register_ddscalar<double, 2>(m, "DD2Scalar");
    register_ddscalar<double, 3>(m, "DD3Scalar");
    register_ddscalar<double, 4>(m, "DD4Scalar");
    register_ddscalar<double, 5>(m, "DD5Scalar");
    register_ddscalar<double, 6>(m, "DD6Scalar");
    register_ddscalar<double, 7>(m, "DD7Scalar");
    register_ddscalar<double, 8>(m, "DD8Scalar");
}