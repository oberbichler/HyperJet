#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>

#include <hyperjet.h>

template <typename TScalar, hyperjet::index TSize>
void register_ddscalar(pybind11::module& m, const std::string& name)
{
    using namespace pybind11::literals;

    namespace py = pybind11;

    using Type = hyperjet::DDScalar<TScalar, TSize>;

    py::class_<Type>(m, name.c_str())
        .def(py::init(&Type::constant), "f"_a=0.0)
        .def_property("f", py::overload_cast<>(&Type::f, py::const_), &Type::set_f)
        // static methods
        .def_static("atan2", &Type::atan2)
        // methods
        .def("__abs__", &Type::abs)
        .def("__len__", &Type::size)
        .def("__pow__", &Type::pow)
        .def("__repr__", &Type::to_string)
        .def("abs", &Type::abs)
        .def("acos", &Type::acos)
        .def("arccos", &Type::acos)
        .def("arcsin", &Type::asin)
        .def("arctan", &Type::atan)
        .def("arctan2", &Type::atan2)
        .def("asin", &Type::asin)
        .def("atan", &Type::atan)
        .def("cos", &Type::cos)
        .def("sin", &Type::sin)
        .def("sqrt", &Type::sqrt)
        .def("tan", &Type::tan)
        // operators
        .def(-py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def(py::self == double())
        .def(py::self != double())
        .def(py::self < double())
        .def(py::self > double())
        .def(py::self <= double())
        .def(py::self >= double())
        .def(double() == py::self)
        .def(double() != py::self)
        .def(double() < py::self)
        .def(double() > py::self)
        .def(double() <= py::self)
        .def(double() >= py::self)
        .def(py::self + py::self)
        .def(py::self + double())
        .def(py::self - py::self)
        .def(py::self - double())
        .def(py::self * py::self)
        .def(py::self * double())
        .def(py::self / py::self)
        .def(py::self / double())
        .def(py::self += py::self)
        .def(py::self -= py::self)
        // .def(py::self *= py::self)
        // .def(py::self *= double())
        // .def(py::self /= py::self)
        // .def(py::self /= double())
        .def(double() + py::self)
        .def(double() - py::self)
        .def(double() * py::self)
        .def(double() / py::self)
        // serialization
        .def(py::pickle(
            [](const Type& self) {
                return py::make_tuple(self.m_data);
            },
            [](py::tuple tuple) {
                if (hyperjet::length(tuple) != 1) {
                    throw std::runtime_error("Invalid state!");
                }

                const auto data = tuple[0].cast<Type::Data>();

                return Type(data);
            }))
        .def("__copy__", [](const Type& self) { return self; })
        .def("__deepcopy__", [](const Type& self, py::dict& memo) { return self; }, "memodict"_a);
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