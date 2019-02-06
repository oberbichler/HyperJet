#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <HyperJet/HyperJet.h>
#include <HyperJet/Jet.h>

PYBIND11_MODULE(HyperJet, m) {
    m.doc() = "HyperJet by Thomas Oberbichler";
    m.attr("__author__") = "Thomas Oberbichler";
    m.attr("__copyright__") = "Copyright (c) 2019, Thomas Oberbichler";
    m.attr("__version__") = HYPERJET_VERSION;
    m.attr("__email__") = "thomas.oberbichler@gmail.com";
    m.attr("__status__") = "Development";

    namespace py = pybind11;

#if defined(EIGEN_USE_BLAS)
    m.attr("USE_BLAS") = true;
#else
    m.attr("USE_BLAS") = false;
#endif // EIGEN_USE_BLAS

    {
    using Type = HyperJet::HyperJet<double>;

    py::class_<Type>(m, "HyperJet")
        .def(py::init<int>())
        .def(py::init<double, Type::Vector>())
        .def(py::init<double, Type::Vector, Type::Matrix>())
        .def_property("f", &Type::f, [](Type& self, double value) {
            self.f() = value;})
        .def_property_readonly("g", &Type::g)
        .def_property_readonly("h", &Type::h)
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
        .def(py::self *= py::self)
        .def(py::self *= double())
        .def(py::self /= py::self)
        .def(py::self /= double())
        .def(double() + py::self)
        .def(double() - py::self)
        .def(double() * py::self)
        .def(double() / py::self)
        .def("__repr__", &Type::toString)
        .def("__len__", &Type::size)
        .def("sqrt", &Type::sqrt)
        .def("cos", &Type::cos)
        .def("sin", &Type::sin)
        .def("tan", &Type::tan)
        .def("acos", &Type::acos)
        .def("asin", &Type::asin)
        .def("atan", &Type::atan)
        .def_static("atan2", &Type::atan2)
        .def("arccos", &Type::acos)
        .def("arcsin", &Type::asin)
        .def("arctan", &Type::atan)
        .def("arctan2", &Type::atan2)
        .def("__pow__", &Type::pow<int>)
        .def("__pow__", &Type::pow<double>)
    ;
    }

    {
    using Type = HyperJet::Jet<double>;

    py::class_<Type>(m, "Jet")
        .def(py::init<int>())
        .def(py::init<double, Type::Vector>())
        .def(py::init<double, Type::Vector>())
        .def_property("f", &Type::f, [](Type& self, double value) {
            self.f() = value;})
        .def_property_readonly("g", &Type::g)
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
        .def(py::self *= py::self)
        .def(py::self *= double())
        .def(py::self /= py::self)
        .def(py::self /= double())
        .def(double() + py::self)
        .def(double() - py::self)
        .def(double() * py::self)
        .def(double() / py::self)
        .def("__repr__", &Type::toString)
        .def("__len__", &Type::size)
        .def("sqrt", &Type::sqrt)
        .def("cos", &Type::cos)
        .def("sin", &Type::sin)
        .def("tan", &Type::tan)
        .def("acos", &Type::acos)
        .def("asin", &Type::asin)
        .def("atan", &Type::atan)
        .def_static("atan2", &Type::atan2)
        .def("arccos", &Type::acos)
        .def("arcsin", &Type::asin)
        .def("arctan", &Type::atan)
        .def("arctan2", &Type::atan2)
        .def("__pow__", &Type::pow<int>)
        .def("__pow__", &Type::pow<double>)
    ;
    }
}
