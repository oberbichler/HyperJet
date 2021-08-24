#include "common.h"

void bind_sscalar(pybind11::module &m)
{
    using T = hj::SScalar<double>;

    auto cls = py::class_<T>(m, "SScalar");

    // constructor
    cls
        .def(py::init<>())
        .def(py::init<T::Scalar>(), "f"_a = 0)
        .def(py::init<T::Scalar, T::Data>(), "f"_a = 0, "d"_a);

    // static methods
    cls
        .def_static("constant", &T::constant, "value"_a)
        .def_static("variable", &T::variable, "name"_a, "value"_a);

    // properties
    cls
        .def_property_readonly("f", &T::f)
        .def_property_readonly("size", &T::size);

    // methods
    cls
        .def("__abs__", &T::abs)
        .def("__len__", &T::size)
        .def("__pow__", &T::pow)
        .def("__repr__", &T::to_string)
        .def("abs", &T::abs)
        .def("d", &T::d, "variable"_a)
        .def("eval", &T::eval, "d"_a);

    // methods: arithmetic operations
    cls
        .def("reciprocal", &T::reciprocal)
        .def("sqrt", &T::sqrt)
        .def("cbrt", &T::cbrt)
        .def("cos", &T::cos)
        .def("sin", &T::sin)
        .def("tan", &T::tan)
        .def("acos", &T::acos)
        .def("asin", &T::asin)
        .def("atan", &T::atan)
        .def("atan2", &T::atan2)
        .def("arccos", &T::acos)
        .def("arcsin", &T::asin)
        .def("arctan", &T::atan)
        .def("arctan2", &T::atan2)
        .def("hypot", py::overload_cast<const T &, const T &>(&T::hypot))
        .def("hypot", py::overload_cast<const T &, const T &, const T &>(&T::hypot));

    // methods: hyperbolic functions
    cls
        .def("cosh", &T::cosh)
        .def("sinh", &T::sinh)
        .def("tanh", &T::tanh)
        .def("acosh", &T::acosh)
        .def("arccosh", &T::acosh)
        .def("asinh", &T::asinh)
        .def("arcsinh", &T::asinh)
        .def("atanh", &T::atanh)
        .def("arctanh", &T::atanh);

    // methods: exponents and logarithms
    cls
        .def("exp", &T::exp)
        .def("log", py::overload_cast<>(&T::log, py::const_))
        .def("log", py::overload_cast<typename T::Scalar>(&T::log, py::const_), "base"_a)
        .def("log2", &T::log2)
        .def("log10", &T::log10);

    // operators
    cls
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
        .def(double() / py::self);
}