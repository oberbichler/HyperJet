#pragma once

#include <Eigen/Core>

#include <hyperjet.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

namespace hj = hyperjet;

namespace py = pybind11;
using namespace py::literals;

template <typename T>
auto bind(py::module &m, const std::string &name)
{
    py::class_<T> cls(m, name.c_str());

    // constructor
    cls
        .def(py::init(py::overload_cast<typename T::Scalar, hj::index>(&T::constant)), "f"_a=0, "size"_a)
        .def(py::init(py::overload_cast<const typename T::Data&>(&T::create)), "data"_a)
        .def(py::init(py::overload_cast<typename T::Scalar>(&T::constant)), "f"_a=0);

    if constexpr(T::order() == 1) {
        cls
            .def(py::init(&T::from_gradient), "f"_a, "g"_a);
    } else {
        cls
            .def(py::init(&T::from_arrays), "f"_a, "g"_a, "hm"_a);
    }

    // properties
    cls
        .def_property("f", py::overload_cast<>(&T::f, py::const_), &T::set_f);
    
    // static read-only properties
    cls
        .def_property_readonly_static("is_dynamic", [](py::object) { return T::is_dynamic(); })
        .def_property_readonly_static("order", [](py::object) { return T::order(); });
    
    // read-only properties
    cls
        .def_property_readonly("data", py::overload_cast<>(&T::adata))
        .def_property_readonly("g", py::overload_cast<>(&T::ag))
        .def_property_readonly("size", &T::size);
    
    // static methods
    cls
        .def_static("constant", py::overload_cast<typename T::Scalar, hj::index>(&T::constant), "f"_a, "size"_a)
        .def_static("constant", py::overload_cast<typename T::Scalar>(&T::constant), "f"_a)
        .def_static("empty", py::overload_cast<>(&T::empty))
        .def_static("empty", py::overload_cast<hj::index>(&T::empty), "size"_a)
        .def_static("variable", py::overload_cast<hj::index, double, hj::index>(&T::variable), "i"_a, "f"_a, "size"_a)
        .def_static("zero", py::overload_cast<>(&T::zero))
        .def_static("zero", py::overload_cast<hj::index>(&T::zero), "size"_a);

    if constexpr(T::is_dynamic()) {
        cls
            .def_static("variables", [](const std::vector<typename T::Scalar>& values) { return T::variables(values); }, "values"_a);
    } else {
        cls
            .def_static("variable", py::overload_cast<hj::index, double>(&T::variable), "i"_a, "f"_a)
            .def_static("variables", [](const std::array<typename T::Scalar, T::static_size()>& values) { return T::template variables<T::static_size()>(values); }, "values"_a);
    }

    // methods
    cls
        .def("__abs__", &T::abs)
        .def("__len__", &T::size)
        .def("__pow__", &T::pow)
        .def("__repr__", &T::to_string)
        .def("abs", &T::abs)
        .def("eval", &T::eval, "d"_a)
        .def("h", py::overload_cast<hj::index, hj::index>(&T::h), "row"_a, "col"_a)
        .def("set_h", py::overload_cast<hj::index, hj::index, typename T::Scalar>(&T::set_h), "row"_a, "col"_a, "value"_a)
        .def("hm", py::overload_cast<std::string>(&T::hm, py::const_), "mode"_a="full")
        .def("set_hm", &T::set_hm, "value"_a);
    
    if constexpr(T::is_dynamic()) {
        cls
            .def("resize", &T::resize, "size"_a)
            .def("pad_right", &T::pad_right, "new_size"_a)
            .def("pad_left", &T::pad_left, "new_size"_a);
    }

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
        .def("hypot", py::overload_cast<const T&, const T&>(&T::hypot))
        .def("hypot", py::overload_cast<const T&, const T&, const T&>(&T::hypot));
    
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

    // serialization
    cls
        .def(py::pickle(
            [](const T& self) {
                return py::make_tuple(self.m_data);
            },
            [](py::tuple tuple) {
                if (hyperjet::length(tuple) != 1) {
                    throw std::runtime_error("Invalid state!");
                }

                const auto data = tuple[0].cast<typename T::Data>();

                if constexpr(T::is_dynamic()) {
                    return T(data, T::size_from_data_length(hj::length(data)));
                } else {
                    return T(data);
                }
            }))
        .def("__copy__", [](const T& self) { return self; })
        .def("__deepcopy__", [](const T& self, py::dict& memo) { return self; }, "memodict"_a);

    m.def("hypot", py::overload_cast<const T&, const T&>(&T::hypot));
    m.def("hypot", py::overload_cast<const T&, const T&, const T&>(&T::hypot));

    return cls;
}