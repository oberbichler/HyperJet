#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include <hyperjet.h>

template <hyperjet::index TOrder, typename TScalar, hyperjet::index TSize>
void register_ddscalar(pybind11::module& m, const std::string& name)
{
    using namespace pybind11::literals;

    namespace py = pybind11;
    namespace hj = hyperjet;

    using Type = hyperjet::DDScalar<TOrder, TScalar, TSize>;

    auto py_class = py::class_<Type>(m, name.c_str())
        // constructor
        .def(py::init(py::overload_cast<TScalar, hj::index>(&Type::constant)), "f"_a=0, "size"_a)
        .def(py::init(py::overload_cast<const typename Type::Data&>(&Type::create)), "data"_a)
        // properties
        .def_property("f", py::overload_cast<>(&Type::f, py::const_), &Type::set_f)
        // static read-only properties
        .def_property_readonly_static("is_dynamic", [](py::object) { return Type::is_dynamic(); })
        .def_property_readonly_static("order", [](py::object) { return Type::order(); })
        // read-only properties
        .def_property_readonly("data", py::overload_cast<>(&Type::adata))
        .def_property_readonly("g", py::overload_cast<>(&Type::ag))
        .def_property_readonly("size", &Type::size)
        // static methods
        .def_static("constant", py::overload_cast<TScalar, hj::index>(&Type::constant), "f"_a, "size"_a)
        .def_static("constant", py::overload_cast<TScalar>(&Type::constant), "f"_a)
        .def_static("empty", py::overload_cast<>(&Type::empty))
        .def_static("empty", py::overload_cast<hj::index>(&Type::empty), "size"_a)
        .def_static("variable", py::overload_cast<hj::index, double, hj::index>(&Type::variable), "i"_a, "f"_a, "size"_a)
        .def_static("variables", &Type::variables, "values"_a)
        .def_static("zero", py::overload_cast<>(&Type::zero))
        .def_static("zero", py::overload_cast<hj::index>(&Type::zero), "size"_a)
        // methods
        .def("__abs__", &Type::abs)
        .def("__len__", &Type::size)
        .def("__pow__", &Type::pow)
        .def("__repr__", &Type::to_string)
        .def("abs", &Type::abs)
        .def("eval", &Type::eval, "d"_a)
        .def("h", py::overload_cast<hj::index, hj::index>(&Type::h), "row"_a, "col"_a)
        .def("set_h", py::overload_cast<hj::index, hj::index, TScalar>(&Type::set_h), "row"_a, "col"_a, "value"_a)
        .def("hm", py::overload_cast<std::string>(&Type::hm, py::const_), "mode"_a="full")
        .def("set_hm", &Type::set_hm, "value"_a)
        // methods: arithmetic operations
        .def("reciprocal", &Type::reciprocal)
        .def("sqrt", &Type::sqrt)
        .def("cbrt", &Type::cbrt)
        .def("cos", &Type::cos)
        .def("sin", &Type::sin)
        .def("tan", &Type::tan)
        .def("acos", &Type::acos)
        .def("asin", &Type::asin)
        .def("atan", &Type::atan)
        .def("atan2", &Type::atan2)
        .def("arccos", &Type::acos)
        .def("arcsin", &Type::asin)
        .def("arctan", &Type::atan)
        .def("arctan2", &Type::atan2)
        // methods: hyperbolic functions
        .def("cosh", &Type::cosh)
        .def("sinh", &Type::sinh)
        .def("tanh", &Type::tanh)
        .def("acosh", &Type::acosh)
        .def("arccosh", &Type::acosh)
        .def("asinh", &Type::asinh)
        .def("arcsinh", &Type::asinh)
        .def("atanh", &Type::atanh)
        .def("arctanh", &Type::atanh)
        // methods: exponents and logarithms
        .def("exp", &Type::exp)
        .def("log", py::overload_cast<>(&Type::log, py::const_))
        .def("log", py::overload_cast<TScalar>(&Type::log, py::const_), "base"_a)
        .def("log2", &Type::log2)
        .def("log10", &Type::log10)
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
        .def(py::self *= py::self)
        .def(py::self *= double())
        .def(py::self /= py::self)
        .def(py::self /= double())
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

                const auto data = tuple[0].cast<typename Type::Data>();

                if constexpr(Type::is_dynamic()) {
                    return Type(data, hj::size_from_data_length<TOrder>(data));
                } else {
                    return Type(data);
                }
            }))
        .def("__copy__", [](const Type& self) { return self; })
        .def("__deepcopy__", [](const Type& self, py::dict& memo) { return self; }, "memodict"_a);

    if constexpr(Type::order() == 1) {
        // FIXME: add from_gradient
    } else {
        py_class
            .def(py::init(&Type::from_arrays), "f"_a, "g"_a, "hm"_a);
    }

    if constexpr(Type::is_dynamic()) {
        py_class
            // constructor
            .def(py::init(py::overload_cast<TScalar>(&Type::constant)), "f"_a=0)
            // methods
            .def("resize", &Type::resize, "size"_a)
            .def("pad_right", &Type::pad_right, "new_size"_a)
            .def("pad_left", &Type::pad_left, "new_size"_a);
    } else {
        py_class
            // constructor
            .def(py::init(py::overload_cast<TScalar>(&Type::constant)), "f"_a=0)
            // static methods
            .def_static("variable", py::overload_cast<hj::index, double>(&Type::variable), "i"_a, "f"_a);
    }
}

PYBIND11_MODULE(hyperjet, m)
{
    m.doc() = "HyperJet by Thomas Oberbichler";
    m.attr("__author__") = "Thomas Oberbichler";
    m.attr("__copyright__") = "Copyright (c) 2019-2021, Thomas Oberbichler";
    m.attr("__version__") = HYPERJET_VERSION;
    m.attr("__email__") = "thomas.oberbichler@gmail.com";
    m.attr("__status__") = "Development";

    register_ddscalar<1, double, -1>(m, "DScalar");
    register_ddscalar<1, double, 1>(m, "D1Scalar");
    register_ddscalar<1, double, 2>(m, "D2Scalar");
    register_ddscalar<1, double, 3>(m, "D3Scalar");
    register_ddscalar<1, double, 4>(m, "D4Scalar");
    register_ddscalar<1, double, 5>(m, "D5Scalar");
    register_ddscalar<1, double, 6>(m, "D6Scalar");
    register_ddscalar<1, double, 7>(m, "D7Scalar");
    register_ddscalar<1, double, 8>(m, "D8Scalar");
    register_ddscalar<1, double, 9>(m, "D9Scalar");
    register_ddscalar<1, double, 10>(m, "D10Scalar");
    register_ddscalar<1, double, 11>(m, "D11Scalar");
    register_ddscalar<1, double, 12>(m, "D12Scalar");
    register_ddscalar<1, double, 13>(m, "D13Scalar");
    register_ddscalar<1, double, 14>(m, "D14Scalar");
    register_ddscalar<1, double, 15>(m, "D15Scalar");
    register_ddscalar<1, double, 16>(m, "D16Scalar");

    register_ddscalar<2, double, -1>(m, "DDScalar");
    register_ddscalar<2, double, 1>(m, "DD1Scalar");
    register_ddscalar<2, double, 2>(m, "DD2Scalar");
    register_ddscalar<2, double, 3>(m, "DD3Scalar");
    register_ddscalar<2, double, 4>(m, "DD4Scalar");
    register_ddscalar<2, double, 5>(m, "DD5Scalar");
    register_ddscalar<2, double, 6>(m, "DD6Scalar");
    register_ddscalar<2, double, 7>(m, "DD7Scalar");
    register_ddscalar<2, double, 8>(m, "DD8Scalar");
    register_ddscalar<2, double, 9>(m, "DD9Scalar");
    register_ddscalar<2, double, 10>(m, "DD10Scalar");
    register_ddscalar<2, double, 11>(m, "DD11Scalar");
    register_ddscalar<2, double, 12>(m, "DD12Scalar");
    register_ddscalar<2, double, 13>(m, "DD13Scalar");
    register_ddscalar<2, double, 14>(m, "DD14Scalar");
    register_ddscalar<2, double, 15>(m, "DD15Scalar");
    register_ddscalar<2, double, 16>(m, "DD16Scalar");
}