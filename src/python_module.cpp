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
    namespace hj = hyperjet;

    using Type = hyperjet::DDScalar<TScalar, TSize>;

    auto py_class = py::class_<Type>(m, name.c_str())
        // constructor
        .def(py::init(py::overload_cast<TScalar, hj::index>(&Type::constant)), "f"_a=0, "size"_a)
        .def(py::init(py::overload_cast<const typename Type::Data&>(&Type::create)), "data"_a)
        // properties
        .def_property("f", py::overload_cast<>(&Type::f, py::const_), &Type::set_f)
        // static read-only properties
        .def_property_readonly_static("is_dynamic", [](py::object) { return Type::is_dynamic(); })
        // read-only properties
        .def_property_readonly("data", py::overload_cast<>(&Type::data, py::const_))
        .def_property_readonly("size", &Type::size)
        // static methods
        .def_static("constant", py::overload_cast<TScalar, hj::index>(&Type::constant), "f"_a, "size"_a)
        .def_static("constant", py::overload_cast<TScalar>(&Type::constant), "f"_a)
        .def_static("empty", py::overload_cast<>(&Type::empty))
        .def_static("empty", py::overload_cast<hj::index>(&Type::empty), "size"_a)
        .def_static("variable", py::overload_cast<hj::index, double, hj::index>(&Type::variable), "i"_a, "f"_a, "size"_a)
        .def_static("variables", py::overload_cast<std::vector<TScalar>>(&Type::variables), "values"_a)
        .def_static("zero", py::overload_cast<>(&Type::zero))
        .def_static("zero", py::overload_cast<hj::index>(&Type::zero), "size"_a)
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
                    return Type(data, hj::size_from_data_length(data));
                } else {
                    return Type(data);
                }
            }))
        .def("__copy__", [](const Type& self) { return self; })
        .def("__deepcopy__", [](const Type& self, py::dict& memo) { return self; }, "memodict"_a);

    if constexpr(Type::is_dynamic()) {
        py_class
            // constructor
            .def(py::init(py::overload_cast<TScalar>(&Type::constant)), "f"_a=0)
            // methods
            .def("resize", &Type::resize, "size"_a);
    } else {
        py_class
            // constructor
            .def(py::init(py::overload_cast<TScalar>(&Type::constant)), "f"_a=0)
            // static methods
            .def_static("variable", py::overload_cast<hj::index, double>(&Type::variable), "i"_a, "f"_a)
            .def_static("variables", py::overload_cast<std::array<TScalar, TSize>>(&Type::variables), "values"_a);
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

    register_ddscalar<double, -1>(m, "DDScalar");
    register_ddscalar<double, 1>(m, "DD1Scalar");
    register_ddscalar<double, 2>(m, "DD2Scalar");
    register_ddscalar<double, 3>(m, "DD3Scalar");
    register_ddscalar<double, 4>(m, "DD4Scalar");
    register_ddscalar<double, 5>(m, "DD5Scalar");
    register_ddscalar<double, 6>(m, "DD6Scalar");
    register_ddscalar<double, 7>(m, "DD7Scalar");
    register_ddscalar<double, 8>(m, "DD8Scalar");
    register_ddscalar<double, 9>(m, "DD9Scalar");
    register_ddscalar<double, 10>(m, "DD10Scalar");
    register_ddscalar<double, 11>(m, "DD11Scalar");
    register_ddscalar<double, 12>(m, "DD12Scalar");
    register_ddscalar<double, 13>(m, "DD13Scalar");
    register_ddscalar<double, 14>(m, "DD14Scalar");
    register_ddscalar<double, 15>(m, "DD15Scalar");
    register_ddscalar<double, 16>(m, "DD16Scalar");
}