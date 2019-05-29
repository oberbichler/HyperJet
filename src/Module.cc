#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <HyperJet/HyperJet.h>
#include <HyperJet/Jet.h>

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

    {
    using Type = hyperjet::HyperJet<double>;

    py::class_<Type>(m, "HyperJet")
        // constructors
        .def(py::init<int>(), "size"_a)
        .def(py::init<double, Type::Vector>(), "f"_a, "g"_a)
        .def(py::init<double, Type::Vector, Type::Matrix>(), "f"_a, "g"_a,
            "h"_a)
        // properties
        .def_property("f", py::overload_cast<>(&Type::f),
            [](Type& self, double value) {
                self.f() = value;
            })
        .def_property("g", py::overload_cast<>(&Type::g),
            [](Type& self, Eigen::Ref<const Type::Vector> value) {
                if (value.size() != self.size()) {
                    throw std::runtime_error("Invalid shape!");
                }
                self.g() = value;
            })
        .def_property("h", py::overload_cast<>(&Type::h),
            [](Type& self, Eigen::Ref<const Type::Matrix> value) {
                if (value.rows() != self.size() ||
                    value.cols() != self.size()) {
                    throw std::runtime_error("Invalid shape!");
                }
                self.h() = value;
            })
        // static methods
        .def_static("atan2", &Type::atan2)
        .def_static("variables", [](const std::vector<Type::Scalar> values) {
            const auto nb_variables = values.size();
            std::vector<Type> variables(nb_variables);
            for (int i = 0; i < nb_variables; i++) {
                Type::Vector g = Type::Vector::Zero(nb_variables);
                g[i] = 1;
                variables[i] = Type(values[i], g);
            }
            return variables;
        }, "values"_a)
        .def_static("variables", [](const std::vector<Type::Scalar> values,
            const int size, const int offset) {
            const auto nb_variables = values.size();
            std::vector<Type> variables(nb_variables);
            for (int i = 0; i < nb_variables; i++) {
                Type::Vector g = Type::Vector::Zero(size);
                g[offset + i] = 1;
                variables[i] = Type(values[i], g);
            }
            return variables;
        }, "values"_a, "size"_a, "offset"_a)
        // methods
        .def("__float__", py::overload_cast<>(&Type::f))
        .def("__len__", &Type::size)
        .def("__pow__", &Type::pow<double>)
        .def("__pow__", &Type::pow<int>)
        .def("__repr__", &Type::to_string)
        .def("acos", &Type::acos)
        .def("arccos", &Type::acos)
        .def("arcsin", &Type::asin)
        .def("arctan", &Type::atan)
        .def("arctan2", &Type::atan2)
        .def("asin", &Type::asin)
        .def("atan", &Type::atan)
        .def("cos", &Type::cos)
        .def("enlarge", py::overload_cast<size_t, size_t>(&Type::enlarge,
            py::const_), "left"_a=0, "right"_a=0)
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
        .def(py::pickle([](const Type& self) {
                return py::make_tuple(self.f(), self.g(), self.h());
            }, [](py::tuple tuple) {
                if (tuple.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                
                auto f = tuple[0].cast<double>();
                auto g = tuple[1].cast<Type::Vector>();
                auto h = tuple[2].cast<Type::Matrix>();

                return Type(f, g, h);
            }
        ))
        .def("__copy__", [](const Type& self) { return self; })
        .def("__deepcopy__", [](const Type& self, py::dict& memo) {
            return self; }, "memodict"_a)
    ;
    }

    {
    using Type = hyperjet::Jet<double>;

    py::class_<Type>(m, "Jet")
        // constructors
        .def(py::init<int>(), "size"_a)
        .def(py::init<double, Type::Vector>(), "f"_a, "g"_a)
        // properties
        .def_property("f", py::overload_cast<>(&Type::f),
            [](Type& self, double value) {
                self.f() = value;
            })
        .def_property("g", py::overload_cast<>(&Type::g),
            [](Type& self, Eigen::Ref<const Type::Vector> value) {
                if (value.size() != self.size()) {
                    throw std::runtime_error("Invalid shape!");
                }
                self.g() = value;
            })
        // static methods
        .def_static("atan2", &Type::atan2)
        .def_static("variables", [](const std::vector<Type::Scalar> values) {
            const auto nb_variables = values.size();
            std::vector<Type> variables(nb_variables);
            for (int i = 0; i < nb_variables; i++) {
                Type::Vector g = Type::Vector::Zero(nb_variables);
                g[i] = 1;
                variables[i] = Type(values[i], g);
            }
            return variables;
        }, "values"_a)
        .def_static("variables", [](const std::vector<Type::Scalar> values,
            const int size, const int offset) {
            const auto nb_variables = values.size();
            std::vector<Type> variables(nb_variables);
            for (int i = 0; i < nb_variables; i++) {
                Type::Vector g = Type::Vector::Zero(size);
                g[offset + i] = 1;
                variables[i] = Type(values[i], g);
            }
            return variables;
        }, "values"_a, "size"_a, "offset"_a)
        // methods
        .def("__float__", py::overload_cast<>(&Type::f))
        .def("__len__", &Type::size)
        .def("__pow__", &Type::pow<double>)
        .def("__pow__", &Type::pow<int>)
        .def("__repr__", &Type::to_string)
        .def("acos", &Type::acos)
        .def("arccos", &Type::acos)
        .def("arcsin", &Type::asin)
        .def("arctan", &Type::atan)
        .def("arctan2", &Type::atan2)
        .def("asin", &Type::asin)
        .def("atan", &Type::atan)
        .def("cos", &Type::cos)
        .def("enlarge", py::overload_cast<size_t, size_t>(&Type::enlarge,
            py::const_), "left"_a=0, "right"_a=0)
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
        .def(py::pickle([](const Type& self) {
                return py::make_tuple(self.f(), self.g());
            }, [](py::tuple tuple) {
                if (tuple.size() != 2) {
                    throw std::runtime_error("Invalid state!");
                }
                
                auto f = tuple[0].cast<double>();
                auto g = tuple[1].cast<Type::Vector>();

                return Type(f, g);
            }
        ))
        .def("__copy__", [](const Type& self) { return self; })
        .def("__deepcopy__", [](const Type& self, py::dict& memo) {
            return self; }, "memodict"_a)
    ;
    }
}
