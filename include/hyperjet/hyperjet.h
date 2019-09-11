//     __  __                          __     __
//    / / / /_  ______  ___  _____    / /__  / /_
//   / /_/ / / / / __ \/ _ \/ ___/_  / / _ \/ __/
//  / __  / /_/ / /_/ /  __/ /  / /_/ /  __/ /_
// /_/ /_/\__, / .___/\___/_/   \____/\___/\__/
//       /____/_/
//
// Copyright (c) 2019 Thomas Oberbichler

#pragma once

#include <Eigen/Core>

#include <string>

namespace hyperjet {

template <typename T = double>
class Jet {
public:     // types
    using Scalar = T;
    using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;

private:    // variables
    T m_f;
    Vector m_g;

public:     // constructors
    EIGEN_STRONG_INLINE
    Jet()
    : m_f(0), m_g(0)
    {
    }

    EIGEN_STRONG_INLINE
    Jet(const int size)
    : m_f(0), m_g(Vector::Zero(size))
    {
    }

    EIGEN_STRONG_INLINE
    Jet(const T f, const int size)
    : m_f(f), m_g(Vector::Zero(size))
    {
    }

    template <typename Derived>
    EIGEN_STRONG_INLINE
    Jet(const T f, const Eigen::DenseBase<Derived>& g)
    : m_f(f), m_g(g)
    {
    }

public:     // static methods
    static Jet<T> variable(const double value, const int size, const int index)
    {
        Jet<T> result(value, size);
        result.g(index) = 1;
        return result;
    }

public:     // methods
    T& f()
    {
        return m_f;
    }

    T f() const
    {
        return m_f;
    }

    Eigen::Ref<Vector> g()
    {
        return m_g;
    }

    Eigen::Ref<const Vector> g() const
    {
        return m_g;
    }

    T& g(int index)
    {
        return m_g(index);
    }

    T g(const int index) const
    {
        return m_g(index);
    }


    inline int size() const
    {
        return static_cast<int>(m_g.size());
    }

    Jet enlarge(const int left, const int right) const
    {
        if (left < 0) {
            throw std::runtime_error("Negative value for 'left'");
        }

        if (right < 0) {
            throw std::runtime_error("Negative value for 'right'");
        }

        Jet result(static_cast<int>(this->size() + left + right));

        result.m_f = m_f;

        if (!left) {
            result.m_g.segment(left, this->size()) = m_g;
        } else {
            result.m_g.segment(left, this->size()) = m_g;
        }

        return result;
    }

    Jet operator-() const
    {
        const auto f = -m_f;
        const auto g = -m_g;
        return Jet(f, g);
    }

    Jet operator+(const Jet& rhs) const
    {
        const auto f = m_f + rhs.m_f;
        const auto g = m_g + rhs.m_g;
        return Jet(f, g);
    }

    Jet operator+(const T rhs) const
    {
        const auto f = m_f + rhs;
        const auto g = m_g;
        return Jet(f, g);
    }

    Jet operator-(const Jet& rhs) const
    {
        const auto f = m_f - rhs.m_f;
        const auto g = m_g - rhs.m_g;
        return Jet(f, g);
    }

    Jet operator-(const T rhs) const
    {
        const auto f = m_f - rhs;
        const auto g = m_g;
        return Jet(f, g);
    }

    Jet operator*(const Jet& rhs) const
    {
        const auto f = m_f * rhs.m_f;
        const auto g = m_f * rhs.m_g + rhs.m_f * m_g;
        return Jet(f, g);
    }

    Jet operator*(const T rhs) const
    {
        const auto f = m_f * rhs;
        const auto g = rhs * m_g;
        return Jet(f, g);
    }

    Jet operator/(const Jet& rhs) const
    {
        const auto f = m_f / rhs.m_f;
        const auto g = m_g / rhs.m_f - m_f * rhs.m_g / (rhs.m_f * rhs.m_f);
        return Jet(f, g);
    }

    Jet operator/(const T rhs) const
    {
        const auto f = m_f / rhs;
        const auto g = m_g / rhs;
        return Jet(f, g);
    }

    Jet& operator+=(const Jet& rhs)
    {
        m_f += rhs.m_f;
        m_g += rhs.m_g;
        return *this;
    }

    Jet& operator-=(const Jet& rhs)
    {
        m_f -= rhs.m_f;
        m_g -= rhs.m_g;
        return *this;
    }

    Jet& operator*=(const Jet& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    Jet& operator*=(const T rhs)
    {
        m_f *= rhs;
        m_g *= rhs;
        return *this;
    }

    Jet& operator/=(const Jet& rhs)
    {
        *this = *this / rhs;
        return *this;
    }

    Jet& operator/=(const T rhs)
    {
        m_f /= rhs;
        m_g /= rhs;
        return *this;
    }

    friend Jet operator+(const T lhs, const Jet& rhs)
    {
        return rhs.operator+(lhs);
    }

    friend Jet operator-(const T lhs, const Jet& rhs)
    {
        const auto f = lhs - rhs.m_f;
        const auto g = -rhs.m_g;
        return Jet(f, g);
    }

    friend Jet operator*(const T lhs, const Jet& rhs)
    {
        return rhs.operator*(lhs);
    }

    friend Jet operator/(const T lhs, const Jet& rhs)
    {
        const auto f = lhs / rhs.m_f;
        const auto g = -lhs * rhs.m_g / (rhs.m_f * rhs.m_f);
        return Jet(f, g);
    }

    inline Jet abs() const
    {
        return m_f < 0 ? -(*this) : *this;
    }

    inline Jet sqrt() const
    {
        const auto f = std::sqrt(m_f);
        const auto g = m_g / (2 * f);
        return Jet(f, g);
    }

    inline Jet cos() const
    {
        const auto f = std::cos(m_f);
        const auto g = -std::sin(m_f) * m_g;
        return Jet(f, g);
    }

    inline Jet sin() const
    {
        const auto f = std::sin(m_f);
        const auto g = std::cos(m_f) * m_g;
        return Jet(f, g);
    }

    inline Jet tan() const
    {
        const auto f = std::tan(m_f);
        const auto g = m_g * (f * f + 1);
        return Jet(f, g);
    }

    inline Jet acos() const
    {
        const auto f = std::acos(m_f);
        const auto g = -m_g / std::sqrt(-m_f * m_f + 1);
        return Jet(f, g);
    }

    inline Jet asin() const
    {
        const auto f = std::asin(m_f);
        const auto g = m_g / std::sqrt(1 - m_f * m_f);
        return Jet(f, g);
    }

    inline Jet atan() const
    {
        const auto f = std::atan(m_f);
        const auto g = m_g / (m_f * m_f + 1);
        return Jet(f, g);
    }

    static inline Jet atan2(const Jet& a, const Jet& b)
    {
        const auto tmp = a.m_f * a.m_f + b.m_f * b.m_f;

        const auto f = std::atan2(a.m_f, b.m_f);
        const auto g = (a.m_g * b.m_f - a.m_f * b.m_g) / tmp;
        return Jet(f, g);
    }

    template <typename U>
    inline Jet pow(const U b) const
    {
        const auto f = std::pow(m_f, b);
        const auto g = b * std::pow(m_f, b - U(1)) * m_g;
        return Jet(f, g);
    }

    bool operator==(const Jet& rhs) const
    {
        return m_f == rhs.m_f;
    }

    bool operator!=(const Jet& rhs) const
    {
        return m_f != rhs.m_f;
    }

    bool operator<(const Jet& rhs) const
    {
        return m_f < rhs.m_f;
    }

    bool operator>(const Jet& rhs) const
    {
        return m_f > rhs.m_f;
    }

    bool operator<=(const Jet& rhs) const
    {
        return m_f <= rhs.m_f;
    }

    bool operator>=(const Jet& rhs) const
    {
        return m_f >= rhs.m_f;
    }

    bool operator==(const T rhs) const
    {
        return m_f == rhs;
    }

    bool operator!=(const T rhs) const
    {
        return m_f != rhs;
    }

    bool operator<(const T rhs) const
    {
        return m_f < rhs;
    }

    bool operator>(const T rhs) const
    {
        return m_f > rhs;
    }

    bool operator<=(const T rhs) const
    {
        return m_f <= rhs;
    }

    bool operator>=(const T rhs) const
    {
        return m_f >= rhs;
    }

    friend bool operator==(const T lhs, const Jet& rhs)
    {
        return rhs.operator==(lhs);
    }

    friend bool operator!=(const T lhs, const Jet& rhs)
    {
        return rhs.operator!=(lhs);
    }

    friend bool operator<(const T lhs, const Jet& rhs)
    {
        return rhs.operator>(lhs);
    }

    friend bool operator>(const T lhs, const Jet& rhs)
    {
        return rhs.operator<(lhs);
    }

    friend bool operator<=(const T lhs, const Jet& rhs)
    {
        return rhs.operator>=(lhs);
    }

    friend bool operator>=(const T lhs, const Jet& rhs)
    {
        return rhs.operator<=(lhs);
    }

    std::string to_string() const
    {
        return "Jet<" + std::to_string(m_f) + ">";
    }

public:     // python
    static void register_python(pybind11::module& m)
    {
        using namespace pybind11::literals;
        namespace py = pybind11;

        using Type = hyperjet::Jet<double>;

        const std::string name = "Jet";

        py::class_<Type>(m, name.c_str())
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
            .def_static("variable", &Type::variable, "value"_a, "size"_a, "index"_a)
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
            .def("__abs__", &Type::abs)
            .def("__len__", &Type::size)
            .def("__pow__", &Type::pow<double>)
            .def("__pow__", &Type::pow<int>)
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
            .def("enlarge", py::overload_cast<int, int>(&Type::enlarge,
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
}; // class Jet

template <typename T = double>
class HyperJet {
public:     // types
    using Scalar = T;
    using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

private:    // variables
    T m_f;
    Vector m_g;
    Matrix m_h;

public:     // constructors
    EIGEN_STRONG_INLINE
    HyperJet()
    : m_f(0), m_g(0), m_h(0, 0)
    {
    }

    EIGEN_STRONG_INLINE
    HyperJet(const int size)
    : m_f(0), m_g(Vector::Zero(size)), m_h(Matrix::Zero(size, size))
    {
    }

    EIGEN_STRONG_INLINE
    HyperJet(const T value, const int size)
    : m_f(value), m_g(Vector::Zero(size)), m_h(Matrix::Zero(size, size))
    {
    }

    template <typename Derived>
    EIGEN_STRONG_INLINE
    HyperJet(const T f, const Eigen::DenseBase<Derived>& g)
    : m_f(f), m_g(g), m_h(Matrix::Zero(g.size(), g.size()))
    {
    }

    template <typename Derived1, typename Derived2>
    EIGEN_STRONG_INLINE
    HyperJet(const T f, const Eigen::DenseBase<Derived1>& g, const Eigen::DenseBase<Derived2>& h)
    : m_f(f), m_g(g), m_h(h)
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (h.rows() != h.cols()) {
            throw new std::runtime_error("Hessian is not a square matrix");
        }

        if (g.size() != h.rows()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif
    }

public:     // static methods
    static HyperJet<T> variable(const double value, const int size,
        const int index)
    {
        HyperJet<T> result(value, size);
        result.g(index) = 1;
        return result;
    }

public:     // methods
    T& f()
    {
        return m_f;
    }

    T f() const
    {
        return m_f;
    }

    Eigen::Ref<Vector> g()
    {
        return m_g;
    }

    Eigen::Ref<const Vector> g() const
    {
        return m_g;
    }

    T& g(int index)
    {
        return m_g(index);
    }

    T g(const int index) const
    {
        return m_g(index);
    }

    Eigen::Ref<Matrix> h()
    {
        return m_h;
    }

    Eigen::Ref<const Matrix> h() const
    {
        return m_h;
    }

    T& h(const int row, const int col)
    {
        return m_h(row, col);
    }

    T h(const int row, const int col) const
    {
        return m_h(row, col);
    }

    inline int size() const
    {
        return static_cast<int>(m_g.size());
    }

    HyperJet enlarge(const int left, const int right) const
    {
        if (left < 0) {
            throw std::runtime_error("Negative value for 'left'");
        }

        if (right < 0) {
            throw std::runtime_error("Negative value for 'right'");
        }

        HyperJet result(static_cast<int>(this->size() + left + right));

        result.m_f = m_f;

        if (!left) {
            result.m_g.segment(left, this->size()) = m_g;
            result.m_h.block(left, left, this->size(), this->size()) = m_h;
        } else {
            result.m_g.segment(left, this->size()) = m_g;
            result.m_h.block(left, left, this->size(), this->size()) = m_h;
        }

        return result;
    }

    HyperJet operator-() const
    {
        const auto f = -m_f;
        const auto g = -m_g;
        const auto h = -m_h;
        return HyperJet(f, g, h);
    }

    HyperJet operator+(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        const auto f = m_f + rhs.m_f;
        const auto g = m_g + rhs.m_g;
        const auto h = m_h + rhs.m_h;
        return HyperJet(f, g, h);
    }

    HyperJet operator+(const T rhs) const
    {
        const auto f = m_f + rhs;
        const auto g = m_g;
        const auto h = m_h;
        return HyperJet(f, g, h);
    }

    HyperJet operator-(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        const auto f = m_f - rhs.m_f;
        const auto g = m_g - rhs.m_g;
        const auto h = m_h - rhs.m_h;
        return HyperJet(f, g, h);
    }

    HyperJet operator-(const T rhs) const
    {
        const auto f = m_f - rhs;
        const auto g = m_g;
        const auto h = m_h;
        return HyperJet(f, g, h);
    }

    HyperJet operator*(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        const auto f = m_f * rhs.m_f;
        const auto g = m_f * rhs.m_g + rhs.m_f * m_g;
        const auto h = m_f * rhs.m_h + rhs.m_f * m_h + m_g.transpose() * rhs.m_g
            + rhs.m_g.transpose() * m_g;
        return HyperJet(f, g, h);
    }

    HyperJet operator*(const T rhs) const
    {
        const auto f = m_f * rhs;
        const auto g = rhs * m_g;
        const auto h = rhs * m_h;
        return HyperJet(f, g, h);
    }

    HyperJet operator/(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        const auto f = m_f / rhs.m_f;
        const auto g = m_g / rhs.m_f - m_f * rhs.m_g / (rhs.m_f * rhs.m_f);
        const auto h = (2 * m_f * rhs.m_g.transpose() * rhs.m_g +
            std::pow(rhs.m_f, 2) * m_h - rhs.m_f * (m_g.transpose() * rhs.m_g +
            rhs.m_g.transpose() * m_g + m_f * rhs.m_h)) / std::pow(rhs.m_f, 3);
        return HyperJet(f, g, h);
    }

    HyperJet operator/(const T rhs) const
    {
        const auto f = m_f / rhs;
        const auto g = m_g / rhs;
        const auto h = m_h / rhs;
        return HyperJet(f, g, h);
    }

    HyperJet& operator+=(const HyperJet& rhs)
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        m_f += rhs.m_f;
        m_g += rhs.m_g;
        m_h += rhs.m_h;
        return *this;
    }

    HyperJet& operator-=(const HyperJet& rhs)
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        m_f -= rhs.m_f;
        m_g -= rhs.m_g;
        m_h -= rhs.m_h;
        return *this;
    }

    HyperJet& operator*=(const HyperJet& rhs)
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        *this = *this * rhs;
        return *this;
    }

    HyperJet& operator*=(const T rhs)
    {
        m_f *= rhs;
        m_g *= rhs;
        m_h *= rhs;
        return *this;
    }

    HyperJet& operator/=(const HyperJet& rhs)
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        *this = *this / rhs;
        return *this;
    }

    HyperJet& operator/=(const T rhs)
    {
        m_f /= rhs;
        m_g /= rhs;
        m_h /= rhs;
        return *this;
    }

    friend HyperJet operator+(const T lhs, const HyperJet& rhs)
    {
        return rhs.operator+(lhs);
    }

    friend HyperJet operator-(const T lhs, const HyperJet& rhs)
    {
        const auto f = lhs - rhs.m_f;
        const auto g = -rhs.m_g;
        const auto h = -rhs.m_h;
        return HyperJet(f, g, h);
    }

    friend HyperJet operator*(const T lhs, const HyperJet& rhs)
    {
        return rhs.operator*(lhs);
    }

    friend HyperJet operator/(const T lhs, const HyperJet& rhs)
    {
        const auto f = lhs / rhs.m_f;
        const auto g = -lhs * rhs.m_g / (rhs.m_f * rhs.m_f);
        const auto h = (2 * lhs * rhs.m_g.transpose() * rhs.m_g - rhs.m_f * (lhs * rhs.m_h)) / std::pow(rhs.m_f, 3);
        return HyperJet(f, g, h);
    }

    inline HyperJet abs() const
    {
        return m_f < 0 ? -(*this) : *this;
    }

    inline HyperJet sqrt() const
    {
        const auto f = std::sqrt(m_f);
        const auto g = m_g / (2 * f);
        const auto h = (2 * m_h - m_g.transpose() * m_g / m_f) / (4 * std::sqrt(m_f));
        return HyperJet(f, g, h);
    }

    inline HyperJet cos() const
    {
        const auto f = std::cos(m_f);
        const auto g = -std::sin(m_f) * m_g;
        const auto h = -std::cos(m_f) * m_g.transpose() * m_g - std::sin(m_f) * m_h;
        return HyperJet(f, g, h);
    }

    inline HyperJet sin() const
    {
        const auto f = std::sin(m_f);
        const auto g = std::cos(m_f) * m_g;
        const auto h = -std::sin(m_f) * m_g.transpose() * m_g + std::cos(m_f) * m_h;
        return HyperJet(f, g, h);
    }

    inline HyperJet tan() const
    {
        const auto f = std::tan(m_f);
        const auto g = m_g * (f * f + 1);
        const auto h = (2 * m_g.transpose() * m_g * f + m_h)*(f * f + 1);
        return HyperJet(f, g, h);
    }

    inline HyperJet acos() const
    {
        const auto f = std::acos(m_f);
        const auto g = -m_g / std::sqrt(-m_f * m_f + 1);
        const auto h = -(m_g.transpose() * m_g * m_f / (-m_f * m_f + 1) + m_h) /
            std::sqrt(-m_f * m_f + 1);
        return HyperJet(f, g, h);
    }

    inline HyperJet asin() const
    {
        const auto f = std::asin(m_f);
        const auto g = m_g / std::sqrt(1 - m_f * m_f);
        const auto h = (m_f * m_g.transpose() * m_g - (m_f * m_f - 1) * m_h)
            / std::pow(1 - m_f * m_f, 1.5);
        return HyperJet(f, g, h);
    }

    inline HyperJet atan() const
    {
        const auto f = std::atan(m_f);
        const auto g = m_g / (m_f * m_f + 1);
        const auto h = (m_h - 2 * m_f * m_g.transpose() * m_g /
            (m_f * m_f + 1)) / (m_f * m_f + 1);
        return HyperJet(f, g, h);
    }

    static inline HyperJet atan2(const HyperJet& a, const HyperJet& b)
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (a.size() != b.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        const auto tmp = a.m_f * a.m_f + b.m_f * b.m_f;

        const auto f = std::atan2(a.m_f, b.m_f);
        const auto g = (a.m_g * b.m_f - a.m_f * b.m_g) / tmp;
        const auto h = (
            2 * (a.m_f * a.m_g + b.m_f * b.m_g).transpose() * (a.m_f * b.m_g) -
            2 * (a.m_f * a.m_g + b.m_f * b.m_g).transpose() * (b.m_f * a.m_g) +
            tmp * (b.m_f * a.m_h - a.m_f * b.m_h + b.m_g.transpose() * a.m_g -
            a.m_g.transpose() * b.m_g)) / std::pow(tmp, 2);
        return HyperJet(f, g, h);
    }

    template <typename U>
    inline HyperJet pow(const U b) const
    {
        const auto f = std::pow(m_f, b);
        const auto g = b * std::pow(m_f, b - U(1)) * m_g;
        const auto h = b * (b * m_g.transpose() * m_g + m_f * m_h -
            m_g.transpose() * m_g) * std::pow(m_f, b - U(2));
        return HyperJet(f, g, h);
    }

    bool operator==(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        return m_f == rhs.m_f;
    }

    bool operator!=(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        return m_f != rhs.m_f;
    }

    bool operator<(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        return m_f < rhs.m_f;
    }

    bool operator>(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        return m_f > rhs.m_f;
    }

    bool operator<=(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        return m_f <= rhs.m_f;
    }

    bool operator>=(const HyperJet& rhs) const
    {
    #if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
    #endif

        return m_f >= rhs.m_f;
    }

    bool operator==(const T rhs) const
    {
        return m_f == rhs;
    }

    bool operator!=(const T rhs) const
    {
        return m_f != rhs;
    }

    bool operator<(const T rhs) const
    {
        return m_f < rhs;
    }

    bool operator>(const T rhs) const
    {
        return m_f > rhs;
    }

    bool operator<=(const T rhs) const
    {
        return m_f <= rhs;
    }

    bool operator>=(const T rhs) const
    {
        return m_f >= rhs;
    }

    friend bool operator==(const T lhs, const HyperJet& rhs)
    {
        return rhs.operator==(lhs);
    }

    friend bool operator!=(const T lhs, const HyperJet& rhs)
    {
        return rhs.operator!=(lhs);
    }

    friend bool operator<(const T lhs, const HyperJet& rhs)
    {
        return rhs.operator>(lhs);
    }

    friend bool operator>(const T lhs, const HyperJet& rhs)
    {
        return rhs.operator<(lhs);
    }

    friend bool operator<=(const T lhs, const HyperJet& rhs)
    {
        return rhs.operator>=(lhs);
    }

    friend bool operator>=(const T lhs, const HyperJet& rhs)
    {
        return rhs.operator<=(lhs);
    }

    std::string to_string() const
    {
        return "HyperJet<" + std::to_string(m_f) + ">";
    }

public:     // python
    static void register_python(pybind11::module& m)
    {
        using namespace pybind11::literals;
        namespace py = pybind11;

        using Type = hyperjet::HyperJet<double>;

        const std::string name = "HyperJet";

        py::class_<Type>(m, name.c_str())
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
            .def_static("variable", &Type::variable, "value"_a, "size"_a, "index"_a)
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
            .def("__abs__", &Type::abs)
            .def("__len__", &Type::size)
            .def("__pow__", &Type::pow<double>)
            .def("__pow__", &Type::pow<int>)
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
            .def("enlarge", py::overload_cast<int, int>(&Type::enlarge,
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
};

// --- Support for std Operators

using std::abs;
using std::pow;
using std::sqrt;
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;

// --- Operators for Jet

template <typename T>
inline Jet<T> abs(const Jet<T>& a)
{
    return a.abs();
}

template <typename T>
inline Jet<T> pow(const Jet<T>& a, const int b)
{
    return a.pow(b);
}

template <typename T>
inline Jet<T> pow(const Jet<T>& a, const double b)
{
    return a.pow(b);
}

template <typename T>
inline Jet<T> sqrt(const Jet<T>& a)
{
    return a.sqrt();
}

template <typename T>
inline Jet<T> cos(const Jet<T>& a)
{
    return a.cos();
}

template <typename T>
inline Jet<T> sin(const Jet<T>& a)
{
    return a.sin();
}

template <typename T>
inline Jet<T> tan(const Jet<T>& a)
{
    return a.tan();
}

template <typename T>
inline Jet<T> acos(const Jet<T>& a)
{
    return a.acos();
}

template <typename T>
inline Jet<T> asin(const Jet<T>& a)
{
    return a.asin();
}

template <typename T>
inline Jet<T> atan(const Jet<T>& a)
{
    return a.atan();
}

template <typename T>
inline Jet<T> atan2(const Jet<T>& a, const Jet<T>& b)
{
    return Jet<T>::atan2(a, b);
}

// --- Operators for HyperJet

template <typename T>
inline HyperJet<T> abs(const HyperJet<T>& a)
{
    return a.abs();
}

template <typename T>
inline HyperJet<T> pow(const HyperJet<T>& a, const int b)
{
    return a.pow(b);
}

template <typename T>
inline HyperJet<T> pow(const HyperJet<T>& a, const double b)
{
    return a.pow(b);
}

template <typename T>
inline HyperJet<T> sqrt(const HyperJet<T>& a)
{
    return a.sqrt();
}

template <typename T>
inline HyperJet<T> cos(const HyperJet<T>& a)
{
    return a.cos();
}

template <typename T>
inline HyperJet<T> sin(const HyperJet<T>& a)
{
    return a.sin();
}

template <typename T>
inline HyperJet<T> tan(const HyperJet<T>& a)
{
    return a.tan();
}

template <typename T>
inline HyperJet<T> acos(const HyperJet<T>& a)
{
    return a.acos();
}

template <typename T>
inline HyperJet<T> asin(const HyperJet<T>& a)
{
    return a.asin();
}

template <typename T>
inline HyperJet<T> atan(const HyperJet<T>& a)
{
    return a.atan();
}

template <typename T>
inline HyperJet<T> atan2(const HyperJet<T>& a, const HyperJet<T>& b)
{
    return HyperJet<T>::atan2(a, b);
}

// --- Utility

template <int TDerivatives>
auto zero(const int size)
{
    static_assert(0 <= TDerivatives && TDerivatives <= 2, "Invalid Parameter");

    if constexpr(TDerivatives == 0) {
        return double{0};
    } else if constexpr(TDerivatives == 1) {
        return Jet{size};
    } else if constexpr(TDerivatives == 2) {
        return HyperJet{size};
    }
}

template <int TDerivatives>
auto constant(const double value, const int size)
{
    static_assert(0 <= TDerivatives && TDerivatives <= 2, "Invalid Parameter");

    if constexpr(TDerivatives == 0) {
        return double{value};
    } else if constexpr(TDerivatives == 1) {
        return Jet{value, size};
    } else if constexpr(TDerivatives == 2) {
        return HyperJet{value, size};
    }
}

template <int TDerivatives>
auto variable(const double value, const int size, const int index)
{
    static_assert(0 <= TDerivatives && TDerivatives <= 2, "Invalid Parameter");

    if constexpr(TDerivatives == 0) {
        return value;
    } else if constexpr(TDerivatives == 1) {
        return Jet<>::variable(value, size, index);
    } else if constexpr(TDerivatives == 2) {
        return HyperJet<>::variable(value, size, index);
    }
}

template <typename T>
auto explode(const T& value, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> h)
{
    using namespace hyperjet;

    if constexpr(std::is_same<T, double>()) {
        return value;
    } else if constexpr(std::is_same<T, Jet<double>>()) {
        if (g.size() >= 0) {
            // FIXME: check size
            g = value.g();
        }
        return value.f();
    } else if constexpr(std::is_same<T, HyperJet<double>>()) {
        if (g.size() >= 0) {
            // FIXME: check size
            g = value.g();
        }
        if (h.size() >= 0) {
            // FIXME: check size
            h = value.h();
        }
        return value.f();
    }
}

} // namespace hyperjet

namespace Eigen {

// --- Support for Jet

template<typename T>
struct NumTraits<hyperjet::Jet<T>> {
    using Real = hyperjet::Jet<T>;
    using NonInteger = hyperjet::Jet<T>;
    using Nested = hyperjet::Jet<T>;
    using Literal = hyperjet::Jet<T>;

    static Real dummy_precision()
    {
        return hyperjet::Jet<T>(1e-12, 0);
    }

    static inline Real epsilon()
    {
        return Real(std::numeric_limits<T>::epsilon());
    }

    static inline int digits10()
    {
        return NumTraits<T>::digits10();
    }

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 3,
        HasFloatingPoint = 1,
        RequireInitialization = 1
    };

    template<bool Vectorized>
    struct Div {
        enum {
#if defined(EIGEN_VECTORIZE_AVX)
            AVX = true,
#else
            AVX = false,
#endif
            Cost = 3
        };
    };

    static inline Real highest()
    {
        return Real(std::numeric_limits<T>::max());
    }

    static inline Real lowest()
    {
        return Real(-std::numeric_limits<T>::max());
    }
}; // struct NumTraits<hyperjet::Jet<T>>

template <typename BinaryOp, typename T>
struct ScalarBinaryOpTraits<hyperjet::Jet<T>, T, BinaryOp> {
    typedef hyperjet::Jet<T> ReturnType;
};

template <typename BinaryOp, typename T>
struct ScalarBinaryOpTraits<T, hyperjet::Jet<T>, BinaryOp> {
    typedef hyperjet::Jet<T> ReturnType;
};

// --- Support for HyperJet

template<typename T>
struct NumTraits<hyperjet::HyperJet<T>> {
    using Real = hyperjet::HyperJet<T>;
    using NonInteger = hyperjet::HyperJet<T>;
    using Nested = hyperjet::HyperJet<T>;
    using Literal = hyperjet::HyperJet<T>;

    static Real dummy_precision()
    {
        return hyperjet::HyperJet<T>(1e-12, 0);
    }

    static inline Real epsilon()
    {
        return Real(std::numeric_limits<T>::epsilon());
    }

    static inline int digits10()
    {
        return NumTraits<T>::digits10();
    }

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 3,
        HasFloatingPoint = 1,
        RequireInitialization = 1
    };

    template<bool Vectorized>
    struct Div {
        enum {
#if defined(EIGEN_VECTORIZE_AVX)
            AVX = true,
#else
            AVX = false,
#endif
            Cost = 3
        };
    };

    static inline Real highest()
    {
        return Real(std::numeric_limits<T>::max());
    }

    static inline Real lowest()
    {
        return Real(-std::numeric_limits<T>::max());
    }
}; // struct NumTraits<hyperjet::HyperJet<T>>

template <typename BinaryOp, typename T>
struct ScalarBinaryOpTraits<hyperjet::HyperJet<T>, T, BinaryOp> {
    typedef hyperjet::HyperJet<T> ReturnType;
};

template <typename BinaryOp, typename T>
struct ScalarBinaryOpTraits<T, hyperjet::HyperJet<T>, BinaryOp> {
    typedef hyperjet::HyperJet<T> ReturnType;
};

}  // namespace Eigen