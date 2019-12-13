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

#if defined(_MSC_VER)
#define HYPERJET_INLINE __forceinline
#else
#define HYPERJET_INLINE __attribute__((always_inline)) inline
#endif

using index = std::ptrdiff_t;

template <typename T>
HYPERJET_INLINE index length(const T& container)
{
    return static_cast<index>(container.size());
}

template <typename T = double, index TSize = Eigen::Dynamic>
class Jet {
public: // types
    using Scalar = T;
    using Vector = Eigen::Matrix<T, 1, TSize>;

private: // variables
    T m_f;
    Vector m_g;

public: // constructors
    HYPERJET_INLINE
    Jet()
        : m_f(0)
        , m_g(0)
    {
    }

    HYPERJET_INLINE
    Jet(const index size)
        : m_f(0)
        , m_g(Vector::Zero(size))
    {
    }

    HYPERJET_INLINE
    Jet(const T f, const index size)
        : m_f(f)
        , m_g(Vector::Zero(size))
    {
    }

    template <typename Derived>
    HYPERJET_INLINE
    Jet(const T f, const Eigen::DenseBase<Derived>& g)
        : m_f(f)
        , m_g(g)
    {
    }

public: // static methods
    static Jet<T, TSize> variable(const double value, const index size, const index index)
    {
        Jet<T, TSize> result(value, size);
        result.g(index) = 1;
        return result;
    }

    static std::vector<Jet<T, TSize>> variables(const std::vector<Jet<T, TSize>::Scalar> values)
    {
        const auto nb_variables = length(values);
        std::vector<Jet<T, TSize>> variables(nb_variables);
        for (index i = 0; i < nb_variables; i++) {
            variables[i] = variable(values[i], nb_variables, i);
        }
        return variables;
    }

public: // methods
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

    T& g(index index)
    {
        return m_g(index);
    }

    T g(const index index) const
    {
        return m_g(index);
    }

    HYPERJET_INLINE index size() const
    {
        return length(m_g);
    }

    Jet enlarge(const index left, const index right) const
    {
        if (left < 0) {
            throw std::runtime_error("Negative value for 'left'");
        }

        if (right < 0) {
            throw std::runtime_error("Negative value for 'right'");
        }

        Jet result(static_cast<index>(this->size() + left + right));

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

    HYPERJET_INLINE Jet abs() const
    {
        return m_f < 0 ? -(*this) : *this;
    }

    HYPERJET_INLINE Jet sqrt() const
    {
        const auto f = std::sqrt(m_f);
        const auto g = m_g / (2 * f);
        return Jet(f, g);
    }

    HYPERJET_INLINE Jet cos() const
    {
        const auto f = std::cos(m_f);
        const auto g = -std::sin(m_f) * m_g;
        return Jet(f, g);
    }

    HYPERJET_INLINE Jet sin() const
    {
        const auto f = std::sin(m_f);
        const auto g = std::cos(m_f) * m_g;
        return Jet(f, g);
    }

    HYPERJET_INLINE Jet tan() const
    {
        const auto f = std::tan(m_f);
        const auto g = m_g * (f * f + 1);
        return Jet(f, g);
    }

    HYPERJET_INLINE Jet acos() const
    {
        const auto f = std::acos(m_f);
        const auto g = -m_g / std::sqrt(-m_f * m_f + 1);
        return Jet(f, g);
    }

    HYPERJET_INLINE Jet asin() const
    {
        const auto f = std::asin(m_f);
        const auto g = m_g / std::sqrt(1 - m_f * m_f);
        return Jet(f, g);
    }

    HYPERJET_INLINE Jet atan() const
    {
        const auto f = std::atan(m_f);
        const auto g = m_g / (m_f * m_f + 1);
        return Jet(f, g);
    }

    static HYPERJET_INLINE Jet atan2(const Jet& a, const Jet& b)
    {
        const auto tmp = a.m_f * a.m_f + b.m_f * b.m_f;

        const auto f = std::atan2(a.m_f, b.m_f);
        const auto g = (a.m_g * b.m_f - a.m_f * b.m_g) / tmp;
        return Jet(f, g);
    }

    template <typename U>
    HYPERJET_INLINE Jet pow(const U b) const
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

public: // python
    static void register_python(pybind11::module& m)
    {
        using namespace pybind11::literals;
        namespace py = pybind11;

        using Type = hyperjet::Jet<double>;

        const std::string name = "Jet";

        py::class_<Type>(m, name.c_str())
            // constructors
            .def(py::init<index>(), "size"_a)
            .def(py::init<double, Type::Vector>(), "f"_a, "g"_a)
            // properties
            .def_property("f", py::overload_cast<>(&Type::f),
                [](Type& self, double value) {
                    self.f() = value;
                })
            .def_property("g", py::overload_cast<>(&Type::g),
                [](Type& self, Eigen::Ref<const Type::Vector> value) {
                    if (length(value) != length(self)) {
                        throw std::runtime_error("Invalid shape!");
                    }
                    self.g() = value;
                })
            // static methods
            .def_static("atan2", &Type::atan2)
            .def_static("variable", &Type::variable, "value"_a, "size"_a, "index"_a)
            .def_static("variables", &Type::variables, "values"_a)
            .def_static("variables", [](const std::vector<Type::Scalar> values, const index size, const index offset) {
                const auto nb_variables = length(values);
                std::vector<Type> variables(nb_variables);
                for (index i = 0; i < nb_variables; i++) {
                    Type::Vector g = Type::Vector::Zero(size);
                    g[offset + i] = 1;
                    variables[i] = Type(values[i], g);
                }
                return variables;
            },
                "values"_a, "size"_a, "offset"_a)
            // methods
            .def("__abs__", &Type::abs)
            .def("__len__", &Type::size)
            .def("__pow__", &Type::pow<double>)
            .def("__pow__", &Type::pow<index>)
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
            .def("enlarge", py::overload_cast<index, index>(&Type::enlarge, py::const_), "left"_a = 0, "right"_a = 0)
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
            .def(py::pickle([](const Type& self) { return py::make_tuple(self.f(), self.g()); }, [](py::tuple tuple) {
                    if (length(tuple) != 2) {
                        throw std::runtime_error("Invalid state!");
                    }

                    auto f = tuple[0].cast<double>();
                    auto g = tuple[1].cast<Type::Vector>();

                    return Type(f, g); }))
            .def("__copy__", [](const Type& self) { return self; })
            .def("__deepcopy__", [](const Type& self, py::dict& memo) { return self; }, "memodict"_a);
    }
}; // class Jet

template <typename T = double, index TSize = Eigen::Dynamic>
class HyperJet {
public: // types
    using Scalar = T;
    using Vector = Eigen::Matrix<T, 1, TSize>;
    using Matrix = Eigen::Matrix<T, TSize, TSize>;

private: // variables
    T m_f;
    Vector m_g;
    Matrix m_h;

public: // constructors
    HYPERJET_INLINE
    HyperJet()
        : m_f(0)
        , m_g(0)
        , m_h(0, 0)
    {
    }

    HYPERJET_INLINE
    HyperJet(const index size)
        : m_f(0)
        , m_g(Vector::Zero(size))
        , m_h(Matrix::Zero(size, size))
    {
    }

    HYPERJET_INLINE
    HyperJet(const T value, const index size)
        : m_f(value)
        , m_g(Vector::Zero(size))
        , m_h(Matrix::Zero(size, size))
    {
    }

    template <typename Derived>
    HYPERJET_INLINE
    HyperJet(const T f, const Eigen::DenseBase<Derived>& g)
        : m_f(f)
        , m_g(g)
        , m_h(Matrix::Zero(length(g), length(g)))
    {
    }

    template <typename Derived1, typename Derived2>
    HYPERJET_INLINE
    HyperJet(const T f, const Eigen::DenseBase<Derived1>& g, const Eigen::DenseBase<Derived2>& h)
        : m_f(f)
        , m_g(g)
        , m_h(h)
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (h.rows() != h.cols()) {
            throw new std::runtime_error("Hessian is not a square matrix");
        }

        if (length(g) != h.rows()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif
    }

public: // static methods
    static HyperJet<T, TSize> variable(const double value, const index size,
        const index index)
    {
        HyperJet<T, TSize> result(value, size);
        result.g(index) = 1;
        return result;
    }

    static std::vector<HyperJet<T, TSize>> variables(const std::vector<HyperJet<T, TSize>::Scalar> values)
    {
        const auto nb_variables = length(values);
        std::vector<HyperJet<T, TSize>> variables(nb_variables);
        for (index i = 0; i < nb_variables; i++) {
            variables[i] = variable(values[i], nb_variables, i);
        }
        return variables;
    }

public: // methods
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

    T& g(index index)
    {
        return m_g(index);
    }

    T g(const index index) const
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

    T& h(const index row, const index col)
    {
        return m_h(row, col);
    }

    T h(const index row, const index col) const
    {
        return m_h(row, col);
    }

    HYPERJET_INLINE index size() const
    {
        return static_cast<index>(length(m_g));
    }

    HyperJet enlarge(const index left, const index right) const
    {
        if (left < 0) {
            throw std::runtime_error("Negative value for 'left'");
        }

        if (right < 0) {
            throw std::runtime_error("Negative value for 'right'");
        }

        HyperJet result(static_cast<index>(this->size() + left + right));

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
        if (size() != length(rhs)) {
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
        if (size() != length(rhs)) {
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
        if (size() != length(rhs)) {
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
        if (size() != length(rhs)) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        const auto f = m_f / rhs.m_f;
        const auto g = m_g / rhs.m_f - m_f * rhs.m_g / (rhs.m_f * rhs.m_f);
        const auto h = (2 * m_f * rhs.m_g.transpose() * rhs.m_g + std::pow(rhs.m_f, 2) * m_h - rhs.m_f * (m_g.transpose() * rhs.m_g + rhs.m_g.transpose() * m_g + m_f * rhs.m_h)) / std::pow(rhs.m_f, 3);
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
        if (size() != length(rhs)) {
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
        if (size() != length(rhs)) {
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
        if (size() != length(rhs)) {
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
        if (size() != length(rhs)) {
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

    HYPERJET_INLINE HyperJet abs() const
    {
        return m_f < 0 ? -(*this) : *this;
    }

    HYPERJET_INLINE HyperJet sqrt() const
    {
        const auto f = std::sqrt(m_f);
        const auto g = m_g / (2 * f);
        const auto h = (2 * m_h - m_g.transpose() * m_g / m_f) / (4 * std::sqrt(m_f));
        return HyperJet(f, g, h);
    }

    HYPERJET_INLINE HyperJet cos() const
    {
        const auto f = std::cos(m_f);
        const auto g = -std::sin(m_f) * m_g;
        const auto h = -std::cos(m_f) * m_g.transpose() * m_g - std::sin(m_f) * m_h;
        return HyperJet(f, g, h);
    }

    HYPERJET_INLINE HyperJet sin() const
    {
        const auto f = std::sin(m_f);
        const auto g = std::cos(m_f) * m_g;
        const auto h = -std::sin(m_f) * m_g.transpose() * m_g + std::cos(m_f) * m_h;
        return HyperJet(f, g, h);
    }

    HYPERJET_INLINE HyperJet tan() const
    {
        const auto f = std::tan(m_f);
        const auto g = m_g * (f * f + 1);
        const auto h = (2 * m_g.transpose() * m_g * f + m_h) * (f * f + 1);
        return HyperJet(f, g, h);
    }

    HYPERJET_INLINE HyperJet acos() const
    {
        const auto f = std::acos(m_f);
        const auto g = -m_g / std::sqrt(-m_f * m_f + 1);
        const auto h = -(m_g.transpose() * m_g * m_f / (-m_f * m_f + 1) + m_h) / std::sqrt(-m_f * m_f + 1);
        return HyperJet(f, g, h);
    }

    HYPERJET_INLINE HyperJet asin() const
    {
        const auto f = std::asin(m_f);
        const auto g = m_g / std::sqrt(1 - m_f * m_f);
        const auto h = (m_f * m_g.transpose() * m_g - (m_f * m_f - 1) * m_h)
            / std::pow(1 - m_f * m_f, 1.5);
        return HyperJet(f, g, h);
    }

    HYPERJET_INLINE HyperJet atan() const
    {
        const auto f = std::atan(m_f);
        const auto g = m_g / (m_f * m_f + 1);
        const auto h = (m_h - 2 * m_f * m_g.transpose() * m_g / (m_f * m_f + 1)) / (m_f * m_f + 1);
        return HyperJet(f, g, h);
    }

    static HYPERJET_INLINE HyperJet atan2(const HyperJet& a, const HyperJet& b)
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (length(a) != length(b)) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        const auto tmp = a.m_f * a.m_f + b.m_f * b.m_f;

        const auto f = std::atan2(a.m_f, b.m_f);
        const auto g = (a.m_g * b.m_f - a.m_f * b.m_g) / tmp;
        const auto h = (2 * (a.m_f * a.m_g + b.m_f * b.m_g).transpose() * (a.m_f * b.m_g) - 2 * (a.m_f * a.m_g + b.m_f * b.m_g).transpose() * (b.m_f * a.m_g) + tmp * (b.m_f * a.m_h - a.m_f * b.m_h + b.m_g.transpose() * a.m_g - a.m_g.transpose() * b.m_g)) / std::pow(tmp, 2);
        return HyperJet(f, g, h);
    }

    template <typename U>
    HYPERJET_INLINE HyperJet pow(const U b) const
    {
        const auto f = std::pow(m_f, b);
        const auto g = b * std::pow(m_f, b - U(1)) * m_g;
        const auto h = b * (b * m_g.transpose() * m_g + m_f * m_h - m_g.transpose() * m_g) * std::pow(m_f, b - U(2));
        return HyperJet(f, g, h);
    }

    bool operator==(const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != length(rhs)) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f == rhs.m_f;
    }

    bool operator!=(const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != length(rhs)) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f != rhs.m_f;
    }

    bool operator<(const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != length(rhs)) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f < rhs.m_f;
    }

    bool operator>(const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != length(rhs)) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f > rhs.m_f;
    }

    bool operator<=(const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != length(rhs)) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f <= rhs.m_f;
    }

    bool operator>=(const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != length(rhs)) {
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

public: // python
    static void register_python(pybind11::module& m)
    {
        using namespace pybind11::literals;
        namespace py = pybind11;

        using Type = hyperjet::HyperJet<double>;

        const std::string name = "HyperJet";

        py::class_<Type>(m, name.c_str())
            // constructors
            .def(py::init<index>(), "size"_a)
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
                    if (length(value) != length(self)) {
                        throw std::runtime_error("Invalid shape!");
                    }
                    self.g() = value;
                })
            .def_property("h", py::overload_cast<>(&Type::h),
                [](Type& self, Eigen::Ref<const Type::Matrix> value) {
                    if (value.rows() != length(self) || value.cols() != length(self)) {
                        throw std::runtime_error("Invalid shape!");
                    }
                    self.h() = value;
                })
            // static methods
            .def_static("atan2", &Type::atan2)
            .def_static("variable", &Type::variable, "value"_a, "size"_a, "index"_a)
            .def_static("variables", &Type::variables, "values"_a)
            .def_static("variables", [](const std::vector<Type::Scalar> values, const index size, const index offset) {
                const auto nb_variables = length(values);
                std::vector<Type> variables(nb_variables);
                for (index i = 0; i < nb_variables; i++) {
                    Type::Vector g = Type::Vector::Zero(size);
                    g[offset + i] = 1;
                    variables[i] = Type(values[i], g);
                }
                return variables;
            },
                "values"_a, "size"_a, "offset"_a)
            // methods
            .def("__abs__", &Type::abs)
            .def("__len__", &Type::size)
            .def("__pow__", &Type::pow<double>)
            .def("__pow__", &Type::pow<index>)
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
            .def("enlarge", py::overload_cast<index, index>(&Type::enlarge, py::const_), "left"_a = 0, "right"_a = 0)
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
            .def(py::pickle([](const Type& self) { return py::make_tuple(self.f(), self.g(), self.h()); }, [](py::tuple tuple) {
                    if (length(tuple) != 3) {
                        throw std::runtime_error("Invalid state!");
                    }

                    auto f = tuple[0].cast<double>();
                    auto g = tuple[1].cast<Type::Vector>();
                    auto h = tuple[2].cast<Type::Matrix>();

                    return Type(f, g, h); }))
            .def("__copy__", [](const Type& self) { return self; })
            .def("__deepcopy__", [](const Type& self, py::dict& memo) { return self; }, "memodict"_a);
    }
};

// --- Support for std Operators

using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::pow;
using std::sin;
using std::sqrt;
using std::tan;

// --- Operators for Jet

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> abs(const Jet<T, TSize>& a)
{
    return a.abs();
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> pow(const Jet<T, TSize>& a, const index b)
{
    return a.pow(b);
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> pow(const Jet<T, TSize>& a, const double b)
{
    return a.pow(b);
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> sqrt(const Jet<T, TSize>& a)
{
    return a.sqrt();
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> cos(const Jet<T, TSize>& a)
{
    return a.cos();
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> sin(const Jet<T, TSize>& a)
{
    return a.sin();
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> tan(const Jet<T, TSize>& a)
{
    return a.tan();
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> acos(const Jet<T, TSize>& a)
{
    return a.acos();
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> asin(const Jet<T, TSize>& a)
{
    return a.asin();
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> atan(const Jet<T, TSize>& a)
{
    return a.atan();
}

template <typename T, index TSize>
HYPERJET_INLINE Jet<T, TSize> atan2(const Jet<T, TSize>& a, const Jet<T, TSize>& b)
{
    return Jet<T, TSize>::atan2(a, b);
}

// --- Operators for HyperJet

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> abs(const HyperJet<T, TSize>& a)
{
    return a.abs();
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> pow(const HyperJet<T, TSize>& a, const index b)
{
    return a.pow(b);
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> pow(const HyperJet<T, TSize>& a, const double b)
{
    return a.pow(b);
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> sqrt(const HyperJet<T, TSize>& a)
{
    return a.sqrt();
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> cos(const HyperJet<T, TSize>& a)
{
    return a.cos();
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> sin(const HyperJet<T, TSize>& a)
{
    return a.sin();
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> tan(const HyperJet<T, TSize>& a)
{
    return a.tan();
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> acos(const HyperJet<T, TSize>& a)
{
    return a.acos();
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> asin(const HyperJet<T, TSize>& a)
{
    return a.asin();
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> atan(const HyperJet<T, TSize>& a)
{
    return a.atan();
}

template <typename T, index TSize>
HYPERJET_INLINE HyperJet<T, TSize> atan2(const HyperJet<T, TSize>& a, const HyperJet<T, TSize>& b)
{
    return HyperJet<T, TSize>::atan2(a, b);
}

// --- Utility

template <index TDerivatives>
auto zero(const index size)
{
    static_assert(0 <= TDerivatives && TDerivatives <= 2, "Invalid Parameter");

    if constexpr (TDerivatives == 0) {
        return double{0};
    } else if constexpr (TDerivatives == 1) {
        return Jet{size};
    } else if constexpr (TDerivatives == 2) {
        return HyperJet{size};
    }
}

template <index TDerivatives>
auto constant(const double value, const index size)
{
    static_assert(0 <= TDerivatives && TDerivatives <= 2, "Invalid Parameter");

    if constexpr (TDerivatives == 0) {
        return double{value};
    } else if constexpr (TDerivatives == 1) {
        return Jet{value, size};
    } else if constexpr (TDerivatives == 2) {
        return HyperJet{value, size};
    }
}

template <index TDerivatives>
auto variable(const double value, const index size, const index index)
{
    static_assert(0 <= TDerivatives && TDerivatives <= 2, "Invalid Parameter");

    if constexpr (TDerivatives == 0) {
        return value;
    } else if constexpr (TDerivatives == 1) {
        return Jet<>::variable(value, size, index);
    } else if constexpr (TDerivatives == 2) {
        return HyperJet<>::variable(value, size, index);
    }
}

template <typename T>
auto explode(const T& value, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> h)
{
    using namespace hyperjet;

    if constexpr (std::is_same<T, double>()) {
        return value;
    } else if constexpr (std::is_same<T, Jet<double>>()) {
        if (length(g) >= 0) {
            // FIXME: check size
            g = value.g();
        }
        return value.f();
    } else if constexpr (std::is_same<T, HyperJet<double>>()) {
        if (length(g) >= 0) {
            // FIXME: check size
            g = value.g();
        }
        if (length(h) >= 0) {
            // FIXME: check size
            h = value.h();
        }
        return value.f();
    }
}

template <typename T>
auto add_explode(const T& value, Eigen::Ref<Eigen::VectorXd> g, Eigen::Ref<Eigen::MatrixXd> h)
{
    using namespace hyperjet;

    if constexpr(std::is_same<T, double>()) {
        return value;
    } else if constexpr(std::is_same<T, Jet<double>>()) {
        if (length(g) >= 0) {
            // FIXME: check size
            g += value.g();
        }
        return value.f();
    } else if constexpr(std::is_same<T, HyperJet<double>>()) {
        if (length(g) >= 0) {
            // FIXME: check size
            g += value.g();
        }
        if (length(h) >= 0) {
            // FIXME: check size
            h += value.h();
        }
        return value.f();
    }
}

} // namespace hyperjet