//     __  __                          __     __
//    / / / /_  ______  ___  _____    / /__  / /_
//   / /_/ / / / / __ \/ _ \/ ___/_  / / _ \/ __/
//  / __  / /_/ / /_/ /  __/ /  / /_/ /  __/ /_
// /_/ /_/\__, / .___/\___/_/   \____/\___/\__/
//       /____/_/
//
// Copyright (c) 2019-2020 Thomas Oberbichler

#pragma once

#include "fwd.h"

namespace hyperjet {

template <typename TScalar, index TSize>
class Jet {
public: // types
    using Type = Jet<TScalar, TSize>;
    using Scalar = TScalar;
    using Vector = Eigen::Matrix<Scalar, 1, TSize>;
    using Matrix = Eigen::Matrix<Scalar, TSize, TSize>;

private: // variables
    Scalar m_f;
    Vector m_g;

public: // constructors
    HYPERJET_INLINE Jet()
        : m_f(0)
        , m_g(init_size(TSize))
    {
        m_g.setZero();
    }

    HYPERJET_INLINE Jet(const TScalar f)
        : m_f(f)
        , m_g(init_size(TSize))
    {
        m_g.setZero();
    }

    HYPERJET_INLINE Jet(const Scalar f, Eigen::Ref<const Vector> g)
        : m_f(f)
        , m_g(g)
    {
    }

public: // methods
    HYPERJET_INLINE static Type empty()
    {
        assert(TSize != -1);
        return Type(Scalar(), Vector(TSize));
    }

    HYPERJET_INLINE static Type empty(const index size)
    {
        assert((TSize == -1 && size != -1) || (TSize != -1 && size == TSize));
        return Type(Scalar(), Vector(size));
    }

    HYPERJET_INLINE static Type zero()
    {
        static_assert(TSize != -1);
        return Type(Scalar(0), Vector::Zero(TSize), Matrix::Zero(TSize, TSize));
    }

    HYPERJET_INLINE static Type zero(const index size)
    {
        assert((TSize == -1 && size != -1) || (TSize != -1 && size == TSize));
        return Type(Scalar(0), Vector::Zero(size));
    }

    HYPERJET_INLINE static Type constant(const TScalar f)
    {
        assert(TSize != -1);
        return Type(Scalar(f), Vector::Zero(TSize), Matrix::Zero(TSize, TSize));
    }

    HYPERJET_INLINE static Type constant(const index size, const TScalar f)
    {
        assert((TSize == -1 && size != -1) || (TSize != -1 && size == TSize));
        return Type(Scalar(f), Vector::Zero(size));
    }

    HYPERJET_INLINE static Type variable(const index i, const TScalar f)
    {
        assert(TSize != -1);
        auto result = Type(Scalar(f), Vector::Zero(TSize));
        result.g(i) = Scalar(1);
        return result;
    }

    HYPERJET_INLINE static Type variable(const index size, const index i, const TScalar f)
    {
        assert((TSize == -1 && size != -1) || (TSize != -1 && size == TSize));
        auto result = Type(Scalar(f), Vector::Zero(size));
        result.g(i) = Scalar(1);
        return result;
    }

    HYPERJET_INLINE static std::vector<Type> variables(std::vector<Scalar> values)
    {
        const auto nb_variables = length(values);
        
        std::vector<Type> variables(nb_variables);
        
        for (index i = 0; i < nb_variables; i++) {
            variables[i] = Type::variable(values.size(), i, values[i]);
        }
        
        return variables;
    }

    HYPERJET_INLINE static std::vector<Type> variables(std::vector<Scalar> values, const index size, const index offset)
    {
        const auto nb_variables = length(values);
        
        std::vector<Type> variables(nb_variables);
        
        for (index i = 0; i < nb_variables; i++) {
            variables[i] = Type::variable(size, offset + i, values[i]);
        }
        
        return variables;
    }

    index size() const
    {
        if constexpr (TSize < 0) {
            return m_g.size();
        } else {
            return TSize;
        }
    }

    Scalar& f()
    {
        return m_f;
    }

    Scalar f() const
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

    TScalar g(const index i) const
    {
        assert(TSize == -1 || (0 <= i && i < TSize));
        return m_g(i);
    }

    TScalar& g(const index i)
    {
        assert(TSize == -1 || (0 <= i && i < TSize));
        return m_g(i);
    }

    Type enlarge(const index left, const index right) const
    {
        if (left < 0) {
            throw std::runtime_error("Negative value for 'left'");
        }

        if (right < 0) {
            throw std::runtime_error("Negative value for 'right'");
        }

        auto result = Type::zero(static_cast<index>(this->size() + left + right));

        result.m_f = m_f;

        if (!left) {
            result.g().segment(left, this->size()) = m_g;
        } else {
            result.g().segment(left, this->size()) = m_g;
        }

        return result;
    }

    auto backward(const std::vector<Jet<TScalar, Dynamic>>& xs) const
    {
        const index size = xs[0].size();

        auto result = Jet<TScalar, Dynamic>::zero(size);

        result.f() = f();
        
        backward_to(xs, result.g());

        return result;
    }

    void backward_to(const std::vector<Jet<TScalar, Dynamic>>& xs, Eigen::Ref<Eigen::VectorXd> g) const
    {
        const index size = g.size();

        for (index i = 0; i < size; i++) {
            for (index r = 0; r < this->size(); r++) {
                assert(xs[r].size() == size);
                g(i) += this->g(r) * xs[r].g(i);
            }
        }
    }

    auto forward(const std::vector<Jet<TScalar, Dynamic>>& xs) const
    {
        const index size = length(xs);

        auto result = Jet<TScalar, Dynamic>::zero(size);
        
        result.f() = f();

        forward_to(xs, result.g());

        return result;
    }

    void forward_to(const std::vector<Jet<TScalar, Dynamic>>& xs, Eigen::Ref<Eigen::VectorXd> g) const
    {
        const index size = g.size();

        for (index i = 0; i < size; i++) {
            g(i) = xs[i].g().dot(this->g());
        }
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

public: // operators
    friend std::ostream& operator<<(std::ostream& out, const Jet<TScalar, TSize>& value)
    {
        out << value.m_f << "j";
        return out;
    }

    // negative

    Type operator-() const
    {
        const auto f = -m_f;
        const auto g = -m_g;
        return Type(f, g);
    }

    // addition

    Type operator+(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != rhs.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        const auto f = m_f + rhs.m_f;
        const auto g = m_g + rhs.m_g;
        return Jet(f, g);
    }

    Type operator+(const Scalar rhs) const
    {
        const auto f = m_f + rhs;
        const auto g = m_g;
        return Type(f, g);
    }

    friend Type operator+(const Scalar lhs, const Type& rhs)
    {
        return rhs.operator+(lhs);
    }

    Type& operator+=(const Type& rhs)
    {
        if constexpr (check_size()) {
            if (size() != rhs.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        m_f += rhs.m_f;
        m_g += rhs.m_g;
        return *this;
    }

    Type& operator+=(const Scalar& rhs)
    {
        m_f += rhs.rhs;
        return *this;
    }

    // subtraction

    Type operator-(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != rhs.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        const auto f = m_f - rhs.m_f;
        const auto g = m_g - rhs.m_g;
        return Type(f, g);
    }

    Type operator-(const Scalar rhs) const
    {
        const auto f = m_f - rhs;
        const auto g = m_g;
        return Type(f, g);
    }

    friend Type operator-(const Scalar lhs, const Type& rhs)
    {
        return -rhs + lhs;
    }

    Type& operator-=(const Type& rhs)
    {
        if constexpr (check_size()) {
            if (size() != rhs.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        m_f -= rhs.m_f;
        m_g -= rhs.m_g;
        return *this;
    }

    Type& operator-=(const Scalar& rhs)
    {
        m_f -= rhs;
        return *this;
    }

    // multiplication

    Type operator*(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != rhs.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        const auto f = m_f * rhs.m_f;
        const auto g = m_f * rhs.m_g + rhs.m_f * m_g;
        return Type(f, g);
    }

    Type operator*(const Scalar rhs) const
    {
        const auto f = m_f * rhs;
        const auto g = m_g * rhs;
        return Type(f, g);
    }

    friend Type operator*(const Scalar lhs, const Type& rhs)
    {
        return rhs.operator*(lhs);
    }

    Type& operator*=(const Type& rhs)
    {
        if constexpr (check_size()) {
            if (size() != rhs.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        *this = *this * rhs;
        return *this;
    }

    Type& operator*=(const Scalar& rhs)
    {
        m_f *= rhs;
        m_g *= rhs;
        return *this;
    }

    // division

    Type operator/(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != rhs.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }
        using std::pow;

        const auto f = m_f / rhs.m_f;
        const auto g = m_g / rhs.m_f - m_f * rhs.m_g / (rhs.m_f * rhs.m_f);
        return Type(f, g);
    }

    Type operator/(const Scalar rhs) const
    {
        const auto f = m_f / rhs;
        const auto g = m_g / rhs;
        return Type(f, g);
    }

    friend Type operator/(const Scalar lhs, const Type& rhs)
    {
        using std::pow;

        const auto f = lhs / rhs.m_f;
        const auto g = -lhs * rhs.m_g / (rhs.m_f * rhs.m_f);
        return Type(f, g);
    }

    Type& operator/=(const Type& rhs)
    {
        if constexpr (check_size()) {
            if (size() != rhs.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        *this = *this / rhs;
        return *this;
    }

    Type& operator/=(const Scalar& rhs)
    {
        m_f /= rhs;
        m_g /= rhs;
        return *this;
    }

    // abs
    
    HYPERJET_INLINE Type abs() const
    {
        return m_f < 0 ? -(*this) : *this;
    }

    // pow

    template <typename TExponent>
    HYPERJET_INLINE Type pow(const TExponent b) const
    {
        using std::pow;

        const auto f = pow(m_f, b);
        const auto g = b * pow(m_f, b - TExponent(1)) * m_g;
        return Type(f, g);
    }

    // sqrt

    HYPERJET_INLINE Type sqrt() const
    {
        using std::sqrt;

        const auto f = sqrt(m_f);
        const auto g = m_g / (2 * f);
        return Type(f, g);
    }

    // cos

    HYPERJET_INLINE Type cos() const
    {
        using std::cos;
        using std::sin;

        const auto f = cos(m_f);
        const auto g = -sin(m_f) * m_g;
        return Type(f, g);
    }

    // sin

    HYPERJET_INLINE Type sin() const
    {
        using std::cos;
        using std::sin;

        const auto f = sin(m_f);
        const auto g = cos(m_f) * m_g;
        return Type(f, g);
    }

    // tan

    HYPERJET_INLINE Type tan() const
    {
        using std::tan;

        const auto f = tan(m_f);
        const auto g = m_g * (f * f + 1);
        return Type(f, g);
    }

    // acos

    HYPERJET_INLINE Type acos() const
    {
        using std::acos;
        using std::sqrt;

        const auto f = acos(m_f);
        const auto g = -m_g / sqrt(-m_f * m_f + 1);
        return Type(f, g);
    }

    // asin

    HYPERJET_INLINE Type asin() const
    {
        using std::asin;
        using std::pow;
        using std::sqrt;

        const auto f = asin(m_f);
        const auto g = m_g / sqrt(1 - m_f * m_f);
        return Type(f, g);
    }

    // atan

    HYPERJET_INLINE Type atan() const
    {
        using std::atan;

        const auto f = atan(m_f);
        const auto g = m_g / (m_f * m_f + 1);
        return Type(f, g);
    }

    // atan2

    static HYPERJET_INLINE Type atan2(const Type& a, const Type& b)
    {
        if constexpr (check_size()) {
            if (a.size() != b.size()) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }
        using std::atan2;
        using std::pow;

        const auto tmp = a.m_f * a.m_f + b.m_f * b.m_f;

        const auto f = atan2(a.m_f, b.m_f);
        const auto g = (a.m_g * b.m_f - a.m_f * b.m_g) / tmp;
        return Type(f, g);
    }

    // comparison

    bool operator==(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != length(rhs)) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        return m_f == rhs.m_f;
    }

    bool operator!=(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != length(rhs)) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        return m_f != rhs.m_f;
    }

    bool operator<(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != length(rhs)) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        return m_f < rhs.m_f;
    }

    bool operator>(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != length(rhs)) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        return m_f > rhs.m_f;
    }

    bool operator<=(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != length(rhs)) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        return m_f <= rhs.m_f;
    }

    bool operator>=(const Type& rhs) const
    {
        if constexpr (check_size()) {
            if (size() != length(rhs)) {
                throw new std::runtime_error("Dimensions do not match");
            }
        }

        return m_f >= rhs.m_f;
    }

    bool operator==(const Scalar rhs) const
    {
        return m_f == rhs;
    }

    bool operator!=(const Scalar rhs) const
    {
        return m_f != rhs;
    }

    bool operator<(const Scalar rhs) const
    {
        return m_f < rhs;
    }

    bool operator>(const Scalar rhs) const
    {
        return m_f > rhs;
    }

    bool operator<=(const Scalar rhs) const
    {
        return m_f <= rhs;
    }

    bool operator>=(const Scalar rhs) const
    {
        return m_f >= rhs;
    }

    friend bool operator==(const Scalar lhs, const Type& rhs)
    {
        return rhs.operator==(lhs);
    }

    friend bool operator!=(const Scalar lhs, const Type& rhs)
    {
        return rhs.operator!=(lhs);
    }

    friend bool operator<(const Scalar lhs, const Type& rhs)
    {
        return rhs.operator>(lhs);
    }

    friend bool operator>(const Scalar lhs, const Type& rhs)
    {
        return rhs.operator<(lhs);
    }

    friend bool operator<=(const Scalar lhs, const Type& rhs)
    {
        return rhs.operator>=(lhs);
    }

    friend bool operator>=(const Scalar lhs, const Type& rhs)
    {
        return rhs.operator<=(lhs);
    }

public: // python
    template <typename TModule>
    static void register_python(TModule& m, const std::string name)
    {
        using namespace pybind11::literals;
        namespace py = pybind11;

        auto python_class = py::class_<Type>(m, name.c_str())
            // constructors
            .def(py::init<Scalar>(), "f"_a)
            .def(py::init<Scalar, Eigen::Ref<const Vector>>(), "f"_a, "g"_a)
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
            .def_static("empty", py::overload_cast<index>(&Type::empty), "size"_a)
            .def_static("zero", py::overload_cast<index>(&Type::zero), "size"_a)
            .def_static("variable", py::overload_cast<index, index, Scalar>(&Type::variable), "size"_a, "index"_a, "f"_a)
            .def_static("variables", py::overload_cast<std::vector<Scalar>>(&Type::variables), "values"_a)
            .def_static("variables", py::overload_cast<std::vector<Scalar>, index, index>(&Type::variables), "values"_a, "size"_a, "offset"_a)
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
            .def("backward", &Type::backward, "xs"_a)
            .def("backward_to", &Type::backward_to, "xs"_a, "g"_a)
            .def("cos", &Type::cos)
            .def("enlarge", py::overload_cast<index, index>(&Type::enlarge, py::const_), "left"_a = 0, "right"_a = 0)
            .def("forward", &Type::forward, "xs"_a)
            .def("forward_to", &Type::forward_to, "xs"_a, "g"_a)
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
};

// abs

using std::abs;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> abs(const Jet<TScalar, TSize>& a)
{
    return a.abs();
}

// pow

using std::pow;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> pow(const Jet<TScalar, TSize>& a, const index b)
{
    return a.pow(b);
}

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> pow(const Jet<TScalar, TSize>& a, const TScalar b)
{
    return a.pow(b);
}

// sqrt

using std::sqrt;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> sqrt(const Jet<TScalar, TSize>& a)
{
    return a.sqrt();
}

// cos

using std::cos;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> cos(const Jet<TScalar, TSize>& a)
{
    return a.cos();
}

// sin

using std::sin;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> sin(const Jet<TScalar, TSize>& a)
{
    return a.sin();
}

// tan

using std::tan;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> tan(const Jet<TScalar, TSize>& a)
{
    return a.tan();
}

// acos

using std::acos;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> acos(const Jet<TScalar, TSize>& a)
{
    return a.acos();
}

// asin

using std::asin;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> asin(const Jet<TScalar, TSize>& a)
{
    return a.asin();
}

// atan

using std::atan;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> atan(const Jet<TScalar, TSize>& a)
{
    return a.atan();
}

// atan2

using std::atan2;

template <typename TScalar, index TSize>
HYPERJET_INLINE Jet<TScalar, TSize> atan2(const Jet<TScalar, TSize>& a, const Jet<TScalar, TSize>& b)
{
    return Jet<TScalar, TSize>::atan2(a, b);
}

} // namespace hyperjet
