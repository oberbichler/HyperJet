#pragma once

#include <Eigen/Core>

#include <string>

namespace hyperjet {

template <typename T = double>
class Jet {
public:     // Types
    using Scalar = T;
    using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;

private:    // Variables
    T m_f;
    Vector m_g;

public:     // Methods
    EIGEN_STRONG_INLINE
    Jet()
    : m_f(0)
    , m_g(0)
    { }

    EIGEN_STRONG_INLINE
    Jet(
        const int size)
    : m_f(0)
    , m_g(Vector::Zero(size))
    { }

    EIGEN_STRONG_INLINE
    Jet(
        const T f,
        const int size)
    : m_f(f)
    , m_g(Vector::Zero(size))
    { }

    template <typename Derived>
    EIGEN_STRONG_INLINE
    Jet(
        const T f,
        const Eigen::DenseBase<Derived>& g)
    : m_f(f)
    , m_g(g)
    { }

    static Jet<T>
    variable(const double value, const int size, const int index)
    {
        Jet<T> result(value, size);
        result.g(index) = 1;
        return result;
    }

    T&
    f()
    {
        return m_f;
    }

    T
    f() const
    {
        return m_f;
    }

    Eigen::Ref<Vector>
    g()
    {
        return m_g;
    }

    Eigen::Ref<const Vector>
    g() const
    {
        return m_g;
    }

    T&
    g(int index)
    {
        return m_g(index);
    }

    T
    g(const int index) const
    {
        return m_g(index);
    }


    inline size_t
    size() const
    {
        return m_g.size();
    }

    Jet
    enlarge(
        const size_t left,
        const size_t right) const
    {
        Jet result(static_cast<int>(this->size() + left + right));

        result.m_f = m_f;

        if (!left) {
            result.m_g.segment(left, this->size()) = m_g;
        } else {
            result.m_g.segment(left, this->size()) = m_g;
        }

        return result;
    }

    Jet
    operator-() const
    {
        const auto f = -m_f;
        const auto g = -m_g;
        return Jet(f, g);
    }

    Jet
    operator+(
        const Jet& rhs) const
    {
        const auto f = m_f + rhs.m_f;
        const auto g = m_g + rhs.m_g;
        return Jet(f, g);
    }

    Jet
    operator+(
        const T rhs) const
    {
        const auto f = m_f + rhs;
        const auto g = m_g;
        return Jet(f, g);
    }

    Jet
    operator-(
        const Jet& rhs) const
    {
        const auto f = m_f - rhs.m_f;
        const auto g = m_g - rhs.m_g;
        return Jet(f, g);
    }

    Jet
    operator-(
        const T rhs) const
    {
        const auto f = m_f - rhs;
        const auto g = m_g;
        return Jet(f, g);
    }

    Jet
    operator*(
        const Jet& rhs) const
    {
        const auto f = m_f * rhs.m_f;
        const auto g = m_f * rhs.m_g + rhs.m_f * m_g;
        return Jet(f, g);
    }

    Jet
    operator*(
        const T rhs) const
    {
        const auto f = m_f * rhs;
        const auto g = rhs * m_g;
        return Jet(f, g);
    }

    Jet
    operator/(
        const Jet& rhs) const
    {
        const auto f = m_f / rhs.m_f;
        const auto g = m_g / rhs.m_f - m_f * rhs.m_g / (rhs.m_f * rhs.m_f);
        return Jet(f, g);
    }

    Jet
    operator/(
        const T rhs) const
    {
        const auto f = m_f / rhs;
        const auto g = m_g / rhs;
        return Jet(f, g);
    }

    Jet&
    operator+=(
        const Jet& rhs)
    {
        m_f += rhs.m_f;
        m_g += rhs.m_g;
        return *this;
    }

    Jet&
    operator-=(
        const Jet& rhs)
    {
        m_f -= rhs.m_f;
        m_g -= rhs.m_g;
        return *this;
    }

    Jet&
    operator*=(
        const Jet& rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    Jet&
    operator*=(
        const T rhs)
    {
        m_f *= rhs;
        m_g *= rhs;
        return *this;
    }

    Jet&
    operator/=(
        const Jet& rhs)
    {
        *this = *this / rhs;
        return *this;
    }

    Jet&
    operator/=(
        const T rhs)
    {
        m_f /= rhs;
        m_g /= rhs;
        return *this;
    }

    friend Jet
    operator+(
        const T lhs,
        const Jet& rhs)
    {
        return rhs.operator+(lhs);
    }

    friend Jet
    operator-(
        const T lhs,
        const Jet& rhs)
    {
        const auto f = lhs - rhs.m_f;
        const auto g = -rhs.m_g;
        return Jet(f, g);
    }

    friend Jet
    operator*(
        const T lhs,
        const Jet& rhs)
    {
        return rhs.operator*(lhs);
    }

    friend Jet
    operator/(
        const T lhs,
        const Jet& rhs)
    {
        const auto f = lhs / rhs.m_f;
        const auto g = -lhs * rhs.m_g / (rhs.m_f * rhs.m_f);
        return Jet(f, g);
    }

    inline Jet
    abs() const
    {
        return m_f < 0 ? -(*this) : *this;
    }

    inline Jet
    sqrt() const
    {
        const auto f = std::sqrt(m_f);
        const auto g = m_g / (2 * f);
        return Jet(f, g);
    }

    inline Jet
    cos() const
    {
        const auto f = std::cos(m_f);
        const auto g = -std::sin(m_f) * m_g;
        return Jet(f, g);
    }

    inline Jet
    sin() const
    {
        const auto f = std::sin(m_f);
        const auto g = std::cos(m_f) * m_g;
        return Jet(f, g);
    }

    inline Jet
    tan() const
    {
        const auto f = std::tan(m_f);
        const auto g = m_g * (f * f + 1);
        return Jet(f, g);
    }

    inline Jet
    acos() const
    {
        const auto f = std::acos(m_f);
        const auto g = -m_g / std::sqrt(-m_f * m_f + 1);
        return Jet(f, g);
    }

    inline Jet
    asin() const
    {
        const auto f = std::asin(m_f);
        const auto g = m_g / std::sqrt(1 - m_f * m_f);
        return Jet(f, g);
    }

    inline Jet
    atan() const
    {
        const auto f = std::atan(m_f);
        const auto g = m_g / (m_f * m_f + 1);
        return Jet(f, g);
    }

    static inline Jet
    atan2(
        const Jet& a,
        const Jet& b)
    {
        const auto tmp = a.m_f * a.m_f + b.m_f * b.m_f;

        const auto f = std::atan2(a.m_f, b.m_f);
        const auto g = (a.m_g * b.m_f - a.m_f * b.m_g) / tmp;
        return Jet(f, g);
    }

    template <typename U>
    inline Jet
    pow(
        const U b) const
    {
        const auto f = std::pow(m_f, b);
        const auto g = b * std::pow(m_f, b - U(1)) * m_g;
        return Jet(f, g);
    }

    bool
    operator==(
        const Jet& rhs) const
    {
        return m_f == rhs.m_f;
    }

    bool
    operator!=(
        const Jet& rhs) const
    {
        return m_f != rhs.m_f;
    }

    bool
    operator<(
        const Jet& rhs) const
    {
        return m_f < rhs.m_f;
    }

    bool
    operator>(
        const Jet& rhs) const
    {
        return m_f > rhs.m_f;
    }

    bool
    operator<=(
        const Jet& rhs) const
    {
        return m_f <= rhs.m_f;
    }

    bool
    operator>=(
        const Jet& rhs) const
    {
        return m_f >= rhs.m_f;
    }

    bool
    operator==(
        const T rhs) const
    {
        return m_f == rhs;
    }

    bool
    operator!=(
        const T rhs) const
    {
        return m_f != rhs;
    }

    bool
    operator<(
        const T rhs) const
    {
        return m_f < rhs;
    }

    bool
    operator>(
        const T rhs) const
    {
        return m_f > rhs;
    }

    bool
    operator<=(
        const T rhs) const
    {
        return m_f <= rhs;
    }

    bool
    operator>=(
        const T rhs) const
    {
        return m_f >= rhs;
    }

    friend bool
    operator==(
        const T lhs,
        const Jet& rhs)
    {
        return rhs.operator==(lhs);
    }

    friend bool
    operator!=(
        const T lhs,
        const Jet& rhs)
    {
        return rhs.operator!=(lhs);
    }

    friend bool
    operator<(
        const T lhs,
        const Jet& rhs)
    {
        return rhs.operator>(lhs);
    }

    friend bool
    operator>(
        const T lhs,
        const Jet& rhs)
    {
        return rhs.operator<(lhs);
    }

    friend bool
    operator<=(
        const T lhs,
        const Jet& rhs)
    {
        return rhs.operator>=(lhs);
    }

    friend bool
    operator>=(
        const T lhs,
        const Jet& rhs)
    {
        return rhs.operator<=(lhs);
    }

    std::string
    to_string() const
    {
        return "Jet<" + std::to_string(m_f) + ">";
    }
}; // class Jet

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

template <typename T>
inline Jet<T>
abs(
    const Jet<T>& a)
{
    return a.abs();
}

template <typename T>
inline Jet<T>
pow(
    const Jet<T>& a,
    const int b)
{
    return a.pow(b);
}

template <typename T>
inline Jet<T>
pow(
    const Jet<T>& a,
    const double b)
{
    return a.pow(b);
}

template <typename T>
inline Jet<T>
sqrt(
    const Jet<T>& a)
{
    return a.sqrt();
}

template <typename T>
inline Jet<T>
cos(
    const Jet<T>& a)
{
    return a.cos();
}

template <typename T>
inline Jet<T>
sin(
    const Jet<T>& a)
{
    return a.sin();
}

template <typename T>
inline Jet<T>
tan(
    const Jet<T>& a)
{
    return a.tan();
}

template <typename T>
inline Jet<T>
acos(
    const Jet<T>& a)
{
    return a.acos();
}

template <typename T>
inline Jet<T>
asin(
    const Jet<T>& a)
{
    return a.asin();
}

template <typename T>
inline Jet<T>
atan(
    const Jet<T>& a)
{
    return a.atan();
}

template <typename T>
inline Jet<T>
atan2(
    const Jet<T>& a,
    const Jet<T>& b)
{
    return Jet<T>::atan2(a, b);
}

} // namespace hyperjet

namespace Eigen {

template<typename T>
struct NumTraits<hyperjet::Jet<T>> {
    using Real = hyperjet::Jet<T>;
    using NonInteger = hyperjet::Jet<T>;
    using Nested = hyperjet::Jet<T>;
    using Literal = hyperjet::Jet<T>;

    static Real
    dummy_precision()
    {
        return hyperjet::Jet<T>(1e-12, 0);
    }

    static inline Real
    epsilon()
    {
        return Real(std::numeric_limits<T>::epsilon());
    }

    static inline int
    digits10()
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

    static inline Real
    highest()
    {
        return Real(std::numeric_limits<T>::max());
    }

    static inline Real
    lowest()
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

}  // namespace Eigen