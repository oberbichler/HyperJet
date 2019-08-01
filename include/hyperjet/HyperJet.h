#pragma once

#include <Eigen/Core>

#include <string>

namespace hyperjet {

template <typename T>
class HyperJet {
public:     // Types
    using Scalar = T;
    using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

private:    // Variables
    T m_f;
    Vector m_g;
    Matrix m_h;

public:     // Methods
    EIGEN_STRONG_INLINE
    HyperJet()
    : m_f(0)
    , m_g(0)
    , m_h(0, 0)
    { }

    EIGEN_STRONG_INLINE
    HyperJet(
        const int size)
    : m_f(0)
    , m_g(Vector::Zero(size))
    , m_h(Matrix::Zero(size, size))
    { }

    EIGEN_STRONG_INLINE
    HyperJet(
        const T value,
        const int size)
    : m_f(value)
    , m_g(Vector::Zero(size))
    , m_h(Matrix::Zero(size, size))
    { }

    template <typename Derived>
    EIGEN_STRONG_INLINE
    HyperJet(
        const T f,
        const Eigen::DenseBase<Derived>& g)
    : m_f(f)
    , m_g(g)
    , m_h(Matrix::Zero(g.size(), g.size()))
    { }

    template <typename Derived1, typename Derived2>
    EIGEN_STRONG_INLINE
    HyperJet(
        const T f,
        const Eigen::DenseBase<Derived1>& g,
        const Eigen::DenseBase<Derived2>& h)
    : m_f(f)
    , m_g(g)
    , m_h(h)
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

    Eigen::Ref<Matrix>
    h()
    {
        return m_h;
    }

    Eigen::Ref<const Matrix>
    h() const
    {
        return m_h;
    }

    T&
    h(const int row, const int col)
    {
        return m_h(row, col);
    }

    T
    h(const int row, const int col) const
    {
        return m_h(row, col);
    }

    inline size_t
    size() const
    {
        return m_g.size();
    }

    HyperJet
    enlarge(
        const size_t left,
        const size_t right) const
    {
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

    HyperJet
    operator-() const
    {
        const auto f = -m_f;
        const auto g = -m_g;
        const auto h = -m_h;
        return HyperJet(f, g, h);
    }

    HyperJet
    operator+(
        const HyperJet& rhs) const
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

    HyperJet
    operator+(
        const T rhs) const
    {
        const auto f = m_f + rhs;
        const auto g = m_g;
        const auto h = m_h;
        return HyperJet(f, g, h);
    }

    HyperJet
    operator-(
        const HyperJet& rhs) const
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

    HyperJet
    operator-(
        const T rhs) const
    {
        const auto f = m_f - rhs;
        const auto g = m_g;
        const auto h = m_h;
        return HyperJet(f, g, h);
    }

    HyperJet
    operator*(
        const HyperJet& rhs) const
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

    HyperJet
    operator*(
        const T rhs) const
    {
        const auto f = m_f * rhs;
        const auto g = rhs * m_g;
        const auto h = rhs * m_h;
        return HyperJet(f, g, h);
    }

    HyperJet
    operator/(
        const HyperJet& rhs) const
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

    HyperJet
    operator/(
        const T rhs) const
    {
        const auto f = m_f / rhs;
        const auto g = m_g / rhs;
        const auto h = m_h / rhs;
        return HyperJet(f, g, h);
    }

    HyperJet&
    operator+=(
        const HyperJet& rhs)
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

    HyperJet&
    operator-=(
        const HyperJet& rhs)
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

    HyperJet&
    operator*=(
        const HyperJet& rhs)
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        *this = *this * rhs;
        return *this;
    }

    HyperJet&
    operator*=(
        const T rhs)
    {
        m_f *= rhs;
        m_g *= rhs;
        m_h *= rhs;
        return *this;
    }

    HyperJet&
    operator/=(
        const HyperJet& rhs)
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        *this = *this / rhs;
        return *this;
    }

    HyperJet&
    operator/=(
        const T rhs)
    {
        m_f /= rhs;
        m_g /= rhs;
        m_h /= rhs;
        return *this;
    }

    friend HyperJet
    operator+(
        const T lhs,
        const HyperJet& rhs)
    {
        return rhs.operator+(lhs);
    }

    friend HyperJet
    operator-(
        const T lhs,
        const HyperJet& rhs)
    {
        const auto f = lhs - rhs.m_f;
        const auto g = -rhs.m_g;
        const auto h = -rhs.m_h;
        return HyperJet(f, g, h);
    }

    friend HyperJet
    operator*(
        const T lhs,
        const HyperJet& rhs)
    {
        return rhs.operator*(lhs);
    }

    friend HyperJet
    operator/(
        const T lhs,
        const HyperJet& rhs)
    {
        const auto f = lhs / rhs.m_f;
        const auto g = -lhs * rhs.m_g / (rhs.m_f * rhs.m_f);
        const auto h = (2 * lhs * rhs.m_g.transpose() * rhs.m_g - rhs.m_f * (lhs * rhs.m_h)) / std::pow(rhs.m_f, 3);
        return HyperJet(f, g, h);
    }

    inline HyperJet
    abs() const
    {
        return m_f < 0 ? -(*this) : *this;
    }

    inline HyperJet
    sqrt() const
    {
        const auto f = std::sqrt(m_f);
        const auto g = m_g / (2 * f);
        const auto h = (2 * m_h - m_g.transpose() * m_g / m_f) / (4 * std::sqrt(m_f));
        return HyperJet(f, g, h);
    }

    inline HyperJet
    cos() const
    {
        const auto f = std::cos(m_f);
        const auto g = -std::sin(m_f) * m_g;
        const auto h = -std::cos(m_f) * m_g.transpose() * m_g - std::sin(m_f) * m_h;
        return HyperJet(f, g, h);
    }

    inline HyperJet
    sin() const
    {
        const auto f = std::sin(m_f);
        const auto g = std::cos(m_f) * m_g;
        const auto h = -std::sin(m_f) * m_g.transpose() * m_g + std::cos(m_f) * m_h;
        return HyperJet(f, g, h);
    }

    inline HyperJet
    tan() const
    {
        const auto f = std::tan(m_f);
        const auto g = m_g * (f * f + 1);
        const auto h = (2 * m_g.transpose() * m_g * f + m_h)*(f * f + 1);
        return HyperJet(f, g, h);
    }

    inline HyperJet
    acos() const
    {
        const auto f = std::acos(m_f);
        const auto g = -m_g / std::sqrt(-m_f * m_f + 1);
        const auto h = -(m_g.transpose() * m_g * m_f / (-m_f * m_f + 1) + m_h) /
            std::sqrt(-m_f * m_f + 1);
        return HyperJet(f, g, h);
    }

    inline HyperJet
    asin() const
    {
        const auto f = std::asin(m_f);
        const auto g = m_g / std::sqrt(1 - m_f * m_f);
        const auto h = (m_f * m_g.transpose() * m_g - (m_f * m_f - 1) * m_h)
            / std::pow(1 - m_f * m_f, 1.5);
        return HyperJet(f, g, h);
    }

    inline HyperJet
    atan() const
    {
        const auto f = std::atan(m_f);
        const auto g = m_g / (m_f * m_f + 1);
        const auto h = (m_h - 2 * m_f * m_g.transpose() * m_g /
            (m_f * m_f + 1)) / (m_f * m_f + 1);
        return HyperJet(f, g, h);
    }

    static inline HyperJet
    atan2(
        const HyperJet& a,
        const HyperJet& b)
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
    inline HyperJet
    pow(
        const U b) const
    {
        const auto f = std::pow(m_f, b);
        const auto g = b * std::pow(m_f, b - U(1)) * m_g;
        const auto h = b * (b * m_g.transpose() * m_g + m_f * m_h -
            m_g.transpose() * m_g) * std::pow(m_f, b - U(2));
        return HyperJet(f, g, h);
    }

    bool
    operator==(
        const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f == rhs.m_f;
    }

    bool
    operator!=(
        const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f != rhs.m_f;
    }

    bool
    operator<(
        const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f < rhs.m_f;
    }

    bool
    operator>(
        const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f > rhs.m_f;
    }

    bool
    operator<=(
        const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

        return m_f <= rhs.m_f;
    }

    bool
    operator>=(
        const HyperJet& rhs) const
    {
#if defined(HYPERJET_EXCEPTIONS)
        if (size() != rhs.size()) {
            throw new std::runtime_error("Dimensions do not match");
        }
#endif

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
        const HyperJet& rhs)
    {
        return rhs.operator==(lhs);
    }

    friend bool
    operator!=(
        const T lhs,
        const HyperJet& rhs)
    {
        return rhs.operator!=(lhs);
    }

    friend bool
    operator<(
        const T lhs,
        const HyperJet& rhs)
    {
        return rhs.operator>(lhs);
    }

    friend bool
    operator>(
        const T lhs,
        const HyperJet& rhs)
    {
        return rhs.operator<(lhs);
    }

    friend bool
    operator<=(
        const T lhs,
        const HyperJet& rhs)
    {
        return rhs.operator>=(lhs);
    }

    friend bool
    operator>=(
        const T lhs,
        const HyperJet& rhs)
    {
        return rhs.operator<=(lhs);
    }

    std::string
    to_string() const
    {
        return "HyperJet<" + std::to_string(m_f) + ">";
    }
};

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
inline HyperJet<T>
abs(
    const HyperJet<T>& a)
{
    return a.abs();
}

template <typename T>
inline HyperJet<T>
pow(
    const HyperJet<T>& a,
    const int b)
{
    return a.pow(b);
}

template <typename T>
inline HyperJet<T>
pow(
    const HyperJet<T>& a,
    const double b)
{
    return a.pow(b);
}

template <typename T>
inline HyperJet<T>
sqrt(
    const HyperJet<T>& a)
{
    return a.sqrt();
}

template <typename T>
inline HyperJet<T>
cos(
    const HyperJet<T>& a)
{
    return a.cos();
}

template <typename T>
inline HyperJet<T>
sin(
    const HyperJet<T>& a)
{
    return a.sin();
}

template <typename T>
inline HyperJet<T>
tan(
    const HyperJet<T>& a)
{
    return a.tan();
}

template <typename T>
inline HyperJet<T>
acos(
    const HyperJet<T>& a)
{
    return a.acos();
}

template <typename T>
inline HyperJet<T>
asin(
    const HyperJet<T>& a)
{
    return a.asin();
}

template <typename T>
inline HyperJet<T>
atan(
    const HyperJet<T>& a)
{
    return a.atan();
}

template <typename T>
inline HyperJet<T>
atan2(
    const HyperJet<T>& a,
    const HyperJet<T>& b)
{
    return HyperJet<T>::atan2(a, b);
}

} // namespace hyperjet

namespace Eigen {

template<typename T>
struct NumTraits<hyperjet::HyperJet<T>> {
    using Real = hyperjet::HyperJet<T>;
    using NonInteger = hyperjet::HyperJet<T>;
    using Nested = hyperjet::HyperJet<T>;
    using Literal = hyperjet::HyperJet<T>;

    static Real
    dummy_precision()
    {
        return hyperjet::HyperJet<T>(1e-12, 0);
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