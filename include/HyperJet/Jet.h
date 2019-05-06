#pragma once

#include <Eigen/Core>

#include <string>

namespace HyperJet {

template <typename T>
class Jet {
public:     // Types
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
    
    inline size_t
    size() const
    {
        return m_g.size();
    }
    
    Jet
    enlarge(
        const size_t size,
        const bool left) const
    {
        Jet result(static_cast<int>(this->size() + size));

        result.m_f = m_f;
        
        if (!left) {
            result.m_g.head(this->size()) = m_g;
        } else {
            result.m_g.tail(this->size()) = m_g;
        }

        return result;
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
        const auto g = m_g / std::sqrt(-m_f * m_f + 1);
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
    toString() const
    {
        return "Jet<" + std::to_string(m_f) + ">";
    }
};

} // namespace HyperJet