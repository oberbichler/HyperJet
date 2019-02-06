#include <Eigen/Core>

#include <string>

namespace HyperJet {

template <typename T>
class HyperJet {
public:     // Types
    using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

private:    // Variables
    T m_f;
    Vector m_g;
    Matrix m_h;

public:     // Methods
    EIGEN_STRONG_INLINE
    HyperJet(
        const int size)
    : m_f(0)
    , m_g(Vector::Zero(size))
    , m_h(Matrix::Zero(size, size))
    { }

    template <typename Derived>
    EIGEN_STRONG_INLINE
    HyperJet(
        const T f,
        const Derived g)
    : m_f(f)
    , m_g(g)
    , m_h(Matrix::Zero(g.size(), g.size()))
    { }

    template <typename Derived1, typename Derived2>
    EIGEN_STRONG_INLINE
    HyperJet(
        const T f,
        const Derived1 g,
        const Derived2 h)
    : m_f(f)
    , m_g(g)
    , m_h(h)
    { }

    T&
    f()
    {
        return m_f;
    }

    Eigen::Ref<Vector>
    g()
    {
        return m_g;
    }

    Eigen::Ref<Matrix>
    h()
    {
        return m_h;
    }
    
    inline size_t
    size() const
    {
        return m_g.size();
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
        m_f += rhs.m_f;
        m_g += rhs.m_g;
        m_h += rhs.m_h;
        return *this;
    }

    HyperJet&
    operator-=(
        const HyperJet& rhs)
    {
        m_f -= rhs.m_f;
        m_g -= rhs.m_g;
        m_h -= rhs.m_h;
        return *this;
    }

    HyperJet&
    operator*=(
        const HyperJet& rhs)
    {
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
        const auto g = m_g / std::sqrt(-m_f * m_f + 1);
        const auto h = (m_g.transpose() * m_g * m_f / (-m_f * m_f + 1) + m_h) /
            std::sqrt(-m_f * m_f + 1);
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
        return m_f == rhs.m_f;
    }

    bool
    operator!=(
        const HyperJet& rhs) const
    {
        return m_f != rhs.m_f;
    }

    bool
    operator<(
        const HyperJet& rhs) const
    {
        return m_f < rhs.m_f;
    }

    bool
    operator>(
        const HyperJet& rhs) const
    {
        return m_f > rhs.m_f;
    }

    bool
    operator<=(
        const HyperJet& rhs) const
    {
        return m_f <= rhs.m_f;
    }

    bool
    operator>=(
        const HyperJet& rhs) const
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
    toString() const
    {
        return "HyperJet<" + std::to_string(m_f) + ">";
    }
};

} // namespace HyperJet