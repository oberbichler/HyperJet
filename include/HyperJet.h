#include <string>

template <typename T>
class HyperJet {
public:
    using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

private:
    T m_v;
    Vector m_g;
    Matrix m_j;

public:
    HyperJet(
        const int size)
    : m_v(0)
    , m_g(Vector::Zero(size))
    , m_j(Matrix::Zero(size, size))
    { }

    template <typename Derived>
    EIGEN_STRONG_INLINE
    HyperJet(
        const T v,
        const Derived g)
    : m_v(v)
    , m_g(g)
    , m_j(Matrix::Zero(g.size(), g.size()))
    { }

    template <typename Derived1, typename Derived2>
    EIGEN_STRONG_INLINE
    HyperJet(
        const T v,
        const Derived1 g,
        const Derived2 j)
    : m_v(v)
    , m_g(g)
    , m_j(j)
    { }

    T&
    v()
    {
        return m_v;
    }

    T
    vc() const
    {
        return m_v;
    }

    Eigen::Ref<Vector>
    g()
    {
        return m_g;
    }

    Eigen::Ref<Matrix>
    j()
    {
        return m_j;
    }
    
    HyperJet
    operator-() const
    {
        const auto v = -m_v;
        const auto g = -m_g;
        const auto j = -m_j;
        return HyperJet(v, g, j);
    }

    HyperJet
    operator+(
        const HyperJet& rhs) const
    {
        const auto v = m_v + rhs.m_v;
        const auto g = m_g + rhs.m_g;
        const auto j = m_j + rhs.m_j;
        return HyperJet(v, g, j);
    }

    HyperJet
    operator+(
        const T rhs) const
    {
        const auto v = m_v + rhs;
        const auto g = m_g;
        const auto j = m_j;
        return HyperJet(v, g, j);
    }
    
    HyperJet
    operator-(
        const HyperJet &rhs) const
    {
        const auto v = m_v - rhs.m_v;
        const auto g = m_g - rhs.m_g;
        const auto j = m_j - rhs.m_j;
        return HyperJet(v, g, j);
    }

    HyperJet
    operator-(
        const T rhs) const
    {
        const auto v = m_v - rhs;
        const auto g = m_g;
        const auto j = m_j;
        return HyperJet(v, g, j);
    }

    HyperJet
    operator*(
        const HyperJet &rhs) const
    {
        const auto v = m_v * rhs.m_v;
        const auto g = m_v * rhs.m_g + rhs.m_v * m_g;
        const auto j = m_v * rhs.m_j + rhs.m_v * m_j + m_g.transpose() * rhs.m_g
            + rhs.m_g.transpose() * m_g;
        return HyperJet(v, g, j);
    }

    HyperJet
    operator*(
        const T rhs) const
    {
        const auto v = m_v * rhs;
        const auto g = rhs * m_g;
        const auto j = rhs * m_j;
        return HyperJet(v, g, j);
    }

    HyperJet
    operator/(
        const HyperJet &rhs) const
    {
        const auto v = m_v / rhs.m_v;
        const auto g = m_g / rhs.m_v - m_v * rhs.m_g / (rhs.m_v * rhs.m_v);
        const auto j = (2 * m_v * rhs.m_g.transpose() * rhs.m_g +
            std::pow(rhs.m_v, 2) * m_j - rhs.m_v * (m_g.transpose() * rhs.m_g +
            rhs.m_g.transpose() * m_g + m_v * rhs.m_j)) / std::pow(rhs.m_v, 3);
        return HyperJet(v, g, j);
    }

    HyperJet
    operator/(
        const T rhs) const
    {
        const auto v = m_v / rhs;
        const auto g = m_g / rhs;
        const auto j = m_j / rhs;
        return HyperJet(v, g, j);
    }

    HyperJet&
    operator+=(
        const HyperJet &rhs)
    {
        m_v += rhs.m_v;
        m_g += rhs.m_g;
        m_j += rhs.m_j;
        return *this;
    }

    HyperJet&
    operator-=(
        const HyperJet &rhs)
    {
        m_v -= rhs.m_v;
        m_g -= rhs.m_g;
        m_j -= rhs.m_j;
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
    operator*=
    (
        const T rhs)
    {
        m_v *= rhs;
        m_g *= rhs;
        m_j *= rhs;
        return *this;
    }

    HyperJet&
    operator/=
    (
        const HyperJet& rhs)
    {
        *this = *this / rhs;
        return *this;
    }

    HyperJet&
    operator/=
    (
        const T rhs)
    {
        m_v /= rhs;
        m_g /= rhs;
        m_j /= rhs;
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
        const auto v = lhs - rhs.m_v;
        const auto g = -rhs.m_g;
        const auto j = -rhs.m_j;
        return HyperJet(v, g, j);
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
        const auto v = lhs / rhs.m_v;
        const auto g = -lhs * rhs.m_g / (rhs.m_v * rhs.m_v);
        const auto j = (2 * lhs * rhs.m_g.transpose() * rhs.m_g - rhs.m_v * (lhs * rhs.m_j)) / std::pow(rhs.m_v, 3);
        return HyperJet(v, g, j);
    }

    HyperJet
    sqrt() const
    {
        const auto v = std::sqrt(m_v);
        const auto g = m_g / (2 * v);
        const auto j = (2 * m_j - m_g.transpose() * m_g / m_v) / (4 * std::sqrt(m_v));
        return HyperJet(v, g, j);
    }

    HyperJet
    cos() const
    {
        const auto v = std::cos(m_v);
        const auto g = -std::sin(m_v) * m_g;
        const auto j = -std::cos(m_v) * m_g.transpose() * m_g - std::sin(m_v) * m_j;
        return HyperJet(v, g, j);
    }

    HyperJet
    sin() const
    {
        const auto v = std::sin(m_v);
        const auto g = std::cos(m_v) * m_g;
        const auto j = -std::sin(m_v) * m_g.transpose() * m_g + std::cos(m_v) * m_j;
        return HyperJet(v, g, j);
    }

    HyperJet
    tan() const
    {
        const auto v = std::tan(m_v);
        const auto g = m_g * (v * v + 1);
        const auto j = (2 * m_g.transpose() * m_g * v + m_j)*(v * v + 1);
        return HyperJet(v, g, j);
    }

    HyperJet
    acos() const
    {
        const auto v = std::acos(m_v);
        const auto g = -m_g / std::sqrt(-m_v * m_v + 1);
        const auto j = -(m_g.transpose() * m_g * m_v / (-m_v * m_v + 1) + m_j) /
            std::sqrt(-m_v * m_v + 1);
        return HyperJet(v, g, j);
    }

    HyperJet
    asin() const
    {
        const auto v = std::asin(m_v);
        const auto g = m_g / std::sqrt(-m_v * m_v + 1);
        const auto j = (m_g.transpose() * m_g * m_v / (-m_v * m_v + 1) + m_j) /
            std::sqrt(-m_v * m_v + 1);
        return HyperJet(v, g, j);
    }

    HyperJet
    atan() const
    {
        const auto v = std::atan(m_v);
        const auto g = m_g / (m_v * m_v + 1);
        const auto j = (m_j - 2 * m_v * m_g.transpose() * m_g /
            (m_v * m_v + 1)) / (m_v * m_v + 1);
        return HyperJet(v, g, j);
    }

    static HyperJet
    atan2(
        const HyperJet& a,
        const HyperJet& b)
    {
        const auto tmp = a.m_v * a.m_v + b.m_v * b.m_v;

        const auto v = std::atan2(a.m_v, b.m_v);
        const auto g = (a.m_g * b.m_v - a.m_v * b.m_g) / tmp;
        const auto j = (
            2 * (a.m_v * a.m_g + b.m_v * b.m_g).transpose() * (a.m_v * b.m_g) -
            2 * (a.m_v * a.m_g + b.m_v * b.m_g).transpose() * (b.m_v * a.m_g) +
            tmp * (b.m_v * a.m_j - a.m_v * b.m_j + b.m_g.transpose() * a.m_g -
            a.m_g.transpose() * b.m_g)) / std::pow(tmp, 2);
        return HyperJet(v, g, j);

        // const auto j = (
        //     2*(f(a, b)*Derivative(f(a, b), b) + g(a, b)*Derivative(g(a, b), b))*f(a, b)*Derivative(g(a, b), a) - 
        //     2*(f(a, b)*Derivative(f(a, b), b) + g(a, b)*Derivative(g(a, b), b))*g(a, b)*Derivative(f(a, b), a) +
        //     (f(a, b)**2 + g(a, b)**2)*
        //     ( + Derivative(g(a, b), b)*Derivative(f(a, b), a) - Derivative(f(a, b), b)*Derivative(g(a, b), a)))
        //     /(f(a, b)**2 + g(a, b)**2)**2;
    }

    bool
    operator==(
        const HyperJet& rhs) const
    {
        return m_v == rhs.m_v;
    }

    bool
    operator!=(
        const HyperJet& rhs) const
    {
        return m_v != rhs.m_v;
    }

    bool
    operator<(
        const HyperJet& rhs) const
    {
        return m_v < rhs.m_v;
    }

    bool
    operator>(
        const HyperJet& rhs) const
    {
        return m_v > rhs.m_v;
    }

    bool
    operator<=(
        const HyperJet& rhs) const
    {
        return m_v <= rhs.m_v;
    }

    bool
    operator>=(
        const HyperJet& rhs) const
    {
        return m_v >= rhs.m_v;
    }

    bool
    operator==(
        const T rhs) const
    {
        return m_v == rhs;
    }

    bool
    operator!=(
        const T rhs) const
    {
        return m_v != rhs;
    }

    bool
    operator<(
        const T rhs) const
    {
        return m_v < rhs;
    }

    bool
    operator>(
        const T rhs) const
    {
        return m_v > rhs;
    }

    bool
    operator<=(
        const T rhs) const
    {
        return m_v <= rhs;
    }

    bool
    operator>=(
        const T rhs) const
    {
        return m_v >= rhs;
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
        return "HyperJet<" + std::to_string(m_v) + ">";
    }
};