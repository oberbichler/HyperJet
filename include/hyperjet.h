#include <array> // array
#include <assert.h> // assert
#include <cstddef> // ptrdiff_t
#include <initializer_list> // initializer_list
#include <ostream> // ostream
#include <string> // string
#include <sstream> // stringstream

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

template <typename TScalar, index TSize>
class DDScalar
{
public:
    using Type = DDScalar<TScalar, TSize>;
    using Scalar = TScalar;
    using Data = std::array<Scalar, (TSize + 1) * (TSize + 2) / 2>;

    Data m_data;

    DDScalar()
    {
    }

    DDScalar(const TScalar f)
    {
        m_data[0] = f;
    }

    DDScalar(std::initializer_list<TScalar> data)
    {
        std::copy(data.begin(), data.end(), m_data.begin());
    }

    DDScalar(const Data& data) : m_data(data)
    {
    }

    constexpr index size() const
    {
        // size = (std::sqrt(1 + 8 * length(m_data)) - 3) / 2;
        return TSize;
    }

    static Type constant(const Scalar f)
    {
        Type result;
        result.f() = f;
        return result;
    }

    template <index TIndex>
    static Type variable(const Scalar f)
    {
        Type result;
        result.f() = f;
        result.g(TIndex) = 1;
        return result;
    }

    Scalar& f()
    {
        return m_data[0];
    }

    Scalar f() const
    {
        return m_data[0];
    }

    void set_f(const double value)
    {
        m_data[0] = value;
    }

    Scalar& g(const index i)
    {
        assert(i < TSize);

        return m_data[1 + i];
    }

    Scalar g(const index i) const
    {
        assert(i < TSize);

        return m_data[1 + i];
    }

    Scalar& h(const index i)
    {
        assert(i < TSize * (TSize + 1) / 2);

        return m_data[1 + TSize + i];
    }

    Scalar h(const index i) const
    {
        assert(i < TSize * (TSize + 1) / 2);

        return m_data[1 + TSize + i];
    }

    Scalar& h(const index i, const index j)
    {
        assert(i < TSize);
        assert(j < TSize);

        if (i > j) {
            return m_data[1 + TSize + (2 * TSize - 1 - j) * j / 2 + i];
        }

        return m_data[1 + TSize + (2 * TSize - 1 - i) * i / 2 + j];
    }

    Scalar h(const index i, const index j) const
    {
        assert(i < TSize);
        assert(j < TSize);

        if (i > j) {
            return m_data[1 + TSize + (2 * TSize - 1 - j) * j / 2 + i];
        }

        return m_data[1 + TSize + (2 * TSize - 1 - i) * i / 2 + j];
    }

    friend std::ostream& operator<<(std::ostream& out, const Type& value)
    {
        out << value.m_data[0] << "hj";
        return out;
    }

    std::string to_string()
    {
        std::stringstream output;

        output << *this;

        return output.str();
    }

    // --- neg

    Type operator -() const
    {
        Type result;

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = -m_data[i];
        }

        return result;
    }

    // --- add

    Type operator +(const Type& b) const
    {
        Type result;

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = m_data[i] + b.m_data[i];
        }

        return result;
    }

    Type operator +(const Scalar b) const
    {
        Type result;

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = m_data[i];
        }

        result.m_data[0] += b;

        return result;
    }

    friend Type operator +(const Scalar a, const Type& b)
    {
        return b + a;
    }

    Type& operator +=(const Type& b)
    {
        for (index i = 0; i < length(m_data); i++) {
            m_data[i] += b.m_data[i];
        }

        return *this;
    }

    Type& operator +=(const Scalar& b)
    {
        m_data[0] += b;

        return *this;
    }

    // --- sub

    Type operator -(const Type& b) const
    {
        Type result;

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = m_data[i] - b.m_data[i];
        }

        return result;
    }

    Type operator -(const Scalar b) const
    {
        return -b + *this;
    }

    friend Type operator -(const Scalar a, const Type& b)
    {
        Type result;

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = -b.m_data[i];
        }

        result.m_data[0] += a;

        return result;
    }

    Type& operator -=(const Type& b)
    {
        for (index i = 0; i < length(m_data); i++) {
            m_data[i] -= b.m_data[i];
        }

        return *this;
    }

    Type& operator -=(const Scalar& b)
    {
        m_data[0] -= b;

        return *this;
    }

    // --- mul

    Type operator *(const Type& b) const
    {
        const double d_a = b.m_data[0];
        const double d_b = m_data[0];

        Type result;

        result.m_data[0] = m_data[0] * b.m_data[0];

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += m_data[1 + i] * b.m_data[1 + j] + m_data[1 + j] * b.m_data[1 + i];
            }
        }

        return result;
    }

    Type operator *(const Scalar b) const
    {
        Type result;

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = m_data[i] * b;
        }

        return result;
    }
    
    friend Type operator *(const Scalar a, const Type& b)
    {
        return b * a;
    }

    Type& operator *=(const Type& b)
    {
        const Data a_m_data = m_data;

        const double d_a = b.m_data[0];
        const double d_b = m_data[0];

        m_data[0] *= b.m_data[0];

        for (index i = 1; i < length(m_data); i++) {
            m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += a_m_data[1 + i] * b.m_data[1 + j] + a_m_data[1 + j] * b.m_data[1 + i];
            }
        }

        return *this;
    }

    Type& operator *=(const Scalar& b)
    {
        for (index i = 0; i < length(m_data); i++) {
            m_data[i] = m_data[i] * b;
        }

        return *this;
    }

    // --- div

    Type operator /(const Type& b) const
    {
        const double d_a = 1 / b.m_data[0];
        const double d_b = -m_data[0] / std::pow(b.m_data[0], 2);
        const double dd_ab = -1 / std::pow(b.m_data[0], 2);
        const double dd_bb = 2 * m_data[0] / std::pow(b.m_data[0], 3);

        Type result;

        result.m_data[0] = m_data[0] * d_a;

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd_bb * b.m_data[1 + i] * b.m_data[1 + j] + dd_ab * (m_data[1 + i] * b.m_data[1 + j] + m_data[1 + j] * b.m_data[1 + i]);
            }
        }

        return result;
    }

    Type operator /(const Scalar b) const
    {
        return 1 / b * (*this);
    }

    friend Type operator /(const Scalar a, const Type& b)
    {
        return 0;;
    }

    Type& operator /=(const Type& b)
    {
        const Data a_m_data = m_data;

        const double d_a = 1 / b.m_data[0];
        const double d_b = -m_data[0] / std::pow(b.m_data[0], 2);
        const double dd_ab = -1 / std::pow(b.m_data[0], 2);
        const double dd_bb = 2 * m_data[0] / std::pow(b.m_data[0], 3);

        m_data[0] = m_data[0] * d_a;

        for (index i = 1; i < length(m_data); i++) {
            m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd_bb * b.m_data[1 + i] * b.m_data[1 + j] + dd_ab * (a_m_data[1 + i] * b.m_data[1 + j] + a_m_data[1 + j] * b.m_data[1 + i]);
            }
        }

        return *this;
    }

    Type& operator /=(const Scalar& b)
    {
        operator *=(1 / b);

        return *this;
    }

    // --- pow

    Type pow(const Scalar b) const
    {
        using std::pow;

        const double d = b * pow(m_data[0], b - 1);
        const double dd = (b - 1) * b * pow(m_data[0], b - 2);

        Type result;

        result.m_data[0] = pow(m_data[0], b);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    Type sqrt() const
    {
        using std::pow;
        using std::sqrt;

        const double d = 1 / (2 * sqrt(m_data[0]));
        const double dd = -d / (2 * m_data[0]);

        Type result;

        result.m_data[0] = sqrt(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    // --- trig

    Type cos() const
    {
        using std::cos;
        using std::sin;

        const double d = -sin(m_data[0]);
        const double dd = -cos(m_data[0]);

        Type result;

        result.m_data[0] = cos(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    Type sin() const
    {
        using std::cos;
        using std::sin;

        const double d = cos(m_data[0]);
        const double dd = -sin(m_data[0]);

        Type result;

        result.m_data[0] = sin(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    Type tan() const
    {
        using std::tan;

        const double tmp = tan(m_data[0]);

        const double d = tmp * tmp + 1;
        const double dd = d * 2 * tmp;

        Type result;

        result.m_data[0] = tmp;

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    Type acos() const
    {
        using std::acos;
        using std::sqrt;

        const double tmp = 1 - m_data[0] * m_data[0];

        const double d = -1 / sqrt(tmp);
        const double dd = d * m_data[0] / tmp;

        Type result;

        result.m_data[0] = acos(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    Type asin() const
    {
        using std::asin;
        using std::sqrt;

        const double tmp = 1 - m_data[0] * m_data[0];

        const double d = 1 / sqrt(tmp);
        const double dd = d * m_data[0] / tmp;

        Type result;

        result.m_data[0] = asin(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    Type atan() const
    {
        using std::atan;

        const double d = 1 / (m_data[0] * m_data[0] + 1);
        const double dd = -d * d * 2 * m_data[0];

        Type result;

        result.m_data[0] = atan(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    static Type atan2(const Type& a, const Type& b)
    {
        using std::atan2;

        const double tmp = a.m_data[0] * a.m_data[0] + b.m_data[0] * b.m_data[0];

        const double d_a = b.m_data[0] / tmp;
        const double d_b = -a.m_data[0] / tmp;
        const double d_aa = d_b * d_a * 2; // = -d_bb
        const double d_ab = d_b * d_b - d_a * d_a;

        Type result;

        result.m_data[0] = atan2(a.m_data[0], b.m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d_a * a. m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &result.m_data[1 + TSize];

        for (index i = 0; i < TSize; i++) {
            for (index j = i; j < TSize; j++) {
                *it++ += d_aa * (a.m_data[1 + i] * a.m_data[1 + j] - b.m_data[1 + i] * b.m_data[1 + j]) + d_ab * (a.m_data[1 + i] * b.m_data[1 + j] + b.m_data[1 + i] * a.m_data[1 + j]);
            }
        }

        return result;
    }

    // abs

    Type abs() const
    {
        return m_data[0] < 0 ? -(*this) : *this;
    }

    // comparison

    bool operator==(const Type& b) const
    {
        return m_data[0] == b.m_data[0];
    }

    bool operator!=(const Type& b) const
    {
        return m_data[0] != b.m_data[0];
    }

    bool operator<(const Type& b) const
    {
        return m_data[0] < b.m_data[0];
    }

    bool operator>(const Type& b) const
    {
        return m_data[0] > b.m_data[0];
    }

    bool operator<=(const Type& b) const
    {
        return m_data[0] <= b.m_data[0];
    }

    bool operator>=(const Type& b) const
    {
        return m_data[0] >= b.m_data[0];
    }

    bool operator==(const Scalar b) const
    {
        return m_data[0] == b;
    }

    bool operator!=(const Scalar b) const
    {
        return m_data[0] != b;
    }

    bool operator<(const Scalar b) const
    {
        return m_data[0] < b;
    }

    bool operator>(const Scalar b) const
    {
        return m_data[0] > b;
    }

    bool operator<=(const Scalar b) const
    {
        return m_data[0] <= b;
    }

    bool operator>=(const Scalar b) const
    {
        return m_data[0] >= b;
    }

    friend bool operator==(const Scalar a, const Type& b)
    {
        return b.operator==(a);
    }

    friend bool operator!=(const Scalar a, const Type& b)
    {
        return b.operator!=(a);
    }

    friend bool operator<(const Scalar a, const Type& b)
    {
        return b.operator>(a);
    }

    friend bool operator>(const Scalar a, const Type& b)
    {
        return b.operator<(a);
    }

    friend bool operator<=(const Scalar a, const Type& b)
    {
        return b.operator>=(a);
    }

    friend bool operator>=(const Scalar a, const Type& b)
    {
        return b.operator<=(a);
    }
};

// std::pow

using std::pow;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> pow(const DDScalar<TScalar, TSize>& a, const index b)
{
    return a.pow(b);
}

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> pow(const DDScalar<TScalar, TSize>& a, const double b)
{
    return a.pow(b);
}

// std::sqrt

using std::sqrt;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> sqrt(const DDScalar<TScalar, TSize>& a)
{
    return a.sqrt();
}

// std::cos

using std::cos;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> cos(const DDScalar<TScalar, TSize>& a)
{
    return a.cos();
}

// std::sin

using std::sin;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> sin(const DDScalar<TScalar, TSize>& a)
{
    return a.sin();
}

// std::tan

using std::tan;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> tan(const DDScalar<TScalar, TSize>& a)
{
    return a.tan();
}

// std::acos

using std::acos;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> acos(const DDScalar<TScalar, TSize>& a)
{
    return a.acos();
}

// std::asin

using std::asin;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> asin(const DDScalar<TScalar, TSize>& a)
{
    return a.asin();
}

// std::atan

using std::atan;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> atan(const DDScalar<TScalar, TSize>& a)
{
    return a.atan();
}

// std::atan2

using std::atan2;

template <typename TScalar, index TSize>
DDScalar<TScalar, TSize> atan2(const DDScalar<TScalar, TSize>& a, const DDScalar<TScalar, TSize>& b)
{
    return DDScalar<TScalar, TSize>::atan2(a, b);
}

} // namespace hyperjet

#ifdef EIGEN_WORLD_VERSION

namespace Eigen {

template <typename T>
struct NumTraits;

template <typename TScalar, std::ptrdiff_t TSize>
struct NumTraits<hyperjet::DDScalar<TScalar, TSize>> : NumTraits<TScalar>
{
    using Real = hyperjet::DDScalar<TScalar, TSize>;
    using NonInteger = hyperjet::DDScalar<TScalar, TSize>;
    using Nested = hyperjet::DDScalar<TScalar, TSize>;

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 3
    };
};

template <typename BinOp, typename TScalar, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<hyperjet::DDScalar<TScalar, TSize>, TScalar, BinOp>
{
    using ReturnType = hyperjet::DDScalar<TScalar, TSize>;
};

template <typename BinOp, typename TScalar, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<TScalar, hyperjet::DDScalar<TScalar, TSize>, BinOp>
{
    using ReturnType = hyperjet::DDScalar<TScalar, TSize>;
};

} // namespace Eigen

#endif