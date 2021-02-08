#include <array> // array
#include <assert.h> // assert
#include <cstddef> // ptrdiff_t
#include <initializer_list> // initializer_list
#include <ostream> // ostream
#include <sstream> // stringstream
#include <string> // string
#include <type_traits> // conditional
#include <vector> // vector

namespace hyperjet {

#if defined(_MSC_VER)
#define HYPERJET_INLINE __forceinline
#else
#define HYPERJET_INLINE __attribute__((always_inline)) inline
#endif

using index = std::ptrdiff_t;

const index Dynamic = -1;

template <typename T>
HYPERJET_INLINE index length(const T& container)
{
    return static_cast<index>(container.size());
}

HYPERJET_INLINE constexpr bool throw_exceptions()
{
#if defined(HYPERJET_NO_EXCEPTIONS)
    return false;
#else
    return true;
#endif
}

HYPERJET_INLINE index data_length_from_size(const index size)
{
    return (size + 1) * (size + 2) / 2;
}

template <typename T>
HYPERJET_INLINE index size_from_data_length(const T& container)
{
    const index s = static_cast<index>(std::sqrt(1 + 8 * length(container)) - 3) / 2;

    if constexpr (throw_exceptions()) {
        if (data_length_from_size(s) != length(container)) {
            throw std::runtime_error("Invalid length");
        }
    } else {
        assert(data_length_from_size(s) == length(container) == 0 && "Invalid length");
    }

    return s;
}

template <index TSize>
HYPERJET_INLINE void check_valid_size(const index size)
{
    if constexpr (TSize == Dynamic) {
        if constexpr (throw_exceptions()) {
            if (size < 0) {
                throw std::runtime_error("Negative size");
            }
        } else {
            assert(size >= 0 && "Negative size");
        }
    } else {
        if constexpr (throw_exceptions()) {
            if (size != TSize) {
                throw std::runtime_error("Invalid size");
            }
        } else {
            assert(size == TSize && "Invalid size");
        }
    }
}

template <index TSize>
HYPERJET_INLINE void check_equal_size(const index size_a, const index size_b)
{
    if constexpr (TSize == Dynamic) {
        if constexpr (throw_exceptions()) {
            if (size_a != size_b) {
                throw std::runtime_error("Incompatible size");
            }
        } else {
            assert(size_a == size_b && "Incompatible size");
        }
    }
}

template <typename TScalar, index TSize>
class DDScalar {
public:
    using Type = DDScalar<TScalar, TSize>;
    using Scalar = TScalar;
    using Data = typename std::conditional<TSize == Dynamic, std::vector<Scalar>, std::array<Scalar, (TSize + 1) * (TSize + 2) / 2>>::type;

    index m_size;
    Data m_data;

    DDScalar()
    {
        if constexpr (is_dynamic()) {
            m_size = 1;
            m_data = Data(1);
        } else {
            static_assert(TSize >= 0);
        }
    }

    DDScalar(const TScalar f)
    {
        if constexpr (is_dynamic()) {
            m_size = 1;
            m_data = Data(1);
        } else {
            static_assert(TSize >= 0);
        }
        m_data[0] = f;
    }

    DDScalar(const Data& data)
        : m_data(data)
    {
        static_assert(!is_dynamic());
    }

    DDScalar(const Data& data, const index size)
        : m_data(data)
        , m_size(size)
    {
        static_assert(is_dynamic());
    }

    DDScalar(std::initializer_list<TScalar> data)
        : m_size(size_from_data_length(data))
    {
        std::copy(data.begin(), data.end(), m_data.begin());
    }

    Data& data()
    {
        return m_data;
    }

    const Data& data() const
    {
        return m_data;
    }

    constexpr index size() const
    {
        if (is_dynamic()) {
            return m_size;
        } else {
            return TSize;
        }
    }

    void resize(const index size)
    {
        static_assert(is_dynamic());
        m_size = size;
        const index n = data_length_from_size(size);
        m_data.resize(n);
    }

    static constexpr bool is_dynamic()
    {
        return TSize == Dynamic;
    }

    static Type create(const Data& data)
    {
        if constexpr (is_dynamic()) {
            const auto s = size_from_data_length(data);
            Type result(data, s);
            return result;
        } else {
            const auto s = size_from_data_length(data);
            check_valid_size<TSize>(s);
            Type result(data);
            return result;
        }
    }

    static Type empty()
    {
        if constexpr (is_dynamic()) {
            Data data(1);
            Type result(data, 0);
            return result;
        } else {
            Data data;
            Type result(data);
            return result;
        }
    }

    static Type empty(const index size)
    {
        if constexpr (is_dynamic()) {
            const index n = data_length_from_size(size);
            const Data data(n);
            Type result(data, size);
            return result;
        } else {
            check_valid_size<TSize>(size);
            return empty();
        }
    }

    static Type zero()
    {
        if constexpr (is_dynamic()) {
            Data data(1);
            Type result(data, 0);
            return result;
        } else {
            Data data;
            data.fill(0);
            Type result(data);
            return result;
        }
    }

    static Type zero(const index size)
    {
        if constexpr (is_dynamic()) {
            const Data data(data_length_from_size(size), 0);
            Type result(data, size);
            return result;
        } else {
            check_valid_size<TSize>(size);
            return zero();
        }
    }

    static Type constant(const Scalar f)
    {
        Type result = zero();
        result.f() = f;
        return result;
    }

    static Type constant(const Scalar f, const index size)
    {
        if constexpr (is_dynamic()) {
            Type result = zero(size);
            result.f() = f;
            return result;
        } else {
            check_valid_size<TSize>(size);
            return constant(f);
        }
    }

    static Type variable(const index i, const Scalar f)
    {
        static_assert(!is_dynamic());
        Type result = zero();
        result.f() = f;
        result.g(i) = 1;
        return result;
    }

    static Type variable(const index i, const Scalar f, const index size)
    {
        if constexpr (is_dynamic()) {
            Type result = zero(size);
            result.f() = f;
            result.g(i) = 1;
            return result;
        } else {
            check_valid_size<TSize>(size);
            return variable(i, f);
        }
    }

    static std::array<Type, TSize < 0 ? 0 : TSize> variables(std::array<Scalar, TSize < 0 ? 0 : TSize> values)
    {
        static_assert(!is_dynamic());

        std::array<Type, TSize < 0 ? 0 : TSize> vars;

        for (index i = 0; i < TSize; i++) {
            vars[i] = variable(i, values[i]);
        }

        return vars;
    }

    static std::vector<Type> variables(std::vector<Scalar> values)
    {
        const index s = length(values);

        if constexpr(!is_dynamic()) {
            assert(s == TSize);
        }

        std::vector<Type> vars(s);
        for (index i = 0; i < s; i++) {
            vars[i] = variable(i, values[i], s);
        }
        return vars;
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
        f() = value;
    }

    Scalar& g(const index i)
    {
        assert(i < size());

        return m_data[1 + i];
    }

    Scalar g(const index i) const
    {
        assert(0 <= i && i < size());

        return m_data[1 + i];
    }

    void set_g(const index i, const double value)
    {
        g(i) = value;
    }

    Scalar& h(const index i)
    {
        assert(0 <= i && i < size() * (size() + 1) / 2);

        return m_data[1 + TSize + i];
    }

    Scalar h(const index i) const
    {
        assert(0 <= i && i < size() * (size() + 1) / 2);

        return m_data[1 + TSize + i];
    }

    void set_h(const index i, const double value)
    {
        h(i) = value;
    }

    Scalar& h(const index i, const index j)
    {
        assert(0 <= i && i < size());
        assert(i <= j && j < size());

        return m_data[1 + TSize + (2 * TSize - 1 - i) * i / 2 + j];
    }

    Scalar h(const index i, const index j) const
    {
        assert(0 <= i && i < size());
        assert(i <= j && j < size());

        return m_data[1 + TSize + (2 * TSize - 1 - i) * i / 2 + j];
    }

    void set_h(const index i, const index j, const double value)
    {
        h(i, j) = value;
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

    Type operator-() const
    {
        Type result = Type::empty(size());

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = -m_data[i];
        }

        return result;
    }

    // --- add

    Type operator+(const Type& b) const
    {
        check_equal_size<TSize>(size(), b.size());

        Type result = *this;

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] += b.m_data[i];
        }

        return result;
    }

    Type operator+(const Scalar b) const
    {
        Type result = *this;

        result.m_data[0] += b;

        return result;
    }

    friend Type operator+(const Scalar a, const Type& b)
    {
        return b + a;
    }

    Type& operator+=(const Type& b)
    {
        check_equal_size<TSize>(size(), b.size());

        for (index i = 0; i < length(m_data); i++) {
            m_data[i] += b.m_data[i];
        }

        return *this;
    }

    Type& operator+=(const Scalar& b)
    {
        m_data[0] += b;

        return *this;
    }

    // --- sub

    Type operator-(const Type& b) const
    {
        check_equal_size<TSize>(size(), b.size());

        Type result = *this;

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] -= b.m_data[i];
        }

        return result;
    }

    Type operator-(const Scalar b) const
    {
        return -b + *this;
    }

    friend Type operator-(const Scalar a, const Type& b)
    {
        Type result = Type::empty(b.size());

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = -b.m_data[i];
        }

        result.m_data[0] += a;

        return result;
    }

    Type& operator-=(const Type& b)
    {
        check_equal_size<TSize>(size(), b.size());

        for (index i = 0; i < length(m_data); i++) {
            m_data[i] -= b.m_data[i];
        }

        return *this;
    }

    Type& operator-=(const Scalar& b)
    {
        m_data[0] -= b;

        return *this;
    }

    // --- mul

    Type operator*(const Type& b) const
    {
        check_equal_size<TSize>(size(), b.size());

        const double d_a = b.m_data[0];
        const double d_b = m_data[0];

        Type result = Type::empty(size());

        result.m_data[0] = m_data[0] * b.m_data[0];

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                *it++ += m_data[1 + i] * b.m_data[1 + j] + m_data[1 + j] * b.m_data[1 + i];
            }
        }

        return result;
    }

    Type operator*(const Scalar b) const
    {
        Type result = Type::empty(size());

        for (index i = 0; i < length(result.m_data); i++) {
            result.m_data[i] = m_data[i] * b;
        }

        return result;
    }

    friend Type operator*(const Scalar a, const Type& b)
    {
        return b * a;
    }

    Type& operator*=(const Type& b)
    {
        check_equal_size<TSize>(size(), b.size());

        const Data a_m_data = m_data;

        const double d_a = b.m_data[0];
        const double d_b = m_data[0];

        m_data[0] *= b.m_data[0];

        for (index i = 1; i < length(m_data); i++) {
            m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                *it++ += a_m_data[1 + i] * b.m_data[1 + j] + a_m_data[1 + j] * b.m_data[1 + i];
            }
        }

        return *this;
    }

    Type& operator*=(const Scalar& b)
    {
        for (index i = 0; i < length(m_data); i++) {
            m_data[i] = m_data[i] * b;
        }

        return *this;
    }

    // --- div

    Type operator/(const Type& b) const
    {
        check_equal_size<TSize>(size(), b.size());

        const double d_a = 1 / b.m_data[0];
        const double d_b = -m_data[0] / std::pow(b.m_data[0], 2);
        const double dd_ab = -1 / std::pow(b.m_data[0], 2);
        const double dd_bb = 2 * m_data[0] / std::pow(b.m_data[0], 3);

        Type result = Type::empty(size());

        result.m_data[0] = m_data[0] * d_a;

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                *it++ += dd_bb * b.m_data[1 + i] * b.m_data[1 + j] + dd_ab * (m_data[1 + i] * b.m_data[1 + j] + m_data[1 + j] * b.m_data[1 + i]);
            }
        }

        return result;
    }

    Type operator/(const Scalar b) const
    {
        return 1 / b * (*this);
    }

    friend Type operator/(const Scalar a, const Type& b)
    {
        return 0;
        ;
    }

    Type& operator/=(const Type& b)
    {
        check_equal_size<TSize>(size(), b.size());

        const Data a_m_data = m_data;

        const double d_a = 1 / b.m_data[0];
        const double d_b = -m_data[0] / std::pow(b.m_data[0], 2);
        const double dd_ab = -1 / std::pow(b.m_data[0], 2);
        const double dd_bb = 2 * m_data[0] / std::pow(b.m_data[0], 3);

        m_data[0] = m_data[0] * d_a;

        for (index i = 1; i < length(m_data); i++) {
            m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                *it++ += dd_bb * b.m_data[1 + i] * b.m_data[1 + j] + dd_ab * (a_m_data[1 + i] * b.m_data[1 + j] + a_m_data[1 + j] * b.m_data[1 + i]);
            }
        }

        return *this;
    }

    Type& operator/=(const Scalar& b)
    {
        operator*=(1 / b);

        return *this;
    }

    // --- pow

    Type pow(const Scalar b) const
    {
        using std::pow;

        const double d = b * pow(m_data[0], b - 1);
        const double dd = (b - 1) * b * pow(m_data[0], b - 2);

        Type result = Type::empty(size());

        result.m_data[0] = pow(m_data[0], b);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
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

        Type result = Type::empty(size());

        result.m_data[0] = sqrt(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
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

        Type result = Type::empty(size());

        result.m_data[0] = cos(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
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

        Type result = Type::empty(size());

        result.m_data[0] = sin(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
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

        Type result = Type::empty(size());

        result.m_data[0] = tmp;

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
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

        Type result = Type::empty(size());

        result.m_data[0] = acos(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
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

        Type result = Type::empty(size());

        result.m_data[0] = asin(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
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

        Type result = Type::empty(size());

        result.m_data[0] = atan(m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d * m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                *it++ += dd * m_data[1 + i] * m_data[1 + j];
            }
        }

        return result;
    }

    Type atan2(const Type& b) const
    {
        using std::atan2;

        const double tmp = m_data[0] * m_data[0] + b.m_data[0] * b.m_data[0];

        const double d_a = b.m_data[0] / tmp;
        const double d_b = -m_data[0] / tmp;
        const double d_aa = d_b * d_a * 2; // = -d_bb
        const double d_ab = d_b * d_b - d_a * d_a;

        Type result = Type::empty(size());

        result.m_data[0] = atan2(m_data[0], b.m_data[0]);

        for (index i = 1; i < length(result.m_data); i++) {
            result.m_data[i] = d_a * m_data[i] + d_b * b.m_data[i];
        }

        auto* it = &result.m_data[1 + size()];

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                *it++ += d_aa * (m_data[1 + i] * m_data[1 + j] - b.m_data[1 + i] * b.m_data[1 + j]) + d_ab * (m_data[1 + i] * b.m_data[1 + j] + b.m_data[1 + i] * m_data[1 + j]);
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
    return a.atan2(b);
}

} // namespace hyperjet

#ifdef EIGEN_WORLD_VERSION

namespace Eigen {

template <typename T>
struct NumTraits;

template <typename TScalar, std::ptrdiff_t TSize>
struct NumTraits<hyperjet::DDScalar<TScalar, TSize>> : NumTraits<TScalar> {
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
struct ScalarBinaryOpTraits<hyperjet::DDScalar<TScalar, TSize>, TScalar, BinOp> {
    using ReturnType = hyperjet::DDScalar<TScalar, TSize>;
};

template <typename BinOp, typename TScalar, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<TScalar, hyperjet::DDScalar<TScalar, TSize>, BinOp> {
    using ReturnType = hyperjet::DDScalar<TScalar, TSize>;
};

} // namespace Eigen

#endif