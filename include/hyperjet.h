#pragma once

#include <array> // array
#include <assert.h> // assert
#include <cstddef> // ptrdiff_t
#include <cmath> // acos, acosh, asin, asinh, atan, atanh, atan2, cbrt, cos, cosh, exp, hypot, log, log2, log10, pow, sin, sinh, sqrt, tan, tanh
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

template <index TOrder, typename TScalar = double, index TSize = Dynamic>
class DDScalar {
    static constexpr index StaticDataSize = TOrder == 1 ? 1 + TSize : (TSize + 1) * (TSize + 2) / 2;
    static constexpr index DataSize = TSize < 0 ? Dynamic : StaticDataSize;

    using DynamicStorage = std::vector<TScalar>;
    using StaticStorage = std::array<TScalar, StaticDataSize>;

public:
    using Type = DDScalar<TOrder, TScalar, TSize>;
    using Scalar = TScalar;
    using Data = typename std::conditional<TSize == Dynamic, DynamicStorage, StaticStorage>::type;

    index m_size;
    Data m_data;

public:
    HYPERJET_INLINE static index data_length_from_size(const index size)
    {
        return TOrder == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
    }

    HYPERJET_INLINE static index size_from_data_length(const index n)
    {
        const index s = TOrder == 1 ? n - 1 : static_cast<index>(std::sqrt(1 + 8 * n) - 3) / 2;

        if constexpr (throw_exceptions()) {
            if (data_length_from_size(s) != n) {
                throw std::runtime_error("Invalid length");
            }
        } else {
            assert(data_length_from_size(s) == n == 0 && "Invalid length");
        }

        return s;
    }

private:
    HYPERJET_INLINE static void check_valid_size(const index size)
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

    HYPERJET_INLINE static void check_equal_size(const index size_a, const index size_b)
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

    struct Zero {
        HYPERJET_INLINE Zero operator*(const Scalar) const
        {
            return Zero();
        }

        template <typename T>
        HYPERJET_INLINE T operator+(const T b) const
        {
            return b;
        }

        template <typename T>
        HYPERJET_INLINE friend T operator+(const T a, const Zero)
        {
            return a;
        }
    };

    struct MinusOne {
        HYPERJET_INLINE Scalar operator*(const Scalar b) const
        {
            return -b;
        }
    };

    struct One {
        HYPERJET_INLINE MinusOne operator-() const
        {
            return MinusOne();
        }

        HYPERJET_INLINE Scalar operator*(const Scalar b) const
        {
            return b;
        }
    };

    template <bool TIncrement, typename TDa, typename TDaa>
    HYPERJET_INLINE void unary(const Data& a, const Scalar f, const TDa da, const TDaa daa, Data& r) const noexcept
    {
        const index n = data_length();

        r[0] = f;

        if constexpr (TOrder < 1 || std::is_same<TDa, Zero>()) {
            return;
        }

        for (index i = 1; i < n; i++) {
            if constexpr (TIncrement) {
                r[i] += da * a[i];
            } else {
                r[i] = da * a[i];
            }
        }

        if constexpr (TOrder < 2 || std::is_same<TDaa, Zero>()) {
            return;
        }

        index k = 1 + size();

        for (index i = 0; i < size(); i++) {
            const auto ca = daa * a[1 + i];

            for (index j = i; j < size(); j++) {
                r[k++] += ca * a[1 + j];
            }
        }
    }

    template <bool TIncrement, typename TDa, typename TDb, typename TDaa, typename TDab, typename TDbb>
    HYPERJET_INLINE void binary(const Data& a, const Data& b, const Scalar f, const TDa da, const TDb db, const TDaa daa, const TDab dab, const TDbb dbb, Data& r) const noexcept
    {
        const index n = data_length();

        r[0] = f;

        if constexpr (TOrder < 1 || (std::is_same_v<TDa, Zero> && std::is_same_v<TDb, Zero>)) {
            return;
        } else {
            for (index i = 1; i < n; i++) {
                if constexpr (TIncrement) {
                    r[i] += da * a[i] + db * b[i];
                } else {
                    r[i] = da * a[i] + db * b[i];
                }
            }
        }

        if constexpr (TOrder < 2 || (std::is_same_v<TDaa, Zero> && std::is_same_v<TDab, Zero> && std::is_same_v<TDbb, Zero>)) {
            return;
        } else {
            index k = 1 + size();

            for (index i = 0; i < size(); i++) {
                const auto ca = daa * a[1 + i] + dab * b[1 + i];
                const auto cb = dab * a[1 + i] + dbb * b[1 + i];

                for (index j = i; j < size(); j++) {
                    r[k++] += ca * a[1 + j] + cb * b[1 + j];
                }
            }
        }
    }

    template <bool TIncrement, typename TDa, typename TDb, typename TDc, typename TDaa, typename TDab, typename TDac, typename TDbb, typename TDbc, typename TDcc>
    HYPERJET_INLINE void ternary(const Data& a, const Data& b, const Data& c, const Scalar f, const TDa da, const TDb db, const TDc dc, const TDaa daa, const TDab dab, const TDac dac, const TDbb dbb, const TDbc dbc, const TDcc dcc, Data& r) const noexcept
    {
        const index n = data_length();

        r[0] = f;

        if constexpr (TOrder < 1 || (std::is_same_v<TDa, Zero> && std::is_same_v<TDb, Zero> && std::is_same_v<TDc, Zero>)) {
            return;
        } else {
            for (index i = 1; i < n; i++) {
                if constexpr (TIncrement) {
                    r[i] += da * a[i] + db * b[i] + dc * c[i];
                } else {
                    r[i] = da * a[i] + db * b[i] + dc * c[i];
                }
            }
        }

        if constexpr (TOrder < 2 || (std::is_same_v<TDaa, Zero> && std::is_same_v<TDab, Zero> && std::is_same_v<TDac, Zero> && std::is_same_v<TDbb, Zero> && std::is_same_v<TDbc, Zero> && std::is_same_v<TDcc, Zero>)) {
            return;
        } else {
            index k = 1 + size();

            for (index i = 0; i < size(); i++) {
                const auto ca = daa * a[1 + i] + dab * b[1 + i] + dac * c[1 + i];
                const auto cb = dab * a[1 + i] + dbb * b[1 + i] + dbc * c[1 + i];
                const auto cc = dac * a[1 + i] + dbc * b[1 + i] + dcc * c[1 + i];

                for (index j = i; j < size(); j++) {
                    r[k++] += ca * a[1 + j] + cb * b[1 + j] + cc * c[1 + j];
                }
            }
        }
    }

public:
    DDScalar()
    {
        static_assert(0 < order() && order() <= 2);

        if constexpr (is_dynamic()) {
            m_size = 1;
            m_data = Data(1);
        } else {
            static_assert(TSize >= 0);
        }
    }

    DDScalar(const TScalar f)
    {
        static_assert(0 < order() && order() <= 2);

        if constexpr (is_dynamic()) {
            m_size = 1;
            m_data = Data(1);
        } else {
            static_assert(TSize >= 0);
        }
        m_data[0] = f;
        std::fill(m_data.begin() + 1, m_data.end(), 0);
    }

    DDScalar(const Data& data)
        : m_data(data)
    {
        static_assert(0 < order() && order() <= 2);

        static_assert(!is_dynamic());
    }

    DDScalar(const Data& data, const index size)
        : m_data(data)
        , m_size(size)
    {
        static_assert(0 < order() && order() <= 2);

        static_assert(is_dynamic());
    }

    DDScalar(std::initializer_list<TScalar> data)
        : m_size(size_from_data_length(length(data)))
    {
        static_assert(0 < order() && order() <= 2);

        std::copy(data.begin(), data.end(), m_data.begin());
    }

    static constexpr index order()
    {
        return TOrder;
    }

    Data& data()
    {
        return m_data;
    }

    Scalar const* ptr() const
    {
        return m_data.data();
    }

    Scalar* ptr()
    {
        return m_data.data();
    }

    const Data& data() const
    {
        return m_data;
    }

    static constexpr index static_size()
    {
        return TSize;
    }

    constexpr index size() const
    {
        if constexpr (is_dynamic()) {
            return m_size;
        } else {
            return TSize;
        }
    }

    constexpr index data_length() const
    {
        if constexpr (is_dynamic()) {
            return length(m_data);
        } else {
            return StaticDataSize;
        }
    }

    void resize(const index size)
    {
        static_assert(is_dynamic());
        m_size = size;
        const index n = data_length_from_size(size);
        m_data.resize(n);
    }

    Type pad_left(const index new_size) const
    {
        static_assert(is_dynamic());

        Type result = empty(new_size);

        const index head = new_size - size();

        auto source = m_data.cbegin();
        auto target = result.m_data.begin();

        *target++ = *source++;

        for (index i = 0; i < head; i++) {
            *target++ = Scalar(0);
        }

        for (index i = 0; i < size(); i++) {
            *target++ = *source++;
        }

        if constexpr (order() == 1) {
            return result;
        }

        for (index i = 0; i < head; i++) {
            for (index j = i; j < new_size; j++) {
                *target++ = Scalar(0);
            }
        }

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                *target++ = *source++;
            }
        }

        return result;
    }

    Type pad_right(const index new_size) const
    {
        static_assert(is_dynamic());

        Type result = empty(new_size);

        const index tail = new_size - size();

        auto source = m_data.cbegin();
        auto target = result.m_data.begin();

        for (index i = 0; i < size() + 1; i++) {
            *target++ = *source++;
        }

        for (index i = 0; i < tail; i++) {
            *target++ = Scalar(0);
        }

        if (order() == 1) {
            return result;
        }

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                *target++ = *source++;
            }

            for (index j = 0; j < tail; j++) {
                *target++ = Scalar(0);
            }
        }

        for (index i = 0; i < tail; i++) {
            for (index j = i; j < tail; j++) {
                *target++ = Scalar(0);
            }
        }

        return result;
    }

    static constexpr bool is_dynamic()
    {
        return TSize == Dynamic;
    }

    static Type create(const Data& data)
    {
        if constexpr (is_dynamic()) {
            const auto s = size_from_data_length(length(data));
            Type result(data, s);
            return result;
        } else {
            const auto s = size_from_data_length(length(data));
            check_valid_size(s);
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
            check_valid_size(size);
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
            check_valid_size(size);
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
            check_valid_size(size);
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
            check_valid_size(size);
            return variable(i, f);
        }
    }

    static std::vector<Type> variables(const std::vector<Scalar>& values)
    {
        const index s = length(values);

        if constexpr (!is_dynamic()) {
            assert(s == TSize);
        }

        std::vector<Type> vars(s);
        for (index i = 0; i < s; i++) {
            vars[i] = variable(i, values[i], s);
        }
        return vars;
    }

    template <index T>
    static std::conditional_t<TSize == Dynamic, std::vector<Type>, std::array<Type, T>> variables(const std::array<Scalar, T>& values)
    {
        if constexpr (!is_dynamic()) {
            static_assert(T == TSize);
        }

        const index s = length(values);

        if constexpr (is_dynamic()) {
            std::vector<Type> vars(s);
            for (index i = 0; i < s; i++) {
                vars[i] = variable(i, values[i], s);
            }
            return vars;
        } else {
            std::array<Type, TSize> vars;
            for (index i = 0; i < s; i++) {
                vars[i] = variable(i, values[i], s);
            }
            return vars;
        }
    }

    Scalar& f()
    {
        return m_data[0];
    }

    Scalar f() const
    {
        return m_data[0];
    }

    void set_f(const Scalar value)
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

    void set_g(const index i, const Scalar value)
    {
        g(i) = value;
    }

    Scalar& h(const index i)
    {
        assert(0 <= i && i < size() * (size() + 1) / 2);

        return m_data[1 + size() + i];
    }

    Scalar h(const index i) const
    {
        assert(0 <= i && i < size() * (size() + 1) / 2);

        return m_data[1 + size() + i];
    }

    void set_h(const index i, const Scalar value)
    {
        h(i) = value;
    }

    Scalar& h(const index i, const index j)
    {
        assert(0 <= i && i < size());
        assert(0 <= j && j < size());

        if (i < j) {
            return m_data[1 + size() + (2 * size() - 1 - i) * i / 2 + j];
        } else {
            return m_data[1 + size() + (2 * size() - 1 - j) * j / 2 + i];
        }
    }

    Scalar h(const index i, const index j) const
    {
        assert(0 <= i && i < size());
        assert(0 <= j && j < size());

        if (i < j) {
            return m_data[1 + size() + (2 * size() - 1 - i) * i / 2 + j];
        } else {
            return m_data[1 + size() + (2 * size() - 1 - j) * j / 2 + i];
        }
    }

    void set_h(const index i, const index j, const Scalar value)
    {
        h(i, j) = value;
    }

#if defined EIGEN_WORLD_VERSION
    using Vector = Eigen::Matrix<TScalar, 1, TSize>;
    using Matrix = Eigen::Matrix<TScalar, TSize, TSize>;
    using DataVector = Eigen::Matrix<TScalar, 1, DataSize>;

    Eigen::Ref<const Vector> ag() const
    {
        return Eigen::Map<Vector>(ptr() + 1, size());
    }

    Eigen::Ref<Vector> ag()
    {
        return Eigen::Map<Vector>(ptr() + 1, size());
    }

    Matrix hm(const std::string mode) const
    {
        Matrix result(size(), size());

        hm(mode, result);

        return result;
    }

    void hm(const std::string mode, Eigen::Ref<Matrix> out) const
    {
        index it = 0;

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                out(i, j) = h(it++);
            }
        }

        if (mode == "zeros") {
            for (index i = 0; i < size(); i++) {
                for (index j = 0; j < i; j++) {
                    out(i, j) = 0;
                }
            }
        } else if (mode == "full") {
            for (index i = 0; i < size(); i++) {
                for (index j = 0; j < i; j++) {
                    out(i, j) = out(j, i);
                }
            }
        } else {
            throw std::runtime_error("Invalid value for 'mode'");
        }
    }

    void set_hm(const Eigen::Ref<const Matrix>& value)
    {
        index it = 0;

        for (index i = 0; i < size(); i++) {
            for (index j = i; j < size(); j++) {
                h(it++) = value(i, j);
            }
        }
    }

    Eigen::Ref<DataVector> adata()
    {
        return Eigen::Map <DataVector >(ptr(), length(m_data));
    }

    void set_adata(const Eigen::Ref<DataVector> value)
    {
        Eigen::Map<DataVector>(ptr(), length(m_data)) = value;
    }

    static Type from_gradient(const TScalar f, const Eigen::Ref<const Vector>& g)
    {
        static_assert(order() == 1);

        Type result = empty(length(g));

        result.f() = f;
        result.ag() = g;

        return result;
    }

    static Type from_arrays(const TScalar f, const Eigen::Ref<const Vector>& g, const Eigen::Ref<const Matrix>& hm)
    {
        static_assert(order() == 2);

        assert(g.size() == hm.rows() && g.size() == hm.cols());

        Type result = empty(length(g));

        result.f() = f;
        result.ag() = g;
        result.set_hm(hm);

        return result;
    }

    Scalar eval(typename std::conditional < TSize == Dynamic, std::vector<Scalar>, std::array<Scalar, TSize<0 ? 0 : TSize>>::type d) const
    {
        Scalar result = f();

        for (index i = 0; i < size(); i++) {
            result += d[i] * g(i);
        }

        if constexpr (order() == 1) {
            return result;
        }

        Scalar t(0);

        for (index i = 0; i < size(); i++) {
            Scalar s(0);

            index k = 1 + size() + i;

            for (index j = 0; j < i; j++) {
                s += d[j] * m_data[k];
                k += size() - j - 1;
            }

            for (index j = i; j < size(); j++) {
                s += d[j] * m_data[k++];
            }

            t += d[i] * s;
        }

        result += 0.5 * t;

        return result;
    }

#endif

    void set_zero()
    {
        std::fill(m_data.begin(), m_data.end(), 0);
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

        for (index i = 0; i < data_length(); i++) {
            result.m_data[i] = -m_data[i];
        }

        return result;
    }

    // --- add

    Type operator+(const Type& b) const
    {
        check_equal_size(size(), b.size());

        Type result = Type::empty(size());

        const auto f = m_data[0] + b.m_data[0];

        const auto da = One();
        const auto db = One();

        const auto daa = Zero();
        const auto dab = Zero();
        const auto dbb = Zero();

        binary<false>(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

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
        check_equal_size(size(), b.size());

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
        check_equal_size(size(), b.size());

        Type result = Type::empty(size());

        const auto f = m_data[0] - b.m_data[0];

        const auto da = One();
        const auto db = -One();

        const auto daa = Zero();
        const auto dab = Zero();
        const auto dbb = Zero();

        binary<false>(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

        return result;
    }

    Type operator-(const Scalar b) const
    {
        return -b + *this;
    }

    friend Type operator-(const Scalar a, const Type& b)
    {
        Type result = Type::empty(b.size());

        for (index i = 0; i < result.data_length(); i++) {
            result.m_data[i] = -b.m_data[i];
        }

        result.m_data[0] += a;

        return result;
    }

    Type& operator-=(const Type& b)
    {
        check_equal_size(size(), b.size());

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
        check_equal_size(size(), b.size());

        Type result = Type::empty(size());

        const auto f = m_data[0] * b.m_data[0];

        const auto da = b.m_data[0];
        const auto db = m_data[0];

        const auto daa = Zero();
        const auto dab = One();
        const auto dbb = Zero();

        binary<false>(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

        return result;
    }

    Type operator*(const Scalar b) const
    {
        Type result = Type::empty(size());

        for (index i = 0; i < data_length(); i++) {
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
        check_equal_size(size(), b.size());

        const Data a_m_data = m_data;

        const Scalar da = b.m_data[0];
        const Scalar db = m_data[0];

        m_data[0] *= b.m_data[0];

        for (index i = 1; i < length(m_data); i++) {
            m_data[i] = da * m_data[i] + db * b.m_data[i];
        }

        if constexpr (order() == 1)
            return *this;

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
        check_equal_size(size(), b.size());

        const Scalar tmp = 1 / b.m_data[0];

        const auto f = m_data[0] * tmp;
        const auto da = tmp;
        const auto db = -m_data[0] / std::pow(b.m_data[0], 2);
        const auto daa = Zero();
        const auto dab = -1 / std::pow(b.m_data[0], 2);
        const auto dbb = 2 * m_data[0] / std::pow(b.m_data[0], 3);

        Type result = Type::empty(size());

        binary<false>(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

        return result;
    }

    Type operator/(const Scalar b) const
    {
        return Scalar(1) / b * (*this);
    }

    friend Type operator/(const Scalar a, const Type& b)
    {
        Type result = Type::empty(b.size());

        const auto f = a / b.m_data[0];
        const auto db = -a / std::pow(b.m_data[0], 2);
        const auto dbb = 2 * a / std::pow(b.m_data[0], 3);

        b.unary<false>(b.m_data, f, db, dbb, result.m_data);

        return result;
    }

    Type& operator/=(const Type& b)
    {
        check_equal_size(size(), b.size());

        const Data a_m_data = m_data;

        const auto f = a_m_data[0] / b.m_data[0];
        const auto da = 1 / b.m_data[0];
        const auto db = -a_m_data[0] / std::pow(b.m_data[0], 2);
        const auto daa = Zero();
        const auto dab = -1 / std::pow(b.m_data[0], 2);
        const auto dbb = 2 * a_m_data[0] / std::pow(b.m_data[0], 3);

        binary<false>(a_m_data, b.m_data, f, da, db, daa, dab, dbb, m_data);

        return *this;
    }

    Type& operator/=(const Scalar& b)
    {
        operator*=(1 / b);

        return *this;
    }

    // --- arithmetic operations

    Type pow(const Scalar b) const
    {
        using std::pow;

        Type result = Type::empty(size());

        const auto f = pow(m_data[0], b);
        const auto da = b * pow(m_data[0], b - 1);
        const auto daa = (b - 1) * b * pow(m_data[0], b - 2);

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type sqrt() const
    {
        using std::pow;
        using std::sqrt;

        Type result = Type::empty(size());

        const auto f = sqrt(m_data[0]);
        const auto da = 1 / (2 * f);
        const auto daa = -da / (2 * m_data[0]);

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type cbrt() const
    {
        using std::cbrt;

        Type result = Type::empty(size());

        const auto f = cbrt(m_data[0]);
        const auto da = 1 / (3 * f * f);
        const auto daa = -da * 2 / (3 * m_data[0]);

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type reciprocal() const
    {
        Type result = Type::empty(size());

        const auto f = 1 / m_data[0];
        const auto da = -f * f;
        const auto daa = -2 * f * da;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    // --- trigonometric functions

    Type cos() const
    {
        using std::cos;
        using std::sin;

        Type result = Type::empty(size());

        const auto f = cos(m_data[0]);
        const auto da = -sin(m_data[0]);
        const auto daa = -cos(m_data[0]);

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type sin() const
    {
        using std::cos;
        using std::sin;

        Type result = Type::empty(size());

        const auto f = sin(m_data[0]);
        const auto da = cos(m_data[0]);
        const auto daa = -sin(m_data[0]);

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type tan() const
    {
        using std::tan;

        Type result = Type::empty(size());

        const auto f = tan(m_data[0]);
        const auto da = f * f + 1;
        const auto daa = da * 2 * f;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type acos() const
    {
        using std::acos;
        using std::sqrt;

        Type result = Type::empty(size());

        const Scalar tmp = 1 - m_data[0] * m_data[0];

        const auto f = acos(m_data[0]);
        const auto da = -1 / sqrt(tmp);
        const auto daa = da * m_data[0] / tmp;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type asin() const
    {
        using std::asin;
        using std::sqrt;

        Type result = Type::empty(size());

        const Scalar tmp = 1 - m_data[0] * m_data[0];

        const auto f = asin(m_data[0]);
        const auto da = 1 / sqrt(tmp);
        const auto daa = da * m_data[0] / tmp;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type atan() const
    {
        using std::atan;

        Type result = Type::empty(size());

        const auto f = atan(m_data[0]);
        const auto da = 1 / (m_data[0] * m_data[0] + 1);
        const auto daa = -da * da * 2 * m_data[0];

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type atan2(const Type& b) const
    {
        using std::atan2;

        Type result = Type::empty(size());

        const Scalar tmp = m_data[0] * m_data[0] + b.m_data[0] * b.m_data[0];

        const auto f = atan2(m_data[0], b.m_data[0]);
        const auto da = b.m_data[0] / tmp;
        const auto db = -m_data[0] / tmp;
        const auto daa = db * da * 2;
        const auto dab = db * db - da * da;
        const auto dbb = -daa;

        binary<false>(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

        return result;
    }

    static Type hypot(const Type& a, const Type& b)
    {
        using std::hypot;
        
        check_equal_size(a.size(), b.size());

        Type result = Type::empty(a.size());

        const auto f = hypot(a.m_data[0], b.m_data[0]);
        const auto f3 = f * f * f;
        const auto da = a.m_data[0] / f;
        const auto db = b.m_data[0] / f;
        const auto daa = b.m_data[0] * b.m_data[0] / f3;
        const auto dab = -a.m_data[0] * b.m_data[0] / f3;
        const auto dbb = a.m_data[0] * a.m_data[0] / f3;

        a.binary<false>(a.m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

        return result;
    }

    static Type hypot(const Type& a, const Type& b, const Type& c)
    {
        using std::hypot;
        
        check_equal_size(a.size(), b.size());

        Type result = Type::empty(a.size());

        const auto f = hypot(a.m_data[0], b.m_data[0], c.m_data[0]);
        const auto f3 = f * f * f;
        const auto a2 = a.m_data[0] * a.m_data[0];
        const auto b2 = b.m_data[0] * b.m_data[0];
        const auto c2 = c.m_data[0] * c.m_data[0];
        const auto da = a.m_data[0] / f;
        const auto db = b.m_data[0] / f;
        const auto dc = c.m_data[0] / f;
        const auto daa = (b2 + c2) / f3;
        const auto dab = -(a.m_data[0] * b.m_data[0]) / f3;
        const auto dac = -(a.m_data[0] * c.m_data[0]) / f3;
        const auto dbb = (a2 + c2) / f3;
        const auto dbc = -(b.m_data[0] * c.m_data[0]) / f3;
        const auto dcc = (a2 + b2) / f3;

        a.ternary<false>(a.m_data, b.m_data, c.m_data, f, da, db, dc, daa, dab, dac, dbb, dbc, dcc, result.m_data);

        return result;
    }

    // --- hyperbolic functions

    Type cosh() const
    {
        using std::cosh;
        using std::sinh;

        Type result = Type::empty(size());

        const auto f = cosh(m_data[0]);
        const auto da = sinh(m_data[0]);
        const auto daa = f;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type sinh() const
    {
        using std::cosh;
        using std::sinh;

        Type result = Type::empty(size());

        const auto f = sinh(m_data[0]);
        const auto da = cosh(m_data[0]);
        const auto daa = f;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type tanh() const
    {
        using std::tanh;

        Type result = Type::empty(size());

        const auto f = tanh(m_data[0]);
        const auto da = 1 - f * f;
        const auto daa = -2 * f * da;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type acosh() const
    {
        using std::acosh;
        using std::sqrt;

        Type result = Type::empty(size());

        const auto f = acosh(m_data[0]);
        const auto da = 1 / (sqrt(m_data[0] - 1) * sqrt(m_data[0] + 1));
        const auto daa = -da * m_data[0] / ((m_data[0] - 1) * (m_data[0] + 1));

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type asinh() const
    {
        using std::asinh;
        using std::sqrt;

        Type result = Type::empty(size());

        const auto f = asinh(m_data[0]);
        const auto da = 1 / sqrt(1 + m_data[0] * m_data[0]);
        const auto daa = -da * m_data[0] / (1 + m_data[0] * m_data[0]);

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type atanh() const
    {
        using std::atanh;
        using std::pow;

        Type result = Type::empty(size());

        const auto f = atanh(m_data[0]);
        const auto da = 1 / (1 - m_data[0] * m_data[0]);
        const auto daa = 2 * m_data[0] / pow(m_data[0] * m_data[0] - 1, 2);

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    // exponents and logarithms

    Type exp() const
    {
        using std::exp;

        Type result = Type::empty(size());

        const auto f = exp(m_data[0]);
        const auto da = f;
        const auto daa = f;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type log() const
    {
        using std::log;

        Type result = Type::empty(size());

        const auto f = log(m_data[0]);
        const auto da = 1 / m_data[0];
        const auto daa = -da * da;

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type log(const TScalar base) const
    {
        using std::log;

        Type result = Type::empty(size());

        const auto f = log(m_data[0]) / log(base);
        const auto da = 1 / (m_data[0] * log(base));
        const auto daa = -da / m_data[0];

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type log2() const
    {
        using std::log;
        using std::log2;

        Type result = Type::empty(size());

        const auto f = log2(m_data[0]);
        const auto da = 1 / (m_data[0] * log(2));
        const auto daa = -da / m_data[0];

        unary<false>(m_data, f, da, daa, result.m_data);

        return result;
    }

    Type log10() const
    {
        using std::log;
        using std::log10;

        Type result = Type::empty(size());

        const auto f = log10(m_data[0]);
        const auto da = 1 / (m_data[0] * log(10));
        const auto daa = -da / m_data[0];

        unary<false>(m_data, f, da, daa, result.m_data);

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

// std::abs

using std::abs;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> abs(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.abs();
}

// std::pow

using std::pow;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> pow(const DDScalar<TOrder, TScalar, TSize>& a, const index b)
{
    return a.pow(b);
}

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> pow(const DDScalar<TOrder, TScalar, TSize>& a, const TScalar b)
{
    return a.pow(b);
}

// std::sqrt

using std::sqrt;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> sqrt(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.sqrt();
}

// std::cbrt

using std::cbrt;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> cbrt(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.cbrt();
}

// std::cos

using std::cos;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> cos(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.cos();
}

// std::sin

using std::sin;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> sin(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.sin();
}

// std::tan

using std::tan;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> tan(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.tan();
}

// std::acos

using std::acos;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> acos(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.acos();
}

// std::asin

using std::asin;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> asin(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.asin();
}

// std::atan

using std::atan;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> atan(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.atan();
}

// std::atan2

using std::atan2;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> atan2(const DDScalar<TOrder, TScalar, TSize>& a, const DDScalar<TOrder, TScalar, TSize>& b)
{
    return a.atan2(b);
}

// std::hypot

using std::hypot;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> hypot(const DDScalar<TOrder, TScalar, TSize>& a, const DDScalar<TOrder, TScalar, TSize>& b)
{
    return DDScalar<TOrder, TScalar, TSize>::hypot(a, b);
}

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> hypot(const DDScalar<TOrder, TScalar, TSize>& a, const DDScalar<TOrder, TScalar, TSize>& b, const DDScalar<TOrder, TScalar, TSize>& c)
{
    return DDScalar<TOrder, TScalar, TSize>::hypot(a, b, c);
}

// std::cosh

using std::cosh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> cosh(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.cosh();
}

// std::sinh

using std::sinh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> sinh(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.sinh();
}

// std::tanh

using std::tanh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> tanh(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.tanh();
}

// std::acosh

using std::acosh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> acosh(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.acosh();
}

// std::asin

using std::asinh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> asinh(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.asinh();
}

// std::atan

using std::atanh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> atanh(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.atanh();
}

// std::exp

using std::exp;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> exp(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.exp();
}

// std::log

using std::log;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> log(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.log();
}

// std::log2

using std::log2;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> log2(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.log2();
}

// std::log10

using std::log10;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> log10(const DDScalar<TOrder, TScalar, TSize>& a)
{
    return a.log10();
}

} // namespace hyperjet

#if defined EIGEN_WORLD_VERSION

namespace Eigen {

template <typename T>
struct NumTraits;

template <hyperjet::index TOrder, typename TScalar, std::ptrdiff_t TSize>
struct NumTraits<hyperjet::DDScalar<TOrder, TScalar, TSize>> : NumTraits<TScalar> {
    using Real = hyperjet::DDScalar<TOrder, TScalar, TSize>;
    using NonInteger = hyperjet::DDScalar<TOrder, TScalar, TSize>;
    using Nested = hyperjet::DDScalar<TOrder, TScalar, TSize>;

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

template <typename BinOp, hyperjet::index TOrder, typename TScalar, hyperjet::index TSize>
struct ScalarBinaryOpTraits<hyperjet::DDScalar<TOrder, TScalar, TSize>, TScalar, BinOp> {
    using ReturnType = hyperjet::DDScalar<TOrder, TScalar, TSize>;
};

template <typename BinOp, hyperjet::index TOrder, typename TScalar, hyperjet::index TSize>
struct ScalarBinaryOpTraits<TScalar, hyperjet::DDScalar<TOrder, TScalar, TSize>, BinOp> {
    using ReturnType = hyperjet::DDScalar<TOrder, TScalar, TSize>;
};

} // namespace Eigen

#endif
