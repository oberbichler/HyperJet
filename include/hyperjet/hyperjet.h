#pragma once

#include <algorithm> // copy, fill
#include <array>     // array
#include <assert.h>  // assert
#include <cmath> // acos, acosh, asin, asinh, atan, atanh, atan2, cbrt, cos, cosh, exp, hypot, log, log2, log10, pow, sin, sinh, sqrt, tan, tanh
#include <cstddef>          // ptrdiff_t
#include <initializer_list> // initializer_list
#include <ostream>          // ostream
#include <sstream>          // stringstream
#include <stdexcept>        // runtime_error
#include <string>           // string
#include <type_traits>      // conditional
#include <unordered_map>    // unordered_map
#include <utility>          // move, pair
#include <vector>           // vector

namespace hyperjet {

#if defined(_MSC_VER)
#define HYPERJET_INLINE __forceinline
#define HYPERJET_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define HYPERJET_INLINE __attribute__((always_inline)) inline
#define HYPERJET_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

using index = std::ptrdiff_t;

const index Dynamic = -1;

template <typename T> HYPERJET_INLINE index length(const T &container) {
  return static_cast<index>(container.size());
}

HYPERJET_INLINE constexpr bool throw_exceptions() {
#if defined(HYPERJET_NO_EXCEPTIONS)
  return false;
#else
  return true;
#endif
}

template <index TOrder, typename TScalar = double, index TSize = Dynamic>
class DDScalar {
  static constexpr index StaticDataSize =
      TOrder == 1 ? 1 + TSize : (TSize + 1) * (TSize + 2) / 2;
  static constexpr index DataSize = TSize < 0 ? Dynamic : StaticDataSize;

  using DynamicStorage = std::vector<TScalar>;
  using StaticStorage = std::array<TScalar, StaticDataSize>;

  // A statically sized scalar knows its size from TSize, so it carries no
  // field for it. The empty stand-in keeps m_size spellable in both variants.
  struct NoSize {};

public:
  using Type = DDScalar<TOrder, TScalar, TSize>;
  using Scalar = TScalar;
  using Data = typename std::conditional<TSize == Dynamic, DynamicStorage,
                                         StaticStorage>::type;

  using SizeStorage = std::conditional_t<TSize == Dynamic, index, NoSize>;

  HYPERJET_NO_UNIQUE_ADDRESS SizeStorage m_size;
  Data m_data;

public:
  HYPERJET_INLINE static index data_length_from_size(const index size) {
    return TOrder == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
  }

  HYPERJET_INLINE static index size_from_data_length(const index n) {
    const index s =
        TOrder == 1 ? n - 1 : static_cast<index>(std::sqrt(1 + 8 * n) - 3) / 2;

    if constexpr (throw_exceptions()) {
      if (s < 0 || data_length_from_size(s) != n) {
        throw std::runtime_error("Invalid length");
      }
    } else {
      assert(s >= 0 && data_length_from_size(s) == n && "Invalid length");
    }

    return s;
  }

private:
  HYPERJET_INLINE static void check_valid_size(const index size) {
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

  HYPERJET_INLINE static void check_valid_index(const index i,
                                                const index size) {
    if constexpr (throw_exceptions()) {
      if (i < 0 || size <= i) {
        throw std::runtime_error("Invalid index");
      }
    } else {
      assert(0 <= i && i < size && "Invalid index");
    }
  }

  HYPERJET_INLINE static void check_pad_size(const index size,
                                             const index new_size) {
    if constexpr (throw_exceptions()) {
      if (new_size < size) {
        throw std::runtime_error("Invalid new size");
      }
    } else {
      assert(new_size >= size && "Invalid new size");
    }
  }

  HYPERJET_INLINE static void check_equal_size(const index size_a,
                                               const index size_b) {
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
    static HYPERJET_INLINE Zero operator()(const Scalar) { return Zero(); }

    template <typename T> HYPERJET_INLINE T operator+(const T b) const {
      return b;
    }

    template <typename T>
    HYPERJET_INLINE friend T operator+(const T a, const Zero) {
      return a;
    }
  };

  struct MinusOne {
    static HYPERJET_INLINE Scalar operator()(const Scalar b) { return -b; }
  };

  struct One {
    HYPERJET_INLINE MinusOne operator-() const { return MinusOne(); }

    static HYPERJET_INLINE Scalar operator()(const Scalar b) { return b; }
  };

  // Filling the Hessian triangle in one pass halves its memory traffic, but it
  // also shortens the gradient pass to the gradient alone, and below a handful
  // of variables that one long vectorised pass is worth more than the traffic
  // saved. Measured crossover at second order, Apple M1 Pro, clang 21: n=6 is
  // 14% slower fused, n=8 3% faster, n=16 35% faster.
  //
  // size() is a compile-time constant for the static variants, so the branch
  // folds away there.
  static constexpr index hessian_fusion_threshold = 8;

  template <typename T>
  static constexpr bool is_tag_v =
      std::is_same_v<T, Zero> || std::is_same_v<T, One> ||
      std::is_same_v<T, MinusOne>;

  template <typename TCoeff>
  HYPERJET_INLINE static auto mul(const TCoeff &coeff,
                                  const Scalar &val) noexcept {
    if constexpr (is_tag_v<TCoeff>) {
      return TCoeff{}(val);
    } else {
      return coeff * val;
    }
  }

  // The chain rule term da * H_a has the same shape as the gradient term, so
  // one pass over the whole data array fills the Hessian slots as well. Only a
  // curvature term makes the triangle need a loop of its own.
  //
  // r never aliases a: the in-place operators copy their operand first, which
  // is what lets the triangle read a's gradient after r's has been written.
  template <typename TDa, typename TDaa>
  HYPERJET_INLINE void unary(const Data &a, const Scalar f, const TDa da,
                             const TDaa daa, Data &r) const noexcept {
    r[0] = f;

    if constexpr (TOrder < 1 || std::is_same_v<TDa, Zero>) {
      return;
    } else if constexpr (TOrder < 2 || std::is_same_v<TDaa, Zero>) {
      const index n = data_length();

      for (index i = 1; i < n; i++) {
        r[i] = mul(da, a[i]);
      }
    } else if (size() < hessian_fusion_threshold) {
      const index n = data_length();

      for (index i = 1; i < n; i++) {
        r[i] = mul(da, a[i]);
      }

      index k = 1 + size();

      for (index i = 0; i < size(); i++) {
        const auto ca = mul(daa, a[1 + i]);

        for (index j = i; j < size(); j++) {
          r[k++] += ca * a[1 + j];
        }
      }
    } else {
      for (index i = 1; i <= size(); i++) {
        r[i] = mul(da, a[i]);
      }

      index k = 1 + size();

      for (index i = 0; i < size(); i++) {
        const auto ca = mul(daa, a[1 + i]);

        for (index j = i; j < size(); j++) {
          r[k] = mul(da, a[k]) + ca * a[1 + j];
          k++;
        }
      }
    }
  }

  // See unary() for why the Hessian is one pass and why aliasing is not a
  // concern here.
  template <typename TDa, typename TDb, typename TDaa, typename TDab,
            typename TDbb>
  HYPERJET_INLINE void binary(const Data &a, const Data &b, const Scalar f,
                              const TDa da, const TDb db, const TDaa daa,
                              const TDab dab, const TDbb dbb,
                              Data &r) const noexcept {
    r[0] = f;

    if constexpr (TOrder < 1 ||
                  (std::is_same_v<TDa, Zero> && std::is_same_v<TDb, Zero>)) {
      return;
    } else if constexpr (TOrder < 2 || (std::is_same_v<TDaa, Zero> &&
                                        std::is_same_v<TDab, Zero> &&
                                        std::is_same_v<TDbb, Zero>)) {
      const index n = data_length();

      for (index i = 1; i < n; i++) {
        r[i] = mul(da, a[i]) + mul(db, b[i]);
      }
    } else if (size() < hessian_fusion_threshold) {
      const index n = data_length();

      for (index i = 1; i < n; i++) {
        r[i] = mul(da, a[i]) + mul(db, b[i]);
      }

      index k = 1 + size();

      for (index i = 0; i < size(); i++) {
        const auto ca = mul(daa, a[1 + i]) + mul(dab, b[1 + i]);
        const auto cb = mul(dab, a[1 + i]) + mul(dbb, b[1 + i]);

        for (index j = i; j < size(); j++) {
          r[k++] += ca * a[1 + j] + cb * b[1 + j];
        }
      }
    } else {
      for (index i = 1; i <= size(); i++) {
        r[i] = mul(da, a[i]) + mul(db, b[i]);
      }

      index k = 1 + size();

      for (index i = 0; i < size(); i++) {
        const auto ca = mul(daa, a[1 + i]) + mul(dab, b[1 + i]);
        const auto cb = mul(dab, a[1 + i]) + mul(dbb, b[1 + i]);

        for (index j = i; j < size(); j++) {
          r[k] = mul(da, a[k]) + mul(db, b[k]) + ca * a[1 + j] + cb * b[1 + j];
          k++;
        }
      }
    }
  }

  // See unary() for why the Hessian is one pass and why aliasing is not a
  // concern here.
  template <typename TDa, typename TDb, typename TDc, typename TDaa,
            typename TDab, typename TDac, typename TDbb, typename TDbc,
            typename TDcc>
  HYPERJET_INLINE void ternary(const Data &a, const Data &b, const Data &c,
                               const Scalar f, const TDa da, const TDb db,
                               const TDc dc, const TDaa daa, const TDab dab,
                               const TDac dac, const TDbb dbb, const TDbc dbc,
                               const TDcc dcc, Data &r) const noexcept {
    r[0] = f;

    if constexpr (TOrder < 1 ||
                  (std::is_same_v<TDa, Zero> && std::is_same_v<TDb, Zero> &&
                   std::is_same_v<TDc, Zero>)) {
      return;
    } else if constexpr (TOrder < 2 || (std::is_same_v<TDaa, Zero> &&
                                        std::is_same_v<TDab, Zero> &&
                                        std::is_same_v<TDac, Zero> &&
                                        std::is_same_v<TDbb, Zero> &&
                                        std::is_same_v<TDbc, Zero> &&
                                        std::is_same_v<TDcc, Zero>)) {
      const index n = data_length();

      for (index i = 1; i < n; i++) {
        r[i] = mul(da, a[i]) + mul(db, b[i]) + mul(dc, c[i]);
      }
    } else if (size() < hessian_fusion_threshold) {
      const index n = data_length();

      for (index i = 1; i < n; i++) {
        r[i] = mul(da, a[i]) + mul(db, b[i]) + mul(dc, c[i]);
      }

      index k = 1 + size();

      for (index i = 0; i < size(); i++) {
        const auto ca =
            mul(daa, a[1 + i]) + mul(dab, b[1 + i]) + mul(dac, c[1 + i]);
        const auto cb =
            mul(dab, a[1 + i]) + mul(dbb, b[1 + i]) + mul(dbc, c[1 + i]);
        const auto cc =
            mul(dac, a[1 + i]) + mul(dbc, b[1 + i]) + mul(dcc, c[1 + i]);

        for (index j = i; j < size(); j++) {
          r[k++] += ca * a[1 + j] + cb * b[1 + j] + cc * c[1 + j];
        }
      }
    } else {
      for (index i = 1; i <= size(); i++) {
        r[i] = mul(da, a[i]) + mul(db, b[i]) + mul(dc, c[i]);
      }

      index k = 1 + size();

      for (index i = 0; i < size(); i++) {
        const auto ca =
            mul(daa, a[1 + i]) + mul(dab, b[1 + i]) + mul(dac, c[1 + i]);
        const auto cb =
            mul(dab, a[1 + i]) + mul(dbb, b[1 + i]) + mul(dbc, c[1 + i]);
        const auto cc =
            mul(dac, a[1 + i]) + mul(dbc, b[1 + i]) + mul(dcc, c[1 + i]);

        for (index j = i; j < size(); j++) {
          r[k] = mul(da, a[k]) + mul(db, b[k]) + mul(dc, c[k]) + ca * a[1 + j] +
                 cb * b[1 + j] + cc * c[1 + j];
          k++;
        }
      }
    }
  }

public:
  DDScalar() {
    static_assert(0 < order() && order() <= 2);

    if constexpr (is_dynamic()) {
      m_size = 1;
      m_data = Data(1);
    } else {
      static_assert(TSize >= 0);
    }
  }

  DDScalar(const TScalar f) {
    static_assert(0 < order() && order() <= 2);

    if constexpr (is_dynamic()) {
      m_size = 1;
      m_data = Data(1);
    } else {
      static_assert(TSize >= 0);
    }
    this->f() = f;
    std::fill(m_data.begin() + 1, m_data.end(), 0);
  }

  DDScalar(const Data &data) : m_data(data) {
    static_assert(0 < order() && order() <= 2);

    static_assert(!is_dynamic());
  }

  // Takes ownership instead of copying. The factories build their data as a
  // temporary, and for the dynamic variant copying it in meant a second
  // allocation and a second pass over the whole array.
  DDScalar(Data &&data, const index size)
      : m_size(size), m_data(std::move(data)) {
    static_assert(0 < order() && order() <= 2);

    static_assert(is_dynamic());
  }

  DDScalar(const Data &data, const index size) : m_size(size), m_data(data) {
    static_assert(0 < order() && order() <= 2);

    static_assert(is_dynamic());
  }

  DDScalar(std::initializer_list<TScalar> data) {
    static_assert(0 < order() && order() <= 2);

    const index s = size_from_data_length(length(data));

    check_valid_size(s);

    if constexpr (is_dynamic()) {
      m_size = s;
      m_data.assign(data.begin(), data.end());
    } else {
      std::copy(data.begin(), data.end(), m_data.begin());
    }
  }

  static constexpr index order() { return TOrder; }

  // The object parameter is a forwarding reference so that a result can be
  // read straight from the expression that produced it: an lvalue reference
  // would not bind to a temporary, and (a * b).f() would not compile. As with
  // std::vector::operator[], a reference obtained from a temporary is only
  // valid within the full expression.
  auto &data(this auto &&self) { return self.m_data; }

  auto ptr(this auto &&self) { return self.m_data.data(); }

  static constexpr index static_size() { return TSize; }

  constexpr index size() const {
    if constexpr (is_dynamic()) {
      return m_size;
    } else {
      return TSize;
    }
  }

  constexpr index data_length() const {
    if constexpr (is_dynamic()) {
      return length(m_data);
    } else {
      return StaticDataSize;
    }
  }

  Type pad_left(const index new_size) const {
    static_assert(is_dynamic());

    check_pad_size(size(), new_size);

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

  Type pad_right(const index new_size) const {
    static_assert(is_dynamic());

    check_pad_size(size(), new_size);

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

  static constexpr bool is_dynamic() { return TSize == Dynamic; }

  static Type create(const Data &data) {
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

  HYPERJET_INLINE static Type empty() {
    if constexpr (is_dynamic()) {
      // Size zero, not the default constructor's size one.
      return Type(Data(1), 0);
    } else {
      // The default constructor already leaves m_data indeterminate, which is
      // what empty() means. Building a local Data and copying it in was both a
      // wasted pass over the whole array -- 2.6 kB at order 2 with 24
      // variables -- and a read of indeterminate doubles, which is undefined.
      return Type();
    }
  }

  static Type empty(const index size) {
    if constexpr (is_dynamic()) {
      check_valid_size(size);
      return Type(Data(data_length_from_size(size)), size);
    } else {
      check_valid_size(size);
      return empty();
    }
  }

  HYPERJET_INLINE static Type zero() {
    if constexpr (is_dynamic()) {
      return Type(Data(1), 0);
    } else {
      // Filling in place rather than filling a local and copying it in: the
      // copy was a second pass over the array, and variables() pays it once per
      // variable.
      Type result;
      result.m_data.fill(0);
      return result;
    }
  }

  static Type zero(const index size) {
    if constexpr (is_dynamic()) {
      check_valid_size(size);
      return Type(Data(data_length_from_size(size), 0), size);
    } else {
      check_valid_size(size);
      return zero();
    }
  }

  static Type constant(const Scalar f) {
    Type result = zero();
    result.f() = f;
    return result;
  }

  static Type constant(const Scalar f, const index size) {
    if constexpr (is_dynamic()) {
      Type result = zero(size);
      result.f() = f;
      return result;
    } else {
      check_valid_size(size);
      return constant(f);
    }
  }

  static Type variable(const index i, const Scalar f) {
    static_assert(!is_dynamic());
    check_valid_index(i, TSize);
    Type result = zero();
    result.f() = f;
    result.g(i) = 1;
    return result;
  }

  static Type variable(const index i, const Scalar f, const index size) {
    if constexpr (is_dynamic()) {
      Type result = zero(size);
      check_valid_index(i, size);
      result.f() = f;
      result.g(i) = 1;
      return result;
    } else {
      check_valid_size(size);
      return variable(i, f);
    }
  }

  static std::vector<Type> variables(const std::vector<Scalar> &values) {
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

  // std::array is sized by std::size_t, so T has to be as well or the size
  // can never be deduced from the argument.
  template <std::size_t T>
  static std::conditional_t<TSize == Dynamic, std::vector<Type>,
                            std::array<Type, T>>
  variables(const std::array<Scalar, T> &values) {
    if constexpr (!is_dynamic()) {
      static_assert(index(T) == TSize);
    }

    const index s = length(values);

    if constexpr (is_dynamic()) {
      std::vector<Type> vars(s);
      for (index i = 0; i < s; i++) {
        vars[i] = variable(i, values[i], s);
      }
      return vars;
    } else {
      std::array<Type, T> vars;
      for (index i = 0; i < s; i++) {
        vars[i] = variable(i, values[i], s);
      }
      return vars;
    }
  }

  auto &f(this auto &&self) { return self.m_data[0]; }

  void set_f(const Scalar value) { m_data[0] = value; }

  auto &g(this auto &&self, const index i) {
    assert(0 <= i && i < self.size());

    return self.m_data[1 + i];
  }

  void set_g(const index i, const Scalar value) { g(i) = value; }

  auto &h(this auto &&self, const index i)
    requires(order() == 2)
  {
    assert(0 <= i && i < self.size() * (self.size() + 1) / 2);

    return self.m_data[1 + self.size() + i];
  }

  void set_h(const index i, const Scalar value)
    requires(order() == 2)
  {
    h(i) = value;
  }

  auto &h(this auto &&self, const index i, const index j)
    requires(order() == 2)
  {
    assert(0 <= i && i < self.size());
    assert(0 <= j && j < self.size());

    if (i < j) {
      return self
          .m_data[1 + self.size() + (2 * self.size() - 1 - i) * i / 2 + j];
    } else {
      return self
          .m_data[1 + self.size() + (2 * self.size() - 1 - j) * j / 2 + i];
    }
  }

  void set_h(const index i, const index j, const Scalar value)
    requires(order() == 2)
  {
    h(i, j) = value;
  }

#if defined EIGEN_WORLD_VERSION
  using Vector = Eigen::Matrix<TScalar, 1, TSize>;
  using Matrix = Eigen::Matrix<TScalar, TSize, TSize>;
  using DataVector = Eigen::Matrix<TScalar, 1, DataSize>;

  Eigen::Ref<const Vector> ag() const {
    return Eigen::Map<Vector>(ptr() + 1, size());
  }

  Eigen::Ref<Vector> ag() { return Eigen::Map<Vector>(ptr() + 1, size()); }

  Matrix hm(const std::string mode) const
    requires(order() == 2)
  {
    Matrix result(size(), size());

    hm(mode, result);

    return result;
  }

  void hm(const std::string mode, Eigen::Ref<Matrix> out) const
    requires(order() == 2)
  {
    check_equal_size(size(), out.rows());
    check_equal_size(size(), out.cols());

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

  void set_hm(const Eigen::Ref<const Matrix> &value)
    requires(order() == 2)
  {
    check_equal_size(size(), value.rows());
    check_equal_size(size(), value.cols());

    index it = 0;

    for (index i = 0; i < size(); i++) {
      for (index j = i; j < size(); j++) {
        h(it++) = value(i, j);
      }
    }
  }

  Eigen::Ref<DataVector> adata() {
    return Eigen::Map<DataVector>(ptr(), length(m_data));
  }

  void set_adata(const Eigen::Ref<DataVector> value) {
    Eigen::Map<DataVector>(ptr(), length(m_data)) = value;
  }

  static Type from_gradient(const TScalar f,
                            const Eigen::Ref<const Vector> &g) {
    static_assert(order() == 1);

    Type result = empty(length(g));

    result.f() = f;
    result.ag() = g;

    return result;
  }

  static Type from_arrays(const TScalar f, const Eigen::Ref<const Vector> &g,
                          const Eigen::Ref<const Matrix> &hm) {
    static_assert(order() == 2);

    assert(g.size() == hm.rows() && g.size() == hm.cols());

    Type result = empty(length(g));

    result.f() = f;
    result.ag() = g;
    result.set_hm(hm);

    return result;
  }

  Scalar eval(typename std::conditional < TSize == Dynamic, std::vector<Scalar>,
              std::array<Scalar, TSize<0 ? 0 : TSize>>::type d) const {
    check_equal_size(size(), length(d));

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

  void set_zero() { std::fill(m_data.begin(), m_data.end(), 0); }

  friend std::ostream &operator<<(std::ostream &out, const Type &value) {
    out << value.f() << "hj";
    return out;
  }

  std::string to_string() {
    std::stringstream output;

    output << *this;

    return output.str();
  }

  // --- neg

  HYPERJET_INLINE Type operator-() const {
    Type result = Type::empty(size());

    for (index i = 0; i < data_length(); i++) {
      result.m_data[i] = -m_data[i];
    }

    return result;
  }

  // --- add

  HYPERJET_INLINE Type operator+(const Type &b) const {
    check_equal_size(size(), b.size());

    Type result = Type::empty(size());

    const auto f = this->f() + b.f();

    const auto da = One();
    const auto db = One();

    const auto daa = Zero();
    const auto dab = Zero();
    const auto dbb = Zero();

    binary(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type operator+(const Scalar b) const {
    Type result = *this;

    result.f() += b;

    return result;
  }

  friend Type operator+(const Scalar a, const Type &b) { return b + a; }

  HYPERJET_INLINE Type &operator+=(const Type &b) {
    check_equal_size(size(), b.size());

    for (index i = 0; i < length(m_data); i++) {
      m_data[i] += b.m_data[i];
    }

    return *this;
  }

  HYPERJET_INLINE Type &operator+=(const Scalar &b) {
    f() += b;

    return *this;
  }

  // --- sub

  HYPERJET_INLINE Type operator-(const Type &b) const {
    check_equal_size(size(), b.size());

    Type result = Type::empty(size());

    const auto f = this->f() - b.f();

    const auto da = One();
    const auto db = -One();

    const auto daa = Zero();
    const auto dab = Zero();
    const auto dbb = Zero();

    binary(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type operator-(const Scalar b) const { return -b + *this; }

  friend Type operator-(const Scalar a, const Type &b) {
    Type result = Type::empty(b.size());

    for (index i = 0; i < result.data_length(); i++) {
      result.m_data[i] = -b.m_data[i];
    }

    result.f() += a;

    return result;
  }

  HYPERJET_INLINE Type &operator-=(const Type &b) {
    check_equal_size(size(), b.size());

    for (index i = 0; i < length(m_data); i++) {
      m_data[i] -= b.m_data[i];
    }

    return *this;
  }

  HYPERJET_INLINE Type &operator-=(const Scalar &b) {
    f() -= b;

    return *this;
  }

  // --- mul

  HYPERJET_INLINE Type operator*(const Type &b) const {
    check_equal_size(size(), b.size());

    Type result = Type::empty(size());

    const auto f = this->f() * b.f();

    const auto da = b.f();
    const auto db = this->f();

    const auto daa = Zero();
    const auto dab = One();
    const auto dbb = Zero();

    binary(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type operator*(const Scalar b) const {
    Type result = Type::empty(size());

    for (index i = 0; i < data_length(); i++) {
      result.m_data[i] = m_data[i] * b;
    }

    return result;
  }

  friend Type operator*(const Scalar a, const Type &b) { return b * a; }

  HYPERJET_INLINE Type &operator*=(const Type &b) {
    check_equal_size(size(), b.size());

    // aliased operands would be read again after being overwritten below
    if (this == &b) {
      return *this = *this * b;
    }

    const Data a_m_data = m_data;

    const Scalar da = b.f();
    const Scalar db = this->f();

    f() *= b.f();

    for (index i = 1; i < length(m_data); i++) {
      m_data[i] = da * m_data[i] + db * b.m_data[i];
    }

    if constexpr (order() == 1)
      return *this;

    auto *it = &m_data[1 + size()];

    for (index i = 0; i < size(); i++) {
      for (index j = i; j < size(); j++) {
        *it++ += a_m_data[1 + i] * b.m_data[1 + j] +
                 a_m_data[1 + j] * b.m_data[1 + i];
      }
    }

    return *this;
  }

  HYPERJET_INLINE Type &operator*=(const Scalar &b) {
    for (index i = 0; i < length(m_data); i++) {
      m_data[i] = m_data[i] * b;
    }

    return *this;
  }

  // --- div

  HYPERJET_INLINE Type operator/(const Type &b) const {
    using std::pow;

    check_equal_size(size(), b.size());

    const Scalar tmp = 1 / b.f();

    const auto f = this->f() * tmp;
    const auto da = tmp;
    const auto db = -this->f() / pow(b.f(), 2);
    const auto daa = Zero();
    const auto dab = -1 / pow(b.f(), 2);
    const auto dbb = 2 * this->f() / pow(b.f(), 3);

    Type result = Type::empty(size());

    binary(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type operator/(const Scalar b) const {
    return Scalar(1) / b * (*this);
  }

  friend Type operator/(const Scalar a, const Type &b) {
    using std::pow;

    Type result = Type::empty(b.size());

    const auto f = a / b.f();
    const auto db = -a / pow(b.f(), 2);
    const auto dbb = 2 * a / pow(b.f(), 3);

    b.unary(b.m_data, f, db, dbb, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type &operator/=(const Type &b) {
    using std::pow;

    check_equal_size(size(), b.size());

    // aliased operands would be read again after being overwritten below
    if (this == &b) {
      return *this = *this / b;
    }

    const Data a_m_data = m_data;

    const auto f = a_m_data[0] / b.f();
    const auto da = 1 / b.f();
    const auto db = -a_m_data[0] / pow(b.f(), 2);
    const auto daa = Zero();
    const auto dab = -1 / pow(b.f(), 2);
    const auto dbb = 2 * a_m_data[0] / pow(b.f(), 3);

    binary(a_m_data, b.m_data, f, da, db, daa, dab, dbb, m_data);

    return *this;
  }

  HYPERJET_INLINE Type &operator/=(const Scalar &b) {
    operator*=(1 / b);

    return *this;
  }

  // --- arithmetic operations

  HYPERJET_INLINE Type pow(const double b) const {
    using std::pow;

    Type result = Type::empty(size());

    const auto f = pow(this->f(), b);
    const auto da = b * pow(this->f(), b - 1);
    const auto daa = (b - 1) * b * pow(this->f(), b - 2);

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type sqrt() const {
    using std::pow;
    using std::sqrt;

    Type result = Type::empty(size());

    const auto f = sqrt(this->f());
    const auto da = 1 / (2 * f);
    const auto daa = -da / (2 * this->f());

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type cbrt() const {
    using std::cbrt;

    Type result = Type::empty(size());

    const auto f = cbrt(this->f());
    const auto da = 1 / (3 * f * f);
    const auto daa = -da * 2 / (3 * this->f());

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type reciprocal() const {
    Type result = Type::empty(size());

    const auto f = 1 / this->f();
    const auto da = -f * f;
    const auto daa = -2 * f * da;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  // --- trigonometric functions

  HYPERJET_INLINE Type cos() const {
    using std::cos;
    using std::sin;

    Type result = Type::empty(size());

    const auto f = cos(this->f());
    const auto da = -sin(this->f());
    const auto daa = -cos(this->f());

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type sin() const {
    using std::cos;
    using std::sin;

    Type result = Type::empty(size());

    const auto f = sin(this->f());
    const auto da = cos(this->f());
    const auto daa = -sin(this->f());

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type tan() const {
    using std::tan;

    Type result = Type::empty(size());

    const auto f = tan(this->f());
    const auto da = f * f + 1;
    const auto daa = da * 2 * f;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type acos() const {
    using std::acos;
    using std::sqrt;

    Type result = Type::empty(size());

    const Scalar tmp = 1 - this->f() * this->f();

    const auto f = acos(this->f());
    const auto da = -1 / sqrt(tmp);
    const auto daa = da * this->f() / tmp;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type asin() const {
    using std::asin;
    using std::sqrt;

    Type result = Type::empty(size());

    const Scalar tmp = 1 - this->f() * this->f();

    const auto f = asin(this->f());
    const auto da = 1 / sqrt(tmp);
    const auto daa = da * this->f() / tmp;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type atan() const {
    using std::atan;

    Type result = Type::empty(size());

    const auto f = atan(this->f());
    const auto da = 1 / (this->f() * this->f() + 1);
    const auto daa = -da * da * 2 * this->f();

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type atan2(const Type &b) const {
    using std::atan2;

    Type result = Type::empty(size());

    const Scalar tmp = this->f() * this->f() + b.f() * b.f();

    const auto f = atan2(this->f(), b.f());
    const auto da = b.f() / tmp;
    const auto db = -this->f() / tmp;
    const auto daa = db * da * 2;
    const auto dab = db * db - da * da;
    const auto dbb = -daa;

    binary(m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

    return result;
  }

  static Type hypot(const Type &a, const Type &b) {
    using std::hypot;

    check_equal_size(a.size(), b.size());

    Type result = Type::empty(a.size());

    const auto f = hypot(a.f(), b.f());
    const auto f3 = f * f * f;
    const auto da = a.f() / f;
    const auto db = b.f() / f;
    const auto daa = b.f() * b.f() / f3;
    const auto dab = -a.f() * b.f() / f3;
    const auto dbb = a.f() * a.f() / f3;

    a.binary(a.m_data, b.m_data, f, da, db, daa, dab, dbb, result.m_data);

    return result;
  }

  static Type hypot(const Type &a, const Type &b, const Type &c) {
    using std::hypot;

    check_equal_size(a.size(), b.size());

    Type result = Type::empty(a.size());

    const auto f = hypot(a.f(), b.f(), c.f());
    const auto f3 = f * f * f;
    const auto a2 = a.f() * a.f();
    const auto b2 = b.f() * b.f();
    const auto c2 = c.f() * c.f();
    const auto da = a.f() / f;
    const auto db = b.f() / f;
    const auto dc = c.f() / f;
    const auto daa = (b2 + c2) / f3;
    const auto dab = -(a.f() * b.f()) / f3;
    const auto dac = -(a.f() * c.f()) / f3;
    const auto dbb = (a2 + c2) / f3;
    const auto dbc = -(b.f() * c.f()) / f3;
    const auto dcc = (a2 + b2) / f3;

    a.ternary(a.m_data, b.m_data, c.m_data, f, da, db, dc, daa, dab, dac, dbb,
              dbc, dcc, result.m_data);

    return result;
  }

  // --- hyperbolic functions

  HYPERJET_INLINE Type cosh() const {
    using std::cosh;
    using std::sinh;

    Type result = Type::empty(size());

    const auto f = cosh(this->f());
    const auto da = sinh(this->f());
    const auto daa = f;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type sinh() const {
    using std::cosh;
    using std::sinh;

    Type result = Type::empty(size());

    const auto f = sinh(this->f());
    const auto da = cosh(this->f());
    const auto daa = f;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type tanh() const {
    using std::tanh;

    Type result = Type::empty(size());

    const auto f = tanh(this->f());
    const auto da = 1 - f * f;
    const auto daa = -2 * f * da;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type acosh() const {
    using std::acosh;
    using std::sqrt;

    Type result = Type::empty(size());

    const auto f = acosh(this->f());
    const auto da = 1 / (sqrt(this->f() - 1) * sqrt(this->f() + 1));
    const auto daa = -da * this->f() / ((this->f() - 1) * (this->f() + 1));

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type asinh() const {
    using std::asinh;
    using std::sqrt;

    Type result = Type::empty(size());

    const auto f = asinh(this->f());
    const auto da = 1 / sqrt(1 + this->f() * this->f());
    const auto daa = -da * this->f() / (1 + this->f() * this->f());

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type atanh() const {
    using std::atanh;
    using std::pow;

    Type result = Type::empty(size());

    const auto f = atanh(this->f());
    const auto da = 1 / (1 - this->f() * this->f());
    const auto daa = 2 * this->f() / pow(this->f() * this->f() - 1, 2);

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  // exponents and logarithms

  HYPERJET_INLINE Type exp() const {
    using std::exp;

    Type result = Type::empty(size());

    const auto f = exp(this->f());
    const auto da = f;
    const auto daa = f;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type log() const {
    using std::log;

    Type result = Type::empty(size());

    const auto f = log(this->f());
    const auto da = 1 / this->f();
    const auto daa = -da * da;

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type log(const TScalar base) const {
    using std::log;

    Type result = Type::empty(size());

    const auto f = log(this->f()) / log(base);
    const auto da = 1 / (this->f() * log(base));
    const auto daa = -da / this->f();

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type log2() const {
    using std::log;
    using std::log2;

    Type result = Type::empty(size());

    const auto f = log2(this->f());
    const auto da = 1 / (this->f() * log(2));
    const auto daa = -da / this->f();

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  HYPERJET_INLINE Type log10() const {
    using std::log;
    using std::log10;

    Type result = Type::empty(size());

    const auto f = log10(this->f());
    const auto da = 1 / (this->f() * log(10));
    const auto daa = -da / this->f();

    unary(m_data, f, da, daa, result.m_data);

    return result;
  }

  // abs

  HYPERJET_INLINE Type abs() const { return f() < 0 ? -(*this) : *this; }

  // comparison

  bool operator==(const Type &b) const { return f() == b.f(); }

  bool operator!=(const Type &b) const { return f() != b.f(); }

  bool operator<(const Type &b) const { return f() < b.f(); }

  bool operator>(const Type &b) const { return f() > b.f(); }

  bool operator<=(const Type &b) const { return f() <= b.f(); }

  bool operator>=(const Type &b) const { return f() >= b.f(); }

  bool operator==(const Scalar b) const { return f() == b; }

  bool operator!=(const Scalar b) const { return f() != b; }

  bool operator<(const Scalar b) const { return f() < b; }

  bool operator>(const Scalar b) const { return f() > b; }

  bool operator<=(const Scalar b) const { return f() <= b; }

  bool operator>=(const Scalar b) const { return f() >= b; }

  friend bool operator==(const Scalar a, const Type &b) {
    return b.operator==(a);
  }

  friend bool operator!=(const Scalar a, const Type &b) {
    return b.operator!=(a);
  }

  friend bool operator<(const Scalar a, const Type &b) {
    return b.operator>(a);
  }

  friend bool operator>(const Scalar a, const Type &b) {
    return b.operator<(a);
  }

  friend bool operator<=(const Scalar a, const Type &b) {
    return b.operator>=(a);
  }

  friend bool operator>=(const Scalar a, const Type &b) {
    return b.operator<=(a);
  }
};

// std::abs

using std::abs;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
abs(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.abs();
}

// std::pow

using std::pow;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> pow(const DDScalar<TOrder, TScalar, TSize> &a,
                                     const index b) {
  return a.pow(b);
}

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize> pow(const DDScalar<TOrder, TScalar, TSize> &a,
                                     const double b) {
  return a.pow(b);
}

// std::sqrt

using std::sqrt;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
sqrt(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.sqrt();
}

// std::cbrt

using std::cbrt;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
cbrt(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.cbrt();
}

// std::cos

using std::cos;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
cos(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.cos();
}

// std::sin

using std::sin;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
sin(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.sin();
}

// std::tan

using std::tan;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
tan(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.tan();
}

// std::acos

using std::acos;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
acos(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.acos();
}

// std::asin

using std::asin;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
asin(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.asin();
}

// std::atan

using std::atan;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
atan(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.atan();
}

// std::atan2

using std::atan2;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
atan2(const DDScalar<TOrder, TScalar, TSize> &a,
      const DDScalar<TOrder, TScalar, TSize> &b) {
  return a.atan2(b);
}

// std::hypot

using std::hypot;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
hypot(const DDScalar<TOrder, TScalar, TSize> &a,
      const DDScalar<TOrder, TScalar, TSize> &b) {
  return DDScalar<TOrder, TScalar, TSize>::hypot(a, b);
}

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
hypot(const DDScalar<TOrder, TScalar, TSize> &a,
      const DDScalar<TOrder, TScalar, TSize> &b,
      const DDScalar<TOrder, TScalar, TSize> &c) {
  return DDScalar<TOrder, TScalar, TSize>::hypot(a, b, c);
}

// std::cosh

using std::cosh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
cosh(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.cosh();
}

// std::sinh

using std::sinh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
sinh(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.sinh();
}

// std::tanh

using std::tanh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
tanh(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.tanh();
}

// std::acosh

using std::acosh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
acosh(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.acosh();
}

// std::asin

using std::asinh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
asinh(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.asinh();
}

// std::atan

using std::atanh;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
atanh(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.atanh();
}

// std::exp

using std::exp;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
exp(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.exp();
}

// std::log

using std::log;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
log(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.log();
}

// std::log2

using std::log2;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
log2(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.log2();
}

// std::log10

using std::log10;

template <index TOrder, typename TScalar, index TSize>
DDScalar<TOrder, TScalar, TSize>
log10(const DDScalar<TOrder, TScalar, TSize> &a) {
  return a.log10();
}

// Derivatives keyed by name rather than by index. The variable set does not
// have to be known in advance: an operation takes the union of the names of its
// operands. Second order is not implemented yet, hence the constraint.
template <index TOrder, typename TScalar = double>
  requires(0 < TOrder && TOrder <= 2)
class SScalar {
public: // types
  using Type = SScalar<TOrder, TScalar>;
  using Scalar = TScalar;

  // The interface type for construction and eval. A map is what callers, and
  // in particular a Python dict, naturally provide.
  using Data = std::unordered_map<std::string, TScalar>;

  static constexpr index order() { return TOrder; }

private: // types
  // The storage. Sorted by name, so a combination of two values is a linear
  // merge rather than a sequence of hash lookups, and the position of a name is
  // its index -- which is what a second-order Hessian would be laid out over.
  using Terms = std::vector<std::pair<std::string, TScalar>>;

  // The Hessian is dense over the names the value carries, laid out as the
  // upper triangle in the same packing DDScalar uses. It is not sparse within
  // that set: any nonlinear function contributes the full outer product of its
  // own gradient, so the triangle fills up after a couple of operations. What
  // stays sparse is the set of names itself.
  using Hessian = std::vector<TScalar>;

  struct NoHessian {};

  using HessianStorage = std::conditional_t<TOrder == 2, Hessian, NoHessian>;

private: // variables
  // The default constructor is defaulted, so without an initializer m_f would
  // be indeterminate and reading it undefined.
  Scalar m_f{};
  Terms m_d;
  HYPERJET_NO_UNIQUE_ADDRESS HessianStorage m_h;

private: // methods
  HYPERJET_INLINE static index hessian_length(const index k) {
    return k * (k + 1) / 2;
  }

  // Position of (i, j) in the upper triangle over k names.
  HYPERJET_INLINE static index at(const index k, index i, index j) {
    if (j < i) {
      std::swap(i, j);
    }

    return (2 * k - 1 - i) * i / 2 + j;
  }

  // Position of each requested name in the storage, or -1 where the value does
  // not carry it. One lookup per name, so a projection costs k log n rather
  // than a lookup per matrix entry.
  std::vector<index> positions(const std::vector<std::string> &names) const {
    std::vector<index> result;
    result.reserve(names.size());

    for (const auto &name : names) {
      const auto it = std::lower_bound(m_d.begin(), m_d.end(), name, before);

      result.push_back(it != m_d.end() && it->first == name
                           ? static_cast<index>(it - m_d.begin())
                           : -1);
    }

    return result;
  }

  HYPERJET_INLINE static bool before(const std::pair<std::string, TScalar> &e,
                                     const std::string &name) {
    return e.first < name;
  }

  // r = phi(f): g_r = da * g, H_r = da * H + daa * g (x) g. The names do not
  // change, so nothing has to be merged.
  template <typename TDa, typename TDaa>
  HYPERJET_INLINE SScalar apply(const Scalar f, const TDa &da,
                                const TDaa &daa) const {
    SScalar r;
    r.m_f = f;
    r.m_d = m_d;

    for (auto &d : r.m_d) {
      d.second = da * d.second;
    }

    if constexpr (TOrder == 2) {
      const index k = length(m_d);
      r.m_h.resize(hessian_length(k));

      index p = 0;

      for (index i = 0; i < k; i++) {
        const TScalar ca = daa * m_d[i].second;

        for (index j = i; j < k; j++) {
          r.m_h[p] = da * m_h[p] + ca * m_d[j].second;
          p++;
        }
      }
    }

    return r;
  }

  // Merges the names of a and b in one pass, leaving the derivatives at zero.
  // At second order it also records where each operand's names land in the
  // union, which is what its Hessian has to be scattered along.
  HYPERJET_INLINE static void merge_names(const Terms &a, const Terms &b,
                                          Terms &out, std::vector<index> &pa,
                                          std::vector<index> &pb) {
    const index ka = length(a);
    const index kb = length(b);

    out.reserve(ka + kb);
    pa.reserve(ka);
    pb.reserve(kb);

    index i = 0;
    index j = 0;

    while (i < ka && j < kb) {
      if (a[i].first < b[j].first) {
        pa.push_back(length(out));
        out.emplace_back(a[i].first, TScalar(0));
        i++;
      } else if (b[j].first < a[i].first) {
        pb.push_back(length(out));
        out.emplace_back(b[j].first, TScalar(0));
        j++;
      } else {
        pa.push_back(length(out));
        pb.push_back(length(out));
        out.emplace_back(a[i].first, TScalar(0));
        i++;
        j++;
      }
    }

    for (; i < ka; i++) {
      pa.push_back(length(out));
      out.emplace_back(a[i].first, TScalar(0));
    }

    for (; j < kb; j++) {
      pb.push_back(length(out));
      out.emplace_back(b[j].first, TScalar(0));
    }
  }

  // Scatters da * h, which is laid out over p.size() names, into the union.
  template <typename TDa>
  HYPERJET_INLINE static void
  scatter(Hessian &out, const index k, const Hessian &h,
          const std::vector<index> &p, const TDa &da) {
    const index n = length(p);

    for (index i = 0; i < n; i++) {
      for (index j = i; j < n; j++) {
        out[at(k, p[i], p[j])] += da * h[at(n, i, j)];
      }
    }
  }

  // r = phi(a, b) over the union of their names. The second-order part has the
  // same shape as in DDScalar::binary.
  template <typename TDa, typename TDb, typename TDaa, typename TDab,
            typename TDbb>
  HYPERJET_INLINE static SScalar
  combine(const Scalar f, const SScalar &a, const SScalar &b, const TDa &da,
          const TDb &db, const TDaa &daa, const TDab &dab, const TDbb &dbb) {
    SScalar r;
    r.m_f = f;

    std::vector<index> pa;
    std::vector<index> pb;

    merge_names(a.m_d, b.m_d, r.m_d, pa, pb);

    const index k = length(r.m_d);

    // both gradients in the union index space
    std::vector<TScalar> ga(k, TScalar(0));
    std::vector<TScalar> gb(k, TScalar(0));

    for (index i = 0; i < length(a.m_d); i++) {
      ga[pa[i]] = a.m_d[i].second;
    }

    for (index i = 0; i < length(b.m_d); i++) {
      gb[pb[i]] = b.m_d[i].second;
    }

    for (index i = 0; i < k; i++) {
      r.m_d[i].second = da * ga[i] + db * gb[i];
    }

    if constexpr (TOrder == 2) {
      r.m_h.assign(hessian_length(k), TScalar(0));

      scatter(r.m_h, k, a.m_h, pa, da);
      scatter(r.m_h, k, b.m_h, pb, db);

      index p = 0;

      for (index i = 0; i < k; i++) {
        const TScalar ca = daa * ga[i] + dab * gb[i];
        const TScalar cb = dab * ga[i] + dbb * gb[i];

        for (index j = i; j < k; j++) {
          r.m_h[p++] += ca * ga[j] + cb * gb[j];
        }
      }
    }

    return r;
  }

public: // constructors
  SScalar() = default;

  SScalar(const Scalar f) : m_f(f), m_d() {}

  // The Hessian of a value built from a gradient is zero, as it is for a
  // variable and for a constant.
  SScalar(const Scalar f, const Data &d) : m_f(f), m_d(d.begin(), d.end()) {
    std::sort(m_d.begin(), m_d.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    if constexpr (TOrder == 2) {
      m_h.assign(hessian_length(length(m_d)), TScalar(0));
    }
  }

  static SScalar constant(const Scalar value) { return SScalar(value, Data()); }

  static SScalar variable(const std::string &name, const Scalar value) {
    return from_terms(value, Terms{{name, Scalar(1)}});
  }

  // Build a set of variables in one call. The order given here is the order
  // g() and hm() project onto when handed the same names, so a caller can use
  // one list for both.
  static std::vector<SScalar>
  variables(const std::vector<std::pair<std::string, Scalar>> &values) {
    std::vector<SScalar> result;
    result.reserve(values.size());

    for (const auto &[name, value] : values) {
      result.push_back(variable(name, value));
    }

    return result;
  }

private: // methods
  // Takes the storage as it is, already sorted. A second constructor would make
  // SScalar(f, {{"x", 1.0}}) ambiguous between Data and Terms.
  HYPERJET_INLINE static SScalar from_terms(const Scalar f, Terms d) {
    SScalar r;
    r.m_f = f;
    r.m_d = std::move(d);

    if constexpr (TOrder == 2) {
      r.m_h.assign(hessian_length(length(r.m_d)), TScalar(0));
    }

    return r;
  }

public:
public: // methods
  index size() const { return m_d.size(); }

  Scalar f() const { return m_f; }

  Scalar d(const std::string &variable) const {
    const auto it = std::lower_bound(m_d.begin(), m_d.end(), variable, before);

    if (it == m_d.end() || it->first != variable)
      return Scalar(0);

    return it->second;
  }

  // The names this value carries, in the order the Hessian is laid out over.
  std::vector<std::string> names() const {
    std::vector<std::string> r;
    r.reserve(m_d.size());

    for (const auto &[name, derivative] : m_d) {
      r.push_back(name);
    }

    return r;
  }

  // Projection onto an explicit, ordered list of names -- the bridge from
  // named derivatives to code that wants arrays, an assembly step for
  // instance. A name the value does not carry reads as zero, exactly as d()
  // and dd() do, so a local value projects onto a larger global set without
  // having to be padded first. The caller fixes the order, which is what makes
  // the result composable: two values with different name sets project onto
  // the same layout.
  std::vector<Scalar> g(const std::vector<std::string> &names) const {
    const auto p = positions(names);

    std::vector<Scalar> result;
    result.reserve(names.size());

    for (const auto i : p) {
      result.push_back(i < 0 ? Scalar(0) : m_d[i].second);
    }

    return result;
  }

  Scalar dd(const std::string &a, const std::string &b) const
    requires(TOrder == 2)
  {
    const auto ia = std::lower_bound(m_d.begin(), m_d.end(), a, before);
    const auto ib = std::lower_bound(m_d.begin(), m_d.end(), b, before);

    if (ia == m_d.end() || ia->first != a || ib == m_d.end() ||
        ib->first != b) {
      return Scalar(0);
    }

    return m_h[at(length(m_d), ia - m_d.begin(), ib - m_d.begin())];
  }
  // The Hessian over an explicit, ordered list of names, row-major and
  // symmetric. Same zero-for-unknown rule as g().
  std::vector<Scalar> hm(const std::vector<std::string> &names) const
    requires(TOrder == 2)
  {
    const auto p = positions(names);
    const index n = length(names);
    const index k = length(m_d);

    std::vector<Scalar> result(static_cast<std::size_t>(n * n), Scalar(0));

    for (index i = 0; i < n; i++) {
      if (p[i] < 0) {
        continue;
      }

      for (index j = 0; j < n; j++) {
        if (p[j] < 0) {
          continue;
        }

        result[i * n + j] = m_h[at(k, p[i], p[j])];
      }
    }

    return result;
  }

  Scalar eval(const Data &d) const {
    Scalar r(m_f);

    for (const auto &[name, derivative] : m_d) {
      const auto it = d.find(name);

      if (it == d.end())
        continue;

      r += derivative * it->second;
    }

    return r;
  }

  friend std::ostream &operator<<(std::ostream &out, const Type &value) {
    out << value.m_f;

    for (const auto &d : value.m_d) {
      if (d.second < 0)
        out << " " << d.second << "*d" << d.first;
      else
        out << " +" << d.second << "*d" << d.first;
    }

    return out;
  }

  std::string to_string() {
    std::stringstream output;

    output << *this;

    return output.str();
  }

  // operators: negate

  Type operator-() const { return apply(-m_f, Scalar(-1), Scalar(0)); }

  // operators: add

  Type operator+(const Type &b) const {
    const auto f = m_f + b.f();
    const auto da = Scalar(1);
    const auto db = Scalar(1);
    const auto daa = Scalar(0);
    const auto dab = Scalar(0);
    const auto dbb = Scalar(0);

    return combine(f, *this, b, da, db, daa, dab, dbb);
  }

  Type operator+(const Scalar b) const {
    Type result = *this;

    result.m_f += b;

    return result;
  }

  friend Type operator+(const Scalar a, const Type &b) { return b + a; }

  // The in-place forms delegate, which keeps the merge in one place and makes
  // aliased operands correct without a guard of their own.
  Type &operator+=(const Type &b) { return *this = *this + b; }

  Type &operator+=(const Scalar &b) {
    m_f += b;

    return *this;
  }

  // operators: sub

  Type operator-(const Type &b) const {
    const auto f = m_f - b.f();
    const auto da = Scalar(1);
    const auto db = Scalar(-1);
    const auto daa = Scalar(0);
    const auto dab = Scalar(0);
    const auto dbb = Scalar(0);

    return combine(f, *this, b, da, db, daa, dab, dbb);
  }

  Type operator-(const Scalar b) const { return -b + *this; }

  friend Type operator-(const Scalar a, const Type &b) {
    Type result = -b;

    result.m_f += a;

    return result;
  }

  Type &operator-=(const Type &b) { return *this = *this - b; }

  Type &operator-=(const Scalar &b) {
    m_f -= b;

    return *this;
  }

  // operators: mul

  Type operator*(const Type &b) const {
    const auto f = m_f * b.f();
    const auto da = b.f();
    const auto db = m_f;
    const auto daa = Scalar(0);
    const auto dab = Scalar(1);
    const auto dbb = Scalar(0);

    return combine(f, *this, b, da, db, daa, dab, dbb);
  }

  Type operator*(const Scalar b) const { return apply(m_f * b, b, Scalar(0)); }

  friend Type operator*(const Scalar a, const Type &b) { return b * a; }

  Type &operator*=(const Type &b) { return *this = *this * b; }

  Type &operator*=(const Scalar &b) { return *this = *this * b; }

  // operators: div

  Type operator/(const Type &b) const {
    using std::pow;

    const auto tmp = 1 / b.f();

    const auto f = m_f * tmp;
    const auto da = tmp;
    const auto db = -m_f / pow(b.f(), 2);
    const auto daa = Scalar(0);
    const auto dab = -Scalar(1) / pow(b.f(), 2);
    const auto dbb = 2 * m_f / pow(b.f(), 3);

    return combine(f, *this, b, da, db, daa, dab, dbb);
  }

  Type operator/(const Scalar b) const { return Scalar(1) / b * (*this); }

  friend Type operator/(const Scalar a, const Type &b) {
    using std::pow;

    const auto f = a / b.f();
    const auto db = -a / pow(b.f(), 2);
    const auto dbb = 2 * a / pow(b.f(), 3);

    return b.apply(f, db, dbb);
  }

  Type &operator/=(const Type &b) { return *this = *this / b; }

  Type &operator/=(const Scalar &b) {
    operator*=(1 / b);

    return *this;
  }

  // arithmetic

  Type pow(const Scalar b) const {
    using std::pow;

    const auto f = pow(m_f, b);
    const auto da = b * pow(m_f, b - Scalar(1));
    const auto daa = (b - 1) * b * pow(m_f, b - 2);

    return apply(f, da, daa);
  }

  Type sqrt() const {
    using std::sqrt;

    const auto f = sqrt(m_f);
    const auto da = Scalar(1) / (Scalar(2) * f);
    const auto daa = -da / (2 * m_f);

    return apply(f, da, daa);
  }

  Type cbrt() const {
    using std::cbrt;

    const auto f = cbrt(m_f);
    const auto da = Scalar(1) / (Scalar(3) * f * f);
    const auto daa = -da * 2 / (3 * m_f);

    return apply(f, da, daa);
  }

  Type reciprocal() const {
    const auto f = Scalar(1) / m_f;
    const auto da = -f * f;
    const auto daa = -2 * f * da;

    return apply(f, da, daa);
  }

  // trigonometric

  Type cos() const {
    using std::cos;
    using std::sin;

    const auto f = cos(m_f);
    const auto da = -sin(m_f);
    const auto daa = -cos(m_f);

    return apply(f, da, daa);
  }

  Type sin() const {
    using std::cos;
    using std::sin;

    const auto f = sin(m_f);
    const auto da = cos(m_f);
    const auto daa = -sin(m_f);

    return apply(f, da, daa);
  }

  Type tan() const {
    using std::tan;

    const auto f = tan(m_f);
    const auto da = f * f + Scalar(1);
    const auto daa = da * 2 * f;

    return apply(f, da, daa);
  }

  Type acos() const {
    using std::acos;
    using std::sqrt;

    const auto tmp = Scalar(1) - m_f * m_f;

    const auto f = acos(m_f);
    const auto da = -Scalar(1) / sqrt(tmp);
    const auto daa = da * m_f / tmp;

    return apply(f, da, daa);
  }

  Type asin() const {
    using std::asin;
    using std::sqrt;

    const auto tmp = Scalar(1) - m_f * m_f;

    const auto f = asin(m_f);
    const auto da = Scalar(1) / sqrt(tmp);
    const auto daa = da * m_f / tmp;

    return apply(f, da, daa);
  }

  Type atan() const {
    using std::atan;

    const auto f = atan(m_f);
    const auto da = Scalar(1) / (m_f * m_f + Scalar(1));
    const auto daa = -da * da * 2 * m_f;

    return apply(f, da, daa);
  }

  Type atan2(const Type &b) const {
    using std::atan2;

    const auto tmp = m_f * m_f + b.m_f * b.m_f;

    const auto f = atan2(m_f, b.m_f);
    const auto da = b.m_f / tmp;
    const auto db = -m_f / tmp;
    const auto daa = db * da * 2;
    const auto dab = db * db - da * da;
    const auto dbb = -daa;

    return combine(f, *this, b, da, db, daa, dab, dbb);
  }

  static Type hypot(const Type &a, const Type &b) {
    using std::hypot;

    const auto f = hypot(a.m_f, b.m_f);
    const auto f3 = f * f * f;
    const auto da = a.m_f / f;
    const auto db = b.m_f / f;
    const auto daa = b.m_f * b.m_f / f3;
    const auto dab = -a.m_f * b.m_f / f3;
    const auto dbb = a.m_f * a.m_f / f3;

    return combine(f, a, b, da, db, daa, dab, dbb);
  }

  // hypot(a, b, c) == hypot(hypot(a, b), c), so the two-argument form carries
  // it and the chain rule takes care of the derivatives. That is exact, and it
  // saves a three-operand merge that nothing else would need.
  static Type hypot(const Type &a, const Type &b, const Type &c) {
    return hypot(hypot(a, b), c);
  }

  // hyperbolic

  Type cosh() const {
    using std::cosh;
    using std::sinh;

    const auto f = cosh(m_f);
    const auto da = sinh(m_f);
    const auto daa = f;

    return apply(f, da, daa);
  }

  Type sinh() const {
    using std::cosh;
    using std::sinh;

    const auto f = sinh(m_f);
    const auto da = cosh(m_f);
    const auto daa = f;

    return apply(f, da, daa);
  }

  Type tanh() const {
    using std::tanh;

    const auto f = tanh(m_f);
    const auto da = 1 - f * f;
    const auto daa = -2 * f * da;

    return apply(f, da, daa);
  }

  Type acosh() const {
    using std::acosh;
    using std::sqrt;

    const auto f = acosh(m_f);
    const auto da = 1 / (sqrt(m_f - 1) * sqrt(m_f + 1));
    const auto daa = -da * m_f / ((m_f - 1) * (m_f + 1));

    return apply(f, da, daa);
  }

  Type asinh() const {
    using std::asinh;
    using std::sqrt;

    const auto f = asinh(m_f);
    const auto da = 1 / sqrt(1 + m_f * m_f);
    const auto daa = -da * m_f / (1 + m_f * m_f);

    return apply(f, da, daa);
  }

  Type atanh() const {
    using std::atanh;
    using std::pow;

    const auto f = atanh(m_f);
    const auto da = 1 / (1 - m_f * m_f);
    const auto daa = 2 * m_f / pow(m_f * m_f - 1, 2);

    return apply(f, da, daa);
  }

  // exponents and logarithms

  Type exp() const {
    using std::exp;

    const auto f = exp(m_f);
    const auto da = f;
    const auto daa = f;

    return apply(f, da, daa);
  }

  Type log() const {
    using std::log;

    const auto f = log(m_f);
    const auto da = 1 / m_f;
    const auto daa = -da * da;

    return apply(f, da, daa);
  }

  Type log(const TScalar base) const {
    using std::log;

    const auto f = log(m_f) / log(base);
    const auto da = 1 / (m_f * log(base));
    const auto daa = -da / m_f;

    return apply(f, da, daa);
  }

  Type log2() const {
    using std::log;
    using std::log2;

    const auto f = log2(m_f);
    const auto da = 1 / (m_f * log(2));
    const auto daa = -da / m_f;

    return apply(f, da, daa);
  }

  Type log10() const {
    using std::log;
    using std::log10;

    const auto f = log10(m_f);
    const auto da = 1 / (m_f * log(10));
    const auto daa = -da / m_f;

    return apply(f, da, daa);
  }

  // abs

  Type abs() const { return m_f < 0 ? -(*this) : *this; }

  // comparison

  bool operator==(const Type &b) const { return m_f == b.m_f; }

  bool operator!=(const Type &b) const { return m_f != b.m_f; }

  bool operator<(const Type &b) const { return m_f < b.m_f; }

  bool operator>(const Type &b) const { return m_f > b.m_f; }

  bool operator<=(const Type &b) const { return m_f <= b.m_f; }

  bool operator>=(const Type &b) const { return m_f >= b.m_f; }

  bool operator==(const Scalar b) const { return m_f == b; }

  bool operator!=(const Scalar b) const { return m_f != b; }

  bool operator<(const Scalar b) const { return m_f < b; }

  bool operator>(const Scalar b) const { return m_f > b; }

  bool operator<=(const Scalar b) const { return m_f <= b; }

  bool operator>=(const Scalar b) const { return m_f >= b; }

  friend bool operator==(const Scalar a, const Type &b) {
    return b.operator==(a);
  }

  friend bool operator!=(const Scalar a, const Type &b) {
    return b.operator!=(a);
  }

  friend bool operator<(const Scalar a, const Type &b) {
    return b.operator>(a);
  }

  friend bool operator>(const Scalar a, const Type &b) {
    return b.operator<(a);
  }

  friend bool operator<=(const Scalar a, const Type &b) {
    return b.operator>=(a);
  }

  friend bool operator>=(const Scalar a, const Type &b) {
    return b.operator<=(a);
  }
};

// std::abs

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> abs(const SScalar<TOrder, TScalar> &a) {
  return a.abs();
}

// std::pow

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> pow(const SScalar<TOrder, TScalar> &a, const index b) {
  return a.pow(b);
}

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> pow(const SScalar<TOrder, TScalar> &a,
                             const TScalar b) {
  return a.pow(b);
}

// std::sqrt

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> sqrt(const SScalar<TOrder, TScalar> &a) {
  return a.sqrt();
}

// std::cbrt

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> cbrt(const SScalar<TOrder, TScalar> &a) {
  return a.cbrt();
}

// std::cos

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> cos(const SScalar<TOrder, TScalar> &a) {
  return a.cos();
}

// std::sin

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> sin(const SScalar<TOrder, TScalar> &a) {
  return a.sin();
}

// std::tan

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> tan(const SScalar<TOrder, TScalar> &a) {
  return a.tan();
}

// std::acos

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> acos(const SScalar<TOrder, TScalar> &a) {
  return a.acos();
}

// std::asin

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> asin(const SScalar<TOrder, TScalar> &a) {
  return a.asin();
}

// std::atan

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> atan(const SScalar<TOrder, TScalar> &a) {
  return a.atan();
}

// std::atan2

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> atan2(const SScalar<TOrder, TScalar> &a,
                               const SScalar<TOrder, TScalar> &b) {
  return a.atan2(b);
}

// std::hypot

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> hypot(const SScalar<TOrder, TScalar> &a,
                               const SScalar<TOrder, TScalar> &b) {
  return SScalar<TOrder, TScalar>::hypot(a, b);
}

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> hypot(const SScalar<TOrder, TScalar> &a,
                               const SScalar<TOrder, TScalar> &b,
                               const SScalar<TOrder, TScalar> &c) {
  return SScalar<TOrder, TScalar>::hypot(a, b, c);
}

// std::cosh

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> cosh(const SScalar<TOrder, TScalar> &a) {
  return a.cosh();
}

// std::sinh

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> sinh(const SScalar<TOrder, TScalar> &a) {
  return a.sinh();
}

// std::tanh

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> tanh(const SScalar<TOrder, TScalar> &a) {
  return a.tanh();
}

// std::acosh

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> acosh(const SScalar<TOrder, TScalar> &a) {
  return a.acosh();
}

// std::asin

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> asinh(const SScalar<TOrder, TScalar> &a) {
  return a.asinh();
}

// std::atan

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> atanh(const SScalar<TOrder, TScalar> &a) {
  return a.atanh();
}

// std::exp

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> exp(const SScalar<TOrder, TScalar> &a) {
  return a.exp();
}

// std::log

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> log(const SScalar<TOrder, TScalar> &a) {
  return a.log();
}

// std::log2

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> log2(const SScalar<TOrder, TScalar> &a) {
  return a.log2();
}

// std::log10

template <index TOrder, typename TScalar>
SScalar<TOrder, TScalar> log10(const SScalar<TOrder, TScalar> &a) {
  return a.log10();
}

} // namespace hyperjet

#if defined EIGEN_WORLD_VERSION

namespace Eigen {

template <typename T> struct NumTraits;

template <hyperjet::index TOrder, typename TScalar, std::ptrdiff_t TSize>
struct NumTraits<hyperjet::DDScalar<TOrder, TScalar, TSize>>
    : NumTraits<TScalar> {
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

template <typename BinOp, hyperjet::index TOrder, typename TScalar,
          hyperjet::index TSize>
struct ScalarBinaryOpTraits<hyperjet::DDScalar<TOrder, TScalar, TSize>, TScalar,
                            BinOp> {
  using ReturnType = hyperjet::DDScalar<TOrder, TScalar, TSize>;
};

template <typename BinOp, hyperjet::index TOrder, typename TScalar,
          hyperjet::index TSize>
struct ScalarBinaryOpTraits<TScalar, hyperjet::DDScalar<TOrder, TScalar, TSize>,
                            BinOp> {
  using ReturnType = hyperjet::DDScalar<TOrder, TScalar, TSize>;
};

} // namespace Eigen

#endif
