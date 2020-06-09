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

template <int TOrder, typename TScalar, index TSize>
struct Space;

template <typename TScalar, index TSize>
struct Space<0, TScalar, TSize> {
    using Scalar = TScalar;

    template <index TDimension>
    using Vector = Eigen::Matrix<Scalar, 1, TDimension>;

    template <index TRows, index TCols>
    using Matrix = Eigen::Matrix<Scalar, TRows, TCols>;

    static Scalar empty()
    {
        static_assert(-1 <= TSize);

        return Scalar();
    }

    static Scalar constant(TScalar value)
    {
        static_assert(-1 <= TSize);

        return Scalar(value);
    }

    static Scalar variable(index i, Scalar value)
    {
        static_assert(-1 <= TSize);

        return value;
    }

    template <index TOffset, index TSize>
    HYPERJET_INLINE static Eigen::Matrix<Scalar, 1, TSize> variables(Eigen::Matrix<TScalar, 1, TSize> value)
    {
        static_assert(-1 <= TSize);

        Eigen::Matrix<Scalar, 1, TSize> result;

        for (index i = 0; i < TSize; i++) {
            result(i) = variable(TOffset + i, value(i));
        }

        return result;
    }

    static TScalar f(const Scalar& variable)
    {
        return variable;
    }

    static void set_f(Scalar& variable, TScalar value)
    {
        variable = value;
    }

    static TScalar g(const Scalar& variable, index i)
    {
        return TScalar(0);
    }

    static void set_g(Scalar& variable, index i, TScalar value)
    {
    }

    static TScalar h(const Scalar& variable, index i, index j)
    {
        return TScalar(0);
    }

    static void set_h(Scalar& variable, index i, index j, TScalar value)
    {
    }

    static TScalar explode(const Scalar& variable, Eigen::Ref<Eigen::Matrix<TScalar, 1, TSize>> g, Eigen::Ref<Eigen::Matrix<TScalar, TSize, TSize>> h)
    {
        return variable;
    }
};

template <typename TScalar, index TSize>
struct Space<1, TScalar, TSize> {
    using Scalar = Jet<TScalar, TSize>;

    template <index TDimension>
    using Vector = Eigen::Matrix<Scalar, 1, TDimension>;

    template <index TRows, index TCols>
    using Matrix = Eigen::Matrix<Scalar, TRows, TCols>;

    static Scalar empty()
    {
        static_assert(-1 <= TSize);

        return Scalar::empty();
    }

    static Scalar constant(TScalar value)
    {
        static_assert(-1 <= TSize);

        return Scalar(value);
    }

    static Scalar variable(index i, TScalar value)
    {
        static_assert(-1 <= TSize);

        Scalar result(value);
        result.g(i) = 1.0;
        return result;
    }

    template <index TOffset, index TSize>
    HYPERJET_INLINE static Eigen::Matrix<Scalar, 1, TSize> variables(Eigen::Matrix<TScalar, 1, TSize> value)
    {
        static_assert(-1 <= TSize);

        Eigen::Matrix<Scalar, 1, TSize> result;

        for (index i = 0; i < TSize; i++) {
            result(i) = variable(TOffset + i, value(i));
        }

        return result;
    }

    static TScalar f(const Scalar& variable)
    {
        return variable.f();
    }

    static void set_f(Scalar& variable, TScalar value)
    {
        variable.f() = value;
    }

    static TScalar g(const Scalar& variable, index i)
    {
        return variable.g(i);
    }

    static void set_g(Scalar& variable, index i, TScalar value)
    {
        variable.g(i) = value;
    }

    static TScalar h(const Scalar& variable, index i, index j)
    {
        return TScalar(0);
    }

    static void set_h(Scalar& variable, index i, index j, TScalar value)
    {
    }

    static TScalar explode(const Scalar& variable, Eigen::Ref<typename Scalar::Vector> g, Eigen::Ref<typename Scalar::Matrix> h)
    {
        if (g.size() >= 0) {
            assert(g.size() == variable.g().size());
            g = variable.g();
        }

        return variable.f();
    }
};

template <typename TScalar, index TSize>
struct Space<2, TScalar, TSize> {
    using Scalar = HyperJet<TScalar, TSize>;

    template <index TDimension>
    using Vector = Eigen::Matrix<Scalar, 1, TDimension>;

    template <index TRows, index TCols>
    using Matrix = Eigen::Matrix<Scalar, TRows, TCols>;

    static Scalar empty()
    {
        static_assert(-1 <= TSize);

        return Scalar::empty();
    }

    static Scalar constant(TScalar value)
    {
        static_assert(-1 <= TSize);

        return Scalar(value);
    }

    static Scalar variable(index i, TScalar value)
    {
        static_assert(-1 <= TSize);

        Scalar result(value);
        result.g(i) = 1.0;
        return result;
    }

    template <index TOffset, index TSize>
    HYPERJET_INLINE static Eigen::Matrix<Scalar, 1, TSize> variables(Eigen::Matrix<TScalar, 1, TSize> value)
    {
        static_assert(-1 <= TSize);

        Eigen::Matrix<Scalar, 1, TSize> result;

        for (index i = 0; i < TSize; i++) {
            result(i) = variable(TOffset + i, value(i));
        }

        return result;
    }

    static TScalar f(const Scalar& variable)
    {
        return variable.f();
    }

    static void set_f(Scalar& variable, TScalar value)
    {
        variable.f() = value;
    }

    static TScalar g(const Scalar& variable, index i)
    {
        return variable.g(i);
    }

    static void set_g(Scalar& variable, index i, TScalar value)
    {
        variable.g(i) = value;
    }

    static TScalar h(const Scalar& variable, index i, index j)
    {
        return variable.h(i, j);
    }

    static void set_h(Scalar& variable, index i, index j, TScalar value)
    {
        variable.h(i, j) = value;
    }

    static TScalar explode(const Scalar& variable, Eigen::Ref<typename Scalar::Vector> g, Eigen::Ref<typename Scalar::Matrix> h)
    {
        if (g.size() >= 0) {
            assert(g.size() == variable.g().size());
            g = variable.g();
        }

        if (h.size() >= 0) {
            assert(h.rows() == variable.h().rows() && h.cols() == variable.h().cols());
            h = variable.h();
        }

        return variable.f();
    }
};

// type definitions

using Space0d1 = Space<0, double, 1>;
using Space0d2 = Space<0, double, 2>;
using Space0d3 = Space<0, double, 3>;
using Space0d4 = Space<0, double, 4>;
using Space0d5 = Space<0, double, 5>;
using Space0d6 = Space<0, double, 6>;
using Space0d7 = Space<0, double, 7>;
using Space0d8 = Space<0, double, 8>;
using Space0d9 = Space<0, double, 9>;
using Space0d10 = Space<0, double, 10>;
using Space0d11 = Space<0, double, 11>;
using Space0d12 = Space<0, double, 12>;
using Space0d13 = Space<0, double, 13>;
using Space0d14 = Space<0, double, 14>;
using Space0d15 = Space<0, double, 15>;
using Space0d16 = Space<0, double, 16>;

using Space1d1 = Space<1, double, 1>;
using Space1d2 = Space<1, double, 2>;
using Space1d3 = Space<1, double, 3>;
using Space1d4 = Space<1, double, 4>;
using Space1d5 = Space<1, double, 5>;
using Space1d6 = Space<1, double, 6>;
using Space1d7 = Space<1, double, 7>;
using Space1d8 = Space<1, double, 8>;
using Space1d9 = Space<1, double, 9>;
using Space1d10 = Space<1, double, 10>;
using Space1d11 = Space<1, double, 11>;
using Space1d12 = Space<1, double, 12>;
using Space1d13 = Space<1, double, 13>;
using Space1d14 = Space<1, double, 14>;
using Space1d15 = Space<1, double, 15>;
using Space1d16 = Space<1, double, 16>;

using Space2d1 = Space<2, double, 1>;
using Space2d2 = Space<2, double, 2>;
using Space2d3 = Space<2, double, 3>;
using Space2d4 = Space<2, double, 4>;
using Space2d5 = Space<2, double, 5>;
using Space2d6 = Space<2, double, 6>;
using Space2d7 = Space<2, double, 7>;
using Space2d8 = Space<2, double, 8>;
using Space2d9 = Space<2, double, 9>;
using Space2d10 = Space<2, double, 10>;
using Space2d11 = Space<2, double, 11>;
using Space2d12 = Space<2, double, 12>;
using Space2d13 = Space<2, double, 13>;
using Space2d14 = Space<2, double, 14>;
using Space2d15 = Space<2, double, 15>;
using Space2d16 = Space<2, double, 16>;

} // namespace hyperjet