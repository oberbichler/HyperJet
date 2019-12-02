//     __  __                          __     __
//    / / / /_  ______  ___  _____    / /__  / /_
//   / /_/ / / / / __ \/ _ \/ ___/_  / / _ \/ __/
//  / __  / /_/ / /_/ /  __/ /  / /_/ /  __/ /_
// /_/ /_/\__, / .___/\___/_/   \____/\___/\__/
//       /____/_/
//
// Copyright (c) 2019 Thomas Oberbichler

#pragma once

#include "hyperjet.h"

namespace Eigen {

template <typename T>
struct NumTraits;

template <>
struct NumTraits<hyperjet::Jet<double>> : NumTraits<double> {
    using Real = hyperjet::Jet<double>;
    using NonInteger = hyperjet::Jet<double>;
    using Nested = hyperjet::Jet<double>;
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

template <>
struct NumTraits<hyperjet::HyperJet<double>> : NumTraits<double> {
    using Real = hyperjet::HyperJet<double>;
    using NonInteger = hyperjet::HyperJet<double>;
    using Nested = hyperjet::HyperJet<double>;
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

template <typename BinOp>
struct ScalarBinaryOpTraits<hyperjet::Jet<double>, double, BinOp> {
    using ReturnType = hyperjet::Jet<double>;
};

template <typename BinOp>
struct ScalarBinaryOpTraits<hyperjet::HyperJet<double>, double, BinOp> {
    using ReturnType = hyperjet::HyperJet<double>;
};

template <typename BinOp>
struct ScalarBinaryOpTraits<double, hyperjet::Jet<double>, BinOp> {
    using ReturnType = hyperjet::Jet<double>;
};

template <typename BinOp>
struct ScalarBinaryOpTraits<double, hyperjet::HyperJet<double>, BinOp> {
    using ReturnType = hyperjet::HyperJet<double>;
};

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)                     \
    using Matrix##SizeSuffix##TypeSuffix = Matrix<Type, Size, Size, 0, Size, Size>; \
    using Vector##SizeSuffix##TypeSuffix = Matrix<Type, Size, 1, 0, Size, 1>;       \
    using RowVector##SizeSuffix##TypeSuffix = Matrix<Type, 1, Size, 1, 1, Size>;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)                    \
    using Matrix##Size##X##TypeSuffix = Matrix<Type, Size, -1, 0, Size, -1>; \
    using Matrix##X##Size##TypeSuffix = Matrix<Type, -1, Size, 0, -1, Size>;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
    EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2)         \
    EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3)         \
    EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4)         \
    EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, -1, X)        \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2)      \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3)      \
    EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(hyperjet::Jet<double>, hg)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(hyperjet::HyperJet<double>, hg)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

} // namespace Eigen