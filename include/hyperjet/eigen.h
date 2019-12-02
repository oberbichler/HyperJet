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

#include <algorithm>

namespace Eigen {

template <typename T>
struct NumTraits;

template <std::ptrdiff_t TSize>
struct NumTraits<hyperjet::Jet<double, TSize>> : NumTraits<double> {
    using Real = hyperjet::Jet<double, TSize>;
    using NonInteger = hyperjet::Jet<double, TSize>;
    using Nested = hyperjet::Jet<double, TSize>;
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

template <std::ptrdiff_t TSize>
struct NumTraits<hyperjet::HyperJet<double, TSize>> : NumTraits<double> {
    using Real = hyperjet::HyperJet<double, TSize>;
    using NonInteger = hyperjet::HyperJet<double, TSize>;
    using Nested = hyperjet::HyperJet<double, TSize>;
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

template <typename BinOp, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<hyperjet::Jet<double, TSize>, double, BinOp> {
    using ReturnType = hyperjet::Jet<double, TSize>;
};

template <typename BinOp, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<hyperjet::HyperJet<double, TSize>, double, BinOp> {
    using ReturnType = hyperjet::HyperJet<double, TSize>;
};

template <typename BinOp, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<double, hyperjet::Jet<double, TSize>, BinOp> {
    using ReturnType = hyperjet::Jet<double, TSize>;
};

template <typename BinOp, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<double, hyperjet::HyperJet<double, TSize>, BinOp> {
    using ReturnType = hyperjet::HyperJet<double, TSize>;
};

} // namespace Eigen