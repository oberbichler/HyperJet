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
#include "jet.h"
#include "hyperjet.h"

namespace Eigen {

template <typename T>
struct NumTraits;

template <typename TScalar, std::ptrdiff_t TSize>
struct NumTraits<hyperjet::Jet<TScalar, TSize>> : NumTraits<TScalar> {
    using Real = hyperjet::Jet<TScalar, TSize>;
    using NonInteger = hyperjet::Jet<TScalar, TSize>;
    using Nested = hyperjet::Jet<TScalar, TSize>;
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

template <typename TScalar, std::ptrdiff_t TSize>
struct NumTraits<hyperjet::HyperJet<TScalar, TSize>> : NumTraits<TScalar> {
    using Real = hyperjet::HyperJet<TScalar, TSize>;
    using NonInteger = hyperjet::HyperJet<TScalar, TSize>;
    using Nested = hyperjet::HyperJet<TScalar, TSize>;
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
struct ScalarBinaryOpTraits<hyperjet::Jet<TScalar, TSize>, TScalar, BinOp> {
    using ReturnType = hyperjet::Jet<TScalar, TSize>;
};

template <typename BinOp, typename TScalar, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<hyperjet::HyperJet<TScalar, TSize>, TScalar, BinOp> {
    using ReturnType = hyperjet::HyperJet<TScalar, TSize>;
};

template <typename BinOp, typename TScalar, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<TScalar, hyperjet::Jet<TScalar, TSize>, BinOp> {
    using ReturnType = hyperjet::Jet<TScalar, TSize>;
};

template <typename BinOp, typename TScalar, std::ptrdiff_t TSize>
struct ScalarBinaryOpTraits<TScalar, hyperjet::HyperJet<TScalar, TSize>, BinOp> {
    using ReturnType = hyperjet::HyperJet<TScalar, TSize>;
};

} // namespace Eigen