//     __  __                          __     __
//    / / / /_  ______  ___  _____    / /__  / /_
//   / /_/ / / / / __ \/ _ \/ ___/_  / / _ \/ __/
//  / __  / /_/ / /_/ /  __/ /  / /_/ /  __/ /_
// /_/ /_/\__, / .___/\___/_/   \____/\___/\__/
//       /____/_/
//
// Copyright (c) 2019-2020 Thomas Oberbichler

#pragma once

#include <Eigen/Core>

#include <assert.h>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

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

constexpr index Dynamic = -1;

constexpr index init_size(const int size)
{
    return size != -1 ? size : 0;
}

constexpr bool check_size()
{
#if defined(HYPERJET_EXCEPTIONS)
    return true;
#else
    return false;
#endif
}

} // namespace hyperjet