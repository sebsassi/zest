#pragma once

#include <cstddef>
#include <cmath>

namespace zest
{
namespace zt
{

/**
    @brief Zernike polynomial normalizations.
*/
enum class ZernikeNorm { NORMED, UNNORMED };

template <ZernikeNorm NORM>
[[nodiscard]] constexpr double normalization(std::size_t n) noexcept
{
    if constexpr (NORM == ZernikeNorm::NORMED)
        return 1.0;
    else if constexpr (NORM == ZernikeNorm::UNNORMED)
        return double(2*n + 3);
}

template <ZernikeNorm FROM, ZernikeNorm TO>
    requires (FROM == TO)
[[nodiscard]] constexpr double conversion_factor(std::size_t n) noexcept
{
    return 1.0;
}


template <ZernikeNorm FROM, ZernikeNorm TO>
    requires (FROM != TO)
[[nodiscard]] constexpr double conversion_factor(std::size_t n) noexcept
{
    if constexpr (TO == ZernikeNorm::NORMED)
        return 1.0/std::sqrt(double(2*n + 3));
    else
        return std::sqrt(double(2*n + 3));
}

} // namespace zt
} // namespace zest