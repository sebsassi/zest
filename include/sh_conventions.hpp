#pragma once

#include <cmath>

namespace zest
{
namespace st
{

/**
    @brief Spherical harmonic phase conventions.
*/
enum class SHPhase { NONE = 1, CS = -1 };

/**
    @brief Spherical harmonic normalization conventions
*/
enum class SHNorm
{
    /** geodesy (4 pi) normalization */
    GEO,
    /** quantum mechanics (unit norm) normalization */
    QM
};


template <SHNorm NORM>
[[nodiscard]] constexpr double normalization() noexcept
{
    if constexpr (NORM == SHNorm::QM)
        return 1.0;
    else if constexpr (NORM == SHNorm::GEO)
        return 1.0/(4.0*std::numbers::pi);
}

/**
    @brief Normalization constant for converting between spherical harmonics conventions.

    @tparam FROM source normalization convention
    @tparam TO destination normalization convention

    @return normalization constant
*/
template <SHNorm FROM, SHNorm TO>
    requires (FROM == TO)
[[nodiscard]] constexpr double conversion_const() noexcept
{
    return 1.0;
}

/**
    @brief Normalization constant for converting between spherical harmonics conventions.

    @tparam FROM source normalization convention
    @tparam TO destination normalization convention

    @return normalization constant
*/
template <SHNorm FROM, SHNorm TO>
[[nodiscard]] constexpr double conversion_const() noexcept
{
    double from;
    if constexpr (FROM == SHNorm::QM)
        from = 1.0;
    else if constexpr (FROM == SHNorm::GEO)
        from = std::sqrt(4.0*std::numbers::pi);

    double to;
    if constexpr (TO == SHNorm::QM)
        to = 1.0;
    else if constexpr (TO == SHNorm::GEO)
        to = 1.0/std::sqrt(4.0*std::numbers::pi);

    return from*to;
}

} // namespace st
} // namespace zest