#pragma once

#include <cmath>

namespace zest
{
namespace st
{

/**
    @brief Spherical harmonic phase conventions.
*/
enum class SHPhase { NONE = -1, CS = 1 };

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

/**
    @brief Normalization constant of spherical harmonics coefficients.

    @tparam NORM normalization convention

    @return normalization constant
*/
template <SHNorm NORM>
[[nodiscard]] constexpr double normalization() noexcept
{
    if constexpr (NORM == SHNorm::QM)
        return 1.0;
    else if constexpr (NORM == SHNorm::GEO)
        return 1.0/(4.0*std::numbers::pi);
}

/**
    @brief Constant for converting between spherical harmonics conventions.

    @tparam FROM source normalization convention
    @tparam TO destination normalization convention

    @return conversion constant
*/
template <SHNorm FROM, SHNorm TO>
    requires (FROM == TO)
[[nodiscard]] constexpr double conversion_const() noexcept
{
    return 1.0;
}

/**
    @brief Constant for converting between spherical harmonics conventions.

    @tparam FROM source normalization convention
    @tparam TO destination normalization convention

    @return conversion constant
*/
template <SHNorm FROM, SHNorm TO>
[[nodiscard]] constexpr double conversion_const() noexcept
{
    constexpr double inv_sqrt_4pi = 0.5*std::numbers::inv_sqrtpi;
    constexpr double sqrt_4pi = 1.0/inv_sqrt_4pi;
    double from;
    if constexpr (FROM == SHNorm::QM)
        from = 1.0;
    else if constexpr (FROM == SHNorm::GEO)
        from = sqrt_4pi;

    double to;
    if constexpr (TO == SHNorm::QM)
        to = 1.0;
    else if constexpr (TO == SHNorm::GEO)
        to = inv_sqrt_4pi;

    return from*to;
}

} // namespace st
} // namespace zest