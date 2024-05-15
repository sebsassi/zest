#pragma once

#include <cmath>

namespace zest
{
namespace st
{

/* Switch for Condon-Shortely phase convention */
enum class SHPhase { NONE = 1, CS = -1 };

/*
Spherical harmonic normalization convention
    GEO geodesy (4 pi) normalization
    QM quantum mechanics (unit norm) normalization
*/
enum class SHNorm { GEO, QM/*, SCHMIDT, UNNORMALIZED, ORTHONORMAL*/ };

/*
Normalization constant for converting between spherical harmonics conventions.
*/
template <SHNorm FROM, SHNorm TO>
    requires (FROM == TO)
constexpr double conversion_const() noexcept
{
    return 1.0;
}

/*
Normalization constant for converting between spherical harmonics conventions.
*/
template <SHNorm FROM, SHNorm TO>
constexpr double conversion_const() noexcept
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

}
}