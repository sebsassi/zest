/*
Copyright (c) 2024 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/
#pragma once

#include <cmath>
#include <numbers>

namespace zest
{
namespace st
{

/**
    @brief Spherical harmonic phase conventions.
*/
enum class SHPhase { none = -1, cs = 1 };

/**
    @brief Spherical harmonic normalization conventions
*/
enum class SHNorm
{
    /** geodesy (4 pi) normalization */
    geo,
    /** quantum mechanics (unit norm) normalization */
    qm
};

/**
    @brief Normalization constant of spherical harmonics coefficients.

    @tparam sh_norm_param normalization convention

    @return normalization constant
*/
template <SHNorm sh_norm_param>
[[nodiscard]] constexpr double normalization() noexcept
{
    if constexpr (sh_norm_param == SHNorm::qm)
        return 1.0;
    else if constexpr (sh_norm_param == SHNorm::geo)
        return 1.0/(4.0*std::numbers::pi);
}

/**
    @brief Constant for converting between spherical harmonics conventions.

    @tparam from source normalization convention
    @tparam to destination normalization convention

    @return conversion constant
*/
template <SHNorm from, SHNorm to>
[[nodiscard]] constexpr double conversion_const() noexcept
{
    constexpr double inv_sqrt_4pi = 0.5*std::numbers::inv_sqrtpi;
    constexpr double sqrt_4pi = 1.0/inv_sqrt_4pi;
    double norm;
    if constexpr (from == SHNorm::qm)
        norm = 1.0;
    else if constexpr (from == SHNorm::geo)
        norm = sqrt_4pi;

    if constexpr (to == SHNorm::qm)
        norm *= 1.0;
    else if constexpr (to == SHNorm::geo)
        norm *= inv_sqrt_4pi;

    return norm;
}

template <SHNorm from, SHNorm to>
    requires (from == to)
[[nodiscard]] constexpr double conversion_const() noexcept
{
    return 1.0;
}

} // namespace st
} // namespace zest