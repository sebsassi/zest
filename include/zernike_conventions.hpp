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

/**
    @brief Normalization of Zernike polynomials.

    @tparam NORM normalization convention

    @return normalization constant
*/
template <ZernikeNorm NORM>
[[nodiscard]] inline double normalization(std::size_t n) noexcept
{
    if constexpr (NORM == ZernikeNorm::NORMED)
        return 1.0;
    else if constexpr (NORM == ZernikeNorm::UNNORMED)
        return double(2*n + 3);
}

/**
    @brief Constant for converting between Zernike polynomial conventions.

    @tparam FROM source normalization convention
    @tparam TO destination normalization convention

    @return conversion constant
*/
template <ZernikeNorm FROM, ZernikeNorm TO>
    requires (FROM == TO)
[[nodiscard]] inline double conversion_factor(std::size_t n) noexcept
{
    return 1.0;
}

/**
    @brief Constant for converting between Zernike polynomial conventions.

    @tparam FROM source normalization convention
    @tparam TO destination normalization convention

    @return conversion constant
*/
template <ZernikeNorm FROM, ZernikeNorm TO>
    requires (FROM != TO)
[[nodiscard]] inline double conversion_factor(std::size_t n) noexcept
{
    if constexpr (TO == ZernikeNorm::NORMED)
        return 1.0/std::sqrt(double(2*n + 3));
    else
        return std::sqrt(double(2*n + 3));
}

} // namespace zt
} // namespace zest