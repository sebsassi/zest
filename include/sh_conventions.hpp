/*
Copyright (c) 2024-2026 Sebastian Sassi

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

#include <concepts>
#include <numbers>

namespace zest::st
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
    four_pi,
    /** quantum mechanics (unit norm) normalization */
    unit
};

template <SHNorm norm, SHPhase phase>
struct SHConvention
{
    static constexpr SHNorm sh_norm = norm;
    static constexpr SHPhase sh_phase = phase;
};

using Geo = SHConvention<SHNorm::four_pi, SHPhase::none>;
using Acoustics = SHConvention<SHNorm::unit, SHPhase::none>;
using QM = SHConvention<SHNorm::unit, SHPhase::cs>;

template <typename T>
concept sh_convention = std::same_as<
    std::remove_cvref_t<T>,
    SHConvention<std::remove_cvref_t<T>::sh_norm, std::remove_cvref_t<T>::sh_phase>>;

template <typename T>
concept sh_tagged = std::derived_from<
    std::remove_cvref_t<T>,
    SHConvention<std::remove_cvref_t<T>::sh_norm, std::remove_cvref_t<T>::sh_phase>>;

template <sh_tagged T>
consteval SHNorm sh_norm_of() { return std::remove_cvref_t<T>::sh_norm; }

template <typename T>
    requires sh_tagged<typename std::remove_cvref_t<T>::shape_type>
consteval SHNorm sh_norm_of() { return std::remove_cvref_t<T>::shape_type::sh_norm; }

template <sh_tagged T>
consteval SHNorm sh_phase_of() { return std::remove_cvref_t<T>::sh_norm; }

template <typename T>
    requires sh_tagged<typename std::remove_cvref_t<T>::shape_type>
consteval SHPhase sh_phase_of() { return std::remove_cvref_t<T>::shape_type::sh_phase; }

template <typename T>
    requires sh_tagged<T> || sh_tagged<typename std::remove_cvref_t<T>::shape_type>
using convention_of = SHConvention<sh_norm_of<T>(), sh_phase_of<T>()>;

/**
    @brief Normalization constant of spherical harmonics coefficients.

    @tparam sh_norm normalization convention

    @return normalization constant
*/
template <SHNorm sh_norm>
[[nodiscard]] constexpr double normalization() noexcept
{
    if constexpr (sh_norm == SHNorm::unit)
        return 1.0;
    else if constexpr (sh_norm == SHNorm::four_pi)
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
    if constexpr (from == SHNorm::unit)
        norm = 1.0;
    else if constexpr (from == SHNorm::four_pi)
        norm = sqrt_4pi;

    if constexpr (to == SHNorm::unit)
        norm *= 1.0;
    else if constexpr (to == SHNorm::four_pi)
        norm *= inv_sqrt_4pi;

    return norm;
}

template <SHNorm from, SHNorm to>
    requires (from == to)
[[nodiscard]] constexpr double conversion_const() noexcept
{
    return 1.0;
}

} // namespace zest::st

