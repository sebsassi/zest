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

#include <span>
#include <vector>

#include "real_sh_expansion.hpp"
#include "zernike_expansion.hpp"
#include "sh_conventions.hpp"

namespace zest
{
namespace st
{

/**
    @brief Compute cross power spectrum of two spherical harmonic expansions.

    @param a spherical harmonic expansion
    @param b spherical harmonic expansion
    @param out output buffer for the cross power spectrum
*/
template <st::real_sh_expansion ExpansionType>
void cross_power_spectrum(
    ExpansionType&& a, ExpansionType&& b, std::span<double> out) noexcept
{
    constexpr double norm = normalization<ExpansionType::sh_norm>();
    std::size_t min_order
            = std::min(std::min(a.order(), b.order()), out.size() - 1);
    
    for (std::size_t l = 0; l < min_order; ++l)
    {
        out[l] = a(l, 0)[0]*b(l, 0)[0];
        for (std::size_t m = 1; m <= l; ++m)
            out[l] += a(l, m)[0]*b(l, m)[0] + a(l, m)[1]*b(l, m)[1];
        out[l] *= norm;
    }
}

/**
    @brief Compute cross power spectrum of two spherical harmonic expansions.

    @param a spherical harmonic expansions
    @param b spherical harmonic expansions

    @return `std::vector` storing the the cross power spectrum
*/
template <st::real_sh_expansion ExpansionType>
[[nodiscard]] std::vector<double> cross_power_spectrum(
    ExpansionType&& a, ExpansionType&& b)
{
    std::size_t min_order = std::min(a.order(), b.order());
    std::vector<double> out(min_order);
    cross_power_spectrum(a, b, out);
    return out;
}

/**
    @brief Compute power spectrum of a spherical harmonic expansions.

    @param expansion spherical harmonic expansion
    @param out output buffer for the power spectrum
*/
template <st::real_sh_expansion ExpansionType>
void power_spectrum(
    ExpansionType&& expansion, std::span<double> out) noexcept
{
    constexpr double norm = normalization<ExpansionType::sh_norm>();
    std::size_t min_order = std::min(out.size(), expansion.order());
    
    for (std::size_t l = 0; l < expansion.order(); ++l)
    {
        out[l] = expansion(l, 0)[0]*expansion(l, 0)[0];
        for (std::size_t m = 1; m <= l; ++m)
            out[l] += expansion(l, m)[0]*expansion(l, m)[0]
                    + expansion(l, m)[1]*expansion(l, m)[1];
        out[l] *= norm;
    }
}

/**
    @brief Compute power spectrum of a spherical harmonic expansions.

    @param expansion spherical harmonic expansion

    @return `std::vector` storing the power spectrum
*/
template <st::real_sh_expansion ExpansionType>
[[nodiscard]] std::vector<double> power_spectrum(ExpansionType&& expansion)
{
    std::vector<double> out(expansion.order());
    power_spectrum(std::forward(expansion), out);
    return out;
}

} // namespace st

namespace zt
{

/**
    @brief Compute power spectrum of a Zernike expansion.

    @param expansion Zernike expansion.
    @param out place to store the power spectrum.
*/
template <zt::zernike_expansion ExpansionType>
void power_spectrum(
    ExpansionType&& expansion,
    RadialZernikeSpan<double, ExpansionType::zernike_norm> out) noexcept
{
    constexpr double norm = st::normalization<ExpansionType::sh_norm>();
    std::size_t min_order = std::min(out.order(), expansion.order());

    for (std::size_t n = 0; n < min_order; ++n)
    {
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            out(n, l) = expansion(n, l, 0)[0]*expansion(n, l, 0)[0];
            for (std::size_t m = 1; m <= l; ++m)
                out(n, l) += expansion(n, l, m)[0]*expansion(n, l, m)[0]
                        + expansion(n, l, m)[1]*expansion(n, l, m)[1];
            out(n, l) *= norm;
        }
    }
}

/**
    @brief Compute power spectrum of a Zernike expansions.

    @param expansion Zernike expansion

    @return `std::vector` storing the power spectrum.
*/
template <zt::zernike_expansion ExpansionType>
[[nodiscard]] std::vector<double> power_spectrum(ExpansionType&& expansion)
{
    using SpectrumSpan = RadialZernikeSpan<double, ExpansionType::zernike_norm>;
    std::vector<double> res(RadialZernikeLayout::size(expansion.order()));
    power_spectrum(
        std::forward(expansion), SpectrumSpan(res, expansion.order()));
    return res;
}

} // namespace zt

} // namespace zest