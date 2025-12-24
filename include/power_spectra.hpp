/*
Copyright (c) 2024, 2025 Sebastian Sassi

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

#include <vector>

#include "real_sh_expansion.hpp"
#include "sh_conventions.hpp"
#include "zernike_expansion.hpp"

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
template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
void
cross_power_spectrum(
    SHSpan<const double, indexing_mode, sh_norm, sh_phase> a,
    SHSpan<const double, indexing_mode, sh_norm, sh_phase> b,
    std::span<double> out) noexcept
{
    constexpr double norm = normalization<sh_norm>();
    std::size_t min_order
            = std::min(std::min(a.order(), b.order()), out.size());

    for (std::size_t l = 0; l < min_order; ++l)
    {
        auto a_l = a[l];
        auto b_l = b[l];
        auto& out_l = out[l];
        if constexpr (indexing_mode == IndexingMode::negative)
        {
            out_l = 0.0;
            for (auto m : a_l.indices())
                out_l += a_l[m]*b_l[m];
        }
        else
        {
            out_l = a_l[0, 0]*b_l[0, 0];
            for (auto m : a_l.indices(1))
                out_l += a_l[m, 0]*b_l[m, 0] + a_l[m, 1]*b_l[m, 1];
            out_l *= norm;
        }
    }
}

/**
    @brief Compute cross power spectrum of two spherical harmonic expansions.

    @param a spherical harmonic expansions
    @param b spherical harmonic expansions

    @return `std::vector` storing the the cross power spectrum
*/
template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
[[nodiscard]] std::vector<double>
cross_power_spectrum(
    SHSpan<const double, indexing_mode, sh_norm, sh_phase> a,
    SHSpan<const double, indexing_mode, sh_norm, sh_phase> b)
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
template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
void
power_spectrum(
    SHSpan<const double, indexing_mode, sh_norm, sh_phase> expansion,
    std::span<double> out) noexcept
{
    constexpr double norm = normalization<sh_norm>();
    std::size_t min_order = std::min(out.size(), expansion.order());

    for (std::size_t l = 0; l < min_order; ++l)
    {
        auto expansion_l = expansion[l];
        auto& out_l = out[l];
        if constexpr (indexing_mode == IndexingMode::negative)
        {
            out_l = 0.0;
            for (auto m : expansion_l.indices())
                out_l += expansion_l[m]*expansion_l[m];
        }
        else
        {
            out_l = expansion_l[0, 0]*expansion_l[0, 0];
            for (auto m : expansion_l.indices(1))
            for (std::size_t m = 1; m <= l; ++m)
                out_l += expansion_l[m, 0]*expansion_l[m, 0]
                        + expansion_l[m, 1]*expansion_l[m, 1];
            out_l *= norm;
        }
    }
}

/**
    @brief Compute power spectrum of a spherical harmonic expansions.

    @param expansion spherical harmonic expansion

    @return `std::vector` storing the power spectrum
*/
template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
[[nodiscard]] std::vector<double>
power_spectrum(SHSpan<const double, indexing_mode, sh_norm, sh_phase> expansion)
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
template <IndexingMode indexing_mode, ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase>
void
power_spectrum(
    ZernikeSpan<const double, indexing_mode, zernike_norm, sh_norm, sh_phase> expansion,
    RadialZernikeSpan<double, zernike_norm> out) noexcept
{
    constexpr double norm = st::normalization<sh_norm>();
    std::size_t min_order = std::min(out.order(), expansion.order());

    for (std::size_t n = 0; n < min_order; ++n)
    {
        auto expansion_n = expansion[n];
        auto out_n = out[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            auto& out_nl = out_n[l];
            if (indexing_mode == IndexingMode::negative)
            {
                out_nl = 0.0;
                for (auto m : expansion_nl.indices())
                    out_nl += expansion_nl[m]*expansion_nl[m];
            }
            else
            {
                out_nl = expansion_nl[0, 0]*expansion_nl[0, 0];
                for (auto m : expansion_nl.indices())
                    out_nl += expansion_nl[m, 0]*expansion_nl[m, 0]
                            + expansion_nl[m, 1]*expansion_nl[m, 1];
                out_nl *= norm;
            }
        }
    }
}

/**
    @brief Compute power spectrum of a Zernike expansions.

    @param expansion Zernike expansion

    @return `std::vector` storing the power spectrum.
*/
template <IndexingMode indexing_mode, ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase>
[[nodiscard]] std::vector<double>
power_spectrum(
    ZernikeSpan<const double, indexing_mode, zernike_norm, sh_norm, sh_phase>& expansion)
{
    using SpectrumSpan = RadialZernikeSpan<double, zernike_norm>;
    std::vector<double> res(SpectrumSpan::shape_type::size(expansion.order()));
    power_spectrum(
        std::forward(expansion), SpectrumSpan(res, expansion.order()));
    return res;
}

} // namespace zt

} // namespace zest
