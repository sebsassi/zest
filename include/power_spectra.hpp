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

#include <cstddef>
#include <vector>

#include "sh_concepts.hpp"
#include "sh_conventions.hpp"
#include "zernike_concepts.hpp"
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
template <any_sh_expansion ExpansionTypeA, compatible_with<ExpansionTypeA> ExpansionTypeB>
    requires std::floating_point<value_type_of<ExpansionTypeA>>
        && std::floating_point<value_type_of<ExpansionTypeB>>
        && st::has_inner_rank<ExpansionTypeA, 0>
        && st::has_inner_rank<ExpansionTypeB, 0>
void cross_power_spectrum(
    const ExpansionTypeA& a, const ExpansionTypeB& b, std::span<double> out) noexcept
{
    constexpr st::SHNorm sh_norm = sh_norm_of<ExpansionTypeA>();
    constexpr IndexingMode indexing_mode = indexing_mode_of<ExpansionTypeA>();

    std::size_t min_order
            = std::min(std::min(a.order(), b.order()), out.size());

    for (std::size_t l = 0; l < min_order; ++l)
    {
        auto a_l = a[l];
        auto b_l = b[l];
        auto& out_l = out[l];
        if constexpr (indexing_mode == IndexingMode::symmetric)
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
        }
        if constexpr (sh_norm == st::SHNorm::qm)
            out_l *= 1.0/(4.0*std::numbers::pi);
    }
}

/**
    @brief Compute cross power spectrum of two spherical harmonic expansions.

    @param a spherical harmonic expansion
    @param b spherical harmonic expansion
*/
template <any_sh_expansion ExpansionTypeA, compatible_with<ExpansionTypeA> ExpansionTypeB>
    requires std::floating_point<value_type_of<ExpansionTypeA>>
        && std::floating_point<value_type_of<ExpansionTypeB>>
        && st::has_inner_rank<ExpansionTypeA, 0>
        && st::has_inner_rank<ExpansionTypeB, 0>
[[nodiscard]] std::vector<double>
cross_power_spectrum(
    const ExpansionTypeA& a, const ExpansionTypeB& b) noexcept
{
    std::size_t min_order = std::min(a.order(), b.order());
    std::vector<double> res(min_order);
    cross_power_spectrum(a, b, res);
    return res;
}

/**
    @brief Compute power spectrum of a spherical harmonic expansions.

    @param expansion spherical harmonic expansion
    @param out output buffer for the power spectrum
*/
template <any_sh_expansion ExpansionType>
    requires std::floating_point<value_type_of<ExpansionType>>
        && st::has_inner_rank<ExpansionType, 0>
void power_spectrum(const ExpansionType& expansion, std::span<double> out) noexcept
{
    constexpr st::SHNorm sh_norm = sh_norm_of<ExpansionType>();
    constexpr IndexingMode indexing_mode = indexing_mode_of<ExpansionType>();

    std::size_t min_order = std::min(out.size(), expansion.order());

    for (std::size_t l = 0; l < min_order; ++l)
    {
        auto expansion_l = expansion[l];
        auto& out_l = out[l];
        if constexpr (indexing_mode == IndexingMode::symmetric)
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
        }
        if constexpr (sh_norm == st::SHNorm::qm)
            out_l *= 1.0/(4.0*std::numbers::pi);
    }
}

/**
    @brief Compute power spectrum of a spherical harmonic expansions.

    @param expansion spherical harmonic expansion

    @return `std::vector` storing the power spectrum
*/
template <any_sh_expansion ExpansionType>
    requires std::floating_point<value_type_of<ExpansionType>>
        && st::has_inner_rank<ExpansionType, 0>
[[nodiscard]] std::vector<double>
power_spectrum(const ExpansionType& expansion)
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
template <any_zernike_expansion ExpansionType>
    requires std::floating_point<value_type_of<ExpansionType>>
        && zt::has_inner_rank<ExpansionType, 0>
void power_spectrum(
    const ExpansionType& expansion,
    RadialZernikeSpan<double, zernike_norm_of<ExpansionType>()> out) noexcept
{
    constexpr st::SHNorm sh_norm = st::sh_norm_of<ExpansionType>();
    constexpr IndexingMode indexing_mode = indexing_mode_of<ExpansionType>();
    std::size_t min_order = std::min(out.order(), expansion.order());

    for (std::size_t n = 0; n < min_order; ++n)
    {
        auto expansion_n = expansion[n];
        auto out_n = out[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            auto& out_nl = out_n[l];
            if constexpr (indexing_mode == IndexingMode::symmetric)
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
            }
            if constexpr (sh_norm == st::SHNorm::qm)
                out_nl *= 3.0/(4.0*std::numbers::pi);
        }
    }
}

/**
    @brief Compute power spectrum of a Zernike expansions.

    @param expansion Zernike expansion

    @return `std::vector` storing the power spectrum.
*/
template <any_zernike_expansion ExpansionType>
    requires std::floating_point<value_type_of<ExpansionType>>
        && zt::has_inner_rank<ExpansionType, 0>
[[nodiscard]] RadialZernikeExpansion<double, zernike_norm_of<ExpansionType>()>
power_spectrum(const ExpansionType& expansion)
{
    using OutType = RadialZernikeExpansion<double, zernike_norm_of<ExpansionType>()>;
    OutType res{expansion.order()};
    power_spectrum(expansion, (typename OutType::view)(res));
    return res;
}

} // namespace zt

} // namespace zest
