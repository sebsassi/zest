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

#include <complex>
#include <utility>

#include "array_complex_view.hpp"
#include "sh_concepts.hpp"
#include "sh_conventions.hpp"
#include "sh_expansion.hpp"
#include "zernike_concepts.hpp"
#include "zernike_conventions.hpp"
#include "zernike_expansion.hpp"

namespace zest
{

/**
    @brief Convert real spherical harmonic expansion of a real function to a
    complex spherical harmonic expansion.

    @tparam ExpansionType Type of expansion.

    @param expansion

    @return Complex view of the expansion.

    @note IMPORTANT: This function modifies the input data! The output is just a
    new view over the same data.
*/
constexpr auto
encode_as_complex_expansion(
    st::complete_sh_expansion<Indexing::zero_based> auto&& expansion
) noexcept
    requires std::same_as<value_type_of<std::remove_reference_t<decltype(expansion)>>, double>
        && st::has_inner_rank<std::remove_reference_t<decltype(expansion)>, 0>
{
    using ReturnType = st::ComplexEncodedRealSHSpan<
        std::complex<double>, st::convention_of<std::remove_reference_t<decltype(expansion)>>>;

    constexpr double complex_conversion_norm = 1.0/std::numbers::sqrt2;

    for (auto l : std::forward<decltype(expansion)>(expansion).indices())
    {
        auto expansion_l = std::forward<decltype(expansion)>(expansion)[l];
        for (auto m : expansion_l.indices(1))
        {
            expansion_l[m, 0] *= complex_conversion_norm;
            expansion_l[m, 1] *= -complex_conversion_norm;
        }
    }

    return ReturnType{
            as_complex_span(std::forward<decltype(expansion)>(expansion).flatten()),
            std::forward<decltype(expansion)>(expansion).order()};
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a
    real spherical harmonic expansion.

    @tparam ExpansionType Type of expansion.

    @param expansion

    @return Real view of the expansion.

    @note IMPORTANT: This function modifies the input data! The output is just
    a new view over the same data.
*/
constexpr auto
decode_as_real_expansion(
    st::complex_encoded_real_sh_expansion auto&& expansion) noexcept
{
    using ReturnType = st::SHSpan<
            double, Indexing::zero_based,
            st::convention_of<std::remove_reference_t<decltype(expansion)>>>;

    constexpr double complex_conversion_norm = std::numbers::sqrt2;

    ReturnType res{
            as_float_span(std::forward<decltype(expansion)>(expansion).flatten()),
            std::forward<decltype(expansion)>(expansion).order()};

    for (auto l : res.indices())
    {
        auto res_l = res[l];
        for (auto m : res_l.indices(1))
        {
            res_l[m, 0] *= complex_conversion_norm;
            res_l[m, 1] *= -complex_conversion_norm;
        }
    }

    return res;
}

/**
    @brief Convert real Zernike expansion of a real function to a complex
    Zernike expansion.

    @tparam ExpansionType Type of expansion.

    @param expansion

    @return Complex view of the expansion.

    @note IMPORTANT This function modifies the input data and merely produces a
    new view over the same data.
*/
template <zt::anisotropic_zernike_expansion<Indexing::zero_based> ExpansionType>
constexpr auto
encode_as_complex_expansion(
    zt::anisotropic_zernike_expansion<Indexing::zero_based> auto&& expansion
) noexcept
    requires std::same_as<value_type_of<ExpansionType>, double>
        && zt::has_inner_rank<ExpansionType, 0>
{
    using ReturnType = zt::ComplexEncodedRealZernikeSpan<
        std::complex<double>, zt::convention_of<std::remove_reference_t<decltype(expansion)>>>;

    constexpr double complex_conversion_norm = 1.0/std::numbers::sqrt2;

    for (auto n : std::forward<decltype(expansion)>(expansion).indices())
    {
        auto expansion_n = std::forward<decltype(expansion)>(expansion)[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];

            for (auto m : expansion_nl.indices(1))
            {
                expansion_nl[m][0] *= complex_conversion_norm;
                expansion_nl[m][1] *= -complex_conversion_norm;
            }
        }
    }

    return ReturnType{
            as_complex_span(std::forward<decltype(expansion)>(expansion).flatten()),
            std::forward<decltype(expansion)>(expansion).order()};
}

/**
    @brief Convert complex Zernike expansion of a real function to a real
    Zernike expansion.

    @tparam ExpansionType Type of expansion.

    @param expansion

    @return Real view of the expansion.

    @note IMPORTANT This function modifies the input data and merely produces a
    new view over the same data.
*/
constexpr auto
decode_as_real_expansion(
    zt::complex_encoded_real_zernike_expansion auto&& expansion) noexcept
{
    using ReturnType = zt::ZernikeSpan<
        double, Indexing::zero_based,
        zt::convention_of<std::remove_reference_t<decltype(expansion)>>>;

    constexpr double complex_conversion_norm = std::numbers::sqrt2;

    ReturnType res{
            as_float_span(std::forward<decltype(expansion)>(expansion).flatten()),
            std::forward<decltype(expansion)>(expansion).order()};

    for (auto n : res.indices())
    {
        auto res_n = res[n];
        for (auto l : res_n.indices())
        {
            auto res_nl = res_n[l];

            for (auto m : res_nl.indices(1))
            {
                res_nl[m][0] *= complex_conversion_norm;
                res_nl[m][1] *= -complex_conversion_norm;
            }
        }
    }

    return res;
}

/**
    @brief Convert real spherical harmonic expansion of a real function to a
    complex spherical harmonic expansion.

    @tparam ExpansionType Type of expansion.

    @param expansion

    @return Complex view of the expansion.

    @note IMPORTANT This function modifies the input data and merely produces a
    new view over the same data.
*/
constexpr auto
encode_as_complex_expansion(
    st::zernike_sh_subspan<Indexing::zero_based> auto&& expansion
) noexcept
    requires std::same_as<value_type_of<std::remove_reference_t<decltype(expansion)>>, double>
        && st::has_inner_rank<std::remove_reference_t<decltype(expansion)>, 0>
{
    using ReturnType = typename zt::ComplexEncodedRealZernikeSpan<
            std::complex<double>, zt::convention_of<std::remove_reference_t<decltype(expansion)>>
        >::template subspan_type<1>;

    constexpr double complex_conversion_norm = 1.0/std::numbers::sqrt2;

    for (auto l : std::forward<decltype(expansion)>(expansion).indices())
    {
        auto expansion_l = expansion[l];

        for (auto m : expansion_l.indices(1))
        {
            expansion_l[m][0] *= complex_conversion_norm;
            expansion_l[m][1] *= -complex_conversion_norm;
        }
    }

    return ReturnType{
            as_complex_span(std::forward<decltype(expansion)>(expansion).flatten()),
            std::forward<decltype(expansion)>(expansion).order()};
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a
    real spherical harmonic expansion.

    @tparam ExpansionType Type of expansion.

    @param expansion

    @return Real view of the expansion.

    @note IMPORTANT This function modifies the input data and merely produces a
    new view over the same data.
*/
constexpr auto
decode_as_real_expansion(
    st::complex_encoded_zernike_sh_subspan auto&& expansion) noexcept
{
    using ReturnType = typename zt::ZernikeExpansion<
            double, Indexing::zero_based,
            zt::convention_of<std::remove_reference_t<decltype(expansion)>>
        >::template subspan_type<1>;

    constexpr double complex_conversion_norm = std::numbers::sqrt2;

    ReturnType res{
            as_float_span(std::forward<decltype(expansion)>(expansion).flatten()),
            std::forward<decltype(expansion)>(expansion).order()};

    for (auto l : res.indices())
    {
        auto res_l = res[l];

        for (auto m : res_l.indices(1))
        {
            res_l[m][0] *= complex_conversion_norm;
            res_l[m][1] *= -complex_conversion_norm;
        }
    }

    return res;
}

} // namespace zest
