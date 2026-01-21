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

    @tparam dest_sh_norm normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note IMPORTANT: This function modifies the input data! The output is just
    a new view over the same data.
*/
template <st::complete_sh_expansion<IndexingMode::zero_based> ExpansionType>
    requires std::same_as<typename std::remove_cvref_t<ExpansionType>::value_type, double>
constexpr st::ComplexEncodedRealSHSpan<
    std::complex<double>, st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()>
encode_as_complex_expansion(ExpansionType&& expansion) noexcept
{
    using ReturnType = st::ComplexEncodedRealSHSpan<
        std::complex<double>, st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()>;

    constexpr double complex_conversion_norm = 1.0/std::numbers::sqrt2;

    for (auto l : std::forward<ExpansionType>(expansion).indices())
    {
        auto expansion_l = std::forward<ExpansionType>(expansion)[l];
        for (auto m : expansion_l.indices(1))
        {
            expansion_l[m, 0] *= complex_conversion_norm;
            expansion_l[m, 1] *= -complex_conversion_norm;
        }
    }

    return ReturnType(
            as_complex_span(std::forward<ExpansionType>(expansion).flatten()),
            std::forward<ExpansionType>(expansion).order());
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a
    real spherical harmonic expansion.

    @tparam dest_sh_norm normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note IMPORTANT: This function modifies the input data! The output is just
    a new view over the same data.
*/
template <st::complex_encoded_real_sh_expansion ExpansionType>
constexpr st::SHSpan<
    double, IndexingMode::zero_based,
    st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()>
decode_as_real_expansion(ExpansionType&& expansion) noexcept
{
    using ReturnType = st::SHSpan<
        double, IndexingMode::zero_based,
        st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()>;

    constexpr double complex_conversion_norm = std::numbers::sqrt2;

    ReturnType res(
            as_float_span(std::forward<ExpansionType>(expansion).flatten()),
            std::forward<ExpansionType>(expansion).order());

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
    @brief Convert real Zernike expansion of a real function to a complex Zernike expansion.

    @tparam dest_zernike_norm Zernike normalization convention of the output view
    @tparam dest_sh_norm spherical harmonic normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion Zernike expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <zt::zernike_expansion<IndexingMode::zero_based> ExpansionType>
    requires std::same_as<typename std::remove_cvref_t<ExpansionType>::value_type, double>
constexpr zt::ComplexEncodedRealZernikeSpan<
    std::complex<double>, zt::zernike_norm_of<ExpansionType>(),
    st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()>
encode_as_complex_expansion(ExpansionType&& expansion) noexcept
{
    using ReturnType = zt::ComplexEncodedRealZernikeSpan<
        std::complex<double>, zt::zernike_norm_of<ExpansionType>(),
        st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()>;

    constexpr double complex_conversion_norm = 1.0/std::numbers::sqrt2;

    for (auto n : std::forward<ExpansionType>(expansion).indices())
    {
        auto expansion_n = std::forward<ExpansionType>(expansion)[n];
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

    return ReturnType(
            as_complex_span(std::forward<ExpansionType>(expansion).flatten()),
            std::forward<ExpansionType>(expansion).order());
}

/**
    @brief Convert complex Zernike expansion of a real function to a real Zernike expansion.

    @tparam dest_zernike_norm Zernike normalization convention of the output view
    @tparam dest_sh_norm spherical harmonic normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion Zernike expansion

    @return view of the expansion transformed to a real expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <zt::complex_encoded_real_zernike_expansion ExpansionType>
constexpr zt::ZernikeSpan<
    double, IndexingMode::zero_based, zt::zernike_norm_of<ExpansionType>(),
    st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()>
decode_as_real_expansion(ExpansionType&& expansion) noexcept
{
    using ReturnType = zt::ZernikeSpan<
        double, IndexingMode::zero_based, zt::zernike_norm_of<ExpansionType>(),
        st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()>;

    constexpr double complex_conversion_norm = std::numbers::sqrt2;
    constexpr double shcnorm = complex_conversion_norm;

    ReturnType res(
            as_float_span(std::forward<ExpansionType>(expansion).flatten()),
            std::forward<ExpansionType>(expansion).order());

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
    @brief Convert real spherical harmonic expansion of a real function to a complex spherical harmonic expansion.

    @tparam DEST_NORM normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <st::zernike_sh_subspan<IndexingMode::zero_based> ExpansionType>
    requires std::same_as<typename std::remove_cvref_t<ExpansionType>::value_type, double>
constexpr typename zt::ComplexEncodedRealZernikeSpan<
    std::complex<double>, zt::zernike_norm_of<ExpansionType>(),
    st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()
>::template subspan_type<1>
encode_as_complex_expansion(ExpansionType&& expansion) noexcept
{
    using ReturnType = typename zt::ComplexEncodedRealZernikeSpan<
            std::complex<double>, zt::zernike_norm_of<ExpansionType>(),
            st::sh_norm_of<ExpansionType>(), st::sh_phase_of<ExpansionType>()
        >::template subspan_type<1>;

    constexpr double complex_conversion_norm = 1.0/std::numbers::sqrt2;

    for (auto l : std::forward<ExpansionType>(expansion).indices())
    {
        auto expansion_l = expansion[l];

        for (auto m : expansion_l.indices(1))
        {
            expansion_l[m][0] *= complex_conversion_norm;
            expansion_l[m][1] *= -complex_conversion_norm;
        }
    }

    return ReturnType(
            as_complex_span(std::forward<ExpansionType>(expansion).flatten()),
            std::forward<ExpansionType>(expansion).order());
}

/**
    @brief Convert complex spherical harmonic expansion of a real function to a real spherical harmonic expansion.

    @tparam DEST_NORM normalization convention of the output view
    @tparam dest_sh_phase phase convention of the output view

    @param expansion spherical harmonic expansion

    @return view of the expansion transformed to a complex expansion

    @note This function modifies the input data and merely produces a new view over the same data.
*/
template <
    zt::ZernikeNorm dest_zernike_norm, st::SHNorm dest_sh_norm, st::SHPhase dest_sh_phase, 
    zt::ZernikeNorm source_zernike_norm, st::SHNorm source_sh_norm, st::SHPhase source_sh_phase
>
constexpr typename zt::ZernikeExpansion<
    double, IndexingMode::zero_based, source_zernike_norm, source_sh_norm, source_sh_phase
>::template subspan_type<1>
decode_as_real_expansion(
    typename zt::ComplexEncodedRealZernikeSpan<
        std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
    >::template subspan_type<1> expansion) noexcept
{
    using ExpansionType = typename zt::ComplexEncodedRealZernikeSpan<
            std::complex<double>, source_zernike_norm, dest_sh_norm, dest_sh_phase
        >::template subspan_type<1>;
    using ReturnType = typename zt::ZernikeExpansion<
            double, IndexingMode::zero_based, dest_zernike_norm, source_sh_norm, source_sh_phase
        >::template subspan_type<1>;

    constexpr double shnorm = st::conversion_const<source_sh_norm, dest_sh_norm>();
    constexpr double complex_conversion_norm = std::numbers::sqrt2;
    constexpr double shcnorm = shnorm*complex_conversion_norm;

    // NOTE: When `expansion.order() == 0`, the argument becomes `0 - 1`.
    // But this is fine because this operation is well-defined for unsigned
    // integers, and the result is never used for anything, because there will
    // be zero loop iterations.
    const double znorm = zt::conversion_factor<source_zernike_norm, dest_zernike_norm>(expansion.order() - 1UL);
    const double zshnorm = shnorm*znorm;
    const double zshcnorm = shcnorm*znorm;

    ReturnType res(as_float_span(expansion.flatten()), expansion.order());

    for (auto l : res.indices())
    {
        auto res_l = res[l];
        res_l[0][0] *= zshnorm;
        res_l[0][1] *= zshnorm;

        if constexpr (dest_sh_phase == source_sh_phase)
        {
            for (auto m : res_l.indices(1))
            {
                res_l[m][0] *= zshcnorm;
                res_l[m][1] *= -zshcnorm;
            }
        }
        else
        {
            double prefactor = zshcnorm;
            for (auto m : res_l.indices(1))
            {
                prefactor *= -1.0;
                res_l[m][0] *= prefactor;
                res_l[m][1] *= -prefactor;
            }
        }
    }

    return res;
}

} // namespace zest
