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

#include <complex>
#include <concepts>
#include <type_traits>

#include "sh_conventions.hpp"
#include "array_complex_view.hpp"
#include "shaped_array.hpp"
#include "shaped_span.hpp"
#include "spans.hpp"


namespace zest::st
{

/**
    @brief Tagged shape represnting layout and conventions of associated
    Legendre function values.

    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns>
using AssociatedLegendreShape = TaggedShape<
    TriangleShape<IndexingMode::nonnegative, Ns...>, SHTag<sh_norm_param, sh_phase_param>>;

/**
    @brief Tagged shape representing layout and conventions of spherical
    harmonic data.

    @tparam IndexingMode determines azimuthal index order
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    IndexingMode indexing_mode_param, SHNorm sh_norm_param, SHPhase sh_phase_param,
    std::size_t... Ns
>
using SHShape = TaggedShape<
    std::conditional_t<(indexing_mode_param == IndexingMode::negative),
        TriangleShape<indexing_mode_param, Ns...>,
        TriangleShape<indexing_mode_param, 2, Ns...>>,
    SHTag<sh_norm_param, sh_phase_param>>;

/**
    @brief A non-owning view for storing spherical harmonic data.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode_param,
    SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns
>
using SHSpan = ShapedSpan<
    ElementType, SHShape<indexing_mode_param, sh_norm_param, sh_phase_param, Ns...>>;

/**
    @brief A non-owning view for storing real spherical harmonic data encoded
    as complex numbers.

    @tparam ElementType type of elements
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <complex_float ElementType, SHNorm sh_norm_param, SHPhase sh_phase_param>
using ComplexEncodedRealSHSpan = ShapedSpan<
    ElementType, AssociatedLegendreShape<sh_norm_param, sh_phase_param>>;

template <
    complex_or_real_float ElementType, SHNorm sh_norm_param, SHPhase sh_phase_param,
    std::size_t... Ns
>
using AssociatedLegendreSpan = ShapedSpan<
    ElementType, AssociatedLegendreShape<sh_norm_param, sh_phase_param, Ns...>>;

/**
    @brief Convenient alias for `RealSHSpan` with orthonormal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode_param,
    std::size_t... Ns
>
using SHSpanAcoustics = SHSpan<
    ElementType, indexing_mode_param, SHNorm::qm, SHPhase::none, Ns...>;
/**
    @brief Convenient alias for `RealSHSpan` with orthonormal spherical
    harmonics with Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode_param,
    std::size_t... Ns
>
using SHSpanQM = SHSpan<ElementType, indexing_mode_param, SHNorm::qm, SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `RealSHSpan` with 4-pi normal spherical
    harmonics and no Condon-Shortley phase

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode_param,
    std::size_t... Ns
>
using SHSpanGeo = SHSpan<ElementType, indexing_mode_param, SHNorm::geo, SHPhase::none, Ns...>;

/**
    @brief A container for spherical harmonic data.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode_param,
    SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns>
using SHExpansion = ShapedArray<
    ElementType, SHShape<indexing_mode_param, sh_norm_param, sh_phase_param, Ns...>>;

/**
    @brief Convenient alias for `SHExpansion` with orthonormal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode_param,
    std::size_t... Ns
>
using SHExpansionAcoustics = SHExpansion<
    ElementType, indexing_mode_param, SHNorm::qm, SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `SHExpansion` with orthonormal spherical
    harmonics with Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode_param,
    std::size_t... Ns
>
using SHExpansionQM = SHExpansion<
    ElementType, indexing_mode_param, SHNorm::qm, SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `SHExpansion` with 4-pi normal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType type of elements
    @tparam IndexingMode determines azimuthal index order
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode_param,
    std::size_t... Ns
>
using SHExpansionGeo = SHExpansion<
    ElementType, indexing_mode_param, SHNorm::geo, SHPhase::none, Ns...>;

namespace detail
{

template <typename T>
concept has_sh_conventions = std::derived_from<T, SHTag<T::sh_norm, T::sh_phase>>;

} // namespace detail

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
template <
    SHNorm dest_sh_norm, SHPhase dest_sh_phase, SHNorm source_sh_norm, SHPhase source_sh_phase
>
constexpr ComplexEncodedRealSHSpan<std::complex<double>, dest_sh_norm, dest_sh_phase>
to_complex_expansion(
    SHSpan<double, IndexingMode::nonnegative, source_sh_norm, source_sh_phase>& expansion) noexcept
{
    using ExpansionType = SHSpan<
        double, IndexingMode::nonnegative, source_sh_norm, source_sh_phase>;
    using ReturnType = ComplexEncodedRealSHSpan<
        std::complex<double>, dest_sh_norm, dest_sh_phase>;

    constexpr double sh_norm
        = conversion_const<std::remove_cvref_t<ExpansionType>::shape::sh_norm, dest_sh_norm>();
    constexpr double cnorm = 1.0/std::numbers::sqrt2;
    constexpr double norm = sh_norm*cnorm;

    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        expansion_l[0, 0] *= sh_norm;
        expansion_l[0, 1] *= sh_norm;

        if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::shape::sh_phase)
        {
            for (auto m : expansion_l.indices(1))
            {
                expansion_l[m, 0] *= norm;
                expansion_l[m, 1] *= -norm;
            }
        }
        else
        {
            double prefactor = norm;
            for (auto m : expansion_l.indices(1))
            {
                prefactor *= -1.0;
                expansion_l[m, 0] *= prefactor;
                expansion_l[m, 1] *= -prefactor;
            }
        }
    }

    return ReturnType(as_complex_span(expansion.flatten()), expansion.order());
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
template <
    SHNorm dest_sh_norm, SHPhase dest_sh_phase, SHNorm source_sh_norm, SHPhase source_sh_phase
>
constexpr SHSpan<double, IndexingMode::nonnegative, dest_sh_norm, dest_sh_phase>
to_real_expansion(
    ComplexEncodedRealSHSpan<std::complex<double>, source_sh_norm, source_sh_phase>& expansion) noexcept
{
    using ExpansionType = ComplexEncodedRealSHSpan<
        std::complex<double>, source_sh_norm, source_sh_phase>;
    using ReturnType = SHSpan<
        double, IndexingMode::nonnegative, dest_sh_norm, dest_sh_phase>;

    constexpr double sh_norm
        = conversion_const<std::remove_cvref_t<ExpansionType>::shape::sh_norm, dest_sh_norm>();
    constexpr double cnorm = std::numbers::sqrt2;
    constexpr double norm = sh_norm*cnorm;

    ReturnType res(as_float_span(expansion.flatten()), expansion.order());

    for (auto l : res.indices())
    {
        auto res_l = res[l];
        res_l[0, 0] *= sh_norm;
        res_l[0, 1] *= sh_norm;

        if constexpr (dest_sh_phase == std::remove_cvref_t<ExpansionType>::shape::phase)
        {
            for (auto m : res_l.indices(1))
            {
                res_l[m, 0] *= norm;
                res_l[m, 1] *= -norm;
            }
        }
        else
        {
            double prefactor = norm;
            for (auto m : res_l.indices(1))
            {
                prefactor *= -1.0;
                res_l[m, 0] *= prefactor;
                res_l[m, 1] *= -prefactor;
            }
        }
    }

    return res;
}

} // namespace zest::st

