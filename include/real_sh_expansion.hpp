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

#include <concepts>
#include <type_traits>

#include "sh_conventions.hpp"
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
    TriangleShape<IndexingMode::zero_based, Ns...>, SHTag<sh_norm_param, sh_phase_param>>;

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
    std::conditional_t<(indexing_mode_param == IndexingMode::symmetric),
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

template <
    complex_or_real_float ElementType, SHNorm sh_norm_param, SHPhase sh_phase_param,
    std::size_t... Ns
>
using AssociatedLegendreExpansion = ShapedArray<
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

} // namespace zest::st

