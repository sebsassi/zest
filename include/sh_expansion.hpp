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
#include <span>

#include "sh_conventions.hpp"
#include "sh_shapes.hpp"
#include "shaped_array.hpp"
#include "shaped_span.hpp"

namespace zest::st
{

/**
    @brief A non-owning view of associated Legendre function data.

    @tparam ElementType Type of elements.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase phase Convention of the spherical harmonics.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, SHNorm sh_norm, SHPhase sh_phase,
    std::size_t... inner_extents
>
using AssociatedLegendreSpan = ShapedSpan<
    ElementType, AssociatedLegendreShape<sh_norm, sh_phase, inner_extents...>>;

/**
    @brief A container for associated Legendre function data.

    @tparam ElementType Type of elements.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, SHNorm sh_norm, SHPhase sh_phase,
    std::size_t... inner_extents
>
using AssociatedLegendreExpansion = ShapedArray<
    ElementType, AssociatedLegendreShape<sh_norm, sh_phase, inner_extents...>>;

/**
    @brief A non-owning view of a multidimensional array of associated Legendre
    function data.

    @tparam ElementType Type of elements.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, SHNorm sh_norm, SHPhase sh_phase,
    std::size_t... outer_extents
>
using AssociatedLegendreTensorSpan = ShapedSpan<
    ElementType, AssociatedLegendreTensorShape<sh_norm, sh_phase, outer_extents...>>;

/**
    @brief A non-owning view of a vector of associated Legendre
    function data.

    @tparam ElementType Type of elements.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
*/
template <
    complex_or_real_float ElementType, SHNorm sh_norm, SHPhase sh_phase
>
using AssociatedLegendreVectorSpan = AssociatedLegendreTensorSpan<
    ElementType, sh_norm, sh_phase, std::dynamic_extent>;

/**
    @brief A container for a multidimensional array of associated Legendre
    function data.

    @tparam ElementType Type of elements.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, SHNorm sh_norm, SHPhase sh_phase,
    std::size_t... outer_extents
>
using AssociatedLegendreExpansionTensor = ShapedArray<
    ElementType, AssociatedLegendreTensorShape<sh_norm, sh_phase, outer_extents...>>;

/**
    @brief A container for a vector of associated Legendre function data.

    @tparam ElementType Type of elements.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
*/
template <
    complex_or_real_float ElementType, SHNorm sh_norm, SHPhase sh_phase
>
using AssociatedLegendreExpansionVector = AssociatedLegendreExpansionTensor<
    ElementType, sh_norm, sh_phase, std::dynamic_extent>;

/**
    @brief A non-owning view of spherical harmonic data.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order
    @tparam sh_norm Normalization convention of the spherical harmonics
    @tparam sh_phase Phase convention of the spherical harmonics
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    SHNorm sh_norm, SHPhase sh_phase, std::size_t... inner_extents
>
using SHSpan = ShapedSpan<
    ElementType, SHShape<indexing_mode, sh_norm, sh_phase, inner_extents...>>;

/**
    @brief Convenient alias for `SHSpan` with orthonormal spherical harmonics
    and no Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... inner_extents
>
using SHSpanAcoustics = SHSpan<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::none, inner_extents...>;
/**
    @brief Convenient alias for `SHSpan` with orthonormal spherical harmonics
    with Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... inner_extents
>
using SHSpanQM = SHSpan<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::cs, inner_extents...>;

/**
    @brief Convenient alias for `SHSpan` with 4-pi normal spherical harmonics
    and no Condon-Shortley phase

    @tparam ElementType type of elements
    @tparam indexing_mode determines azimuthal index order
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... Ns
>
using SHSpanGeo = SHSpan<
    ElementType, indexing_mode, SHNorm::geo, SHPhase::none, Ns...>;

/**
    @brief A container for spherical harmonic data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    SHNorm sh_norm, SHPhase sh_phase, std::size_t... inner_extents
>
using SHExpansion = ShapedArray<
    ElementType, SHShape<indexing_mode, sh_norm, sh_phase, inner_extents...>>;

/**
    @brief Convenient alias for `SHExpansion` with orthonormal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... inner_extents
>
using SHExpansionAcoustics = SHExpansion<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::none, inner_extents...>;

/**
    @brief Convenient alias for `SHExpansion` with orthonormal spherical
    harmonics with Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... inner_extents
>
using SHExpansionQM = SHExpansion<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::cs, inner_extents...>;

/**
    @brief Convenient alias for `SHExpansion` with 4-pi normal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... inner_extents
>
using SHExpansionGeo = SHExpansion<
    ElementType, indexing_mode, SHNorm::geo, SHPhase::none, inner_extents...>;

/**
    @brief A non-owning view of a multidimensional array of spherical harmonic
    data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    SHNorm sh_norm, SHPhase sh_phase, std::size_t... outer_extents
>
using SHTensorSpan = ShapedSpan<
    ElementType, SHTensorShape<indexing_mode, sh_norm, sh_phase, outer_extents...>>;

/**
    @brief Convenient alias for `SHTensorSpan` with orthonormal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... outer_extents
>
using SHTensorSpanAcoustics = SHTensorSpan<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::none, outer_extents...>;
/**
    @brief Convenient alias for `SHTensorSpan` with orthonormal spherical
    harmonics with Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... outer_extents
>
using SHTensorSpanQM = SHTensorSpan<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::cs, outer_extents...>;

/**
    @brief Convenient alias for `SHTensorSpan` with 4-pi normal spherical
    harmonics and no Condon-Shortley phase

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... outer_extents
>
using SHTensorSpanGeo = SHTensorSpan<
    ElementType, indexing_mode, SHNorm::geo, SHPhase::none, outer_extents...>;

/**
    @brief A non-owning view of a vector of spherical harmonic data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    SHNorm sh_norm, SHPhase sh_phase
>
using SHVectorSpan = SHTensorSpan<
    ElementType, indexing_mode, sh_norm, sh_phase, std::dynamic_extent>;

/**
    @brief Convenient alias for `SHVectorSpan` with orthonormal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode
>
using SHVectorSpanAcoustics = SHVectorSpan<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::none>;
/**
    @brief Convenient alias for `SHVectorSpan` with orthonormal spherical
    harmonics with Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode
>
using SHVectorSpanQM = SHVectorSpan<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::cs>;

/**
    @brief Convenient alias for `SHVectorSpan` with 4-pi normal spherical
    harmonics and no Condon-Shortley phase

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode
>
using SHVectorSpanGeo = SHVectorSpan<
    ElementType, indexing_mode, SHNorm::geo, SHPhase::none>;

/**
    @brief A container for a multidimensional array of spherical harmonic data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    SHNorm sh_norm, SHPhase sh_phase, std::size_t... outer_extents
>
using SHExpansionTensor = ShapedArray<
    ElementType, SHTensorShape<indexing_mode, sh_norm, sh_phase, outer_extents...>>;

/**
    @brief Convenient alias for `SHExpansionTensor` with orthonormal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... outer_extents
>
using SHExpansionTensorAcoustics = SHExpansionTensor<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::none, outer_extents...>;

/**
    @brief Convenient alias for `SHExpansionTensor` with orthonormal spherical
    harmonics with Condon-Shortley phase.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... outer_extents
>
using SHExpansionTensorQM = SHExpansionTensor<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::cs, outer_extents...>;

/**
    @brief Convenient alias for `SHExpansionTensor` with 4-pi normal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    std::size_t... outer_extents
>
using SHExpansionTensorGeo = SHExpansionTensor<
    ElementType, indexing_mode, SHNorm::geo, SHPhase::none, outer_extents...>;

/**
    @brief A container for a vector of spherical harmonic data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode,
    SHNorm sh_norm, SHPhase sh_phase
>
using SHExpansionVector = SHExpansionTensor<
    ElementType, indexing_mode, sh_norm, sh_phase, std::dynamic_extent>;

/**
    @brief Convenient alias for `SHExpansionVector` with orthonormal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode
>
using SHExpansionVectorAcoustics = SHExpansionVector<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::none>;

/**
    @brief Convenient alias for `SHExpansionVector` with orthonormal spherical
    harmonics with Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode
>
using SHExpansionVectorQM = SHExpansionVector<
    ElementType, indexing_mode, SHNorm::qm, SHPhase::cs>;

/**
    @brief Convenient alias for `SHExpansionVector` with 4-pi normal spherical
    harmonics and no Condon-Shortley phase.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <
    complex_or_real_float ElementType, IndexingMode indexing_mode
>
using SHExpansionVectorGeo = SHExpansionVector<
    ElementType, indexing_mode, SHNorm::geo, SHPhase::none>;

/**
    @brief A non-owning view for storing real spherical harmonic data encoded
    as complex numbers.

    @tparam ElementType Type of elements.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
*/
template <complex_float ElementType, SHNorm sh_norm, SHPhase sh_phase>
using ComplexEncodedRealSHSpan = ShapedSpan<
    ElementType, AssociatedLegendreShape<sh_norm, sh_phase>>;

} // namespace zest::st
