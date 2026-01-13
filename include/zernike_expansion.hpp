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

#include "sequence.hpp"
#include "sh_conventions.hpp"
#include "shaped_array.hpp"
#include "shaped_span.hpp"
#include "zernike_conventions.hpp"
#include "zernike_shapes.hpp"

namespace zest::zt
{

/**
    @brief A non-owning view of 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... inner_extents>
using RadialZernikeSpan = ShapedSpan<
    ElementType, RadialZernikeShape<zernike_norm, inner_extents...>>;

/**
    @brief A container of 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... inner_extents>
using RadialZernikeExpansion = ShapedArray<
    ElementType, RadialZernikeShape<zernike_norm, inner_extents...>>;

/**
    @brief A non-owning view of multidimensional arrays of 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... outer_extents>
using RadialZernikeTensorSpan = ShapedSpan<
    ElementType, RadialZernikeTensorShape<zernike_norm, outer_extents...>>;

/**
    @brief A non-owning view of a vector of 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
*/
template <typename ElementType, ZernikeNorm zernike_norm>
using RadialZernikeVectorSpan = RadialZernikeTensorSpan<
    ElementType, zernike_norm, std::dynamic_extent>;

/**
    @brief A container for storing multidimensional arrays of 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... outer_extents>
using RadialZernikeExpansionTensor = ShapedArray<
    ElementType, RadialZernikeTensorShape<zernike_norm, outer_extents...>>;

/**
    @brief A container for storing a vector of 3D radial Zernike polynomial
    data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
*/
template <typename ElementType, ZernikeNorm zernike_norm>
using RadialZernikeExpansionVector = RadialZernikeExpansionTensor<
    ElementType, zernike_norm, std::dynamic_extent>;

/**
    @brief A non-owning view of 3D Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... inner_extents
>
using ZernikeSpan = ShapedSpan<
    ElementType, ZernikeShape<indexing_mode, zernike_norm, sh_norm, sh_phase, inner_extents...>>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeSpanAcoustics = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeSpanNormalAcoustics = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeSpanQM = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs,
    inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeSpanNormalQM = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs,
    inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeSpanGeo = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none,
    inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeSpanNormalGeo = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none,
    inner_extents...>;

/**
    @brief A container for Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... inner_extents
>
using ZernikeExpansion = ShapedArray<
    ElementType, ZernikeShape<indexing_mode, zernike_norm,
    sh_norm, sh_phase, inner_extents...>>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeExpansionAcoustics = ZernikeExpansion<
    ElementType, indexing_mode, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::none, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeExpansionNormalAcoustics = ZernikeExpansion<
    ElementType, indexing_mode, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::none, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeExpansionQM = ZernikeExpansion<
    ElementType, indexing_mode, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::cs, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeExpansionNormalQM = ZernikeExpansion<
    ElementType, indexing_mode, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::cs, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeExpansionGeo = ZernikeExpansion<
    ElementType, indexing_mode, ZernikeNorm::unnormed,
    st::SHNorm::geo, st::SHPhase::none, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... inner_extents>
using ZernikeExpansionNormalGeo = ZernikeExpansion<
    ElementType, indexing_mode, ZernikeNorm::normed,
    st::SHNorm::geo, st::SHPhase::none, inner_extents...>;

/**
    @brief A non-owning view of a multidimensional array pf 3D Zernike function
    data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... outer_extents
>
using ZernikeTensorSpan = ShapedSpan<
    ElementType, ZernikeTensorShape<indexing_mode, zernike_norm, sh_norm, sh_phase, outer_extents...>>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeTensorSpanAcoustics = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeTensorSpanNormalAcoustics = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeTensorSpanQM = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeTensorSpanNormalQM = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeTensorSpanGeo = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeTensorSpanNormalGeo = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none,
    outer_extents...>;

/**
    @brief A non-owning view of a vector of 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
*/
template <
    typename ElementType, IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase
>
using ZernikeVectorSpan = ZernikeTensorSpan<
    ElementType, indexing_mode, zernike_norm, sh_norm, sh_phase, std::dynamic_extent>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeVectorSpanAcoustics = ZernikeVectorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeVectorSpanNormalAcoustics = ZernikeVectorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeVectorSpanQM = ZernikeVectorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeVectorSpanNormalQM = ZernikeVectorSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeVectorSpanGeo = ZernikeVectorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeVectorSpanNormalGeo = ZernikeVectorSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief A container for mutlidimensional arrays of Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... outer_extents
>
using ZernikeExpansionTensor = ShapedArray<
    ElementType, ZernikeTensorShape<indexing_mode, zernike_norm, sh_norm, sh_phase, outer_extents...>>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeExpansionTensorAcoustics = ZernikeExpansionTensor<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeExpansionTensorNormalAcoustics = ZernikeExpansionTensor<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::none,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeExpansionTensorQM = ZernikeExpansionTensor<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeExpansionTensorNormalQM = ZernikeExpansionTensor<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam outer_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeExpansionTensorGeo = ZernikeExpansionTensor<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none,
    outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... outer_extents>
using ZernikeExpansionTensorNormalGeo = ZernikeExpansionTensor<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none,
    outer_extents...>;

/**
    @brief A container for a vector of Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
*/
template <
    typename ElementType, IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase
>
using ZernikeExpansionVector = ZernikeExpansionTensor<
    ElementType, indexing_mode, zernike_norm, sh_norm, sh_phase, std::dynamic_extent>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeExpansionVectorAcoustics = ZernikeExpansionVector<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeExpansionVectorNormalAcoustics = ZernikeExpansionVector<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam outer_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeExpansionVectorQM = ZernikeExpansionVector<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeExpansionVectorNormalQM = ZernikeExpansionVector<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam outer_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeExpansionVectorGeo = ZernikeExpansionVector<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
*/
template <typename ElementType, IndexingMode indexing_mode>
using ZernikeExpansionVectorNormalGeo = ZernikeExpansionVector<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none>;

/**
    @brief A complex encoded view of real 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase,
    std::size_t... inner_extents
>
using ComplexEncodedRealZernikeSpan = ShapedSpan<
    ElementType, ZernikeNonnegativeShape<zernike_norm, sh_norm, sh_phase, inner_extents...>>;

} // namespace zest::zt
