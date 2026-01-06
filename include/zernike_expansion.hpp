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

#include <type_traits>

#include "sequence.hpp"
#include "sh_conventions.hpp"
#include "shape.hpp"
#include "shaped_array.hpp"
#include "shaped_span.hpp"
#include "zernike_conventions.hpp"

namespace zest::zt
{

/**
    @brief Shape representing an efficient layout of 3D radial Zernike
    polynomials, tagged with normalization conventions.

    @tparam zernike_norm Zernike function normalization conventions.
    @tparam Ns Extents representing an inner multidimensional array structure.
*/
template <ZernikeNorm zernike_norm, std::size_t... Ns>
using RadialZernikeShape = TaggedShape<
    TensorSequenceShape<EvenTriangleSequence, Ns...>, ZernikeTag<zernike_norm>>;

/**
    @brief Shape representing an efficient layout of a multidimensional array
    of 3D radial Zernike polynomial sequences, tagged with normalization
    conventions.

    @tparam zernike_norm Zernike function normalization convention.
    @tparam Ns Extents representing the outer multidimensional array structure.
*/
template <ZernikeNorm zernike_norm, std::size_t... Ns>
using RadialZernikeTensorShape = TaggedShape<
    SequenceTensorShape<EvenTriangleSequence, Ns...>, ZernikeTag<zernike_norm>>;

/**
    @brief Shape representing an efficient layout of 3D Zernike functions,
    tagged with normalizaton and phase conventions.

    @tparam indexing_mode Determines azimuthal index layout.
    @tparam zernike_norm Radial Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam Ns Extents representing an inner multidimensional array structure.
*/
template <
    IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... Ns
>
using ZernikeShape = TaggedShape<
    std::conditional_t<(indexing_mode == IndexingMode::symmetric), 
        TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode>, Ns...>,
        TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode>, 2, Ns...>>,
    ZernikeTag<zernike_norm>, st::SHTag<sh_norm, sh_phase>>;

/**
    @brief Shape representing an efficient layout of a multidimensional array
    of 3D Zernike functions, tagged with normalization and phase conventions.

    @tparam indexing_mode Determines azimuthal index layout.
    @tparam zernike_norm Radial Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam Ns Extents representing an inner multidimensional array structure.
*/
template <
    IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... Ns
>
using ZernikeTensorShape = TaggedShape<
    std::conditional_t<(indexing_mode == IndexingMode::symmetric),
        SequenceTensorShape<ZernikeTetrahedralSequence<indexing_mode>, Ns...>,
        CompositeShape<
            TensorShape<Ns...>,
            TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode>, 2>>>,
        ZernikeTag<zernike_norm>, st::SHTag<sh_norm, sh_phase>>;

/**
    @brief Shape representing an efficient layout of 3D Zernike functions with
    only nonnegative azimuthal indices, tagged with normalization and phase
    conventions.

    @tparam zernike_norm Radial Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam Ns Extents representing an inner multidimensional array structure.
*/
template <
    ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... Ns
>
using ZernikeNonnegativeShape = TaggedShape<
    TensorSequenceShape<ZernikeTetrahedralSequence<IndexingMode::zero_based>, Ns...>,
    ZernikeTag<zernike_norm>, st::SHTag<sh_norm, sh_phase>>;

/**
    @brief A non-owning view of 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... Ns>
using RadialZernikeSpan = ShapedSpan<
    ElementType, RadialZernikeShape<zernike_norm, Ns...>>;

/**
    @brief A container of 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... Ns>
using RadialZernikeExpansion = ShapedArray<
    ElementType, RadialZernikeShape<zernike_norm, Ns...>>;

/**
    @brief A non-owning view of multidimensional arrays of 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... Ns>
using RadialZernikeTensorSpan = ShapedSpan<
    ElementType, RadialZernikeTensorShape<zernike_norm, Ns...>>;

/**
    @brief A container for storing multidimensional arrays of 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... Ns>
using RadialZernikeExpansionTensor = ShapedArray<
    ElementType, RadialZernikeTensorShape<zernike_norm, Ns...>>;

/**
    @brief A non-owning view of 3D Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <
    typename ElementType, IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... Ns
>
using ZernikeSpan = ShapedSpan<
    ElementType, ZernikeShape<indexing_mode, zernike_norm, sh_norm, sh_phase, Ns...>>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeSpanAcoustics = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeSpanNormalAcoustics = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeSpanQM = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeSpanNormalQM = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeSpanGeo = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeSpanNormalGeo = ZernikeSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none,
    Ns...>;

/**
    @brief A container for Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <
    typename ElementType, IndexingMode indexing_mode_param, ZernikeNorm zernike_norm_param,
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param, std::size_t... Ns
>
using ZernikeExpansion = ShapedArray<
    ElementType, ZernikeShape<indexing_mode_param, zernike_norm_param,
    sh_norm_param, sh_phase_param, Ns...>>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionAcoustics = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionNormalAcoustics = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam Ns extents representing inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionQM = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam Ns extents representing inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionNormalQM = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam Ns extents representing inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionGeo = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::geo, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam Ns extents representing inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionNormalGeo = ZernikeExpansion<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::geo, st::SHPhase::none, Ns...>;

/**
    @brief A non-owning view of a multidimensional array pf 3D Zernike function
    data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <
    typename ElementType, IndexingMode indexing_mode, ZernikeNorm zernike_norm,
    st::SHNorm sh_norm, st::SHPhase sh_phase, std::size_t... Ns
>
using ZernikeTensorSpan = ShapedSpan<
    ElementType, ZernikeTensorShape<indexing_mode, zernike_norm, sh_norm, sh_phase, Ns...>>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeTensorSpanAcoustics = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeTensorSpanNormalAcoustics = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeTensorSpanQM = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeTensorSpanNormalQM = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeTensorSpanGeo = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none,
    Ns...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode, std::size_t... Ns>
using ZernikeTensorSpanNormalGeo = ZernikeTensorSpan<
    ElementType, indexing_mode, ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none,
    Ns...>;

/**
    @brief A container for mutlidimensional arrays of Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <
    typename ElementType, IndexingMode indexing_mode_param, ZernikeNorm zernike_norm_param,
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param, std::size_t... Ns
>
using ZernikeExpansionTensor = ShapedArray<
    ElementType, ZernikeTensorShape<indexing_mode_param, zernike_norm_param,
    sh_norm_param, sh_phase_param, Ns...>>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionTensorAcoustics = ZernikeExpansionTensor<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionTensorNormalAcoustics = ZernikeExpansionTensor<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam Ns extents representing inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionTensorQM = ZernikeExpansionTensor<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::qm, st::SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam Ns extents representing inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionTensorNormalQM = ZernikeExpansionTensor<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::qm, st::SHPhase::cs, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam Ns extents representing inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionTensorGeo = ZernikeExpansionTensor<
    ElementType, indexing_mode_param, ZernikeNorm::unnormed,
    st::SHNorm::geo, st::SHPhase::none, Ns...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam Ns extents representing inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeExpansionTensorNormalGeo = ZernikeExpansionTensor<
    ElementType, indexing_mode_param, ZernikeNorm::normed,
    st::SHNorm::geo, st::SHPhase::none, Ns...>;

/**
    @brief A complex encoded view of real 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam indexing_mode Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam sh_norm Spherical harmonic normalization convention.
    @tparam sh_phase Spherical harmonic phase convention.
    @tparam Ns Extents representing inner multidimensional array structure.
*/
template <
    typename ElementType, ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase,
    std::size_t... Ns
>
using ComplexEncodedRealZernikeSpan = ShapedSpan<
    ElementType, ZernikeNonnegativeShape<zernike_norm, sh_norm, sh_phase, Ns...>>;

} // namespace zest::zt

