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
    @brief A non-owning view of isotropic 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... inner_extents>
using IsotropicRadialZernikeSpan = ShapedSpan<
    ElementType, IsotropicRadialZernikeShape<zernike_norm, inner_extents...>>;

/**
    @brief A container of isotropic 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... inner_extents>
using IsotropicRadialZernikeExpansion = ShapedArray<
    ElementType, IsotropicRadialZernikeShape<zernike_norm, inner_extents...>>;

/**
    @brief A non-owning view of multidimensional arrays of isotropic 3D radial
    Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... outer_extents>
using IsotropicRadialZernikeTensorSpan = ShapedSpan<
    ElementType, IsotropicRadialZernikeTensorShape<zernike_norm, outer_extents...>>;

/**
    @brief A non-owning view of a vector of isotropic 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
*/
template <typename ElementType, ZernikeNorm zernike_norm>
using IsotropicRadialZernikeVectorSpan = IsotropicRadialZernikeTensorSpan<
    ElementType, zernike_norm, std::dynamic_extent>;

/**
    @brief A container for storing multidimensional arrays of isotropic 3D
    radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, ZernikeNorm zernike_norm, std::size_t... outer_extents>
using IsotropicRadialZernikeExpansionTensor = ShapedArray<
    ElementType, IsotropicRadialZernikeTensorShape<zernike_norm, outer_extents...>>;

/**
    @brief A container for storing a vector of isotropic 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
*/
template <typename ElementType, ZernikeNorm zernike_norm>
using IsotropicRadialZernikeExpansionVector = IsotropicRadialZernikeExpansionTensor<
    ElementType, zernike_norm, std::dynamic_extent>;

/**
    @brief A non-owning view of 3D Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, Indexing indexing, ZernikeNorm zernike_norm,
    st::sh_convention Convention, std::size_t... inner_extents
>
using ZernikeSpan = ShapedSpan<
    ElementType,
    ZernikeShape<
        indexing, zernike_norm, Convention, inner_extents...>>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanAcoustics = ZernikeSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics,
    inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanNormalAcoustics = ZernikeSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanQM = ZernikeSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::QM, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanNormalQM = ZernikeSpan<
    ElementType, indexing, ZernikeNorm::normed, st::QM, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanGeo = ZernikeSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Geo, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanNormalGeo = ZernikeSpan<
    ElementType, indexing, ZernikeNorm::normed, st::Geo, inner_extents...>;

/**
    @brief A container for Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, Indexing indexing, ZernikeNorm zernike_norm,
    st::sh_convention Convention, std::size_t... inner_extents
>
using ZernikeExpansion = ShapedArray<
    ElementType, ZernikeShape<indexing, zernike_norm, Convention, inner_extents...>>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionAcoustics = ZernikeExpansion<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionNormalAcoustics = ZernikeExpansion<
    ElementType, indexing, ZernikeNorm::normed, st::Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionQM = ZernikeExpansion<
    ElementType, indexing, ZernikeNorm::unnormed, st::QM, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionNormalQM = ZernikeExpansion<
    ElementType, indexing, ZernikeNorm::normed, st::QM, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionGeo = ZernikeExpansion<
    ElementType, indexing, ZernikeNorm::unnormed, st::Geo, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionNormalGeo = ZernikeExpansion<
    ElementType, indexing, ZernikeNorm::normed, st::Geo, inner_extents...>;

/**
    @brief A non-owning view of a multidimensional array pf 3D Zernike function
    data.

    @tparam ElementType Type of elements.
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, Indexing indexing, ZernikeNorm zernike_norm,
    st::sh_convention Convention, std::size_t... outer_extents
>
using ZernikeTensorSpan = ShapedSpan<
    ElementType,
    ZernikeTensorShape<indexing, zernike_norm, Convention, outer_extents...>>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanAcoustics = ZernikeTensorSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanNormalAcoustics = ZernikeTensorSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanQM = ZernikeTensorSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::QM, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanNormalQM = ZernikeTensorSpan<
    ElementType, indexing, ZernikeNorm::normed, st::QM, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanGeo = ZernikeTensorSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Geo, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanNormalGeo = ZernikeTensorSpan<
    ElementType, indexing, ZernikeNorm::normed, st::Geo, outer_extents...>;

/**
    @brief A non-owning view of a vector of 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <
    typename ElementType, Indexing indexing, ZernikeNorm zernike_norm,
    st::sh_convention Convention
>
using ZernikeVectorSpan = ZernikeTensorSpan<
    ElementType, indexing, zernike_norm, Convention, std::dynamic_extent>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanAcoustics = ZernikeVectorSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanNormalAcoustics = ZernikeVectorSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanQM = ZernikeVectorSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::QM>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanNormalQM = ZernikeVectorSpan<
    ElementType, indexing, ZernikeNorm::normed, st::QM>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanGeo = ZernikeVectorSpan<
    ElementType, indexing, ZernikeNorm::unnormed, st::Geo>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanNormalGeo = ZernikeVectorSpan<
    ElementType, indexing, ZernikeNorm::normed, st::Geo>;

/**
    @brief A container for mutlidimensional arrays of Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, Indexing indexing, ZernikeNorm zernike_norm,
    st::sh_convention Convention, std::size_t... outer_extents
>
using ZernikeExpansionTensor = ShapedArray<
    ElementType,
    ZernikeTensorShape<indexing, zernike_norm, Convention, outer_extents...>>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorAcoustics = ZernikeExpansionTensor<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorNormalAcoustics = ZernikeExpansionTensor<
    ElementType, indexing, ZernikeNorm::normed, st::Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorQM = ZernikeExpansionTensor<
    ElementType, indexing, ZernikeNorm::unnormed, st::QM, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorNormalQM = ZernikeExpansionTensor<
    ElementType, indexing, ZernikeNorm::normed, st::QM, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorGeo = ZernikeExpansionTensor<
    ElementType, indexing, ZernikeNorm::unnormed, st::Geo, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorNormalGeo = ZernikeExpansionTensor<
    ElementType, indexing, ZernikeNorm::normed, st::Geo, outer_extents...>;

/**
    @brief A container for a vector of Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <
    typename ElementType, Indexing indexing, ZernikeNorm zernike_norm,
    st::sh_convention Convention
>
using ZernikeExpansionVector = ZernikeExpansionTensor<
    ElementType, indexing, zernike_norm, Convention, std::dynamic_extent>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorAcoustics = ZernikeExpansionVector<
    ElementType, indexing, ZernikeNorm::unnormed, st::Acoustics>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorNormalAcoustics = ZernikeExpansionVector<
    ElementType, indexing, ZernikeNorm::normed, st::Acoustics>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorQM = ZernikeExpansionVector<
    ElementType, indexing, ZernikeNorm::unnormed, st::QM>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorNormalQM = ZernikeExpansionVector<
    ElementType, indexing, ZernikeNorm::normed, st::QM>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorGeo = ZernikeExpansionVector<
    ElementType, indexing, ZernikeNorm::unnormed, st::Geo>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorNormalGeo = ZernikeExpansionVector<
    ElementType, indexing, ZernikeNorm::normed, st::Geo>;

/**
    @brief A complex encoded view of real 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, ZernikeNorm zernike_norm, st::sh_convention Convention,
    std::size_t... inner_extents
>
using ComplexEncodedRealZernikeSpan = ShapedSpan<
    ElementType, ZernikeNonnegativeShape<zernike_norm, Convention, inner_extents...>>;

/**
    @brief A non-owning view of isotropic 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    std::floating_point ElementType,
    zt::ZernikeNorm zernike_norm, st::sh_convention Convention,
    std::size_t... inner_extents>
using IsotropicZernikeSpan = zest::ShapedSpan<
    ElementType, IsotropicZernikeShape<zernike_norm, Convention, inner_extents...>>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanAcoustics = IsotropicZernikeSpan<
    ElementType, ZernikeNorm::unnormed, st::Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanNormalAcoustics = IsotropicZernikeSpan<
    ElementType, ZernikeNorm::unnormed, st::Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanQM = IsotropicZernikeSpan<
    ElementType, ZernikeNorm::unnormed, st::QM, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanNormalQM = IsotropicZernikeSpan<
    ElementType, ZernikeNorm::normed, st::QM, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanGeo = IsotropicZernikeSpan<
    ElementType, ZernikeNorm::unnormed, st::Geo, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanNormalGeo = IsotropicZernikeSpan<
    ElementType, ZernikeNorm::normed, st::Geo, inner_extents...>;

/**
    @brief A container for isotropic 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    std::floating_point ElementType,
    zest::zt::ZernikeNorm zernike_norm, st::sh_convention Convention,
    std::size_t... inner_extents>
using IsotropicZernikeExpansion = zest::ShapedArray<
    ElementType, IsotropicZernikeShape<zernike_norm, Convention, inner_extents...>>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionAcoustics = IsotropicZernikeExpansion<
    ElementType, ZernikeNorm::unnormed, st::Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionNormalAcoustics = IsotropicZernikeExpansion<
    ElementType, ZernikeNorm::unnormed, st::Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionQM = IsotropicZernikeExpansion<
    ElementType, ZernikeNorm::unnormed, st::QM, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionNormalQM = IsotropicZernikeExpansion<
    ElementType, ZernikeNorm::normed, st::QM, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with unnormalized
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionGeo = IsotropicZernikeExpansion<
    ElementType, ZernikeNorm::unnormed, st::Geo, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with orthonormal
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionNormalGeo = IsotropicZernikeExpansion<
    ElementType, ZernikeNorm::normed, st::Geo, inner_extents...>;

/**
    @brief A non-owning view of a multidimensional array of isotropic 3D Zernike
    function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, ZernikeNorm zernike_norm,
    st::sh_convention Convention, std::size_t... outer_extents
>
using IsotropicZernikeTensorSpan = ShapedSpan<
    ElementType,
    IsotropicZernikeTensorShape<zernike_norm, Convention, outer_extents...>>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanAcoustics = IsotropicZernikeTensorSpan<
    ElementType, ZernikeNorm::unnormed, st::Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanNormalAcoustics = IsotropicZernikeTensorSpan<
    ElementType, ZernikeNorm::unnormed, st::Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanQM = IsotropicZernikeTensorSpan<
    ElementType, ZernikeNorm::unnormed, st::QM, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanNormalQM = IsotropicZernikeTensorSpan<
    ElementType, ZernikeNorm::normed, st::QM, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with unnormalized
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanGeo = IsotropicZernikeTensorSpan<
    ElementType, ZernikeNorm::unnormed, st::Geo, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with orthonormal
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanNormalGeo = IsotropicZernikeTensorSpan<
    ElementType, ZernikeNorm::normed, st::Geo, outer_extents...>;

/**
    @brief A non-owning view of a vector of isotropic 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <typename ElementType, ZernikeNorm zernike_norm, st::sh_convention Convention>
using IsotropicZernikeVectorSpan = IsotropicZernikeTensorSpan<
    ElementType, zernike_norm, Convention, std::dynamic_extent>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanAcoustics = IsotropicZernikeVectorSpan<
    ElementType, ZernikeNorm::unnormed, st::Acoustics>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanNormalAcoustics = IsotropicZernikeVectorSpan<
    ElementType, ZernikeNorm::unnormed, st::Acoustics>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanQM = IsotropicZernikeVectorSpan<
    ElementType, ZernikeNorm::unnormed, st::QM>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanNormalQM = IsotropicZernikeVectorSpan<
    ElementType, ZernikeNorm::normed, st::QM>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with unnormalized
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanGeo = IsotropicZernikeVectorSpan<
    ElementType, ZernikeNorm::unnormed, st::Geo>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with orthonormal
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanNormalGeo = IsotropicZernikeVectorSpan<
    ElementType, ZernikeNorm::normed, st::Geo>;

/**
    @brief A container for mutlidimensional arrays of isotropic Zernike function
    data.

    @tparam ElementType Type of elements
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, ZernikeNorm zernike_norm,
    st::sh_convention Convention, std::size_t... outer_extents
>
using IsotropicZernikeExpansionTensor = ShapedArray<
    ElementType,
    IsotropicZernikeTensorShape<zernike_norm, Convention, outer_extents...>>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    unnormalized Zernike functions, orthonormal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorAcoustics = IsotropicZernikeExpansionTensor<
    ElementType, ZernikeNorm::unnormed, st::Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    orthnormal Zernike functions, orthonormal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorNormalAcoustics = IsotropicZernikeExpansionTensor<
    ElementType, ZernikeNorm::normed, st::Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    unnormalized Zernike functions, orthonormal spherical harmonics, and
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorQM = IsotropicZernikeExpansionTensor<
    ElementType, ZernikeNorm::unnormed, st::QM, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    orthonormal Zernike functions, orthonormal spherical harmonics, and
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorNormalQM = IsotropicZernikeExpansionTensor<
    ElementType, ZernikeNorm::normed, st::QM, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    unnormalized Zernike functions, 4-pi normal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorGeo = IsotropicZernikeExpansionTensor<
    ElementType, ZernikeNorm::unnormed, st::Geo, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    orthonormal Zernike functions, 4-pi normal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorNormalGeo = IsotropicZernikeExpansionTensor<
    ElementType, ZernikeNorm::normed, st::Geo, outer_extents...>;

/**
    @brief A container for a vector of isotropic Zernike function data.

    @tparam ElementType Type of elements
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <typename ElementType, ZernikeNorm zernike_norm, st::sh_convention Convention>
using IsotropicZernikeExpansionVector = IsotropicZernikeExpansionTensor<
    ElementType, zernike_norm, Convention, std::dynamic_extent>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    unnormalized Zernike functions, orthonormal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorAcoustics = IsotropicZernikeExpansionVector<
    ElementType, ZernikeNorm::unnormed, st::Acoustics>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    orthnormal Zernike functions, orthonormal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorNormalAcoustics = IsotropicZernikeExpansionVector<
    ElementType, ZernikeNorm::normed, st::Acoustics>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    unnormalized Zernike functions, orthonormal spherical harmonics, and
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorQM = IsotropicZernikeExpansionVector<
    ElementType, ZernikeNorm::unnormed, st::QM>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    orthonormal Zernike functions, orthonormal spherical harmonics, and
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorNormalQM = IsotropicZernikeExpansionVector<
    ElementType, ZernikeNorm::normed, st::QM>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    unnormalized Zernike functions, 4-pi normal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorGeo = IsotropicZernikeExpansionVector<
    ElementType, ZernikeNorm::unnormed, st::Geo>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    orthonormal Zernike functions, 4-pi normal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorNormalGeo = IsotropicZernikeExpansionVector<
    ElementType, ZernikeNorm::normed, st::Geo>;

} // namespace zest::zt
