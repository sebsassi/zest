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
template <typename ElementType, zernike_norm_convention Convention, std::size_t... inner_extents>
using RadialZernikeSpan = ShapedSpan<
    ElementType, RadialZernikeShape<Convention, inner_extents...>>;

/**
    @brief A container of 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, zernike_norm_convention Convention, std::size_t... inner_extents>
using RadialZernikeExpansion = ShapedArray<
    ElementType, RadialZernikeShape<Convention, inner_extents...>>;

/**
    @brief A non-owning view of multidimensional arrays of 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <typename ElementType, zernike_norm_convention Convention, std::size_t... outer_extents>
using RadialZernikeTensorSpan = ShapedSpan<
    ElementType, RadialZernikeTensorShape<Convention, outer_extents...>>;

/**
    @brief A non-owning view of a vector of 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
*/
template <typename ElementType, zernike_norm_convention Convention>
using RadialZernikeVectorSpan = RadialZernikeTensorSpan<
    ElementType, Convention, std::dynamic_extent>;

/**
    @brief A container for storing multidimensional arrays of 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, zernike_norm_convention Convention, std::size_t... outer_extents>
using RadialZernikeExpansionTensor = ShapedArray<
    ElementType, RadialZernikeTensorShape<Convention, outer_extents...>>;

/**
    @brief A container for storing a vector of 3D radial Zernike polynomial
    data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
*/
template <typename ElementType, zernike_norm_convention Convention>
using RadialZernikeExpansionVector = RadialZernikeExpansionTensor<
    ElementType, Convention, std::dynamic_extent>;

/**
    @brief A non-owning view of isotropic 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, zernike_norm_convention Convention, std::size_t... inner_extents>
using IsotropicRadialZernikeSpan = ShapedSpan<
    ElementType, IsotropicRadialZernikeShape<Convention, inner_extents...>>;

/**
    @brief A container of isotropic 3D radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, zernike_norm_convention Convention, std::size_t... inner_extents>
using IsotropicRadialZernikeExpansion = ShapedArray<
    ElementType, IsotropicRadialZernikeShape<Convention, inner_extents...>>;

/**
    @brief A non-owning view of multidimensional arrays of isotropic 3D radial
    Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <typename ElementType, zernike_norm_convention Convention, std::size_t... outer_extents>
using IsotropicRadialZernikeTensorSpan = ShapedSpan<
    ElementType, IsotropicRadialZernikeTensorShape<Convention, outer_extents...>>;

/**
    @brief A non-owning view of a vector of isotropic 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
*/
template <typename ElementType, zernike_norm_convention Convention>
using IsotropicRadialZernikeVectorSpan = IsotropicRadialZernikeTensorSpan<
    ElementType, Convention, std::dynamic_extent>;

/**
    @brief A container for storing multidimensional arrays of isotropic 3D
    radial Zernike polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, zernike_norm_convention Convention, std::size_t... outer_extents>
using IsotropicRadialZernikeExpansionTensor = ShapedArray<
    ElementType, IsotropicRadialZernikeTensorShape<Convention, outer_extents...>>;

/**
    @brief A container for storing a vector of isotropic 3D radial Zernike
    polynomial data.

    @tparam ElementType Type of elements in the view.
    @tparam zernike_norm Zernike function normalization convention.
*/
template <typename ElementType, zernike_norm_convention Convention>
using IsotropicRadialZernikeExpansionVector = IsotropicRadialZernikeExpansionTensor<
    ElementType, Convention, std::dynamic_extent>;

/**
    @brief A non-owning view of 3D Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, Indexing indexing, zernike_convention Convention,
    std::size_t... inner_extents
>
using ZernikeSpan = ShapedSpan<
    ElementType,
    ZernikeShape<indexing, Convention, inner_extents...>>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanAcoustics = ZernikeSpan<
    ElementType, indexing, Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanNormedAcoustics = ZernikeSpan<
    ElementType, indexing, NormedAcoustics, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanQM = ZernikeSpan<
    ElementType, indexing, QM, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanNormedQM = ZernikeSpan<
    ElementType, indexing, NormedQM, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanGeo = ZernikeSpan<
    ElementType, indexing, Geo, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeSpanNormedGeo = ZernikeSpan<
    ElementType, indexing, NormedGeo, inner_extents...>;

/**
    @brief A container for Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, Indexing indexing, zernike_convention Convention,
    std::size_t... inner_extents
>
using ZernikeExpansion = ShapedArray<
    ElementType, ZernikeShape<indexing, Convention, inner_extents...>>;

/**
    @brief Convenient alias for `ZernikeExpansion` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionAcoustics = ZernikeExpansion<
    ElementType, indexing, Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionNormedAcoustics = ZernikeExpansion<
    ElementType, indexing, NormedAcoustics, inner_extents...>;

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
    ElementType, indexing, QM, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionNormedQM = ZernikeExpansion<
    ElementType, indexing, NormedQM, inner_extents...>;

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
    ElementType, indexing, Geo, inner_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansion` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.
    @tparam inner_extents extents of an inner multidimensional array structure

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... inner_extents>
using ZernikeExpansionNormedGeo = ZernikeExpansion<
    ElementType, indexing, NormedGeo, inner_extents...>;

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
    typename ElementType, Indexing indexing, zernike_convention Convention,
    std::size_t... outer_extents
>
using ZernikeTensorSpan = ShapedSpan<
    ElementType,
    ZernikeTensorShape<indexing, Convention, outer_extents...>>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanAcoustics = ZernikeTensorSpan<
    ElementType, indexing, Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanNormedAcoustics = ZernikeTensorSpan<
    ElementType, indexing, NormedAcoustics, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanQM = ZernikeTensorSpan<
    ElementType, indexing, QM, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanNormedQM = ZernikeTensorSpan<
    ElementType, indexing, NormedQM, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanGeo = ZernikeTensorSpan<
    ElementType, indexing, Geo, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeTensorSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeTensorSpanNormedGeo = ZernikeTensorSpan<
    ElementType, indexing, NormedGeo, outer_extents...>;

/**
    @brief A non-owning view of a vector of 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <
    typename ElementType, Indexing indexing, zernike_convention Convention
>
using ZernikeVectorSpan = ZernikeTensorSpan<
    ElementType, indexing, Convention, std::dynamic_extent>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanAcoustics = ZernikeVectorSpan<
    ElementType, indexing, Acoustics>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanNormedAcoustics = ZernikeVectorSpan<
    ElementType, indexing, NormedAcoustics>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanQM = ZernikeVectorSpan<
    ElementType, indexing, QM>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanNormedQM = ZernikeVectorSpan<
    ElementType, indexing, NormedQM>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanGeo = ZernikeVectorSpan<
    ElementType, indexing, Geo>;

/**
    @brief Convenient alias for `ZernikeVectorSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeVectorSpanNormedGeo = ZernikeVectorSpan<
    ElementType, indexing, NormedGeo>;

/**
    @brief A container for mutlidimensional arrays of Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, Indexing indexing, zernike_convention Convention,
    std::size_t... outer_extents
>
using ZernikeExpansionTensor = ShapedArray<
    ElementType,
    ZernikeTensorShape<indexing, Convention, outer_extents...>>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorAcoustics = ZernikeExpansionTensor<
    ElementType, indexing, Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorNormedAcoustics = ZernikeExpansionTensor<
    ElementType, indexing, NormedAcoustics, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorQM = ZernikeExpansionTensor<
    ElementType, indexing, QM, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorNormedQM = ZernikeExpansionTensor<
    ElementType, indexing, NormedQM, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorGeo = ZernikeExpansionTensor<
    ElementType, indexing, Geo, outer_extents...>;

/**
    @brief Convenient alias for `ZernikeExpansionTensor` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, Indexing indexing, std::size_t... outer_extents>
using ZernikeExpansionTensorNormedGeo = ZernikeExpansionTensor<
    ElementType, indexing, NormedGeo, outer_extents...>;

/**
    @brief A container for a vector of Zernike function data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <
    typename ElementType, Indexing indexing, zernike_convention Convention
>
using ZernikeExpansionVector = ZernikeExpansionTensor<
    ElementType, indexing, Convention, std::dynamic_extent>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorAcoustics = ZernikeExpansionVector<
    ElementType, indexing, Acoustics>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthnormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorNormedAcoustics = ZernikeExpansionVector<
    ElementType, indexing, NormedAcoustics>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorQM = ZernikeExpansionVector<
    ElementType, indexing, QM>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorNormedQM = ZernikeExpansionVector<
    ElementType, indexing, NormedQM>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorGeo = ZernikeExpansionVector<
    ElementType, indexing, Geo>;

/**
    @brief Convenient alias for `ZernikeExpansionVector` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam indexing Determines azimuthal index order.
*/
template <typename ElementType, Indexing indexing>
using ZernikeExpansionVectorNormedGeo = ZernikeExpansionVector<
    ElementType, indexing, NormedGeo>;

/**
    @brief A complex encoded view of real 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, zernike_convention Convention, std::size_t... inner_extents
>
using ComplexEncodedRealZernikeSpan = ShapedSpan<
    ElementType, ZernikeNonnegativeShape<Convention, inner_extents...>>;

/**
    @brief A non-owning view of isotropic 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, zernike_convention Convention, std::size_t... inner_extents>
using IsotropicZernikeSpan = zest::ShapedSpan<
    ElementType, IsotropicZernikeShape<Convention, inner_extents...>>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanAcoustics = IsotropicZernikeSpan<
    ElementType, Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanNormedAcoustics = IsotropicZernikeSpan<
    ElementType, NormedAcoustics, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanQM = IsotropicZernikeSpan<
    ElementType, QM, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanNormedQM = IsotropicZernikeSpan<
    ElementType, NormedQM, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanGeo = IsotropicZernikeSpan<
    ElementType, Geo, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeSpan` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeSpanNormedGeo = IsotropicZernikeSpan<
    ElementType, NormedGeo, inner_extents...>;

/**
    @brief A container for isotropic 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, zernike_convention Convention, std::size_t... inner_extents>
using IsotropicZernikeExpansion = zest::ShapedArray<
    ElementType, IsotropicZernikeShape<Convention, inner_extents...>>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionAcoustics = IsotropicZernikeExpansion<
    ElementType, Acoustics, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionNormedAcoustics = IsotropicZernikeExpansion<
    ElementType, NormedAcoustics, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionQM = IsotropicZernikeExpansion<
    ElementType, QM, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionNormedQM = IsotropicZernikeExpansion<
    ElementType, NormedQM, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with unnormalized
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionGeo = IsotropicZernikeExpansion<
    ElementType, Geo, inner_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansion` with orthonormal
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... inner_extents>
using IsotropicZernikeExpansionNormedGeo = IsotropicZernikeExpansion<
    ElementType, NormedGeo, inner_extents...>;

/**
    @brief A non-owning view of a multidimensional array of isotropic 3D Zernike
    function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, zernike_convention Convention, std::size_t... outer_extents
>
using IsotropicZernikeTensorSpan = ShapedSpan<
    ElementType,
    IsotropicZernikeTensorShape<Convention, outer_extents...>>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanAcoustics = IsotropicZernikeTensorSpan<
    ElementType, Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanNormedAcoustics = IsotropicZernikeTensorSpan<
    ElementType, NormedAcoustics, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanQM = IsotropicZernikeTensorSpan<
    ElementType, QM, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanNormedQM = IsotropicZernikeTensorSpan<
    ElementType, NormedQM, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with unnormalized
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanGeo = IsotropicZernikeTensorSpan<
    ElementType, Geo, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeTensorSpan` with orthonormal
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeTensorSpanNormedGeo = IsotropicZernikeTensorSpan<
    ElementType, NormedGeo, outer_extents...>;

/**
    @brief A non-owning view of a vector of isotropic 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <typename ElementType, zernike_convention Convention>
using IsotropicZernikeVectorSpan = IsotropicZernikeTensorSpan<
    ElementType, Convention, std::dynamic_extent>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanAcoustics = IsotropicZernikeVectorSpan<
    ElementType, Acoustics>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanNormedAcoustics = IsotropicZernikeVectorSpan<
    ElementType, NormedAcoustics>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with unnormalized
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanQM = IsotropicZernikeVectorSpan<
    ElementType, QM>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with orthonormal
    Zernike functions, orthonormal spherical harmonics, and Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanNormedQM = IsotropicZernikeVectorSpan<
    ElementType, NormedQM>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with unnormalized
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanGeo = IsotropicZernikeVectorSpan<
    ElementType, Geo>;

/**
    @brief Convenient alias for `IsotropicZernikeVectorSpan` with orthonormal
    Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley
    phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeVectorSpanNormedGeo = IsotropicZernikeVectorSpan<
    ElementType, NormedGeo>;

/**
    @brief A container for mutlidimensional arrays of isotropic Zernike function
    data.

    @tparam ElementType Type of elements
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, zernike_convention Convention, std::size_t... outer_extents
>
using IsotropicZernikeExpansionTensor = ShapedArray<
    ElementType,
    IsotropicZernikeTensorShape<Convention, outer_extents...>>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    unnormalized Zernike functions, orthonormal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorAcoustics = IsotropicZernikeExpansionTensor<
    ElementType, Acoustics, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    orthnormal Zernike functions, orthonormal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorNormedAcoustics = IsotropicZernikeExpansionTensor<
    ElementType, NormedAcoustics, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    unnormalized Zernike functions, orthonormal spherical harmonics, and
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorQM = IsotropicZernikeExpansionTensor<
    ElementType, QM, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    orthonormal Zernike functions, orthonormal spherical harmonics, and
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorNormedQM = IsotropicZernikeExpansionTensor<
    ElementType, NormedQM, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    unnormalized Zernike functions, 4-pi normal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorGeo = IsotropicZernikeExpansionTensor<
    ElementType, Geo, outer_extents...>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionTensor` with
    orthonormal Zernike functions, 4-pi normal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
    @tparam outer_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, std::size_t... outer_extents>
using IsotropicZernikeExpansionTensorNormedGeo = IsotropicZernikeExpansionTensor<
    ElementType, NormedGeo, outer_extents...>;

/**
    @brief A container for a vector of isotropic Zernike function data.

    @tparam ElementType Type of elements
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <typename ElementType, zernike_convention Convention>
using IsotropicZernikeExpansionVector = IsotropicZernikeExpansionTensor<
    ElementType, Convention, std::dynamic_extent>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    unnormalized Zernike functions, orthonormal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorAcoustics = IsotropicZernikeExpansionVector<
    ElementType, Acoustics>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    orthnormal Zernike functions, orthonormal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorNormedAcoustics = IsotropicZernikeExpansionVector<
    ElementType, NormedAcoustics>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    unnormalized Zernike functions, orthonormal spherical harmonics, and
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorQM = IsotropicZernikeExpansionVector<
    ElementType, QM>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    orthonormal Zernike functions, orthonormal spherical harmonics, and
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorNormedQM = IsotropicZernikeExpansionVector<
    ElementType, NormedQM>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    unnormalized Zernike functions, 4-pi normal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorGeo = IsotropicZernikeExpansionVector<
    ElementType, Geo>;

/**
    @brief Convenient alias for `IsotropicZernikeExpansionVector` with
    orthonormal Zernike functions, 4-pi normal spherical harmonics, and no
    Condon-Shortley phase.

    @tparam ElementType Type of elements in the view.
*/
template <typename ElementType>
using IsotropicZernikeExpansionVectorNormedGeo = IsotropicZernikeExpansionVector<
    ElementType, NormedGeo>;

} // namespace zest::zt
