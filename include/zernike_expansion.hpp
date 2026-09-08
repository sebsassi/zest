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
    @brief A non-owning view of a vector of isotropic 3D Zernike function data.

    @tparam ElementType Type of elements.
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <typename ElementType, zernike_convention Convention>
using IsotropicZernikeVectorSpan = IsotropicZernikeTensorSpan<
    ElementType, Convention, std::dynamic_extent>;

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
    @brief A container for a vector of isotropic Zernike function data.

    @tparam ElementType Type of elements
    @tparam zernike_norm Zernike function normalization convention.
    @tparam Convention Spherical harmonic convention.
*/
template <typename ElementType, zernike_convention Convention>
using IsotropicZernikeExpansionVector = IsotropicZernikeExpansionTensor<
    ElementType, Convention, std::dynamic_extent>;

} // namespace zest::zt
