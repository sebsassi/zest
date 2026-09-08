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
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, sh_convention Convention,
    std::size_t... inner_extents
>
using AssociatedLegendreSpan = ShapedSpan<
    ElementType, AssociatedLegendreShape<Convention, inner_extents...>>;

/**
    @brief A container for associated Legendre function data.

    @tparam ElementType Type of elements.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, sh_convention Convention,
    std::size_t... inner_extents
>
using AssociatedLegendreExpansion = ShapedArray<
    ElementType, AssociatedLegendreShape<Convention, inner_extents...>>;

/**
    @brief A non-owning view of a multidimensional array of associated Legendre
    function data.

    @tparam ElementType Type of elements.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, sh_convention Convention,
    std::size_t... outer_extents
>
using AssociatedLegendreTensorSpan = ShapedSpan<
    ElementType, AssociatedLegendreTensorShape<Convention, outer_extents...>>;

/**
    @brief A non-owning view of a vector of associated Legendre
    function data.

    @tparam ElementType Type of elements.
    @tparam Convention Spherical harmonic convention.
*/
template <complex_or_real_float ElementType, sh_convention Convention>
using AssociatedLegendreVectorSpan = AssociatedLegendreTensorSpan<
    ElementType, Convention, std::dynamic_extent>;

/**
    @brief A container for a multidimensional array of associated Legendre
    function data.

    @tparam ElementType Type of elements.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, sh_convention Convention,
    std::size_t... outer_extents
>
using AssociatedLegendreExpansionTensor = ShapedArray<
    ElementType, AssociatedLegendreTensorShape<Convention, outer_extents...>>;

/**
    @brief A container for a vector of associated Legendre function data.

    @tparam ElementType Type of elements.
    @tparam Convention Spherical harmonic convention.
*/
template <complex_or_real_float ElementType, sh_convention Convention>
using AssociatedLegendreExpansionVector = AssociatedLegendreExpansionTensor<
    ElementType, Convention, std::dynamic_extent>;

/**
    @brief A non-owning view of spherical harmonic data.

    @tparam ElementType Type of elements
    @tparam indexing Determines azimuthal index order.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, Indexing indexing,
    sh_convention Convention, std::size_t... inner_extents
>
using SHSpan = ShapedSpan<
    ElementType, SHShape<indexing, Convention, inner_extents...>>;

/**
    @brief A container for spherical harmonic data.

    @tparam ElementType Type of elements.
    @tparam indexing Determines azimuthal index order.
    @tparam Convention Spherical harmonic convention.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, Indexing indexing,
    sh_convention Convention, std::size_t... inner_extents
>
using SHExpansion = ShapedArray<
    ElementType, SHShape<indexing, Convention, inner_extents...>>;

/**
    @brief A non-owning view of a multidimensional array of spherical harmonic
    data.

    @tparam ElementType Type of elements.
    @tparam indexing Determines azimuthal index order.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, Indexing indexing,
    sh_convention Convention, std::size_t... outer_extents
>
using SHTensorSpan = ShapedSpan<
    ElementType, SHTensorShape<indexing, Convention, outer_extents...>>;

/**
    @brief A non-owning view of a vector of spherical harmonic data.

    @tparam ElementType Type of elements.
    @tparam indexing Determines azimuthal index order.
    @tparam Convention Spherical harmonic convention.
*/
template <
    complex_or_real_float ElementType, Indexing indexing,
    sh_convention Convention
>
using SHVectorSpan = SHTensorSpan<
    ElementType, indexing, Convention, std::dynamic_extent>;

/**
    @brief A container for a multidimensional array of spherical harmonic data.

    @tparam ElementType Type of elements.
    @tparam indexing Determines azimuthal index order.
    @tparam Convention Spherical harmonic convention.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    complex_or_real_float ElementType, Indexing indexing,
    sh_convention Convention, std::size_t... outer_extents
>
using SHExpansionTensor = ShapedArray<
    ElementType, SHTensorShape<indexing, Convention, outer_extents...>>;

/**
    @brief A container for a vector of spherical harmonic data.

    @tparam ElementType Type of elements.
    @tparam indexing Determines azimuthal index order.
    @tparam Convention Spherical harmonic convention.
*/
template <
    complex_or_real_float ElementType, Indexing indexing,
    sh_convention Convention
>
using SHExpansionVector = SHExpansionTensor<
    ElementType, indexing, Convention, std::dynamic_extent>;

/**
    @brief A non-owning view for storing real spherical harmonic data encoded
    as complex numbers.

    @tparam ElementType Type of elements.
    @tparam Convention Spherical harmonic convention.
*/
template <complex_float ElementType, sh_convention Convention>
using ComplexEncodedRealSHSpan = ShapedSpan<
    ElementType, AssociatedLegendreShape<Convention>>;

} // namespace zest::st
