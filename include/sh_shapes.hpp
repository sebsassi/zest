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

#include "sequence.hpp"
#include "sh_conventions.hpp"
#include "triangle_shapes.hpp"

namespace zest::st
{

/**
    @brief Shape representing an efficient layout of associated Legendre
    functions, tagged with normalization and phase conventions.

    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam InnerExtents Extents on an inner multidimensional array structure.
*/
template <SHNorm sh_norm, SHPhase sh_phase, std::size_t... InnerExtents>
using AssociatedLegendreShape = TaggedShape<
    TriangleShape<IndexingMode::zero_based, InnerExtents...>, SHTag<sh_norm, sh_phase>>;

/**
    @brief Shape representing an efficient layout of a multidimensional array
    of associated Legendre functions, tagged with normalization and phase
    conventions.

    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam OuterExtents Extents of an outer multidimensional array structure.
*/
template <SHNorm sh_norm, SHPhase sh_phase, std::size_t... OuterExtents>
using AssociatedLegendreTensorShape = TaggedShape<
    TriangleTensorShape<IndexingMode::zero_based, OuterExtents...>, SHTag<sh_norm, sh_phase>>;

/**
    @brief Tagged shape representing layout and conventions of spherical
    harmonic data.

    @tparam IndexingMode Determines azimuthal index order.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam InnerExtents extents of an inner multidimensional array structure.
*/
template <
    IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase,
    std::size_t... InnerExtents
>
using SHShape = TaggedShape<
    std::conditional_t<(indexing_mode == IndexingMode::symmetric),
        TriangleShape<indexing_mode, InnerExtents...>,
        TriangleShape<indexing_mode, 2, InnerExtents...>>,
    SHTag<sh_norm, sh_phase>>;

/**
    @brief Shape representing an efficient layout of a multidimensional array of
    spherical harmonics, tagged with normalization and phase conventions.

    @tparam IndexingMode Determines azimuthal index order.
    @tparam sh_norm Normalization convention of the spherical harmonics.
    @tparam sh_phase Phase convention of the spherical harmonics.
    @tparam OuterExtents Extents of an outer multidimensional array structure.
*/
template <
    IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase,
    std::size_t... OuterExtents
>
using SHTensorShape = TaggedShape<
    std::conditional_t<(indexing_mode == IndexingMode::symmetric),
        TriangleTensorShape<indexing_mode, OuterExtents...>,
        CompositeShape<
            TensorShape<OuterExtents...>,
            TriangleShape<indexing_mode, 2>>>,
    SHTag<sh_norm, sh_phase>>;


} // namespace zest::st
