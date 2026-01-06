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

#include <cstddef>

#include "sequence.hpp"
#include "sh_conventions.hpp"
#include "spans.hpp"

namespace zest::st
{

/**
    @brief Shape representing an efficient layout of associated Legendre
    functions, tagged with normalization and phase conventions.

    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns>
using AssociatedLegendreShape = TaggedShape<
    TriangleShape<IndexingMode::zero_based, Ns...>, SHTag<sh_norm_param, sh_phase_param>>;

/**
    @brief Shape representing an efficient layout of a multidimensional array
    of associated Legendre functions, tagged with normalization and phase
    conventions.

    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns>
using AssociatedLegendreTensorShape = TaggedShape<
    TriangleTensorShape<IndexingMode::zero_based, Ns...>, SHTag<sh_norm_param, sh_phase_param>>;

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
    @brief Shape representing an efficient layout of a multidimensional array of
    spherical harmonics, tagged with normalization and phase conventions.

    @tparam IndexingMode determines azimuthal index order
    @tparam sh_norm_param normalization convention of the spherical harmonics
    @tparam sh_phase_param phase convention of the spherical harmonics
    @tparam Ns extents representing inner multidimensional array structure
*/
template <
    IndexingMode indexing_mode_param, SHNorm sh_norm_param, SHPhase sh_phase_param,
    std::size_t... Ns
>
using SHTensorShape = TaggedShape<
    std::conditional_t<(indexing_mode_param == IndexingMode::symmetric),
        TriangleTensorShape<indexing_mode_param, Ns...>,
        CompositeShape<
            TensorShape<Ns...>,
            TriangleShape<indexing_mode_param, 2>>>,
    SHTag<sh_norm_param, sh_phase_param>>;


} // namespace zest::st
