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

#include "sh_conventions.hpp"
#include "sequence.hpp"
#include "shape.hpp"
#include "shaped_span.hpp"
#include "zernike_conventions.hpp"

namespace zest
{

template <IndexingMode indexing_mode_param, std::size_t... Ns>
using TriangleShape = TensorSequenceShape<TriangleSequence<indexing_mode_param>, Ns...>;

template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using TriangleSpan = ShapedSpan<ElementType, TriangleShape<indexing_mode_param, Ns...>>;

// namespace st
// {
//
// template <SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns>
// using AssociatedLegendreShape = TaggedShape<TriangleShape<IndexingMode::nonnegative, Ns...>, SHTag<sh_norm_param, sh_phase_param>>;
//
// template <IndexingMode indexing_mode_param, SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns>
// using SHShape = TaggedShape<
//     std::conditional_t<(indexing_mode_param == IndexingMode::negative),
//         TriangleShape<indexing_mode_param, Ns...>,
//         TriangleShape<indexing_mode_param, 2, Ns...>>,
//     SHTag<sh_norm_param, sh_phase_param>>;
//
// template <
//     typename ElementType, IndexingMode indexing_mode_param, SHNorm sh_norm_param,
//     SHPhase sh_phase_param, std::size_t... Ns>
// using SHLMSpan = ShapedSpan<ElementType, SHShape<indexing_mode_param, sh_norm_param, sh_phase_param, Ns...>>;
//
// template <typename ElementType, SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns>
// using AssociatedLegendreSpan = ShapedSpan<ElementType, AssociatedLegendreShape<sh_norm_param, sh_phase_param, Ns...>>;
//
// } // namespace st
//
// namespace zt
// {
//
// template <ZernikeNorm zernike_norm_param, std::size_t... Ns>
// using RadialZernikeShape = TaggedShape<TensorSequenceShape<EvenTriangleSequence, Ns...>, ZernikeTag<zernike_norm_param>>;
//
// template <
//     IndexingMode indexing_mode_param, ZernikeNorm zernike_norm_param,
//     st::SHNorm sh_norm_param, st::SHPhase sh_phase_param, std::size_t... Ns>
// using ZernikeShape = TaggedShape<
//     std::conditional_t<(indexing_mode_param == IndexingMode::negative), 
//         TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode_param>, Ns...>,
//         TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode_param>, 2, Ns...>>,
//     ZernikeTag<zernike_norm_param>, st::SHTag<sh_norm_param, sh_phase_param>>;
//
// template <typename ElementType, ZernikeNorm zernike_norm_param, std::size_t... Ns>
// using RadialZernikeSpan = ShapedSpan<ElementType, RadialZernikeShape<zernike_norm_param, Ns...>>;
//
// template <
//     typename ElementType, typename LayoutType, IndexingMode indexing_mode_param,
//     ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param,
//     std::size_t... Ns>
// using ZernikeNLMSpan = ShapedSpan<ElementType, ZernikeShape<indexing_mode_param, zernike_norm_param, sh_norm_param, sh_phase_param, Ns...>>;
//
// } // namespace zt

} // namespace zest
