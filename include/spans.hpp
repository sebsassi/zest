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
#include "shaped_span.hpp"
#include "sequence.hpp"
#include "shape.hpp"
#include "zernike_conventions.hpp"

namespace zest
{

template <IndexingMode indexing_mode_param, std::size_t... Ns>
using TriangleShape = TensorSequenceShape<TriangleSequence<indexing_mode_param>, Ns...>;

template <std::size_t... Ns>
using AssociatedLegendreShape = TriangleShape<IndexingMode::nonnegative, Ns...>;

template <IndexingMode indexing_mode_param, std::size_t... Ns>
using SHShape = std::conditional_t<(indexing_mode_param == IndexingMode::negative),
    TriangleShape<indexing_mode_param, Ns...>,
    TriangleShape<indexing_mode_param, 2, Ns...>>;

template <std::size_t... Ns>
using RadialZernikeShape = TensorSequenceShape<EvenTriangleSequence, Ns...>;

template <IndexingMode indexing_mode_param, std::size_t... Ns>
using ZernikeShape = std::conditional_t<(indexing_mode_param == IndexingMode::negative), 
    TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode_param>, Ns...>,
    TensorSequenceShape<ZernikeTetrahedralSequence<indexing_mode_param>, 2, Ns...>>;

template <typename ElementType, IndexingMode indexing_mode_param, std::size_t... Ns>
using TriangleSpan = ShapedSpan<ElementType, TriangleShape<indexing_mode_param, Ns...>>;

namespace st
{

// TODO: Preserve tags
template <
    typename ElementType, IndexingMode indexing_mode_param, SHNorm sh_norm_param,
    SHPhase sh_phase_param, std::size_t... Ns>
class SHLMSpan: public ShapedSpan<ElementType, SHShape<indexing_mode_param, Ns...>>
{
private:
    using Base = ShapedSpan<ElementType, SHShape<indexing_mode_param, Ns...>>;

public:
    using Base::ShapedSpan;
    using Base::data;
    using Base::shape;

    using ConstView = SHLMSpan<const ElementType, indexing_mode_param, sh_norm_param, sh_phase_param, Ns...>;

    static constexpr SHNorm norm = sh_norm_param;
    static constexpr SHPhase phase = sh_phase_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), shape());
    }
};

// TODO: Preserve tags
template <typename ElementType, SHNorm sh_norm_param, SHPhase sh_phase_param, std::size_t... Ns>
class AssociatedLegendreSpan: public TriangleSpan<ElementType, IndexingMode::nonnegative, Ns...>
{
private:
    using Base = TriangleSpan<ElementType, IndexingMode::nonnegative, Ns...>;

public:
    using Base::ShapedSpan;
    using Base::data;
    using Base::shape;

    using ConstView = AssociatedLegendreSpan<const ElementType, sh_norm_param, sh_phase_param, Ns...>;

    static constexpr SHNorm norm = sh_norm_param;
    static constexpr SHPhase phase = sh_phase_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), shape());
    }
};

} // namespace st

namespace zt
{

// TODO: Preserve tags
template <typename ElementType, ZernikeNorm zernike_norm_param, std::size_t... Ns>
class RadialZernikeSpan : public ShapedSpan<ElementType, RadialZernikeShape<Ns...>>
{
private:
    using Base = ShapedSpan<ElementType, RadialZernikeShape<Ns...>>;

public:
    using Base::ShapedSpan;
    using Base::data;
    using Base::shape;

    using ConstView = RadialZernikeSpan<const ElementType, zernike_norm_param, Ns...>;

    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), shape());
    }
};

// TODO: Preserve tags
template <
    typename ElementType, typename LayoutType, IndexingMode indexing_mode_param,
    ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param,
    std::size_t... Ns>
class ZernikeNLMSpan : public ShapedSpan<ElementType, ZernikeShape<indexing_mode_param, Ns...>>
{
private:
    using Base = ShapedSpan<ElementType, ZernikeShape<indexing_mode_param, Ns...>>;

public:
    using Base::ShapedSpan;
    using Base::data;
    using Base::shape;

    using ConstView = ZernikeNLMSpan<
        const ElementType, LayoutType, indexing_mode_param, zernike_norm_param, sh_norm_param, sh_phase_param>;

    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm sh_norm = sh_norm_param;
    static constexpr st::SHPhase sh_phase = sh_phase_param;

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), shape());
    }
};

} // namespace zt

} // namespace zest
