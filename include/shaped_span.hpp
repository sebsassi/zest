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

#include <cassert>
#include <cstddef>
#include <span>

#include "utility.hpp"

namespace zest
{

template <typename T>
concept shaped_contiguous_buffer = requires (T x)
    {
        { x.data() } -> std::same_as<typename T::pointer>;
        { x.shape() } -> std::same_as<const typename T::shape_type>;
    };

template <typename ElementType, typename ShapeType>
class ShapedSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using reference = element_type&;
    using const_reference = const element_type&;
    using pointer = element_type*;
    using const_pointer = const element_type*;
    using shape_type = ShapeType;
    using index_type = shape_type::index_type;
    using index_range = ShapeType::index_range;
    using view = ShapedSpan<element_type, ShapeType>;
    using const_view = ShapedSpan<const element_type, ShapeType>;

    template <std::size_t N>
    using subspan_type = ShapedSpan<
        element_type, typename shape_type::template subshape_type<N>>;

    template <std::size_t N>
    using const_subspan_type = ShapedSpan<
        const element_type, typename shape_type::template subshape_type<N>>;

    constexpr ShapedSpan() = default;

    constexpr ShapedSpan(pointer data, const shape_type::extent_type& extents):
        m_data(data), m_shape(extents) {}
    constexpr ShapedSpan(pointer data, const shape_type& shape):
        m_data(data), m_shape(shape) {}

    template <typename... ExtentTypes>
    constexpr ShapedSpan(std::span<value_type> data, const ExtentTypes&... extents):
        m_data(data.data()), m_shape(extents...) { assert(data.size() == m_shape.size()); }

    constexpr ShapedSpan(std::span<value_type> data, const shape_type& shape):
        m_data(data), m_shape(shape) { assert(data.size() == m_shape.size()); }

    template <shaped_contiguous_buffer T>
        requires std::same_as<typename T::shape_type, shape_type>
    constexpr ShapedSpan(T& shaped_buffer):
        m_data(shaped_buffer.data()), m_shape(shaped_buffer.shape()) {}

    [[nodiscard]] constexpr operator
    const_view() const noexcept { return const_view(m_data, m_shape); }

    [[nodiscard]] constexpr auto
    tagless() const noexcept requires tagged<shape_type>
    {
        return ShapedSpan<element_type, typename shape_type::untag>(m_data, m_shape);
    }

    [[nodiscard]] explicit constexpr operator
    std::span<value_type>() const noexcept { return flatten(); }

    template <typename NewShapeType>
    [[nodiscard]] constexpr auto
    reshape(const NewShapeType& shape) const noexcept
    {
        assert(shape.size() == m_shape.size());
        return ShapedSpan<element_type, NewShapeType>(m_data, shape);
    }

    [[nodiscard]] constexpr const ShapeType&
    shape() const noexcept { return m_shape; }

    [[nodiscard]] constexpr size_type
    order() const noexcept requires sequence_shaped<shape_type> { return m_shape.order(); }

    [[nodiscard]] constexpr const ShapeType::extent_type&
    extents() const noexcept { return m_shape.extents(); }

    [[nodiscard]] constexpr size_type
    extent(size_type i) const noexcept requires tensor_shaped<shape_type> { return m_shape.extent(i); }

    [[nodiscard]] constexpr size_type
    size() const noexcept { return m_shape.size(); }

    [[nodiscard]] constexpr bool
    is_empty() const noexcept { return size() == 0; }

    [[nodiscard]] constexpr pointer
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr auto
    flatten() const noexcept
    {
        return std::span<element_type, shape_type::linear_extent>(m_data, m_shape.size());
    }

    [[nodiscard]] constexpr index_range
    indices() const noexcept { return m_shape.indices(); }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] constexpr reference
    operator()(Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] constexpr reference
    operator[](Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] constexpr auto
    operator()(Inds... indices) const noexcept
    {
        return subspan_type<sizeof...(Inds)>(
            m_data + m_shape(indices...), m_shape.subshape(indices...));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] constexpr auto
    operator[](Inds... indices) const noexcept
    {
        return subspan_type<sizeof...(Inds)>(
            m_data + m_shape(indices...), m_shape.subshape(indices...));
    }

private:
    pointer m_data{};
    [[no_unique_address]] shape_type m_shape{};
};

} //namespace zest
