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
#include <vector>

#include "utility.hpp"
#include "shaped_span.hpp"

namespace zest
{

template <typename ElementType, typename ShapeType, typename Allocator = std::allocator<ElementType>>
class ShapedArray
{
public:
    using value_type = ElementType;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = std::allocator_traits<allocator_type>::pointer;
    using const_pointer = std::allocator_traits<allocator_type>::const_pointer;
    using shape_type = ShapeType;
    using index_type = shape_type::index_type;
    using index_range = ShapeType::index_range;
    using view = ShapedSpan<value_type, shape_type>;
    using const_view = ShapedSpan<const value_type, shape_type>;

    template <std::size_t N>
    using subspan_type = ShapedSpan<
        value_type, typename shape_type::template subshape_type<N>>;

    template <std::size_t N>
    using const_subspan_type = ShapedSpan<
        const value_type, typename shape_type::template subshape_type<N>>;

    ShapedArray() = default;

    template <typename... ExtentTypes>
        requires std::constructible_from<shape_type, ExtentTypes...>
    explicit ShapedArray(const ExtentTypes&... extents):
        m_shape(extents...) { m_data.resize(m_shape.size()); }

    explicit ShapedArray(const shape_type& shape):
        m_data(shape.size()), m_shape(shape) {}

    template <typename... ExtentTypes>
        requires std::constructible_from<shape_type, ExtentTypes...>
    [[nodiscard]] static constexpr size_type
    size(const ExtentTypes&... extents) noexcept { return shape_type::size(extents...); }

    [[nodiscard]] explicit operator
    view() noexcept { return view(m_data.data(), m_shape); }

    [[nodiscard]] explicit operator
    const_view() const noexcept { return const_view(m_data.data(), m_shape); }

    [[nodiscard]] auto
    tagless() noexcept requires tagged<shape_type>
    {
        return ShapedSpan<value_type, typename shape_type::untag>(m_data, m_shape);
    }

    [[nodiscard]] auto
    tagless() const noexcept requires tagged<shape_type>
    {
        return ShapedSpan<value_type, typename shape_type::untag>(m_data, m_shape);
    }

    [[nodiscard]] explicit(shape_type::rank != 1) operator
    std::span<value_type>() noexcept { return flatten(); }

    [[nodiscard]] explicit(shape_type::rank != 1) operator
    std::span<const value_type>() const noexcept { return flatten(); }

    void reshape(const shape_type& shape)
    {
        m_shape = shape;
        m_data.resize(m_shape.size());
    }

    template <typename... ExtentTypes>
        requires std::constructible_from<shape_type, ExtentTypes...>
    void reshape(const ExtentTypes&... extents)
    {
        reshape(shape_type(extents...));
    }

    template <typename NewShapeType>
    [[nodiscard]] auto
    view_as(const NewShapeType& shape) noexcept
    {
        assert(m_shape.size() == shape.size());
        return ShapedSpan<value_type, NewShapeType>(m_data.data(), shape);
    }

    template <typename NewShapeType>
    [[nodiscard]] auto
    reshape(const NewShapeType& shape) const noexcept
    {
        assert(m_shape.size() == shape.size());
        return ShapedSpan<const value_type, NewShapeType>(m_data.data(), shape);
    }

    [[nodiscard]] const shape_type&
    shape() const noexcept { return m_shape; }

    [[nodiscard]] size_type
    order() const noexcept requires sequence_shaped<shape_type> { return m_shape.order(); }

    [[nodiscard]] const ShapeType::extent_type&
    extents() const noexcept { return m_shape.extents(); }

    [[nodiscard]] constexpr size_type
    extent(size_type i) const noexcept requires tensor_shaped<shape_type> { return m_shape.extent(i); }

    [[nodiscard]] size_type
    size() const noexcept { return m_data.size(); }

    [[nodiscard]] const_pointer
    data() const noexcept { return m_data.data(); }

    [[nodiscard]] pointer
    data() noexcept { return m_data.data(); }

    [[nodiscard]] std::span<const value_type, shape_type::linear_extent>
    flatten() const noexcept
    {
        return std::span<const value_type, shape_type::linear_extent>(m_data);
    }

    [[nodiscard]] std::span<value_type, shape_type::linear_extent>
    flatten() noexcept
    {
        return std::span<value_type, shape_type::linear_extent>(m_data);
    }

    [[nodiscard]] index_range
    indices() const noexcept { return m_shape.indices(); }

    [[nodiscard]] index_range
    indices(index_type index) const noexcept { return m_shape.indices(index); }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] const_reference
    operator()(Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] reference
    operator()(Inds... indices) noexcept { return m_data[m_shape(indices...)]; }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] const_reference
    operator[](Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] reference
    operator[](Inds... indices) noexcept { return m_data[m_shape(indices...)]; }

    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator()(Inds... indices) const noexcept
    {
        return const_subspan_type<sizeof...(Inds)>(
            m_data.data() + m_shape(indices...), m_shape.subshape(indices...));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator()(Inds... indices) noexcept
    {
        return subspan_type<sizeof...(Inds)>(
            m_data.data() + m_shape(indices...), m_shape.subshape(indices...));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator[](Inds... indices) const noexcept
    {
        return const_subspan_type<sizeof...(Inds)>(
            m_data.data() + m_shape(indices...), m_shape.subshape(indices...));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator[](Inds... indices) noexcept
    {
        return subspan_type<sizeof...(Inds)>(
            m_data.data() + m_shape(indices...), m_shape.subshape(indices...));
    }

private:
    std::vector<ElementType, Allocator> m_data{};
    shape_type m_shape{};
};

} // namespace zest
