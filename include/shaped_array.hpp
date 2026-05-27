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

/**
    @brief A container for contiguous multidimensional data.

    @tparam ElementType Type of elements of the view.
    @tparam ShapeType Type defining the multidimensional shape of the data.
    @tparam Allocator Allocator type for the internal buffer.

    This class allows for definition of containers of contiguous
    multidimensional data of arbitrary shape. It generalizes the concept of
    tensor and multidimensional arrays to datasets whose indices don't
    necessarily form a rectangular grid.
*/
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

    static constexpr size_type rank = shape_type::rank;

    /**
        @brief `N`th subspan view.

        @tparam N The depth of the view in dimensions.

        Given a `ShapedSpan` of dimension `M`, `subspan_type` corresponds to
        the `M - N` dimensional view of the last `M - N` dimensions.
    */
    template <std::size_t N>
    using subspan_type = ShapedSpan<
        value_type, typename shape_type::template subshape_type<N>>;

    /**
        @brief Constant `N`th subspan view.

        @tparam N The depth of the view in dimensions.

        this is a const variant of `subspan_view`.
    */
    template <std::size_t N>
    using const_subspan_type = ShapedSpan<
        const value_type, typename shape_type::template subshape_type<N>>;

    ShapedArray() = default;

    /**
        @brief Construct a shaped array from extents of the shape.

        @tparam ExtentTypes Types of the extents of the shape.

        @param extents Extents of the shape.
    */
    template <typename... ExtentTypes>
        requires std::constructible_from<shape_type, ExtentTypes...>
    explicit ShapedArray(const ExtentTypes&... extents):
        m_shape(extents...) { m_data.resize(m_shape.size()); }

    /**
        @brief Construct a shaped array from a shape.

        @param extents Extents of the shape.
    */
    explicit ShapedArray(const shape_type& shape):
        m_data(shape.size()), m_shape(shape) {}

    /**
        @brief Compute the size of a shaped span gives its extents.

        @tparam ExtentTypes Types of the extents of the shape.

        @param extents Extents of the shape.
    */
    template <typename... ExtentTypes>
        requires std::constructible_from<shape_type, ExtentTypes...>
    [[nodiscard]] static constexpr size_type
    size(const ExtentTypes&... extents) noexcept { return shape_type::size(extents...); }

    /**
        @brief Convert to a view.
    */
    [[nodiscard]] explicit operator
    view() noexcept { return view(m_data.data(), m_shape); }

    /**
        @brief Convert to a constant view.
    */
    [[nodiscard]] explicit operator
    const_view() const noexcept { return const_view(m_data.data(), m_shape); }

    /**
        @brief Get a view of a shaped array with a tagged shape with the tags
        removed.
    */
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

    /**
        @brief Convert to a `std::span`.
    */
    [[nodiscard]] explicit(shape_type::rank != 1) operator
    std::span<value_type>() noexcept { return flatten(); }

    [[nodiscard]] explicit(shape_type::rank != 1) operator
    std::span<const value_type>() const noexcept { return flatten(); }

    /**
        @brief Give a new shape to the data.
    */
    void reshape(const shape_type& shape)
    {
        m_shape = shape;
        m_data.resize(m_shape.size());
    }

    /**
        @brief Give a new shape to the data from extents.
    */
    template <typename... ExtentTypes>
        requires std::constructible_from<shape_type, ExtentTypes...>
    void reshape(const ExtentTypes&... extents)
    {
        reshape(shape_type(extents...));
    }

    /**
        @brief View the data with another shape.
    */
    template <typename NewShapeType>
    [[nodiscard]] auto
    view_as(const NewShapeType& shape) noexcept
    {
        assert(m_shape.size() == shape.size());
        return ShapedSpan<value_type, NewShapeType>(m_data.data(), shape);
    }

    /**
        @brief View the data with another shape.
    */
    template <typename NewShapeType>
    [[nodiscard]] auto
    reshape(const NewShapeType& shape) const noexcept
    {
        assert(m_shape.size() == shape.size());
        return ShapedSpan<const value_type, NewShapeType>(m_data.data(), shape);
    }

    /**
        @brief Shape of the view.
    */
    [[nodiscard]] const shape_type&
    shape() const noexcept { return m_shape; }

    /**
        @brief Order of the view if the shape is a sequenced shape.
    */
    [[nodiscard]] size_type
    order() const noexcept requires sequence_shaped<shape_type> { return m_shape.order(); }

    /**
        @brief Extents of the view.
    */
    [[nodiscard]] const ShapeType::extent_type&
    extents() const noexcept { return m_shape.extents(); }

    /**
        @brief Extent of a tensor-like view along a given dimension.
    */
    [[nodiscard]] constexpr size_type
    extent(size_type i) const noexcept requires tensor_shaped<shape_type> { return m_shape.extent(i); }

    /**
        @brief Size of the view.
    */
    [[nodiscard]] size_type
    size() const noexcept { return m_data.size(); }

    /**
        @brief Constant pointer to the underlying data.
    */
    [[nodiscard]] const_pointer
    data() const noexcept { return m_data.data(); }

    /**
        @brief Pointer to the underlying data.
    */
    [[nodiscard]] pointer
    data() noexcept { return m_data.data(); }

    /**
        @brief Flatten to a constant one-dimensional `std::span` view.
    */
    [[nodiscard]] std::span<const value_type, shape_type::linear_extent>
    flatten() const noexcept
    {
        return std::span<const value_type, shape_type::linear_extent>(m_data);
    }

    /**
        @brief Flatten to a one-dimensional `std::span` view.
    */
    [[nodiscard]] std::span<value_type, shape_type::linear_extent>
    flatten() noexcept
    {
        return std::span<value_type, shape_type::linear_extent>(m_data);
    }

    /**
        @brief Index range of the outermost dimension.
    */
    [[nodiscard]] index_range
    indices() const noexcept { return m_shape.indices(); }

    /**
        @brief Index range of the outermost dimension, starting at `index`.
    */
    [[nodiscard]] index_range
    indices(index_type index) const noexcept { return m_shape.indices(index); }

    /**
        @brief Access the elements at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] const_reference
    operator()(Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    /**
        @brief Access the elements at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] reference
    operator()(Inds... indices) noexcept { return m_data[m_shape(indices...)]; }

    /**
        @brief Access the elements at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] const_reference
    operator[](Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    /**
        @brief Access the elements at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] reference
    operator[](Inds... indices) noexcept { return m_data[m_shape(indices...)]; }

    /**
        @brief Get the subspan at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator()(Inds... indices) const noexcept
    {
        return const_subspan_type<sizeof...(Inds)>(
            m_data.data() + m_shape(indices...), m_shape.subshape(indices...));
    }

    /**
        @brief Get the subspan at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator()(Inds... indices) noexcept
    {
        return subspan_type<sizeof...(Inds)>(
            m_data.data() + m_shape(indices...), m_shape.subshape(indices...));
    }

    /**
        @brief Get the subspan at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator[](Inds... indices) const noexcept
    {
        return const_subspan_type<sizeof...(Inds)>(
            m_data.data() + m_shape(indices...), m_shape.subshape(indices...));
    }

    /**
        @brief Get the subspan at the given indices.
    */
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
