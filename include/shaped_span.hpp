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

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>

#include "utility.hpp"
#include "utility_concepts.hpp"

namespace zest
{

/**
    @brief A non-owning view of contiguous multidimensional data.

    @tparam ElementType Type of elements of the view.
    @tparam ShapeType Type defining the multidimensional shape of the data.

    This class allows for definition of non-owning views of contiguous
    multidimensional data of arbitrary shape. It generalizes the concept of
    tensor and multidimensional array views to datasets whose indices don't
    necessarily form a rectangular grid.
*/
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
    using index_range = shape_type::index_range;
    using view = ShapedSpan<element_type, ShapeType>;
    using const_view = ShapedSpan<const element_type, ShapeType>;

    static constexpr size_type rank = shape_type::rank;

    /**
        @brief `N`th subspan view.

        @tparam N The depth of the view in dimensions.

        Given a `ShapedSpan` of dimension `M`, `subspan_type` corresponds to
        the `M - N` dimensional view of the last `M - N` dimensions.
    */
    template <std::size_t N>
    using subspan_type = ShapedSpan<
        element_type, typename shape_type::template subshape_type<N>>;

    /**
        @brief Constant `N`th subspan view.

        @tparam N The depth of the view in dimensions.

        this is a const variant of `subspan_view`.
    */
    template <std::size_t N>
    using const_subspan_type = ShapedSpan<
        const element_type, typename shape_type::template subshape_type<N>>;

    constexpr ShapedSpan() = default;

    /**
        @brief Construct a shaped span from a pointer to data and extents of
        the shape.

        @tparam ExtentTypes Types of the extents of the shape.

        @param data Pointer to the beginning of the data.
        @param extents Extents of the shape.
    */
    template <typename... ExtentTypes>
        requires std::constructible_from<shape_type, ExtentTypes...>
    constexpr ShapedSpan(pointer data, const ExtentTypes&... extents):
        m_data{data}, m_shape{extents...} {}

    /**
        @brief Construct a shaped span from a pointer to data and extents of
        the shape.

        @param data Pointer to the beginning of the data.
        @param extents Extents of the shape.
    */
    constexpr ShapedSpan(pointer data, const shape_type::extent_type& extents):
        m_data{data}, m_shape{extents} {}

    /**
        @brief Construct a shaped span from a pointer and a shape.

        @param data Pointer to the beginning of the data.
        @param shape Shape of the view.
    */
    constexpr ShapedSpan(pointer data, const shape_type& shape):
        m_data{data}, m_shape{shape} {}

    /**
        @brief Construct a shaped span from `std::span` and extents of the shape.

        @tparam ExtentTypes Types of the extents of the shape.

        @param data `std::span` of the data.
        @param extents Extents of the shape.
    */
    template <typename... ExtentTypes>
        requires std::constructible_from<shape_type, ExtentTypes...>
    constexpr ShapedSpan(std::span<element_type> data, const ExtentTypes&... extents):
        m_data{data.data()}, m_shape{extents...} { assert(data.size() >= m_shape.size()); }

    /**
        @brief Construct a shaped span from `std::span` and extents of the shape.

        @param data `std::span` of the data.
        @param extents Extents of the shape.
    */
    constexpr ShapedSpan(std::span<element_type> data, const shape_type::extent_type& extents):
        m_data{data.data()}, m_shape{extents} { assert(data.size() >= m_shape.size()); }

    /**
        @brief Construct a shaped span from `std::span` and a shape.

        @param data `std::span` of the data.
        @param shape Shape of the view.
    */
    constexpr ShapedSpan(std::span<element_type> data, const shape_type& shape):
        m_data{data.data()}, m_shape{shape} { assert(data.size() >= m_shape.size()); }

    /**
        @brief Construct a shaped span from another shaped buffer.

        @tparam T The type of the buffer.

        @param shaped_buffer The other buffer.
    */
    template <shaped_contiguous_buffer T>
        requires std::same_as<typename std::remove_cvref_t<T>::shape_type, shape_type>
            && std::same_as<typename std::remove_cvref_t<T>::value_type, value_type>
    constexpr ShapedSpan(T&& shaped_buffer):
        m_data{std::forward<T>(shaped_buffer).data()},
        m_shape{std::forward<T>(shaped_buffer).shape()} {}

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
        @brief Convert to a constant view.
    */
    [[nodiscard]] explicit constexpr operator
    const_view() const noexcept { return const_view(m_data, m_shape); }

    /**
        @brief Get a view of a shaped span with a tagged shape with the tags
        removed.
    */
    [[nodiscard]] constexpr auto
    tagless() const noexcept requires tagged<shape_type>
    {
        return ShapedSpan<element_type, typename shape_type::untag>(m_data, m_shape);
    }

    /**
        @brief Convert to a `std::span`.
    */
    [[nodiscard]] explicit(shape_type::rank != 1) constexpr operator
    std::span<value_type>() const noexcept { return flatten(); }

    /**
        @brief View the data with a new shape.

        @tparam NewShapeType Type of the new shape.

        @param shape New shape of the data.
    */
    template <typename NewShapeType>
    [[nodiscard]] constexpr auto
    view_as(const NewShapeType& shape) const noexcept
    {
        assert(shape.size() == m_shape.size());
        return ShapedSpan<element_type, NewShapeType>(m_data, shape);
    }

    template <typename T>
        requires (!std::is_const_v<element_type> || std::is_const_v<T>)
            && representable_as<value_type, std::remove_cv_t<T>>
    [[nodiscard]] constexpr auto
    represent_as() const noexcept
    {
        using VT = std::conditional<std::is_volatile_v<element_type>, std::add_volatile_t<T>, T>;
        using CVT = std::conditional<std::is_const_v<element_type>, std::add_const_t<VT>, VT>;
        return ShapedSpan<CVT, shape_type>(reinterpret_cast<CVT*>(m_data), shape);
    }

    /**
        @brief Shape of the view.
    */
    [[nodiscard]] constexpr const shape_type&
    shape() const noexcept { return m_shape; }

    /**
        @brief Order of the view if the shape is a sequenced shape.
    */
    [[nodiscard]] constexpr size_type
    order() const noexcept requires sequence_shaped<shape_type> { return m_shape.order(); }

    /**
        @brief Extents of the view.
    */
    [[nodiscard]] constexpr const ShapeType::extent_type&
    extents() const noexcept { return m_shape.extents(); }

    /**
        @brief Extent of a tensor-like view along a given dimension.
    */
    [[nodiscard]] constexpr size_type
    extent(size_type i) const noexcept requires tensor_shaped<shape_type> { return m_shape.extent(i); }

    /**
        @brief Size of the view.
    */
    [[nodiscard]] constexpr size_type
    size() const noexcept { return m_shape.size(); }

    /**
        @brief Check if the view is empty.
    */
    [[nodiscard]] constexpr bool
    is_empty() const noexcept { return size() == 0; }

    /**
        @brief Pointer to the underlying data.
    */
    [[nodiscard]] constexpr pointer
    data() const noexcept { return m_data; }

    /**
        @brief Flatten to a one-dimensional `std::span` view.
    */
    [[nodiscard]] constexpr auto
    flatten() const noexcept
    {
        return std::span<element_type, shape_type::linear_extent>(m_data, m_shape.size());
    }

    /**
        @brief Index range of the outermost dimension.
    */
    [[nodiscard]] constexpr index_range
    indices() const noexcept { return m_shape.indices(); }

    /**
        @brief Index range of the outermost dimension, starting at `index`.
    */
    [[nodiscard]] constexpr index_range
    indices(index_type index) const noexcept { return m_shape.indices(index); }

    /**
        @brief Access the elements at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] constexpr reference
    operator()(Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    /**
        @brief Access the elements at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] constexpr reference
    operator[](Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    /**
        @brief Get the subspan at the given indices.
    */
    template <std::integral... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] constexpr auto
    operator()(Inds... indices) const noexcept
    {
        return subspan_type<sizeof...(Inds)>(
            m_data + m_shape(indices...), m_shape.subshape(indices...));
    }

    /**
        @brief Get the subspan at the given indices.
    */
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
