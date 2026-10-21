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

#include <type_traits>
#include <concepts>

namespace zest
{

template <typename T>
concept shape = requires (T t, typename T::extent_type e)
{
    typename std::remove_cvref_t<T>::size_type;
    typename std::remove_cvref_t<T>::index_type;
    typename std::remove_cvref_t<T>::index_range;
    typename std::remove_cvref_t<T>::extent_type;

    requires std::same_as<
        std::remove_const_t<decltype(std::remove_cvref_t<T>::rank)>,
        typename std::remove_cvref_t<T>::size_type>;
    requires std::same_as<
        std::remove_const_t<decltype(std::remove_cvref_t<T>::linear_extent)>,
        typename std::remove_cvref_t<T>::size_type>;

    { t.size() } -> std::same_as<typename std::remove_cvref_t<T>::size_type>;
    { t.indices() } -> std::same_as<typename std::remove_cvref_t<T>::index_range>;

    requires std::same_as<
        std::remove_cvref_t<decltype(t.extents())>,
        typename std::remove_cvref_t<T>::extent_type>;
};

template <typename T>
concept composite_shape = shape<T>
    && requires
{
    typename T::outer_shape_type;
    typename T::inner_shape_type;
};

template <typename T>
concept shaped_contiguous_buffer
    = requires
    {
        typename std::remove_cvref_t<T>::element_type;
        requires std::same_as<
            typename std::remove_cvref_t<T>::value_type,
            std::remove_cvref_t<typename std::remove_cvref_t<T>::element_type>>;
        requires std::unsigned_integral<typename std::remove_cvref_t<T>::size_type>;
        requires std::same_as<
            typename std::remove_cvref_t<T>::reference,
            typename std::remove_cvref_t<T>::element_type&>;
        requires std::same_as<
            typename std::remove_cvref_t<T>::const_reference,
            const typename std::remove_cvref_t<T>::element_type&>;
        requires std::same_as<
            typename std::remove_cvref_t<T>::pointer,
            typename std::remove_cvref_t<T>::element_type*>;
        requires std::same_as<
            typename std::remove_cvref_t<T>::const_pointer,
            const typename std::remove_cvref_t<T>::element_type*>;
        requires shape<typename std::remove_cvref_t<T>::shape_type>;
        requires std::integral<typename std::remove_cvref_t<T>::index_type>;
        typename std::remove_cvref_t<T>::index_range;
        typename std::remove_cvref_t<T>::view;
        typename std::remove_cvref_t<T>::const_view;
    }
    && requires (std::remove_cvref_t<T> x)
    {
        { x.data() } -> std::same_as<typename std::remove_reference_t<T>::pointer>;
        { x.shape() } -> std::same_as<const typename std::remove_cvref_t<T>::shape_type&>;
    };

template <typename T, typename Shape>
concept shaped_like
    = std::same_as<typename std::remove_cvref_t<T>::shape_type, Shape>;

template <typename T, typename Shape>
concept contiguous_buffer_shaped_like
    = shaped_contiguous_buffer<T> && shaped_like<T, Shape>;

/**
    @brief Check if a type is indexed by a sequence of index tuples.

    The defining property of a sequence shaped layout is that its extents,
    regardless of dimensionality, are determined by a single integer parameter,
    the order.
*/
template <typename T>
concept sequence_shaped = requires (T x)
    {
        { x.order() } -> std::same_as<typename T::size_type>;
    };

/**
    @brief Check if a type looks tensor shaped.

    A tensor shaped array has extents, which are independent of where we are in
    the array.
*/
template <typename T>
concept tensor_shaped
    =  requires (T x, typename std::remove_cvref_t<T>::size_type i)
    {
        { x.extent(i) } -> std::same_as<typename std::remove_cvref_t<T>::size_type>;
    };

/**
    @brief Check if a shape has inner tensor structure.
*/
template <typename T, std::size_t outer_rank, std::size_t inner_rank>
concept has_inner_tensor_structure
    = (std::remove_cvref_t<T>::rank == outer_rank && inner_rank == 0)
        || ((std::remove_cvref_t<T>::rank == outer_rank + inner_rank)
        && tensor_shaped<typename std::remove_cvref_t<T>::template subshape_type<outer_rank>>);

} // namespace zest
