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

#include <array>
#include <concepts>
#include <complex>

namespace zest
{

template <typename T>
concept arithmetic = std::integral<T> || std::floating_point<T>;

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

template <typename T, std::size_t outer_rank, std::size_t inner_rank>
concept has_inner_tensor_structure = (std::remove_cvref_t<T>::rank == outer_rank + inner_rank);

template <typename T>
concept shaped_contiguous_buffer = requires (T x)
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

    { x.data() } -> std::same_as<typename std::remove_cvref_t<T>::pointer>;
    { x.shape() } -> std::same_as<const typename std::remove_cvref_t<T>::shape_type&>;
};

template <typename T, typename Shape>
concept shaped_like
    = std::same_as<typename std::remove_cvref_t<T>::shape_type, Shape>;

template <typename T, typename Shape>
concept contiguous_buffer_shaped_like
    = shaped_contiguous_buffer<T> && shaped_like<T, Shape>;

template <typename T, typename Rep>
concept representable_as
    = std::same_as<T, Rep>
    || ((sizeof(T) == sizeof(Rep))
        && std::is_trivially_copyable_v<T> && std::is_trivially_copyable_v<Rep>
        && std::same_as<typename T::rep, Rep>);

template <typename Rep, typename T>
concept represents = representable_as<T, Rep>;

template <shaped_contiguous_buffer T>
using value_type_of = typename std::remove_cvref_t<T>::value_type;

template <shaped_contiguous_buffer T>
using shape_type_of = typename std::remove_cvref_t<T>::shape_type;

template <shaped_contiguous_buffer T, std::size_t depth>
    requires (depth <= T::rank)
using subshape_type_of = typename std::remove_cvref_t<T>::shape_type::template subshape_type<depth>;
/**
    @brief Check if a type contains tags.

    A tagged type is `T` identified by the presence of a member type
    `T::untag`, which is the type stripped of all its tags.
*/
template <typename T>
concept tagged = requires { typename T::untag; };

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

    A tensor shaped type has a set of extents which can be indexed by its dimensions.
*/
template <typename T>
concept tensor_shaped
    =  requires (T x, typename std::remove_cvref_t<T>::size_type i)
    {
        { x.extent(i) } -> std::same_as<typename std::remove_cvref_t<T>::size_type>;
    };

/**
    @brief Check if a type is a complex floating point number.
*/
template <typename T>
concept complex_float = std::same_as<
    std::remove_const_t<T>, std::complex<typename std::remove_const_t<T>::value_type>>;

/**
    @brief Check if a type is a complex or a real floating point number.
*/
template <typename T>
concept complex_or_real_float
    = std::floating_point<std::remove_const_t<T>> || complex_float<T>;

/**
    @brief Check if a type can be used as a tag.

    A type may be used as a tag if it:
        1. Contains no non-static member variables
        2. Is an aggreggate type as defined by the C++ standard

    In practice, a tag type only contains definitions for member types and
    static member variables.
*/
template <typename T>
concept is_tag = (std::is_empty_v<T> && std::is_aggregate_v<T>);

namespace detail
{

template <typename T>
struct remove_tags_helper { using type = T; };

template <typename T>
    requires requires { typename T::untag; }
struct remove_tags_helper<T> { using type = typename T::untag; };

} // namespace detail

template <typename T>
using remove_tags = detail::remove_tags_helper<T>::type;

/**
    @brief Function concept taking Cartesian coordinates as inputs.
*/
template <typename Func, typename T>
concept cartesian_function
    = std::invocable<Func, T> && std::constructible_from<std::array<double, 3>>;

/**
    @brief Function concept taking spherical angles as inputs.
*/
template <typename Func>
concept spherical_function = std::invocable<Func, double, double>;

/**
    @brief Function concept taking spherical coordinates as inputs.
*/
template <typename Func, typename RadiusType>
concept ball_function = std::invocable<Func, double, double, RadiusType>;

/**
    @brief Function concept taking radial coordinate as input.
*/
template <typename Func, typename RadiusType>
concept isotropic_function = std::invocable<Func, RadiusType>;

} // namespace zest
