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
