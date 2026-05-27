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

#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace zest
{

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
concept tag_type = (std::is_empty_v<T> && std::is_aggregate_v<T>);

namespace detail
{

template <typename T, std::size_t N, std::size_t... I>
[[nodiscard]] constexpr std::array<T, sizeof...(I)>
take_last_impl(const std::array<T, N>& arr, std::index_sequence<I...> /*unused*/) noexcept
{
    return {std::get<N - sizeof...(I) + I>(arr)...};
}

template <typename T, std::size_t... I>
[[nodiscard]] constexpr auto
take_last_impl(const T& tuple, std::index_sequence<I...> /*unused*/) noexcept
{
    return std::make_tuple(std::get<std::tuple_size_v<T> - sizeof...(I) + I>(tuple)...);
}

template <typename T, std::size_t N, std::size_t... I>
[[nodiscard]] constexpr std::array<T, sizeof...(I)>
take_first_impl(const std::array<T, N>& arr, std::index_sequence<I...> /*unused*/) noexcept
{
    return {std::get<I>(arr)...};
}

template <typename T, std::size_t... I>
[[nodiscard]] constexpr auto
take_first_impl(const T& tuple, std::index_sequence<I...> /*unused*/) noexcept
{
    return std::make_tuple(std::get<I>(tuple)...);
}

template <typename T, std::convertible_to<T> S, std::size_t... I>
[[nodiscard]] constexpr std::array<T, sizeof...(I) + 1>
append_impl(
    const std::array<T, sizeof...(I)>& arr, const S& element,
    std::index_sequence<I...> /*unused*/) noexcept
{
    return {std::get<I>(arr)..., element};
}

template <typename T, typename S, std::size_t... I>
[[nodiscard]] constexpr auto
append_impl(
    const T& tuple, const S& element, std::index_sequence<I...> /*unused*/) noexcept
{
    return std::make_tuple(std::get<I>(tuple)..., element);
}

template <typename T, std::convertible_to<T> S, std::size_t... I>
[[nodiscard]] constexpr std::array<T, sizeof...(I) + 1>
prepend_impl(
    const S& element, const std::array<T, sizeof...(I)>& arr,
    std::index_sequence<I...> /*unused*/) noexcept
{
    return {element, std::get<I>(arr)...};
}

template <typename T, typename S, std::size_t... I>
[[nodiscard]] constexpr auto
prepend_impl(
    const S& element, const T& tuple, std::index_sequence<I...> /*unused*/) noexcept
{
    return std::make_tuple(element, std::get<I>(tuple)...);
}

template <typename T, std::size_t... I, std::size_t... J>
[[nodiscard]] constexpr std::array<T, sizeof...(I) + sizeof...(J)>
concatenate_impl(
    const std::array<T, sizeof...(I)>& arr_i, const std::array<T, sizeof...(J)>& arr_j,
    std::index_sequence<I...> /*unused*/, std::index_sequence<J...> /*unused*/) noexcept
{
    return {std::get<I>(arr_i)..., std::get<J>(arr_j)...};
}

template <typename T, typename S, std::size_t... I, std::size_t... J>
[[nodiscard]] constexpr auto
concatenate_impl(
    const T& tuple_t, const S& tuple_s,
    std::index_sequence<I...> /*unused*/, std::index_sequence<J...> /*unused*/) noexcept
{
    return std::make_tuple(std::get<I>(tuple_t)..., std::get<J>(tuple_s)...);
}

} // namespace detail

/**
    @brief Take last `N` elements of a tuple-like object.

    @tparam N Number of elements to take.
    @tparam T A tuple-like type.

    @param tuple_like A tuple-like object.

    @return A tuple-like object consisting of the last `N` elements of the
    input.
*/
template <std::size_t N, typename T>
    requires (N <= std::tuple_size_v<T>)
[[nodiscard]] constexpr auto
take_last(const T& tuple_like) noexcept
{
    return detail::take_last_impl(tuple_like, std::make_index_sequence<N>());
}

/**
    @brief Take first `N` elements of a tuple-like object.

    @tparam N Number of elements to take.
    @tparam T A tuple-like type.

    @param tuple_like A tuple-like object.

    @return A tuple-like object consisting of the first `N` elements of the
    input.
*/
template <std::size_t N, typename T>
    requires (N <= std::tuple_size_v<T>)
[[nodiscard]] constexpr auto
take_first(const T& tuple_like) noexcept
{
    return detail::take_first_impl(tuple_like, std::make_index_sequence<N>());
}

/**
    @brief Append an element to a tuple-like object

    @tparam T A tuple-like type.
    @tparam S Type of appended element.

    @param tuple_like Tuple-like object.
    @param element Object to append.

    @return Input with `element` appended.
*/
template <typename T, typename S>
[[nodiscard]] constexpr auto
append(const T& tuple_like, const S& element) noexcept
{
    return detail::append_impl(
        tuple_like, element, std::make_index_sequence<std::tuple_size_v<T>>{});
}

/**
    @brief Prepend an element to a tuple-like object

    @tparam T A tuple-like type.
    @tparam S Type of prepended element.

    @param element Object to prepend.
    @param tuple_like Tuple-like object.

    @return Input with `element` prepended.
*/
template <typename T, typename S>
[[nodiscard]] constexpr auto
prepend(const T& element, const S& tuple_like) noexcept
{
    return detail::prepend_impl(
        element, tuple_like, std::make_index_sequence<std::tuple_size_v<S>>{});
}

/**
    @brief Concatenate two tuple-like objects.

    @tparam T A tuple-like type.
    @tparam S A tuple-like type.

    @param tuple_like_t
    @param tuple_like_s

    @return Concatenation of the input objects.
*/
template <typename T, typename S>
[[nodiscard]] constexpr auto
concatenate(const T& tuple_like_t, const S& tuple_like_s) noexcept
{
    return detail::concatenate_impl(
        tuple_like_t, tuple_like_s,
        std::make_index_sequence<std::tuple_size_v<T>>{},
        std::make_index_sequence<std::tuple_size_v<S>>{});
}

/**
    @brief Compute the product of the elements of an array.

    @tparam T Value type of the array.
    @tparam N Size of the array.

    @param arr Input array.

    @return Product of the elements of the array.
*/
template <typename T, std::size_t N>
    requires std::is_arithmetic_v<T> && (N > 0)
[[nodiscard]] constexpr T
product(const std::array<T, N>& arr) noexcept
{
    T res = arr[0];
    for (std::size_t i = 1; i < N; ++i)
        res *= arr[i];
    return res;
}

template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] constexpr T
product([[maybe_unused]] const std::array<T, 0>& arr) noexcept
{
    return T{1};
}

/**
    @brief Compute the product of numbers.

    @tparam Ts Types of the arguments.

    @param x Input values.

    @return Product of the values.
*/
template <typename... Ts>
    requires (std::is_arithmetic_v<Ts> && ...)
[[nodiscard]] constexpr auto
product(Ts... x) noexcept
{
    return (x*...);
}

} // namespace zest
