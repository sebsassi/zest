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

template <typename T>
concept tagged = requires { typename T::untag; };

template <typename T>
concept sequence_shaped = requires (T x)
    {
        { x.order() } -> std::same_as<typename T::size_type>;
    };

template <typename T>
concept tensor_shaped
    =  requires (T x, typename std::remove_cvref_t<T>::size_type i)
    {
        { x.extent(i) } -> std::same_as<typename std::remove_cvref_t<T>::size_type>;
    };

template <typename T>
concept complex_float = std::same_as<
    std::remove_const_t<T>, std::complex<typename std::remove_const_t<T>::value_type>>;

template <typename T>
concept complex_or_real_float
    = std::floating_point<std::remove_const_t<T>> || complex_float<T>;

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

template <std::size_t N, typename T>
    requires (N <= std::tuple_size_v<T>)
[[nodiscard]] constexpr auto
take_last(const T& tuple_like) noexcept
{
    return detail::take_last_impl(tuple_like, std::make_index_sequence<N>());
}

template <std::size_t N, typename T>
    requires (N <= std::tuple_size_v<T>)
[[nodiscard]] constexpr auto
take_first(const T& tuple_like) noexcept
{
    return detail::take_first_impl(tuple_like, std::make_index_sequence<N>());
}

template <typename T, typename S>
[[nodiscard]] constexpr auto
append(const T& tuple_like, const S& element) noexcept
{
    return detail::append_impl(
        tuple_like, element, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <typename T, typename S>
[[nodiscard]] constexpr auto
prepend(const T& element, const S& tuple_like) noexcept
{
    return detail::prepend_impl(
        element, tuple_like, std::make_index_sequence<std::tuple_size_v<S>>{});
}

template <typename T, typename S>
[[nodiscard]] constexpr auto
concatenate(const T& tuple_like_t, const S& tuple_like_s) noexcept
{
    return detail::concatenate_impl(
        tuple_like_t, tuple_like_s,
        std::make_index_sequence<std::tuple_size_v<T>>{},
        std::make_index_sequence<std::tuple_size_v<S>>{});
}

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

template <typename... Ts>
    requires (std::is_arithmetic_v<Ts> && ...)
[[nodiscard]] constexpr auto
product(Ts... x) noexcept
{
    return (x*...);
}

} // namespace zest
