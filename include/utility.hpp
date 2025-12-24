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

#include <complex>
#include <concepts>
#include <type_traits>

namespace zest
{

template <typename T>
concept sequenced = requires (T x)
    {
        { x.order() } -> std::same_as<typename T::size_type>;
    };

template <typename T>
concept complex_float
    = std::same_as<std::remove_const_t<T>, std::complex<typename std::remove_const_t<T>::value_type>>;

template <typename T>
concept complex_or_real_float
    = std::floating_point<std::remove_const_t<T>> || complex_float<T>;

template <typename T>
concept tag_type = (std::is_empty_v<T> && std::is_aggregate_v<T>);

template <typename T, tag_type... Tags>
struct Tag: public T, public Tags...
{
    using T::T;
};

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
take_first_impl(const std::array<T, sizeof...(I)>& arr)
{
    return {std::get<I>(arr)...};
}

template <typename T, std::size_t... I>
[[nodiscard]] constexpr auto
take_first_impl(const T& tuple, std::index_sequence<I...> /*unused*/) noexcept
{
    return std::make_tuple(std::get<I>(tuple)...);
}

} // namespace detail

template <typename T, std::size_t N>
    requires (std::tuple_size_v<T> <= N)
[[nodiscard]] constexpr auto
take_last(const T& tuple_like) noexcept
{
    return take_last_impl(tuple_like, std::make_index_sequence<N>());
}

template <typename T, std::size_t N>
    requires (std::tuple_size_v<T> <= N)
[[nodiscard]] constexpr auto
take_first(const T& tuple_like) noexcept
{
    return take_first_impl(tuple_like, std::make_index_sequence<N>());
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr T
product(const std::array<T, N>& arr) noexcept
{
    T res = arr[0];
    for (std::size_t i = 1; i < N; ++i)
        res *= arr[i];
    return res;
}

} // namespace zest
