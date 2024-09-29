/*
Copyright (c) 2024 Sebastian Sassi

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

#include <span>
#include <complex>

namespace zest
{
    
/**
    @brief Convenience function for viewing a contiguous sequence of `std::array<T, 2>` as contiguous sequence of `std::complex<T>` for some type `T`.

    @tparam T type of array elements

    @param x sequence of `std::array<T, 2>`

    @return view of input as sequence of `std::complex<T>`

    @note `T` has the same caveats it has for `std::complex`.
*/
template <typename T>
[[nodiscard]] constexpr std::span<std::complex<T>>
as_complex_span(std::span<std::array<T, 2>> x) noexcept
{
    return std::span<std::complex<T>>(
            reinterpret_cast<std::complex<T>*>(x.data()), x.size());
}

/**
    @brief Convenience function for viewing a contiguous sequence of `std::complex<T>` as contiguous sequence of `std::array<T, 2>`. for some type `T`.

    @tparam T type of array elements

    @param x sequence of `std::complex<T>`

    @return view of input as sequence of `std::array<T, 2>`

    @note `T` has the same caveats it has for `std::complex`.
*/
template <typename T>
[[nodiscard]] constexpr std::span<std::array<T, 2>>
as_array_span(std::span<std::complex<T>> x) noexcept
{
    return std::span<std::array<T, 2>>(
            reinterpret_cast<std::array<T, 2>*>(x.data()), x.size());
}

} // namespace zest