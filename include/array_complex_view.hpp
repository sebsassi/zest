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