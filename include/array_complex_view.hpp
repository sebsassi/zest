#pragma once

#include <span>
#include <complex>

namespace zest
{
    
/*
Convenience function for viewing a contiguous sequence of `std::array<T, 2>` as contiguous sequence of `std::complex<T>`.
*/
template <typename T>
[[nodiscard]] std::span<std::complex<T>>
as_complex_span(std::span<std::array<T, 2>> x) noexcept
{
    return std::span<std::complex<T>>(
            reinterpret_cast<std::complex<T>*>(x.data()), x.size());
}

/*
Convenience function for viewing a contiguous sequence of `std::complex<T>` as contiguous sequence of `std::array<T, 2>`.
*/
template <typename T>
[[nodiscard]] std::span<std::array<T, 2>>
as_array_span(std::span<std::complex<T>> x) noexcept
{
    return std::span<std::array<T, 2>>(
            reinterpret_cast<std::array<T, 2>*>(x.data()), x.size());
}

}