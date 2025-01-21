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

#include <cstdlib>
#include <cstddef>
#include <limits>
#include <new>

namespace zest
{

namespace detail
{

constexpr bool is_power_of_two(std::size_t n) noexcept
{
    return !(n & (n - 1UL));
}

// Count trailing zeros
[[nodiscard]] constexpr std::size_t ctz(std::size_t n) noexcept
{
    if (!n) return 8*sizeof(std::size_t);

    n = (n ^ (n - 1)) >> 1;
    std::size_t count = 0;
    for (; n; ++count)
        n >>= 1;
    return count;
}


// Smallest number greater than or equal to `n` divisible by `power_of_two`
template <std::size_t power_of_two>
    requires (is_power_of_two(power_of_two))
[[nodiscard]] constexpr std::size_t next_divisible(std::size_t n) noexcept
{
    constexpr std::size_t shift = detail::ctz(power_of_two);
    return ((n + (power_of_two - 1)) >> shift) << shift;
}

} // namespace detail

/**
    @brief Descriptor corresponding to no alignment.
*/
struct NoAlignment
{
    static constexpr std::size_t bytes = 1;

    /**
        @brief Number of elements that fit in a SIMD vector of given type.

        @tparam T type of elements
    */
    template <typename T>
    [[nodiscard]] static constexpr std::size_t
    vector_size() noexcept { return 1; }
};

/**
    @brief Descriptor for SIMD vector alignment.

    @tparam byte_alignment number of bytes to align to

    @note `byte_alignment` must be a power of two.
*/
template <std::size_t byte_alignment>
struct VectorAlignment
{
    static constexpr std::size_t bytes = byte_alignment;

    /**
        @brief Number of elements that fit in a SIMD vector of given type.

        @tparam T type of elements
    */
    template <typename T>
    [[nodiscard]] static constexpr std::size_t vector_size() noexcept
    {
        return bytes/sizeof(T);
    }
};

/**
    @brief Descriptor for SSE (16 byte) alignment.
*/
using SSEAlignment = VectorAlignment<16>;

/**
    @brief Descriptor for AVX (32 byte) alignment.
*/
using AVXAlignment = VectorAlignment<32>;

/**
    @brief Descriptor for AVX512 (64 byte) alignment.
*/
using AVX512Alignment = VectorAlignment<64>;

/**
    @brief Descriptor for cache line (64 byte) alignment.
*/
using CacheLineAlignment = VectorAlignment<64>;

template <typename T>
concept valid_simd_alignment = std::same_as<T, NoAlignment>
    || std::same_as<T, SSEAlignment> || std::same_as<T, AVXAlignment>
    || std::same_as<T, AVX512Alignment>;

/**
    @brief Figure out the number of bytes needed to store a number of elements with given byte alignment.

    @tparam T type of allocated object
    @tparam BYTE_ALIGNMENT number of bytes to align to

    @param n number of elements

    @return number of bytes
*/
template<typename T, valid_simd_alignment Alignment>
[[nodiscard]] constexpr std::size_t aligned_size(std::size_t n) noexcept
{
    if constexpr (std::same_as<Alignment, NoAlignment>)
        return n*sizeof(T);
    else
        return detail::next_divisible<Alignment::bytes>(n*sizeof(T));
}

/**
    @brief Allocator class for allocating aligned memory.

    @tparam T type of allocated object
    @tparam BYTE_ALIGNMENT number of bytes to align to
*/
template<typename T, valid_simd_alignment Alignment>
struct AlignedAllocator
{
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind { using other = AlignedAllocator<T, Alignment>; };
 
    [[nodiscard]] T* allocate(std::size_t n)
    {
        constexpr std::size_t max_count
            = (std::numeric_limits<std::size_t>::max() - Alignment::bytes)/sizeof(T);
        if (n > max_count) throw std::bad_array_new_length();
 
        auto p = static_cast<T*>(std::aligned_alloc(Alignment::bytes, aligned_size<T, Alignment>(n)));
        if (!p) throw std::bad_alloc();

        return p;
    }
 
    void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept
    {
        std::free(p);
    }
};

} // namespace zest