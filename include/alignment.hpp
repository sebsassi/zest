#pragma once

#include <cstdlib>
#include <cstddef>
#include <limits>
#include <concepts>

namespace zest
{

namespace detail
{

constexpr bool is_power_of_two(std::size_t n)
{
    return !(n & (n - 1UL));
}

[[nodiscard]] constexpr std::size_t
ctz(std::size_t n)
{
    if (!n) return 8*sizeof(std::size_t);

    n = (n ^ (n - 1)) >> 1;
    std::size_t count = 0;
    for (; n; ++count)
        n >>= 1;
    return count;
}

template <std::size_t POWER_OF_TWO>
[[nodiscard]] constexpr std::size_t
next_divisible(std::size_t n)
{
    constexpr std::size_t shift = detail::ctz(POWER_OF_TWO);
    return ((n + (POWER_OF_TWO - 1)) >> shift) << shift;
}

}



/**
    @brief Descriptor corresponding to no alignment.
*/
struct NoAlignment
{
    static constexpr std::size_t byte_alignment = 1;
    
    template <typename T>
    [[nodiscard]] static constexpr std::size_t
    vector_size() noexcept { return 1; }
};

/**
    @brief Descriptor for SIMD vector alignment.

    @tparam BYTE_ALIGNMENT number of bytes to align to

    @note `BYTE_ALIGNMENT` must be a power of two.
*/
template <std::size_t BYTE_ALIGNMENT>
struct VectorAlignment
{
    static constexpr std::size_t byte_alignment = BYTE_ALIGNMENT;

    /**
        @brief Number of elements that fit in a SIMD vector of given type.

        @tparam T type of elements
    */
    template <typename T>
    [[nodiscard]] static constexpr std::size_t vector_size() noexcept
    {
        return byte_alignment/sizeof(T);
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
        return detail::next_divisible<Alignment::byte_alignment>(n*sizeof(T));
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
            = (std::numeric_limits<std::size_t>::max() - Alignment::byte_alignment)/sizeof(T);
        if (n > max_count) throw std::bad_array_new_length();
 
        auto p = static_cast<T*>(std::aligned_alloc(Alignment::byte_alignment, aligned_size<T, Alignment>(n)));
        if (!p) throw std::bad_alloc();

        return p;
    }
 
    void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept
    {
        std::free(p);
    }
};

}