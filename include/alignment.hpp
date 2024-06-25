#pragma once

#include <cstdlib>
#include <cstddef>
#include <limits>
#include <concepts>

namespace zest
{

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

    @note `BYTE_ALIGNMENT` should normally be a power of two.
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

/**
    @brief Figure out the number of bytes needed to store a number of elements with given byte alignment.

    @tparam T type of allocated object
    @tparam BYTE_ALIGNMENT number of bytes to align to

    @param n number of elements

    @return number of bytes
*/
template<typename T, std::size_t BYTE_ALIGNMENT>
[[nodiscard]] constexpr std::size_t aligned_size(std::size_t n) noexcept
{
    if constexpr (BYTE_ALIGNMENT == 1)
        return n*sizeof(T);
    else if constexpr (!(BYTE_ALIGNMENT & (BYTE_ALIGNMENT - 1UL)))
    {
        if (n*sizeof(T) & (BYTE_ALIGNMENT - 1UL))
            return ((n*sizeof(T)) & (~(BYTE_ALIGNMENT - 1UL))) + BYTE_ALIGNMENT;
        else
            return n*sizeof(T);
    }
    else
    {
        if (n*sizeof(T) % BYTE_ALIGNMENT)
            return (1UL + (n*sizeof(T))/BYTE_ALIGNMENT)*BYTE_ALIGNMENT;
        else
            return n*sizeof(T);
    }
}

/**
    @brief Allocator class for allocating aligned memory.

    @tparam T type of allocated object
    @tparam BYTE_ALIGNMENT number of bytes to align to
*/
template<typename T, std::size_t BYTE_ALIGNMENT>
struct AlignedAllocator
{
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind { using other = AlignedAllocator<T, BYTE_ALIGNMENT>; };
 
    [[nodiscard]] T* allocate(std::size_t n)
    {
        constexpr std::size_t max_count
            = (std::numeric_limits<std::size_t>::max() - BYTE_ALIGNMENT)/sizeof(T);
        if (n > max_count) throw std::bad_array_new_length();
 
        auto p = static_cast<T*>(std::aligned_alloc(BYTE_ALIGNMENT, aligned_size<T, BYTE_ALIGNMENT>(n)));
        if (!p) throw std::bad_alloc();

        return p;
    }
 
    void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept
    {
        std::free(p);
    }
};

}