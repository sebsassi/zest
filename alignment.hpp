#pragma once

#include <cstdlib>
#include <cstddef>
#include <limits>
#include <concepts>

namespace zest
{

struct NoAlignment
{
    static constexpr std::size_t byte_alignment = 1;
    
    template <typename T>
    static constexpr std::size_t vector_size() { return 1; }
};

template <std::size_t BYTE_ALIGNMENT>
struct VectorAlignment
{
    static constexpr std::size_t byte_alignment = BYTE_ALIGNMENT;

    template <typename T>
    static constexpr std::size_t vector_size()
    {
        return std::max(1, byte_alignment/sizeof(T));
    }
};

using SSEAlignment = VectorAlignment<16>;
using AVXAlignment = VectorAlignment<32>;
using AVX512Alignment = VectorAlignment<64>;
using CacheLineAlignment = VectorAlignment<64>;

/*
Figure out the number of bytes needed to store `n` elements with given byte alignment.
*/
template<typename T, std::size_t BYTE_ALIGNMENT>
constexpr std::size_t aligned_size(std::size_t n)
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

template<typename T, std::size_t BYTE_ALIGNMENT>
struct AlignedAllocator
{
    typedef T value_type;

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