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

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>

#include "utility.hpp"

namespace zest
{

namespace array::detail
{

template <std::integral SizeType, std::size_t rank, std::integral IndexType, std::integral... Inds>
    requires (0 <= sizeof...(Inds) && sizeof...(Inds) + 1 <= rank)
[[nodiscard]] constexpr auto
index(const std::array<SizeType, rank>& extents, IndexType ind, Inds... inds) noexcept
{
    auto impl = [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        assert(ind < IndexType(extents[0]) && ((IndexType(inds) < IndexType(extents[1 + I])) && ...));
        IndexType res = ind;
        ([&]{ res = res*IndexType(extents[1 + I]) + IndexType(inds); }(), ...);
        if constexpr (sizeof...(Inds) + 1 == rank)
            return res;
        else
            return product(take_last<rank - sizeof...(Inds) - 1>(extents))*res;
    };
    return impl(std::make_index_sequence<sizeof...(Inds)>{});
}

// template <std::size_t static_extent, std::size_t... static_extents, std::integral SizeType, std::size_t N, std::integral IndexType, std::integral... Inds>
// [[nodiscard]] constexpr auto
// index(const std::array<SizeType, N>& dynamic_extents, IndexType ind, Inds... inds) noexcept
// {
//     auto impl = [&]<std::size_t... I>(std::index_sequence<I...>)
//     {
//         std::size_t i = (static_extent == std::dynamic_extent) ? 1 : 0;
//         IndexType res = ind;
//         ([&]{
//             if constexpr (static_extents == std::dynamic_extent)
//             {
//                 res = res*IndexType(dynamic_extents[i]) + IndexType(inds);
//                 ++i;
//             }
//             else
//                 res = res*IndexType(static_extents) + IndexType(inds);
//         }(),...);
//         return res;
//     };
//     return impl(std::make_index_sequence<sizeof...(Inds)>{});
// }

} // namespace array::detail

/**
    @brief Iterator presenting an infinite arithmetic sequence of integer indices with arbitrary stride.

    @tparam IndexType type of the index
    @tparam stride_param stride of the index
*/
template <std::integral IndexType, IndexType stride_param>
class IndexIterator
{
public:
    using index_type = IndexType;
    using size_type = std::size_t;
    using difference_type = index_type;

    static constexpr index_type stride = stride_param;

    constexpr IndexIterator() = default;
    explicit constexpr IndexIterator(index_type index): m_index(index) {}

    /**
        @brief Increment index by stride.
    */
    constexpr IndexIterator& operator++() noexcept
    {
        m_index += stride;
        return *this;
    }

    /**
        @brief Decrement index by stride.
    */
    constexpr IndexIterator& operator--() noexcept
    {
        m_index -= stride;
        return *this;
    }

    /**
        @brief Increment index by stride.
    */
    constexpr IndexIterator operator++(int) noexcept
    {
        auto out = IndexIterator{m_index};
        m_index += stride;
        return out;
    }

    /**
        @brief Decrement index by stride.
    */
    constexpr IndexIterator operator--(int) noexcept
    {
        auto out = IndexIterator{m_index};
        m_index -= stride;
        return out;
    }

    /**
        @brief Increment index by multiple strides.
    */
    constexpr IndexIterator& operator+=(index_type n) noexcept
    { m_index += n*stride; return *this; }


    /**
        @brief Decrement index by multiple strides.
    */
    constexpr IndexIterator& operator-=(index_type n) noexcept
    { m_index += n*stride; return *this; }

    /**
        @brief Add `n` strides to index.
    */
    [[nodiscard]] constexpr IndexIterator
    operator+(difference_type n) const noexcept
    { return IndexIterator{m_index + n*stride}; }

    /**
        @brief Subtract `n` strides from index.
    */
    [[nodiscard]] constexpr IndexIterator
    operator-(difference_type n) const noexcept
    { return IndexIterator{m_index - n*stride}; }

    /**
        @brief Get value of index.
    */
    [[nodiscard]] constexpr index_type operator*() noexcept
    {
        return m_index;
    }

    /**
        @brief Get value of index `n` strides forward from current index.
    */
    [[nodiscard]] constexpr index_type operator[](index_type n) noexcept
    {
        return m_index + n*stride;
    }

    [[nodiscard]] constexpr bool operator==(const IndexIterator& other) const noexcept = default;
    [[nodiscard]] constexpr auto operator<=>(const IndexIterator& other) const noexcept = default;

    /**
        @brief Get value of index.
    */
    [[nodiscard]] constexpr index_type index() const noexcept { return m_index; }

private:
    index_type m_index{};
};

template <std::integral IndexType, IndexType stride_param>
class BasicIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<index_type, stride_param>;

    explicit constexpr BasicIndexRange(index_type begin, index_type end):
        m_begin(std::min(begin, end)), m_end(end) {}

    /**
        @brief Iterator to the beginning of the range.
    */
    [[nodiscard]] constexpr iterator begin() const noexcept { return iterator{m_begin}; }

    /**
        @brief Iterator to the end of the range.
    */
    [[nodiscard]] constexpr iterator end() const noexcept { return iterator{m_end}; }

private:
    index_type m_begin{};
    index_type m_end{};
};

template <
    std::integral IndexType, IndexType begin_param, IndexType end_param,
    IndexType stride_param>
class StaticBasicIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<index_type, stride_param>;

    [[nodiscard]] constexpr iterator 
    begin() const noexcept { return iterator{begin_param}; }

    [[nodiscard]] constexpr iterator
    end() const noexcept { return iterator{end_param}; }
};

template <std::integral IndexType, IndexType begin, IndexType end>
using StaticStandardIndexRange = StaticBasicIndexRange<
    IndexType, begin, end, IndexType{1}>;

template <std::integral IndexType>
using SingleIndexRange = StaticBasicIndexRange<
    IndexType, IndexType{0}, IndexType{1}, IndexType{1}>;

/**
    @brief Range of integer indices.

    @tparam IndexType type of the index
*/
template <std::integral IndexType>
class StandardIndexRange: public BasicIndexRange<IndexType, IndexType{1}>
{
public:
    using index_type = BasicIndexRange<IndexType, IndexType{1}>::index_type;
    using iterator = BasicIndexRange<IndexType, IndexType{1}>::iterator;

    /**
        @brief Constructs a range of indices `[0, end)`.

        @param end end of index range
    */
    StandardIndexRange(index_type end): 
        BasicIndexRange<index_type, index_type{1}>(index_type{}, end) {};

    /**
        @brief Constructs a range of indices `[begin, end)`.

        @param begin start of index range
        @param end end of index range
    */
    StandardIndexRange(index_type begin, index_type end):
        BasicIndexRange<index_type, index_type{1}>(begin, end) {};
};

/**
    @brief Range of even or odd integer indices.

    @tparam IndexType type of the index
*/
template <std::integral IndexType>
class ParityIndexRange: public BasicIndexRange<IndexType, IndexType{2}>
{
public:
    using index_type = BasicIndexRange<IndexType, IndexType{2}>::index_type;
    using iterator = BasicIndexRange<IndexType, IndexType{2}>::iterator;

    /**
        @brief Constructs a range of indices `[(end + 1) % 2, end + 1)`.

        @param end end of index range
    */
    explicit constexpr ParityIndexRange(index_type end):
        BasicIndexRange<index_type, index_type{2}>(1 & (end + 1), end + 1) {}

    /**
        @brief Constructs a range of indices `[2*floor(begin/2) + (end + 1) % 2, end + 1)`.

        @param begin start of index range
        @param end end of index range
    */
    explicit constexpr ParityIndexRange(index_type begin, index_type end):
        BasicIndexRange<index_type, index_type{2}>(begin + (1 & (begin ^ (end + 1))), end + 1) {}
};

/**
    @brief Range of integer indices symmetric about zero.

    @tparam IndexType type of the index
*/
template <std::signed_integral IndexType>
class SymmetricIndexRange: public BasicIndexRange<IndexType, IndexType{1}>
{
public:
    using index_type = BasicIndexRange<IndexType, IndexType{1}>::index_type;
    using iterator = BasicIndexRange<IndexType, IndexType{1}>::iterator;

    /**
        @brief Constructs a range of indices `(-end, end)`.

        @param end end of index range
    */
    explicit constexpr SymmetricIndexRange(index_type end):
        BasicIndexRange<index_type, index_type{1}>(1 - end, end) {}

    /**
        @brief Constructs a range of indices `[begin, end)`.

        @param begin start of index range
        @param end end of index range
    */
    constexpr SymmetricIndexRange(index_type begin, index_type end): 
        BasicIndexRange<index_type, index_type{1}>(begin, end) {}
};

} // namespace zest
