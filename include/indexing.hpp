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

/**
    @brief A basic strided range of integer indices.

    @tparam IndexType Type of the indices.
    @tparam stride_param Stride of the index range.

    This class describes a basic strided index range, which start from some
    index `begin`, and increments by `stride_param` until the index is equal
    to or greater than some `end` index. That is, the `end` index is never
    contained in the range.
*/
template <std::integral IndexType, IndexType stride_param>
class BasicIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<index_type, stride_param>;

    /**
        @brief Constructs an index range from its starting point and end point.

        @param begin Starting index of the range.
        @param end End index of the range.
    */
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

/**
    @brief A basic strided range of integer indices defined entirely at compile
    time.

    @tparam IndexType Type of the indices.
    @tparam begin_param Starting index of the range.
    @tparam end_param End index of the range.
    @tparam stride_param Stride of the index range.

    This class is like `BasicIndexRange`, but its start and end index are
    defined at compile time.
*/
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

/**
    @brief A basic index range with unit stride.

    @tparam IndexType type of the index

    This class defines the usual standard index range that contains all indices
    in the range `[begin, end)`.
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
    @brief A basic index range with unit stride defined entirely at compile
    time.

    @tparam IndexType Type of the indices.
    @tparam begin_param Starting index of the range.
    @tparam end_param End index of the range.

    This class is like `StandardIndexRange`, but its start and end index are
    defined at compile time.
*/
template <std::integral IndexType, IndexType begin, IndexType end>
using StaticStandardIndexRange = StaticBasicIndexRange<
    IndexType, begin, end, IndexType{1}>;

/**
    @brief Index range to a single-element collection.

    @tparam IndexType Type of the index.

    This class defines an index range for a collection of one element,
    containing only the index zero.
*/
template <std::integral IndexType>
using SingleIndexRange = StaticStandardIndexRange<
    IndexType, IndexType{0}, IndexType{1}>;

/**
    @brief An enum for describing the parity of a thing.
*/
enum class Parity
{
    none,   /// Has no parity.
    mixed,  /// Could be of either even or odd parity.
    even,   /// Has even parity.
    odd     /// Has odd parity.
};

/**
    @brief Range of even or odd integer indices.

    @tparam IndexType Type of the index.
    @tparam parity Parity of the indices.

    This class describes an index range, where the indices are either even or
    odd. Depending on the `parity` template parameter, an object of the class
    may be constructed to either contain only even, only odd, or possibly either
    even or odd indices. Both `Parity::odd` and `Parity::even` guarantee the
    range contains only indices of the given parity.
*/
template <std::integral IndexType, Parity parity = Parity::mixed>
    requires (parity != Parity::none)
class ParityIndexRange: public BasicIndexRange<IndexType, IndexType{2}>
{
public:
    using index_type = BasicIndexRange<IndexType, IndexType{2}>::index_type;
    using iterator = BasicIndexRange<IndexType, IndexType{2}>::iterator;

    /**
        @brief Constructs a range of indices `[(end + 1) % 2, end + 1)`.

        @param end end of index range

        The index range has the following limits for given `parity`:
            - `Parity::mixed`: `[(end + 1) % 2, end + 1)`
            - `Parity::even`: `[0, end + 1)`
            - `Parity::odd`: `[1, end + 1)`
    */
    explicit constexpr ParityIndexRange(index_type end) requires (parity == Parity::mixed):
        BasicIndexRange<index_type, index_type{2}>(1 & (end + 1), end + 1) {}

    explicit constexpr ParityIndexRange(index_type end) requires (parity == Parity::even):
        BasicIndexRange<index_type, index_type{2}>(0, (end + 1) & (~1UL)) {}

    explicit constexpr ParityIndexRange(index_type end) requires (parity == Parity::odd):
        BasicIndexRange<index_type, index_type{2}>(1, (end & (~1UL)) + 1) {}

    /**
        @brief Constructs a range of indices `[2*floor(begin/2) + (end + 1) % 2, end + 1)`.

        @param begin start of index range
        @param end end of index range

        This index range has the following limits for given `parity`:
            - `Parity::mixed`: `[2*floor(begin/2) + (end + 1) % 2, end + 1)`
            - `Parity::even`: `[2*floor(begin/2), end + 1)`
            - `Parity::odd`: `[2*floor(begin/2) + 1, end + 1)`
    */
    explicit constexpr ParityIndexRange(index_type begin, index_type end) requires (parity == Parity::mixed):
        BasicIndexRange<index_type, index_type{2}>(begin + (1 & (begin ^ (end + 1))), end + 1) {}

    explicit constexpr ParityIndexRange(index_type begin, index_type end) requires (parity == Parity::even):
        BasicIndexRange<index_type, index_type{2}>((begin + 1) & (~1UL), (end + 1) & (~1UL)) {}

    explicit constexpr ParityIndexRange(index_type begin, index_type end) requires (parity == Parity::odd):
        BasicIndexRange<index_type, index_type{2}>((begin & (~1UL)) + 1, (end & (~1UL)) + 1) {}
};

/**
    @brief Range of integer indices symmetric about zero.

    @tparam IndexType Type of the index.

    This class behaves more or less like a `StandardIndexRange`, but if
    constructed with only `end`, it gives a range `(-end, end)` instead of
    `[0, end)`.
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
