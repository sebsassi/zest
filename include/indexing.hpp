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

#include <cstddef>
#include <concepts>

namespace zest
{

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
    { m_index += n; return *this; }


    /**
        @brief Decrement index by multiple strides.
    */
    constexpr IndexIterator& operator-=(index_type n) noexcept
    { m_index += n; return *this; }

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

    [[nodiscard]] constexpr bool
    operator==(const IndexIterator& b) const noexcept
    { return m_index == b.index(); }

    [[nodiscard]] constexpr bool
    operator!=(const IndexIterator& b) const noexcept
    { return m_index != b.index(); }

    [[nodiscard]] constexpr bool
    operator<=(const IndexIterator& b) const noexcept
    { return m_index <= b.index(); }

    [[nodiscard]] constexpr bool
    operator>=(const IndexIterator& b) const noexcept
    { return m_index >= b.index(); }

    [[nodiscard]] constexpr bool
    operator<(const IndexIterator& b) const noexcept
    { return m_index < b.index(); }

    [[nodiscard]] constexpr bool
    operator>(const IndexIterator& b) const noexcept
    { return m_index > b.index(); }

    /**
        @brief Get value of index.
    */
    [[nodiscard]] constexpr index_type index() const noexcept
    { return m_index; }

private:
    index_type m_index{};
};

/**
    @brief Range of integer indices.

    @tparam IndexType type of the index
*/
template <std::integral IndexType>
class StandardIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<index_type, 1UL>;

    /**
        @brief Constructs a range of indices `[0, end)`.

        @param end end of index range
    */
    explicit constexpr StandardIndexRange(index_type end):
        m_begin(0), m_end(end) {}

    /**
        @brief Constructs a range of indices `[begin, end)`.

        @param begin start of index range
        @param end end of index range
    */
    constexpr StandardIndexRange(index_type begin, index_type end):
        m_begin(begin), m_end(end) {}

    /**
        @brief Iterator to the beginning of the range.
    */
    [[nodiscard]] constexpr iterator begin() const noexcept
    { return iterator{m_begin}; }

    /**
        @brief Iterator to the end of the range.
    */
    [[nodiscard]] constexpr iterator end() const noexcept
    { return iterator{m_end}; }
private:
    index_type m_begin{};
    index_type m_end{};
};

/**
    @brief Range of even or odd integer indices.

    @tparam IndexType type of the index
*/
template <std::integral IndexType>
class ParityIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<index_type, 2UL>;

    /**
        @brief Constructs a range of indices `[end % 2, end)`.

        @param end end of index range
    */
    explicit constexpr ParityIndexRange(index_type end):
        m_begin(end & 1), m_end(end) {}
    
    /**
        @brief Constructs a range of indices `[2*floor(begin/2) + end % 2, end)`.

        @param begin start of index range
        @param end end of index range
    */
    constexpr ParityIndexRange(index_type begin, index_type end):
        m_begin((begin & ~1UL) + (end & 1)), m_end(end) {}

    /**
        @brief Iterator to the beginning of the range.
    */
    [[nodiscard]] constexpr iterator begin() const noexcept
    { return iterator{m_begin}; }

    /**
        @brief Iterator to the end of the range.
    */
    [[nodiscard]] constexpr iterator end() const noexcept
    { return iterator{m_end}; }
private:
    index_type m_begin;
    index_type m_end;
};

/**
    @brief Range of integer indices symmetric about zero.

    @tparam IndexType type of the index
*/
template <std::signed_integral IndexType>
class SymmetricIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<int, 1UL>;

    /**
        @brief Constructs a range of indices `(-end, end)`.

        @param end end of index range
    */
    explicit constexpr SymmetricIndexRange(index_type end):
        m_begin(1 - end), m_end(end) {}

    /**
        @brief Constructs a range of indices `[begin, end)`.

        @param begin start of index range
        @param end end of index range
    */
    constexpr SymmetricIndexRange(index_type begin, index_type end):
        m_begin(begin), m_end(end) {}

    /**
        @brief Iterator to the beginning of the range.
    */
    [[nodiscard]] constexpr iterator begin() const noexcept
    { return iterator{m_begin}; }
    
    /**
        @brief Iterator to the end of the range.
    */
    [[nodiscard]] constexpr iterator end() const noexcept
    { return iterator{m_end}; }
private:
    index_type m_begin{};
    index_type m_end{};
};

} // namespace zest