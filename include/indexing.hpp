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

namespace zest
{

template <typename IndexType, IndexType stride_param>
class IndexIterator
{
public:
    using index_type = IndexType;
    using size_type = std::size_t;
    using difference_type = index_type;

    static constexpr index_type stride = stride_param;

    constexpr IndexIterator() = default;
    explicit constexpr IndexIterator(index_type index): m_index(index) {}

    constexpr IndexIterator& operator++() noexcept
    {
        m_index += stride;
        return *this;
    }
    constexpr IndexIterator& operator--() noexcept
    {
        m_index -= stride;
        return *this;
    }

    constexpr IndexIterator operator++(int) noexcept
    {
        auto out = IndexIterator{m_index};
        m_index += stride;
        return out;
    }

    constexpr IndexIterator operator--(int) noexcept
    {
        auto out = IndexIterator{m_index};
        m_index -= stride;
        return out;
    }

    constexpr IndexIterator& operator+=(index_type n) noexcept
    { m_index += n; return *this; }

    constexpr IndexIterator& operator-=(index_type n) noexcept
    { m_index += n; return *this; }

    [[nodiscard]] constexpr IndexIterator
    operator+(difference_type n) const noexcept
    { return IndexIterator{m_index + n}; }

    [[nodiscard]] constexpr IndexIterator
    operator-(difference_type n) const noexcept
    { return IndexIterator{m_index - n}; }

    [[nodiscard]] constexpr index_type operator*() noexcept
    {
        return m_index;
    }

    [[nodiscard]] constexpr index_type operator[](index_type n) noexcept
    {
        return m_index + n;
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

    [[nodiscard]] constexpr index_type index() const noexcept
    { return m_index; }

private:
    index_type m_index{};
};

template <typename IndexType>
class StandardIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<index_type, 1UL>;

    explicit constexpr StandardIndexRange(index_type end):
        m_begin(0), m_end(end) {}
    constexpr StandardIndexRange(index_type begin, index_type end):
        m_begin(begin), m_end(end) {}

    [[nodiscard]] constexpr iterator begin() const noexcept
    { return iterator{m_begin}; }
    [[nodiscard]] constexpr iterator end() const noexcept
    { return iterator{m_end}; }
private:
    index_type m_begin{};
    index_type m_end{};
};

template <typename IndexType>
class ParityIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<index_type, 2UL>;

    explicit constexpr ParityIndexRange(index_type end):
        m_begin(end & 1), m_end(end) {}
    constexpr ParityIndexRange(index_type begin, index_type end):
        m_begin((begin & ~1UL) + (end & 1)), m_end(end) {}

    [[nodiscard]] constexpr iterator begin() const noexcept
    { return iterator{m_begin}; }
    [[nodiscard]] constexpr iterator end() const noexcept
    { return iterator{m_end}; }
private:
    index_type m_begin;
    index_type m_end;
};

template <typename IndexType>
class SymmetricIndexRange
{
public:
    using index_type = IndexType;
    using iterator = IndexIterator<int, 1UL>;

    explicit constexpr SymmetricIndexRange(index_type end):
        m_begin(1 - end), m_end(end) {}
    constexpr SymmetricIndexRange(index_type begin, index_type end):
        m_begin(begin), m_end(end) {}

    [[nodiscard]] constexpr iterator begin() const noexcept
    { return iterator{m_begin}; }
    [[nodiscard]] constexpr iterator end() const noexcept
    { return iterator{m_end}; }
private:
    index_type m_begin{};
    index_type m_end{};
};

} // namespace zest