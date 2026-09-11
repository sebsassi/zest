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

#include <cstddef>
#include <span>

#include "md_array.hpp"

namespace zest
{

template <typename ElementType, std::size_t buffer_count_param, std::size_t extent_param>
    requires (buffer_count_param > 0)
class BufferChain
{
public:
    static constexpr std::size_t buffer_count = buffer_count_param;
    static constexpr std::size_t extent = extent_param;

    BufferChain() = default;
    BufferChain(std::size_t size) requires (extent == std::dynamic_extent):
        m_buffer{size} {}

    void resize(std::size_t size) requires (extent == std::dynamic_extent)
    {
        m_buffer.reshape(size);
    }

    [[nodiscard]] std::size_t buffer_size() const noexcept
    {
        return m_buffer.template extent<1>();
    }

    template <std::size_t index>
        requires (index < buffer_count_param)
    [[nodiscard]] std::span<double, extent> previous() noexcept
    {
        return m_buffer[m_chain[index]].flatten();
    }

    template <std::size_t index>
        requires (index < buffer_count_param)
    [[nodiscard]] std::span<const ElementType, extent> previous() const noexcept
    {
        return m_buffer[m_chain[index]].flatten();
    }

    [[nodiscard]] std::span<ElementType, extent> current() noexcept
    {
        return m_buffer[m_chain[0]].flatten();
    }

    [[nodiscard]] std::span<const ElementType, extent> current() const noexcept
    {
        return m_buffer[m_chain[0]].flatten();
    }

    [[nodiscard]] std::span<ElementType, extent> next() noexcept
    {
        return m_buffer[m_chain.back()].flatten();
    }

    [[nodiscard]] std::span<const ElementType, extent> next() const noexcept
    {
        return m_buffer[m_chain.back()].flatten();
    }

    void advance()
    {
        const std::size_t back = m_chain.back();
        for (std::size_t i = buffer_count - 1; i > 0; --i)
            m_chain[i] = m_chain[i - 1];

        m_chain[0] = back;
    }

private:
    MDArray<ElementType, buffer_count_param, extent> m_buffer;
    std::array<std::size_t, buffer_count_param> m_chain
        = []<std::size_t... I>(std::index_sequence<I...>)
            {
                return std::array{I...,};
            }(std::make_index_sequence<buffer_count_param>{});
};

} // namespace zest
