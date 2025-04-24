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
#include <vector>
#include <span>
#include <array>

#include "md_span.hpp"

namespace zest
{

/**
    @brief Multidimensional array container.

    @tparam ElementType type of array elements
    @tparam rank_param number of array dimensions
*/
template <typename ElementType, std::size_t rank_param>
class MDArray
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using index_type = std::size_t;
    using data_handle_type = element_type*;
    using View = MDSpan<element_type, rank_param>;
    using ConstView = MDSpan<const element_type, rank_param>;

    /**
        @brief Rank of the array.
    */
    static constexpr std::size_t rank() { return rank_param; }

    MDArray() = default;
    MDArray(const std::array<std::size_t, rank_param>& extents):
        m_data(detail::prod(extents)), m_size(detail::prod(extents)), m_extents(extents) {}

    [[nodiscard]] operator View() const noexcept
    {
        return View(m_data.data(), m_size, m_extents);
    }

    [[nodiscard]] operator ConstView() const noexcept
    {
        return ConstView(m_data.data(), m_size, m_extents);
    }

    [[nodiscard]] operator std::span<element_type>() noexcept
    {
        return m_data;
    }
    
    [[nodiscard]] operator std::span<const element_type>() const noexcept
    {
        return m_data;
    }
    
    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] data_handle_type data() const noexcept
    {
        return m_data.data();
    }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] size_type size() const noexcept
    {
        return m_size;
    }

    /**
        @brief Check if the size is zero.
    */
    [[nodiscard]] bool empty() const noexcept
    {
        return m_size == 0;
    }

    /**
        @brief Dimensions of the array.
    */
    [[nodiscard]] const std::array<std::size_t, rank_param>&
    extents() const noexcept
    {
        return m_extents;
    }

    /**
        @brief Dimension of the array along an axis.
    */
    [[nodiscard]] std::size_t
    extent(std::size_t i) const noexcept
    {
        return m_extents[i];
    }

    /**
        @brief Change the shape of the array.
        
        @param extents shape of the new array
    */
    void reshape(const std::array<size_type, rank_param>& extents)
    {
        m_size = detail::prod(extents);
        m_data.resize(m_size);
        m_extents = extents;
    }

    template <typename... Ts>
        requires (sizeof...(Ts) == rank_param)
    [[nodiscard]] element_type& operator()(Ts... inds) noexcept
    {
        return m_data[detail::index(m_extents, inds...)];
    }

    template <typename... Ts>
        requires (sizeof...(Ts) == rank_param)
    [[nodiscard]] const element_type& operator()(Ts... inds) const noexcept
    {
        return m_data[detail::index(m_extents, inds...)];
    }

    template <typename... Ts>
        requires (sizeof...(Ts) < rank_param)
    [[nodiscard]] MDSpan<element_type, rank_param - sizeof...(Ts)>
    operator()(Ts... inds) noexcept
    {
        const index_type ind = detail::index(m_extents, inds...);
        const std::array<index_type, rank_param - sizeof...(Ts)> new_extents = detail::last<rank_param - sizeof...(Ts)>(m_extents);
        const size_type new_size = detail::prod(new_extents);
        return MDSpan<element_type, rank_param - sizeof...(Ts)>(m_data.data() + ind*new_size, new_size, new_extents);
    }

    template <typename... Ts>
        requires (sizeof...(Ts) < rank_param)
    [[nodiscard]] MDSpan<const element_type, rank_param - sizeof...(Ts)>
    operator()(Ts... inds) const noexcept
    {
        const index_type ind = detail::index(m_extents, inds...);
        const std::array<index_type, rank_param - sizeof...(Ts)> new_extents = detail::last<rank_param - sizeof...(Ts)>(m_extents);
        const size_type new_size = detail::prod(new_extents);
        return MDSpan<element_type, rank_param - sizeof...(Ts)>(m_data.data() + ind*new_size, new_size, new_extents);
    }

    template <typename T>
        requires (rank_param == 1UL)
    [[nodiscard]] element_type& operator[](T i) noexcept
    {
        return (*this)(i);
    }

    template <typename T>
        requires (rank_param == 1UL)
    [[nodiscard]] const element_type& operator[](T i) const noexcept
    {
        return (*this)(i);
    }

    template <typename T>
    [[nodiscard]] MDSpan<element_type, rank_param - 1UL>
    operator[](T i) noexcept
    {
        return (*this)(i);
    }

    template <typename T>
    [[nodiscard]] MDSpan<const element_type, rank_param - 1UL>
    operator[](T i) const noexcept
    {
        return (*this)(i);
    }

private:
    std::vector<element_type> m_data;
    size_type m_size{};
    std::array<size_type, rank_param> m_extents{};
};

} // namespace zest
