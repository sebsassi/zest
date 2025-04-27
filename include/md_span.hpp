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
#include <span>
#include <array>

namespace zest
{

namespace detail
{

template <std::size_t M, typename T>
auto last(T a) noexcept
{
    std::array<typename std::remove_cvref<T>::type::value_type, M> res{};
    for (std::size_t i = 0; i < M; ++i)
        res[i] = a[(a.size() - M) + i];
    return res;
}

template <typename T>
auto prod(T a) noexcept
{
    auto res = a[0];
    for (std::size_t i = 1; i < a.size(); ++i)
        res *= a[i];
    return res;
}

template <typename size_type, std::size_t N, std::size_t I, typename index_type>
[[nodiscard]] constexpr index_type
index_impl([[maybe_unused]] const std::array<size_type, N>& extents, index_type ind) noexcept
{
    return ind;
}

template <typename size_type, std::size_t N, std::size_t I, typename index_type, typename... Ts>
[[nodiscard]] constexpr index_type
index_impl(const std::array<size_type, N>& extents, index_type ind, index_type next, Ts... inds) noexcept
{
    if constexpr (I < N)
        return index_impl<size_type, N, I + 1UL>(extents, ind*extents[I] + next, inds...);
    else
        return ind;
}

template <typename size_type, std::size_t N, typename... Ts>
[[nodiscard]] constexpr auto
index(const std::array<size_type, N>& extents, Ts... inds) noexcept
{
    return index_impl<size_type, N, 1UL>(extents, inds...);
}

} // namespace detail

// A future version of this library using C++23 may do away with this class.

/**
    @brief Poor man's mdspan for a non-owning multidimensional array view.

    @tparam ElementType type of array elements
    @tparam rank_param number of array dimensions
*/
template <typename ElementType, std::size_t rank_param>
class MDSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using index_type = std::size_t;
    using data_handle_type = element_type*;
    using ConstView = MDSpan<const element_type, rank_param>;

    /**
        @brief Rank of the array.
    */
    static constexpr std::size_t rank() { return rank_param; }

    constexpr MDSpan() noexcept = default;
    constexpr MDSpan(
        data_handle_type data, const std::array<std::size_t, rank_param>& extents) noexcept:
        m_data(data), m_size(detail::prod(extents)), m_extents(extents) {}

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_extents);
    }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return std::span<element_type>(m_data, m_size);
    }
    
    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] constexpr data_handle_type data() const noexcept
    {
        return m_data;
    }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] constexpr size_type size() const noexcept
    {
        return m_size;
    }

    /**
        @brief Check if the size is zero.
    */
    [[nodiscard]] constexpr bool empty() const noexcept
    {
        return m_size == 0;
    }

    /**
        @brief Dimensions of the array.
    */
    [[nodiscard]] constexpr const std::array<std::size_t, rank_param>&
    extents() const noexcept
    {
        return m_extents;
    }

    /**
        @brief Dimension of the array along an axis.
    */
    [[nodiscard]] constexpr std::size_t extent(std::size_t i) const noexcept
    {
        return m_extents[i];
    }

    /**
        @brief Get flattened view of the array.
    */
    [[nodiscard]] constexpr std::span<element_type> flatten() const noexcept
    {
        return std::span<element_type>(m_data, m_size);
    }

    template <typename... Ts>
        requires (sizeof...(Ts) == rank_param)
    [[nodiscard]] constexpr element_type& operator()(Ts... inds) const noexcept
    {
        return m_data[detail::index(m_extents, inds...)];
    }

    template <typename... Ts>
        requires (sizeof...(Ts) < rank_param)
    [[nodiscard]] constexpr MDSpan<element_type, rank_param - sizeof...(Ts)>
    operator()(Ts... inds) const noexcept
    {
        const index_type ind = detail::index(m_extents, inds...);
        const std::array<index_type, rank_param - sizeof...(Ts)> new_extents = detail::last<rank_param - sizeof...(Ts)>(m_extents);
        const size_type new_size = detail::prod(new_extents);
        return MDSpan<element_type, rank_param - sizeof...(Ts)>(m_data + ind*new_size, new_size, new_extents);
    }

    template <typename T>
        requires (rank_param == 1UL)
    [[nodiscard]] constexpr element_type& operator[](T i) const noexcept
    {
        return (*this)(i);
    }

    template <typename T>
    [[nodiscard]] constexpr MDSpan<element_type, rank_param - 1UL>
    operator[](T i) const noexcept
    {
        return (*this)(i);
    }

protected:
    template <typename T, std::size_t dimension>
    friend class MDSpan;

    template <typename T, std::size_t dimension>
    friend class MDArray;

    constexpr MDSpan(
        data_handle_type data, size_type size, const std::array<std::size_t, rank_param>& extents) noexcept:
        m_data(data), m_size(size), m_extents(extents) {}

private:
    data_handle_type m_data{};
    size_type m_size{};
    std::array<size_type, rank_param> m_extents{};
};

} // namespace zest
