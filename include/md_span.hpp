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

} // namespace detail

// A future version of this library using C++23 may do away with this class.

/**
    @brief Poor man's mdspan for a non-owning multidimensional array view.

    @tparam ElementType type of array elements
    @tparam ndim number of array dimensions
*/
template <typename ElementType, std::size_t ndim>
class MDSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using index_type = std::size_t;
    using data_handle_type = element_type*;
    using ConstView = MDSpan<const element_type, ndim>;

    /**
        @brief Rank of the array.
    */
    static constexpr std::size_t rank() { return ndim; }

    constexpr MDSpan() noexcept = default;
    constexpr MDSpan(
        data_handle_type data, const std::array<std::size_t, ndim>& extents) noexcept:
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
    [[nodiscard]] constexpr const std::array<std::size_t, ndim>&
    extents() const noexcept
    {
        return m_extents;
    }

    /**
        @brief Dimension of the array along an axis.
    */
    [[nodiscard]] constexpr std::size_t
    extent(std::size_t i) const noexcept
    {
        return m_extents[i];
    }

    template <typename... Ts>
        requires (sizeof...(Ts) == ndim)
    [[nodiscard]] constexpr element_type& operator()(Ts... inds) const noexcept
    {
        return m_data[idx(inds...)];
    }

    template <typename... Ts>
        requires (sizeof...(Ts) < ndim)
    [[nodiscard]] constexpr MDSpan<element_type, ndim - sizeof...(Ts)>
    operator()(Ts... inds) const noexcept
    {
        index_type ind = idx(inds...);
        std::array<index_type, ndim - sizeof...(Ts)> extents = detail::last<ndim - sizeof...(Ts)>(m_extents);
        return MDSpan<element_type, ndim - sizeof...(Ts)>(m_data + ind*detail::prod(extents), extents);
    }

    template <typename T>
        requires (ndim == 1UL)
    [[nodiscard]] constexpr element_type& operator[](T i) const noexcept
    {
        return (*this)(i);
    }

    template <typename T>
    [[nodiscard]] constexpr MDSpan<element_type, ndim - 1UL>
    operator[](T i) const noexcept
    {
        return (*this)(i);
    }

protected:
    friend MDSpan<std::remove_const_t<element_type>, ndim>;

    constexpr MDSpan(
        data_handle_type data, size_type size, const std::array<std::size_t, ndim>& extents) noexcept:
        m_data(data), m_size(size), m_extents(extents) {}

private:
    template <typename... Ts>
    [[nodiscard]] constexpr index_type
    idx(index_type ind, Ts... inds) const noexcept
    {
        return idx_impl<1>(ind, inds...);
    }

    template <std::size_t N, typename... Ts>
    [[nodiscard]] constexpr index_type
    idx_impl(index_type ind, index_type next, Ts... inds) const noexcept
    {
        if constexpr (N < ndim)
            return idx_impl<N + 1>(ind*m_extents[N] + next, inds...);
        else
            return ind;
    }

    template <std::size_t N>
    [[nodiscard]] constexpr index_type
    idx_impl(index_type ind) const noexcept
    {
        return ind;
    }
    
    data_handle_type m_data{};
    size_type m_size{};
    std::array<size_type, ndim> m_extents{};
};

} // namespace zest