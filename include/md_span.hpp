#pragma once

#include <cstddef>
#include <span>
#include <array>

namespace zest
{

namespace detail
{

template <std::size_t M, typename T>
auto last(T a)
{
    std::array<typename std::remove_cvref<T>::type::value_type, M> res{};
    for (std::size_t i = 0; i < M; ++i)
        res[i] = a[(a.size() - M) + i];
    return res;
}

template <typename T>
auto prod(T a)
{
    auto res = a[0];
    for (std::size_t i = 1; i < a.size(); ++i)
        res *= a[i];
    return res;
}

} // namespace detail

/**
    @brief Poor man's mdspan for modeling dynamic multidimensional arrays.

    @tparam ElementType type of array elements
    @tparam NDIM number of array dimensions
*/
template <typename ElementType, std::size_t NDIM>
class MDSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using index_type = std::size_t;
    using data_handle_type = element_type*;

    constexpr MDSpan() = default;
    constexpr MDSpan(
        data_handle_type data, const std::array<std::size_t, NDIM>& extents):
        m_data(data), m_size(detail::prod(extents)), m_extents(extents) {}

    [[nodiscard]] operator std::span<element_type>()
    {
        return std::span<element_type>(m_data, m_size);
    }
    
    [[nodiscard]] constexpr data_handle_type data() const noexcept
    {
        return m_data;
    }
    
    [[nodiscard]] constexpr size_type size() const noexcept
    {
        return m_size;
    }

    [[nodiscard]] constexpr bool empty() const noexcept
    {
        return m_size == 0;
    }

    [[nodiscard]] constexpr const std::array<std::size_t, NDIM>&
    extents() const noexcept
    {
        return m_extents;
    }

    template <typename... Ts>
        requires (sizeof...(Ts) == NDIM)
    [[nodiscard]] constexpr element_type& operator()(Ts... inds) const noexcept
    {
        return m_data[idx(inds...)];
    }

    template <typename... Ts>
        requires (sizeof...(Ts) < NDIM)
    [[nodiscard]] constexpr MDSpan<element_type, NDIM - sizeof...(Ts)>
    operator()(Ts... inds) const noexcept
    {
        index_type ind = idx(inds...);
        std::array<index_type, NDIM - sizeof...(Ts)> extents = detail::last<NDIM - sizeof...(Ts)>(m_extents);
        return MDSpan<element_type, NDIM - sizeof...(Ts)>(m_data + ind*detail::prod(extents), extents);
    }

    template <typename T>
        requires (NDIM == 1UL)
    [[nodiscard]] constexpr element_type& operator[](T i)
    {
        return (*this)(i);
    }

    template <typename T>
    [[nodiscard]] constexpr MDSpan<element_type, NDIM - 1UL> operator[](T i)
    {
        return (*this)(i);
    }

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
        if constexpr (N < NDIM)
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
    
    data_handle_type m_data;
    size_type m_size;
    std::array<size_type, NDIM> m_extents;
};

} // namespace zest