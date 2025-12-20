/*
Copyright (c) 2024, 2025 Sebastian Sassi

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
#include <cstddef>
#include <span>
#include <tuple>
#include <utility>

#include "indexing.hpp"

namespace zest
{

namespace detail
{

template <typename T, std::size_t N, std::size_t... I>
[[nodiscard]] constexpr std::array<T, sizeof...(I)>
take_last_impl(const std::array<T, N>& arr, std::index_sequence<I...>) noexcept
{
    return {std::get<N - sizeof...(I) + I>(arr)...};
}

template <typename T, std::size_t... I>
[[nodiscard]] constexpr auto
take_last_impl(const T& tuple, std::index_sequence<I...>) noexcept
{
    return std::make_tuple(std::get<std::tuple_size_v<T> - sizeof...(I) + I>(tuple)...);
}

template <typename T, std::size_t N>
    requires (std::tuple_size_v<T> <= N)
[[nodiscard]] constexpr auto
take_last(const T& tuple_like) noexcept
{
    return take_last_impl(tuple_like, std::make_index_sequence<N>());
}

template <typename T, std::size_t N, std::size_t... I>
[[nodiscard]] constexpr std::array<T, sizeof...(I)>
take_first_impl(const std::array<T, sizeof...(I)>& arr)
{
    return {std::get<I>(arr)...};
}

template <typename T, std::size_t... I>
[[nodiscard]] constexpr auto
take_first_impl(const T& tuple, std::index_sequence<I...>) noexcept
{
    return std::make_tuple(std::get<I>(tuple)...);
}

template <typename T, std::size_t N>
    requires (std::tuple_size_v<T> <= N)
[[nodiscard]] constexpr auto
take_first(const T& tuple_like) noexcept
{
    return take_first_impl(tuple_like, std::make_index_sequence<N>());
}

} // namespace detail

class NullShape
{
public:
    using size_type = std::size_t;
    using index_type = size_type;
    using index_range = SingleIndexRange<index_type>;
    using extent_type = size_type;

    template <std::size_t N>
    using subshape_type = NullShape;

    static constexpr size_type rank = 0;

    [[nodiscard]] static constexpr size_type size([[maybe_unused]] extent_type p) noexcept { return 0; }
    [[nodiscard]] constexpr size_type size() const noexcept { return 0; }
    [[nodiscard]] constexpr subshape_type subshape([[maybe_unused]] index_type i) const noexcept { return subshape_type{}; }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }
    [[nodiscard]] constexpr index_range indices() const noexcept { return index_range{}; }
};

template <typename SequenceType>
class SequencedShape
{
public:
    using sequence = SequenceType;
    using size_type = typename sequence::size_type;
    using index_type = typename sequence::index_type;
    using index_range = typename sequence::index_range;
    using extent_type = size_type;

    static constexpr size_type rank = sequence::rank;
    static constexpr std::size_t linear_extent = std::dynamic_extent;

    template <std::size_t N>
    using subshape_type = std::conditional_t<(N < rank),
        SequencedShape<typename sequence::template sublayout_t<N>>,
        NullShape>;

    constexpr SequencedShape() = default;
    explicit constexpr SequencedShape(extent_type extent): m_extent(extent), m_size(size(extent)) {}

    [[nodiscard]] static constexpr size_type size(extent_type extent) noexcept { return sequence::size(extent); }
    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
    [[nodiscard]] constexpr extent_type extents() const noexcept { return m_extent; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr auto subshape(Inds... indices) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(sequence::subextent(indices...));
    }

    template <typename... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto subshape([[maybe_unused]] Inds... indices) const noexcept
    {
        return NullShape{};
    }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr size_type operator()(Inds... indices) const noexcept { return sequence::index(indices...); }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }

    [[nodiscard]] constexpr index_range indices() const noexcept { return sequence::indices(m_extent); }
private:
    extent_type m_extent{};
    size_type m_size{};
};

template <std::size_t... Ns>
class TensorShape
{
public:
    using size_type = std::size_t;
    using index_type = size_type;
    using index_range = StandardIndexRange<size_type>;
    using extent_type = std::array<size_type, sizeof...(Ns)>;

    static constexpr size_type rank = sizeof...(Ns);
    static constexpr std::size_t linear_extent = std::dynamic_extent;

private:
    static constexpr std::array<std::size_t, sizeof...(Ns)> static_extents = {Ns...};

    template <typename T>
    struct subshape_helper;

    template <std::size_t... Inds>
    struct subshape_helper<std::index_sequence<Inds...>>
    {
        using type = TensorShape<std::get<Inds>(static_extents)...>;
    };

public:
    template <std::size_t N>
    using subshape_type = std::conditional_t<(N < rank),
        typename subshape_helper<std::make_index_sequence<N>>::type,
        NullShape>;

    constexpr TensorShape() = default;
    explicit constexpr TensorShape(extent_type extents): m_extents(extents), m_size(size(extents)) {}

    [[nodiscard]] static constexpr size_type size(extent_type extents) noexcept { return product(extents); }
    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
    [[nodiscard]] constexpr extent_type extents() const noexcept { return m_extents; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(take_last<rank - sizeof...(Inds)>(m_extents));
    }

    template <typename... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto subshape([[maybe_unused]] Inds... inds) const noexcept { return NullShape{}; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr size_type operator()(Inds... indices) const noexcept { return array::detail::index(m_extents, indices...); }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }
    [[nodiscard]] constexpr index_range indices() const noexcept { return index_range(m_extents[0]); }

private:
    template <std::size_t... I>
    [[nodiscard]] constexpr std::array<size_type, sizeof...(I)>
    subextents(std::index_sequence<I...>) const noexcept
    {
        return {std::get<rank - sizeof...(I) + I>(m_extents)...};
    }

    extent_type m_extents{};
    size_type m_size{};
};

namespace detail
{

template <typename T>
struct DynamicTensorShapeHelper;

template <std::size_t... Ns>
struct DynamicTensorShapeHelper<std::index_sequence<Ns...>>
{
    using type = TensorShape<(std::dynamic_extent + 0*Ns)...>;
};

} // namespace detail

template <std::size_t N>
using DynamicTensorShape = detail::DynamicTensorShapeHelper<std::make_index_sequence<N>>::type;

template <std::size_t... Ns>
    requires ((Ns != std::dynamic_extent) && ...)
class TensorShape<Ns...>
{
public:
    using size_type = std::size_t;
    using index_type = size_type;
    using index_range = StandardIndexRange<size_type>;
    using extent_type = std::array<size_type, sizeof...(Ns)>;

    static constexpr size_type rank = sizeof...(Ns);

    template <std::size_t N>
    using subshape_type = std::conditional_t<(N < rank), TensorShape</* IMPLEMENTATION */>, NullShape>;

private:
    static constexpr extent_type s_extents = std::array<size_type, rank>{Ns...};

public:
    static constexpr std::size_t linear_extent = product(std::array<size_type, rank>{Ns...});

    constexpr TensorShape() = default;
    explicit constexpr TensorShape([[maybe_unused]] extent_type extents) {}

    [[nodiscard]] static constexpr size_type size([[maybe_unused]] extent_type extents) noexcept { return linear_extent; }
    [[nodiscard]] constexpr size_type size() const noexcept { return linear_extent; }
    [[nodiscard]] constexpr extent_type extents() const noexcept { return s_extents; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr auto subshape([[maybe_unused]] Inds... inds) const noexcept { return subshape_impl(std::make_index_sequence<rank - sizeof...(Inds)>()); }

    template <typename... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto subshape([[maybe_unused]] Inds... inds) const noexcept { return NullShape{}; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr size_type operator()(Inds... indices) const noexcept { return array::detail::index(s_extents, indices...); }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }
    [[nodiscard]] constexpr index_range indices() const noexcept { return index_range(s_extents[0]); }

private:
    template <std::size_t... I>
    [[nodiscard]] static consteval auto subshape_impl(std::index_sequence<I...>) noexcept
    {
        return TensorShape<std::get<rank - sizeof...(I) + I>(s_extents)...>{};
    }
};

template <typename S1, typename S2>
class Combine
{
public:
    using size_type = typename S1::size_type;
    using index_type = typename S1::index_type;
    using index_range = typename S1::index_range;
    using extent_type = typename std::tuple<typename S1::extent_type, typename S2::extent_type>;

    static constexpr size_type rank = S1::rank + S2::rank;
    static constexpr std::size_t linear_extent = (S1::linear_extent == std::dynamic_extent || S2::linear_extent == std::dynamic_extent) ?
        std::dynamic_extent : S1::linear_extent*S2::linear_extent;

    constexpr Combine() = default;
    constexpr Combine(S1::extent_type first_extents, S2::extent_type second_extents): m_shapes(first_extents, second_extents), m_size(size(first_extents, second_extents)) {}

    [[nodiscard]] static constexpr size_type size(S1::extent_type first_extents, S2::extent_type second_extents) noexcept { return S1::size(first_extents)*S2::size(second_extents); }
    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
    [[nodiscard]] constexpr extent_type extents() const noexcept { return {m_shapes.extents(), m_shapes.extents()}; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < S1::rank)
    [[nodiscard]] constexpr auto subshape(Inds... indices) const noexcept
    {
        Combine<typename S1::template subshape_t<sizeof...(Inds)>, S2>(m_shapes.first.subextents(indices...), m_shapes.second.extents());
    }

    template <typename... Inds>
        requires (sizeof...(Inds) == S1::rank)
    [[nodiscard]] constexpr auto subshape(Inds... indices) const noexcept
    {
        return S2(m_shapes.second.extents());
    }

    template <typename... Inds>
        requires (S1::rank <= sizeof...(Inds) && sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto subshape(Inds... indices) const noexcept
    {
        using subshape_type = typename S2::template subshape_t<sizeof...(Inds) - S1::rank>;
        return subshape_type(m_shapes.second.subextents(indices...));
    }

    template <typename... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto subshape([[maybe_unused]] Inds... indices) const noexcept
    {
        return NullShape{};
    }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr size_type operator()(Inds... indices) const noexcept { return index(indices...); }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }
    [[nodiscard]] constexpr index_range indices() const noexcept { return m_shapes.first.indices(); }

private:
    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= S1::rank)
    [[nodiscard]] constexpr size_type index(Inds... indices)
    {
        return m_shapes.first.index(indices...)*m_shapes.second.size();
    }

    template <typename... Inds>
        requires (S1::rank < sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr size_type index(Inds... indices)
    {
        return split_index(std::make_index_sequence<S1::rank>{}, std::make_index_sequence<sizeof...(Inds) - S1::rank>{}, std::make_tuple(indices...));
    }

    template <std::size_t... I, std::size_t... J, typename T>
    [[nodiscard]] constexpr size_type split_index(std::index_sequence<I...>, std::index_sequence<J...>, T indices)
    {
        return S1::index(std::get<I>(indices)...)*m_shapes.second.size() + array::detail::index(std::get<S1::rank + J>(indices)...);
    }

    std::pair<S1, S2> m_shapes{};
    size_type m_size{};
};

template <typename SequenceType, std::size_t... Ns>
using TensorSequenceShape = Combine<SequencedShape<SequenceType>, TensorShape<Ns...>>;

} // namespace zest
