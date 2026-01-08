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
#include <cassert>
#include <cstddef>
#include <span>
#include <tuple>
#include <utility>

#include "utility.hpp"
#include "indexing.hpp"

namespace zest
{

class NullShape
{
public:
    using size_type = std::size_t;
    using index_type = size_type;
    using index_range = SingleIndexRange<index_type>;
    using extent_type = size_type;

    static constexpr size_type rank = 0;
    static constexpr std::size_t linear_extent = 0;

    template <std::size_t N>
        requires (N > 0)
    using subshape_type = NullShape;

    constexpr NullShape() = default;
    explicit constexpr NullShape(extent_type /*unused*/) {}

    [[nodiscard]] static constexpr size_type
    size([[maybe_unused]] extent_type /*unused*/) noexcept { return 0; }

    [[nodiscard]] constexpr size_type
    size() const noexcept { return 0; }

    [[nodiscard]] constexpr extent_type
    extents() const noexcept { return 0; }

    [[nodiscard]] constexpr index_range
    indices() const noexcept { return index_range{}; }
};

template <typename T>
concept shape = requires (T x, typename T::extent_type ex, typename T::index_type ind)
    {
        { T::size(ex) } -> std::same_as<typename T::size_type>;
        { x.size() } -> std::same_as<typename T::size_type>;
        { x.extents() } -> std::same_as<typename T::extent_type>;
        { x.indices() } -> std::same_as<typename T::index_range>;
        { x.indices(ind) } -> std::same_as<typename T::index_range>;
    }
    && std::same_as<decltype(T::rank), typename T::size_type>
    && std::same_as<decltype(T::linear_extent), typename T::size_type>
    && std::same_as<typename T::template subshape_type<T::rank>, NullShape>;

template <typename SequenceType>
class SequencedShape
{
public:
    using sequence_type = SequenceType;
    using size_type = typename sequence_type::size_type;
    using index_type = typename sequence_type::index_type;
    using index_range = typename sequence_type::index_range;
    using extent_type = size_type;

    static constexpr size_type rank = sequence_type::rank;
    static constexpr std::size_t linear_extent = std::dynamic_extent;

private:
    template <std::size_t N> struct subshape_helper;

    template <std::size_t N>
        requires (0 < N && N < rank)
    struct subshape_helper<N>
    {
        using type = SequencedShape<typename sequence_type::template subsequence_type<N>>;
    };

    template <std::size_t N>
        requires (N == rank)
    struct subshape_helper<N>
    {
        using type = NullShape;
    };

public:
    template <std::size_t N>
        requires (0 < N && N <= rank)
    using subshape_type = subshape_helper<N>::type;

    constexpr SequencedShape() = default;
    explicit constexpr SequencedShape(extent_type order):
        m_order(order), m_size(size(order)) {}

    [[nodiscard]] static constexpr size_type
    size(extent_type order) noexcept { return sequence_type::size(order); }

    [[nodiscard]] constexpr size_type
    size() const noexcept { return m_size; }

    [[nodiscard]] constexpr extent_type
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr extent_type
    extents() const noexcept { return m_order; }

    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr auto
    subshape(Inds... indices) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(sequence_type::subextent(index_type(indices)...));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... indices) const noexcept
    {
        return NullShape{};
    }

    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr index_type
    operator()(Inds... indices) const noexcept { return sequence_type::index(index_type(indices)...); }

    [[nodiscard]] constexpr index_range
    indices() const noexcept { return index_range(m_order); }

    [[nodiscard]] constexpr index_range
    indices(index_type index) const noexcept
    {
        assert(index < order());
        return index_range(index, index_type(m_order));
    }

private:
    extent_type m_order{};
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
    static constexpr std::array<std::size_t, sizeof...(Ns)> static_extents = {Ns...};

private:
    template <std::size_t N, typename T>
    struct subshape_helper;

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) == rank - N && 1 <= N && N < rank)
    struct subshape_helper<N, std::index_sequence<Inds...>>
    {
        using type = TensorShape<std::get<N + Inds>(static_extents)...>;
    };

    template <std::size_t N>
        requires (N == rank)
    struct subshape_helper<N, std::index_sequence<>>
    {
        using type = NullShape;
    };

public:
    template <std::size_t N>
        requires (0 < N && N <= rank)
    using subshape_type = subshape_helper<N, std::make_index_sequence<rank - N>>::type;

    constexpr TensorShape() = default;

    explicit constexpr TensorShape(size_type extents) requires (rank == 1):
        m_extents({extents}), m_size(size({extents})) {}

    explicit constexpr TensorShape(const extent_type& extents):
        m_extents(extents), m_size(size(extents)) {}

    [[nodiscard]] static constexpr size_type
    size(extent_type extents) noexcept { return product(extents); }

    [[nodiscard]] constexpr size_type
    size() const noexcept { return m_size; }

    [[nodiscard]] constexpr extent_type
    extents() const noexcept { return m_extents; }

    [[nodiscard]] constexpr size_type
    extent(size_type i) const noexcept { return m_extents[i]; }

    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(take_last<rank - sizeof...(Inds)>(m_extents));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept { return NullShape{}; }

    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr index_type
    operator()(Inds... indices) const noexcept
    {
        return array::detail::index(m_extents, index_type(indices)...);
    }

    [[nodiscard]] constexpr index_range
    indices() const noexcept { return index_range(m_extents[0]); }

    [[nodiscard]] constexpr index_range
    indices(index_type index) const noexcept
    {
        assert(index < extent(0));
        return index_range(m_extents[0]);
    }

private:
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
using DynamicTensorShape
    = detail::DynamicTensorShapeHelper<std::make_index_sequence<N>>::type;

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
    static constexpr std::size_t linear_extent = product(std::array{Ns...});
    static constexpr extent_type static_extents = std::array{Ns...};

private:
    template <std::size_t N, typename T>
    struct subshape_helper;

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) == rank - N && 1 <= N && N < rank)
    struct subshape_helper<N, std::index_sequence<Inds...>>
    {
        using type = TensorShape<std::get<N + Inds>(static_extents)...>;
    };

    template <std::size_t N>
        requires (N == rank)
    struct subshape_helper<N, std::index_sequence<>>
    {
        using type = NullShape;
    };

public:
    template <std::size_t N>
        requires (0 < N && N <= rank)
    using subshape_type = subshape_helper<N, std::make_index_sequence<rank - N>>::type;

    constexpr TensorShape() = default;
    explicit constexpr TensorShape([[maybe_unused]] size_type extent) requires (rank == 1) {}
    explicit constexpr TensorShape([[maybe_unused]] extent_type extents) {}

    [[nodiscard]] static constexpr size_type
    size([[maybe_unused]] extent_type extents) noexcept { return linear_extent; }

    [[nodiscard]] static constexpr size_type
    size() noexcept { return linear_extent; }

    [[nodiscard]] constexpr extent_type
    extents() const noexcept { return static_extents; }

    [[nodiscard]] constexpr size_type
    extent(size_type i) const noexcept { return static_extents[i]; }

    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>{};
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept { return NullShape{}; }

    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr index_type
    operator()(Inds... indices) const noexcept
    {
        return array::detail::index(static_extents, index_type(indices)...);
    }

    [[nodiscard]] constexpr index_range
    indices() const noexcept { return index_range(static_extents[0]); }

    [[nodiscard]] constexpr index_range
    indices(index_type index) const noexcept
    {
        assert(index < extent(0));
        return index_range(index, static_extents[0]);
    }
};

template <typename S1, typename S2>
class CompositeShape
{
public:
    using size_type = typename S1::size_type;
    using index_type = typename S1::index_type;
    using index_range = typename S1::index_range;
    using extent_type = std::tuple<typename S1::extent_type, typename S2::extent_type>;

    static constexpr size_type rank = S1::rank + S2::rank;
    static constexpr std::size_t linear_extent
        = (S1::linear_extent == std::dynamic_extent || S2::linear_extent == std::dynamic_extent) ?
            std::dynamic_extent : S1::linear_extent*S2::linear_extent;
private:
    template <std::size_t N>
    struct subshape_helper;

    template <std::size_t N>
        requires (N < S1::rank)
    struct subshape_helper<N>
    {
        using type = CompositeShape<typename S1::template subshape_type<N>, S2>;
    };

    template <std::size_t N>
        requires (N == S1::rank)
    struct subshape_helper<N>
    {
        using type = S2;
    };

    template <std::size_t N>
        requires (S1::rank < N && N < rank)
    struct subshape_helper<N>
    {
        using type = typename S2::template subshape_type<N - S1::rank>;
    };

    template <std::size_t N>
        requires (N == rank)
    struct subshape_helper<N>
    {
        using type = NullShape;
    };

public:
    template <std::size_t N>
        requires (N > 0)
    using subshape_type = subshape_helper<N>::type;

    constexpr CompositeShape() = default;

    template <typename E1, typename E2>
        requires std::constructible_from<S1, E1> && std::constructible_from<S2, E2>
    constexpr CompositeShape(const E1& first_extents, const E2& second_extents):
        m_shapes(S1{first_extents}, S2{second_extents}) {}

    template <typename E>
        requires std::constructible_from<S1, E>
    explicit constexpr CompositeShape(const E& first_extents)
        requires (S1::linear_extent == std::dynamic_extent
            && S2::linear_extent != std::dynamic_extent):
        m_shapes(S1{first_extents}, S2{}) {}

    template <typename E>
        requires std::constructible_from<S2, E>
    explicit constexpr CompositeShape(const E& second_extents)
        requires (S1::linear_extent != std::dynamic_extent
            && S2::linear_extent == std::dynamic_extent):
        m_shapes(S1{}, S2{second_extents}) {}

    constexpr CompositeShape(const S1& s1, const S2& s2): m_shapes(s1, s2) {}

    [[nodiscard]] static constexpr size_type
    size(const S1::extent_type& first_extents, const S2::extent_type& second_extents) noexcept
    {
        return S1::size(first_extents)*S2::size(second_extents);
    }

    [[nodiscard]] static constexpr size_type
    size(const S1::extent_type& first_extents)
        requires (S1::linear_extent == std::dynamic_extent && S2::linear_extent != std::dynamic_extent)
    {
        return S1::size(first_extents)*S2::linear_extent;
    }

    [[nodiscard]] static constexpr size_type
    size(const S2::extent_type& second_extents)
        requires (S1::linear_extent != std::dynamic_extent && S2::linear_extent == std::dynamic_extent)
    {
        return S1::linear_extent*S2::size(second_extents);
    }

    [[nodiscard]] static constexpr size_type
    size() noexcept requires (linear_extent != std::dynamic_extent) { return linear_extent; }

    [[nodiscard]] constexpr size_type
    size() const noexcept requires (linear_extent == std::dynamic_extent)
    {
        return m_shapes.first.size()*m_shapes.second.size();
    }

    [[nodiscard]] constexpr S1::extent_type
    order() const noexcept requires sequence_shaped<S1>
    {
        return m_shapes.first.extents();
    }

    [[nodiscard]] constexpr extent_type
    extents() const noexcept { return {m_shapes.first.extents(), m_shapes.second.extents()}; }

    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < S1::rank)
    [[nodiscard]] constexpr auto
    subshape(Inds... indices) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(
            m_shapes.first.subshape((typename S1::index_type)(indices)...), m_shapes.second);
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == S1::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... indices) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(m_shapes.second);
    }

    template <std::integral... Inds>
        requires (S1::rank < sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr auto
    subshape(Inds... indices) const noexcept
    {
        auto impl = [&]<std::size_t... I, typename T>(std::index_sequence<I...>, T index_tuple)
        {
            return subshape_type<sizeof...(Inds)>(m_shapes.second.subshape(std::get<S1::rank + I>(index_tuple)...));
        };
        return impl(
            std::make_index_sequence<sizeof...(Inds) - S1::rank>{},
            std::make_tuple(indices...));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... indices) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(NullShape{});
    }

    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr index_type
    operator()(Inds... indices) const noexcept { return index(index_type(indices)...); }

    [[nodiscard]] constexpr index_range
    indices() const noexcept { return m_shapes.first.indices(); }

    [[nodiscard]] constexpr index_range
    indices(index_type index) const noexcept { return m_shapes.first.indices(index); }

private:
    template <std::integral... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= S1::rank)
    [[nodiscard]] constexpr index_type
    index(Inds... indices) const noexcept
    {
        return m_shapes.first((typename S1::index_type)(indices)...)*m_shapes.second.size();
    }

    template <std::integral... Inds>
        requires (S1::rank < sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr index_type
    index(Inds... indices) const noexcept
    {
        auto impl = [&]<std::size_t... I, std::size_t... J, typename T>
        (std::index_sequence<I...>, std::index_sequence<J...>, T index_tuple)
        {
            return m_shapes.first((typename S1::index_type)(std::get<I>(index_tuple))...)*m_shapes.second.size()
                + m_shapes.second((typename S2::index_type)(std::get<S1::rank + J>(index_tuple))...);
        };
        return impl(
            std::make_index_sequence<S1::rank>{},
            std::make_index_sequence<sizeof...(Inds) - S1::rank>{},
            std::make_tuple(indices...));
    }

    std::pair<S1, S2> m_shapes{};
};

template <typename SequenceType, std::size_t... Ns>
using TensorSequenceShape = std::conditional_t<(sizeof...(Ns) > 0),
    CompositeShape<SequencedShape<SequenceType>, TensorShape<Ns...>>,
    SequencedShape<SequenceType>>;

template <typename SequenceType, std::size_t... Ns>
using SequenceTensorShape = std::conditional_t<(sizeof...(Ns) > 0),
    CompositeShape<TensorShape<Ns...>, SequencedShape<SequenceType>>,
    SequencedShape<SequenceType>>;

template <typename ShapeType, tag_type... Tags>
struct TaggedShape: public ShapeType, public Tags...
{
    using untag = ShapeType;

    template <std::size_t N>
    using subshape_type = TaggedShape<typename ShapeType::template subshape_type<N>, Tags...>;

    using ShapeType::ShapeType;

    explicit TaggedShape(const ShapeType& other): ShapeType(other) {}
};

} // namespace zest
