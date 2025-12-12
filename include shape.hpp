#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <tuple>

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
    using subshape_type = NullShape;

    static constexpr size_type rank = 0;

    [[nodiscard]] static constexpr size_type size([[maybe_unused]] extent_type p) noexcept { return 0; }
    [[nodiscard]] constexpr size_type size() const noexcept { return 0; }
    [[nodiscard]] constexpr subshape_type subshape([[maybe_unused]] index_type i) const noexcept { return subshape_type{}; }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }
    [[nodiscard]] constexpr index_range indices() const noexcept { return index_range{}; }
};

template <typename LayoutType>
class SequencedShape
{
public:
    using layout = LayoutType;
    using size_type = typename layout::size_type;
    using index_type = typename layout::index_type;
    using index_range = typename layout::index_range;
    using extent_type = size_type;

    static constexpr size_type rank = layout::rank;

    SequencedShape(extent_type extent): m_extent(extent), m_size(layout::size(extent)) {}

    [[nodiscard]] static constexpr size_type size(extent_type extent) noexcept { return layout::size(extent); }
    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
    [[nodiscard]] constexpr extent_type extents() const noexcept { return m_extent; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr auto subshape(Inds... indices) const noexcept
    {
        return SequencedShape<typename layout::template sublayout_t<sizeof...(Inds)>>(layout::subextent(indices...));
    }

    template <typename... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr auto subshape([[maybe_unused]] Inds... indices) const noexcept
    {
        return NullShape{};
    }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr size_type operator()(Inds... indices) const noexcept { return layout::index(indices...); }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }

    [[nodiscard]] constexpr index_range indices() const noexcept { return layout::indices(m_extent); }
private:
    extent_type m_extent;
    size_type m_size;
};

template <std::size_t N>
class ArrayShape
{
public:
    using size_type = std::size_t;
    using index_type = size_type;
    using index_range = StandardIndexRange<size_type>;
    using extent_type = std::array<size_type, N>;
    using subshape_type = std::conditional<(N > 1), ArrayShape<N - 1>, NullShape>;

    static constexpr size_type rank = N;

    ArrayShape(extent_type extents): m_extents(extents), m_size(product(extents)) {}

    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
    [[nodiscard]] constexpr extent_type extents() const noexcept { return m_extents; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < rank)
    [[nodiscard]] constexpr subshape_type subshape([[maybe_unused]] Inds... inds) const noexcept { return subshape_type(subextents<sizeof...(Inds)>()); }

    template <typename... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr subshape_type subshape([[maybe_unused]] Inds... inds) const noexcept { return NullShape{}; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr size_type operator()(Inds... indices) const noexcept { return array::detail::index(m_extents, indices...); }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }
    [[nodiscard]] constexpr index_range indices() const noexcept { return index_range(m_extents[0]); }

private:

    extent_type m_extents;
    size_type m_size;
};

template <typename LayoutType, std::size_t N>
class ArraySequenceShape
{
public:
    using layout = LayoutType;
    using size_type = typename layout::size_type;
    using index_type = typename layout::index_type;
    using index_range = typename layout::index_range;
    using extent_type = typename std::pair<size_type, std::array<size_type, N>>;

    static constexpr size_type rank = layout::rank + N;

    ArraySequenceShape(size_type layout_extent, std::array<size_type, N> array_extents): m_extents(layout_extent, array_extents), m_size() {}

    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
    [[nodiscard]] constexpr extent_type extents() const noexcept { return m_extents; }

    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) < layout::rank)
    [[nodiscard]] constexpr auto subshape(Inds... indices) const noexcept
    {
        return ArraySequenceShape<typename layout::template sublayout_t<sizeof...(Inds)>, N>(layout::subextent(indices...), m_extents.second);
    }

    template <typename... Inds>
        requires (layout::rank <= sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr auto subshape([[maybe_unused]] Inds... indices)
    {
        return ArrayShape<N>(subarray_extents<sizeof...(Inds)>());
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
    [[nodiscard]] constexpr index_range indices() const noexcept { return index_range(m_extents[0]); }

private:
    template <typename... Inds>
        requires (1 <= sizeof...(Inds) && sizeof...(Inds) <= layout::rank)
    [[nodiscard]] constexpr size_type index(Inds... indices)
    {
        return layout::index(indices...)*m_array_size;
    }

    template <typename... Inds>
        requires (layout::rank < sizeof...(Inds) && sizeof...(Inds) <= rank)
    [[nodiscard]] constexpr size_type index(Inds... indices)
    {
        return split_index(std::make_index_sequence<layout::rank>{}, std::make_index_sequence<sizeof...(Inds) - layout::rank>{}, std::make_tuple(indices...));
    }

    template <std::size_t... I, std::size_t... J, typename T>
    [[nodiscard]] constexpr size_type split_index(std::index_sequence<I...>, std::index_sequence<J...>, T indices)
    {
        return layout::index(std::get<I>(indices)...)*m_array_size + array::detail::index(std::get<layout::rank + J>(indices)...);
    }

    extent_type m_extents;
    size_type m_array_size;
    size_type m_size;
};

} // namespace zest
