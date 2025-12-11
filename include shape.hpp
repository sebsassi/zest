#pragma once

#include <cstddef>
#include <array>

#include "indexing.hpp"

namespace zest
{

class NullShape
{
public:
    using size_type = std::size_t;
    using index_type = std::size_t;
    using index_range = SingleIndexRange<index_type>;
    using extent_type = std::size_t;
    using subshape_type = NullShape;

    static constexpr size_type rank = 0;

    [[nodiscard]] static constexpr size_type size(extent_type p) noexcept { return 0; }
    [[nodiscard]] constexpr size_type size() const noexcept { return 0; }
    [[nodiscard]] constexpr subshape_type subshape(index_type i) const noexcept { return subshape_type{}; }
    [[nodiscard]] constexpr size_type operator()() const noexcept { return 0; }
    [[nodiscard]] constexpr index_range indices() const noexcept { return index_range{}; }
};

template <typename LayoutType>
class SimpleShape
{
public:
    using layout = LayoutType;
    using size_type = typename layout::size_type;
    using index_type = typename elayout::index_type;
    using index_range = typename layout::index_range;
    using extent_type = typename layout::extent_type;
    using subshape_type = SimpleShape<layout::sublayout>;

    static constexpr size_type rank = layout::rank;

    SimpleShape(extent_type extents): m_extents(extents), m_size(size(extents)) {}

    [[nodiscard]] static constexpr size_type size(extent_type extents) noexcept { return layout::size(extents); }
    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
    [[nodiscard]] constexpr subshape_type subshape(index_type i) const noexcept { return subshape_type(layout::sub_extents(i, m_extents)); }

    template <typename... Inds>
        requires (sizeof...(Inds) == rank)
    [[nodiscard]] constexpr size_type operator()(Inds... indices) const noexcept { return layout::index(indices...); }

    [[nodiscard]] constexpr index_range indices() const noexcept { return layout::indices(m_extents); }
private:
    extent_type m_extents;
    size_type m_size;
};

template <typename LayoutType, typename... LayoutTypes>
    requires (LayoutType::rank > 1)
class CompundShape
{
public:
    using layout = LayoutType;
    using size_type = typename layout::size_type;
    using index_type = typename layout::index_type;
    using index_range = typename layout::index_range;
    using extent_type = typename layout::extent_type;
    using subshape_type = CompoundShape<layout::sublayout, LayoutTypes>;

    static constexpr size_type rank = layout::rank + (LayoutTypes::rank + ...);

    SimpleShape(extent_type extent, LayoutTypes::extent_type... extents): m_extent(extent), m_extents(extents...), m_strides() {}

    [[nodiscard]] static constexpr size_type size(extent_type extent, LayoutTypes::extent_type... extents) noexcept { return layout::size(extents)*(LayoutTypes::size(extents) * ...); }

    [[nodiscard]] static constexpr std::array<size_type, sizeof...(LayoutTypes) + 1> strides(extent_type extent, LayoutTypes::extent_type... extents) noexcept
    {
        std::array<size_type, sizeof...(LayoutTypes) + 1> sizes = {layout::size(extent), LayoutTypes::size(extents)...};
        return {layout::size(extent), LayoutTypes::size(extents)...};
    }

    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
    [[nodiscard]] constexpr subshape_type subshape(index_type i) const noexcept { return subshape_type(layout::sub_extents(i, m_extent), m_extents); }

    template <typename... MainInds, typename... OtherInds>
        requires (sizeof...(MainInds) == layout::rank) && (sizeof...(MainInds) + sizeof...(OtherInds) == rank)
    [[nodiscard]] constexpr size_type operator()(MainInds... layout_indices, OtherInds... other_indices) const noexcept { return layout::index(layout_indices); }

    [[nodiscard]] constexpr index_range indices() const noexcept { return layout::indices(m_extent); }

private:
    extent_type m_extent;
    std::tuple<LayoutTypes::extent_type...> m_extents;
    std::array<size_type, sizeof...(LayoutTypes) + 1> m_strides;
    size_type m_size;
}

} // namespace zest
