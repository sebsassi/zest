#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace zest
{

template <typename T>
concept shape = requires (T s, typename T::extent_type p) {
    { T::size(p) } -> std::same_as<typename T::size_type>;
    { s.size(p) } -> std::same_as<typename T::size_type>;
    { s.indices() } -> std::same_as<typename T::index_range>;
    { s.extents() } -> std::same_as<typename T::extent_type>;
};

template <typename ElementType, typename ShapeType>
class View
{
public:
    using value_type = ElementType;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using shape_type = ShapeType;
    using index_type = shape_type::index_type;
    using index_range = ShapeType::index_range;
    using const_view = View<const value_type, ShapeType>;

    constexpr View(pointer data, const shape_type::extent_type& extents): m_data(data), m_shape(extents) {}
    constexpr View(pointer data, const shape_type& shape): m_data(data), m_shape(shape) {}

    [[nodiscard]] constexpr operator const_view() const noexcept { return const_view(m_data, m_shape); }

    [[nodiscard]] constexpr ShapeType shape() const noexcept { return m_shape; }

    [[nodiscard]] constexpr ShapeType::extent_type extents() const noexcept { return m_shape.extents(); }

    [[nodiscard]] constexpr size_type size() const noexcept { return m_shape.size(); }

    [[nodiscard]] constexpr pointer data() const noexcept { return m_data; }

    [[nodiscard]] constexpr std::span<value_type> flatten() const noexcept { return std::span(m_data, m_shape.size()); }

    [[nodiscard]] constexpr index_range indices() const noexcept { return m_shape.indices(); }

    template <typename... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] constexpr reference operator()(Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    template <typename... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] constexpr reference operator[](Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    [[nodiscard]] constexpr auto operator()(index_type index) const noexcept requires (shape_type::rank > 1)
    {
        const auto subshape = m_shape.subshape(index);
        return View<value_type, decltype(subshape)>(m_data + m_shape(index), m_shape.subshape(index));
    }

    [[nodiscard]] constexpr auto operator[](index_type index) const noexcept requires (shape_type::rank > 1)
    {
        const auto subshape = m_shape.subshape(index);
        return View<value_type, decltype(subshape)>(m_data + m_shape(index), m_shape.subshape(index));
    }

private:
    pointer m_data;
    shape_type m_shape;
};

template <typename ElementType, typename ShapeType>
class Buffer
{
public:
    using value_type = ElementType;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using shape_type = ShapeType;
    using index_type = shape_type::index_type;
    using index_range = ShapeType::index_range;
    using view = View<value_type, shape_type>;
    using const_view = View<const value_type, shape_type>;
    using subview = View<value_type, typename shape_type::subshape_type>;
    using const_subview = View<const value_type, typename shape_type::subshape_type>;

    explicit Buffer(shape_type::extent_type shape_parameter): m_data(shape_type::size(shape_parameter)), m_shape(shape_parameter) {}

    [[nodiscard]] operator view() noexcept { return view(m_data.data(), m_data.size(), m_shape); }
    [[nodiscard]] operator const_view() const noexcept { return view(m_data.data(), m_data.size(), m_shape); }

    [[nodiscard]] ShapeType shape() const noexcept { return m_shape; }

    [[nodiscard]] ShapeType::extent_type extents() const noexcept { return m_shape.extents(); }

    [[nodiscard]] size_type size() const noexcept { return m_data.size(); }

    [[nodiscard]] const_pointer data() const noexcept { return m_data.data(); }
    [[nodiscard]] pointer data() noexcept { return m_data.data(); }

    [[nodiscard]] std::span<const value_type> flatten() const noexcept { return std::span(m_data); }
    [[nodiscard]] std::span<value_type> flatten() noexcept { return std::span(m_data); }

    [[nodiscard]] index_range indices() const noexcept { return m_shape.indices(); }

    template <typename... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] const_reference operator()(Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    template <typename... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] reference operator()(Inds... indices) noexcept { return m_data[m_shape(indices...)]; }

    template <typename... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] const_reference operator[](Inds... indices) const noexcept { return m_data[m_shape(indices...)]; }

    template <typename... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] reference operator[](Inds... indices) noexcept { return m_data[m_shape(indices...)]; }

    template <typename... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator()(Inds... indices) const noexcept
    {
        const auto subshape = m_shape.subshape(indices...);
        return View<const value_type, decltype(subshape)>(m_data.data() + m_shape(indices...), subshape);
    }

    template <typename... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] auto operator()(Inds... indices) noexcept
    {
        const auto subshape = m_shape.subshape(indices...);
        return View<value_type, decltype(subshape)>(m_data.data() + m_shape(indices...), subshape);
    }

    template <typename... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] const_subview operator[](Inds... indices) const noexcept
    {
        const auto subshape = m_shape.subshape(indices...);
        return View<const value_type, decltype(subshape)>(m_data.data() + m_shape(indices...), subshape);
    }

    template <typename... Inds>
        requires (sizeof...(Inds) < shape_type::rank)
    [[nodiscard]] subview operator[](Inds... indices) noexcept
    {
        const auto subshape = m_shape.subshape(indices...);
        return View<value_type, decltype(subshape)>(m_data.data() + m_shape(indices...), subshape);
    }

private:
    std::vector<ElementType> m_data;
    shape_type m_shape;
};

} // namespace zest
