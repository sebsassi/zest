#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace zest
{

template <typename T>
concept shape = requires (T s, typename T::parameter_type p, typename T::index_type i) {
    { T::size(p) } -> std::same_as<typename T::size_type>;
    { s.size(p) } -> std::same_as<typename T::size_type>;
    { s.subshape(i) } -> std::same_as<typename T::subshape>;
    { s(i) } -> std::same_as<typename T::size_type>;
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
    using const_view = View<const value_type, ShapeType>;
    using subview = View<value_type, typename shape_type::subshape>;
    using const_subview = View<const value_type, typename shape_type::subshape>;

    constexpr View(pointer data, const shape_type::parameter_type& shape_parameter): m_data(data), m_size(shape_type::size(shape_parameter)), m_shape(shape_parameter) {}
    constexpr View(pointer data, const shape_type& shape): m_data(data), m_size(shape.size()), m_shape(shape) {}
    constexpr View(pointer data, size_type size, const shape_type& shape): m_data(data), m_size(size), m_shape(shape) {}

    [[nodiscard]] constexpr operator const_view() const noexcept { return const_view(m_data, m_size, m_shape); }

    [[nodiscard]] constexpr ShapeType shape() const noexcept { return m_shape; }

    [[nodiscard]] constexpr ShapeType::parameter_type shape_parameter() const noexcept { return m_shape.parameter(); }

    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }

    [[nodiscard]] constexpr const_pointer data() const noexcept { return m_data; }
    [[nodiscard]] constexpr pointer data() noexcept { return m_data; }

    [[nodiscard]] constexpr std::span<const value_type> flatten() const noexcept { return std::span(m_data); }
    [[nodiscard]] constexpr std::span<value_type> flatten() noexcept { return std::span(m_data); }

    template <typename... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] constexpr reference operator()(Inds... indices) noexcept { return m_data[m_shape(indices...)]; }

    template <typename... Inds>
        requires (sizeof...(Inds) == shape_type::rank)
    [[nodiscard]] constexpr reference operator[](Inds... indices) noexcept { return m_data[m_shape(indices...)]; }

    [[nodiscard]] subview operator()(index_type index) noexcept
    {
        const auto subshape = m_shape.subshape(index);
        return subview(m_data + m_shape(index), m_shape.subshape(index));
    }

    [[nodiscard]] subview operator[](index_type index) noexcept
    {
        const auto subshape = m_shape.subshape(index);
        return subview(m_data + m_shape(index), m_shape.subshape(index));
    }

private:
    pointer m_data;
    size_type m_size;
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
    using view = View<value_type, shape_type>;
    using const_view = View<const value_type, shape_type>;
    using subview = View<value_type, typename shape_type::subshape>;
    using const_subview = View<const value_type, typename shape_type::subshape>;

    explicit Buffer(shape_type::parameter_type shape_parameter): m_data(shape_type::size(shape_parameter)), m_shape(shape_parameter) {}

    [[nodiscard]] operator view() noexcept { return view(m_data.data(), m_data.size(), m_shape); }
    [[nodiscard]] operator const_view() const noexcept { return view(m_data.data(), m_data.size(), m_shape); }

    [[nodiscard]] ShapeType shape() const noexcept { return m_shape; }

    [[nodiscard]] ShapeType::parameter_type shape_parameter() const noexcept { return m_shape.parameter; }

    [[nodiscard]] size_type size() const noexcept { return m_data.size(); }

    [[nodiscard]] const_pointer data() const noexcept { return m_data.data(); }
    [[nodiscard]] pointer data() noexcept { return m_data.data(); }

    [[nodiscard]] std::span<const value_type> flatten() const noexcept { return std::span(m_data); }
    [[nodiscard]] std::span<value_type> flatten() noexcept { return std::span(m_data); }

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

    [[nodiscard]] const_subview operator()(index_type index) const noexcept
    {
        const auto subshape = m_shape.subshape(index);
        return const_subview(m_data.data() + m_shape(index), m_shape.subshape(index));
    }

    [[nodiscard]] subview operator()(index_type index) noexcept
    {
        const auto subshape = m_shape.subshape(index);
        return subview(m_data.data() + m_shape(index), m_shape.subshape(index));
    }

    [[nodiscard]] const_subview operator[](index_type index) const noexcept
    {
        const auto subshape = m_shape.subshape(index);
        return const_subview(m_data.data() + m_shape(index), m_shape.subshape(index));
    }

    [[nodiscard]] subview operator[](index_type index) noexcept
    {
        const auto subshape = m_shape.subshape(index);
        return subview(m_data.data() + m_shape(index), m_shape.subshape(index));
    }

private:
    std::vector<ElementType> m_data;
    shape_type m_shape;
};

} // namespace zest
