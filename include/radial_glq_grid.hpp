/*
Copyright (c) 2024-2026 Sebastian Sassi

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
#include <vector>

#include "alignment.hpp"
#include "gauss_legendre.hpp"
#include "shape.hpp"
#include "shaped_array.hpp"
#include "shaped_span.hpp"
#include "utility.hpp"

namespace zest::zt
{

template <typename AlignmentType = zest::CacheLineAlignment>
struct RadialGridLayout
{
    using Alignment = AlignmentType;

    /**
        @brief Number of grid points.

        @param order order of Zernike expansion
    */
    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        constexpr std::size_t vector_size
                = AlignmentType::template vector_size<double>();
        const std::size_t min_size = order;
        if constexpr (std::is_same_v<AlignmentType, zest::NoAlignment>)
            return min_size;
        else
            return zest::detail::next_divisible<vector_size>(min_size);
    }

    /**
        @brief Shape of the grid.

        @param order order of Zernike expansion
    */
    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    extents(std::size_t order) noexcept
    {
        return {size(order)};
    }
};

template <typename AlignmentType = CacheLineAlignment, std::size_t... inner_extent_params>
class RadialGLQGridShape:
    public TensorShape<std::dynamic_extent, inner_extent_params...>
{
private:
    using Base = TensorShape<std::dynamic_extent, inner_extent_params...>;

    template <std::size_t N, typename T>
    struct subshape_helper;

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) == Base::rank - N && 1 <= N && N < Base::rank)
    struct subshape_helper<N, std::index_sequence<Inds...>>
    {
        using type = TensorShape<std::get<N + Inds>(Base::static_extents)...>;
    };

    template <std::size_t N>
        requires (N == Base::rank)
    struct subshape_helper<N, std::index_sequence<>>
    {
        using type = NullShape;
    };

public:
    using layout_type = RadialGridLayout<AlignmentType>;
    using size_type = typename Base::size_type;
    using Base::extents;
    using Base::size;

    template <std::size_t N>
    using subshape_type = subshape_helper<N, std::make_index_sequence<Base::rank - N>>;

    using Alignment = AlignmentType;

    RadialGLQGridShape() = default;
    RadialGLQGridShape(size_type order) requires (Base::dynamic_rank == 1):
        Base{layout_type::extents(order)[0]}, m_order{order} {}

    RadialGLQGridShape(size_type order, size_type inner_extent)
        requires (Base::dynamic_rank == 2):
        Base{append(layout_type::extents(order), inner_extent)}, m_order{order} {}

    RadialGLQGridShape(
        size_type order, const std::array<size_type, Base::rank - 1>& inner_extents):
        Base{concatenate(layout_type::extents(order), inner_extents)}, m_order{order} {}

    RadialGLQGridShape(size_type order, const Base::extent_type& extents):
        Base{extents}, m_order{order} {}

    [[nodiscard]] constexpr size_type
    order() const noexcept { return m_order; }

    [[nodiscard]] static constexpr size_type
    size(size_type order) requires (Base::dynamic_rank == 1)
    {
        return Base::size(layout_type::extents(order));
    }

    [[nodiscard]] static constexpr size_type
    size(size_type order, size_type inner_extent) requires (Base::dynamic_rank == 2)
    {
        return Base::size(append(layout_type::extents(order), inner_extent));
    }

    [[nodiscard]] static constexpr size_type
    size(size_type order, const std::array<size_type, Base::dynamic_rank - 1>& inner_extents)
    {
        return Base::size(concatenate(layout_type::extents(order), inner_extents));
    }

    [[nodiscard]] static constexpr size_type
    size(size_type order, const std::array<size_type, Base::rank - 1>& inner_extents)
        requires (Base::dynamic_rank != Base::rank)
    {
        return Base::size(concatenate(layout_type::extents(order), inner_extents));
    }

    template <std::integral... Inds>
        requires (0 < sizeof...(Inds) && sizeof...(Inds) < Base::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(
                take_last<Base::rank - sizeof...(Inds)>(extents()));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == Base::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept { return zest::NullShape{}; }

private:

    size_type m_order{};
};

template <typename AlignmentType = CacheLineAlignment, std::size_t... outer_extent_params>
    requires (sizeof...(outer_extent_params) > 0)
class RadialGLQGridTensorShape:
    public zest::TensorShape<outer_extent_params..., std::dynamic_extent>
{
private:
    using Base = zest::TensorShape<outer_extent_params..., std::dynamic_extent>;

    template <std::size_t N, typename T>
    struct subshape_helper;

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) + 1 == Base::rank - N && 1 <= N && N < Base::rank - 1)
    struct subshape_helper<N, std::index_sequence<Inds...>>
    {
        using type = RadialGLQGridTensorShape<
                AlignmentType, std::get<Inds>(Base::static_extents)...
            >;
    };

    template <std::size_t N>
        requires (N == Base::rank - 1)
    struct subshape_helper<N, std::index_sequence<>>
    {
        using type = RadialGLQGridShape<AlignmentType>;
    };

    template <std::size_t N>
        requires (N == Base::rank)
    struct subshape_helper<N, std::index_sequence<>>
    {
        using type = zest::NullShape;
    };

public:
    using layout_type = RadialGridLayout<AlignmentType>;
    using size_type = typename Base::size_type;
    using Base::extents;
    using Base::size;

    template <std::size_t N>
        requires (0 < N && N <= Base::rank)
    using subshape_type = subshape_helper<
            N, std::make_index_sequence<Base::rank - std::min(N + 1, Base::rank)>
        >::type;

    RadialGLQGridTensorShape() = default;

    RadialGLQGridTensorShape(size_type outer_extent, size_type order)
        requires (Base::dynamic_rank == 2):
        Base(prepend(outer_extent, layout_type::extents(order))), m_order(order) {}

    RadialGLQGridTensorShape(
        const std::array<size_type, Base::dynamic_rank - 1>& outer_extents, size_type order):
        Base(concatenate(outer_extents, layout_type::extents(order))), m_order(order) {}

    RadialGLQGridTensorShape(
        const std::array<size_type, Base::rank - 1>& outer_extents, size_type order)
        requires (Base::dynamic_rank != Base::rank):
        Base(concatenate(outer_extents, layout_type::extents(order))), m_order(order) {}

    [[nodiscard]] constexpr size_type
    size(size_type outer_extent, size_type order) requires (Base::dynamic_rank == 2)
    {
        return Base::size(prepend(outer_extent, layout_type::extents(order)));
    }

    [[nodiscard]] constexpr size_type
    size(const std::array<size_type, Base::dynamic_rank - 1>& outer_extents, size_type order)
    {
        return Base::size(concatenate(outer_extents, layout_type::extents(order)));
    }

    [[nodiscard]] constexpr size_type
    size(const std::array<size_type, Base::rank - 1>& outer_extents, size_type order)
        requires (Base::dynamic_rank != Base::rank)
    {
        return Base::size(concatenate(outer_extents, layout_type::extents(order)));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) < Base::rank - 1)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(
                m_order, zest::take_last<Base::rank - sizeof...(Inds)>(extents()));
    }

    template <std::integral... Inds>
        requires (Base::rank - 1 <= sizeof...(Inds) && sizeof...(Inds) < Base::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(m_order);
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == Base::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept { return zest::NullShape{}; }

    [[nodiscard]] constexpr size_type
    order() const noexcept { return m_order; }

private:

    size_type m_order{};
};

template <typename AlignmentType>
using RadialGLQGridVectorShape
    = RadialGLQGridTensorShape<AlignmentType, std::dynamic_extent>;

/**
    @brief A non-owning view of a radial Gauss-Legendre quadrature grid.

    @tparam ElementType Type of elements in the grid.
    @tparam AlignmentType Byte alignment of the data.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, typename AlignmentType = CacheLineAlignment,
    std::size_t... inner_extents
>
using RadialGLQGridSpan
    = ShapedSpan<ElementType, RadialGLQGridShape<AlignmentType, inner_extents...>>;

/**
    @brief Container for radial Gauss-Legendre quadrature gridded data.

    @tparam ElementType Type of elements in the grid.
    @tparam AlignmentType Byte alignment of the data.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <
    typename ElementType, typename AlignmentType = CacheLineAlignment,
std::size_t... inner_extents
>
using RadialGLQGrid
    = ShapedArray<ElementType, RadialGLQGridShape<AlignmentType, inner_extents...>>;

/**
    @brief A non-owning view of a multidiemensional array of radial
    Gauss-Legendre quadrature grids.

    @tparam ElementType Type of elements in the grid.
    @tparam AlignmentType Byte alignment of the data.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    typename ElementType, typename AlignmentType = CacheLineAlignment,
    std::size_t... outer_extents
>
using RadialGLQGridTensorSpan
    = ShapedSpan<ElementType, RadialGLQGridTensorShape<AlignmentType, outer_extents...>>;

/**
    @brief A non-owning view of an array of radial Gauss-Legendre quadrature
    grids.

    @tparam ElementType Type of elements in the grid.
    @tparam AlignmentType Byte alignment of the data.
*/
template <typename ElementType, typename AlignmentType = CacheLineAlignment>
using SphereGLQGridVectorSpan
    = RadialGLQGridTensorSpan<ElementType, AlignmentType, std::dynamic_extent>;

/**
    @brief Container for a multidimensional array of radial Gauss-Legendre
    quadrature grids.

    @tparam ElementType Type of elements in the grid.
    @tparam AlignmentType Byte alignment of the data.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <
    typename ElementType, typename AlignmentType = CacheLineAlignment,
    std::size_t... outer_extents
>
using RadialGLQGridTensor
    = ShapedArray<ElementType, RadialGLQGridTensorShape<AlignmentType, outer_extents...>>;

/**
    @brief Container for an array of radial Gauss-Legendre quadrature grids.

    @tparam ElementType Type of elements in the grid.
    @tparam AlignmentType Byte alignment of the data.
*/
template <typename ElementType, typename AlignmentType = CacheLineAlignment>
using RadialGLQGridVector
    = RadialGLQGridTensorSpan<ElementType, AlignmentType, std::dynamic_extent>;

/**
    @brief Points defining a Gauss-Legendre quadrature grid on the sphere.

    @tparam LayoutType memory layout of the grid
*/
template <typename AlignmentType = CacheLineAlignment>
class RadialGLQGridPoints
{
public:
    using layout_type = RadialGridLayout<AlignmentType>;
    using alignment_type = AlignmentType;
    RadialGLQGridPoints() = default;
    explicit RadialGLQGridPoints(std::size_t order) { resize(order); }

    /**
        @brief Change the size of the corresponding grid.
    */
    void resize(std::size_t order)
    {
        resize_impl(layout_type::extents(order)[0]);
    }

    /**
        @brief Shape of the corresponding grid.
    */
    [[nodiscard]] std::array<std::size_t, 1> extents() noexcept
    {
        return {m_glq_nodes.size()};
    }

    /**
        @brief Radial Gauss-Legendre nodes.
    */
    [[nodiscard]] std::span<const double> glq_nodes() const noexcept
    {
        return m_glq_nodes;
    }

    /**
        @brief Generate Gauss-Legendre quadrature grid values from a function.

        @tparam FuncType type of function

        @param grid grid to place the values in
        @param f function to generate values
    */
    template <
        isotropic_function FuncType,
        contiguous_buffer_shaped_like<RadialGLQGridShape<layout_type>> GridType,
    >
        requires std::same_as<std::invoke_result_t<FuncType, double>, value_type_of<GridType>>
    void generate_values(GridType&& grid, FuncType&& f)
    {
        resize(grid.order());
        assert(grid.extent(0) == m_glq_nodes.size());

        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        {
            const double radius = m_glq_nodes[i];
            std::forward<GridType>(grid)[i] = std::forward<FuncType>(f)(radius);
        }
    }

    /**
        @brief Generate Gauss-Legendre quadrature grid values from a function.

        @tparam FuncType type of function

        @param f function to generate values
    */
    template <isotropic_function FuncType>
    auto generate_values(FuncType&& f, std::size_t order)
    {
        using ResultType = std::invoke_result_t<FuncType, double>;
        auto grid = RadialGLQGrid<ResultType, alignment_type>(order);
        generate_values(grid, std::forward<FuncType>(f));
        return grid;
    }

private:
    void resize_impl(std::size_t num_rad)
    {
        if (num_rad != m_glq_nodes.size())
        {
            m_glq_nodes.resize(num_rad);
            gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::cos>(
                    m_glq_nodes, m_glq_nodes.size() & 1);
            for (auto& node : m_glq_nodes)
                node = 0.5*(1 + node);
        }
    }

    std::vector<double> m_glq_nodes;
};

} // namespace zest::zt
