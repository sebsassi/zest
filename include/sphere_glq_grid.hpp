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
#include <span>

#include "alignment.hpp"
#include "gauss_legendre.hpp"
#include "shape.hpp"
#include "shaped_array.hpp"
#include "shaped_span.hpp"

namespace zest::st
{

/**
    @brief Longitudinally contiguous layout for storing a Gauss-Legendre
    quadrature grid.

    @tparam AlignmentType byte alignment of the grid
*/
template <typename AlignmentType = CacheLineAlignment>
struct LatLonLayout
{
    using Alignment = AlignmentType;

    /**
        @brief Number of grid points.

        @param order order of spherical harmonic expansion
    */
    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return lat_size(order)*lon_size(order);
    }

    /**
        @brief Shape of the grid.

        @param order order of spherical harmonic expansion
    */
    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    extents(std::size_t order) noexcept
    {
        return {lat_size(order), lon_size(order)};
    }

    /**
        @brief Number of longitudinal Fourier coefficients.
    */
    [[nodiscard]] static constexpr std::size_t
    fft_size(std::size_t order) noexcept
    {
        return (lon_size(order) >> 1) + 1;
    }

    /**
        @brief Stride of the longitudinally Fourier transformed grid.
    */
    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    fft_stride(std::size_t order) noexcept
    {
        return {fft_size(order), 1};
    }

    /**
        @brief Size in latitudinal direction.
    */
    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t order) noexcept
    {
        return order;
    }

    /**
        @brief Size in latitudinal direction.
    */
    [[nodiscard]] static constexpr std::size_t
    lon_size(std::size_t order) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = 2UL*order - std::min(1UL, order);
        if constexpr (std::is_same_v<Alignment, NoAlignment>)
            return min_size;
        else
            return zest::detail::next_divisible<vector_size>(min_size);
    }

    static constexpr std::size_t lat_axis = 0UL;
    static constexpr std::size_t lon_axis = 1UL;
};

/**
    @brief Latitudinally contiguous layout for storing a Gauss-Legendre
    quadrature grid.

    @tparam AlignmentType byte alignment of the grid
*/
template <typename AlignmentType = CacheLineAlignment>
struct LonLatLayout
{
    using Alignment = AlignmentType;

    /**
        @brief Number of grid points.

        @param order order of spherical harmonic expansion
    */
    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return lat_size(order)*lon_size(order);
    }

    /**
        @brief Shape of the grid.

        @param order order of spherical harmonic expansion
    */
    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    extents(std::size_t order) noexcept
    {
        return {lon_size(order), lat_size(order)};
    }

    /**
        @brief Number of longitudinal Fourier coefficients.
    */
    [[nodiscard]] static constexpr std::size_t
    fft_size(std::size_t order) noexcept
    {
        return (lon_size(order) >> 1) + 1;
    }

    /**
        @brief Stride of the longitudinally Fourier transformed grid.
    */
    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    fft_stride(std::size_t order)
    {
        return {lat_size(order), 1};
    }

    /**
        @brief Size in latitudinal direction.
    */
    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t order) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = order;
        if constexpr (std::is_same_v<Alignment, NoAlignment>)
            return min_size;
        else
            return zest::detail::next_divisible<vector_size>(min_size);
    }

    /**
        @brief Size in longitudinal direction.
    */
    [[nodiscard]] static constexpr std::size_t
    lon_size(std::size_t order) noexcept
    {
        return 2UL*order - std::min(1UL, order);
    }

    static constexpr std::size_t lat_axis = 1UL;
    static constexpr std::size_t lon_axis = 0UL;
};

using DefaultLayout = LonLatLayout<>;

template <typename LayoutType, std::size_t... inner_extent_params>
class SphereGLQSubGridShape:
    public TensorShape<std::dynamic_extent, inner_extent_params...>
{
private:
    using Base = TensorShape<std::dynamic_extent, inner_extent_params...>;

    template <std::size_t N, typename T>
    struct subshape_helper;

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) = Base::rank - N && 1 <= N && N < Base::rank)
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
    SphereGLQSubGridShape() = default;

    SphereGLQSubGridShape(Base::size_type order, Base::extent_type extents):
        Base{extents}, m_order{order} {}


    [[nodiscard]] constexpr Base::size_type
    order() const noexcept { return m_order; }

private:
    Base::size_type m_order{};
};

template <typename LayoutType, std::size_t... inner_extent_params>
class SphereGLQGridShape:
    public TensorShape<std::dynamic_extent, std::dynamic_extent, inner_extent_params...>
{
private:
    using Base = TensorShape<std::dynamic_extent, std::dynamic_extent, inner_extent_params...>;

    template <std::size_t N, typename T>
    struct subshape_helper;

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) == Base::rank - N && N == 1)
    struct subshape_helper<N, std::index_sequence<Inds...>>
    {
        using type = SphereGLQSubGridShape<LayoutType, std::get<N + Inds>(Base::static_extents)...>;
    };

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) = Base::rank - N && 1 < N && N < Base::rank)
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
    using size_type = typename Base::size_type;
    using Base::extents;

    template <std::size_t N>
        requires (0 < N && N <= Base::rank)
    using subshape_type = subshape_helper<N, std::make_index_sequence<Base::rank - N>>::type;

    SphereGLQGridShape() = default;

    SphereGLQGridShape(size_type order) requires (Base::dynamic_rank == 2):
        Base(LayoutType::extents(order)), m_order(order) {}

    SphereGLQGridShape(size_type order, size_type inner_extent)
        requires (Base::dynamic_rank == 3):
        Base(append(LayoutType::extents(order), inner_extent)), m_order(order) {}

    SphereGLQGridShape(
        size_type order, const std::array<size_type, Base::dynamic_rank - 2>& inner_extents):
        Base(concatenate(LayoutType::extents(order), inner_extents)), m_order(order) {}

    SphereGLQGridShape(
        size_type order, const std::array<size_type, Base::rank - 2>& inner_extents)
        requires (Base::dynamic_rank != Base::rank):
        Base(concatenate(LayoutType::extents(order), inner_extents)), m_order(order) {}

    template <std::integral... Inds>
        requires (sizeof...(Inds) == 1)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(m_order, take_last<Base::rank - sizeof...(Inds)>(extents()));
    }

    template <std::integral... Inds>
        requires (1 < sizeof...(Inds) && sizeof...(Inds) < Base::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(take_last<Base::rank - sizeof...(Inds)>(extents()));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == Base::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept { return NullShape{}; }

    [[nodiscard]] constexpr size_type
    order() const noexcept { return m_order; }

private:
    size_type m_order{};
};

template <typename LayoutType, std::size_t... outer_extent_params>
    requires (sizeof...(outer_extent_params) > 0)
class SphereGLQGridTensorShape:
    public TensorShape<outer_extent_params..., std::dynamic_extent, std::dynamic_extent>
{
private:
    using Base = TensorShape<outer_extent_params..., std::dynamic_extent, std::dynamic_extent>;

    template <std::size_t N, typename T>
    struct subshape_helper;

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) == Base::rank - N && 1 <= N && N < Base::rank - 2)
    struct subshape_helper<N, std::index_sequence<Inds...>>
    {
        using type = SphereGLQGridTensorShape<LayoutType, std::get<Inds>(Base::static_extents)...>;
    };

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) = Base::rank - N && N == Base::rank - 2)
    struct subshape_helper<N, std::index_sequence<Inds...>>
    {
        using type = SphereGLQGridShape<LayoutType>;
    };

    template <std::size_t N, std::size_t... Inds>
        requires (sizeof...(Inds) = Base::rank - N && N == Base::rank - 1)
    struct subshape_helper<N, std::index_sequence<Inds...>>
    {
        using type = SphereGLQSubGridShape<LayoutType>;
    };

    template <std::size_t N>
        requires (N == Base::rank)
    struct subshape_helper<N, std::index_sequence<>>
    {
        using type = NullShape;
    };

public:
    using size_type = typename Base::size_type;
    using Base::extents;

    template <std::size_t N>
        requires (0 < N && N <= Base::rank)
    using subshape_type = subshape_helper<N, std::make_index_sequence<Base::rank - N>>::type;

    SphereGLQGridTensorShape() = default;

    SphereGLQGridTensorShape(size_type outer_extent, size_type order)
        requires (Base::dynamic_rank == 3):
        Base(prepend(outer_extent, LayoutType::extents(order))), m_order(order) {}

    SphereGLQGridTensorShape(
        const std::array<size_type, Base::dynamic_rank - 2>& outer_extents, size_type order):
        Base(concatenate(outer_extents, LayoutType::extents(order))), m_order(order) {}

    SphereGLQGridTensorShape(
        const std::array<size_type, Base::rank - 2>& outer_extents, size_type order)
        requires (Base::dynamic_rank != Base::rank):
        Base(concatenate(outer_extents, LayoutType::extents(order))), m_order(order) {}

    template <std::integral... Inds>
        requires (sizeof...(Inds) < Base::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept
    {
        return subshape_type<sizeof...(Inds)>(m_order, take_last<Base::rank - sizeof...(Inds)>(extents()));
    }

    template <std::integral... Inds>
        requires (sizeof...(Inds) == Base::rank)
    [[nodiscard]] constexpr auto
    subshape([[maybe_unused]] Inds... inds) const noexcept { return NullShape{}; }

private:
    size_type m_order{};
};

template <typename LayoutType>
using SphereGLQGridVectorShape = SphereGLQGridTensorShape<LayoutType, std::dynamic_extent>;

/**
    @brief A non-owning view of a Gauss-Legendre quadrature grid on the sphere.

    @tparam ElementType Type of elements in the grid.
    @tparam LayoutType Layout of the grid.
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, typename LayoutType = DefaultLayout, std::size_t... inner_extents>
using SphereGLQGridSpan = ShapedSpan<ElementType, SphereGLQGridShape<LayoutType, inner_extents...>>;

/**
    @brief Container for Gauss-Legendre quadrature gridded data on the sphere.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
    @tparam inner_extents Extents of an inner multidimensional array structure.
*/
template <typename ElementType, typename LayoutType = DefaultLayout, std::size_t... inner_extents>
using SphereGLQGrid = ShapedArray<ElementType, SphereGLQGridShape<LayoutType, inner_extents...>>;

/**
    @brief A non-owning view of a multidiemensional array of Gauss-Legendre
    quadrature grids on the sphere.

    @tparam ElementType Type of elements in the grid.
    @tparam LayoutType Layout of the grid.
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <typename ElementType, typename LayoutType = DefaultLayout, std::size_t... outer_extents>
using SphereGLQGridTensorSpan = ShapedSpan<ElementType, SphereGLQGridTensorShape<LayoutType, outer_extents...>>;

/**
    @brief A non-owning view of an array of Gauss-Legendre quadrature grids on
    the sphere.

    @tparam ElementType Type of elements in the grid.
    @tparam LayoutType Layout of the grid.
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
using SphereGLQGridVectorSpan = SphereGLQGridTensorSpan<ElementType, LayoutType, std::dynamic_extent>;

/**
    @brief Container for a multidimensional array of Gauss-Legendre quadrature
    grids on the sphere.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
    @tparam outer_extents Extents of an outer multidimensional array structure.
*/
template <typename ElementType, typename LayoutType = DefaultLayout, std::size_t... outer_extents>
using SphereGLQGridTensor = ShapedArray<ElementType, SphereGLQGridTensorShape<LayoutType, outer_extents...>>;

/**
    @brief Container for an array of Gauss-Legendre quadrature grids on the
    sphere.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
using SphereGLQGridVector = SphereGLQGridTensor<ElementType, LayoutType, std::dynamic_extent>;

/**
    @brief Points defining a Gauss-Legendre quadrature grid on the sphere.

    @tparam LayoutType memory layout of the grid
*/
template <typename LayoutType = DefaultLayout>
class SphereGLQGridPoints
{
public:
    using GridLayout = LayoutType;
    SphereGLQGridPoints() = default;
    explicit SphereGLQGridPoints(std::size_t order) { resize(order); }

    /**
        @brief Change the size of the corresponding grid.
    */
    void resize(std::size_t order)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr std::size_t lat_axis = GridLayout::lat_axis;
        const auto shape = GridLayout::extents(order);
        resize(shape[lon_axis], shape[lat_axis]);
    }

    /**
        @brief Shape of the corresponding grid.
    */
    [[nodiscard]] std::array<std::size_t, 2> extents() noexcept
    {
        return {m_glq_nodes.size(), m_longitudes.size()};
    }

    /**
        @brief Longitude values of the grid points.
    */
    [[nodiscard]] std::span<const double> longitudes() const noexcept
    {
        return m_longitudes;
    }

    /**
        @brief Latitudinal Gauss-Legendre nodes.
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
    template <typename FuncType>
        requires std::same_as<std::invoke_result_t<FuncType, double, double>, double>
    void generate_values(SphereGLQGridSpan<double, GridLayout> grid, FuncType&& f)
    {
        resize(grid.order());

        if constexpr (std::same_as<GridLayout, LatLonLayout<typename LayoutType::Alignment>>)
        {
            for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            {
                const double colatitude = m_glq_nodes[i];
                for (std::size_t j = 0; j < m_longitudes.size(); ++j)
                {
                    const double lon = m_longitudes[j];
                    grid[i, j] = std::forward<FuncType>(f)(lon, colatitude);
                }
            }
        }
        else if constexpr (std::same_as<GridLayout, LonLatLayout<typename LayoutType::Alignment>>)
        {
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
            {
                const double lon = m_longitudes[i];
                for (std::size_t j = 0; j < m_glq_nodes.size(); ++j)
                {
                    const double colatitude = m_glq_nodes[j];
                    grid[i, j] = std::forward<FuncType>(f)(lon, colatitude);
                }
            }
        }
    }

    /**
        @brief Generate Gauss-Legendre quadrature grid values from a function.

        @tparam FuncType type of function

        @param grid grid to place the values in
        @param f function to generate values
    */
    template <typename FuncType>
        requires std::same_as<std::invoke_result_t<FuncType, double, double>, double>
    void generate_values(SphereGLQGrid<double, GridLayout>& grid, FuncType&& f)
    {
        generate_values((typename SphereGLQGrid<double, GridLayout>::view)(grid), std::forward<FuncType>(f));
    }

    /**
        @brief Generate Gauss-Legendre quadrature grid values from a function.

        @tparam FuncType type of function

        @param f function to generate values
    */
    template <typename FuncType>
        requires std::same_as<std::invoke_result_t<FuncType, double, double>, double>
    auto generate_values(FuncType&& f, std::size_t order)
    {
        auto grid = SphereGLQGrid<double, GridLayout>(order);
        generate_values((typename SphereGLQGrid<double, GridLayout>::view)(grid), std::forward<FuncType>(f));
        return grid;
    }

private:
    void resize(std::size_t num_lon, std::size_t num_lat)
    {
        if (num_lon != m_longitudes.size())
        {
            m_longitudes.resize(num_lon);
            const double dlon = (2.0*std::numbers::pi)/double(m_longitudes.size());
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
                m_longitudes[i] = dlon*double(i);
        }
        if (num_lat != m_glq_nodes.size())
        {
            m_glq_nodes.resize(num_lat);
            gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::angle>(m_glq_nodes, m_glq_nodes.size() & 1);
        }
    }

    std::vector<double> m_longitudes;
    std::vector<double> m_glq_nodes;
};

} // namespace zest::st
