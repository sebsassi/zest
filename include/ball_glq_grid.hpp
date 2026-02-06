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
#include <type_traits>
#include <vector>

#include "alignment.hpp"
#include "gauss_legendre.hpp"
#include "shape.hpp"
#include "shaped_array.hpp"
#include "shaped_span.hpp"

namespace zest::zt
{

/**
    @brief Layout for storing a Gauss-Legendre quadrature grid.

    @tparam AlignmentType byte alignment of the grid
*/
template <typename AlignmentType = CacheLineAlignment>
struct LonLatRadLayout
{
    using Alignment = AlignmentType;

    /**
        @brief Number of grid points.

        @param order order of Zernike expansion
    */
    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return lat_size(order)*lon_size(order)*rad_size(order);
    }

    /**
        @brief Shape of the grid.

        @param order order of Zernike expansion
    */
    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    extents(std::size_t order) noexcept
    {
        return {lon_size(order), lat_size(order), rad_size(order)};
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
    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    fft_stride(std::size_t order) noexcept
    {
        return {lat_size(order)*rad_size(order), rad_size(order), 1};
    }

    /**
        @brief Size in latitudinal direction.
    */
    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t order) noexcept
    {
        return order + 1UL;
    }


    /**
        @brief Size in radial direction.
    */
    [[nodiscard]] static constexpr std::size_t
    rad_size(std::size_t order) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = order + 1UL;
        if constexpr (std::is_same_v<Alignment, NoAlignment>)
            return min_size;
        else
            return zest::detail::next_divisible<vector_size>(min_size);
    }

    /**
        @brief Size in latitudinal direction.
    */
    [[nodiscard]] static constexpr std::size_t
    lon_size(std::size_t order) noexcept
    {
        return 2UL*order - std::min(1UL, order);
    }

    static constexpr std::size_t lat_axis = 1UL;
    static constexpr std::size_t lon_axis = 0UL;
    static constexpr std::size_t rad_axis = 2UL;
};

using DefaultLayout = LonLatRadLayout<>;

template <typename LayoutType>
class BallGLQGridShape: public DynamicTensorShape<3>
{
public:
    BallGLQGridShape() = default;
    BallGLQGridShape(size_type order):
        DynamicTensorShape<3>(LayoutType::extents(order)), m_order(order) {}

    [[nodiscard]] constexpr size_type
    order() const noexcept { return m_order; }

private:
    size_type m_order{};
};

/**
    @brief A non-owning view of gridded data in spherical coordinates in the unit ball.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
using BallGLQGridSpan = ShapedSpan<ElementType, BallGLQGridShape<LayoutType>>;

/**
    @brief Container for gridded data in spherical coordinates in the unit ball.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
using BallGLQGrid = ShapedArray<ElementType, BallGLQGridShape<LayoutType>>;

/**
    @brief Points defining a grid in spherical coordinates in the unit ball.

    @tparam LayoutType memory layout of the grid
*/
template <typename LayoutType = DefaultLayout>
class BallGLQGridPoints
{
public:
    using GridLayout = LayoutType;
    BallGLQGridPoints() = default;
    explicit BallGLQGridPoints(std::size_t order) { resize(order); }

    /**
        @brief Change the size of the corresponding grid.
    */
    void resize(std::size_t order)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr std::size_t lat_axis = GridLayout::lat_axis;
        constexpr std::size_t rad_axis = GridLayout::rad_axis;
        const auto shape = GridLayout::extents(order);
        resize(shape[lon_axis], shape[lat_axis], shape[rad_axis]);
    }

    /**
        @brief Longitude values of the grid points.
    */
    [[nodiscard]] std::span<const double> longitudes() const noexcept
    {
        return m_longitudes;
    }

    /**
        @brief Radial Gauss-Legendre nodes.
    */
    [[nodiscard]] std::span<const double> rad_glq_nodes() const noexcept
    {
        return m_rad_glq_nodes;
    }

    /**
        @brief Latitudinal Gauss-Legendre nodes.
    */
    [[nodiscard]] std::span<const double> lat_glq_nodes() const noexcept
    {
        return m_lat_glq_nodes;
    }

    /**
        @brief Generate Gauss-Legendre quadrature grid values from a function.

        @tparam FuncType type of function

        @param grid grid to place the values in
        @param f function to generate values
    */
    template <typename FuncType>
        requires std::same_as<std::invoke_result_t<FuncType, double, double, double>, double>
    void generate_values(BallGLQGridSpan<double, GridLayout> grid, FuncType&& f)
    {
        resize(grid.order());

        if constexpr (std::same_as<GridLayout, LonLatRadLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
            {
                const double lon = m_longitudes[i];
                for (std::size_t j = 0; j < m_lat_glq_nodes.size(); ++j)
                {
                    const double colatitude = m_lat_glq_nodes[j];
                    for (std::size_t k = 0; k < m_rad_glq_nodes.size(); ++k)
                    {
                        const double r = m_rad_glq_nodes[k];
                        grid(i, j, k) = std::forward<FuncType>(f)(lon, colatitude, r);
                    }
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
        requires std::same_as<std::invoke_result_t<FuncType, double, double, double>, double>
    void generate_values(BallGLQGrid<double, GridLayout>& grid, FuncType&& f)
    {
        generate_values((typename BallGLQGrid<double, GridLayout>::view)(grid), std::forward<FuncType>(f));
    }

    /**
        @brief Generate Gauss-Legendre quadrature grid values from a function.

        @tparam FuncType type of function

        @param f function to generate values
    */
    template <typename FuncType>
        requires std::same_as<std::invoke_result_t<FuncType, double, double, double>, double>
    [[nodiscard]] auto generate_values(FuncType&& f, std::size_t order)
    {
        BallGLQGrid<double, GridLayout> grid(order);
        generate_values((typename BallGLQGrid<double, GridLayout>::view)(grid), std::forward<FuncType>(f));
        return grid;
    }

#ifdef ZEST_USE_OMP
    template <ball_glq_grid GridType, typename FuncType>
        requires std::same_as<
            typename std::remove_cvref_t<GridType>::Layout, GridLayout>
    void generate_values(
        BallGLQGridSpan<double> grid, FuncType&& f, std::size_t num_threads)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr std::size_t lat_axis = GridLayout::lat_axis;
        constexpr std::size_t rad_axis = GridLayout::rad_axis;
        const auto shape = grid.shape();
        resize(shape[lon_axis], shape[lat_axis], shape[rad_axis]);

        std::size_t nthreads = (num_threads) ?
                num_threads : std::size_t(omp_get_max_threads());
        if constexpr (std::same_as<GridLayout, LonLatRadLayout<typename GridLayout::Alignment>>)
        {
            #pragma omp parallel for num_threads(nthreads) collapse(2)
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
            {
                for (std::size_t j = 0; j < m_lat_glq_nodes.size(); ++j)
                {
                    const double lon = m_longitudes[i];
                    const double colatitude = m_lat_glq_nodes[j];
                    for (std::size_t k = 0; k < m_rad_glq_nodes.size(); ++k)
                    {
                        const double r = m_rad_glq_nodes[k];
                        grid(i, j, k) = f(r, lon, colatitude);
                    }
                }
            }
        }
    }

    template <typename FuncType>
    [[nodiscard]] auto generate_values(
        FuncType&& f, std::size_t order, std::size_t num_threads)
    {
        using CodomainType = std::invoke_result_t<FuncType, double, double, double>;
        BallGLQGrid<CodomainType, GridLayout> grid(order);
        generate_values(grid, f, num_threads);
        return grid;
    }
#endif

private:
    void resize(std::size_t num_lon, std::size_t num_lat, std::size_t num_rad)
    {
        if (num_lon != m_longitudes.size())
        {
            m_longitudes.resize(num_lon);
            const double dlon = (2.0*std::numbers::pi)/double(m_longitudes.size());
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
                m_longitudes[i] = dlon*double(i);
        }
        if (num_lat != m_lat_glq_nodes.size())
        {
            m_lat_glq_nodes.resize(num_lat);
            gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::angle>(m_lat_glq_nodes, m_lat_glq_nodes.size() & 1);
        }
        if (num_rad != m_rad_glq_nodes.size())
        {
            m_rad_glq_nodes.resize(num_rad);
            gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::cos>(m_rad_glq_nodes, m_rad_glq_nodes.size() & 1);
            for (auto& node : m_rad_glq_nodes)
                node = 0.5*(1.0 + node);
        }
    }

    std::vector<double> m_rad_glq_nodes;
    std::vector<double> m_lat_glq_nodes;
    std::vector<double> m_longitudes;
};

} // namespace zest::zt
