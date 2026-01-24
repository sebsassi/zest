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
#include <complex>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include "pocketfft_spec.hpp"

#include "alignment.hpp"
#include "associated_legendre_recursion.hpp"
#include "gauss_legendre.hpp"
#include "md_span.hpp"
#include "radial_zernike_recursion.hpp"
#include "zernike_expansion.hpp"

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
    size_type m_order;
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

/**
    @brief Class for transforming between a Gauss-Legendre quadrature grid
    representation and Zernike polynomial expansion representation of data in
    the unit baal.

    @tparam zernike_norm_param normalization convention of Zernike functions
    @tparam sh_norm_param normalization convention of spherical harmonics
    @tparam sh_phase_param phase convention of spherical harmonics
    @tparam GridLayoutType
*/
template <
    ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param,
    st::SHPhase sh_phase_param, typename GridLayoutType = DefaultLayout>
class GLQTransformer
{
private:
    template <typename T, std::size_t... Ns>
    using AssLegSpan = st::AssociatedLegendreSpan<T, sh_norm_param, sh_phase_param, Ns...>;

    using AssLegShape = st::AssociatedLegendreShape<sh_norm_param, sh_phase_param>;

    template <typename T, std::size_t... Ns>
    using RadZerSpan = RadialZernikeSpan<T, zernike_norm_param, Ns...>;

    using RadZerShape = RadialZernikeShape<zernike_norm_param>;

public:
    using grid_layout_type = GridLayoutType;

    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm sh_norm = sh_norm_param;
    static constexpr st::SHPhase sh_phase = sh_phase_param;

    GLQTransformer():
        m_pocketfft_shape_grid(3), m_pocketfft_stride_grid(3), 
        m_pocketfft_stride_fft(3) {};
    explicit GLQTransformer(std::size_t order):
        m_zernike_recursion(order), m_ass_leg_recursion(order),
        m_rad_glq_nodes(grid_layout_type::rad_size(order)),
        m_rad_glq_weights(grid_layout_type::rad_size(order)),
        m_lat_glq_nodes(grid_layout_type::lat_size(order)),
        m_lat_glq_weights(grid_layout_type::lat_size(order)),
        m_zernike_grid(grid_layout_type::rad_size(order)*RadZerShape::size(order)),
        m_ass_leg_grid(grid_layout_type::lat_size(order)*AssLegShape::size(order)),
        m_flm_grid(grid_layout_type::rad_size(order)*AssLegShape::size(order)*2),
        m_ffts(grid_layout_type::rad_size(order)*grid_layout_type::lat_size(order)*grid_layout_type::fft_size(order)),
        m_pocketfft_shape_grid(3),
        m_pocketfft_stride_grid(3),
        m_pocketfft_stride_fft(3),
        m_order(order)
    {
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::cos>(
                m_rad_glq_nodes, m_rad_glq_weights,
                m_rad_glq_weights.size() & 1);
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::cos>(
                m_lat_glq_nodes, m_lat_glq_weights,
                m_lat_glq_weights.size() & 1);

        for (auto& node : m_rad_glq_nodes)
            node = 0.5*(1.0 + node);

        RadZerSpan<double, std::dynamic_extent>
        zernike(m_zernike_grid, order, m_rad_glq_nodes.size());

        m_zernike_recursion.generate<zernike_norm_param>(
                m_rad_glq_nodes, zernike);

        AssLegSpan<double, std::dynamic_extent>
        ass_leg(m_ass_leg_grid, order, m_lat_glq_nodes.size());

        m_ass_leg_recursion.generate_real(m_lat_glq_nodes, ass_leg);

        auto shape = grid_layout_type::extents(order);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];
        m_pocketfft_shape_grid[2] = shape[2];

        m_pocketfft_stride_grid[0] = long(shape[1]*shape[2]*sizeof(double));
        m_pocketfft_stride_grid[1] = long(shape[2]*sizeof(double));
        m_pocketfft_stride_grid[2] = sizeof(double);

        auto fft_stride = grid_layout_type::fft_stride(order);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = long(fft_stride[1]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[2] = long(fft_stride[2]*sizeof(std::complex<double>));
    }

    /**
        @brief Order of Zernike expansion.
    */
    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    /**
        @brief Resize transformer for specified expansion order.
    */
    void resize(std::size_t order)
    {
        if (order == m_order) return;

        m_ass_leg_recursion.expand(order);
        m_zernike_recursion.expand(order);

        m_rad_glq_nodes.resize(grid_layout_type::rad_size(order));
        m_rad_glq_weights.resize(grid_layout_type::rad_size(order));
        m_lat_glq_nodes.resize(grid_layout_type::lat_size(order));
        m_lat_glq_weights.resize(grid_layout_type::lat_size(order));

        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::cos>(
                m_rad_glq_nodes, m_rad_glq_weights,
                m_rad_glq_weights.size() & 1);
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::cos>(
                m_lat_glq_nodes, m_lat_glq_weights,
                m_lat_glq_weights.size() & 1);

        for (auto& node : m_rad_glq_nodes)
            node = 0.5*(1.0 + node);

        m_zernike_grid.resize(grid_layout_type::rad_size(order)*RadZerShape::size(order));

        RadZerSpan<double, std::dynamic_extent>
        zernike(m_zernike_grid, order, m_rad_glq_nodes.size());

        m_zernike_recursion.generate<zernike_norm_param>(
                m_rad_glq_nodes, zernike);

        m_ass_leg_grid.resize(grid_layout_type::lat_size(order)*AssLegShape::size(order));
        m_flm_grid.resize(grid_layout_type::rad_size(order)*AssLegShape::size(order)*2);

        AssLegSpan<double, std::dynamic_extent> ass_leg(m_ass_leg_grid, order, m_lat_glq_nodes.size());
        m_ass_leg_recursion.generate_real(m_lat_glq_nodes, ass_leg);

        m_ffts.resize(grid_layout_type::rad_size(order)*grid_layout_type::lat_size(order)*grid_layout_type::fft_size(order));
        std::array<std::size_t, 3> shape = grid_layout_type::extents(order);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];
        m_pocketfft_shape_grid[2] = shape[2];

        m_pocketfft_stride_grid[0] = long(shape[1]*shape[2]*sizeof(double));
        m_pocketfft_stride_grid[1] = long(shape[2]*sizeof(double));
        m_pocketfft_stride_grid[2] = sizeof(double);

        std::array<std::size_t, 3> fft_stride = grid_layout_type::fft_stride(order);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = long(fft_stride[1]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[2] = long(fft_stride[2]*sizeof(std::complex<double>));

        m_order = order;
    }

    /**
        @brief Forward transform from Gauss-Legendre quadrature grid to Zernike coefficients.

        @param values values on the ball quadrature grid
        @param expansion coefficients of the expansion
    */
    void forward_transform(
        BallGLQGridSpan<const double, grid_layout_type> values,
        ZernikeSpan<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion)
    {
        resize(values.order());

        integrate_longitudinal(values);
        apply_weights();

        std::size_t min_order = std::min(expansion.order(), values.order());
        integrate_latitudinal(min_order);

        ZernikeSpan<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase>
        truncated_expansion(expansion.flatten(), min_order);

        integrate_radial(truncated_expansion);
    }

    /**
        @brief Backward transform from Zernike expansion to Gauss-Legendre quadrature grid.

        @param expansion coefficients of the expansion
        @param values values on the ball quadrature grid
    */
    void backward_transform(
        ZernikeSpan<const double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion,
        BallGLQGridSpan<double, grid_layout_type> values)
    {
        resize(values.order());

        std::size_t min_order = std::min(expansion.order(), values.order());

        ZernikeSpan<const double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase>
        truncated_expansion(expansion.flatten(), min_order);

        sum_n(truncated_expansion);
        sum_l(min_order);
        sum_m(values);
    }

    /**
        @brief Forward transform from Gauss-Legendre quadrature grid to Zernike coefficients.

        @param values values on the ball quadrature grid
        @param order order of expansion
    */
    [[nodiscard]] ZernikeExpansion<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase>
    forward_transform(BallGLQGridSpan<const double, grid_layout_type> values, std::size_t order)
    {
        ZernikeExpansion<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase>
        expansion(order);

        forward_transform(values, expansion);
        return expansion;
    }

    /**
        @brief Backward transform from Zernike coefficients to Gauss-Legendre quadrature grid.

        @param values values on the ball quadrature grid
        @param expansion coefficients of the expansion
    */
    [[nodiscard]] BallGLQGrid<double, grid_layout_type>
    backward_transform(
        ZernikeSpan<const double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion,
        std::size_t order)
    {
        BallGLQGrid<double, grid_layout_type> grid(order);
        backward_transform(expansion, grid);
        return grid;
    }

private:
    void integrate_longitudinal(
        BallGLQGridSpan<const double, grid_layout_type> values)
    {
        constexpr std::size_t lon_axis = grid_layout_type::lon_axis;
        constexpr double radial_integral_norm = 0.5;
        constexpr double sh_norm = st::normalization<sh_norm_param>();
        const double fourier_norm = (2.0*std::numbers::pi)/double(values.extent(lon_axis));
        const double prefactor = sh_norm*radial_integral_norm*fourier_norm;
        pocketfft::r2c(
            m_pocketfft_shape_grid, m_pocketfft_stride_grid, m_pocketfft_stride_fft,
            lon_axis, pocketfft::FORWARD, values.flatten().data(), m_ffts.data(),
            prefactor);
    }

    void apply_weights() noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        const std::size_t lat_glq_size = m_lat_glq_weights.size();
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);

        MDSpan<std::complex<double>, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>
        fft(m_ffts, fft_order, lat_glq_size, rad_glq_size);

        if constexpr (std::same_as<grid_layout_type, LonLatRadLayout<typename grid_layout_type::Alignment>>)
        {
            for (std::size_t m = 0; m < fft_order; ++m)
            {
                auto fft_m = fft[m];
                for (std::size_t i = 0; i < lat_glq_size; ++i)
                {
                    auto fft_mi = fft_m[i];
                    const double lat_weight = m_lat_glq_weights[i];
                    for (std::size_t j = 0; j < rad_glq_size; ++j)
                    {
                        const double r = m_rad_glq_nodes[j];
                        const double weight
                                = lat_weight*r*r*m_rad_glq_weights[j];
                        std::complex<double>& x = fft_mi[j];
                        x = {weight*x.real(), -weight*x.imag()};
                    }
                }
            }
        }

        /*
        else if (std::same_as<GridLayout, RadLatLonLayout>)
        {
            for (std::size_t i = 0; i < rad_glq_size; ++i)
            {
                std::span<std::complex<double>> fft_i(
                        m_ffts.begin() + i*fft_order*lat_glq_size, 
                        fft_order*lat_glq_size);
                const double r = m_rad_glq_nodes[i];
                const double radial_weight = r*r*m_glq_weights[i];
                for (std::size_t j = 0; j < lat_glq_size; ++j)
                {
                    std::span<std::complex<double>> fft_ij((
                            fft_i.begin() + j*fft_order, fft_order);
                    const double weight = radial_weight*m_lat_glq_weights[j];
                    for (std::size_t m = 0; m < fft_order; ++m)
                    {
                        std::complex<double>& x = fft_ij[m];
                        x = {weight*x.real(), -weight*x.imag()};
                    }
                }
            }
        }
        */
    }

    void integrate_latitudinal(std::size_t min_order) noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        const std::size_t lat_glq_size = m_lat_glq_weights.size();
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);
        std::ranges::fill(m_flm_grid, 0.0);

        AssLegSpan<double, std::dynamic_extent, 2>
        flm(m_flm_grid, min_order, rad_glq_size);

        AssLegSpan<const double, std::dynamic_extent>
        ass_leg(m_ass_leg_grid, min_order, m_lat_glq_nodes.size());

        MDSpan<const std::complex<double>, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>
        fft(m_ffts, fft_order, lat_glq_size, rad_glq_size);

        if constexpr (std::same_as<grid_layout_type, LonLatRadLayout<typename grid_layout_type::Alignment>>)
        {
            for (auto l : flm.indices())
            {
                auto ass_leg_l = ass_leg[l];
                auto flm_l = flm[l];
                for (auto m : flm_l.indices())
                {
                    auto flm_lm = flm_l[m];
                    auto ass_leg_lm = ass_leg_l[m];
                    auto fft_m = fft[m];
                    for (std::size_t i = 0; i < lat_glq_size; ++i)
                    {
                        const double ass_leg_lmi = ass_leg_lm[i];
                        auto fft_mi = fft_m[i];
                        for (std::size_t j = 0; j < rad_glq_size; ++j)
                        {
                            flm_lm[j, 0] += ass_leg_lmi*fft_mi[j].real();
                            flm_lm[j, 1] += ass_leg_lmi*fft_mi[j].imag();
                        }
                    }
                }
            }
        }
    }

    void integrate_radial(
        ZernikeSpan<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion) noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        std::ranges::fill(expansion.flatten(), 0.0);

        AssLegSpan<const double, std::dynamic_extent, 2>
        flm(m_flm_grid, m_order, rad_glq_size);

        RadZerSpan<const double, std::dynamic_extent>
        zernike(m_zernike_grid, m_order, m_rad_glq_nodes.size());

        if constexpr (std::same_as<grid_layout_type, LonLatRadLayout<typename grid_layout_type::Alignment>>)
        {
            for (auto n : expansion.indices())
            {
                const double norm = normalization<zernike_norm_param>(n);
                auto zernike_n = zernike[n];
                auto expansion_n = expansion[n];

                for (auto l : expansion_n.indices())
                {
                    auto flm_l = flm[l];
                    auto expansion_nl = expansion_n[l];

                    auto zernike_nl = zernike_n[l];
                    for (auto m : expansion_nl.indices())
                    {
                        auto flm_lm = flm_l[m];
                        for (std::size_t i = 0; i < rad_glq_size; ++i)
                        {
                            expansion_nl[m, 0] += zernike_nl[i]*flm_lm[i, 0];
                            expansion_nl[m, 1] += zernike_nl[i]*flm_lm[i, 1];
                        }

                        if constexpr (zernike_norm_param == ZernikeNorm::unnormed)
                        {
                            expansion_nl[m, 0] *= norm;
                            expansion_nl[m, 1] *= norm;
                        }
                    }
                }
            }
        }
    }

    void sum_n(
        ZernikeSpan<const double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion) noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        std::ranges::fill(m_flm_grid, 0.0);

        RadZerSpan<const double, std::dynamic_extent>
        zernike(m_zernike_grid, expansion.order(), rad_glq_size);

        AssLegSpan<double, std::dynamic_extent, 2>
        flm(m_flm_grid, expansion.order(), rad_glq_size);

        for (auto n : expansion.indices())
        {
            auto zernike_n = zernike[n];
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto zernike_nl = zernike_n[l];
                auto expansion_nl = expansion_n[l];
                auto flm_l = flm[l];
                for (auto m : expansion_nl.indices())
                {
                    auto flm_lm = flm_l[m];
                    for (std::size_t i = 0; i < rad_glq_size; ++i)
                    {
                        flm_lm[i, 0] += zernike_nl[i]*expansion_nl[m, 0];
                        flm_lm[i, 1] += zernike_nl[i]*expansion_nl[m, 1];
                    }
                }
            }
        }
    }

    void sum_l(std::size_t min_order) noexcept
    {
        const std::size_t lat_glq_size = m_lat_glq_weights.size();
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);

        AssLegSpan<const double, std::dynamic_extent, 2>
        flm(m_flm_grid, min_order, rad_glq_size);

        AssLegSpan<const double, std::dynamic_extent>
        ass_leg(m_ass_leg_grid, m_order, m_lat_glq_nodes.size());

        std::ranges::fill(m_ffts, std::complex<double>{});

        MDSpan<std::complex<double>, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>
        fft(m_ffts, fft_order, lat_glq_size, rad_glq_size);

        for (auto l : flm.indices())
        {
            auto ass_leg_l = ass_leg[l];
            auto flm_l = flm[l];
            for (auto m : flm_l.indices())
            {
                auto flm_lm = flm_l[m];
                auto ass_leg_lm = ass_leg_l[m];
                const double half_or_one = (m > 0) ? 0.5 : 1.0;

                auto fft_m = fft[m];
                for (std::size_t i = 0; i < lat_glq_size; ++i)
                {
                    const double ass_leg_lm_i = ass_leg_lm[i];
                    const double weight = half_or_one*ass_leg_lm_i;
                    auto fft_mi = fft_m[i];
                    for (std::size_t j = 0; j < rad_glq_size; ++j)
                        fft_mi[j] += std::complex<double>{weight*flm_lm[j, 0], -weight*flm_lm[j, 1]};
                }
            }
        }
    }

    void sum_m(BallGLQGridSpan<double, grid_layout_type> values)
    {
        constexpr std::size_t lon_axis = grid_layout_type::lon_axis;
        constexpr double prefactor = 1.0;
        pocketfft::c2r(
            m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid,
            lon_axis, pocketfft::BACKWARD, m_ffts.data(), values.flatten().data(),
            prefactor);
    }

    RadialZernikeRecursion m_zernike_recursion;
    st::AssociatedLegendreRecursion m_ass_leg_recursion;
    std::vector<double> m_rad_glq_nodes;
    std::vector<double> m_rad_glq_weights;
    std::vector<double> m_lat_glq_nodes;
    std::vector<double> m_lat_glq_weights;
    std::vector<double> m_zernike_grid;
    std::vector<double> m_ass_leg_grid;
    std::vector<double> m_flm_grid;
    std::vector<std::complex<double>> m_ffts;
    std::vector<std::size_t> m_pocketfft_shape_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft;
    std::size_t m_order{};
};

/**
    @brief Convenient alias for `GLQTransformer` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerAcoustics
    = GLQTransformer<
        ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with orthonorml Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerNormalAcoustics
    = GLQTransformer<
        ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::none, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerQM
    = GLQTransformer<
        ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerNormalQM
    = GLQTransformer<
        ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerGeo
    = GLQTransformer<
        ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerNormalGeo
    = GLQTransformer<
        ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none, GridLayout>;

/**
    @brief High-level interface for taking Zernike transforms of functions on
    balls of arbitrary radii.

    @tparam zernike_norm_param normalization convention of Zernike functions
    @tparam sh_norm_param normalization convention of spherical harmonics
    @tparam sh_phase_param phase convention of spherical harmonics
    @tparam GridLayoutType
*/
template <ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param, typename GridLayoutType = DefaultLayout>
class ZernikeTransformer
{
public:
    using grid_layout_type = GridLayoutType;

    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm sh_norm = sh_norm_param;
    static constexpr st::SHPhase sh_phase = sh_phase_param;

    ZernikeTransformer() = default;
    explicit ZernikeTransformer(std::size_t order):
        m_grid(order), m_points(order), m_transformer(order) {}

    /**
        @brief Resize the transformer to work with expansions of different
        order.
    */
    void resize(std::size_t order)
    {
        m_points.resize(order);
        m_grid.reshape(order);
        m_transformer.resize(order);
    }

    /**
        @brief Get Zernike expansion of a function expressed in spherical
        coordinates.

        @tparam FuncType type of function

        @param f function to transform
        @param radius radius of the ball `f` is defined on
        @param expansion buffer to store the expansion
    */
    template <ball_function FuncType>
    void transform(
        FuncType&& f, double radius,
        ZernikeSpan<double, IndexingMode::zero_based, zernike_norm_param, sh_norm_param, sh_phase_param> expansion)
    {
        auto f_scaled = [&](double lon, double colat, double r) {
            return std::forward<FuncType>(f)(lon, colat, r*radius);
        };
        resize(expansion.order());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.forward_transform(m_grid, expansion);
    }

    /**
        @brief Get Zernike expansion of a function expressed in spherical
        coordinates.

        @tparam FuncType type of function

        @param f function to transform
        @param radius radius of the ball `f` is defined on
        @param order order of the expansion

        @returns Zernike expansion
    */
    template <ball_function FuncType>
    [[nodiscard]] ZernikeExpansion<double, IndexingMode::zero_based, zernike_norm_param, sh_norm_param, sh_phase_param>
    transform(FuncType&& f, double radius, std::size_t order)
    {
        auto f_scaled = [&](double lon, double colat, double r) {
            return std::forward<FuncType>(f)(lon, colat, r*radius);
        };
        resize(order);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.forward_transform(m_grid, order);
    }

    /**
        @brief Get Zernike expansion of a function expressed in Cartesian
        coordinates.

        @tparam FuncType type of function

        @param f function to transform
        @param radius radius of the ball `f` is defined on
        @param expansion buffer to store the expansion
    */
    template <cartesian_function FuncType>
    void transform(
        FuncType&& f, double radius,
        ZernikeSpan<double, IndexingMode::zero_based, zernike_norm_param, sh_norm_param, sh_phase_param> expansion)
    {
        auto f_scaled = [&](double lon, double colat, double r) {
            const double rad = r*radius;
            const double scolat = std::sin(colat);
            const std::array<double, 3> x = {
                rad*scolat*std::cos(lon), rad*scolat*std::sin(lon),
                rad*std::cos(colat)
            };
            return std::forward<FuncType>(f)(x);
        };
        resize(expansion.order());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.forward_transform(m_grid, expansion);
    }

    /**
        @brief Get spherical harmonic expansion of a function expressed in
        Cartesian coordinates.

        @tparam FuncType type of function

        @param f function to transform
        @param radius radius of the ball `f` is defined on
        @param order order of the expansion

        @returns Zernike expansion
    */
    template <cartesian_function FuncType>
    [[nodiscard]] ZernikeExpansion<double, IndexingMode::zero_based, zernike_norm_param, sh_norm_param, sh_phase_param>
    transform(FuncType&& f, double radius, std::size_t order)
    {
        auto f_scaled = [&](double lon, double colat, double r) {
            const double rad = r*radius;
            const double scolat = std::sin(colat);
            const std::array<double, 3> x = {
                rad*scolat*std::cos(lon), rad*scolat*std::sin(lon),
                rad*std::cos(colat)
            };
            return std::forward<FuncType>(f)(x);
        };
        resize(order);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.forward_transform(m_grid, order);
    }

private:
    BallGLQGrid<double, grid_layout_type> m_grid;
    BallGLQGridPoints<grid_layout_type> m_points;
    GLQTransformer<zernike_norm_param, sh_norm_param, sh_phase_param, grid_layout_type> m_transformer;
};

/**
    @brief Convenient alias for `ZernikeTransformer` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerAcoustics
    = ZernikeTransformer<
        ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerNormalAcoustics
    = ZernikeTransformer<
        ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::none, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerQM
    = ZernikeTransformer<
        ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerNormalQM
    = ZernikeTransformer<
        ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerGeo
    = ZernikeTransformer<
        ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerNormalGeo
    = ZernikeTransformer<
        ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none, GridLayout>;

} // namespace zest::zt

