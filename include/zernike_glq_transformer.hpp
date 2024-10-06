/*
Copyright (c) 2024 Sebastian Sassi

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

#include <vector>
#include <array>
#include <span>
#include <complex>

#include "pocketfft_spec.hpp"

#include "gauss_legendre.hpp"
#include "plm_recursion.hpp"
#include "zernike_expansion.hpp"
#include "radial_zernike_recursion.hpp"
#include "alignment.hpp"
#include "md_span.hpp"

namespace zest
{
namespace zt
{

/**
    @brief Layout for storing a Gauss-Legendre quadrature grid.

    @tparam AlignmentType byte alignment of the grid
*/
template <typename AlignmentType = CacheLineAlignment>
struct LonLatRadLayout
{
    using Alignment = AlignmentType;

    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return lat_size(order)*lon_size(order)*rad_size(order);
    }
    
    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    shape(std::size_t order) noexcept
    {
        return {lon_size(order), lat_size(order), rad_size(order)};
    }

    [[nodiscard]] static constexpr std::size_t
    fft_size(std::size_t order) noexcept
    {
        return (lon_size(order) >> 1) + 1;
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    fft_stride(std::size_t order) noexcept
    {
        return {lat_size(order)*rad_size(order), rad_size(order), 1};
    }

    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t order) noexcept
    {
        return order + 1UL;
    }

    [[nodiscard]] static constexpr std::size_t
    rad_size(std::size_t order) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = order + 1UL;
        if constexpr (std::is_same_v<Alignment, NoAlignment>)
            return min_size;
        else
            return detail::next_divisible<vector_size>(min_size);
    }

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

/**
    @brief A non-owning view of gridded data in spherical coordinates in the unit ball.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
class BallGLQGridSpan: public MDSpan<ElementType, 3>
{
public:
    using typename MDSpan<ElementType, 3>::element_type;
    using Layout = LayoutType;
    using ConstView = BallGLQGridSpan<const element_type, Layout>;

    using MDSpan<ElementType, 3>::extents;
    using MDSpan<ElementType, 3>::data;
    using MDSpan<ElementType, 3>::size;

    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return Layout::size(order);
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    shape(std::size_t order) noexcept
    {
        return Layout::shape(order);
    }

    BallGLQGridSpan() noexcept = default;
    constexpr BallGLQGridSpan(element_type* data, std::size_t order) noexcept:
        MDSpan<ElementType, 3>(data, Layout::shape(order)), m_order(order) {}
    constexpr BallGLQGridSpan(
        std::span<element_type> buffer, std::size_t order) noexcept:
        MDSpan<ElementType, 3>(buffer.data(), Layout::shape(order)),
        m_order(order) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }
    
    [[nodiscard]] constexpr const std::array<std::size_t, 3>&
    shape() const noexcept { return extents(); }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span<element_type>(data(), size()); }

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), extents(), m_order);
    }

private:
    friend BallGLQGridSpan<std::remove_const_t<element_type>, Layout>;

    constexpr BallGLQGridSpan(
        element_type* data, std::size_t size, const std::array<std::size_t, 3>& extents, std::size_t order) noexcept:
        MDSpan<ElementType, 3>(data, size, extents), m_order(order) {}
    
    std::size_t m_order{};
};

/**
    @brief Container for gridded data in spherical coordinates in the unit ball.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
class BallGLQGrid
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cvref_t<element_type>;
    using Layout = LayoutType;
    using View = BallGLQGridSpan<double>;
    using ConstView = BallGLQGridSpan<const double>;

    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return Layout::size(order);
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    shape(std::size_t order) noexcept
    {
        return Layout::shape(order);
    }

    BallGLQGrid() = default;
    explicit BallGLQGrid(std::size_t order):
        m_data(Layout::size(order)), m_shape(Layout::shape(order)),
        m_order(order) {}

    [[nodiscard]] std::array<std::size_t, 3> shape() { return m_shape; }
    [[nodiscard]] std::size_t order() const noexcept { return m_order; }
    [[nodiscard]] std::span<const element_type> flatten() const noexcept { return m_data; }
    std::span<element_type> flatten() noexcept { return m_data; }

    [[nodiscard]] operator View() noexcept
    {
        return View(m_data, m_order);
    };

    [[nodiscard]] operator ConstView() const noexcept
    {
        return ConstView(m_data, m_order);
    };

    void resize(std::size_t order)
    {
        m_data.resize(Layout::size(order));
        m_shape = Layout::shape(order);
        m_order = order;
    }

    [[nodiscard]] element_type operator()(
        std::size_t i, std::size_t j, std::size_t k) const noexcept
    {
        return m_data[m_shape[2]*(m_shape[1]*i + j) + k];
    }

    [[nodiscard]] element_type& operator()(
        std::size_t i, std::size_t j, std::size_t k) noexcept
    {
        return m_data[m_shape[2]*(m_shape[1]*i + j) + k];
    }
private:
    using Allocator
        = AlignedAllocator<element_type, typename LayoutType::Alignment>;

    std::vector<element_type, Allocator> m_data{};
    std::array<std::size_t, 3> m_shape{};
    std::size_t m_order{};
};

template <typename T>
concept ball_glq_grid
    = std::same_as<
        std::remove_cvref_t<T>,
        BallGLQGridSpan<
            typename std::remove_cvref_t<T>::element_type,
            typename std::remove_cvref_t<T>::Layout>>
    || std::same_as<
        std::remove_cvref_t<T>,
        BallGLQGrid<
            typename std::remove_cvref_t<T>::element_type,
            typename std::remove_cvref_t<T>::Layout>>;

/**
    @brief Points defining a grid in spherical coordinates in the unit ball.
*/
template <typename LayoutType = DefaultLayout>
class BallGLQGridPoints
{
public:
    using GridLayout = LayoutType;
    BallGLQGridPoints() = default;
    BallGLQGridPoints(std::size_t order) { resize(order); }

    void resize(std::size_t order)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr std::size_t lat_axis = GridLayout::lat_axis;
        constexpr std::size_t rad_axis = GridLayout::rad_axis;
        const auto shape = GridLayout::shape(order);
        resize(shape[lon_axis], shape[lat_axis], shape[rad_axis]);
    }

    [[nodiscard]] std::span<const double> longitudes() const noexcept
    {
        return m_longitudes;
    }

    [[nodiscard]] std::span<const double> rad_glq_nodes() const noexcept
    {
        return m_rad_glq_nodes;
    }
    
    [[nodiscard]] std::span<const double> lat_glq_nodes() const noexcept
    {
        return m_lat_glq_nodes;
    }

    template <ball_glq_grid GridType, typename FuncType>
        requires std::same_as<
            typename std::remove_cvref_t<GridType>::Layout, GridLayout>
    void generate_values(GridType&& grid, FuncType&& f)
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
                        grid(i, j, k) = f(r, lon, colatitude);
                    }
                }
            }
        }
    }

    template <typename FuncType>
    [[nodiscard]] auto generate_values(FuncType&& f, std::size_t order)
    {
        using CodomainType = std::invoke_result_t<FuncType, double, double, double>;
        BallGLQGrid<CodomainType, GridLayout> grid(order);
        generate_values(grid, f);
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
            gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::ANGLE>(m_lat_glq_nodes, m_lat_glq_nodes.size() & 1);
        }
        if (num_rad != m_rad_glq_nodes.size())
        {
            m_rad_glq_nodes.resize(num_rad);
            gl::gl_nodes<gl::UnpackedLayout, gl::GLNodeStyle::COS>(m_rad_glq_nodes, m_rad_glq_nodes.size() & 1);
            for (auto& node : m_rad_glq_nodes)
                node = 0.5*(1.0 + node);
        }
    }

    std::vector<double> m_rad_glq_nodes{};
    std::vector<double> m_lat_glq_nodes{};
    std::vector<double> m_longitudes{};
};

/**
    @brief Class for transforming between a Gauss-Legendre quadrature grid representation and Zernike polynomial expansion representation of data in the unit baal.

    @tparam ZERNIKE_NORM normalization convention of Zernike functions
    @tparam SH_NORM normalization convention of spherical harmonics
    @tparam PHASE phase convention of spherical harmonics
    @tparam GridLayoutType
*/
template <ZernikeNorm ZERNIKE_NORM, st::SHNorm SH_NORM, st::SHPhase PHASE, typename GridLayoutType = DefaultLayout>
class GLQTransformer
{
public:
    using GridLayout = GridLayoutType;

    static constexpr ZernikeNorm zernike_norm = ZERNIKE_NORM;
    static constexpr st::SHNorm sh_norm = SH_NORM;
    static constexpr st::SHPhase phase = PHASE;

    GLQTransformer():
        m_pocketfft_shape_grid(3), m_pocketfft_stride_grid(3), 
        m_pocketfft_stride_fft(3) {};
    explicit GLQTransformer(std::size_t order):
        m_zernike_recursion(order), m_plm_recursion(order),
        m_rad_glq_nodes(GridLayout::rad_size(order)),
        m_rad_glq_weights(GridLayout::rad_size(order)),
        m_lat_glq_nodes(GridLayout::lat_size(order)),
        m_lat_glq_weights(GridLayout::lat_size(order)),
        m_zernike_grid(GridLayout::rad_size(order)*RadialZernikeLayout::size(order)),
        m_plm_grid(GridLayout::lat_size(order)*TriangleLayout::size(order)),
        m_flm_grid(GridLayout::rad_size(order)*TriangleLayout::size(order)),
        m_ffts(GridLayout::rad_size(order)*GridLayout::lat_size(order)*GridLayout::fft_size(order)), m_pocketfft_shape_grid(3),
        m_pocketfft_stride_grid(3), m_pocketfft_stride_fft(3), m_order(order)
    {
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::COS>(
                m_rad_glq_nodes, m_rad_glq_weights,
                m_rad_glq_weights.size() & 1);
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::COS>(
                m_lat_glq_nodes, m_lat_glq_weights,
                m_lat_glq_weights.size() & 1);
        
        for (auto& node : m_rad_glq_nodes)
            node = 0.5*(1.0 + node);
        
        RadialZernikeVecSpan<ZERNIKE_NORM, double> zernike(
                m_zernike_grid, order, m_rad_glq_nodes.size());
        m_zernike_recursion.zernike<ZERNIKE_NORM>(
                m_rad_glq_nodes, zernike);

        st::PlmVecSpan<double, SH_NORM, PHASE> plm(
                m_plm_grid, order, m_lat_glq_nodes.size());
        m_plm_recursion.plm_real(m_lat_glq_nodes, plm);

        auto shape = GridLayout::shape(order);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];
        m_pocketfft_shape_grid[2] = shape[2];

        m_pocketfft_stride_grid[0] = long(shape[1]*shape[2]*sizeof(double));
        m_pocketfft_stride_grid[1] = long(shape[2]*sizeof(double));
        m_pocketfft_stride_grid[2] = sizeof(double);

        auto fft_stride = GridLayout::fft_stride(order);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = long(fft_stride[1]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[2] = fft_stride[2]*sizeof(std::complex<double>);
    }

    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    void resize(std::size_t order)
    {
        if (order == m_order) return;

        m_plm_recursion.expand(order);
        m_zernike_recursion.expand(order);

        m_rad_glq_nodes.resize(GridLayout::rad_size(order));
        m_rad_glq_weights.resize(GridLayout::rad_size(order));
        m_lat_glq_nodes.resize(GridLayout::lat_size(order));
        m_lat_glq_weights.resize(GridLayout::lat_size(order));

        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::COS>(
                m_rad_glq_nodes, m_rad_glq_weights,
                m_rad_glq_weights.size() & 1);
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::COS>(
                m_lat_glq_nodes, m_lat_glq_weights,
                m_lat_glq_weights.size() & 1);
        
        for (auto& node : m_rad_glq_nodes)
            node = 0.5*(1.0 + node);
        
        m_zernike_grid.resize(GridLayout::rad_size(order)*RadialZernikeLayout::size(order));
        
        RadialZernikeVecSpan<ZERNIKE_NORM, double> zernike(
                m_zernike_grid, order, m_rad_glq_nodes.size());
        m_zernike_recursion.zernike<ZERNIKE_NORM>(
                m_rad_glq_nodes, zernike);
        
        m_plm_grid.resize(GridLayout::lat_size(order)*TriangleLayout::size(order));
        m_flm_grid.resize(GridLayout::rad_size(order)*TriangleLayout::size(order));

        st::PlmVecSpan<double, SH_NORM, PHASE> plm(
                m_plm_grid, order, m_lat_glq_nodes.size());
        m_plm_recursion.plm_real(m_lat_glq_nodes, plm);

        m_ffts.resize(GridLayout::rad_size(order)*GridLayout::lat_size(order)*GridLayout::fft_size(order));
        std::array<std::size_t, 3> shape = GridLayout::shape(order);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];
        m_pocketfft_shape_grid[2] = shape[2];

        m_pocketfft_stride_grid[0] = long(shape[1]*shape[2]*sizeof(double));
        m_pocketfft_stride_grid[1] = long(shape[2]*sizeof(double));
        m_pocketfft_stride_grid[2] = sizeof(double);

        std::array<std::size_t, 3> fft_stride = GridLayout::fft_stride(order);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = long(fft_stride[1]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[2] = fft_stride[2]*sizeof(std::complex<double>);

        m_order = order;
    }

    void forward_transform(
        BallGLQGridSpan<const double, GridLayout> values,
        ZernikeExpansionSpan<std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion)
    {
        resize(values.order());

        integrate_longitudinal(values);
        apply_weights();

        std::size_t min_order = std::min(expansion.order(), values.order());
        integrate_latitudinal(min_order);
        integrate_radial(expansion, min_order);
    }

    void backward_transform(
        ZernikeExpansionSpan<const std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion,
        BallGLQGridSpan<double, GridLayout> values)
    {
        resize(values.order());

        std::size_t min_order = std::min(expansion.order(), values.order());
        
        sum_n(expansion, min_order);
        sum_l(min_order);
        sum_m(values);
    }
    
    [[nodiscard]] ZernikeExpansion<ZERNIKE_NORM, SH_NORM, PHASE> forward_transform(
        BallGLQGridSpan<const double, GridLayout> values, std::size_t order)
    {
        ZernikeExpansion<ZERNIKE_NORM, SH_NORM, PHASE> expansion(order);
        forward_transform(values, expansion);
        return expansion;
    }

    [[nodiscard]] BallGLQGrid<double, GridLayout> backward_transform(
        ZernikeExpansionSpan<const std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion, std::size_t order)
    {
        BallGLQGrid<double, GridLayout> grid(order);
        backward_transform(expansion, grid);
        return grid;
    }

private:
    void integrate_longitudinal(
        BallGLQGridSpan<const double, GridLayout> values)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr double radial_integral_norm = 0.5;
        constexpr double sh_norm = st::normalization<SH_NORM>();
        const double fourier_norm = (2.0*std::numbers::pi)/double(values.shape()[lon_axis]);
        const double prefactor = sh_norm*radial_integral_norm*fourier_norm;
        pocketfft::r2c(
            m_pocketfft_shape_grid, m_pocketfft_stride_grid, m_pocketfft_stride_fft, lon_axis, pocketfft::FORWARD, values.flatten().data(), m_ffts.data(), prefactor);
    }

    void apply_weights() noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        const std::size_t lat_glq_size = m_lat_glq_weights.size();
        const std::size_t fft_order = GridLayout::fft_size(m_order);

        MDSpan<std::complex<double>, 3> fft(
                m_ffts.data(), {fft_order, lat_glq_size, rad_glq_size});

        if constexpr (std::same_as<GridLayout, LonLatRadLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t m = 0; m < fft_order; ++m)
            {
                MDSpan<std::complex<double>, 2> fft_m = fft[m];
                for (std::size_t i = 0; i < lat_glq_size; ++i)
                {
                    MDSpan<std::complex<double>, 1> fft_mi = fft_m[i];
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
        const std::size_t fft_order = GridLayout::fft_size(m_order);
        std::ranges::fill(m_flm_grid, std::array<double, 2>{});

        TriangleVecSpan<std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, m_order, rad_glq_size);

        st::PlmVecSpan<const double, SH_NORM, PHASE> ass_leg(
                m_plm_grid, m_order, m_lat_glq_nodes.size());

        MDSpan<const std::complex<double>, 3> fft(
                m_ffts.data(), {fft_order, lat_glq_size, rad_glq_size});
        if constexpr (std::same_as<GridLayout, LonLatRadLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t l = 0; l < min_order; ++l)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::span<std::array<double, 2>> flm_lm = flm(l,m);
                    std::span<const double> ass_leg_lm = ass_leg(l,m);
                    MDSpan<const std::complex<double>, 2> fft_m = fft[m];
                    for (std::size_t i = 0; i < lat_glq_size; ++i)
                    {
                        const double ass_leg_lmi = ass_leg_lm[i];
                        MDSpan<const std::complex<double>, 1> fft_mi = fft_m[i];
                        for (std::size_t j = 0; j < rad_glq_size; ++j)
                        {
                            flm_lm[j][0] += ass_leg_lmi*fft_mi[j].real();
                            flm_lm[j][1] += ass_leg_lmi*fft_mi[j].imag();
                        }
                    }
                }
            }
        }
    }

    void integrate_radial(
        ZernikeExpansionSpan<std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion, std::size_t min_order) noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        std::ranges::fill(expansion.flatten(), std::array<double, 2>{});

        TriangleVecSpan<const std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, m_order, rad_glq_size);

        RadialZernikeVecSpan<ZERNIKE_NORM, const double> zernike(
                m_zernike_grid, m_order, m_rad_glq_nodes.size());
        if constexpr (std::same_as<GridLayout, LonLatRadLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t n = 0; n < min_order; ++n)
            {
                const double norm = normalization<ZERNIKE_NORM>(n);
                ZernikeExpansionSHSpan<std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> 
                expansion_n = expansion[n];
                
                for (std::size_t l = n & 1; l <= n; l += 2)
                {
                    std::span<std::array<double, 2>> expansion_nl
                        = expansion_n[l];
                    
                    std::span<const double> zernike_nl = zernike(n,l);
                    for (std::size_t m = 0; m <= l; ++m)
                    {
                        std::span<const std::array<double, 2>>
                        flm_lm = flm(l,m);

                        std::array<double, 2>& coeff = expansion_nl[m];
                        for (std::size_t i = 0; i < rad_glq_size; ++i)
                        {
                            coeff[0] += zernike_nl[i]*flm_lm[i][0];
                            coeff[1] += zernike_nl[i]*flm_lm[i][1];
                        }

                        if constexpr (ZERNIKE_NORM == ZernikeNorm::UNNORMED)
                        {
                            coeff[0] *= norm;
                            coeff[1] *= norm;
                        }
                    }
                }
            }
        }
    }

    void sum_n(
        ZernikeExpansionSpan<const std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion, 
        std::size_t min_order) noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        std::ranges::fill(m_flm_grid, std::array<double, 2>{});

        RadialZernikeVecSpan<ZERNIKE_NORM, const double> zernike(
                m_zernike_grid, m_order, m_rad_glq_nodes.size());

        TriangleVecSpan<std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, m_order, rad_glq_size);
        for (std::size_t n = 0; n < min_order; ++n)
        {
            auto expansion_n = expansion[n];
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                std::span<const double> zernike_nl = zernike(n,l);
                auto expansion_nl = expansion_n[l];
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::span<std::array<double, 2>> flm_lm = flm(l,m);
                    const std::array<double, 2> coeff = expansion_nl[m];
                    for (std::size_t i = 0; i < rad_glq_size; ++i)
                    {
                        flm_lm[i][0] += zernike_nl[i]*coeff[0];
                        flm_lm[i][1] += zernike_nl[i]*coeff[1];
                    }
                }
            }
        }
    }

    void sum_l(std::size_t min_order) noexcept
    {
        const std::size_t lat_glq_size = m_lat_glq_weights.size();
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        const std::size_t fft_order = GridLayout::fft_size(m_order);

        TriangleVecSpan<const std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, m_order, rad_glq_size);
        
        st::PlmVecSpan<const double, SH_NORM, PHASE> ass_leg(
                m_plm_grid, m_order, m_lat_glq_nodes.size());

        std::ranges::fill(m_ffts, std::complex<double>{});
        MDSpan<std::complex<double>, 3> fft(
                m_ffts.data(), {fft_order, lat_glq_size, rad_glq_size});
        for (std::size_t l = 0; l < min_order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::span<const std::array<double, 2>> flm_lm = flm(l,m);
                std::span<const double> plm = ass_leg(l,m);
                const double m_factor = (m > 0) ? 0.5 : 1.0;
                
                MDSpan<std::complex<double>, 2> fft_m = fft[m];
                for (std::size_t i = 0; i < lat_glq_size; ++i)
                {
                    const double plm_i = plm[i];
                    const double weight = m_factor*plm_i;
                    MDSpan<std::complex<double>, 1> fft_mi = fft_m[i];
                    for (std::size_t j = 0; j < rad_glq_size; ++j)
                    {
                        fft_mi[j] += std::complex<double>{
                            weight*flm_lm[j][0], -weight*flm_lm[j][1]
                        };
                    }
                }
            }
        }
    }

    void sum_m(BallGLQGridSpan<double, GridLayout> values)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr double prefactor = 1.0;
        pocketfft::c2r(
            m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid, lon_axis, pocketfft::BACKWARD, m_ffts.data(), values.flatten().data(), prefactor);
    }

    RadialZernikeRecursion m_zernike_recursion{};
    st::PlmRecursion m_plm_recursion{};
    std::vector<double> m_rad_glq_nodes{};
    std::vector<double> m_rad_glq_weights{};
    std::vector<double> m_lat_glq_nodes{};
    std::vector<double> m_lat_glq_weights{};
    std::vector<double> m_zernike_grid{};
    std::vector<double> m_plm_grid{};
    std::vector<std::array<double, 2>> m_flm_grid{};
    std::vector<std::complex<double>> m_ffts{};
    std::vector<std::size_t> m_pocketfft_shape_grid{};
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid{};
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft{};
    std::size_t m_order{};
};

/**
    @brief Convenient alias for `GLQTransformer` with unnormalized Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerAcoustics
    = GLQTransformer<ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::NONE, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with orthonorml Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerOrthoAcoustics
    = GLQTransformer<ZernikeNorm::NORMED, st::SHNorm::QM, st::SHPhase::NONE, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with unnormalized Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerQM
    = GLQTransformer<ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::CS, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with orthonormal Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerOrthoQM
    = GLQTransformer<ZernikeNorm::NORMED, st::SHNorm::QM, st::SHPhase::CS, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with unnormalized Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerGeo
    = GLQTransformer<ZernikeNorm::UNNORMED, st::SHNorm::GEO, st::SHPhase::NONE, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with orthonormal Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerOrthoGeo
    = GLQTransformer<ZernikeNorm::NORMED, st::SHNorm::GEO, st::SHPhase::NONE, GridLayout>;

/**
    @brief Function concept taking Cartesian coordinates as inputs.
*/
template <typename Func>
concept cartesian_function = requires (Func f, std::array<double, 3> x)
{
    { f(x) } -> std::same_as<double>;
};

/**
    @brief Function concept taking spherical coordinates as inputs.
*/
template <typename Func>
concept spherical_function = requires (Func f, double r, double lon, double colat)
{
    { f(r, lon, colat) } -> std::same_as<double>;
};

/**
    @brief High-level interface for taking Zernike transforms of functions on balls of arbitrary radii.

    @tparam ZERNIKE_NORM normalization convention of Zernike functions
    @tparam SH_NORM normalization convention of spherical harmonics
    @tparam PHASE phase convention of spherical harmonics
    @tparam GridLayoutType
*/
template <ZernikeNorm ZERNIKE_NORM, st::SHNorm SH_NORM, st::SHPhase PHASE, typename GridLayoutType = DefaultLayout>
class ZernikeTransformer
{
public:
    using GridLayout = GridLayoutType;
    ZernikeTransformer() = default;
    explicit ZernikeTransformer(std::size_t order):
        m_grid(order), m_points(order), m_transformer(order) {}

    void resize(std::size_t order)
    {
        m_points.resize(order);
        m_grid.resize(order);
        m_transformer.resize(order);
    }

    template <spherical_function Func>
    void transform(
        Func&& f, double radius,
        ZernikeExpansionSpan<std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            return f(r*radius, lon, colat);
        };
        resize(expansion.order());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.forward_transform(m_grid, expansion);
    }

    template <spherical_function Func>
    [[nodiscard]] ZernikeExpansion<ZERNIKE_NORM, SH_NORM, PHASE> transform(
        Func&& f, double radius, std::size_t order)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            return f(r*radius, lon, colat);
        };
        resize(order);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.forward_transform(m_grid, order);
    }

    template <cartesian_function Func>
    void transform(
        Func&& f, double radius,
        ZernikeExpansionSpan<std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            const double rad = r*radius;
            const double scolat = std::sin(colat);
            const std::array<double, 3> x = {
                rad*scolat*std::cos(lon), rad*scolat*std::sin(lon),
                rad*std::cos(colat)
            };
            return f(x);
        };
        resize(expansion.order());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.forward_transform(m_grid, expansion);
    }

    template <cartesian_function Func>
    [[nodiscard]] ZernikeExpansion<ZERNIKE_NORM, SH_NORM, PHASE> transform(
        Func&& f, double radius, std::size_t order)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            const double rad = r*radius;
            const double scolat = std::sin(colat);
            const std::array<double, 3> x = {
                rad*scolat*std::cos(lon), rad*scolat*std::sin(lon),
                rad*std::cos(colat)
            };
            return f(x);
        };
        resize(order);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.forward_transform(m_grid, order);
    }

private:
    BallGLQGrid<double, GridLayout> m_grid;
    BallGLQGridPoints<GridLayout> m_points;
    GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE, GridLayout> m_transformer;
};

/**
    @brief Convenient alias for `ZernikeTransformer` with unnormalized Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerAcoustics
    = ZernikeTransformer<ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::NONE, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with orthonormal Zernike functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerOrthoAcoustics
    = ZernikeTransformer<ZernikeNorm::NORMED, st::SHNorm::QM, st::SHPhase::NONE, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with unnormalized Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerQM
    = ZernikeTransformer<ZernikeNorm::UNNORMED, st::SHNorm::QM, st::SHPhase::CS, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with orthonormal Zernike functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerOrthoQM
    = ZernikeTransformer<ZernikeNorm::NORMED, st::SHNorm::QM, st::SHPhase::CS, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with unnormalized Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerGeo
    = ZernikeTransformer<ZernikeNorm::UNNORMED, st::SHNorm::GEO, st::SHPhase::NONE, GridLayout>;

/**
    @brief Convenient alias for `ZernikeTransformer` with orthonormal Zernike functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using ZernikeTransformerOrthoGeo
    = ZernikeTransformer<ZernikeNorm::NORMED, st::SHNorm::GEO, st::SHPhase::NONE, GridLayout>;

} // namespace zt
} // namespace zest