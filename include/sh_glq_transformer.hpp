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

#include <span>
#include <vector>
#include <complex>
#include <cassert>

#include "pocketfft_spec.hpp"

#include "alignment.hpp"
#include "plm_recursion.hpp"
#include "gauss_legendre.hpp"
#include "md_span.hpp"

namespace zest
{
namespace st
{

/**
    @brief Longitudinally contiguous layout for storing a Gauss-Legendre quadrature grid.

    @tparam AlignmentType byte alignment of the grid
*/
template <typename AlignmentType = CacheLineAlignment>
struct LatLonLayout
{
    using Alignment = AlignmentType;

    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return lat_size(order)*lon_size(order);
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    shape(std::size_t order) noexcept
    {
        return {lat_size(order), lon_size(order)};
    }

    [[nodiscard]] static constexpr std::size_t
    fft_size(std::size_t order) noexcept
    {
        return (lon_size(order) >> 1) + 1;
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    fft_stride(std::size_t order) noexcept
    {
        return {fft_size(order), 1};
    }

    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t order) noexcept
    {
        return order;
    }

    [[nodiscard]] static constexpr std::size_t
    lon_size(std::size_t order) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = 2UL*order - std::min(1UL, order);
        if constexpr (std::is_same_v<Alignment, NoAlignment>)
            return min_size;
        else
            return detail::next_divisible<vector_size>(min_size);
    }

    static constexpr std::size_t lat_axis = 0UL;
    static constexpr std::size_t lon_axis = 1UL;
};

/**
    @brief Latitudinally contiguous layout for storing a Gauss-Legendre quadrature grid.

    @tparam AlignmentType byte alignment of the grid
*/
template <typename AlignmentType = CacheLineAlignment>
struct LonLatLayout
{
    using Alignment = AlignmentType;

    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return lat_size(order)*lon_size(order);
    }
    
    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    shape(std::size_t order) noexcept
    {
        return {lon_size(order), lat_size(order)};
    }

    [[nodiscard]] static constexpr std::size_t
    fft_size(std::size_t order) noexcept
    {
        return (lon_size(order) >> 1) + 1;
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    fft_stride(std::size_t order)
    {
        return {lat_size(order), 1};
    }

    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t order) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = order;
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
};

using DefaultLayout = LonLatLayout<>;

/**
    @brief A non-owning view on data modeling a Gauss-Legendre quadrature grid on the sphere.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
class SphereGLQGridSpan: public MDSpan<ElementType, 2>
{
public:
    using typename MDSpan<ElementType, 2>::element_type;
    using Layout = LayoutType;
    using ConstView = SphereGLQGridSpan<const element_type, Layout>;

    using MDSpan<ElementType, 2>::extents;
    using MDSpan<ElementType, 2>::data;
    using MDSpan<ElementType, 2>::size;

    [[nodiscard]] static constexpr std::size_t size(std::size_t order) noexcept
    {
        return Layout::size(order);
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    shape(std::size_t order) noexcept
    {
        return Layout::shape(order);
    }

    constexpr SphereGLQGridSpan() noexcept = default;
    constexpr SphereGLQGridSpan(element_type* data, std::size_t order) noexcept:
        MDSpan<ElementType, 2>(data, Layout::shape(order)), m_order(order) {}
    constexpr SphereGLQGridSpan(
        std::span<element_type> buffer, std::size_t order) noexcept:
        MDSpan<ElementType, 2>(buffer.data(), Layout::shape(order)),
        m_order(order) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }
    
    [[nodiscard]] constexpr const std::array<std::size_t, 2>&
    shape() const noexcept { return extents(); }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span<element_type>(data(), size()); }

    [[nodiscard]] constexpr operator ConstView() const noexcept
    {
        return ConstView(data(), size(), extents(), m_order);
    }

protected:
    friend SphereGLQGridSpan<std::remove_const_t<element_type>, Layout>;

    constexpr SphereGLQGridSpan(
        element_type* data, std::size_t size, const std::array<std::size_t, 2>& extents, std::size_t order) noexcept:
        MDSpan<ElementType, 2>(data, size, extents), m_order(order) {}

private:
    std::size_t m_order{};
};

/**
    @brief Container for Gauss-Legendre quadrature gridded data on the sphere.

    @tparam ElementType type of elements in the grid
    @tparam LayoutType grid layout
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
class SphereGLQGrid
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cvref_t<element_type>;
    using Layout = LayoutType;
    using View = SphereGLQGridSpan<element_type, Layout>;
    using ConstView = SphereGLQGridSpan<const element_type, Layout>;

    SphereGLQGrid() = default;
    explicit SphereGLQGrid(std::size_t order):
        m_values(Layout::size(order)), m_shape(Layout::shape(order)), 
        m_order(order) {}

    [[nodiscard]] std::array<std::size_t, 2>
    shape() const noexcept { return m_shape; }

    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    [[nodiscard]] std::span<const element_type>
    flatten() const noexcept { return m_values; }

    std::span<element_type> flatten() noexcept { return m_values; }

    [[nodiscard]] operator View() noexcept
    {
        return View(m_values, m_order);
    };

    [[nodiscard]] operator ConstView() const noexcept
    {
        return ConstView(m_values, m_order);
    };

    void resize(std::size_t order)
    {
        m_values.resize(Layout::size(order));
        m_shape = Layout::shape(order);
        m_order = order;
    }

    [[nodiscard]] element_type
    operator()(std::size_t i, std::size_t j) const noexcept
    {
        return m_values[m_shape[1]*i + j];
    }

    [[nodiscard]] element_type& operator()(std::size_t i, std::size_t j) noexcept
    {
        return m_values[m_shape[1]*i + j];
    }

private:
    using Allocator
        = AlignedAllocator<element_type, typename LayoutType::Alignment>;

    std::vector<element_type, Allocator> m_values{};
    std::array<std::size_t, 2> m_shape{};
    std::size_t m_order{};
};

template <typename T>
concept sphere_glq_grid
    = std::same_as<
        std::remove_cvref_t<T>,
        SphereGLQGridSpan<
            typename std::remove_cvref_t<T>::element_type,
            typename std::remove_cvref_t<T>::Layout>>
    || std::same_as<
        std::remove_cvref_t<T>,
        SphereGLQGrid<
            typename std::remove_cvref_t<T>::element_type,
            typename std::remove_cvref_t<T>::Layout>>;

/**
    @brief Points defining a Gauss-Legendre quadrature grid on the sphere.
*/
template <typename LayoutType = DefaultLayout>
class SphereGLQGridPoints
{
public:
    using GridLayout = LayoutType;
    SphereGLQGridPoints() = default;
    explicit SphereGLQGridPoints(std::size_t order) { resize(order); }

    void resize(std::size_t order)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr std::size_t lat_axis = GridLayout::lat_axis;
        const auto shape = GridLayout::shape(order);
        resize(shape[lon_axis], shape[lat_axis]);
    }

    [[nodiscard]] std::array<std::size_t, 2> shape() noexcept
    {
        return {m_glq_nodes.size(), m_longitudes.size()};
    }

    [[nodiscard]] std::span<const double> longitudes() const noexcept
    {
        return m_longitudes;
    }
    [[nodiscard]] std::span<const double> glq_nodes() const noexcept
    {
        return m_glq_nodes;
    }

    template <sphere_glq_grid GridType, typename FuncType>
        requires std::same_as<
            typename std::remove_cvref_t<GridType>::Layout, GridLayout>
    void generate_values(GridType&& grid, FuncType&& f)
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
                    grid(i,j) = f(lon, colatitude);
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
                    grid(i,j) = f(lon, colatitude);
                }
            }
        }
    }

    template <typename FuncType>
    auto generate_values(FuncType&& f, std::size_t order)
    {
        using CodomainType = std::invoke_result_t<FuncType, double, double>;
        SphereGLQGrid<CodomainType, GridLayout> grid(order);
        generate_values(grid, f);
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

    std::vector<double> m_longitudes{};
    std::vector<double> m_glq_nodes{};
};

/**
    @brief Transformations between a Gauss-Legendre quadrature grid representation and spherical harmonic expansion representation of real data.

    @tparam NORM normalization convention of spherical harmonics
    @tparam sh_phase_param phase convention of spherical harmonics
    @tparam GridLayoutType
*/
template <
    SHNorm sh_norm_param, SHPhase sh_phase_param,
    typename GridLayoutType = DefaultLayout>
class GLQTransformer
{
public:
    using GridLayout = GridLayoutType;

    static constexpr SHNorm norm = sh_norm_param;
    static constexpr SHPhase phase = sh_phase_param;

    GLQTransformer(): 
        m_pocketfft_shape_grid(2),
        m_pocketfft_stride_grid(2), 
        m_pocketfft_stride_fft(2) {}
    explicit GLQTransformer(std::size_t order):
        m_recursion(order),
        m_glq_nodes(gl::PackedLayout::size(GridLayout::lat_size(order))),
        m_glq_weights(gl::PackedLayout::size(GridLayout::lat_size(order))),
        m_plm_grid(GridLayout::lat_size(order)*TriangleLayout::size(order)),
        m_ffts(GridLayout::lat_size(order)*GridLayout::fft_size(order)), m_symm_asymm(GridLayout::fft_size(order)*((GridLayout::lat_size(order) + 1) >> 1)*2),
        m_pocketfft_shape_grid(2),
        m_pocketfft_stride_grid(2),
        m_pocketfft_stride_fft(2),
        m_order(order)
    {
        gl::gl_nodes_and_weights<gl::PackedLayout, gl::GLNodeStyle::cos>(
                m_glq_nodes, m_glq_weights, GridLayout::lat_size(order) & 1);
        
        if constexpr (std::same_as<GridLayout, LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            {
                const double z = m_glq_nodes[i];
                PlmSpan<double, sh_norm_param, sh_phase_param> plm(
                        m_plm_grid.data() + i*TriangleLayout::size(order), 
                        order);
                m_recursion.plm_real(z, plm);
            }
        }
        else if constexpr (std::same_as<GridLayout, LonLatLayout<typename GridLayout::Alignment>>)
        {
            PlmVecSpan<double, sh_norm_param, sh_phase_param> plm(m_plm_grid, order, m_glq_nodes.size());
            m_recursion.plm_real(m_glq_nodes, plm);
        }

        auto shape = GridLayout::shape(order);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];

        m_pocketfft_stride_grid[0] = long(shape[1]*sizeof(double));
        m_pocketfft_stride_grid[1] = sizeof(double);

        auto fft_stride = GridLayout::fft_stride(order);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = fft_stride[1]*sizeof(std::complex<double>);
    }

    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    /**
        @brief Resize transformer for specified expansion order

        @param order
    */
    void resize(std::size_t order)
    {
        if (order == m_order) return;

        m_recursion.expand(order);

        m_glq_nodes.resize(gl::PackedLayout::size(GridLayout::lat_size(order)));
        m_glq_weights.resize(gl::PackedLayout::size(GridLayout::lat_size(order)));
        gl::gl_nodes_and_weights<gl::PackedLayout, gl::GLNodeStyle::cos>(
                m_glq_nodes, m_glq_weights, GridLayout::lat_size(order) & 1);
        m_plm_grid.resize(m_glq_weights.size()*TriangleLayout::size(order));
        
        if constexpr (std::same_as<GridLayout, LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            {
                const double z = m_glq_nodes[i];
                PlmSpan<double, sh_norm_param, sh_phase_param> plm(
                        m_plm_grid.data() + i*TriangleLayout::size(order), 
                        order);
                m_recursion.plm_real(z, plm);
            }
        }
        else if constexpr (std::same_as<GridLayout, LonLatLayout<typename GridLayout::Alignment>>)
        {
            PlmVecSpan<double, sh_norm_param, sh_phase_param> plm(m_plm_grid, order, m_glq_nodes.size());
            m_recursion.plm_real(m_glq_nodes, plm);
        }

        m_ffts.resize(GridLayout::lat_size(order)*GridLayout::fft_size(order));
        m_symm_asymm.resize(GridLayout::fft_size(order)*((GridLayout::lat_size(order) + 1) >> 1)*2);

        auto shape = GridLayout::shape(order);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];

        m_pocketfft_stride_grid[0] = long(shape[1]*sizeof(double));
        m_pocketfft_stride_grid[1] = sizeof(double);

        auto fft_stride = GridLayout::fft_stride(order);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = fft_stride[1]*sizeof(std::complex<double>);

        m_order = order;
    }

    /**
        @brief Forward transform from Gauss-Legendre quadrature grid to spherical harmonic coefficients.

        @param values values on the spherical quadrature grid
        @param expansion coefficients of the expansion
    */
    void forward_transform(
        SphereGLQGridSpan<const double, GridLayout> values,
        RealSHExpansionSpan<std::array<double, 2>, sh_norm_param, sh_phase_param> expansion)
    {
        resize(values.order());
        
        integrate_longitudinal(values);

        fft_to_symm_asymm();

        std::size_t min_order = std::min(expansion.order(), values.order());
        integrate_latitudinal(expansion, min_order);
    }

    /**
        @brief Backward transform from spherical harmonic expansion to Gauss-Legendre quadrature grid.

        @param expansion coefficients of the expansion
        @param values values on the spherical quadrature grid
    */
    void backward_transform(
        RealSHExpansionSpan<const std::array<double, 2>, sh_norm_param, sh_phase_param> expansion,
        SphereGLQGridSpan<double, GridLayout> values)
    {
        resize(values.order());

        std::size_t min_order = std::min(expansion.order(), values.order());
        
        sum_l(expansion, min_order);
        symm_asymm_to_fft();
        sum_m(values);
    }

    /**
        @brief Backward transform from spherical harmonic expansion of even or odd parity to Gauss-Legendre quadrature grid.

        @tparam Expansion type of expansion

        @param expansion coefficients of the expansion
        @param values values on the spherical quadrature grid

        @note A spherical harmonic expansion has even/odd parity if the first index of all nonzero coefficients has even/odd parity.
    */
    template <even_odd_real_sh_expansion Expansion>
        requires (std::remove_cvref_t<Expansion>::norm == sh_norm_param)
        && (std::remove_cvref_t<Expansion>::phase == sh_phase_param)
        && std::same_as<
            typename std::remove_cvref_t<Expansion>::value_type, 
            std::array<double, 2>>
        && has_parity<Expansion>
    void backward_transform(
        Expansion&& expansion, SphereGLQGridSpan<double, GridLayout> values)
    {
        resize(values.order());

        std::size_t min_order = std::min(expansion.order(), values.order());
        
        sum_l(std::forward<Expansion>(expansion), min_order, expansion.parity());
        symm_asymm_to_fft();
        sum_m(values);
    }

    /**
        @brief Backward transform from even/odd coefficients of a spherical harmonic expansion to Gauss-Legendre quadrature grid.

        @param values values on the spherical quadrature grid
        @param expansion coefficients of the expansion
        @param parity parity of the coefficients

        @note The parity of a spherical harmonic coefficient is determined by the parity of the first index of the coefficient.
    */
    void backward_transform(
        RealSHExpansionSpan<const std::array<double, 2>, sh_norm_param, sh_phase_param> expansion,
        SphereGLQGridSpan<double, GridLayout> values, Parity parity)
    {
        resize(values.order());

        std::size_t min_order = std::min(expansion.order(), values.order());
        
        sum_l(expansion, min_order, parity);
        symm_asymm_to_fft();
        sum_m(values);
    }
    
    /*
    Forward transform from Gauss-Legendre quadrature grid to spherical harmonic coefficients.

    Parameters:
    `values`: values on the spherical quadrature grid.
    `order`: order of expansion.
    */
    [[nodiscard]] RealSHExpansion<sh_norm_param, sh_phase_param>
    forward_transform(
        SphereGLQGridSpan<const double, GridLayout> values, std::size_t order)
    {
        RealSHExpansion<sh_norm_param, sh_phase_param> expansion(order);
        forward_transform(values, expansion);
        return expansion;
    }

    /**
        @brief Backward transform from spherical harmonic coefficients to Gauss-Legendre quadrature grid.

        @param values values on the spherical quadrature grid
        @param expansion coefficients of the expansion
    */
    [[nodiscard]] SphereGLQGrid<double, GridLayout> backward_transform(
        RealSHExpansionSpan<const std::array<double, 2>, sh_norm_param, sh_phase_param> expansion, std::size_t order)
    {
        SphereGLQGrid<double, GridLayout> grid(order);
        backward_transform(expansion, grid);
        return grid;
    }

    /**
        @brief Backward transform from spherical harmonic expansion of even or odd parity to Gauss-Legendre quadrature grid.

        @tparam Expansion type of expansion

        @param expansion coefficients of the expansion
        @param values values on the spherical quadrature grid

        @note A spherical harmonic expansion has even/odd parity if the first index of all nonzero coefficients has even/odd parity.
    */
    template <even_odd_real_sh_expansion Expansion>
        requires (std::remove_cvref_t<Expansion>::norm == sh_norm_param)
        && (std::remove_cvref_t<Expansion>::phase == sh_phase_param)
        && std::same_as<
            typename std::remove_cvref_t<Expansion>::value_type, 
            std::array<double, 2>>
        && has_parity<Expansion>
    [[nodiscard]] SphereGLQGrid<double, GridLayout> backward_transform(
        Expansion&& expansion, std::size_t order)
    {
        SphereGLQGrid<double, GridLayout> grid(order);
        backward_transform(expansion, grid);
        return grid;
    }

    /**
        @brief Backward transform from even/odd coefficients of a spherical harmonic expansion to Gauss-Legendre quadrature grid.

        @param values values on the spherical quadrature grid
        @param expansion coefficients of the expansion
        @param parity parity of the coefficients

        @note The parity of a spherical harmonic coefficient is determined by the parity of the first index of the coefficient.
    */
    [[nodiscard]] SphereGLQGrid<double, GridLayout> backward_transform(
        RealSHExpansionSpan<const std::array<double, 2>, sh_norm_param, sh_phase_param> expansion, std::size_t order, Parity parity)
    {
        SphereGLQGrid<double, GridLayout> grid(order);
        backward_transform(expansion, grid, parity);
        return grid;
    }

private:
    void integrate_longitudinal(
        SphereGLQGridSpan<const double, GridLayout> values)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr double sh_normalization = normalization<sh_norm_param>();
        const double prefactor = sh_normalization*(2.0*std::numbers::pi)/double(values.shape()[lon_axis]);
        pocketfft::r2c(
            m_pocketfft_shape_grid, m_pocketfft_stride_grid, m_pocketfft_stride_fft, lon_axis, pocketfft::FORWARD, values.flatten().data(), m_ffts.data(), prefactor);
    }

    void apply_gl_weights() noexcept
    {
        const std::size_t num_lat = m_glq_weights.size();
        const std::size_t fft_order = GridLayout::fft_size(m_order);
        if constexpr (std::same_as<GridLayout, LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < num_lat; ++i)
            {
                const double weight = m_glq_weights[i];
                for (std::size_t m = 0; m < fft_order; ++m)
                {
                    std::complex<double>& x = m_ffts[fft_order*i + m];
                    x = {weight*x.real(), -weight*x.imag()};
                }
            }
        }
        else if constexpr (std::same_as<GridLayout, LonLatLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t m = 0; m < fft_order; ++m)
            {
                for (std::size_t i = 0; i < num_lat; ++i)
                {
                    const double weight = m_glq_weights[i];
                    std::complex<double>& x = m_ffts[num_lat*m + i];
                    x = {weight*x.real(), -weight*x.imag()};
                }
            }
        }
    }

    /*
    Apply Gauss-Legendre weights and divide Fourier transforms into symmetric and antisymmetric parts, `f(x) + f(-x)` and `f(x) - f(-x)`, where `x` are the Legendre nodes.

    This division is useful to reduce operation count in the latitudinal integration stage due to symmetry properties of the associated legendre functions. Namely, `P_lm(-x) = (-1)^(l+m)*P_lm(x)`. This means that for `l + m` even, the integration is over the symmetric parts, and for `l + m` odd, it is over the antisymmetric parts.

    The final symmetric and antisymmetric parts have a very specific memory layout. Given number of latitudes `num_lat`, there are `(num_lat + 1)/2` symmetric and antisymmetric components. These are denoted `+` and `-` for symmetric and antisymmetric, respectively. The layout differs in an expected way depending on whether `GridLayout` is latitude or longitude major. Namely, for latitude major order `+` and `-` refer to blocks of `(num_lat + 1)/2` complex numbers, whereas for longitude major order they refer to individual complex numbers.
    
    Given `m` denoting the order of the Fourier transform, these components are then stored in memory as two alternating sequences:
    `m 0 1 2 3 4 5 6 7 8 ...`
    `s + - + - + - + - + ...`
    `a - + - + - + - + - ...`
    In the latitudinal integration step, for even `l` the sequence `s` starting with the symmetric components is chosen, and for odd `l` the sequence `a` starting with the antisymmetric components is chosen. This leads to a moderately cache efficient access pattern in the latitudinal integration step.
    */
    void fft_to_symm_asymm() noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_order);
        const std::size_t num_lat = GridLayout::lat_size(m_order);
        const std::size_t central_offset = num_lat >> 1;
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t south_offset = num_unique_nodes - 1;
        const std::size_t north_offset = central_offset;

        if constexpr (std::same_as<GridLayout, LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < num_unique_nodes; ++i)
            {
                const double weight = m_glq_weights[i];
                std::span<const std::complex<double>> fft_south(
                    m_ffts.begin() + (south_offset - i)*fft_order, fft_order);
                std::span<const std::complex<double>> fft_north(
                    m_ffts.begin() + (north_offset + i)*fft_order, fft_order);
                
                std::span<std::complex<double>> symm_asymm_i(
                    m_symm_asymm.begin() + 2*i*fft_order, fft_order);
                std::span<std::complex<double>> asymm_symm_i(
                    m_symm_asymm.begin() + (2*i + 1)*fft_order, fft_order);

                for (std::size_t m = 0; m < fft_order; ++m)
                {
                    const double sign = (m & 1) ? -1.0 : 1.0;
                    const std::complex<double> south = fft_south[m];
                    const std::complex<double> north = fft_north[m];
                    const std::complex<double> south_weighted = {
                        weight*south.real(), -weight*south.imag()
                    };
                    const std::complex<double> north_weighted = {
                        weight*north.real(), -weight*north.imag()
                    };
                    symm_asymm_i[m] = north_weighted + sign*south_weighted;
                    asymm_symm_i[m] = north_weighted - sign*south_weighted;
                }
            }
        }
        if constexpr (std::same_as<GridLayout, LonLatLayout<typename GridLayout::Alignment>>)
        {
            std::span<std::complex<double>> symm_asymm(
                m_symm_asymm.begin(), num_unique_nodes*fft_order);
            std::span<std::complex<double>> asymm_symm(
                m_symm_asymm.begin() + num_unique_nodes*fft_order, num_unique_nodes*fft_order);

            for (std::size_t m = 0; m < fft_order; ++m)
            {
                const double sign = (m & 1) ? -1.0 : 1.0;
                std::span<const std::complex<double>> fft_m(
                    m_ffts.begin() + num_lat*m, num_lat);

                std::span<std::complex<double>> symm_asymm_m(
                    symm_asymm.begin() + num_unique_nodes*m, num_unique_nodes);
                std::span<std::complex<double>> asymm_symm_m(
                    asymm_symm.begin() + num_unique_nodes*m, num_unique_nodes);
                    
                for (std::size_t i = 0; i < num_unique_nodes; ++i)
                {
                    const double weight = m_glq_weights[i];
                    const std::complex<double> south = fft_m[south_offset - i];
                    const std::complex<double> north = fft_m[north_offset + i];
                    const std::complex<double> south_weighted = {
                        weight*south.real(), -weight*south.imag()
                    };
                    const std::complex<double> north_weighted = {
                        weight*north.real(), -weight*north.imag()
                    };
                    symm_asymm_m[i] = north_weighted + sign*south_weighted;
                    asymm_symm_m[i] = north_weighted - sign*south_weighted;
                }
            }
        }
    }

    void integrate_latitudinal(
        RealSHExpansionSpan<std::array<double, 2>, sh_norm_param, sh_phase_param> expansion, std::size_t min_order) noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_order);
        const std::size_t num_unique_nodes = m_glq_weights.size();

        std::span coeffs = expansion.flatten();
        std::ranges::fill(coeffs, std::array<double, 2>{});
        if constexpr (std::same_as<GridLayout, LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < num_unique_nodes; ++i)
            {
                PlmSpan<double, sh_norm_param, sh_phase_param> plm(
                        m_plm_grid.data() + i*TriangleLayout::size(m_order), 
                        m_order);
                std::span plm_flat = plm.flatten();
                for (std::size_t l = 0; l < min_order; ++l)
                {
                    std::span<const double> plm_l = plm[l];
                    std::span<std::array<double, 2>> expansion_l = expansion[l];
                    std::span<const std::complex<double>> fft(
                        m_symm_asymm.begin() + (2*i + (l & 1))*fft_order, fft_order);
                    for (std::size_t m = 0; m <= l; ++m)
                    {
                        expansion_l[m][0] += plm_l[m]*fft[m].real();
                        expansion_l[m][1] += plm_l[m]*fft[m].imag();
                    }
                }
            }
        }
        else if constexpr (std::same_as<GridLayout, LonLatLayout<typename GridLayout::Alignment>>)
        {
            const std::size_t num_plm = num_unique_nodes;
            for (std::size_t l = 0; l < min_order; ++l)
            {
                std::span<const std::complex<double>> ffts;
                ffts = std::span<const std::complex<double>>(
                    m_symm_asymm.begin() + (l & 1)*num_plm*fft_order, num_plm*fft_order);
                for (std::size_t m = 0; m <= l; ++m)
                {
                    const std::size_t ind = TriangleLayout::idx(l,m);
                    std::span<const double> plm(
                        m_plm_grid.begin() + ind*num_plm, num_plm);
                    std::span<const std::complex<double>> fft(
                        ffts.begin() + m*num_plm, num_plm);
                    
                    std::array<double, 2> coeff{};
                    switch (num_plm & 3)
                    {
                        case 1:
                            coeff[0] = plm[0]*fft[0].real();
                            coeff[1] = plm[0]*fft[0].imag();
                            break;
                        case 2:
                            coeff[0] = plm[0]*fft[0].real() + plm[1]*fft[1].real();
                            coeff[1] = plm[0]*fft[0].imag() + plm[1]*fft[1].imag();
                            break;
                        case 3:
                            coeff[0] = plm[0]*fft[0].real() + plm[1]*fft[1].real() + plm[2]*fft[2].real();
                            coeff[1] = plm[0]*fft[0].imag() + plm[1]*fft[1].imag() + plm[2]*fft[2].imag();
                            break;
                    }

                    /*for (std::size_t i = 0; i < (num_lat & 3); ++i)
                    {
                        coeff[0] += plm[i]*fft[i].real();
                        coeff[1] += plm[i]*fft[i].imag();
                    }*/

                    std::array<double, 8> partial_sum{};
                    for (std::size_t i = (num_plm & 3); i < num_plm; i += 4)
                    {
                        partial_sum[0] += plm[i]*fft[i].real();
                        partial_sum[1] += plm[i]*fft[i].imag();
                        partial_sum[2] += plm[i + 1]*fft[i + 1].real();
                        partial_sum[3] += plm[i + 1]*fft[i + 1].imag();
                        partial_sum[4] += plm[i + 2]*fft[i + 2].real();
                        partial_sum[5] += plm[i + 2]*fft[i + 2].imag();
                        partial_sum[6] += plm[i + 3]*fft[i + 3].real();
                        partial_sum[7] += plm[i + 3]*fft[i + 3].imag();
                    }

                    for (std::size_t i = 0; i < 8; i += 2)
                    {
                        coeff[0] += partial_sum[i];
                        coeff[1] += partial_sum[i + 1];
                    }

                    coeffs[ind] = coeff;
                }
            }
        }
    }

    void sum_l(
        RealSHExpansionSpan<const std::array<double, 2>, sh_norm_param, sh_phase_param> expansion, std::size_t min_order) noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_order);
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t num_plm = num_unique_nodes;

        std::ranges::fill(m_symm_asymm, std::complex<double>{});

        if constexpr (std::same_as<GridLayout, LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < num_plm; ++i)
            {
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + 2*i*fft_order, 2*fft_order);
                PlmSpan<double, sh_norm_param, sh_phase_param> plm(
                        m_plm_grid.data() + i*TriangleLayout::size(m_order), 
                        m_order);
                std::span plm_flat = plm.flatten();
                for (std::size_t l = 0; l < min_order; ++l)
                {
                    std::span<const double> plm_l = plm[l];
                    std::span<const std::array<double, 2>> expansion_l = expansion[l];
                    symm_asymm[(l & 1)*fft_order] += std::complex<double>{
                        plm_l[0]*expansion_l[0][0], -plm_l[0]*expansion_l[0][1]
                    };
                    for (std::size_t m = 1; m <= l; ++m)
                    {
                        const double weight = 0.5*plm_l[m];
                        symm_asymm[(l & 1)*fft_order + m]
                            += std::complex<double>{
                                weight*expansion_l[m][0],
                                -weight*expansion_l[m][1]
                            };
                    }
                }
            }
        }
        else if constexpr (std::same_as<GridLayout, LonLatLayout<typename GridLayout::Alignment>>)
        {
            PlmVecSpan<const double, sh_norm_param, sh_phase_param> ass_leg(
                    m_plm_grid, m_order, num_plm);

            std::span coeffs = expansion.flatten();
            for (std::size_t l = 0; l < min_order; ++l)
            {
                const std::array<double, 2> coeff = expansion(l, 0);
                std::span<const double> plm = ass_leg(l, 0);
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + (l & 1)*num_plm*fft_order, num_plm);
                for (std::size_t i = 0; i < num_plm; ++i)
                {
                    symm_asymm[i] += std::complex<double>{
                        plm[i]*coeff[0], -plm[i]*coeff[1]
                    };
                }

                for (std::size_t m = 1; m <= l; ++m)
                {
                    const std::array<double, 2> coeff = expansion(l, m);
                    std::span<const double> plm = ass_leg(l, m);
                    std::span<std::complex<double>> symm_asymm(
                        m_symm_asymm.begin() + ((l & 1)*fft_order + m)*num_plm, num_plm);
                    for (std::size_t i = 0; i < num_plm; ++i)
                    {
                        const double weight = 0.5*plm[i];
                        symm_asymm[i] += std::complex<double>{
                            weight*coeff[0], -weight*coeff[1]
                        };
                    }
                }
            }
        }
    }

    template <even_odd_real_sh_expansion Expansion>
        requires (std::remove_cvref_t<Expansion>::norm == sh_norm_param)
        && (std::remove_cvref_t<Expansion>::phase == sh_phase_param)
        && std::same_as<
            typename std::remove_cvref_t<Expansion>::value_type, 
            std::array<double, 2>>
    void sum_l(
        Expansion&& expansion, std::size_t min_order, Parity parity) noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_order);
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t num_plm = num_unique_nodes;

        std::ranges::fill(m_symm_asymm, std::complex<double>{});

        if constexpr (std::same_as<GridLayout, LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < num_plm; ++i)
            {
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + 2*i*fft_order, 2*fft_order);
                PlmSpan<double, sh_norm_param, sh_phase_param> plm(
                        m_plm_grid.data() + i*TriangleLayout::size(m_order), 
                        m_order);
                std::span plm_flat = plm.flatten();
                for (std::size_t l = std::size_t(parity); l < min_order; l += 2)
                {
                    std::span<const double> plm_l = plm[l];
                    std::span<const std::array<double, 2>> expansion_l = expansion[l];
                    symm_asymm[(l & 1)*fft_order] += std::complex<double>{
                        plm_l[0]*expansion_l[0][0], -plm_l[0]*expansion_l[0][1]
                    };
                    for (std::size_t m = 1; m <= l; ++m)
                    {
                        const double weight = 0.5*plm_l[m];
                        symm_asymm[(l & 1)*fft_order + m]
                            += std::complex<double>{
                                weight*expansion_l[m][0],
                                -weight*expansion_l[m][1]
                            };
                    }
                }
            }
        }
        else if constexpr (std::same_as<GridLayout, LonLatLayout<typename GridLayout::Alignment>>)
        {
            PlmVecSpan<const double, sh_norm_param, sh_phase_param> ass_leg(
                    m_plm_grid, m_order, num_plm);

            std::span coeffs = expansion.flatten();
            for (std::size_t l = std::size_t(parity); l < min_order; l += 2)
            {
                const std::array<double, 2> coeff = expansion(l, 0);
                std::span<const double> plm = ass_leg(l, 0);
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + (l & 1)*num_plm*fft_order, num_plm);
                for (std::size_t i = 0; i < num_plm; ++i)
                {
                    symm_asymm[i] += std::complex<double>{
                        plm[i]*coeff[0], -plm[i]*coeff[1]
                    };
                }

                for (std::size_t m = 1; m <= l; ++m)
                {
                    const std::array<double, 2> coeff = expansion(l, m);
                    std::span<const double> plm = ass_leg(l, m);
                    std::span<std::complex<double>> symm_asymm(
                        m_symm_asymm.begin() + ((l & 1)*fft_order + m)*num_plm, num_plm);
                    for (std::size_t i = 0; i < num_plm; ++i)
                    {
                        const double weight = 0.5*plm[i];
                        symm_asymm[i] += std::complex<double>{
                            weight*coeff[0], -weight*coeff[1]
                        };
                    }
                }
            }
        }
    }

    // Inverse of `fft_to_symm_asymm`
    void symm_asymm_to_fft() noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_order);
        const std::size_t num_lat = GridLayout::lat_size(m_order);
        const std::size_t central_offset = num_lat >> 1;
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t south_offset = num_unique_nodes - 1;
        const std::size_t north_offset = central_offset;

        if constexpr (std::same_as<GridLayout, LatLonLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t i = 0; i < num_unique_nodes; ++i)
            {
                std::span<const std::complex<double>> symm_asymm_i(
                    m_symm_asymm.begin() + 2*i*fft_order, fft_order);
                std::span<const std::complex<double>> asymm_symm_i(
                    m_symm_asymm.begin() + (2*i + 1)*fft_order, fft_order);

                std::span<std::complex<double>> fft_south(
                    m_ffts.begin() + (south_offset - i)*fft_order, fft_order);
                std::span<std::complex<double>> fft_north(
                    m_ffts.begin() + (north_offset + i)*fft_order, fft_order);

                for (std::size_t m = 0; m < fft_order; ++m)
                {
                    const double sign = (m & 1) ? -1.0 : 1.0;
                    fft_north[m] = symm_asymm_i[m] + asymm_symm_i[m];
                    fft_south[m] = sign*(symm_asymm_i[m] - asymm_symm_i[m]);
                }
            }
        }
        if constexpr (std::same_as<GridLayout, LonLatLayout<typename GridLayout::Alignment>>)
        {
            std::span<std::complex<double>> symm_asymm(
                m_symm_asymm.begin(), num_unique_nodes*fft_order);
            std::span<std::complex<double>> asymm_symm(
                m_symm_asymm.begin() + num_unique_nodes*fft_order, num_unique_nodes*fft_order);

            for (std::size_t m = 0; m < fft_order; ++m)
            {
                const double sign = (m & 1) ? -1.0 : 1.0;
                std::span<std::complex<double>> fft_m(
                    m_ffts.begin() + num_lat*m, num_lat);

                std::span<const std::complex<double>> symm_asymm_m(
                    symm_asymm.begin() + num_unique_nodes*m, num_unique_nodes);
                std::span<const std::complex<double>> asymm_symm_m(
                    asymm_symm.begin() + num_unique_nodes*m, num_unique_nodes);
                    
                for (std::size_t i = 0; i < num_unique_nodes; ++i)
                {
                    fft_m[north_offset + i] = symm_asymm_m[i] + asymm_symm_m[i];
                    fft_m[south_offset - i]
                        = sign*(symm_asymm_m[i] - asymm_symm_m[i]);
                }
            }
        }
    }

    void sum_m(SphereGLQGridSpan<double, GridLayout> values)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr double prefactor = 1.0;
        pocketfft::c2r(
            m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid, lon_axis, pocketfft::BACKWARD, m_ffts.data(), values.flatten().data(), prefactor);
    }

    PlmRecursion m_recursion{};
    std::vector<double> m_glq_nodes{};
    std::vector<double> m_glq_weights{};
    std::vector<double> m_plm_grid{};
    std::vector<std::complex<double>> m_ffts{};
    std::vector<std::complex<double>> m_symm_asymm{};
    std::vector<std::size_t> m_pocketfft_shape_grid{};
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid{};
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft{};
    std::size_t m_order{};
};

/**
    @brief Convenient alias for `GLQTransformer` with orthonormal spherical harmonics and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerAcoustics
    = GLQTransformer<SHNorm::qm, SHPhase::none, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with orthonormal spherical harmonics with Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerQM
    = GLQTransformer<SHNorm::qm, SHPhase::cs, GridLayout>;

/**
    @brief Convenient alias for `GLQTransformer` with 4-pi normal spherical harmonics and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerGeo
    = GLQTransformer<SHNorm::geo, SHPhase::none, GridLayout>;

/**
    @brief Function concept taking Cartesian coordinates as inputs.
*/
template <typename Func>
concept cartesian_function = requires (Func f, std::array<double, 3> x)
{
    { f(x) } -> std::same_as<double>;
};

/**
    @brief Function concept taking spherical angles as inputs.
*/
template <typename Func>
concept spherical_function = requires (Func f, double lon, double colat)
{
    { f(lon, colat) } -> std::same_as<double>;
};

/**
    @brief High-level interface for taking SH transforms of functions on balls of arbitrary radii.

    @tparam sh_norm_param normalization convention of spherical harmonics
    @tparam sh_phase_param phase convention of spherical harmonics
    @tparam GridLayoutType
*/
template <
    st::SHNorm sh_norm_param, st::SHPhase sh_phase_param,
    typename GridLayoutType = DefaultLayout>
class SHTransformer
{
public:
    using GridLayout = GridLayoutType;
    SHTransformer() = default;
    explicit SHTransformer(std::size_t order):
        m_grid(order), m_points(order), m_transformer(order) {}

    void resize(std::size_t order)
    {
        m_points.resize(order);
        m_grid.resize(order);
        m_transformer.resize(order);
    }

    template <spherical_function Func>
    void transform(
        Func&& f,
        RealSHExpansionSpan<std::array<double, 2>, sh_norm_param, sh_phase_param> expansion)
    {
        resize(expansion.order());
        m_points.generate_values(m_grid, f);
        m_transformer.forward_transform(m_grid, expansion);
    }

    template <spherical_function Func>
    [[nodiscard]] RealSHExpansion<sh_norm_param, sh_phase_param> transform(
        Func&& f, double radius, std::size_t order)
    {
        resize(order);
        m_points.generate_values(m_grid, f);
        return m_transformer.forward_transform(m_grid, order);
    }

    template <cartesian_function Func>
    void transform(
        Func&& f, 
        RealSHExpansionSpan<std::array<double, 2>, sh_norm_param, sh_phase_param> expansion)
    {
        auto f_scaled = [&](double lon, double colat) {
            const double scolat = std::sin(colat);
            const std::array<double, 3> x = {
                scolat*std::cos(lon), scolat*std::sin(lon), std::cos(colat)
            };
            return f(x);
        };
        resize(expansion.order());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.forward_transform(m_grid, expansion);
    }

    template <cartesian_function Func>
    [[nodiscard]] RealSHExpansion<sh_norm_param, sh_phase_param> transform(
        Func&& f, std::size_t order)
    {
        auto f_scaled = [&](double lon, double colat) {
            const double scolat = std::sin(colat);
            const std::array<double, 3> x = {
                scolat*std::cos(lon), scolat*std::sin(lon), std::cos(colat)
            };
            return f(x);
        };
        resize(order);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.forward_transform(m_grid, order);
    }

private:
    SphereGLQGrid<double, GridLayout> m_grid;
    SphereGLQGridPoints<GridLayout> m_points;
    GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> m_transformer;
};

/**
    @brief Convenient alias for `SHTransformer` with orthonormal spherical harmonics and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using SHTransformerAcoustics
    = SHTransformer<SHNorm::qm, SHPhase::none, GridLayout>;

/**
    @brief Convenient alias for `SHTransformer` with orthonormal spherical harmonics with Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using SHTransformerQM
    = SHTransformer<SHNorm::qm, SHPhase::cs, GridLayout>;

/**
    @brief Convenient alias for `SHTransformer` with 4-pi normal spherical harmonics and no Condon-Shortley phase.

    @tparam GridLayout
*/
template <typename GridLayout = DefaultLayout>
using SHTransformerGeo
    = SHTransformer<SHNorm::geo, SHPhase::none, GridLayout>;

} // namespace st
} // namespace zest