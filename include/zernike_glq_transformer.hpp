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
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "alignment.hpp"
#include "pocketfft_spec.hpp"

#include "associated_legendre_recursion.hpp"
#include "ball_glq_grid.hpp"
#include "gauss_legendre.hpp"
#include "md_span.hpp"
#include "radial_glq_grid.hpp"
#include "radial_zernike_recursion.hpp"
#include "zernike_expansion.hpp"

namespace zest::zt
{


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

        @param order Order of the Zernike expansion.
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

        @param values Values on the quadrature grid.
        @param expansion Output buffer for coefficients of the Zernike expansion.
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

        @param expansion Coefficients of the Zernike expansion.
        @param values Output buffer for values on the ball quadrature grid.
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
        @brief Forward transform from a quadrature grid to Zernike coefficients.

        @param values Values on the ball quadrature grid.
        @param order Order of the output expansion.

        @return Coefficients of the Zernike expansion.
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
        @brief Backward transform from Zernike coefficients to a quadrature grid.

        @param expansion Coefficients of the expansion.
        @param order Order of the output expansion.

        @return Function values on a ball quadrature grid.
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

        @tparam FuncType Type of function.

        @param f Function to transform.
        @param radius Radius of the ball `f` is defined on.
        @param expansion Output buffer for the Zernike expansion coefficients.
    */
    template <ball_function<double> FuncType>
    void forward_transform(
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

        @tparam FuncType Type of function.

        @param f Function to transform.
        @param radius Radius of the ball `f` is defined on.
        @param order Order of the expansion.

        @returns Zernike expansion
    */
    template <ball_function<double> FuncType>
    [[nodiscard]] ZernikeExpansion<double, IndexingMode::zero_based, zernike_norm_param, sh_norm_param, sh_phase_param>
    forward_transform(FuncType&& f, double radius, std::size_t order)
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

        @tparam FuncType Type of the function.

        @param f Function to transform
        @param radius Radius of the ball `f` is defined on.
        @param expansion Output buffer for the Zernike expansion coefficients.
    */
    template <cartesian_function<double> FuncType>
    void forward_transform(
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

        @tparam FuncType Type of the function.

        @param f Function to transform.
        @param radius Radius of the ball `f` is defined on.
        @param order Order of the expansion.

        @return Coefficients of the Zernike expansion.
    */
    template <cartesian_function<double> FuncType>
    [[nodiscard]] ZernikeExpansion<double, IndexingMode::zero_based, zernike_norm_param, sh_norm_param, sh_phase_param>
    forward_transform(FuncType&& f, double radius, std::size_t order)
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

    /**
        @brief Backward transform from Zernike coefficients to Gauss-Legendre quadrature grid.

        @param expansion Coefficients of the Zernike expansion.
        @param values Output buffer for values on the ball quadrature grid.
    */
    void backward_transform(
        ZernikeSpan<const double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion,
        BallGLQGridSpan<double, grid_layout_type> values)
    {
        m_transformer.backward_transform(expansion, values);
    }

    /**
        @brief Backward transform from Zernike coefficients to Gauss-Legendre quadrature grid.

        @param expansion Coefficients of the Zernike expansion.
        @param order Order of the expansion.

        @return Function values on a ball quadrature grid.
    */
    [[nodiscard]] BallGLQGrid<double, grid_layout_type>
    backward_transform(
        ZernikeSpan<const double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion,
        std::size_t order)
    {
        return m_transformer.backward_transform(expansion, order);
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

template <
    zest::zt::ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase,
    typename AlignmentType = CacheLineAlignment>
class IsotropicGLQTransformer
{
public:
    using grid_layout_type = RadialGridLayout<AlignmentType>;
    using alignment_type = AlignmentType;

    IsotropicGLQTransformer() = default;
    IsotropicGLQTransformer(std::size_t order):
        m_recursion{order},
        m_glq_nodes(grid_layout_type::size(order)),
        m_glq_weights(grid_layout_type::size(order)),
        m_weighted_values(grid_layout_type::size(order)),
        m_order{order}
    {
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::cos>(
                m_glq_nodes, m_glq_weights, m_glq_weights.size() & 1);

        for (auto& node : m_glq_nodes)
            node = 0.5*(1.0 + node);

        m_recursion.set_radii(m_glq_nodes);
    }

    /**
        @brief Resize the transformer to work with expansions of different
        order.

        @param order Order of Zernike expansion.
    */
    void resize(std::size_t order)
    {
        if (order == m_order) return;

        m_glq_nodes.resize(grid_layout_type::size(order));
        m_glq_weights.resize(grid_layout_type::size(order));
        m_weighted_values.resize(grid_layout_type::size(order));
        m_recursion.expand(order);
        m_order = order;

        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::cos>(
                m_glq_nodes, m_glq_weights, m_glq_weights.size() & 1);

        for (auto& node : m_glq_nodes)
            node = 0.5*(1.0 + node);

        m_recursion.set_radii(m_glq_nodes);
    }

    /**
        @brief Forward transform from a radial quadrature grid to Zernike coefficients.

        @param values Values on the radial quadrature grid.
        @param expansion Coefficients of the Zernike expansion.
    */
    void forward_transform(
        RadialGLQGridSpan<const double, alignment_type> values,
        IsotropicZernikeSpan<double, zernike_norm, sh_norm, sh_phase> expansion)
    {
        resize(values.order());

        std::size_t min_order = std::min(values.order(), expansion.order());

        IsotropicZernikeSpan<double, zernike_norm, sh_norm, sh_phase>
        truncated_expansion{expansion.flatten(), min_order};

        for (std::size_t i = 0; i < values.size(); ++i)
        {
            const double r = m_glq_nodes[i];
            m_weighted_values[i] = r*r*m_glq_weights[i]*values[i];
        }

        m_recursion.init();
        for (auto n : truncated_expansion.indices())
        {
            auto radial_zernike = m_recursion.current();
            double& element = truncated_expansion[n];
            element = 0.0;
            for (std::size_t i = 0; i < values.size(); ++i)
                element += m_weighted_values[i]*radial_zernike[i];

            m_recursion.iterate();

            constexpr double radial_integral_norm = 0.5;
            constexpr double spherical_integral = (sh_norm == zest::st::SHNorm::geo) ?
                1.0 : 2.0/std::numbers::inv_sqrtpi;
            constexpr double norm = radial_integral_norm*spherical_integral;
            const double zernike_normalization = normalization<zernike_norm>(n);
            element *= norm*zernike_normalization;
        }
    }

    /**
        @brief Forward transform from a radial quadrature grid to Zernike coefficients.

        @param values Values on the radial quadrature grid.
        @param order Order of the output expansion.

        @return Coefficients of the Zernike expansion.
    */
    [[nodiscard]] IsotropicZernikeExpansion<double, zernike_norm, sh_norm, sh_phase>
    forward_transform(RadialGLQGridSpan<const double, alignment_type> values, std::size_t order)
    {
        IsotropicZernikeExpansion<double, zernike_norm, sh_norm, sh_phase>
        expansion{order};

        forward_transform(values, expansion);
        return expansion;
    }

    /**
        @brief Backward transform from Zernike expansion to a radial quadrature grid.

        @param expansion Coefficients of the Zernike expansion.
        @param values Values on the radial quadrature grid.
    */
    void backward_transform(
        IsotropicZernikeSpan<const double, zernike_norm, sh_norm, sh_phase> expansion,
        RadialGLQGridSpan<double, alignment_type> values)
    {
        resize(values.order());

        std::size_t min_order = std::min(expansion.order(), values.order());

        IsotropicZernikeSpan<const double, zernike_norm, sh_norm, sh_phase>
        truncated_expansion{expansion.flatten(), min_order};

        m_recursion.init();
        for (auto n : truncated_expansion.indices())
        {
            auto radial_zernike = m_recursion.current();
            const double spherical_harmonic = (sh_norm == zest::st::SHNorm::geo) ?
                1.0 : 0.5*std::numbers::inv_sqrtpi;
            const double element = spherical_harmonic*truncated_expansion[n];
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] += element*radial_zernike[i];

            m_recursion.iterate();
        }
    }

    /**
        @brief Backward transform from Zernike coefficients to a quadrature grid.

        @param expansion Coefficients of the expansion.
        @param order Order of the output expansion.

        @return Function values on a radial quadrature grid.
    */
    [[nodiscard]] RadialGLQGrid<double>
    backward_transform(
        IsotropicZernikeSpan<const double, zernike_norm, sh_norm, sh_phase> expansion,
        std::size_t order)
    {
        RadialGLQGrid<double, alignment_type> grid{order};

        backward_transform(expansion, grid);
        return grid;
    }

private:
    IsotropicRadialZernikeRecursion<zernike_norm> m_recursion;
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_weighted_values;
    std::size_t m_order{};
};

/**
    @brief Convenient alias for `IsotropicGLQTransformer` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicGLQTransformerAcoustics
    = IsotropicGLQTransformer<
        ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none, Alignment>;

/**
    @brief Convenient alias for `IsotropicGLQTransformer` with orthonorml Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicGLQTransformerNormalAcoustics
    = IsotropicGLQTransformer<
        ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::none, Alignment>;

/**
    @brief Convenient alias for `IsotropicGLQTransformer` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicGLQTransformerQM
    = IsotropicGLQTransformer<
        ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs, Alignment>;

/**
    @brief Convenient alias for `IsotropicGLQTransformer` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicGLQTransformerNormalQM
    = IsotropicGLQTransformer<
        ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs, Alignment>;

/**
    @brief Convenient alias for `IsotropicGLQTransformer` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicGLQTransformerGeo
    = IsotropicGLQTransformer<
        ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none, Alignment>;

/**
    @brief Convenient alias for `IsotropicGLQTransformer` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicGLQTransformerNormalGeo
    = IsotropicGLQTransformer<
        ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none, Alignment>;

/**
    @brief High-level interface for taking Zernike transforms of functions on
    balls of arbitrary radii.

    @tparam zernike_norm_param normalization convention of Zernike functions
    @tparam sh_norm_param normalization convention of spherical harmonics
    @tparam sh_phase_param phase convention of spherical harmonics
    @tparam GridLayoutType
*/
template <ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param, typename AlignmentType = CacheLineAlignment>
class IsotropicZernikeTransformer
{
public:
    using alignment_type = AlignmentType;

    static constexpr ZernikeNorm zernike_norm = zernike_norm_param;
    static constexpr st::SHNorm sh_norm = sh_norm_param;
    static constexpr st::SHPhase sh_phase = sh_phase_param;

    IsotropicZernikeTransformer() = default;
    explicit IsotropicZernikeTransformer(std::size_t order):
        m_grid(order), m_points(order), m_transformer(order) {}

    /**
        @brief Resize the transformer to work with expansions of different
        order.

        @param order Order of Zernike expansion.
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

        @tparam FuncType Type of function.

        @param f Function to transform.
        @param radius Radius of the ball `f` is defined on.
        @param expansion Buffer to store the expansion.
    */
    template <isotropic_function<double> FuncType>
    void forward_transform(
        FuncType&& f, double radius,
        IsotropicZernikeSpan<double, zernike_norm_param, sh_norm_param, sh_phase_param> expansion)
    {
        auto f_scaled = [&](double r) {
            return std::forward<FuncType>(f)(r*radius);
        };
        resize(expansion.order());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.forward_transform(m_grid, expansion);
    }

    /**
        @brief Get Zernike expansion of a function expressed in spherical
        coordinates.

        @tparam FuncType Type of the function.

        @param f Function to transform.
        @param radius Radius of the ball `f` is defined on.
        @param order Order of the expansion.

        @return Zernike expansion coefficients of the function.
    */
    template <isotropic_function<double> FuncType>
    [[nodiscard]] IsotropicZernikeExpansion<double, zernike_norm_param, sh_norm_param, sh_phase_param>
    forward_transform(FuncType&& f, double radius, std::size_t order)
    {
        auto f_scaled = [&](double r) {
            return std::forward<FuncType>(f)(r*radius);
        };
        resize(order);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.forward_transform(m_grid, order);
    }

    /**
        @brief Backward transform from Zernike coefficients to values on a
        radial quadrature grid.

        @param values Values on the ball quadrature grid.
        @param expansion Coefficients of the expansion.
    */
    void backward_transform(
        IsotropicZernikeSpan<const double, zernike_norm, sh_norm, sh_phase> expansion,
        RadialGLQGridSpan<double, alignment_type> values)
    {
        m_transformer.backward_transform(expansion, values);
    }

    /**
        @brief Backward transform from Zernike coefficients to values on a
        radial quadrature grid.

        @param expansion coefficients of the expansion

        @return Values of the corresponding function on a radial quadrature
        grid.
    */
    [[nodiscard]] RadialGLQGrid<double, alignment_type>
    backward_transform(
        IsotropicZernikeSpan<const double, zernike_norm, sh_norm, sh_phase> expansion,
        std::size_t order)
    {
        return m_transformer.backward_transform(expansion, order);
    }

private:
    RadialGLQGrid<double, alignment_type> m_grid;
    RadialGLQGridPoints<alignment_type> m_points;
    IsotropicGLQTransformer<zernike_norm_param, sh_norm_param, sh_phase_param, alignment_type> m_transformer;
};

/**
    @brief Convenient alias for `IsotropicZernikeTransformer` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicZernikeTransformerAcoustics
    = IsotropicZernikeTransformer<
        ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::none, Alignment>;

/**
    @brief Convenient alias for `IsotropicZernikeTransformer` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and no Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicZernikeTransformerNormalAcoustics
    = IsotropicZernikeTransformer<
        ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::none, Alignment>;

/**
    @brief Convenient alias for `IsotropicZernikeTransformer` with unnormalized Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicZernikeTransformerQM
    = IsotropicZernikeTransformer<
        ZernikeNorm::unnormed, st::SHNorm::qm, st::SHPhase::cs, Alignment>;

/**
    @brief Convenient alias for `IsotropicZernikeTransformer` with orthonormal Zernike
    functions, orthonormal spherical harmonics, and Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicZernikeTransformerNormalQM
    = IsotropicZernikeTransformer<
        ZernikeNorm::normed, st::SHNorm::qm, st::SHPhase::cs, Alignment>;

/**
    @brief Convenient alias for `IsotropicZernikeTransformer` with unnormalized Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicZernikeTransformerGeo
    = IsotropicZernikeTransformer<
        ZernikeNorm::unnormed, st::SHNorm::geo, st::SHPhase::none, Alignment>;

/**
    @brief Convenient alias for `IsotropicZernikeTransformer` with orthonormal Zernike
    functions, 4-pi normal spherical harmonics, and no Condon-Shortley phase.

    @tparam Alignment
*/
template <typename Alignment = CacheLineAlignment>
using IsotropicZernikeTransformerNormalGeo
    = IsotropicZernikeTransformer<
        ZernikeNorm::normed, st::SHNorm::geo, st::SHPhase::none, Alignment>;

} // namespace zest::zt

