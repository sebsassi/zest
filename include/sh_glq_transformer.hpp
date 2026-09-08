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
#include <vector>

#include "pocketfft_spec.hpp"

#include "associated_legendre_recursion.hpp"
#include "gauss_legendre.hpp"
#include "sequence.hpp"
#include "sh_concepts.hpp"
#include "sh_expansion.hpp"
#include "sphere_glq_grid.hpp"
#include "utility_concepts.hpp"

namespace zest::st
{

/**
    @brief Transformations between a Gauss-Legendre quadrature grid
    representation and spherical harmonic expansion representation of real data.

    @tparam Convention Spherical harmonic convention.
    @tparam GridLayoutType Memory layout of the grid.
*/
template <sh_convention Convention, typename GridLayoutType = DefaultLayout>
class GLQTransformer: public Convention
{
private:
    template <typename T, std::size_t... Ns>
    using AssLegSpan = AssociatedLegendreSpan<T, Convention, Ns...>;

    template <typename T>
    using AssLegVecSpan = AssociatedLegendreVectorSpan<T, Convention>;

    using AssLegShape = AssociatedLegendreShape<Convention>;

public:
    using Convention::sh_norm;
    using Convention::sh_phase;

    using grid_layout_type = GridLayoutType;

    template <typename T>
    using grid_span_type = SphereGLQGridSpan<T, grid_layout_type>;

    using expansion_shape = SHShape<Indexing::zero_based, Convention>;
    using grid_shape = SphereGLQGridShape<grid_layout_type>;

    template <typename T>
    using sh_span_type = SHSpan<T, Indexing::zero_based, Convention>;

    GLQTransformer(): 
        m_pocketfft_shape_grid(2),
        m_pocketfft_stride_grid(2), 
        m_pocketfft_stride_fft(2) {}
    explicit GLQTransformer(std::size_t order):
        m_recursion(order),
        m_glq_nodes(gl::PackedLayout::size(grid_layout_type::lat_size(order))),
        m_glq_weights(gl::PackedLayout::size(grid_layout_type::lat_size(order))),
        m_ass_leg_grid(grid_layout_type::lat_size(order)*AssLegShape::size(order)),
        m_ffts(grid_layout_type::lat_size(order)*grid_layout_type::fft_size(order)),
        m_symm_asymm(grid_layout_type::fft_size(order)*((grid_layout_type::lat_size(order) + 1) >> 1)*2),
        m_pocketfft_shape_grid(2),
        m_pocketfft_stride_grid(2),
        m_pocketfft_stride_fft(2),
        m_order(order)
    {
        gl::gl_nodes_and_weights<gl::PackedLayout, gl::GLNodeStyle::cos>(
                m_glq_nodes, m_glq_weights, grid_layout_type::lat_size(order) & 1);

        if constexpr (
            std::same_as<grid_layout_type, LatLonLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegVecSpan<double> ass_leg(m_ass_leg_grid, m_glq_nodes.size(), order);
            for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
                m_recursion.generate_real(m_glq_nodes[i], ass_leg[i]);
        }
        else if constexpr (
            std::same_as<grid_layout_type, LonLatLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegSpan<double, std::dynamic_extent> ass_leg(
                    m_ass_leg_grid, order, m_glq_nodes.size());
            m_recursion.generate_real(m_glq_nodes, ass_leg);
        }

        auto shape = grid_layout_type::extents(order);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];

        m_pocketfft_stride_grid[0] = long(shape[1]*sizeof(double));
        m_pocketfft_stride_grid[1] = sizeof(double);

        auto fft_stride = grid_layout_type::fft_stride(order);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = fft_stride[1]*sizeof(std::complex<double>);
    }

    /**
        @brief Order of spherical harmonic expansion.
    */
    [[nodiscard]] std::size_t order() const noexcept { return m_order; }

    /**
        @brief Resize transformer for specified expansion order.

        @param order New order of the expansion.
    */
    void resize(std::size_t order)
    {
        if (order == m_order) return;

        m_recursion.expand(order);

        m_glq_nodes.resize(gl::PackedLayout::size(grid_layout_type::lat_size(order)));
        m_glq_weights.resize(gl::PackedLayout::size(grid_layout_type::lat_size(order)));
        gl::gl_nodes_and_weights<gl::PackedLayout, gl::GLNodeStyle::cos>(
                m_glq_nodes, m_glq_weights, grid_layout_type::lat_size(order) & 1);
        m_ass_leg_grid.resize(m_glq_weights.size()*AssLegShape::size(order));

        if constexpr (
            std::same_as<grid_layout_type, LatLonLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegVecSpan<double> ass_leg(m_ass_leg_grid, m_glq_nodes.size(), order);
            for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
                m_recursion.generate_real(m_glq_nodes[i], ass_leg[i]);
        }
        else if constexpr (
            std::same_as<grid_layout_type, LonLatLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegSpan<double, std::dynamic_extent> ass_leg(m_ass_leg_grid, order, m_glq_nodes.size());
            m_recursion.generate_real(m_glq_nodes, ass_leg);
        }

        m_ffts.resize(grid_layout_type::lat_size(order)*grid_layout_type::fft_size(order));
        m_symm_asymm.resize(grid_layout_type::fft_size(order)*((grid_layout_type::lat_size(order) + 1) >> 1)*2);

        auto shape = grid_layout_type::extents(order);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];

        m_pocketfft_stride_grid[0] = long(shape[1]*sizeof(double));
        m_pocketfft_stride_grid[1] = sizeof(double);

        auto fft_stride = grid_layout_type::fft_stride(order);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = fft_stride[1]*sizeof(std::complex<double>);

        m_order = order;
    }

    /**
        @brief Forward transform from Gauss-Legendre quadrature grid to
        spherical harmonic coefficients.

        @param values Values on the spherical quadrature grid.
        @param expansion Output buffer for coefficients of the spherical
        harmonic expansion.
    */
    template <
        contiguous_buffer_shaped_like<grid_shape> GridType,
        contiguous_buffer_shaped_like<expansion_shape> ExpansionType
    >
        requires representable_as<value_type_of<GridType>, double>
            && std::same_as<value_type_of<GridType>, value_type_of<ExpansionType>>
    void forward_transform(const GridType& values, ExpansionType&& expansion)
    {
        resize(values.order());

        integrate_longitudinal(values.template represent_as<double>());

        fft_to_symm_asymm();

        std::size_t min_order = std::min(expansion.order(), values.order());

        SHSpan<double, Indexing::zero_based, Convention>
        truncated_expansion{
            std::forward<ExpansionType>(expansion).template represent_as<double>().flatten(),
            min_order
        };

        integrate_latitudinal(truncated_expansion);
    }

    /**
        @brief Forward transform from Gauss-Legendre quadrature grid to
        spherical harmonic coefficients.

        @param values Values on the spherical quadrature grid.
        @param order Order of the spherical harmonic expansion.

        @return Spherical harmonic coefficients of the expansion.
    */
    template <contiguous_buffer_shaped_like<grid_shape> GridType>
        requires representable_as<value_type_of<GridType>, double>
    [[nodiscard]] auto forward_transform(const GridType& values, std::size_t order)
    {
        SHExpansion<value_type_of<GridType>, Indexing::zero_based, Convention>
        expansion{order};

        forward_transform(values, expansion);
        return expansion;
    }

    /**
        @brief Backward transform from spherical harmonic expansion to
        Gauss-Legendre quadrature grid.

        @param expansion Coefficients of the spherical harmonic expansion.
        @param values Output buffer for values on the spherical quadrature grid.
    */
    template <
        contiguous_buffer_shaped_like<expansion_shape> ExpansionType,
        contiguous_buffer_shaped_like<grid_shape> GridType
    >
        requires representable_as<value_type_of<ExpansionType>, double>
            && std::same_as<value_type_of<ExpansionType>, value_type_of<GridType>>
    void backward_transform(const ExpansionType& expansion, GridType&& values)
    {
        resize(values.order());

        std::size_t min_order = std::min(expansion.order(), values.order());

        SHSpan<const double, Indexing::zero_based, Convention>
        truncated_expansion{expansion.template represent_as<double>().flatten(), min_order};

        sum_l(truncated_expansion);
        symm_asymm_to_fft();
        sum_m(std::forward<GridType>(values).template represent_as<double>());
    }

    /**
        @brief Backward transform from spherical harmonic expansion of even or
        odd parity to Gauss-Legendre quadrature grid.

        @tparam Expansion type of expansion

        @param expansion Coefficients of the spherical harmonic expansion.
        @param values Output buffer for values on the spherical quadrature grid.

        @note A spherical harmonic expansion has even/odd parity if the first
        index of all nonzero coefficients has even/odd parity.
    */
    template <
        st::zernike_sh_subspan<Indexing::zero_based> ExpansionType,
        contiguous_buffer_shaped_like<grid_shape> GridType
    >
        requires (st::sh_norm_of<ExpansionType>() == sh_norm)
            && (st::sh_phase_of<ExpansionType>() == sh_phase)
            && st::has_inner_rank<ExpansionType, 0>
            && representable_as<value_type_of<ExpansionType>, double>
    void backward_transform(
        const ExpansionType& expansion, GridType&& values)
    {
        resize(values.order());

        std::size_t min_order = std::min(expansion.order(), values.order());

        ShapedSpan<const double, shape_type_of<ExpansionType>>
        truncated_expansion{expansion.template represent_as<double>().flatten(), min_order};

        sum_l(truncated_expansion);
        symm_asymm_to_fft();
        sum_m(std::forward<GridType>(values));
    }

    /**
        @brief Backward transform from spherical harmonic coefficients to
        Gauss-Legendre quadrature grid.

        @param expansion Coefficients of the spherical harmonic expansion.
        @param order Order of the spherical quadrature grid.

        @return Values on the spherical quadrature grid.
    */
    template <contiguous_buffer_shaped_like<expansion_shape> ExpansionType>
        requires representable_as<value_type_of<ExpansionType>, double>
    [[nodiscard]] auto backward_transform(const ExpansionType& expansion, std::size_t order)
    {
        SphereGLQGrid<value_type_of<ExpansionType>, grid_layout_type> grid(order);
        backward_transform(expansion, grid);
        return grid;
    }

    /**
        @brief Backward transform from spherical harmonic expansion of even or
        odd parity to Gauss-Legendre quadrature grid.

        @tparam Expansion Type of the spherical harmonic expansion.

        @param expansion Coefficients of the spherical harmonic expansion.
        @param order Order of the spherical quadrature grid.

        @note A spherical harmonic expansion has even/odd parity if the first
        index of all nonzero coefficients has even/odd parity.
    */
    template <st::zernike_sh_subspan<Indexing::zero_based> ExpansionType>
        requires (st::sh_norm_of<ExpansionType>() == sh_norm)
            && (st::sh_phase_of<ExpansionType>() == sh_phase)
            && st::has_inner_rank<ExpansionType, 0>
            && representable_as<value_type_of<ExpansionType>, double>
    [[nodiscard]] auto backward_transform(const ExpansionType& expansion, std::size_t order)
    {
        SphereGLQGrid<value_type_of<ExpansionType>, grid_layout_type> grid(order);
        backward_transform(expansion, grid);
        return grid;
    }

private:
    void integrate_longitudinal(
        SphereGLQGridSpan<const double, grid_layout_type> values)
    {
        constexpr std::size_t lon_axis = grid_layout_type::lon_axis;
        constexpr double sh_normalization = normalization<sh_norm>();
        const double prefactor
            = sh_normalization*(2.0*std::numbers::pi)/double(values.extent(lon_axis));
        pocketfft::r2c(
            m_pocketfft_shape_grid, m_pocketfft_stride_grid, m_pocketfft_stride_fft,
            lon_axis, pocketfft::FORWARD, values.flatten().data(), m_ffts.data(),
            prefactor);
    }

    void apply_gl_weights() noexcept
    {
        const std::size_t num_lat = m_glq_weights.size();
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);
        if constexpr (
            std::same_as<grid_layout_type, LatLonLayout<typename grid_layout_type::Alignment>>)
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
        else if constexpr (
            std::same_as<grid_layout_type, LonLatLayout<typename grid_layout_type::Alignment>>)
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
    Apply Gauss-Legendre weights and divide Fourier transforms into symmetric
    and antisymmetric parts, `f(x) + f(-x)` and `f(x) - f(-x)`, where `x` are
    the Legendre nodes.

    This division is useful to reduce operation count in the latitudinal
    integration stage due to symmetry properties of the associated legendre
    functions. Namely, `P_lm(-x) = (-1)^(l+m)*P_lm(x)`. This means that for
    `l + m` even, the integration is over the symmetric parts, and for `l + m`
    odd, it is over the antisymmetric parts.

    The final symmetric and antisymmetric parts have a very specific memory
    layout. Given number of latitudes `num_lat`, there are `(num_lat + 1)/2`
    symmetric and antisymmetric components. These are denoted `+` and `-` for
    symmetric and antisymmetric, respectively. The layout differs in an
    expected way depending on whether `GridLayout` is latitude or longitude
    major. Namely, for latitude major order `+` and `-` refer to blocks of
    `(num_lat + 1)/2` complex numbers, whereas for longitude major order they
    refer to individual complex numbers.

    Given `m` denoting the order of the Fourier transform, these components
    are then stored in memory as two alternating sequences:
    `m 0 1 2 3 4 5 6 7 8 ...`
    `s + - + - + - + - + ...`
    `a - + - + - + - + - ...`
    In the latitudinal integration step, for even `l` the sequence `s`
    starting with the symmetric components is chosen, and for odd `l` the
    sequence `a` starting with the antisymmetric components is chosen. This
    leads to a moderately cache efficient access pattern in the latitudinal
    integration step.
    */
    void fft_to_symm_asymm() noexcept
    {
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);
        const std::size_t num_lat = grid_layout_type::lat_size(m_order);
        const std::size_t central_offset = num_lat >> 1;
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t south_offset = num_unique_nodes - 1;
        const std::size_t north_offset = central_offset;

        if constexpr (
            std::same_as<grid_layout_type, LatLonLayout<typename grid_layout_type::Alignment>>)
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
        if constexpr (
            std::same_as<grid_layout_type, LonLatLayout<typename grid_layout_type::Alignment>>)
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
        SHSpan<double, Indexing::zero_based, Convention> expansion) noexcept
    {
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);
        const std::size_t num_ass_leg = m_glq_weights.size();

        std::ranges::fill(expansion.flatten(), 0.0);
        if constexpr (
            std::same_as<grid_layout_type, LatLonLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegVecSpan<const double> ass_leg(m_ass_leg_grid, num_ass_leg, m_order);
            for (std::size_t i = 0; i < num_ass_leg; ++i)
            {
                auto ass_leg_i = ass_leg[i];
                for (auto l : expansion.indices())
                {
                    auto ass_leg_l = ass_leg_i[l];
                    auto expansion_l = expansion[l];
                    std::span<const std::complex<double>> fft(
                        m_symm_asymm.begin() + (2*i + (l & 1))*fft_order, fft_order);
                    for (auto m : expansion_l.indices())
                    {
                        expansion_l[m, 0] += ass_leg_l[m]*fft[m].real();
                        expansion_l[m, 1] += ass_leg_l[m]*fft[m].imag();
                    }
                }
            }
        }
        else if constexpr (
            std::same_as<grid_layout_type, LonLatLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegSpan<const double, std::dynamic_extent>
            ass_leg(m_ass_leg_grid, m_order, num_ass_leg);

            for (auto l : expansion.indices())
            {
                auto expansion_l = expansion[l];
                auto ass_leg_l = ass_leg[l];
                std::span<const std::complex<double>> ffts;
                ffts = std::span<const std::complex<double>>(
                        m_symm_asymm.begin() + (l & 1)*num_ass_leg*fft_order,
                        num_ass_leg*fft_order);
                for (auto m : expansion_l.indices())
                {
                    auto ass_leg_lm = ass_leg_l[m];
                    std::span<const std::complex<double>> fft(
                        ffts.begin() + m*num_ass_leg, num_ass_leg);

                    std::array<double, 2> coeff{};
                    switch (num_ass_leg & 3)
                    {
                        case 1:
                            coeff[0] = ass_leg_lm[0]*fft[0].real();
                            coeff[1] = ass_leg_lm[0]*fft[0].imag();
                            break;
                        case 2:
                            coeff[0] = ass_leg_lm[0]*fft[0].real()
                                    + ass_leg_lm[1]*fft[1].real();
                            coeff[1] = ass_leg_lm[0]*fft[0].imag()
                                    + ass_leg_lm[1]*fft[1].imag();
                            break;
                        case 3:
                            coeff[0] = ass_leg_lm[0]*fft[0].real()
                                    + ass_leg_lm[1]*fft[1].real()
                                    + ass_leg_lm[2]*fft[2].real();
                            coeff[1] = ass_leg_lm[0]*fft[0].imag()
                                    + ass_leg_lm[1]*fft[1].imag()
                                    + ass_leg_lm[2]*fft[2].imag();
                            break;
                    }

                    std::array<double, 8> partial_sum{};
                    for (std::size_t i = (num_ass_leg & 3); i < num_ass_leg; i += 4)
                    {
                        partial_sum[0] += ass_leg_lm[i]*fft[i].real();
                        partial_sum[1] += ass_leg_lm[i]*fft[i].imag();
                        partial_sum[2] += ass_leg_lm[i + 1]*fft[i + 1].real();
                        partial_sum[3] += ass_leg_lm[i + 1]*fft[i + 1].imag();
                        partial_sum[4] += ass_leg_lm[i + 2]*fft[i + 2].real();
                        partial_sum[5] += ass_leg_lm[i + 2]*fft[i + 2].imag();
                        partial_sum[6] += ass_leg_lm[i + 3]*fft[i + 3].real();
                        partial_sum[7] += ass_leg_lm[i + 3]*fft[i + 3].imag();
                    }

                    for (std::size_t i = 0; i < 8; i += 2)
                    {
                        coeff[0] += partial_sum[i];
                        coeff[1] += partial_sum[i + 1];
                    }

                    expansion_l[m, 0] = coeff[0];
                    expansion_l[m, 1] = coeff[1];
                }
            }
        }
    }

    void sum_l(
        SHSpan<const double, Indexing::zero_based, Convention> expansion) noexcept
    {
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);
        const std::size_t num_ass_leg = m_glq_weights.size();

        std::ranges::fill(m_symm_asymm, std::complex<double>{});

        if constexpr (
            std::same_as<grid_layout_type, LatLonLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegVecSpan<const double> ass_leg(m_ass_leg_grid, num_ass_leg, m_order);
            for (std::size_t i = 0; i < num_ass_leg; ++i)
            {
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + 2*i*fft_order, 2*fft_order);
                auto ass_leg_i = ass_leg[i];
                for (auto l : expansion.indices())
                {
                    auto ass_leg_l = ass_leg_i[l];
                    auto expansion_l = expansion[l];
                    symm_asymm[(l & 1)*fft_order] += std::complex<double>{
                        ass_leg_l[0]*expansion_l[0, 0], -ass_leg_l[0]*expansion_l[0, 1]
                    };
                    for (auto m : expansion_l.indices(1))
                    {
                        const double weight = 0.5*ass_leg_l[m];
                        symm_asymm[(l & 1)*fft_order + m]
                            += std::complex<double>{
                                weight*expansion_l[m, 0],
                                -weight*expansion_l[m, 1]
                            };
                    }
                }
            }
        }
        else if constexpr (
            std::same_as<grid_layout_type, LonLatLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegSpan<const double, std::dynamic_extent>
            ass_leg(m_ass_leg_grid, m_order, num_ass_leg);

            std::span coeffs = expansion.flatten();
            for (auto l : expansion.indices())
            {
                auto expansion_l = expansion[l];
                auto ass_leg_l = ass_leg[l];
                auto ass_leg_l0 = ass_leg_l[0];
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + (l & 1)*num_ass_leg*fft_order, num_ass_leg);
                for (std::size_t i = 0; i < num_ass_leg; ++i)
                {
                    symm_asymm[i] += std::complex<double>{
                        ass_leg_l0[i]*expansion_l[0, 0], -ass_leg_l0[i]*expansion_l[0, 1]
                    };
                }

                for (auto m : expansion_l.indices(1))
                {
                    auto ass_leg_lm = ass_leg_l[m];
                    std::span<std::complex<double>> symm_asymm(
                        m_symm_asymm.begin() + ((l & 1)*fft_order + m)*num_ass_leg,
                        num_ass_leg);
                    for (std::size_t i = 0; i < num_ass_leg; ++i)
                    {
                        const double weight = 0.5*ass_leg_lm[i];
                        symm_asymm[i] += std::complex<double>{
                            weight*expansion_l[m, 0], -weight*expansion_l[m, 1]
                        };
                    }
                }
            }
        }
    }

    template <st::zernike_sh_subspan<Indexing::zero_based> ExpansionType>
        requires (st::sh_norm_of<ExpansionType>() == sh_norm)
            && (st::sh_phase_of<ExpansionType>() == sh_phase)
    void sum_l(const ExpansionType& expansion) noexcept
    {
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);
        const std::size_t num_ass_leg = m_glq_weights.size();

        std::ranges::fill(m_symm_asymm, std::complex<double>{});

        if constexpr (
            std::same_as<grid_layout_type, LatLonLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegVecSpan<const double> ass_leg(m_ass_leg_grid, num_ass_leg, m_order);
            for (std::size_t i = 0; i < num_ass_leg; ++i)
            {
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + 2*i*fft_order, 2*fft_order);
                auto ass_leg_i = ass_leg[i];
                for (auto l : expansion.indices())
                {
                    auto ass_leg_l = ass_leg_i[l];
                    auto expansion_l = expansion[l];
                    symm_asymm[(l & 1)*fft_order] += std::complex<double>{
                        ass_leg_l[0]*expansion_l[0, 0], -ass_leg_l[0]*expansion_l[0, 1]
                    };
                    for (auto m : expansion_l.indices(1))
                    {
                        const double weight = 0.5*ass_leg_l[m];
                        symm_asymm[(l & 1)*fft_order + m]
                            += std::complex<double>{
                                weight*expansion_l[m, 0],
                                -weight*expansion_l[m, 1]
                            };
                    }
                }
            }
        }
        else if constexpr (
            std::same_as<grid_layout_type, LonLatLayout<typename grid_layout_type::Alignment>>)
        {
            AssLegSpan<const double, std::dynamic_extent>
            ass_leg(m_ass_leg_grid, m_order, num_ass_leg);

            for (auto l : expansion.indices())
            {
                auto expansion_l = expansion[l];
                auto ass_leg_l = ass_leg[l];
                auto ass_leg_l0 = ass_leg_l[0];
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + (l & 1)*num_ass_leg*fft_order, num_ass_leg);
                for (std::size_t i = 0; i < num_ass_leg; ++i)
                {
                    symm_asymm[i] += std::complex<double>{
                        ass_leg_l0[i]*expansion_l[0, 0], -ass_leg_l0[i]*expansion_l[0, 1]
                    };
                }

                for (auto m : expansion_l.indices(1))
                {
                    auto ass_leg_lm = ass_leg_l[m];
                    std::span<std::complex<double>> symm_asymm(
                        m_symm_asymm.begin() + ((l & 1)*fft_order + m)*num_ass_leg,
                        num_ass_leg);
                    for (std::size_t i = 0; i < num_ass_leg; ++i)
                    {
                        const double weight = 0.5*ass_leg_lm[i];
                        symm_asymm[i] += std::complex<double>{
                            weight*expansion_l[m, 0], -weight*expansion_l[m, 1]
                        };
                    }
                }
            }
        }
    }

    // Inverse of `fft_to_symm_asymm`
    void symm_asymm_to_fft() noexcept
    {
        const std::size_t fft_order = grid_layout_type::fft_size(m_order);
        const std::size_t num_lat = grid_layout_type::lat_size(m_order);
        const std::size_t central_offset = num_lat >> 1;
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t south_offset = num_unique_nodes - 1;
        const std::size_t north_offset = central_offset;

        if constexpr (
            std::same_as<grid_layout_type, LatLonLayout<typename grid_layout_type::Alignment>>)
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
        if constexpr (
            std::same_as<grid_layout_type, LonLatLayout<typename grid_layout_type::Alignment>>)
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

    void sum_m(SphereGLQGridSpan<double, grid_layout_type> values)
    {
        constexpr std::size_t lon_axis = grid_layout_type::lon_axis;
        constexpr double prefactor = 1.0;
        pocketfft::c2r(
            m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid,
            lon_axis, pocketfft::BACKWARD, m_ffts.data(), values.flatten().data(),
            prefactor);
    }

    AssociatedLegendreRecursion m_recursion;
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_ass_leg_grid;
    std::vector<std::complex<double>> m_ffts;
    std::vector<std::complex<double>> m_symm_asymm;
    std::vector<std::size_t> m_pocketfft_shape_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft;
    std::size_t m_order{};
};

/**
    @brief High-level interface for taking SH transforms of functions on balls
    of arbitrary radii.

    @tparam Convention Spherical harmonic convention.
    @tparam GridLayoutType
*/
template <st::sh_convention Convention, typename GridLayoutType = DefaultLayout>
class SHTransformer: Convention
{
private:
    using Transformer = GLQTransformer<Convention, GridLayoutType>;

public:
    using Convention::sh_norm;
    using Convention::sh_phase;

    using grid_layout_type = GridLayoutType;
    using expansion_shape = typename Transformer::expansion_shape;
    using grid_shape = typename Transformer::grid_shape;

    template <typename T>
    using grid_span_type = SphereGLQGridSpan<T, grid_layout_type>;

    template <typename T>
    using sh_span_type = SHSpan<T, Indexing::zero_based, Convention>;

    SHTransformer() = default;
    explicit SHTransformer(std::size_t order):
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
        @brief Get spherical harmonic expansion of a function expressed in
        spherical coordinates.

        @tparam FuncType type of function

        @param f function to transform
        @param expansion buffer to store the expansion
    */
    template <
        spherical_function FuncType,
        contiguous_buffer_shaped_like<expansion_shape> ExpansionType
    >
        requires representable_as<std::invoke_result_t<FuncType, double, double>, double>
    void forward_transform(FuncType&& f, ExpansionType&& expansion)
    {
        resize(expansion.order());

        using ResultType = std::invoke_result_t<FuncType, double, double>;
        m_points.generate_values(
                m_grid.template represent_as<ResultType>(), std::forward<FuncType>(f));
        m_transformer.forward_transform(
                m_grid.template represent_as<ResultType>(),
                std::forward<ExpansionType>(expansion));
    }

    /**
        @brief Get spherical harmonic expansion of a function expressed in
        spherical coordinates.

        @tparam FuncType type of function

        @param f function to transform
        @param order order of the expansion

        @returns spherical harmonic expansion
    */
    template <spherical_function FuncType>
    [[nodiscard]] auto forward_transform(FuncType&& f, std::size_t order)
    {
        resize(order);

        using ResultType = std::invoke_result_t<FuncType, double, double>;
        m_points.generate_values(
                m_grid.template represent_as<ResultType>(), std::forward<FuncType>(f));
        return m_transformer.forward_transform(
                m_grid.template represent_as<ResultType>(), order);
    }

    template <
        contiguous_buffer_shaped_like<expansion_shape> ExpansionType,
        contiguous_buffer_shaped_like<grid_shape> GridType
    >
        requires representable_as<value_type_of<ExpansionType>, double>
            && std::same_as<value_type_of<ExpansionType>, value_type_of<GridType>>
    void backward_transform(const ExpansionType& expansion, GridType&& values)
    {
        m_transformer.backward_transform(expansion, std::forward<GridType>(values));
    }

    template <
        st::zernike_sh_subspan<Indexing::zero_based> ExpansionType,
        contiguous_buffer_shaped_like<grid_shape> GridType
    >
        requires (st::sh_norm_of<ExpansionType>() == sh_norm)
            && (st::sh_phase_of<ExpansionType>() == sh_phase)
            && st::has_inner_rank<ExpansionType, 0>
            && representable_as<value_type_of<ExpansionType>, double>
    void backward_transform(const ExpansionType& expansion, GridType&& values)
    {
        m_transformer.backward_transform(expansion, std::forward<GridType>(values));
    }

    template <contiguous_buffer_shaped_like<expansion_shape> ExpansionType>
        requires representable_as<value_type_of<ExpansionType>, double>
    [[nodiscard]] auto backward_transform(const ExpansionType& expansion, std::size_t order)
    {
        return m_transformer.backward_transform(expansion, order);
    }

    template <st::zernike_sh_subspan<Indexing::zero_based> ExpansionType>
        requires (st::sh_norm_of<ExpansionType>() == sh_norm)
            && (st::sh_phase_of<ExpansionType>() == sh_phase)
            && st::has_inner_rank<ExpansionType, 0>
            && representable_as<value_type_of<ExpansionType>, double>
    [[nodiscard]] SphereGLQGrid<value_type_of<ExpansionType>, grid_layout_type>
    backward_transform(const ExpansionType& expansion, std::size_t order)
    {
        return m_transformer.backward_transform(expansion, order);
    }

private:
    SphereGLQGrid<double, grid_layout_type> m_grid;
    SphereGLQGridPoints<grid_layout_type> m_points;
    GLQTransformer<Convention, grid_layout_type> m_transformer;
};

} // namespace zest::st

