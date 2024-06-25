#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <utility>
#include <stdexcept>
#include <ranges>
#include <algorithm>
#include <span>

#include <cassert>

#include "real_sh_expansion.hpp"
#include "plm_recursion.hpp"
#include "zernike_expansion.hpp"
#include "radial_zernike_recursion.hpp"
#include "md_span.hpp"

namespace zest
{

namespace detail
{

void recursive_trig(
    MDSpan<std::array<double, 2>, 2> trigs, std::span<const double> angles);

}

namespace st
{

/*
Class for evaluating spherical harmonic expansions on arbitrary grids.
*/
class GridEvaluator
{
public:
    GridEvaluator() = default;
    explicit GridEvaluator(std::size_t max_order);

    /*
    Construct `GridEvaluator` with memory reserved for a combination of expansion and grid size.

    Parameters:
    `max_order`: maximum order of spherical harmonic expansion.
    `lon_size`: size of grid in the longitudinal direction.
    `lat_size`: size of grid in the latittudinal direction.
    */
    GridEvaluator(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size);

    /*
    Resize for a combination of expansion and grid size.

    Parameters:
    `max_order`: maximum order of spherical harmonic expansion.
    `lon_size`: size of grid in the longitudinal direction.
    `lat_size`: size of grid in the latittudinal direction.
    */
    void resize(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size);

    /*
    Evaluate spherical harmonic expansion on a grid.

    Parameters:
    `expansion`: spherical harmonics expansion.
    `longitudes`: longitude values defining the grid points.
    `colatitudes`: colatitude values defining the grid points.

    Returns:
    `std::vector` containing values of the expansion on the grid. The values are ordered as a 2D array with shape `{longitudes.size(), colatitudes.size()}` in row-major order.
    */
    template <real_sh_expansion ExpansionType>
    [[nodiscard]] std::vector<double> evaluate(
        ExpansionType&& expansion, std::span<const double> longitudes, std::span<const double> colatitudes)
    {
        constexpr SHNorm NORM = std::remove_cvref_t<ExpansionType>::norm;
        constexpr SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        if (longitudes.size() == 0 || colatitudes.size() == 0)
            return std::vector<double>{};

        const std::size_t order = expansion.order();
        resize(order, longitudes.size(), colatitudes.size());

        for (std::size_t i = 0; i < m_lat_size; ++i)
            m_cos_colat[i] = std::cos(colatitudes[i]);

        st::PlmVecSpan<double, NORM, PHASE> plm(
                m_plm_grid, order, m_lat_size);
        m_plm_recursion.plm_real(plm, m_cos_colat);

        MDSpan<std::array<double, 2>, 2> cossin_lon(
            m_cossin_lon_grid.data(), {order, m_lon_size});
        detail::recursive_trig(cossin_lon, longitudes);

        sum_l(std::forward<ExpansionType>(expansion));

        std::vector<double> res(m_lon_size*m_lat_size);
        sum_m(MDSpan<double, 2>(res.data(), {m_lon_size, m_lat_size}), order);

        return res;
    }
private:
    template <real_sh_expansion ExpansionType>
    void sum_l(ExpansionType&& expansion) noexcept
    {
        TriangleVecSpan<const double, TriangleLayout> ass_leg(
                m_plm_grid, expansion.order(), m_lat_size);
        for (std::size_t l = 0; l < expansion.order(); ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                const std::array<double, 2> coeff = expansion(l,m);
                std::span<const double> plm = ass_leg(l,m);
                
                std::span<std::array<double, 2>> f_m(
                        m_fm_grid.begin() + m*m_lat_size, m_lat_size);
                for (std::size_t i = 0; i < m_lat_size; ++i)
                {
                    const double weight = plm[i];
                    f_m[i][0] += weight*coeff[0];
                    f_m[i][1] += weight*coeff[1];
                }
            }
        }
    }

    void sum_m(MDSpan<double, 2> values, std::size_t order) noexcept;

    st::PlmRecursion m_plm_recursion;
    std::vector<double> m_plm_grid;
    std::vector<double> m_cos_colat;
    std::vector<std::array<double, 2>> m_cossin_lon_grid;
    std::vector<std::array<double, 2>> m_fm_grid;
    std::size_t m_lon_size;
    std::size_t m_lat_size;
    std::size_t m_max_order;
};

}

namespace zt
{

/*
Class for evaluating Zernike expansions on arbitrary grids.
*/
class GridEvaluator
{
public:
    GridEvaluator() = default;
    explicit GridEvaluator(std::size_t max_order);

    /*
    Construct `GridEvaluator` with memory reserved for a combination of expansion and grid size.

    Parameters:
    `max_order`: maximum order of Zernike expansion.
    `lon_size`: size of grid in the longitudinal direction.
    `lat_size`: size of grid in the latittudinal direction.
    `rad_size`: size of grid in the radial direction.
    */
    GridEvaluator(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size, 
        std::size_t rad_size);

    /*
    Resize for a combination of expansion and grid size.

    Parameters:
    `max_order`: maximum order of Zernike expansion.
    `lon_size`: size of grid in the longitudinal direction.
    `lat_size`: size of grid in the latittudinal direction.
    `rad_size`: size of grid in the radial direction.
    */
    void resize(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size, 
        std::size_t rad_size);

    /*
    Evaluate spherical harmonic expansion on a grid.

    Parameters:
    `expansion`: spherical harmonics expansion.
    `longitudes`: longitude values defining the grid points.
    `colatitudes`: colatitude values defining the grid points.
    `radii`: radius values defining the grid points.

    Returns:
    `std::vector` containing values of the expansion on the grid. The values are ordered as a 3D array with shape `{longitudes.size(), colatitudes.size(), radii.size()}` in row-major order.
    */
    template <zernike_expansion ExpansionType>
    [[nodiscard]] std::vector<double> evaluate(
        ExpansionType&& expansion, std::span<const double> longitudes, std::span<const double> colatitudes, std::span<const double> radii)
    {
        constexpr st::SHNorm NORM = std::remove_cvref_t<ExpansionType>::norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        if (longitudes.size() == 0 || colatitudes.size() == 0 || radii.size() == 0)
            return std::vector<double>{};
        
        const std::size_t order = expansion.order();
        resize(order, longitudes.size(), colatitudes.size(), radii.size());


        RadialZernikeVecSpan<double> zernike(
                m_zernike_grid, order, m_rad_size);
        m_zernike_recursion.zernike<ZernikeNorm::NORMED>(zernike, radii);

        for (std::size_t i = 0; i < m_lat_size; ++i)
            m_cos_colat[i] = std::cos(colatitudes[i]);
        
        st::PlmVecSpan<double, NORM, PHASE> plm(m_plm_grid, order, m_lat_size);
        m_plm_recursion.plm_real(plm, m_cos_colat);

        MDSpan<std::array<double, 2>, 2> cossin_lon(
                m_cossin_lon_grid.data(), {order, m_lon_size});
        detail::recursive_trig(cossin_lon, longitudes);

        sum_n(std::forward<ExpansionType>(expansion));
        sum_l(order);

        std::vector<double> res(m_lon_size*m_lat_size*m_rad_size);
        sum_m(MDSpan<double, 3>(
                res.data(), {m_lon_size, m_lat_size, m_rad_size}), order);

        return res;
    }

private:
    template <zernike_expansion ExpansionType>
    void sum_n(ExpansionType&& expansion) noexcept
    {
        const std::size_t order = expansion.order();
        RadialZernikeVecSpan<const double> zernike(
                m_zernike_grid, order, m_rad_size);

        std::ranges::fill(m_flm_grid, std::array<double, 2>{});

        TriangleVecSpan<std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, order, m_rad_size);

        for (std::size_t n = 0; n < order; ++n)
        {
            auto expansion_n = expansion[n];
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                std::span<const double> zernike_nl = zernike(n,l);

                std::span<const std::array<double, 2>>
                expansion_nl = expansion_n[l];

                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::span<std::array<double, 2>> flm_lm = flm(l,m);
                    const std::array<double, 2> coeff = expansion_nl[m];
                    for (std::size_t i = 0; i < m_rad_size; ++i)
                    {
                        flm_lm[i][0] += zernike_nl[i]*coeff[0];
                        flm_lm[i][1] += zernike_nl[i]*coeff[1];
                    }
                }
            }
        }
    }

    void sum_l(std::size_t order) noexcept;

    void sum_m(MDSpan<double, 3> values, std::size_t order) noexcept;

    RadialZernikeRecursion m_zernike_recursion;
    st::PlmRecursion m_plm_recursion;
    std::vector<double> m_zernike_grid;
    std::vector<double> m_plm_grid;
    std::vector<double> m_cos_colat;
    std::vector<std::array<double, 2>> m_cossin_lon_grid;
    std::vector<std::array<double, 2>> m_flm_grid;
    std::vector<std::array<double, 2>> m_fm_grid;
    std::size_t m_lon_size;
    std::size_t m_lat_size;
    std::size_t m_rad_size;
    std::size_t m_max_order;
};

}

}