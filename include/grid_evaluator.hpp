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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "md_array.hpp"
#include "associated_legendre_recursion.hpp"
#include "md_span.hpp"
#include "radial_zernike_recursion.hpp"
#include "sh_expansion.hpp"
#include "triangle_spans.hpp"
#include "zernike_expansion.hpp"

namespace zest
{

namespace detail
{

/**
    @brief Recursively evaluate `cos` and `sin` for integer multiples of `angles`.

    @param trigs array to store `{cos, sin}` pairs
    @param angles array of input angles
*/
void recursive_trig(
    MDSpan<double, std::dynamic_extent, std::dynamic_extent, 2> trigs,
    std::span<const double> angles) noexcept;

} // namespace detail

namespace st
{

/**
    @brief Class for evaluating spherical harmonic expansions on arbitrary grids.
*/
class GridEvaluator
{
public:
    GridEvaluator() = default;

    /**
        @brief Reserves memory for an expansion of given order.

        @param max_order maximum order of spherical harmonic expansion.
    */
    explicit GridEvaluator(std::size_t max_order);

    /**
        @brief Reserves memory for a combination of expansion and grid size.

        @param max_order maximum order of spherical harmonic expansion.
        @param lon_size size of grid in the longitudinal direction.
        @param lat_size size of grid in the latittudinal direction.
    */
    GridEvaluator(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size);

    /**
        @brief Resize for a combination of expansion and grid size.

        @param max_order maximum order of spherical harmonic expansion.
        @param lon_size size of grid in the longitudinal direction.
        @param lat_size size of grid in the latittudinal direction.
    */
    void resize(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size);

    /**
        @brief Evaluate spherical harmonic expansion on a grid.

        @param expansion spherical harmonics expansion.
        @param longitudes longitude values defining the grid points.
        @param colatitudes colatitude values defining the grid points.

        @return Two dimensional array containing values of the expansion on the
        grid with shape `{longitudes.size(), colatitudes.size()}` in row-major
        order.
    */
    template <SHNorm sh_norm, SHPhase sh_phase>
    [[nodiscard]] DynamicMDArray<double, 2> evaluate(
        SHSpan<const double, IndexingMode::zero_based, sh_norm, sh_phase> expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes)
    {
        if (longitudes.size() == 0 || colatitudes.size() == 0)
            return {};

        const std::size_t order = expansion.order();
        resize(order, longitudes.size(), colatitudes.size());

        for (std::size_t i = 0; i < m_lat_size; ++i)
            m_cos_colat[i] = std::cos(colatitudes[i]);

        AssociatedLegendreSpan<double, sh_norm, sh_phase, std::dynamic_extent>
        ass_leg(m_ass_leg_grid, order, m_lat_size);

        m_ass_leg_recursion.generate_real(m_cos_colat, ass_leg);

        MDSpan<double, std::dynamic_extent, std::dynamic_extent, 2> cossin_lon(
            m_cossin_lon_grid, std::array<std::size_t, 3>{order, m_lon_size, 2});
        zest::detail::recursive_trig(cossin_lon, longitudes);

        sum_l(expansion);

        DynamicMDArray<double, 2> res{m_lon_size, m_lat_size};
        sum_m(res, order);

        return res;
    }

    template <SHNorm sh_norm, SHPhase sh_phase>
    [[nodiscard]] DynamicMDArray<double, 2> evaluate(
        SHSpan<double, IndexingMode::zero_based, sh_norm, sh_phase> expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes)
    {
        using ExpansionType = SHSpan<double, IndexingMode::zero_based, sh_norm, sh_phase>;
        return evaluate((typename ExpansionType::const_view)(expansion), longitudes, colatitudes);
    }

    template <SHNorm sh_norm, SHPhase sh_phase>
    [[nodiscard]] DynamicMDArray<double, 2> evaluate(
        const SHExpansion<double, IndexingMode::zero_based, sh_norm, sh_phase>& expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes)
    {
        using ExpansionType = SHExpansion<double, IndexingMode::zero_based, sh_norm, sh_phase>;
        return evaluate((typename ExpansionType::const_view)(expansion), longitudes, colatitudes);
    }

private:
    template <SHNorm sh_norm, SHPhase sh_phase>
    void sum_l(SHSpan<const double, IndexingMode::zero_based, sh_norm, sh_phase> expansion) noexcept
    {
        AssociatedLegendreSpan<const double, sh_norm, sh_phase, std::dynamic_extent>
        ass_leg(m_ass_leg_grid, expansion.order(), m_lat_size);
        for (auto l : expansion.indices())
        {
            auto expansion_l = expansion[l];
            auto ass_leg_l = ass_leg[l];
            for (auto m : expansion_l.indices())
            {
                auto expansion_lm = expansion_l[m];
                auto ass_leg_lm = ass_leg_l[m];

                std::span<std::array<double, 2>> f_m(
                        m_fm_grid.begin() + m*m_lat_size, m_lat_size);
                for (std::size_t i = 0; i < m_lat_size; ++i)
                {
                    const double weight = ass_leg_lm[i];
                    f_m[i][0] += weight*expansion_lm[0];
                    f_m[i][1] += weight*expansion_lm[1];
                }
            }
        }
    }

    void sum_m(DynamicMDSpan<double, 2> values, std::size_t order) noexcept;

    st::AssociatedLegendreRecursion m_ass_leg_recursion;
    std::vector<double> m_ass_leg_grid;
    std::vector<double> m_cos_colat;
    std::vector<double> m_cossin_lon_grid;
    std::vector<std::array<double, 2>> m_fm_grid;
    std::size_t m_lon_size{};
    std::size_t m_lat_size{};
    std::size_t m_max_order{};
};

} // namespace st

namespace zt
{

/*
Class for evaluating Zernike expansions on arbitrary grids.
*/
class GridEvaluator
{
public:
    GridEvaluator() = default;

    /**
        @brief Reserves memory for an expansion of given order.

        @param max_order maximum order of spherical harmonic expansion.
    */
    explicit GridEvaluator(std::size_t max_order);

    /**
        @brief Reserves memory for a combination of expansion and grid size.

        @param max_order maximum order of spherical harmonic expansion.
        @param lon_size size of grid in the longitudinal direction.
        @param lat_size size of grid in the latittudinal direction.
        @param rad_size size of grid in the radial direction.
    */
    GridEvaluator(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size, 
        std::size_t rad_size);

    /**
        @brief Resize for a combination of expansion and grid size.

        @param max_order maximum order of spherical harmonic expansion.
        @param lon_size size of grid in the longitudinal direction.
        @param lat_size size of grid in the latittudinal direction.
        @param rad_size size of grid in the radial direction.
    */
    void resize(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size, 
        std::size_t rad_size);

    /**
        @brief Evaluate spherical harmonic expansion on a grid.

        @param expansion spherical harmonics expansion.
        @param longitudes longitude values defining the grid points.
        @param colatitudes colatitude values defining the grid points.
        @param radii radius values defining the grid points.

        @return Three dimensional array containing values of the expansion on
        the grid with shape `{longitudes.size(), colatitudes.size(),
        radii.size()}` in row-major order.
    */
    template <ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase>
    [[nodiscard]] DynamicMDArray<double, 3> evaluate(
        ZernikeSpan<const double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes, std::span<const double> radii)
    {
        if (longitudes.size() == 0 || colatitudes.size() == 0 || radii.size() == 0)
            return {};

        const std::size_t order = expansion.order();
        resize(order, longitudes.size(), colatitudes.size(), radii.size());

        RadialZernikeSpan<double, zernike_norm, std::dynamic_extent>
        zernike(m_zernike_grid, order, m_rad_size);

        m_zernike_recursion.generate<zernike_norm>(radii, zernike);

        for (std::size_t i = 0; i < m_lat_size; ++i)
            m_cos_colat[i] = std::cos(colatitudes[i]);

        st::AssociatedLegendreSpan<double, sh_norm, sh_phase, std::dynamic_extent>
        ass_leg(m_ass_leg_grid, order, m_lat_size);

        m_ass_leg_recursion.generate_real(m_cos_colat, ass_leg);

        MDSpan<double, std::dynamic_extent, std::dynamic_extent, 2> cossin_lon(
                m_cossin_lon_grid.data(), std::array{order, m_lon_size});
        zest::detail::recursive_trig(cossin_lon, longitudes);

        sum_n(expansion);
        sum_l(order);

        DynamicMDArray<double, 3> res{m_lon_size, m_lat_size, m_rad_size};
        sum_m(res, order);

        return res;
    }

    template <ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase>
    [[nodiscard]] DynamicMDArray<double, 3> evaluate(
        ZernikeSpan<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes, std::span<const double> radii)
    {
        using ExpansionType = ZernikeSpan<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase>;
        return evaluate((typename ExpansionType::const_view)(expansion), longitudes, colatitudes, radii);
    }

    template <ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase>
    [[nodiscard]] DynamicMDArray<double, 3> evaluate(
        const ZernikeExpansion<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase>& expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes, std::span<const double> radii)
    {
        using ExpansionType = ZernikeExpansion<double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase>;
        return evaluate((typename ExpansionType::const_view)(expansion), longitudes, colatitudes, radii);
    }

private:
    template <ZernikeNorm zernike_norm, st::SHNorm sh_norm, st::SHPhase sh_phase>
    void sum_n(ZernikeSpan<const double, IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion) noexcept
    {
        const std::size_t order = expansion.order();
        RadialZernikeSpan<const double, zernike_norm, std::dynamic_extent>
        zernike(m_zernike_grid, order, m_rad_size);

        std::ranges::fill(m_flm_grid, 0.0);

        TriangleSpan<double, IndexingMode::zero_based, std::dynamic_extent, 2>
        flm(m_flm_grid, order, m_rad_size);

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
                    auto coeff = expansion_nl[m];
                    for (std::size_t i = 0; i < m_rad_size; ++i)
                    {
                        flm_lm[i, 0] += zernike_nl[i]*coeff[0];
                        flm_lm[i, 1] += zernike_nl[i]*coeff[1];
                    }
                }
            }
        }
    }

    void sum_l(std::size_t order) noexcept;

    void sum_m(DynamicMDSpan<double, 3> values, std::size_t order) noexcept;

    RadialZernikeRecursion m_zernike_recursion;
    st::AssociatedLegendreRecursion m_ass_leg_recursion;
    std::vector<double> m_zernike_grid;
    std::vector<double> m_ass_leg_grid;
    std::vector<double> m_cos_colat;
    std::vector<double> m_cossin_lon_grid;
    std::vector<double> m_flm_grid;
    std::vector<std::array<double, 2>> m_fm_grid;
    std::size_t m_lon_size{};
    std::size_t m_lat_size{};
    std::size_t m_rad_size{};
    std::size_t m_max_order{};
};

} // namespace zt

} // namespace zest
