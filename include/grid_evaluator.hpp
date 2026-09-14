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
#include "sequence.hpp"
#include "sh_concepts.hpp"
#include "sh_expansion.hpp"
#include "triangle_spans.hpp"
#include "zernike_concepts.hpp"
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

        @param max_order Maximum order of spherical harmonic expansion.
    */
    explicit GridEvaluator(std::size_t max_order);

    /**
        @brief Reserves memory for a combination of expansion and grid size.

        @param max_order Maximum order of spherical harmonic expansion.
        @param lon_size Size of grid in the longitudinal direction.
        @param lat_size Size of grid in the latittudinal direction.
    */
    GridEvaluator(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size);

    /**
        @brief Resize for a combination of expansion and grid size.

        @param max_order Maximum order of spherical harmonic expansion.
        @param lon_size Size of grid in the longitudinal direction.
        @param lat_size Size of grid in the latittudinal direction.
    */
    void resize(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size);

    /**
        @brief Evaluate spherical harmonic expansion on a grid.

        @param expansion Spherical harmonics expansion.
        @param longitudes Longitude values defining the grid points.
        @param colatitudes Colatitude values defining the grid points.
        @param values Values of the spherical harmonic expansion on the grid.
    */
    template <sh_expansion<Indexing::zero_based> ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
            && st::has_inner_rank<ExpansionType, 0>
    void evaluate(
        const ExpansionType& expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes,
        DynamicMDSpan<double, 2> values)
    {
        if (longitudes.size() == 0 || colatitudes.size() == 0)
            return;

        for (const auto& element : colatitudes)
            assert(0.0 <= element && element <= std::numbers::pi);

        const std::size_t order = expansion.order();
        resize(order, longitudes.size(), colatitudes.size());

        for (std::size_t i = 0; i < m_lat_size; ++i)
            m_cos_colat[i] = std::cos(colatitudes[i]);

        constexpr st::SHNorm sh_norm = sh_norm_of<ExpansionType>();
        constexpr st::SHPhase sh_phase = sh_phase_of<ExpansionType>();

        AssociatedLegendreSpan<double, st::SHConvention<sh_norm, sh_phase>, std::dynamic_extent>
        ass_leg(m_ass_leg_grid, order, m_lat_size);

        m_ass_leg_recursion.generate_real(m_cos_colat, ass_leg);

        MDSpan<double, std::dynamic_extent, std::dynamic_extent, 2> cossin_lon(
            m_cossin_lon_grid, std::array<std::size_t, 3>{order, m_lon_size, 2});
        zest::detail::recursive_trig(cossin_lon, longitudes);

        sum_l(expansion);

        sum_m(values, order);
    }

    /**
        @brief Evaluate spherical harmonic expansion on a grid.

        @param expansion Spherical harmonics expansion.
        @param longitudes Longitude values defining the grid points.
        @param colatitudes Colatitude values defining the grid points.
    */
    template <sh_expansion<Indexing::zero_based> ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
            && st::has_inner_rank<ExpansionType, 0>
    [[nodiscard]] DynamicMDArray<double, 2> evaluate(
        const ExpansionType& expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes)
    {
        DynamicMDArray<double, 2> values{longitudes.size(), colatitudes.size()};
        evaluate(expansion, longitudes, colatitudes, values);
        return values;
    }

private:
    template <sh_expansion<Indexing::zero_based> ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
    void sum_l(const ExpansionType& expansion) noexcept
    {
        constexpr st::SHNorm sh_norm = sh_norm_of<ExpansionType>();
        constexpr st::SHPhase sh_phase = sh_phase_of<ExpansionType>();

        AssociatedLegendreSpan<const double, st::SHConvention<sh_norm, sh_phase>, std::dynamic_extent>
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

        @param max_order Maximum order of Zernike Expansion.
    */
    explicit GridEvaluator(std::size_t max_order);

    /**
        @brief Reserves memory for a combination of expansion and grid size.

        @param max_order Maximum order of Zernike expansion.
        @param lon_size Size of grid in the longitudinal direction.
        @param lat_size Size of grid in the latittudinal direction.
        @param rad_size Size of grid in the radial direction.
    */
    GridEvaluator(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size, 
        std::size_t rad_size);

    /**
        @brief Resize for a combination of expansion and grid size.

        @param max_order Maximum order of Zernike expansion.
        @param lon_size Size of grid in the longitudinal direction.
        @param lat_size Size of grid in the latittudinal direction.
        @param rad_size Size of grid in the radial direction.
    */
    void resize(
        std::size_t max_order, std::size_t lon_size, std::size_t lat_size, 
        std::size_t rad_size);

    /**
        @brief Evaluate spherical harmonic expansion on a grid.

        @param expansion Zernike Expansion.
        @param longitudes Longitude values defining the grid points.
        @param colatitudes Colatitude values defining the grid points.
        @param radii Radius values defining the grid points.
        @param values Values of the Zernike expansion on the grid.
    */
    template <zernike_expansion<Indexing::zero_based> ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
            && zt::has_inner_rank<ExpansionType, 0>
    void evaluate(
        const ExpansionType& expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes,
        std::span<const double> radii, DynamicMDSpan<double, 3> values)
    {
        assert(
            values.extent(0) == longitudes.size()
            && values.extent(1) == colatitudes.size()
            && values.extent(2) == radii.size());

        if (longitudes.size() == 0 || colatitudes.size() == 0 || radii.size() == 0)
            return;

        for (double element : colatitudes)
            assert(0.0 <= element && element <= std::numbers::pi);

        for (double element : radii)
            assert(0.0 <= element && element <= 1.0);

        const std::size_t order = expansion.order();
        resize(order, longitudes.size(), colatitudes.size(), radii.size());

        constexpr st::SHNorm sh_norm = st::sh_norm_of<ExpansionType>();
        constexpr st::SHPhase sh_phase = st::sh_phase_of<ExpansionType>();
        constexpr ZernikeNorm zernike_norm = zernike_norm_of<ExpansionType>();

        RadialZernikeSpan<double, ZernikeNormConvention<zernike_norm>, std::dynamic_extent>
        zernike(m_zernike_grid, order, m_rad_size);

        m_zernike_recursion.generate<ZernikeNormConvention<zernike_norm>>(radii, zernike);

        for (std::size_t i = 0; i < m_lat_size; ++i)
            m_cos_colat[i] = std::cos(colatitudes[i]);

        st::AssociatedLegendreSpan<double, st::SHConvention<sh_norm, sh_phase>, std::dynamic_extent>
        ass_leg(m_ass_leg_grid, order, m_lat_size);

        m_ass_leg_recursion.generate_real(m_cos_colat, ass_leg);

        MDSpan<double, std::dynamic_extent, std::dynamic_extent, 2> cossin_lon(
                m_cossin_lon_grid.data(), std::array{order, m_lon_size});
        zest::detail::recursive_trig(cossin_lon, longitudes);

        sum_n(expansion);
        sum_l(order);

        sum_m(values, order);
    }

    /**
        @brief Evaluate spherical harmonic expansion on a grid.

        @param expansion Zernike Expansion.
        @param longitudes Longitude values defining the grid points.
        @param colatitudes Colatitude values defining the grid points.
        @param radii Radius values defining the grid points.
        @param values Values of the Zernike expansion on the grid.
    */
    template <zernike_expansion<Indexing::zero_based> ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
            && zt::has_inner_rank<ExpansionType, 0>
    [[nodiscard]] DynamicMDArray<double, 3> evaluate(
        const ExpansionType& expansion,
        std::span<const double> longitudes, std::span<const double> colatitudes,
        std::span<const double> radii)
    {
        DynamicMDArray<double, 3> values{longitudes.size(), colatitudes.size(), radii.size()};
        evaluate(expansion, longitudes, colatitudes, radii, values);
        return values;
    }

private:
    template <zernike_expansion<Indexing::zero_based> ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
    void sum_n(const ExpansionType& expansion) noexcept
    {
        constexpr zt::ZernikeNorm zernike_norm = zernike_norm_of<ExpansionType>();

        const std::size_t order = expansion.order();
        RadialZernikeSpan<const double, ZernikeNormConvention<zernike_norm>, std::dynamic_extent>
        zernike(m_zernike_grid, order, m_rad_size);

        std::ranges::fill(m_flm_grid, 0.0);

        TriangleSpan<double, Indexing::zero_based, std::dynamic_extent, 2>
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

class IsotropicGridEvaluator
{
public:
    IsotropicGridEvaluator() = default;

    /**
        @brief Reserves memory for an expansion of given order.

        @param max_order Maximum order of spherical harmonic expansion.
    */
    explicit IsotropicGridEvaluator(std::size_t max_order);

    /**
        @brief Reserves memory for a combination of expansion and grid size.

        @param max_order Maximum order of spherical harmonic expansion.
        @param rad_size Size of grid in the radial direction.
    */
    IsotropicGridEvaluator(std::size_t max_order, std::size_t rad_size);

    /**
        @brief Resize for a combination of expansion and grid size.

        @param max_order Maximum order of spherical harmonic expansion.
        @param rad_size Size of grid in the radial direction.
    */
    void resize(std::size_t max_order, std::size_t rad_size);

    /**
        @brief Evaluate spherical harmonic expansion on a grid.

        @param expansion Isotropic Zernike expansion.
        @param radii Radius values defining the grid points.
        @param values Values of the Zernike expansion at the radial points.
    */
    template <typename ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
            && zt::has_inner_rank<ExpansionType, 0>
    void evaluate(
        const ExpansionType& expansion, std::span<const double> radii, std::span<double> values)
    {
        assert(radii.size() == values.size());
        if (values.size() == 0)
            return;

        for (double element : radii)
            assert(0.0 <= element && element <= 1.0);

        // We generate the values in chunks of 256 to maximize cache
        // utilization.
        constexpr std::size_t chunk_size = 256;
        constexpr double spherical_harmonic = (ExpansionType::sh_norm == zest::st::SHNorm::four_pi) ?
            1.0 : 0.5*std::numbers::inv_sqrtpi;
        const std::size_t chunk_count = radii.size()/chunk_size;

        for (std::size_t i = 0; i < chunk_count; ++i)
        {
            std::span<double, chunk_size> chunk{values.data() + i*chunk_size, chunk_size};
            std::span<const double, chunk_size> point_chunk{radii.data() + i*chunk_size, chunk_size};

            m_zernike_recursion.init<norm_convention_of<ExpansionType>>(point_chunk);

            for (auto n : expansion.indices())
            {
                auto radial_zernike = m_zernike_recursion.current();
                const double element = spherical_harmonic*expansion[n];
                for (std::size_t j = 0; j < chunk_size; ++j)
                    chunk[j] += element*radial_zernike[j];

                m_zernike_recursion.iterate<norm_convention_of<ExpansionType>>();
            }
        }

        const std::size_t remainder = radii.size() - chunk_count*chunk_size;
        std::span<double> chunk{values.data() + chunk_count*chunk_size, remainder};
        std::span<const double> point_chunk{radii.data() + chunk_count*chunk_size, remainder};

        m_zernike_recursion.init<norm_convention_of<ExpansionType>>(point_chunk);

        for (auto n : expansion.indices())
        {
            auto radial_zernike = m_zernike_recursion.current();
            const double element = spherical_harmonic*expansion[n];
            for (std::size_t i = 0; i < remainder; ++i)
                chunk[i] += element*radial_zernike[i];

            m_zernike_recursion.iterate<norm_convention_of<ExpansionType>>();
        }
    }

    /**
        @brief Evaluate spherical harmonic expansion on a grid.

        @param expansion Isotropic Zernike expansion.
        @param radii Radius values defining the grid points.
        @param values Values of the Zernike expansion at the radial points.
    */
    template <typename ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
            && zt::has_inner_rank<ExpansionType, 0>
    [[nodiscard]] std::vector<double> evaluate(
        const ExpansionType& expansion, std::span<const double> radii)
    {
        std::vector<double> values(radii.size());
        evaluate(expansion, radii, values);
        return values;
    }

private:
    IsotropicRadialZernikeRecursion m_zernike_recursion;
};

} // namespace zt

} // namespace zest
