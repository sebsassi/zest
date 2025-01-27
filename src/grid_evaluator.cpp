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
#include "grid_evaluator.hpp"

namespace zest
{

template <std::floating_point T>
std::vector<T> linspace(T start, T stop, std::size_t count)
{
    if (count == 0) return {};
    if (count == 1) return {start};

    std::vector<T> res(count);
    const T step = (stop - start)/T(count - 1);
    for (std::size_t i = 0; i < count - 1; ++i)
        res[i] = start + T(i)*step;
    
    res[count - 1] = stop;

    return res;
}

namespace detail
{

void recursive_trig(
    MDSpan<std::array<double, 2>, 2> trigs, std::span<const double> angles) noexcept
{
    MDSpan<std::array<double, 2>, 1> trigs_0 = trigs[0];
    for (std::size_t i = 0; i < angles.size(); ++i)
        trigs_0[i] = {1.0, 0.0};
    
    if (trigs.extents()[0] == 1) return;

    MDSpan<std::array<double, 2>, 1> trigs_1 = trigs[1];
    for (std::size_t i = 0; i < angles.size(); ++i)
        trigs_1[i] = {std::cos(angles[i]), std::sin(angles[i])};
    
    for (std::size_t i = 2; i < trigs.extents()[0]; ++i)
    {
        MDSpan<std::array<double, 2>, 1> trigs_prev = trigs[i - 1];
        MDSpan<std::array<double, 2>, 1> trigs_next = trigs[i];
        for (std::size_t j = 0; j < angles.size(); ++j)
        {
            trigs_next[j] = {
                trigs_prev[j][0]*trigs_1[j][0] - trigs_prev[j][1]*trigs_1[j][1],
                trigs_prev[j][0]*trigs_1[j][1] + trigs_prev[j][1]*trigs_1[j][0]
            };
        }
    }
}

} // namespace detail

namespace st
{

GridEvaluator::GridEvaluator(std::size_t max_order):
    m_plm_recursion(max_order), m_lon_size{}, m_lat_size{},
    m_max_order(max_order) {}

GridEvaluator::GridEvaluator(
    std::size_t max_order, std::size_t lon_size, std::size_t lat_size):
    m_plm_recursion(max_order), m_plm_grid(PlmLayout::size(max_order)*lat_size),
    m_cos_colat(lat_size), m_cossin_lon_grid(max_order*lon_size),
    m_fm_grid(max_order*lat_size), m_lon_size(lon_size),
    m_lat_size(lat_size), m_max_order(max_order) {}


void GridEvaluator::resize(
    std::size_t max_order, std::size_t lon_size, std::size_t lat_size)
{
    if (m_max_order < max_order)
        m_plm_recursion.expand(max_order);

    if (m_max_order < max_order || lat_size != m_lat_size)
    {
        m_plm_grid.resize(PlmLayout::size(max_order)*lat_size);
        m_fm_grid.resize(max_order*lat_size);
    }

    if (m_max_order < max_order || lon_size != m_lon_size)
        m_cossin_lon_grid.resize(max_order*lon_size);

    if (lat_size != m_lat_size)
        m_cos_colat.resize(lat_size);
    
    m_max_order = max_order;
    m_lon_size = lon_size;
    m_lat_size = lat_size;
}

void GridEvaluator::sum_m(MDSpan<double, 2> values, std::size_t order) noexcept
{
    MDSpan<const std::array<double, 2>, 2> cossin_lon(
        m_cossin_lon_grid.data(), {order, m_lon_size});
    
    for (std::size_t m = 0; m < order; ++m)
    {
        std::span<std::array<double, 2>> f_m(
                m_fm_grid.begin() + m*m_lat_size, m_lat_size);
        MDSpan<const std::array<double, 2>, 1> cossin_lon_m
            = cossin_lon[m];
        for (std::size_t i = 0; i < m_lon_size; ++i)
        {
            MDSpan<double, 1> values_i = values[i];
            const double cos_lon = cossin_lon_m[i][0];
            const double sin_lon = cossin_lon_m[i][1];
            for (std::size_t j = 0; j < m_lat_size; ++j)
            {
                values_i[j] += f_m[j][0]*cos_lon + f_m[j][1]*sin_lon;
            }
        }
    }
}

} // namespace st

namespace zt
{

GridEvaluator::GridEvaluator(std::size_t max_order):
    m_zernike_recursion(max_order), m_plm_recursion(max_order), m_lon_size{}, 
    m_lat_size{}, m_rad_size{}, m_max_order(max_order) {}


GridEvaluator::GridEvaluator(
    std::size_t max_order, std::size_t lon_size, std::size_t lat_size, 
    std::size_t rad_size):
    m_zernike_recursion(max_order), m_plm_recursion(max_order),
    m_zernike_grid(RadialZernikeLayout::size(max_order)*rad_size),
    m_plm_grid(st::PlmLayout::size(max_order)*lat_size),
    m_cos_colat(lat_size), m_cossin_lon_grid(max_order*lon_size),
    m_flm_grid(st::PlmLayout::size(max_order)*rad_size),
    m_fm_grid(max_order*lat_size*rad_size), m_lon_size(lon_size),
    m_lat_size(lat_size), m_rad_size(rad_size), m_max_order(max_order) {}

void GridEvaluator::resize(
    std::size_t max_order, std::size_t lon_size, std::size_t lat_size, 
    std::size_t rad_size)
{
    if (max_order < m_max_order)
    {
        m_zernike_recursion.expand(max_order);
        m_plm_recursion.expand(max_order);
    }

    if (lon_size != m_lon_size || max_order < m_max_order)
        m_cossin_lon_grid.resize(max_order*lon_size);
    
    if (lat_size != m_lat_size)
        m_cos_colat.resize(lat_size);

    if (lat_size != m_lat_size || max_order < m_max_order)
    {
        m_zernike_grid.resize(RadialZernikeLayout::size(max_order)*rad_size);
        m_plm_grid.resize(st::PlmLayout::size(max_order)*lat_size);
    }

    if (rad_size != m_rad_size || max_order < m_max_order)
        m_flm_grid.resize(st::PlmLayout::size(max_order)*rad_size);
    
    if (rad_size != m_rad_size || lat_size != m_lat_size || max_order < m_max_order)
        m_fm_grid.resize(max_order*lat_size*rad_size);
    
    m_max_order = max_order;
    m_lon_size = lon_size;
    m_lat_size = lat_size;
    m_rad_size = rad_size;
}

void GridEvaluator::sum_l(std::size_t order) noexcept
{
    TriangleVecSpan<
        const std::array<double, 2>, TriangleLayout<IndexingMode::nonnegative>>
    flm(m_flm_grid, order, m_rad_size);
    
    TriangleVecSpan<const double, st::PlmLayout>
    ass_leg(m_plm_grid.data(), order, m_lat_size);

    std::ranges::fill(m_fm_grid, std::array<double, 2>{});
    MDSpan<std::array<double, 2>, 3> fm(
            m_fm_grid.data(), {order, m_lat_size, m_rad_size});
    
    for (auto l : flm.indices())
    {
        auto ass_leg_l = ass_leg[l];
        auto flm_l = flm[l];
        for (auto m : flm_l.indices())
        {
            std::span<const std::array<double, 2>> flm_lm = flm_l[m];
            std::span<const double> ass_leg_lm = ass_leg_l[m];
            
            MDSpan<std::array<double, 2>, 2> fm_m = fm[m];
            for (std::size_t i = 0; i < m_lat_size; ++i)
            {
                const double weight = ass_leg_lm[i];
                MDSpan<std::array<double, 2>, 1> fm_mi = fm_m[i];
                for (std::size_t j = 0; j < m_rad_size; ++j)
                {
                    fm_mi[j][0] += weight*flm_lm[j][0];
                    fm_mi[j][1] += weight*flm_lm[j][1];
                }
            }
        }
    }
}

void GridEvaluator::sum_m(MDSpan<double, 3> values, std::size_t order) noexcept
{
    MDSpan<const std::array<double, 2>, 2> cossin_lon(
        m_cossin_lon_grid.data(), {order, m_lon_size});

    MDSpan<const std::array<double, 2>, 3> fm(
            m_fm_grid.data(), {order, m_lat_size, m_rad_size});
    
    for (std::size_t m = 0; m < order; ++m)
    {
        MDSpan<const std::array<double, 2>, 1> cossin_lon_m
            = cossin_lon[m];
        MDSpan<const std::array<double, 2>, 2> fm_m = fm[m];
        for (std::size_t i = 0; i < m_lon_size; ++i)
        {
            MDSpan<double, 2> values_i = values[i];
            const double cos_lon = cossin_lon_m[i][0];
            const double sin_lon = cossin_lon_m[i][1];
            for (std::size_t j = 0; j < m_lat_size; ++j)
            {
                MDSpan<const std::array<double, 2>, 1> fm_mj = fm_m[j];
                MDSpan<double, 1> values_ij = values_i[j];
                for (std::size_t k = 0; k < m_rad_size; ++k)
                    values_ij[k] += fm_mj[k][0]*cos_lon + fm_mj[k][1]*sin_lon;
            }
        }
    }
}

} // namespace zt

} // namespace zest