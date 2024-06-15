#include "grid_evaluator.hpp"

#include <cmath>
#include <utility>
#include <stdexcept>
#include <ranges>
#include <algorithm>

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

void recursive_trig(
    MDSpan<std::array<double, 2>, 2> trigs, std::span<const double> angles)
{
    MDSpan<std::array<double, 2>, 1> trigs_0 = trigs[0];
    for (std::size_t i = 0; i < angles.size(); ++i)
        trigs_0[i] = {1.0, 0.0};
    
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

namespace st
{

GridEvaluator::GridEvaluator(
    std::size_t lmax, std::size_t lon_size, std::size_t lat_size):
    m_plm_recursion(lmax), m_plm_grid(TriangleLayout::size(lmax)*lat_size),
    m_cos_colat(lat_size), m_cossin_lon_grid((lmax + 1)*lon_size),
    m_fm_grid((lmax + 1)*lat_size), m_lon_size(lon_size),
    m_lat_size(lat_size), m_lmax(lmax) {}


void GridEvaluator::resize(
    std::size_t lmax, std::size_t lon_size, std::size_t lat_size)
{
    if (lmax != m_lmax)
        m_plm_recursion.expand(lmax);

    if (lmax != m_lmax || lat_size != m_lat_size)
    {
        m_plm_grid.resize(TriangleLayout::size(lmax)*lat_size);
        m_fm_grid.resize((lmax + 1)*lat_size);
    }

    if (lmax != m_lmax || lon_size != m_lon_size)
        m_cossin_lon_grid.resize((lmax + 1)*lon_size);

    if (lat_size != m_lat_size)
        m_cos_colat.resize(lat_size);
    
    m_lmax = lmax;
    m_lon_size = lon_size;
    m_lat_size = lat_size;
}

void GridEvaluator::sum_m(MDSpan<double, 2> values)
{
    MDSpan<const std::array<double, 2>, 2> cossin_lon(
        m_cossin_lon_grid.data(), {m_lmax, m_lon_size});
    
    for (std::size_t m = 0; m <= m_lmax; ++m)
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

}

namespace zt
{

GridEvaluator::GridEvaluator(
    std::size_t lmax, std::size_t lon_size, std::size_t lat_size, 
    std::size_t rad_size):
    m_zernike_recursion(lmax), m_plm_recursion(lmax),
    m_zernike_grid(RadialZernikeLayout::size(lmax)*rad_size),
    m_plm_grid(TriangleLayout::size(lmax)*lat_size),
    m_cos_colat(lat_size), m_cossin_lon_grid((lmax + 1)*lon_size),
    m_flm_grid(TriangleLayout::size(lmax)*rad_size),
    m_fm_grid((lmax + 1)*lat_size*rad_size), m_lon_size(lon_size),
    m_lat_size(lat_size), m_rad_size(rad_size), m_lmax(lmax) {}

void GridEvaluator::resize(
    std::size_t lmax, std::size_t lon_size, std::size_t lat_size, 
    std::size_t rad_size)
{
    if (lmax != m_lmax)
    {
        m_zernike_recursion.expand(lmax);
        m_plm_recursion.expand(lmax);
    }

    if (lon_size != m_lon_size || lmax != m_lmax)
        m_cossin_lon_grid.resize((lmax + 1)*lon_size);
    
    if (lat_size != m_lat_size)
        m_cos_colat.resize(lat_size);

    if (lat_size != m_lat_size || lmax != m_lmax)
    {
        m_zernike_grid.resize(RadialZernikeLayout::size(lmax)*rad_size);
        m_plm_grid.resize(TriangleLayout::size(lmax)*lat_size);
    }

    if (rad_size != m_rad_size || lmax != m_lmax)
        m_flm_grid.resize(TriangleLayout::size(lmax)*rad_size);
    
    if (rad_size != m_rad_size || lat_size != m_lat_size || lmax != m_lmax)
        m_fm_grid.resize((lmax + 1)*lat_size*rad_size);
    
    m_lmax = lmax;
    m_lon_size = lon_size;
    m_lat_size = lat_size;
    m_rad_size = rad_size;
}

void GridEvaluator::sum_l()
{
    TriangleVecSpan<const std::array<double, 2>, TriangleLayout>
    flm(m_flm_grid, m_lmax, m_rad_size);
    
    TriangleVecSpan<const double, TriangleLayout> ass_leg(
            m_plm_grid, m_lmax, m_lat_size);

    std::ranges::fill(m_fm_grid, std::array<double, 2>{});
    MDSpan<std::array<double, 2>, 3> fm(
            m_fm_grid, {m_lmax, m_lat_size, m_rad_size});
    
    for (std::size_t l = 0; l <= m_lmax; ++l)
    {
        for (std::size_t m = 0; m <= l; ++m)
        {
            std::span<const std::array<double, 2>> flm_lm = flm(l,m);
            std::span<const double> plm = ass_leg(l,m);
            
            MDSpan<std::array<double, 2>, 2> fm_m = fm[m];
            for (std::size_t i = 0; i < m_lat_size; ++i)
            {
                const double weight = plm[i];
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

void GridEvaluator::sum_m(MDSpan<double, 3> values)
{
    MDSpan<const std::array<double, 2>, 2> cossin_lon(
        m_cossin_lon_grid.data(), {m_lmax, m_lon_size});

    MDSpan<const std::array<double, 2>, 3> fm(
            m_fm_grid, {m_lmax, m_lat_size, m_rad_size});
    
    for (std::size_t m = 0; m <= m_lmax; ++m)
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

}

}