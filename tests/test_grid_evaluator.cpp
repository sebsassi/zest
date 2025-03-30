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

#include <random>
#include <cmath>
#include <cassert>


constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::fabs(a[0] - b[0]) < tol && std::fabs(a[1] - b[1]) < tol;
}

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

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_sh_grid_evaluator_does_constant_function()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };

    std::vector<double> longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(
            0.0, std::numbers::pi, num_lat);
    
    std::vector<double> test_grid(num_lon*num_lat);

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            test_grid[i*num_lat + j] = function(
                    longitudes[i], colatitudes[j]);
        }
    }
    
    zest::st::RealSHExpansion<sh_norm_param, sh_phase_param> expansion(order);
    
    expansion(0,0)[0] = 1.0;

    const auto grid = zest::st::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes);
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            if(!is_close(grid[i*num_lat + j], test_grid[i*num_lat + j], tol))
                success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", grid[i*num_lat + j]);
            }
            std::printf("\n");
        }
    }

    return success;
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_sh_grid_evaluator_does_Y10()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(3.0)*z;
    };

    std::vector<double> longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(
            0.0, std::numbers::pi, num_lat);
    
    std::vector<double> test_grid(num_lon*num_lat);

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            test_grid[i*num_lat + j] = function(
                    longitudes[i], colatitudes[j]);
        }
    }
    
    zest::st::RealSHExpansion<sh_norm_param, sh_phase_param> expansion(order);
    
    expansion(1,0) = {1.0, 0.0};

    const auto& grid = zest::st::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes);
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            if(!is_close(grid[i*num_lat + j], test_grid[i*num_lat + j], tol))
                success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", grid[i*num_lat + j]);
            }
            std::printf("\n");
        }
        std::printf("\n");

        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", test_grid[i*num_lat + j]);
            }
            std::printf("\n");
        }
        std::printf("\n");
    }

    return success;
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_sh_grid_evaluator_does_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*(std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon));
    };

    std::vector<double> longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(
            0.0, std::numbers::pi, num_lat);
    
    std::vector<double> test_grid(num_lon*num_lat);

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            test_grid[i*num_lat + j] = function(
                    longitudes[i], colatitudes[j]);
        }
    }
    
    zest::st::RealSHExpansion<sh_norm_param, sh_phase_param> expansion(order);
    
    expansion(3,1) = {1.0, 0.0};
    expansion(4,3) = {0.0, 1.0};

    const auto& grid = zest::st::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes);
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            if(!is_close(grid[i*num_lat + j], test_grid[i*num_lat + j], tol))
                success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", grid[i*num_lat + j]);
            }
            std::printf("\n");
        }
        std::printf("\n");

        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", test_grid[i*num_lat + j]);
            }
            std::printf("\n");
        }
        std::printf("\n");
    }

    return success;
}

template <zest::zt::ZernikeNorm zernike_norm_param, zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_zernike_grid_evaluator_does_constant_function()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 14;
    constexpr std::size_t num_lat = 7;
    constexpr std::size_t num_rad = 7;

    auto function = []([[maybe_unused]] double r, [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double znorm
            = (zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::numbers::sqrt3 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm;
    };

    std::vector<double> longitudes = linspace(0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(0.0, std::numbers::pi, num_lat);
    std::vector<double> radii = linspace(0.0, 1.0, num_rad);
    
    std::vector<double> test_grid(num_lon*num_lat*num_rad);

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                test_grid[(i*num_lat + j)*num_rad + k] = function(radii[k], longitudes[i], colatitudes[j]);
        }
    }
    
    zest::zt::RealZernikeExpansion<zernike_norm_param, sh_norm_param, sh_phase_param> expansion(order);
    
    expansion(0,0,0)[0] = 1.0;

    const auto& grid = zest::zt::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes, radii);
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                if(!is_close(grid[(i*num_lat + j)*num_rad + k], test_grid[(i*num_lat + j)*num_rad + k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                for (std::size_t k = 0; k < num_rad; ++k)
                    std::printf("%f ", grid[(i*num_lat + j)*num_rad + k]);
                std::printf("\n");
            }
            std::printf("\n");
        }
    }

    return success;
}

template <zest::zt::ZernikeNorm zernike_norm_param, zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_zernike_grid_evaluator_does_Z33m2_plus_Z531()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 14;
    constexpr std::size_t num_lat = 7;
    constexpr std::size_t num_rad = 7;
    
    std::vector<double> test_grid(num_lon*num_lat*num_rad);

    auto function = [](double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double znorm3
            = (zernike_norm_param == zest::zt::ZernikeNorm::normed) ? 3.0 : 1.0;
        constexpr double znorm5
            = (zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(13.0) : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*(
            znorm3*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon)
            + znorm5*phase*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon));
    };

    std::vector<double> longitudes = linspace(0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(0.0, std::numbers::pi, num_lat);
    std::vector<double> radii = linspace(0.0, 1.0, num_rad);
    
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                test_grid[(i*num_lat + j)*num_rad + k] = function(radii[k], longitudes[i], colatitudes[j]);
        }
    }
    
    zest::zt::RealZernikeExpansion<zernike_norm_param, sh_norm_param, sh_phase_param> expansion(order);

    expansion(3,3,2) = {0.0, 1.0};
    expansion(5,3,1) = {1.0, 0.0};

    const auto& grid = zest::zt::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes, radii);
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                if(!is_close(grid[(i*num_lat + j)*num_rad + k], test_grid[(i*num_lat + j)*num_rad + k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::printf("test grid\n");
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                for (std::size_t k = 0; k < num_rad; ++k)
                    std::printf("%f ", test_grid[(i*num_lat + j)*num_rad + k]);
                std::printf("\n");
            }
            std::printf("\n");
        }

        std::printf("grid\n");
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                for (std::size_t k = 0; k < num_rad; ++k)
                    std::printf("%f ", grid[(i*num_lat + j)*num_rad + k]);
                std::printf("\n");
            }
            std::printf("\n");
        }
    }

    return success;
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
void test_sh_grid_evaluator()
{
    assert((test_sh_grid_evaluator_does_constant_function<sh_norm_param, sh_phase_param>()));
    assert((test_sh_grid_evaluator_does_Y10<sh_norm_param, sh_phase_param>()));
    assert((test_sh_grid_evaluator_does_Y31_plus_Y4m3<sh_norm_param, sh_phase_param>()));
}

template <zest::zt::ZernikeNorm zernike_norm_param, zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
void test_zernike_grid_evaluator()
{
    assert((test_zernike_grid_evaluator_does_constant_function<zernike_norm_param, sh_norm_param, sh_phase_param>()));
    assert((test_zernike_grid_evaluator_does_Z33m2_plus_Z531<zernike_norm_param, sh_norm_param, sh_phase_param>()));
}

int main()
{
    test_sh_grid_evaluator<zest::st::SHNorm::geo, zest::st::SHPhase::none>();
    test_sh_grid_evaluator<zest::st::SHNorm::geo, zest::st::SHPhase::cs>();
    test_sh_grid_evaluator<zest::st::SHNorm::qm, zest::st::SHPhase::none>();
    test_sh_grid_evaluator<zest::st::SHNorm::qm, zest::st::SHPhase::cs>();

    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::cs>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::normed, zest::st::SHNorm::qm, zest::st::SHPhase::none>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::normed, zest::st::SHNorm::qm, zest::st::SHPhase::cs>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::unnormed, zest::st::SHNorm::geo, zest::st::SHPhase::none>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::unnormed, zest::st::SHNorm::geo, zest::st::SHPhase::cs>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::unnormed, zest::st::SHNorm::qm, zest::st::SHPhase::none>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::unnormed, zest::st::SHNorm::qm, zest::st::SHPhase::cs>();
}