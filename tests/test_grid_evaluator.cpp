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
#include "grid_evaluator.hpp"
#include "sequence.hpp"

#include <cassert>
#include <cmath>

namespace
{

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

template <zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_sh_grid_evaluator_does_constant_function()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };

    std::vector<double> longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(
            0.0, std::numbers::pi, num_lat);

    zest::DynamicMDArray<double, 2> test_grid{num_lon, num_lat};

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            test_grid[i, j] = function(
                    longitudes[i], colatitudes[j]);
        }
    }

    zest::st::SHExpansion<double, zest::IndexingMode::zero_based, sh_norm, sh_phase> expansion(order);

    expansion[0, 0, 0] = 1.0;

    const auto grid = zest::st::GridEvaluator(order)
        .evaluate(expansion, longitudes, colatitudes);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            if(!is_close(grid[i, j], test_grid[i, j], tol))
                success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", grid[i, j]);
            }
            std::printf("\n");
        }
    }

    return success;
}

template <zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_sh_grid_evaluator_does_Y10()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::numbers::sqrt3*z;
    };

    std::vector<double> longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(
            0.0, std::numbers::pi, num_lat);

    zest::DynamicMDArray<double, 2> test_grid{num_lon, num_lat};

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            test_grid[i, j] = function(
                    longitudes[i], colatitudes[j]);
        }
    }

    zest::st::SHExpansion<double, zest::IndexingMode::zero_based, sh_norm, sh_phase> expansion(order);

    expansion[1, 0, 0] = 1.0;

    auto grid = zest::st::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            if(!is_close(grid[i, j], test_grid[i, j], tol))
                success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", grid[i, j]);
            }
            std::printf("\n");
        }
        std::printf("\n");

        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", test_grid[i, j]);
            }
            std::printf("\n");
        }
        std::printf("\n");
    }

    return success;
}

template <zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_sh_grid_evaluator_does_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*(std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon));
    };

    std::vector<double> longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(
            0.0, std::numbers::pi, num_lat);

    zest::DynamicMDArray<double, 2> test_grid{num_lon, num_lat};

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            test_grid[i, j] = function(
                    longitudes[i], colatitudes[j]);
        }
    }

    zest::st::SHExpansion<double, zest::IndexingMode::zero_based, sh_norm, sh_phase> expansion(order);

    expansion[3, 1, 0] = 1.0;
    expansion[4, 3, 1] = 1.0;

    auto grid = zest::st::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            if(!is_close(grid[i, j], test_grid[i, j], tol))
                success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", grid[i, j]);
            }
            std::printf("\n");
        }
        std::printf("\n");

        for (std::size_t i = 0; i < num_lon; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                std::printf("%f ", test_grid[i, j]);
            }
            std::printf("\n");
        }
        std::printf("\n");
    }

    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_zernike_grid_evaluator_does_constant_function()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 14;
    constexpr std::size_t num_lat = 7;
    constexpr std::size_t num_rad = 7;

    auto function = []([[maybe_unused]] double r, [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ? std::numbers::sqrt3 : 1.0;
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm;
    };

    std::vector<double> longitudes = linspace(0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(0.0, std::numbers::pi, num_lat);
    std::vector<double> radii = linspace(0.0, 1.0, num_rad);

    zest::DynamicMDArray<double, 3> test_grid{num_lon, num_lat, num_rad};

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                test_grid[i, j, k] = function(radii[k], longitudes[i], colatitudes[j]);
        }
    }

    zest::zt::ZernikeExpansion<double, zest::IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion(order);

    expansion[0, 0, 0, 0] = 1.0;

    auto grid = zest::zt::GridEvaluator(order)
        .evaluate(expansion, longitudes, colatitudes, radii);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                if(!is_close(grid[i, j, k], test_grid[i, j, k], tol))
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
                    std::printf("%f ", grid[i, j, k]);
                std::printf("\n");
            }
            std::printf("\n");
        }
    }

    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_zernike_grid_evaluator_does_Z33m2_plus_Z531()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 14;
    constexpr std::size_t num_lat = 7;
    constexpr std::size_t num_rad = 7;

    zest::DynamicMDArray<double, 3> test_grid{num_lon, num_lat, num_rad};

    auto function = [](double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double znorm3
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ? 3.0 : 1.0;
        const double znorm5
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ? std::sqrt(13.0) : 1.0;
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::unit) ?
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
                test_grid[i, j, k] = function(radii[k], longitudes[i], colatitudes[j]);
        }
    }

    zest::zt::ZernikeExpansion<double, zest::IndexingMode::zero_based, zernike_norm, sh_norm, sh_phase> expansion(order);

    expansion[3, 3, 2, 1] = 1.0;
    expansion[5, 3, 1, 0] = 1.0;

    auto grid = zest::zt::GridEvaluator(order)
        .evaluate(expansion, longitudes, colatitudes, radii);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                if(!is_close(grid[i, j, k], test_grid[i, j, k], tol))
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
                    std::printf("%f ", test_grid[i, j, k]);
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
                    std::printf("%f ", grid[i, j, k]);
                std::printf("\n");
            }
            std::printf("\n");
        }
    }

    return success;
}

template <zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
void test_sh_grid_evaluator()
{
    assert((test_sh_grid_evaluator_does_constant_function<sh_norm, sh_phase>()));
    assert((test_sh_grid_evaluator_does_Y10<sh_norm, sh_phase>()));
    assert((test_sh_grid_evaluator_does_Y31_plus_Y4m3<sh_norm, sh_phase>()));
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
void test_zernike_grid_evaluator()
{
    assert((test_zernike_grid_evaluator_does_constant_function<zernike_norm, sh_norm, sh_phase>()));
    assert((test_zernike_grid_evaluator_does_Z33m2_plus_Z531<zernike_norm, sh_norm, sh_phase>()));
}

} // namespace

int main()
{
    test_sh_grid_evaluator<zest::st::SHNorm::four_pi, zest::st::SHPhase::none>();
    test_sh_grid_evaluator<zest::st::SHNorm::four_pi, zest::st::SHPhase::cs>();
    test_sh_grid_evaluator<zest::st::SHNorm::unit, zest::st::SHPhase::none>();
    test_sh_grid_evaluator<zest::st::SHNorm::unit, zest::st::SHPhase::cs>();

    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::normed, zest::st::SHNorm::four_pi, zest::st::SHPhase::none>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::normed, zest::st::SHNorm::four_pi, zest::st::SHPhase::cs>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::normed, zest::st::SHNorm::unit, zest::st::SHPhase::none>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::normed, zest::st::SHNorm::unit, zest::st::SHPhase::cs>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::unnormed, zest::st::SHNorm::four_pi, zest::st::SHPhase::none>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::unnormed, zest::st::SHNorm::four_pi, zest::st::SHPhase::cs>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::unnormed, zest::st::SHNorm::unit, zest::st::SHPhase::none>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::unnormed, zest::st::SHNorm::unit, zest::st::SHPhase::cs>();
}
