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

bool test_sh_grid_evaluator_does_constant_function()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;
    
    std::vector<double> test_grid(num_lon*num_lat);

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            test_grid[i*num_lat + j] = 1.0;
        }
    }
    
    zest::st::RealSHExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>
    expansion(order);
    
    expansion(0,0)[0] = 1.0;

    std::vector<double> longitudes = linspace(0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(0.0, std::numbers::pi, num_lon);

    const auto& grid = zest::st::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes);
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            if(!is_close(grid[i*num_lat + j], 1.0, tol))
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

bool test_sh_grid_evaluator_does_Y10()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(3.0)*z;
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
    
    zest::st::RealSHExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>
    expansion(order);
    
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

bool test_sh_grid_evaluator_does_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
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
    
    zest::st::RealSHExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>
    expansion(order);
    
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

bool test_zernike_grid_evaluator_does_constant_function()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 14;
    constexpr std::size_t num_lat = 7;
    constexpr std::size_t num_rad = 7;
    
    std::vector<double> test_grid(num_lon*num_lat*num_rad);

    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                test_grid[(i*num_lat + j)*num_rad + k] = std::sqrt(3.0);
        }
    }
    
    zest::zt::ZernikeExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE> 
    expansion(order);
    
    expansion(0,0,0)[0] = 1.0;

    std::vector<double> longitudes = linspace(0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(0.0, std::numbers::pi, num_lat);
    std::vector<double> radii = linspace(0.0, 1.0, num_rad);

    const auto& grid = zest::zt::GridEvaluator(order).evaluate(
            expansion, longitudes, colatitudes, radii);
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_lon; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_rad; ++k)
                if(!is_close(grid[(i*num_lat + j)*num_rad + k], std::sqrt(3.0), tol))
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
        return std::sqrt(9.0)*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon) + std::sqrt(13.0)*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
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
    
    zest::zt::ZernikeExpansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE> 
    expansion(order);

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

int main()
{
    assert(test_sh_grid_evaluator_does_constant_function());
    assert(test_sh_grid_evaluator_does_Y10());
    assert(test_sh_grid_evaluator_does_Y31_plus_Y4m3());
    assert(test_zernike_grid_evaluator_does_constant_function());
    assert(test_zernike_grid_evaluator_does_Z33m2_plus_Z531());
}