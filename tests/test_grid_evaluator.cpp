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

template <zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_sh_grid_evaluator_does_constant_function()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
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
    
    zest::st::RealSHExpansion<NORM, PHASE> expansion(order);
    
    expansion(0,0)[0] = 1.0;

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
    }

    return success;
}

template <zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_sh_grid_evaluator_does_Y10()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
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
    
    zest::st::RealSHExpansion<NORM, PHASE> expansion(order);
    
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

template <zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_sh_grid_evaluator_does_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 30;
    constexpr std::size_t num_lat = 15;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
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
    
    zest::st::RealSHExpansion<NORM, PHASE> expansion(order);
    
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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_zernike_grid_evaluator_does_constant_function()
{
    constexpr std::size_t order = 6;
    constexpr std::size_t num_lon = 14;
    constexpr std::size_t num_lat = 7;
    constexpr std::size_t num_rad = 7;

    auto function = []([[maybe_unused]] double r, [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ? std::numbers::sqrt3 : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
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
    
    zest::zt::ZernikeExpansion<ZERNIKE_NORM, SH_NORM, PHASE> expansion(order);
    
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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
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
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double znorm3
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ? 3.0 : 1.0;
        constexpr double znorm5
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ? std::sqrt(13.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
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
    
    zest::zt::ZernikeExpansion<ZERNIKE_NORM, SH_NORM, PHASE> expansion(order);

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

template <zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
void test_sh_grid_evaluator()
{
    assert((test_sh_grid_evaluator_does_constant_function<NORM, PHASE>()));
    assert((test_sh_grid_evaluator_does_Y10<NORM, PHASE>()));
    assert((test_sh_grid_evaluator_does_Y31_plus_Y4m3<NORM, PHASE>()));
}

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
void test_zernike_grid_evaluator()
{
    assert((test_zernike_grid_evaluator_does_constant_function<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_zernike_grid_evaluator_does_Z33m2_plus_Z531<ZERNIKE_NORM, SH_NORM, PHASE>()));
}

int main()
{
    test_sh_grid_evaluator<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>();
    test_sh_grid_evaluator<zest::st::SHNorm::GEO, zest::st::SHPhase::CS>();
    test_sh_grid_evaluator<zest::st::SHNorm::QM, zest::st::SHPhase::NONE>();
    test_sh_grid_evaluator<zest::st::SHNorm::QM, zest::st::SHPhase::CS>();

    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::CS>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::QM, zest::st::SHPhase::NONE>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::QM, zest::st::SHPhase::CS>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::UNNORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::UNNORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::CS>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::UNNORMED, zest::st::SHNorm::QM, zest::st::SHPhase::NONE>();
    test_zernike_grid_evaluator<zest::zt::ZernikeNorm::UNNORMED, zest::st::SHNorm::QM, zest::st::SHPhase::CS>();
}