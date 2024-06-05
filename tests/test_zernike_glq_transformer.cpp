#include "zernike_glq_transformer.hpp"

#include <cassert>

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
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

bool test_glq_geo_expansion_expands_Z000()
{
    std::size_t lmax = 5;

    auto function = [](
        [[maybe_unused]] double r, [[maybe_unused]] double lon, 
        [[maybe_unused]] double colat)
    {
        return std::sqrt(3.0);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid grid = points.generate_values(function);
    zest::zt::GLQTransformer transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 0)
        {
            if (is_close(coeffs[i][0], 1.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (const auto& coeff : expansion.coeffs())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

bool test_glq_geo_expansion_expands_Z110()
{
    std::size_t lmax = 5;

    auto function = [](double r, [[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(5.0)*r*std::sqrt(3.0)*z;
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid grid = points.generate_values(function);
    zest::zt::GLQTransformer transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 1)
        {
            if (is_close(coeffs[i][0], 1.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (const auto& coeff : expansion.coeffs())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

bool test_glq_geo_expansion_expands_Z200()
{
    std::size_t lmax = 5;

    auto function = [](
        double r, [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        return std::sqrt(7.0)*(2.5*r*r - 1.5);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid grid = points.generate_values(function);
    zest::zt::GLQTransformer transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 3)
        {
            if (is_close(coeffs[i][0], 1.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (const auto& coeff : expansion.coeffs())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

bool test_glq_geo_expansion_expands_Z221()
{
    std::size_t lmax = 5;

    auto function = [](
        double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(7.0)*r*r*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid grid = points.generate_values(function);
    zest::zt::GLQTransformer transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 5)
        {
            if (is_close(coeffs[i][0], 1.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (const auto& coeff : expansion.coeffs())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

bool test_glq_geo_expansion_expands_Z33m2()
{
    std::size_t lmax = 5;

    auto function = [](
        double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(9.0)*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid grid = points.generate_values(function);
    zest::zt::GLQTransformer transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 11)
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], 1.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (const auto& coeff : expansion.coeffs())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

bool test_glq_geo_expansion_expands_Z531()
{
    std::size_t lmax = 5;

    auto function = [](
        double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(13.0)*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid grid = points.generate_values(function);
    zest::zt::GLQTransformer transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 25)
        {
            if (is_close(coeffs[i][0], 1.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (const auto& coeff : expansion.coeffs())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

bool test_glq_geo_evaluates_Z000()
{
    constexpr std::size_t lmax = 5; 

    auto function = []([[maybe_unused]] double r, [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        return 1.0;
    };
    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid test_grid = points.generate_values(function);

    zest::zt::GLQTransformer transformer(lmax);

    auto expansion = transformer.transform(test_grid, lmax);
    auto grid = transformer.transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
        {
            for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                if (!is_close(grid(i, j, k), test_grid(i, j, k), tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", test_grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
    return success;
}

bool test_glq_geo_evaluates_Z110()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](double r, [[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(5.0)*r*std::sqrt(3.0)*z;
    };
    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid test_grid = points.generate_values(function);

    zest::zt::GLQTransformer transformer(lmax);

    auto expansion = transformer.transform(test_grid, lmax);
    auto grid = transformer.transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
        {
            for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                if (!is_close(grid(i, j, k), test_grid(i, j, k), tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", test_grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
    return success;
}

bool test_glq_geo_evaluates_Z200()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](
        double r, [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        return std::sqrt(7.0)*(2.5*r*r - 1.5);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid test_grid = points.generate_values(function);

    zest::zt::GLQTransformer transformer(lmax);

    auto expansion = transformer.transform(test_grid, lmax);
    auto grid = transformer.transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
        {
            for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                if (!is_close(grid(i, j, k), test_grid(i, j, k), tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", test_grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
    return success;
}

bool test_glq_geo_evaluates_Z221()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](
        double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(7.0)*r*r*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid test_grid = points.generate_values(function);

    zest::zt::GLQTransformer transformer(lmax);

    auto expansion = transformer.transform(test_grid, lmax);
    auto grid = transformer.transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
        {
            for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                if (!is_close(grid(i, j, k), test_grid(i, j, k), tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", test_grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
    return success;
}

bool test_glq_geo_evaluates_Z33m2()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](
        double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(9.0)*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid test_grid = points.generate_values(function);

    zest::zt::GLQTransformer transformer(lmax);

    auto expansion = transformer.transform(test_grid, lmax);
    auto grid = transformer.transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
        {
            for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                if (!is_close(grid(i, j, k), test_grid(i, j, k), tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", test_grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
    return success;
}

bool test_glq_geo_evaluates_Z531()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](
        double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(13.0)*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points(lmax);
    zest::zt::BallGLQGrid test_grid = points.generate_values(function);

    zest::zt::GLQTransformer transformer(lmax);

    auto expansion = transformer.transform(test_grid, lmax);
    auto grid = transformer.transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
        {
            for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                if (!is_close(grid(i, j, k), test_grid(i, j, k), tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", test_grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            {
                for (std::size_t k = 0; k < grid.shape()[2]; ++k)
                    std::printf("%f ", grid(i, j, k));
                std::printf("\n");
            }
            std::printf("\n");
        }
    }
    return success;
}

bool test_uniform_grid_evaluator_does_constant_function()
{
    constexpr std::size_t lmax = 5;
    constexpr std::size_t num_lon = 14;
    constexpr std::size_t num_lat = 7;
    constexpr std::size_t num_rad = 7;
    
    std::vector<double> test_grid(num_lon*num_lat*num_rad);

    for (std::size_t i = 0; i < num_rad; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_lon; ++k)
                test_grid[(i*num_lat + j)*num_lon + k] = std::sqrt(3.0);
        }
    }
    
    zest::zt::ZernikeExpansion expansion(lmax);
    
    expansion(0,0,0)[0] = 1.0;

    const auto& [radii, longitudes, latitudes, grid]
        = zest::zt::UniformGridEvaluator{}.evaluate(
                expansion, {num_rad, num_lat, num_lon});
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_rad; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_lon; ++k)
                if(!is_close(grid[(i*num_lat + j)*num_lon + k], std::sqrt(3.0), tol))
                    success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < num_rad; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                for (std::size_t k = 0; k < num_lon; ++k)
                    std::printf("%f ", grid[(i*num_lat + j)*num_lon + k]);
                std::printf("\n");
            }
            std::printf("\n");
        }
    }

    return success;
}

bool test_uniform_grid_evaluator_does_Z33m2_plus_Z531()
{
    constexpr std::size_t lmax = 5;
    constexpr std::size_t num_lon = 14;
    constexpr std::size_t num_lat = 7;
    constexpr std::size_t num_rad = 7;
    
    std::vector<double> test_grid(num_lon*num_lat*num_rad);

    auto function = [](double r, double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(9.0)*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon) + std::sqrt(13.0)*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    std::vector<double> test_radii = linspace(
            0.0, 1.0, num_rad);
    std::vector<double> test_longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> test_colatitudes = linspace(
            0.0, std::numbers::pi, num_lat);
    
    for (std::size_t i = 0; i < num_rad; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_lon; ++k)
                test_grid[(i*num_lat + j)*num_lon + k] = function(test_radii[i], test_longitudes[k], test_colatitudes[j]);
        }
    }
    
    zest::zt::ZernikeExpansion expansion(lmax);

    expansion(3,3,2) = {0.0, 1.0};
    expansion(5,3,1) = {1.0, 0.0};

    const auto& [radii, longitudes, latitudes, grid]
        = zest::zt::UniformGridEvaluator{}.evaluate(
                expansion, {num_rad, num_lat, num_lon});
    
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < num_rad; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            for (std::size_t k = 0; k < num_lon; ++k)
                if(!is_close(grid[(i*num_lat + j)*num_lon + k], test_grid[(i*num_lat + j)*num_lon + k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::printf("test grid\n");
        for (std::size_t i = 0; i < num_rad; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                for (std::size_t k = 0; k < num_lon; ++k)
                    std::printf("%f ", test_grid[(i*num_lat + j)*num_lon + k]);
                std::printf("\n");
            }
            std::printf("\n");
        }

        std::printf("grid\n");
        for (std::size_t i = 0; i < num_rad; ++i)
        {
            for (std::size_t j = 0; j < num_lat; ++j)
            {
                for (std::size_t k = 0; k < num_lon; ++k)
                    std::printf("%f ", grid[(i*num_lat + j)*num_lon + k]);
                std::printf("\n");
            }
            std::printf("\n");
        }
    }

    return success;
}

int main()
{
    assert(test_glq_geo_expansion_expands_Z000());
    assert(test_glq_geo_expansion_expands_Z110());
    assert(test_glq_geo_expansion_expands_Z200());
    assert(test_glq_geo_expansion_expands_Z221());
    assert(test_glq_geo_expansion_expands_Z33m2());
    assert(test_glq_geo_expansion_expands_Z531());

    assert(test_glq_geo_evaluates_Z000());
    assert(test_glq_geo_evaluates_Z110());
    assert(test_glq_geo_evaluates_Z200());
    assert(test_glq_geo_evaluates_Z221());
    assert(test_glq_geo_evaluates_Z33m2());
    assert(test_glq_geo_evaluates_Z531());
}