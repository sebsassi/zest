#include "sh_glq_transformer.hpp"

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

template <typename GridLayout>
bool test_glq_geo_expansion_expands_Y00()
{
    constexpr std::size_t lmax = 5; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        return 1.0;
    };
    
    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    zest::st::SphereGLQGridPoints points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values<GridLayout>(function, lmax);
    auto expansion = transformer.forward_transform(grid, lmax);

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
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_expansion_expands_Y10()
{
    constexpr std::size_t lmax = 5; 

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(3.0)*z;
    };
    
    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    zest::st::SphereGLQGridPoints points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values<GridLayout>(function, lmax);
    auto expansion = transformer.forward_transform(grid, lmax);

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
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_expansion_expands_Y21()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };
    
    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    zest::st::SphereGLQGridPoints points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values<GridLayout>(function, lmax);
    auto expansion = transformer.forward_transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 4)
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
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_expansion_expands_Y31()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };
    
    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    zest::st::SphereGLQGridPoints points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values<GridLayout>(function, lmax);
    auto expansion = transformer.forward_transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 7)
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
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_expansion_expands_Y4m3()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };
    
    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    zest::st::SphereGLQGridPoints points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values<GridLayout>(function, lmax);
    auto expansion = transformer.forward_transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 13)
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
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_expansion_expands_Y31_plus_Y4m3()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };
    
    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    zest::st::SphereGLQGridPoints points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values<GridLayout>(function, lmax);
    auto expansion = transformer.forward_transform(grid, lmax);

    const auto& coeffs = expansion.coeffs();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 7)
        {
            if (is_close(coeffs[i][0], 1.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else if (i == 13)
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
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_evaluates_Y00()
{
    constexpr std::size_t lmax = 5; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        return 1.0;
    };

    zest::st::SphereGLQGridPoints points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values<GridLayout>(function, lmax);

    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            if (!is_close(grid(i, j), test_grid(i, j), tol))
                success = false;
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", test_grid(i, j));
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", grid(i, j));
            std::printf("\n");
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_evaluates_Y10()
{
    constexpr std::size_t lmax = 5; 

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(3.0)*z;
    };

    zest::st::SphereGLQGridPoints points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values<GridLayout>(function, lmax);

    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            if (!is_close(grid(i, j), test_grid(i, j), tol))
                success = false;
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", test_grid(i, j));
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", grid(i, j));
            std::printf("\n");
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_evaluates_Y21()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::st::SphereGLQGridPoints points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values<GridLayout>(function, lmax);

    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            if (!is_close(grid(i, j), test_grid(i, j), tol))
                success = false;
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", test_grid(i, j));
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", grid(i, j));
            std::printf("\n");
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_evaluates_Y31()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::st::SphereGLQGridPoints points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values<GridLayout>(function, lmax);

    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            if (!is_close(grid(i, j), test_grid(i, j), tol))
                success = false;
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", test_grid(i, j));
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", grid(i, j));
            std::printf("\n");
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_evaluates_Y4m3()
{
    constexpr std::size_t lmax = 5; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };

    zest::st::SphereGLQGridPoints points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values<GridLayout>(function, lmax);

    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            if (!is_close(grid(i, j), test_grid(i, j), tol))
                success = false;
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", test_grid(i, j));
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", grid(i, j));
            std::printf("\n");
        }
    }
    return success;
}

template <typename GridLayout>
bool test_glq_geo_evaluates_Y31_plus_Y4m3()
{
    constexpr std::size_t lmax = 5;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        return std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };

    zest::st::SphereGLQGridPoints points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values<GridLayout>(function, lmax);

    zest::st::GLQTransformerGeo<GridLayout> transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            if (!is_close(grid(i, j), test_grid(i, j), tol))
                success = false;
    }

    if (!success)
    {
        std::printf("test_grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", test_grid(i, j));
            std::printf("\n");
        }
        std::printf("grid\n");
        for (std::size_t i = 0; i < grid.shape()[0]; ++i)
        {
            for (std::size_t j = 0; j < grid.shape()[1]; ++j)
                std::printf("%f ", grid(i, j));
            std::printf("\n");
        }
    }
    return success;
}

template <typename GridLayout>
void test_glq()
{
    assert(test_glq_geo_expansion_expands_Y00<GridLayout>());
    assert(test_glq_geo_expansion_expands_Y10<GridLayout>());
    assert(test_glq_geo_expansion_expands_Y21<GridLayout>());
    assert(test_glq_geo_expansion_expands_Y31<GridLayout>());
    assert(test_glq_geo_expansion_expands_Y4m3<GridLayout>());
    assert(test_glq_geo_expansion_expands_Y31_plus_Y4m3<GridLayout>());

    assert(test_glq_geo_evaluates_Y00<GridLayout>());
    assert(test_glq_geo_evaluates_Y10<GridLayout>());
    assert(test_glq_geo_evaluates_Y21<GridLayout>());
    assert(test_glq_geo_evaluates_Y31<GridLayout>());
    assert(test_glq_geo_evaluates_Y4m3<GridLayout>());
    assert(test_glq_geo_evaluates_Y31_plus_Y4m3<GridLayout>());
}

int main()
{
    test_glq<zest::st::LatLonLayout<>>();
    test_glq<zest::st::LonLatLayout<>>();
}