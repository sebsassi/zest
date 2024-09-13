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

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Y00()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };
    
    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    const double reference_coeff = 1.0;

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 0)
        {
            if (is_close(coeffs[i][0], reference_coeff, tol)
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
        for (std::size_t l = 0; l <= order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Y10()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(3.0)*z;
    };
    
    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    const double reference_coeff = 1.0;

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 1)
        {
            if (is_close(coeffs[i][0], reference_coeff, tol)
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
        for (std::size_t l = 0; l <= order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Y21()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };
    
    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    const double reference_coeff = 1.0;

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 4)
        {
            if (is_close(coeffs[i][0], reference_coeff, tol)
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
        for (std::size_t l = 0; l <= order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Y31()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };
    
    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    const double reference_coeff = 1.0;

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 7)
        {
            if (is_close(coeffs[i][0], reference_coeff, tol)
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
        for (std::size_t l = 0; l <= order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };
    
    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    const double reference_coeff = 1.0;

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 13)
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], reference_coeff, tol))
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
        for (std::size_t l = 0; l <= order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*(std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon));
    };
    
    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    const double reference_coeff = 1.0;

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 7)
        {
            if (is_close(coeffs[i][0], reference_coeff, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else if (i == 13)
        {
            if (is_close(coeffs[i][0], 0.0, tol)
                    && is_close(coeffs[i][1], reference_coeff, tol))
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
        for (std::size_t l = 0; l <= order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf("%lu %lu %f %f\n", l, m, expansion(l,m)[0], expansion(l,m)[1]);
        }
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Y00()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values(function, order);

    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Y10()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(3.0)*z;
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values(function, order);

    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Y21()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values(function, order);

    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Y31()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values(function, order);

    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values(function, order);

    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? -1.0 : 1.0;
        constexpr double shnorm = (NORM == zest::st::SHNorm::QM) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*(std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon));
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid = points.generate_values(function, order);

    zest::st::GLQTransformer<NORM, PHASE, GridLayout> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <typename GridLayout, zest::st::SHNorm NORM, zest::st::SHPhase PHASE>
void test_glq()
{
    assert((test_glq_forward_transform_expands_Y00<GridLayout, NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Y10<GridLayout, NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Y21<GridLayout, NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Y31<GridLayout, NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Y4m3<GridLayout, NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Y31_plus_Y4m3<GridLayout, NORM, PHASE>()));

    assert((test_glq_backward_transform_evaluates_Y00<GridLayout, NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Y10<GridLayout, NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Y21<GridLayout, NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Y31<GridLayout, NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Y4m3<GridLayout, NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Y31_plus_Y4m3<GridLayout, NORM, PHASE>()));
}

int main()
{
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>();
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>();

    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::QM, zest::st::SHPhase::NONE>();
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::GEO, zest::st::SHPhase::CS>();
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::QM, zest::st::SHPhase::CS>();
}