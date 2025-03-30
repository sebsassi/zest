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

constexpr bool test_sphereglqgridspan_const_view_can_be_taken()
{
    zest::st::SphereGLQGridSpan<double> span{};
    [[maybe_unused]] auto const_view
        = zest::st::SphereGLQGridSpan<const double>(span);
    return true;
}

static_assert(test_sphereglqgridspan_const_view_can_be_taken());

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_forward_transform_expands_Y00()
{
    constexpr std::size_t order = 6; 

    auto function = [](
        [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };
    
    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid
        = points.generate_values(function, order);
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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_forward_transform_expands_Y10()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(3.0)*z;
    };
    
    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid
        = points.generate_values(function, order);
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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_forward_transform_expands_Y21()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? 
            -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };
    
    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid
        = points.generate_values(function, order);
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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_forward_transform_expands_Y31()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? 
            -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };
    
    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid
        = points.generate_values(function, order);
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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_forward_transform_expands_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? 
            -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };
    
    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid
        = points.generate_values(function, order);
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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_forward_transform_expands_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? 
            -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*(std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon));
    };
    
    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid
        = points.generate_values(function, order);
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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_backward_transform_evaluates_Y00()
{
    constexpr std::size_t order = 6; 

    auto function = [](
        [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid
        = points.generate_values(function, order);

    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_backward_transform_evaluates_Y10()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(3.0)*z;
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid
        = points.generate_values(function, order);

    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_backward_transform_evaluates_Y21()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? 
            -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid
        = points.generate_values(function, order);

    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_backward_transform_evaluates_Y31()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? 
            -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid
        = points.generate_values(function, order);

    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_backward_transform_evaluates_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? 
            -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid
        = points.generate_values(function, order);

    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

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

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
bool test_glq_backward_transform_evaluates_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6;

    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? 
            -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*(std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon));
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid
        = points.generate_values(function, order);

    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

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

template <typename GridLayout, zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_sh_transform_converges()
{
    constexpr std::size_t order = 100;

    auto function = [](double lon, double colat)
    {
        std::array<double, 3> x = {
            std::sin(colat)*std::cos(lon) - 0.3,
            std::sin(colat)*std::sin(lon) + 0.1,
            std::cos(colat) - 0.043
        };
        std::array<double, 3> y = {
            0.5, 0.5, 0.5
        };
        return std::exp(-(x[0]*y[0] + x[1]*y[1] + x[2]*y[2]));
    };

    zest::st::SphereGLQGridPoints<GridLayout> points{};

    zest::st::SphereGLQGrid<double, GridLayout> test_grid
        = points.generate_values(function, order);

    zest::st::GLQTransformer<sh_norm_param, sh_phase_param, GridLayout> 
    transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < grid.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < grid.shape()[1]; ++j)
            if (!is_close(grid(i, j), test_grid(i, j), tol))
                success = false;
    }

}

template <
    typename GridLayout, zest::st::SHNorm sh_norm_param,
    zest::st::SHPhase sh_phase_param>
void test_glq()
{
    assert((test_glq_forward_transform_expands_Y00<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_forward_transform_expands_Y10<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_forward_transform_expands_Y21<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_forward_transform_expands_Y31<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_forward_transform_expands_Y4m3<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_forward_transform_expands_Y31_plus_Y4m3<GridLayout, sh_norm_param, sh_phase_param>()));

    assert((test_glq_backward_transform_evaluates_Y00<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_backward_transform_evaluates_Y10<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_backward_transform_evaluates_Y21<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_backward_transform_evaluates_Y31<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_backward_transform_evaluates_Y4m3<GridLayout, sh_norm_param, sh_phase_param>()));
    assert((test_glq_backward_transform_evaluates_Y31_plus_Y4m3<GridLayout, sh_norm_param, sh_phase_param>()));

    assert((test_sh_transform_converges<GridLayout, sh_norm_param, sh_phase_param>()));
}

int main()
{
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::geo, zest::st::SHPhase::none>();
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::geo, zest::st::SHPhase::none>();

    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::qm, zest::st::SHPhase::none>();
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::geo, zest::st::SHPhase::cs>();
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::qm, zest::st::SHPhase::cs>();
}