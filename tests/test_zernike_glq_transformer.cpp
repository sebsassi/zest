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
#include "zernike_glq_transformer.hpp"

#include <cassert>
#include <print>

namespace
{

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

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_forward_transform_expands_Z000()
{
    std::size_t order = 6;

    auto function = [](
        [[maybe_unused]] double lon, [[maybe_unused]] double colat,
        [[maybe_unused]] double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                std::numbers::sqrt3 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm;
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
            {
                if (n == 0 && l == 0 && m == 0)
                {
                    if (is_close(expansion_nl[m, 0], reference_coeff, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_nl[m, 0], 0.0, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                for (auto m : expansion_nl.indices())
                {
                    std::println(
                            "({}, {}, {}) {} {}", n, l, m,
                            expansion_nl[m, 0], expansion_nl[m, 1]);
                }
            }
        }
    }
    return success;
}

template <
    zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_forward_transform_expands_Z200()
{
    constexpr double sqrt7 = 2.6457513110645905905016158;

    std::size_t order = 6;

    auto function = [](
        [[maybe_unused]] double lon, [[maybe_unused]] double colat, double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt7 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm*(2.5*r*r - 1.5);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
            {
                if (n == 2 && l == 0 && m == 0)
                {
                    if (is_close(expansion_nl[m, 0], reference_coeff, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_nl[m, 0], 0.0, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                for (auto m : expansion_nl.indices())
                {
                    std::println(
                            "({}, {}, {}) {} {}", n, l, m,
                            expansion_nl[m, 0], expansion_nl[m, 1]);
                }
            }
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_forward_transform_expands_Z110()
{
    constexpr double sqrt5 = 2.2360679774997896964091737;
    std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, double colat, double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt5 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        const double z = std::cos(colat);
        return znorm*shnorm*r*std::numbers::sqrt3*z;
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
            {
                if (n == 1 && l == 1 && m == 0)
                {
                    if (is_close(expansion_nl[m, 0], reference_coeff, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_nl[m, 0], 0.0, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                for (auto m : expansion_nl.indices())
                {
                    std::println(
                            "({}, {}, {}) {} {}", n, l, m,
                            expansion_nl[m, 0], expansion_nl[m, 1]);
                }
            }
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_forward_transform_expands_Z221()
{
    constexpr double sqrt7 = 2.6457513110645905905016158;

    std::size_t order = 6;

    auto function = [](double lon, double colat, double r)
    {
        constexpr double phase = (Convention::sh_phase == zest::st::SHPhase::none) ?
            -1.0 : 1.0;
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt7 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        const double z = std::cos(colat);
        return phase*znorm*shnorm*r*r*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
            {
                if (n == 2 && l == 2 && m == 1)
                {
                    if (is_close(expansion_nl[m, 0], reference_coeff, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_nl[m, 0], 0.0, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                for (auto m : expansion_nl.indices())
                {
                    std::println(
                            "({}, {}, {}) {} {}", n, l, m,
                            expansion_nl[m, 0], expansion_nl[m, 1]);
                }
            }
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_forward_transform_expands_Z33m2()
{
    std::size_t order = 6;

    auto function = [](double lon, double colat, double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ? 3.0 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        const double z = std::cos(colat);
        return znorm*shnorm*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
            {
                if (n == 3 && l == 3 && m == 2)
                {
                    if (is_close(expansion_nl[m, 0], 0.0, tol)
                            && is_close(expansion_nl[m, 1], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_nl[m, 0], 0.0, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                for (auto m : expansion_nl.indices())
                {
                    std::println(
                            "({}, {}, {}) {} {}", n, l, m,
                            expansion_nl[m, 0], expansion_nl[m, 1]);
                }
            }
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_forward_transform_expands_Z531()
{
    constexpr double sqrt13 = 3.6055512754639892931192213;

    std::size_t order = 6;

    auto function = [](double lon, double colat, double r)
    {
        constexpr double phase = (Convention::sh_phase == zest::st::SHPhase::none) ?
            -1.0 : 1.0;
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt13 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        const double z = std::cos(colat);
        return phase*znorm*shnorm*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        auto expansion_n = expansion[n];
        for (auto l : expansion_n.indices())
        {
            auto expansion_nl = expansion_n[l];
            for (auto m : expansion_nl.indices())
            {
                if (n == 5 && l == 3 && m == 1)
                {
                    if (is_close(expansion_nl[m, 0], reference_coeff, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_nl[m, 0], 0.0, tol)
                            && is_close(expansion_nl[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            auto expansion_n = expansion[n];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                for (auto m : expansion_nl.indices())
                {
                    std::println(
                            "({}, {}, {}) {} {}", n, l, m,
                            expansion_nl[m, 0], expansion_nl[m, 1]);
                }
            }
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_backward_transform_evaluates_Z000()
{
    constexpr std::size_t order = 6;

    auto function = [](
        [[maybe_unused]] double lon, [[maybe_unused]] double colat,
        [[maybe_unused]] double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                std::numbers::sqrt3 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm;
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.extent(0); ++i)
    {
        for (std::size_t j = 0; j < grid.extent(1); ++j)
        {
            for (std::size_t k = 0; k < grid.extent(2); ++k)
                if (!is_close(grid[i, j, k], test_grid[i, j, k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", test_grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
        std::println("grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_backward_transform_evaluates_Z110()
{
    constexpr double sqrt5 = 2.2360679774997896964091737;

    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, double colat, double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt5 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        const double z = std::cos(colat);
        return znorm*shnorm*r*std::numbers::sqrt3*z;
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.extent(0); ++i)
    {
        for (std::size_t j = 0; j < grid.extent(1); ++j)
        {
            for (std::size_t k = 0; k < grid.extent(2); ++k)
                if (!is_close(grid[i, j, k], test_grid[i, j, k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", test_grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
        std::println("grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_backward_transform_evaluates_Z200()
{
    constexpr double sqrt7 = 2.6457513110645905905016158;

    constexpr std::size_t order = 6;

    auto function = [](
        [[maybe_unused]] double lon, [[maybe_unused]] double colat, double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt7 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm*(2.5*r*r - 1.5);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.extent(0); ++i)
    {
        for (std::size_t j = 0; j < grid.extent(1); ++j)
        {
            for (std::size_t k = 0; k < grid.extent(2); ++k)
                if (!is_close(grid[i, j, k], test_grid[i, j, k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", test_grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
        std::println("grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_backward_transform_evaluates_Z221()
{
    constexpr double sqrt7 = 2.6457513110645905905016158;

    constexpr std::size_t order = 6;

    auto function = [](double lon, double colat, double r)
    {
        constexpr double phase = (Convention::sh_phase == zest::st::SHPhase::none) ?
            -1.0 : 1.0;
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt7 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        const double z = std::cos(colat);
        return phase*znorm*shnorm*r*r*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.extent(0); ++i)
    {
        for (std::size_t j = 0; j < grid.extent(1); ++j)
        {
            for (std::size_t k = 0; k < grid.extent(2); ++k)
                if (!is_close(grid[i, j, k], test_grid[i, j, k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", test_grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
        std::println("grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_backward_transform_evaluates_Z33m2()
{
    constexpr std::size_t order = 6; 

    auto function = [](double lon, double colat, double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ? 3.0 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        const double z = std::cos(colat);
        return znorm*shnorm*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.extent(0); ++i)
    {
        for (std::size_t j = 0; j < grid.extent(1); ++j)
        {
            for (std::size_t k = 0; k < grid.extent(2); ++k)
                if (!is_close(grid[i, j, k], test_grid[i, j, k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", test_grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
        std::println("grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_glq_backward_transform_evaluates_Z531()
{
    constexpr double sqrt13 = 3.6055512754639892931192213;

    constexpr std::size_t order = 6;

    auto function = [](double lon, double colat, double r)
    {
        constexpr double phase = (Convention::sh_phase == zest::st::SHPhase::none) ?
            -1.0 : 1.0;
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt13 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        const double z = std::cos(colat);
        return phase*znorm*shnorm*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.extent(0); ++i)
    {
        for (std::size_t j = 0; j < grid.extent(1); ++j)
        {
            for (std::size_t k = 0; k < grid.extent(2); ++k)
                if (!is_close(grid[i, j, k], test_grid[i, j, k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", test_grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
        std::println("grid");
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                    std::print("{} ", grid[i, j, k]);
                std::println("");
            }
            std::println("");
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_zernike_transform_converges()
{
    constexpr std::size_t order = 100;

    auto function = [](double lon, double colat, double r)
    {
        std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon) - 0.3,
            r*std::sin(colat)*std::sin(lon) + 0.1,
            r*std::cos(colat) - 0.043
        };
        return std::exp(-(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]));
    };


    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < grid.extent(0); ++i)
    {
        for (std::size_t j = 0; j < grid.extent(1); ++j)
        {
            for (std::size_t k = 0; k < grid.extent(2); ++k)
                if (!is_close(grid[i, j, k], test_grid[i, j, k], tol))
                    success = false;
        }
    }

    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_zernike_transformer_scales_for_nonunit_radius()
{
    constexpr double radius = 2.0;
    constexpr std::size_t order = 100;

    auto function = [](double lon, double colat, double r)
    {
        std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon) - 0.3,
            r*std::sin(colat)*std::sin(lon) + 0.1,
            r*std::cos(colat) - 0.043
        };
        return std::exp(-(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]));
    };

    auto scaled_function = [&](double lon, double colat, double r) {
        return function(lon, colat, r*radius);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(scaled_function, order);

    zest::zt::ZernikeTransformer<zernike_norm, Convention> transformer{};
    auto expansion = transformer.forward_transform(function, radius, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < grid.extent(0); ++i)
    {
        for (std::size_t j = 0; j < grid.extent(1); ++j)
        {
            for (std::size_t k = 0; k < grid.extent(2); ++k)
                if (!is_close(grid[i, j, k], test_grid[i, j, k], tol))
                    success = false;
        }
    }

    if (!success)
    {
        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                {
                    std::print("{:.16e} ", grid[i, j, k]);
                }
                 std::println("");
            }
             std::println("");
        }

        for (std::size_t i = 0; i < grid.extent(0); ++i)
        {
            for (std::size_t j = 0; j < grid.extent(1); ++j)
            {
                for (std::size_t k = 0; k < grid.extent(2); ++k)
                {
                    std::print("{:.16e} ", test_grid[i, j, k]);
                }
                 std::println("");
            }
             std::println("");
        }
    }

    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_isotropic_glq_forward_transform_expands_Z000()
{
    std::size_t order = 6;

    auto function = []([[maybe_unused]] double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                std::numbers::sqrt3 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm;
    };

    zest::zt::RadialGLQGridPoints points{};
    zest::zt::RadialGLQGrid grid = points.generate_values(function, order);
    zest::zt::IsotropicGLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        if (n == 0)
        {
            if (is_close(expansion[n], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[n], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            std::println("{}: {}", n, expansion[n]);
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_isotropic_glq_forward_transform_expands_Z200()
{
    std::size_t order = 6;

    auto function = []([[maybe_unused]] double r)
    {
        constexpr double sqrt7 = 2.6457513110645905905016158;
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt7 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm*(2.5*r*r - 1.5);
    };

    zest::zt::RadialGLQGridPoints points{};
    zest::zt::RadialGLQGrid grid = points.generate_values(function, order);
    zest::zt::IsotropicGLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        if (n == 2)
        {
            if (is_close(expansion[n], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[n], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            std::println("{}: {}", n, expansion[n]);
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_isotropic_glq_forward_transform_expands_Z400()
{
    std::size_t order = 6;

    auto function = []([[maybe_unused]] double r)
    {
        constexpr double sqrt11 = 3.3166247903553998491149327;
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt11 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm*((7.875*r*r - 8.75)*r*r + 1.875);
    };

    zest::zt::RadialGLQGridPoints points{};
    zest::zt::RadialGLQGrid grid = points.generate_values(function, order);
    zest::zt::IsotropicGLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto n : expansion.indices())
    {
        if (n == 4)
        {
            if (is_close(expansion[n], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[n], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (auto n : expansion.indices())
        {
            std::println("{}: {}", n, expansion[n]);
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_isotropic_glq_backward_transform_evaluates_Z000()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double r)
    {
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                std::numbers::sqrt3 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm;
    };

    zest::zt::RadialGLQGridPoints points{};
    zest::zt::RadialGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::IsotropicGLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.size(); ++i)
    {
        if (!is_close(grid[i], test_grid[i], tol))
            success = false;
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < test_grid.size(); ++i)
        {
            std::print("{} ", test_grid[i]);
        }
        std::println("\ngrid");
        for (std::size_t i = 0; i < grid.size(); ++i)
        {
            std::print("{} ", grid[i]);
        }
        std::println("");
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_isotropic_glq_backward_transform_evaluates_Z200()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double r)
    {
        constexpr double sqrt7 = 2.6457513110645905905016158;
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt7 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm*(2.5*r*r - 1.5);
    };

    zest::zt::RadialGLQGridPoints points{};
    zest::zt::RadialGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::IsotropicGLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.size(); ++i)
    {
        if (!is_close(grid[i], test_grid[i], tol))
            success = false;
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < test_grid.size(); ++i)
        {
            std::print("{} ", test_grid[i]);
        }
        std::println("grid");
        for (std::size_t i = 0; i < grid.size(); ++i)
        {
            std::print("{} ", grid[i]);
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_isotropic_glq_backward_transform_evaluates_Z400()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double r)
    {
        constexpr double sqrt11 = 3.3166247903553998491149327;
        constexpr double znorm
            = (zernike_norm == zest::zt::ZernikeNorm::normed) ?
                sqrt11 : 1.0;
        constexpr double shnorm = (Convention::sh_norm == zest::st::SHNorm::unit) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return znorm*shnorm*((7.875*r*r - 8.75)*r*r + 1.875);
    };

    zest::zt::RadialGLQGridPoints points{};
    zest::zt::RadialGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::IsotropicGLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < grid.size(); ++i)
    {
        if (!is_close(grid[i], test_grid[i], tol))
            success = false;
    }

    if (!success)
    {
        std::println("test_grid");
        for (std::size_t i = 0; i < test_grid.size(); ++i)
        {
            std::print("{} ", test_grid[i]);
        }
        std::println("grid");
        for (std::size_t i = 0; i < grid.size(); ++i)
        {
            std::print("{} ", grid[i]);
        }
    }
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
bool test_isotropic_zernike_transform_converges()
{
    constexpr std::size_t order = 100;

    auto function = [](double r)
    {
        return std::exp(-r*r);
    };


    zest::zt::RadialGLQGridPoints points{};
    zest::zt::RadialGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::IsotropicGLQTransformer<zernike_norm, Convention> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t i = 0; i < grid.size(); ++i)
    {
        if (!is_close(grid[i], test_grid[i], tol))
            success = false;
    }

    return success;
}

template <zest::zt::ZernikeNorm zernike_norm, zest::st::sh_convention Convention>
void test_glq()
{
    assert((test_glq_forward_transform_expands_Z000<zernike_norm, Convention>()));
    assert((test_glq_forward_transform_expands_Z200<zernike_norm, Convention>()));
    assert((test_glq_forward_transform_expands_Z110<zernike_norm, Convention>()));
    assert((test_glq_forward_transform_expands_Z221<zernike_norm, Convention>()));
    assert((test_glq_forward_transform_expands_Z33m2<zernike_norm, Convention>()));
    assert((test_glq_forward_transform_expands_Z531<zernike_norm, Convention>()));

    assert((test_glq_backward_transform_evaluates_Z000<zernike_norm, Convention>()));
    assert((test_glq_backward_transform_evaluates_Z110<zernike_norm, Convention>()));
    assert((test_glq_backward_transform_evaluates_Z200<zernike_norm, Convention>()));
    assert((test_glq_backward_transform_evaluates_Z221<zernike_norm, Convention>()));
    assert((test_glq_backward_transform_evaluates_Z33m2<zernike_norm, Convention>()));
    assert((test_glq_backward_transform_evaluates_Z531<zernike_norm, Convention>()));

    assert((test_zernike_transform_converges<zernike_norm, Convention>()));

    assert((test_zernike_transformer_scales_for_nonunit_radius<zernike_norm, Convention>()));

    assert((test_isotropic_glq_forward_transform_expands_Z000<zernike_norm, Convention>()));
    assert((test_isotropic_glq_forward_transform_expands_Z200<zernike_norm, Convention>()));
    assert((test_isotropic_glq_forward_transform_expands_Z400<zernike_norm, Convention>()));

    assert((test_isotropic_glq_backward_transform_evaluates_Z000<zernike_norm, Convention>()));
    assert((test_isotropic_glq_backward_transform_evaluates_Z200<zernike_norm, Convention>()));
    assert((test_isotropic_glq_backward_transform_evaluates_Z400<zernike_norm, Convention>()));

    assert((test_isotropic_zernike_transform_converges<zernike_norm, Convention>()));
}

} // namespace

int main()
{

    test_glq<zest::zt::ZernikeNorm::normed, zest::st::SHConvention<zest::st::SHNorm::four_pi, zest::st::SHPhase::none>>();
    test_glq<zest::zt::ZernikeNorm::unnormed, zest::st::SHConvention<zest::st::SHNorm::unit, zest::st::SHPhase::cs>>();
}
