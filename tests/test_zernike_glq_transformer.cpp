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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, lmax);
    zest::zt::GLQTransformerGeo transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 0 && l == 0 && m == 0)
                {
                    if (is_close(expansion(n,l,m)[0], 1.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion(n,l,m)[0], 0.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n <= lmax; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::printf(
                            "(%lu,%lu,%lu) %f %f\n", n, l, m,
                            expansion(n,l,m)[0], expansion(n,l,m)[1]);
                }
            }
        }
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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, lmax);
    zest::zt::GLQTransformerGeo transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 0 && m == 0)
                {
                    if (is_close(expansion(n,l,m)[0], 1.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion(n,l,m)[0], 0.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n <= lmax; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::printf(
                            "(%lu,%lu,%lu) %f %f\n", n, l, m,
                            expansion(n,l,m)[0], expansion(n,l,m)[1]);
                }
            }
        }
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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, lmax);
    zest::zt::GLQTransformerGeo transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 1 && l == 1 && m == 0)
                {
                    if (is_close(expansion(n,l,m)[0], 1.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion(n,l,m)[0], 0.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n <= lmax; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::printf(
                            "(%lu,%lu,%lu) %f %f\n", n, l, m,
                            expansion(n,l,m)[0], expansion(n,l,m)[1]);
                }
            }
        }
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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, lmax);
    zest::zt::GLQTransformerGeo transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 2 && m == 1)
                {
                    if (is_close(expansion(n,l,m)[0], 1.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion(n,l,m)[0], 0.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n <= lmax; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::printf(
                            "(%lu,%lu,%lu) %f %f\n", n, l, m,
                            expansion(n,l,m)[0], expansion(n,l,m)[1]);
                }
            }
        }
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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, lmax);
    zest::zt::GLQTransformerGeo transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 3 && l == 3 && m == 2)
                {
                    if (is_close(expansion(n,l,m)[0], 0.0, tol)
                            && is_close(expansion(n,l,m)[1], 1.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion(n,l,m)[0], 0.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n <= lmax; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::printf(
                            "(%lu,%lu,%lu) %f %f\n", n, l, m,
                            expansion(n,l,m)[0], expansion(n,l,m)[1]);
                }
            }
        }
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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, lmax);
    zest::zt::GLQTransformerGeo transformer(lmax);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, lmax);

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 5 && l == 3 && m == 1)
                {
                    if (is_close(expansion(n,l,m)[0], 1.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion(n,l,m)[0], 0.0, tol)
                            && is_close(expansion(n,l,m)[1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n <= lmax; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::printf(
                            "(%lu,%lu,%lu) %f %f\n", n, l, m,
                            expansion(n,l,m)[0], expansion(n,l,m)[1]);
                }
            }
        }
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
    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, lmax);

    zest::zt::GLQTransformerGeo transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

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
    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, lmax);

    zest::zt::GLQTransformerGeo transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, lmax);

    zest::zt::GLQTransformerGeo transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, lmax);

    zest::zt::GLQTransformerGeo transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, lmax);

    zest::zt::GLQTransformerGeo transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

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

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, lmax);

    zest::zt::GLQTransformerGeo transformer(lmax);

    auto expansion = transformer.forward_transform(test_grid, lmax);
    auto grid = transformer.backward_transform(expansion, lmax);

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

int main()
{
    assert(test_glq_geo_expansion_expands_Z000());
    assert(test_glq_geo_expansion_expands_Z200());
    assert(test_glq_geo_expansion_expands_Z110());
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