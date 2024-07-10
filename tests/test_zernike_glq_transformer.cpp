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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Z000()
{
    std::size_t order = 6;

    auto function = [](
        [[maybe_unused]] double r, [[maybe_unused]] double lon, 
        [[maybe_unused]] double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(3.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        return znorm*shnorm;
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 0 && l == 0 && m == 0)
                {
                    if (is_close(expansion(n,l,m)[0], reference_coeff, tol)
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
        for (std::size_t n = 0; n <= order; ++n)
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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Z200()
{
    std::size_t order = 6;

    auto function = [](
        double r, [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(7.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        return znorm*shnorm*(2.5*r*r - 1.5);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 0 && m == 0)
                {
                    if (is_close(expansion(n,l,m)[0], reference_coeff, tol)
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
        for (std::size_t n = 0; n <= order; ++n)
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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Z110()
{
    std::size_t order = 6;

    auto function = [](double r, [[maybe_unused]] double lon, double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(5.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        const double z = std::cos(colat);
        return znorm*shnorm*r*std::sqrt(3.0)*z;
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 1 && l == 1 && m == 0)
                {
                    if (is_close(expansion(n,l,m)[0], reference_coeff, tol)
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
        for (std::size_t n = 0; n < order; ++n)
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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Z221()
{
    std::size_t order = 6;

    auto function = [](
        double r, double lon, double colat)
    {
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? 1.0 : -1.0;
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(7.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        const double z = std::cos(colat);
        return phase*znorm*shnorm*r*r*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 2 && m == 1)
                {
                    if (is_close(expansion(n,l,m)[0], reference_coeff, tol)
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
        for (std::size_t n = 0; n <= order; ++n)
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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Z33m2()
{
    std::size_t order = 6;

    auto function = [](
        double r, double lon, double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(9.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        const double z = std::cos(colat);
        return znorm*shnorm*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 3 && l == 3 && m == 2)
                {
                    if (is_close(expansion(n,l,m)[0], 0.0, tol)
                            && is_close(expansion(n,l,m)[1], reference_coeff, tol))
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
        for (std::size_t n = 0; n <= order; ++n)
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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_forward_transform_expands_Z531()
{
    std::size_t order = 6;

    auto function = [](
        double r, double lon, double colat)
    {
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? 1.0 : -1.0;
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(13.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        const double z = std::cos(colat);
        return phase*znorm*shnorm*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid grid = points.generate_values(function, order);
    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);
    zest::zt::ZernikeExpansion expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 5 && l == 3 && m == 1)
                {
                    if (is_close(expansion(n,l,m)[0], reference_coeff, tol)
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
        for (std::size_t n = 0; n <= order; ++n)
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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Z000()
{
    constexpr std::size_t order = 6; 

    auto function = [](
        [[maybe_unused]] double r, [[maybe_unused]] double lon, 
        [[maybe_unused]] double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(3.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        return znorm*shnorm;
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Z110()
{
    constexpr std::size_t order = 6; 

    auto function = [](double r, [[maybe_unused]] double lon, double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(5.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        const double z = std::cos(colat);
        return znorm*shnorm*r*std::sqrt(3.0)*z;
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Z200()
{
    constexpr std::size_t order = 6; 

    auto function = [](
        double r, [[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(7.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        return znorm*shnorm*(2.5*r*r - 1.5);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Z221()
{
    constexpr std::size_t order = 6; 

    auto function = [](
        double r, double lon, double colat)
    {
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? 1.0 : -1.0;
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(7.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        const double z = std::cos(colat);
        return phase*znorm*shnorm*r*r*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Z33m2()
{
    constexpr std::size_t order = 6; 

    auto function = [](
        double r, double lon, double colat)
    {
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(9.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        const double z = std::cos(colat);
        return znorm*shnorm*r*r*r*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
bool test_glq_backward_transform_evaluates_Z531()
{
    constexpr std::size_t order = 6; 

    auto function = [](
        double r, double lon, double colat)
    {
        constexpr double phase = (PHASE == zest::st::SHPhase::NONE) ? 1.0 : -1.0;
        constexpr double znorm
            = (ZERNIKE_NORM == zest::zt::ZernikeNorm::NORMED) ?
                std::sqrt(13.0) : 1.0;
        constexpr double shnorm = (SH_NORM == zest::st::SHNorm::QM) ?
            1.0/std::sqrt(4.0*std::numbers::pi) : 1.0;
        const double z = std::cos(colat);
        return phase*znorm*shnorm*(5.5*r*r - 4.5)*r*r*r*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    zest::zt::BallGLQGridPoints points{};
    zest::zt::BallGLQGrid test_grid = points.generate_values(function, order);

    zest::zt::GLQTransformer<ZERNIKE_NORM, SH_NORM, PHASE> transformer(order);

    auto expansion = transformer.forward_transform(test_grid, order);
    auto grid = transformer.backward_transform(expansion, order);

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

template <zest::zt::ZernikeNorm ZERNIKE_NORM, zest::st::SHNorm SH_NORM, zest::st::SHPhase PHASE>
void test_glq()
{
    assert((test_glq_forward_transform_expands_Z000<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Z200<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Z110<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Z221<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Z33m2<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_forward_transform_expands_Z531<ZERNIKE_NORM, SH_NORM, PHASE>()));

    assert((test_glq_backward_transform_evaluates_Z000<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Z110<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Z200<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Z221<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Z33m2<ZERNIKE_NORM, SH_NORM, PHASE>()));
    assert((test_glq_backward_transform_evaluates_Z531<ZERNIKE_NORM, SH_NORM, PHASE>()));
}

int main()
{
    test_glq<zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>();
    test_glq<zest::zt::ZernikeNorm::UNNORMED, zest::st::SHNorm::QM, zest::st::SHPhase::CS>();
}