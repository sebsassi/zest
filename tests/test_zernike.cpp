#include "../zernike.hpp"

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

bool test_radial_zernike_layout_size_is_correct(std::size_t lmax)
{
    std::size_t i = 0;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
            ++i;
    }

    bool success = true;
    std::size_t size = zest::zt::RadialZernikeLayout::size(lmax);
    if (i != size)
    {
        std::printf("%lu %lu", size, i);
        success = success && false;
    }

    return success;
}

bool test_radial_zernike_layout_indices_are_contiguous(std::size_t lmax)
{
    bool success = true;
    std::size_t i = 0;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            const std::size_t ind = zest::zt::RadialZernikeLayout::idx(n, l);
            if (ind != i)
            {
                std::printf("(%lu, %lu) ind = %lu i = %lu\n", n, l, ind, i);
                success = success && false;
            }
            ++i;
        }
    }

    return success;
}

bool test_zernike_layout_size_is_correct(std::size_t lmax)
{
    std::size_t i = 0;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                ++i;
        }
    }

    bool success = true;
    std::size_t size = zest::zt::ZernikeLayout::size(lmax);
    if (i != size)
    {
        std::printf("%lu %lu", size, i);
        success = success && false;
    }

    return success;
}

bool test_zernike_layout_indices_are_contiguous(std::size_t lmax)
{
    bool success = true;
    std::size_t i = 0;
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                const std::size_t ind = zest::zt::ZernikeLayout::idx(n, l, m);
                if (ind != i)
                {
                    std::printf(
                            "(%lu, %lu, %lu) ind = %lu i = %lu\n",
                            n, l, m, ind,i);
                    success = success && false;
                }
                ++i;
            }
        }
    }

    return success;
}

bool test_radial_zernike_unnormed_recursion_correct_for_lmax_0()
{
    std::vector<double> zernike(zest::zt::RadialZernikeLayout::size(0));
    zest::zt::RadialZernikeRecursion recursion(0);

    recursion.zernike<zest::zt::ZernikeNorm::UNNORMED>(zest::zt::RadialZernikeSpan(std::span(zernike), 0), 1.0);
    bool success = is_close(zernike[0], 1.0, 1.0e-10);

    if (!success)
        std::printf("R00 %f %f\n", zernike[0], 1.0);
    
    return success;
}

bool test_radial_zernike_unnormed_recursion_generates_correct_up_to_lmax_6(double r)
{
    constexpr std::size_t lmax = 6;

    const double R00 = 1.0;

    const double R11 = r;

    const double R20 = 2.5*r*r - 1.5;
    const double R22 = r*r;

    const double R31 = (3.5*r*r - 2.5)*r;
    const double R33 = r*r*r;

    const double R40 = (7.875*r*r - 8.75)*r*r + 1.875;
    const double R42 = (4.5*r*r - 3.5)*r*r;
    const double R44 = r*r*r*r;

    const double R51 = ((12.375*r*r - 15.75)*r*r + 4.375)*r;
    const double R53 = (5.5*r*r - 4.5)*r*r*r;
    const double R55 = r*r*r*r*r;

    const double R60 = ((26.8125*r*r - 43.3125)*r*r + 19.6875)*r*r - 2.1875;
    const double R62 = ((17.875*r*r - 24.75)*r*r + 7.875)*r*r;
    const double R64 = (6.5*r*r - 5.5)*r*r*r*r;
    const double R66 = r*r*r*r*r*r;

    std::vector<double> zernike(zest::zt::RadialZernikeLayout::size(lmax));
    zest::zt::RadialZernikeRecursion recursion(lmax);

    recursion.zernike<zest::zt::ZernikeNorm::UNNORMED>(zest::zt::RadialZernikeSpan(std::span(zernike), lmax), r);
    bool success = is_close(zernike[0], R00, 1.0e-10)
            && is_close(zernike[1], R11, 1.0e-10)
            && is_close(zernike[2], R20, 1.0e-10)
            && is_close(zernike[3], R22, 1.0e-10)
            && is_close(zernike[4], R31, 1.0e-10)
            && is_close(zernike[5], R33, 1.0e-10)
            && is_close(zernike[6], R40, 1.0e-10)
            && is_close(zernike[7], R42, 1.0e-10)
            && is_close(zernike[8], R44, 1.0e-10)
            && is_close(zernike[9], R51, 1.0e-10)
            && is_close(zernike[10], R53, 1.0e-10)
            && is_close(zernike[11], R55, 1.0e-10)
            && is_close(zernike[12], R60, 1.0e-10)
            && is_close(zernike[13], R62, 1.0e-10)
            && is_close(zernike[14], R64, 1.0e-10)
            && is_close(zernike[15], R66, 1.0e-10);
    
    if (success)
        return true;
    else
    {
        std::printf("R00 %f %f\n", zernike[0], R00);
        std::printf("R11 %f %f\n", zernike[1], R11);
        std::printf("R20 %f %f\n", zernike[2], R20);
        std::printf("R22 %f %f\n", zernike[3], R22);
        std::printf("R31 %f %f\n", zernike[4], R31);
        std::printf("R33 %f %f\n", zernike[5], R33);
        std::printf("R40 %f %f\n", zernike[6], R40);
        std::printf("R42 %f %f\n", zernike[7], R42);
        std::printf("R44 %f %f\n", zernike[8], R44);
        std::printf("R51 %f %f\n", zernike[9], R51);
        std::printf("R53 %f %f\n", zernike[10], R53);
        std::printf("R55 %f %f\n", zernike[11], R55);
        std::printf("R60 %f %f\n", zernike[12], R60);
        std::printf("R62 %f %f\n", zernike[13], R62);
        std::printf("R64 %f %f\n", zernike[14], R64);
        std::printf("R66 %f %f\n", zernike[15], R66);
        return false;
    }
}

bool test_radial_zernike_normed_recursion_generates_correct_up_to_lmax_6(double r)
{
    constexpr std::size_t lmax = 6;

    const double R00 = std::sqrt(3.0)*1.0;

    const double R11 = std::sqrt(5.0)*r;

    const double R20 = std::sqrt(7.0)*(2.5*r*r - 1.5);
    const double R22 = std::sqrt(7.0)*r*r;

    const double R31 = std::sqrt(9.0)*(3.5*r*r - 2.5)*r;
    const double R33 = std::sqrt(9.0)*r*r*r;

    const double R40 = std::sqrt(11.0)*((7.875*r*r - 8.75)*r*r + 1.875);
    const double R42 = std::sqrt(11.0)*(4.5*r*r - 3.5)*r*r;
    const double R44 = std::sqrt(11.0)*r*r*r*r;

    const double R51 = std::sqrt(13.0)*((12.375*r*r - 15.75)*r*r + 4.375)*r;
    const double R53 = std::sqrt(13.0)*(5.5*r*r - 4.5)*r*r*r;
    const double R55 = std::sqrt(13.0)*r*r*r*r*r;

    const double R60 = std::sqrt(15.0)*(((26.8125*r*r - 43.3125)*r*r + 19.6875)*r*r - 2.1875);
    const double R62 = std::sqrt(15.0)*((17.875*r*r - 24.75)*r*r + 7.875)*r*r;
    const double R64 = std::sqrt(15.0)*(6.5*r*r - 5.5)*r*r*r*r;
    const double R66 = std::sqrt(15.0)*r*r*r*r*r*r;

    std::vector<double> zernike(zest::zt::RadialZernikeLayout::size(lmax));
    zest::zt::RadialZernikeRecursion recursion(lmax);

    recursion.zernike<zest::zt::ZernikeNorm::NORMED>(zest::zt::RadialZernikeSpan(std::span(zernike), lmax), r);
    bool success = is_close(zernike[0], R00, 1.0e-10)
            && is_close(zernike[1], R11, 1.0e-10)
            && is_close(zernike[2], R20, 1.0e-10)
            && is_close(zernike[3], R22, 1.0e-10)
            && is_close(zernike[4], R31, 1.0e-10)
            && is_close(zernike[5], R33, 1.0e-10)
            && is_close(zernike[6], R40, 1.0e-10)
            && is_close(zernike[7], R42, 1.0e-10)
            && is_close(zernike[8], R44, 1.0e-10)
            && is_close(zernike[9], R51, 1.0e-10)
            && is_close(zernike[10], R53, 1.0e-10)
            && is_close(zernike[11], R55, 1.0e-10)
            && is_close(zernike[12], R60, 1.0e-10)
            && is_close(zernike[13], R62, 1.0e-10)
            && is_close(zernike[14], R64, 1.0e-10)
            && is_close(zernike[15], R66, 1.0e-10);
    
    if (success)
        return true;
    else
    {
        std::printf("R00 %f %f\n", zernike[0], R00);
        std::printf("R11 %f %f\n", zernike[1], R11);
        std::printf("R20 %f %f\n", zernike[2], R20);
        std::printf("R22 %f %f\n", zernike[3], R22);
        std::printf("R31 %f %f\n", zernike[4], R31);
        std::printf("R33 %f %f\n", zernike[5], R33);
        std::printf("R40 %f %f\n", zernike[6], R40);
        std::printf("R42 %f %f\n", zernike[7], R42);
        std::printf("R44 %f %f\n", zernike[8], R44);
        std::printf("R51 %f %f\n", zernike[9], R51);
        std::printf("R53 %f %f\n", zernike[10], R53);
        std::printf("R55 %f %f\n", zernike[11], R55);
        std::printf("R60 %f %f\n", zernike[12], R60);
        std::printf("R62 %f %f\n", zernike[13], R62);
        std::printf("R64 %f %f\n", zernike[14], R64);
        std::printf("R66 %f %f\n", zernike[15], R66);
        return false;
    }
}

bool test_radial_zernike_normed_recursion_is_orthonormal()
{
    constexpr std::size_t lmax = 5;
    zest::zt::RadialZernikeRecursion recursion(lmax);

    const std::size_t glq_order = lmax + 2;
    std::vector<double> glq_nodes(glq_order);
    std::vector<double> glq_weights(glq_order);
    zest::gl::gl_nodes_and_weights<double, zest::gl::GLLayout::UNPACKED, zest::gl::GLNodeStyle::COS>(
            glq_nodes, glq_weights, glq_weights.size());
    
    std::vector<double>
    zernike_grid(glq_order*zest::zt::RadialZernikeLayout::size(lmax));
    
    for (std::size_t i = 0; i < glq_nodes.size(); ++i)
    {
        const double r = 0.5*(1.0 + glq_nodes[i]);
        zest::zt::RadialZernikeSpan<double> zernike(zernike_grid, i, lmax);
        recursion.zernike<zest::zt::ZernikeNorm::NORMED>(zernike, r);
    }

    const std::size_t matrix_size = ((lmax >> 1) + 1)*((lmax >> 1) + 1);
    std::vector<double> inner_products((lmax + 1)*matrix_size);
    
    for (std::size_t i = 0; i < glq_nodes.size(); ++i)
    {
        const double r = 0.5*(1.0 + glq_nodes[i]);
        const double weight = 0.5*r*r*glq_weights[i];
        zest::zt::RadialZernikeSpan<double> zernike(zernike_grid, i, lmax);
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            const std::size_t extent = ((lmax - l) >> 1) + 1;

            std::size_t k1 = 0;
            for (std::size_t n1 = l; n1 <= lmax; n1 += 2, ++k1)
            {
                std::size_t k2 = 0;
                for (std::size_t n2 = l; n2 <= lmax; n2 += 2, ++k2)
                {
                    auto& element = inner_products[matrix_size*l + extent*k1 + k2];
                    element += weight*zernike(n1, l)*zernike(n2, l);
                }
            }
        }
    }

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < lmax; ++l)
    {
        const std::size_t extent = ((lmax - l) >> 1) + 1;
        for (std::size_t i = 0; i < extent; ++i)
        {
            for (std::size_t j = 0; j < extent; ++j)
            {
                const auto& element = inner_products[matrix_size*l + extent*i + j];
                if (i == j && !is_close(element, 1.0, tol))
                    success = success && false;
                if (i != j && !is_close(element, 0.0, tol))
                    success = success && false;
            }
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < lmax; ++l)
        {
            const std::size_t extent = ((lmax - l) >> 1) + 1;
            for (std::size_t i = 0; i < extent; ++i)
            {
                for (std::size_t j = 0; j < extent; ++j)
                {
                    const auto& element = inner_products[matrix_size*l + extent*i + j];
                    std::printf("%f ", element);
                }
                std::printf("\n");
            }
            std::printf("\n");
        }
    }

    return success;
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
    assert(test_radial_zernike_layout_size_is_correct(20));
    assert(test_radial_zernike_layout_indices_are_contiguous(20));
    assert(test_zernike_layout_size_is_correct(20));
    assert(test_zernike_layout_indices_are_contiguous(20));

    assert(test_radial_zernike_unnormed_recursion_correct_for_lmax_0());

    assert(test_radial_zernike_unnormed_recursion_generates_correct_up_to_lmax_6(0.0));
    assert(test_radial_zernike_unnormed_recursion_generates_correct_up_to_lmax_6(0.5));
    assert(test_radial_zernike_unnormed_recursion_generates_correct_up_to_lmax_6(1.0));
    assert(test_radial_zernike_unnormed_recursion_generates_correct_up_to_lmax_6(0.84567497698));

    assert(test_radial_zernike_normed_recursion_generates_correct_up_to_lmax_6(0.0));
    assert(test_radial_zernike_normed_recursion_generates_correct_up_to_lmax_6(0.5));
    assert(test_radial_zernike_normed_recursion_generates_correct_up_to_lmax_6(1.0));
    assert(test_radial_zernike_normed_recursion_generates_correct_up_to_lmax_6(0.84567497698));

    assert(test_radial_zernike_normed_recursion_is_orthonormal());

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

    assert(test_uniform_grid_evaluator_does_constant_function());
    assert(test_uniform_grid_evaluator_does_Z33m2_plus_Z531());
}