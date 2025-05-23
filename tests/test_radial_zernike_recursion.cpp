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
#include "radial_zernike_recursion.hpp"

#include "gauss_legendre.hpp"

#include <cassert>

template <zest::IndexingMode indexing_mode_param>
using ZernikeLayout
    = zest::ZernikeTetrahedralLayout<indexing_mode_param>;

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

bool test_radial_zernike_layout_size_is_correct(std::size_t order)
{
    std::size_t i = 0;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
            ++i;
    }

    bool success = true;
    std::size_t size = zest::zt::RadialZernikeLayout::size(order);
    if (i != size)
    {
        std::printf("%lu %lu", size, i);
        success = success && false;
    }

    return success;
}

bool test_radial_zernike_layout_indices_are_contiguous(std::size_t order)
{
    bool success = true;
    std::size_t i = 0;
    for (std::size_t n = 0; n < order; ++n)
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

template <zest::IndexingMode indexing_mode_param>
bool test_zernike_layout_size_is_correct(std::size_t order)
{
    std::size_t i = 0;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            const int mmin
                = (indexing_mode_param == zest::IndexingMode::nonnegative) ?
                    0 : -int(l);
            for (int m = mmin; m <= int(l); ++m)
                ++i;
        }
    }

    bool success = true;
    std::size_t size = ZernikeLayout<indexing_mode_param>::size(order);
    if (i != size)
    {
        std::printf("%lu %lu", size, i);
        success = success && false;
    }

    return success;
}

template <zest::IndexingMode indexing_mode_param>
bool test_zernike_layout_indices_are_contiguous(std::size_t order)
{
    using index_type = typename ZernikeLayout<indexing_mode_param>::index_type;
    bool success = true;
    std::size_t i = 0;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            const int mmin
                = (indexing_mode_param == zest::IndexingMode::nonnegative) ?
                    0 : -int(l);
            for (int m = mmin; m <= int(l); ++m)
            {
                const std::size_t ind = ZernikeLayout<indexing_mode_param>::idx(index_type(n), index_type(l), index_type(m));
                if (ind != i)
                {
                    std::printf(
                            "(%lu, %lu, %d) ind = %lu i = %lu\n",
                            n, l, m, ind, i);
                    success = success && false;
                }
                ++i;
            }
        }
    }

    return success;
}


template <zest::zt::ZernikeNorm zernike_norm_param>
bool test_radial_zernike_recursion_correct_for_order_1()
{
    constexpr std::size_t order = 1;
    std::vector<double> zernike(zest::zt::RadialZernikeLayout::size(order));
    zest::zt::RadialZernikeRecursion recursion(order);

    const double R00 = 1.0*((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(3.0) : 1.0);

    recursion.zernike<zernike_norm_param>(1.0, zest::zt::RadialZernikeSpan<double, zernike_norm_param>(std::span(zernike), order));
    bool success = is_close(zernike[0], R00, 1.0e-10);

    if (!success)
        std::printf("R00 %f %f\n", zernike[0], R00);
    
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm_param>
bool test_radial_zernike_recursion_generates_correct_up_to_order_7(double r)
{
    constexpr std::size_t order = 7;

    const double R00 = 1.0
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(3.0) : 1.0);

    const double R11 = r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(5.0) : 1.0);

    const double R20 = (2.5*r*r - 1.5)
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(7.0) : 1.0);
    const double R22 = r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(7.0) : 1.0);

    const double R31 = (3.5*r*r - 2.5)*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(9.0) : 1.0);
    const double R33 = r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(9.0) : 1.0);

    const double R40 = ((7.875*r*r - 8.75)*r*r + 1.875)
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(11.0) : 1.0);
    const double R42 = (4.5*r*r - 3.5)*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(11.0) : 1.0);
    const double R44 = r*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(11.0) : 1.0);

    const double R51 = ((12.375*r*r - 15.75)*r*r + 4.375)*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(13.0) : 1.0);
    const double R53 = (5.5*r*r - 4.5)*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(13.0) : 1.0);
    const double R55 = r*r*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(13.0) : 1.0);

    const double R60 = (((26.8125*r*r - 43.3125)*r*r + 19.6875)*r*r - 2.1875)
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(15.0) : 1.0);
    const double R62 = ((17.875*r*r - 24.75)*r*r + 7.875)*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(15.0) : 1.0);
    const double R64 = (6.5*r*r - 5.5)*r*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(15.0) : 1.0);
    const double R66 = r*r*r*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(15.0) : 1.0);

    std::vector<double> zernike(zest::zt::RadialZernikeLayout::size(order));
    zest::zt::RadialZernikeRecursion recursion(order);

    recursion.zernike<zernike_norm_param>(r, zest::zt::RadialZernikeSpan<double, zernike_norm_param>(std::span(zernike), order));
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
    constexpr std::size_t order = 6;
    zest::zt::RadialZernikeRecursion recursion(order);

    const std::size_t glq_order = order + 2;
    std::vector<double> glq_nodes(glq_order);
    std::vector<double> glq_weights(glq_order);
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::cos>(
            glq_nodes, glq_weights, glq_weights.size() & 1);
    
    std::vector<double>
    zernike_grid(glq_order*zest::zt::RadialZernikeLayout::size(order));
    
    for (std::size_t i = 0; i < glq_nodes.size(); ++i)
    {
        const double r = 0.5*(1.0 + glq_nodes[i]);
        zest::zt::RadialZernikeSpan<double, zest::zt::ZernikeNorm::normed> 
        zernike(
                zernike_grid.data() + i*zest::zt::RadialZernikeLayout::size(order), order);
        recursion.zernike<zest::zt::ZernikeNorm::normed>(r, zernike);
    }

    const std::size_t matrix_size = (((order - 1) >> 1) + 1)*(((order - 1) >> 1) + 1);
    std::vector<double> inner_products(order*matrix_size);
    
    for (std::size_t i = 0; i < glq_nodes.size(); ++i)
    {
        const double r = 0.5*(1.0 + glq_nodes[i]);
        const double weight = 0.5*r*r*glq_weights[i];
        zest::zt::RadialZernikeSpan<double, zest::zt::ZernikeNorm::normed> 
        zernike(
                zernike_grid.data() + i*zest::zt::RadialZernikeLayout::size(order), order);
        for (std::size_t l = 0; l < order; ++l)
        {
            const std::size_t extent = ((order - l - 1) >> 1) + 1;

            std::size_t k1 = 0;
            for (std::size_t n1 = l; n1 < order; n1 += 2, ++k1)
            {
                std::size_t k2 = 0;
                for (std::size_t n2 = l; n2 < order; n2 += 2, ++k2)
                {
                    auto& element = inner_products[matrix_size*l + extent*k1 + k2];
                    element += weight*zernike(n1, l)*zernike(n2, l);
                }
            }
        }
    }

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < order; ++l)
    {
        const std::size_t extent = ((order - l - 1) >> 1) + 1;
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
        for (std::size_t l = 0; l < order; ++l)
        {
            const std::size_t extent = ((order - l - 1) >> 1) + 1;
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

template <zest::zt::ZernikeNorm zernike_norm_param>
bool test_radial_zernike_vec_recursion_correct_for_order_1()
{
    constexpr std::size_t order = 1;
    constexpr std::size_t vec_size = 4;
    constexpr std::array<double, vec_size> rad = {0.0, 1.0, 0.5, 0.3};
    std::vector<double> zernike(
            zest::zt::RadialZernikeLayout::size(order)*vec_size);
    zest::zt::RadialZernikeRecursion recursion(order);

    const double R00 = 1.0
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(3.0) : 1.0);

    recursion.zernike<zernike_norm_param>(rad, zest::zt::RadialZernikeVecSpan<double, zernike_norm_param>(zernike, order, vec_size));
    bool success = is_close(zernike[0], R00, 1.0e-10)
            && is_close(zernike[1], R00, 1.0e-10)
            && is_close(zernike[2], R00, 1.0e-10)
            && is_close(zernike[3], R00, 1.0e-10);

    if (!success)
    {
        std::printf("R00(%f) %f\n", rad[0], zernike[0]);
        std::printf("R00(%f) %f\n", rad[1], zernike[1]);
        std::printf("R00(%f) %f\n", rad[2], zernike[2]);
        std::printf("R00(%f) %f\n", rad[3], zernike[3]);
    }
    
    return success;
}

template <zest::zt::ZernikeNorm zernike_norm_param>
bool test_radial_zernike_vec_recursion_generates_correct_up_to_order_7(double r)
{
    constexpr std::size_t vec_size = 1;
    constexpr std::size_t order = 7;
    const std::array<double, 1> x = {r};

    const double R00 = 1.0
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(3.0) : 1.0);

    const double R11 = r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(5.0) : 1.0);

    const double R20 = (2.5*r*r - 1.5)
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(7.0) : 1.0);
    const double R22 = r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(7.0) : 1.0);

    const double R31 = (3.5*r*r - 2.5)*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(9.0) : 1.0);
    const double R33 = r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(9.0) : 1.0);

    const double R40 = ((7.875*r*r - 8.75)*r*r + 1.875)
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(11.0) : 1.0);
    const double R42 = (4.5*r*r - 3.5)*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(11.0) : 1.0);
    const double R44 = r*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(11.0) : 1.0);

    const double R51 = ((12.375*r*r - 15.75)*r*r + 4.375)*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(13.0) : 1.0);
    const double R53 = (5.5*r*r - 4.5)*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(13.0) : 1.0);
    const double R55 = r*r*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(13.0) : 1.0);

    const double R60 = (((26.8125*r*r - 43.3125)*r*r + 19.6875)*r*r - 2.1875)
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(15.0) : 1.0);
    const double R62 = ((17.875*r*r - 24.75)*r*r + 7.875)*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(15.0) : 1.0);
    const double R64 = (6.5*r*r - 5.5)*r*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(15.0) : 1.0);
    const double R66 = r*r*r*r*r*r
        *((zernike_norm_param == zest::zt::ZernikeNorm::normed) ? std::sqrt(15.0) : 1.0);

    std::vector<double> zernike(
            zest::zt::RadialZernikeLayout::size(order)*vec_size);
    zest::zt::RadialZernikeRecursion recursion(order);

    recursion.zernike<zernike_norm_param>(x, zest::zt::RadialZernikeVecSpan<double, zernike_norm_param>(std::span(zernike), order, vec_size));
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

template <zest::zt::ZernikeNorm zernike_norm_param>
bool test_radial_zernike_vec_recursion_end_points_correct_up_to(std::size_t order)
{
    constexpr std::size_t vec_size = 2;
    std::vector<double> reference_buffer(
            zest::zt::RadialZernikeLayout::size(order)*vec_size);
    zest::zt::RadialZernikeVecSpan<double, zernike_norm_param> 
    reference_end_points(reference_buffer, order, vec_size);

    std::vector<double> zero_values(order/2);

    double val = 1.0;
    for (std::size_t n = 0; n < order/2; ++n)
    {
        zero_values[n] = val;
        val *= -(double(n + 1) + 0.5)/double(n + 1);
    }
    
    for (std::size_t n = 0; n < order; ++n)
    {
        if (!(n % 2))
        {
            reference_end_points(n,0)[0]
                = zero_values[n/2]*((n % 2) ? 0.0 : 1.0);
            if constexpr (zernike_norm_param == zest::zt::ZernikeNorm::normed)
                reference_end_points(n,0)[0] *= std::sqrt(double(2*n + 3));
        }
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            if constexpr (zernike_norm_param == zest::zt::ZernikeNorm::normed)
                reference_end_points(n,l)[1] = std::sqrt(2*n + 3);
            else
                reference_end_points(n,l)[1] = 1.0;
        }
    }


    std::vector<double> test_buffer(
            zest::zt::RadialZernikeLayout::size(order)*vec_size);
    zest::zt::RadialZernikeVecSpan<double, zernike_norm_param> test_end_points(
            test_buffer, order, vec_size);

    const std::array<double, 2> x = {0.0, 1.0};
    zest::zt::RadialZernikeRecursion recursion(order);
    recursion.zernike<zernike_norm_param>(x, test_end_points);

    constexpr double tol = 1.0e-13;
    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            success = success && is_close(reference_end_points(n,l)[0], test_end_points(n,l)[0], tol);
            success = success && is_close(reference_end_points(n,l)[1], test_end_points(n,l)[1], tol);
        }
    }

    if (!success)
    {
        for (std::size_t n = 0; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                std::printf("(%lu,%lu): {%f, %f}, {%f, %f}\n", n, l, reference_end_points(n,l)[0], reference_end_points(n,l)[1], test_end_points(n,l)[0], test_end_points(n,l)[1]);
            }
        }
    }

    return success;
}

template <zest::zt::ZernikeNorm zernike_norm_param>
void test_zernike()
{
    assert(test_radial_zernike_recursion_correct_for_order_1<zernike_norm_param>());
    assert(test_radial_zernike_recursion_generates_correct_up_to_order_7<zernike_norm_param>(0.0));
    assert(test_radial_zernike_recursion_generates_correct_up_to_order_7<zernike_norm_param>(1.0));
    assert(test_radial_zernike_recursion_generates_correct_up_to_order_7<zernike_norm_param>(0.3));
    assert(test_radial_zernike_vec_recursion_correct_for_order_1<zernike_norm_param>());
    assert(test_radial_zernike_vec_recursion_generates_correct_up_to_order_7<zernike_norm_param>(0.0));
    assert(test_radial_zernike_vec_recursion_generates_correct_up_to_order_7<zernike_norm_param>(1.0));
    assert(test_radial_zernike_vec_recursion_generates_correct_up_to_order_7<zernike_norm_param>(0.3));
    assert (test_radial_zernike_vec_recursion_end_points_correct_up_to<zernike_norm_param>(30));
}

int main()
{
    assert(test_radial_zernike_layout_size_is_correct(6));
    assert(test_radial_zernike_layout_indices_are_contiguous(6));
    assert(test_zernike_layout_size_is_correct<zest::IndexingMode::nonnegative>(6));
    assert(test_zernike_layout_indices_are_contiguous<zest::IndexingMode::nonnegative>(6));
    assert(test_zernike_layout_size_is_correct<zest::IndexingMode::negative>(6));
    assert(test_zernike_layout_indices_are_contiguous<zest::IndexingMode::negative>(6));

    test_zernike<zest::zt::ZernikeNorm::unnormed>();
    test_zernike<zest::zt::ZernikeNorm::normed>();

    assert(test_radial_zernike_normed_recursion_is_orthonormal());
}