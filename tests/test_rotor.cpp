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
#include "plm_recursion.hpp"
#include "lsq_transformer.hpp"
#include "sh_glq_transformer.hpp"
#include "rotor.hpp"
#include "grid_evaluator.hpp"

#include <random>
#include <cmath>



constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::fabs(a[0] - b[0]) < tol && std::fabs(a[1] - b[1]) < tol;
}

bool to_real_is_inverse_of_to_complex()
{
    constexpr std::size_t order = 6;
    
    using ExpansionSpan = zest::st::RealSHExpansionSpan<std::array<double, 2>, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;

    std::vector<std::array<double, 2>> buffer(ExpansionSpan::Layout::size(order));
    
    ExpansionSpan expansion(buffer, order);
    for (std::size_t l = 0; l < order; ++l)
    {
        for (std::size_t m = 1; m <= l; ++m)
            expansion(l,m) = {
                0.3*double(l) + 0.5*double(m), 0.1*double(l) - 0.2*double(m)
            };
    }

    std::vector<std::array<double, 2>> test_buffer(ExpansionSpan::Layout::size(order));
    std::ranges::copy(buffer, test_buffer.begin());
    ExpansionSpan test_expansion(test_buffer, order);

    zest::st::RealSHExpansionSpan<std::complex<double>, zest::st::SHNorm::QM, zest::st::SHPhase::CS> complex_expansion
            = zest::st::to_complex_expansion<zest::st::SHNorm::QM, zest::st::SHPhase::CS>(expansion);
    zest::st::to_real_expansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>(complex_expansion);

    bool success = true;
    for (std::size_t l = 0; l < order; ++l)
    {
        for (std::size_t m = 0; m <= l; ++m)
            if (!is_close(expansion(l,m), test_expansion(l,m), 1.0e-13))
                success = false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf(
                        "%lu %lu {%f, %f} {%f, %f}\n", l, m,
                        expansion(l,m)[0], expansion(l,m)[1],
                        test_expansion(l,m)[0], test_expansion(l,m)[1]);
        }
    }
    return success;
}

bool test_wigner_d_pi2_is_correct_to_order_5()
{
    constexpr double sqrt5 = 2.2360679774997896964091737;
    constexpr double sqrt7 = 2.6457513110645905905016158;

    constexpr std::size_t order = 5;

    constexpr double d_pi2_0_0_0 = 1.0;

    constexpr double d_pi2_1_0_0 = 0.0;
    constexpr double d_pi2_1_0_1 = 1.0/std::numbers::sqrt2;
    constexpr double d_pi2_1_1_0 = -1.0/std::numbers::sqrt2;
    constexpr double d_pi2_1_1_1 = 1.0/2.0;

    constexpr double d_pi2_2_0_0 = -1.0/2.0;
    constexpr double d_pi2_2_0_1 = 0.0;
    constexpr double d_pi2_2_0_2 = std::numbers::sqrt3/(2.0*std::numbers::sqrt2);
    constexpr double d_pi2_2_1_0 = 0.0;
    constexpr double d_pi2_2_1_1 = -1.0/2.0;
    constexpr double d_pi2_2_1_2 = 1.0/2.0;
    constexpr double d_pi2_2_2_0 = std::numbers::sqrt3/(2.0*std::numbers::sqrt2);
    constexpr double d_pi2_2_2_1 = -1.0/2.0;
    constexpr double d_pi2_2_2_2 = 1.0/4.0;

    constexpr double d_pi2_3_0_0 = 0.0;
    constexpr double d_pi2_3_0_1 = -std::numbers::sqrt3/4.0;
    constexpr double d_pi2_3_0_2 = 0.0;
    constexpr double d_pi2_3_0_3 = sqrt5/4.0;
    constexpr double d_pi2_3_1_0 = std::numbers::sqrt3/4.0;
    constexpr double d_pi2_3_1_1 = -1.0/8.0;
    constexpr double d_pi2_3_1_2 = -sqrt5/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_3_1_3 = std::numbers::sqrt3*sqrt5/8.0;
    constexpr double d_pi2_3_2_0 = 0.0;
    constexpr double d_pi2_3_2_1 = sqrt5/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_3_2_2 = -1.0/2.0;
    constexpr double d_pi2_3_2_3 = std::numbers::sqrt3/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_3_3_0 = -sqrt5/4.0;
    constexpr double d_pi2_3_3_1 = std::numbers::sqrt3*sqrt5/8.0;
    constexpr double d_pi2_3_3_2 = -std::numbers::sqrt3/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_3_3_3 = 1.0/8.0;

    constexpr double d_pi2_4_0_0 = 3.0/8.0;
    constexpr double d_pi2_4_0_1 = 0.0;
    constexpr double d_pi2_4_0_2 = -sqrt5/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_0_3 = 0.0;
    constexpr double d_pi2_4_0_4 = sqrt5*sqrt7/(8.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_1_0 = 0.0;
    constexpr double d_pi2_4_1_1 = 3.0/8.0;
    constexpr double d_pi2_4_1_2 = -1.0/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_1_3 = -sqrt7/8.0;
    constexpr double d_pi2_4_1_4 = sqrt7/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_2_0 = -sqrt5/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_2_1 = 1.0/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_2_2 = 1.0/4.0;
    constexpr double d_pi2_4_2_3 = -sqrt7/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_2_4 = sqrt7/8.0;
    constexpr double d_pi2_4_3_0 = 0.0;
    constexpr double d_pi2_4_3_1 = -sqrt7/8.0;
    constexpr double d_pi2_4_3_2 = sqrt7/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_3_3 = -3.0/8.0;
    constexpr double d_pi2_4_3_4 = 1.0/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_4_0 = sqrt5*sqrt7/(8.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_4_1 = -sqrt7/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_4_2 = sqrt7/8.0;
    constexpr double d_pi2_4_4_3 = -1.0/(4.0*std::numbers::sqrt2);
    constexpr double d_pi2_4_4_4 = 1.0/16.0;


    zest::WignerdPiHalfCollection d_pi2(order);
    
    bool success = is_close(d_pi2(0,0,0), d_pi2_0_0_0, 1.0e-13)
            && is_close(d_pi2(1,0,0), d_pi2_1_0_0, 1.0e-13)
            && is_close(d_pi2(1,0,1), d_pi2_1_0_1, 1.0e-13)
            && is_close(d_pi2(1,1,0), d_pi2_1_1_0, 1.0e-13)
            && is_close(d_pi2(1,1,1), d_pi2_1_1_1, 1.0e-13)
            && is_close(d_pi2(2,0,0), d_pi2_2_0_0, 1.0e-13)
            && is_close(d_pi2(2,0,1), d_pi2_2_0_1, 1.0e-13)
            && is_close(d_pi2(2,0,2), d_pi2_2_0_2, 1.0e-13)
            && is_close(d_pi2(2,1,0), d_pi2_2_1_0, 1.0e-13)
            && is_close(d_pi2(2,1,1), d_pi2_2_1_1, 1.0e-13)
            && is_close(d_pi2(2,1,2), d_pi2_2_1_2, 1.0e-13)
            && is_close(d_pi2(2,2,0), d_pi2_2_2_0, 1.0e-13)
            && is_close(d_pi2(2,2,1), d_pi2_2_2_1, 1.0e-13)
            && is_close(d_pi2(2,2,2), d_pi2_2_2_2, 1.0e-13)
            && is_close(d_pi2(3,0,0), d_pi2_3_0_0, 1.0e-13)
            && is_close(d_pi2(3,0,1), d_pi2_3_0_1, 1.0e-13)
            && is_close(d_pi2(3,0,2), d_pi2_3_0_2, 1.0e-13)
            && is_close(d_pi2(3,0,3), d_pi2_3_0_3, 1.0e-13)
            && is_close(d_pi2(3,1,0), d_pi2_3_1_0, 1.0e-13)
            && is_close(d_pi2(3,1,1), d_pi2_3_1_1, 1.0e-13)
            && is_close(d_pi2(3,1,2), d_pi2_3_1_2, 1.0e-13)
            && is_close(d_pi2(3,1,3), d_pi2_3_1_3, 1.0e-13)
            && is_close(d_pi2(3,2,0), d_pi2_3_2_0, 1.0e-13)
            && is_close(d_pi2(3,2,1), d_pi2_3_2_1, 1.0e-13)
            && is_close(d_pi2(3,2,2), d_pi2_3_2_2, 1.0e-13)
            && is_close(d_pi2(3,2,3), d_pi2_3_2_3, 1.0e-13)
            && is_close(d_pi2(3,3,0), d_pi2_3_3_0, 1.0e-13)
            && is_close(d_pi2(3,3,1), d_pi2_3_3_1, 1.0e-13)
            && is_close(d_pi2(3,3,2), d_pi2_3_3_2, 1.0e-13)
            && is_close(d_pi2(3,3,3), d_pi2_3_3_3, 1.0e-13)
            && is_close(d_pi2(4,0,0), d_pi2_4_0_0, 1.0e-13)
            && is_close(d_pi2(4,0,1), d_pi2_4_0_1, 1.0e-13)
            && is_close(d_pi2(4,0,2), d_pi2_4_0_2, 1.0e-13)
            && is_close(d_pi2(4,0,3), d_pi2_4_0_3, 1.0e-13)
            && is_close(d_pi2(4,0,4), d_pi2_4_0_4, 1.0e-13)
            && is_close(d_pi2(4,1,0), d_pi2_4_1_0, 1.0e-13)
            && is_close(d_pi2(4,1,1), d_pi2_4_1_1, 1.0e-13)
            && is_close(d_pi2(4,1,2), d_pi2_4_1_2, 1.0e-13)
            && is_close(d_pi2(4,1,3), d_pi2_4_1_3, 1.0e-13)
            && is_close(d_pi2(4,1,4), d_pi2_4_1_4, 1.0e-13)
            && is_close(d_pi2(4,2,0), d_pi2_4_2_0, 1.0e-13)
            && is_close(d_pi2(4,2,1), d_pi2_4_2_1, 1.0e-13)
            && is_close(d_pi2(4,2,2), d_pi2_4_2_2, 1.0e-13)
            && is_close(d_pi2(4,2,3), d_pi2_4_2_3, 1.0e-13)
            && is_close(d_pi2(4,2,4), d_pi2_4_2_4, 1.0e-13)
            && is_close(d_pi2(4,3,0), d_pi2_4_3_0, 1.0e-13)
            && is_close(d_pi2(4,3,1), d_pi2_4_3_1, 1.0e-13)
            && is_close(d_pi2(4,3,2), d_pi2_4_3_2, 1.0e-13)
            && is_close(d_pi2(4,3,3), d_pi2_4_3_3, 1.0e-13)
            && is_close(d_pi2(4,3,4), d_pi2_4_3_4, 1.0e-13)
            && is_close(d_pi2(4,4,0), d_pi2_4_4_0, 1.0e-13)
            && is_close(d_pi2(4,4,1), d_pi2_4_4_1, 1.0e-13)
            && is_close(d_pi2(4,4,2), d_pi2_4_4_2, 1.0e-13)
            && is_close(d_pi2(4,4,3), d_pi2_4_4_3, 1.0e-13)
            && is_close(d_pi2(4,4,4), d_pi2_4_4_4, 1.0e-13);
    
    if (success)
        return true;
    else
    {
        std::printf("d_pi2_0_0_0 %f %f\n", d_pi2(0,0,0), d_pi2_0_0_0);
        std::printf("d_pi2_1_0_0 %f %f\n", d_pi2(1,0,0), d_pi2_1_0_0);
        std::printf("d_pi2_1_1_0 %f %f\n", d_pi2(1,1,0), d_pi2_1_1_0);
        std::printf("d_pi2_1_1_1 %f %f\n", d_pi2(1,1,1), d_pi2_1_1_1);
        std::printf("d_pi2_2_0_0 %f %f\n", d_pi2(2,0,0), d_pi2_2_0_0);
        std::printf("d_pi2_2_1_0 %f %f\n", d_pi2(2,1,0), d_pi2_2_1_0);
        std::printf("d_pi2_2_1_1 %f %f\n", d_pi2(2,1,1), d_pi2_2_1_1);
        std::printf("d_pi2_2_2_0 %f %f\n", d_pi2(2,2,0), d_pi2_2_2_0);
        std::printf("d_pi2_2_2_1 %f %f\n", d_pi2(2,2,1), d_pi2_2_2_1);
        std::printf("d_pi2_2_2_2 %f %f\n", d_pi2(2,2,2), d_pi2_2_2_2);
        std::printf("d_pi2_3_0_0 %f %f\n", d_pi2(3,0,0), d_pi2_3_0_0);
        std::printf("d_pi2_3_1_0 %f %f\n", d_pi2(3,1,0), d_pi2_3_1_0);
        std::printf("d_pi2_3_1_1 %f %f\n", d_pi2(3,1,1), d_pi2_3_1_1);
        std::printf("d_pi2_3_2_0 %f %f\n", d_pi2(3,2,0), d_pi2_3_2_0);
        std::printf("d_pi2_3_2_1 %f %f\n", d_pi2(3,2,1), d_pi2_3_2_1);
        std::printf("d_pi2_3_2_2 %f %f\n", d_pi2(3,2,2), d_pi2_3_2_2);
        std::printf("d_pi2_3_3_0 %f %f\n", d_pi2(3,3,0), d_pi2_3_3_0);
        std::printf("d_pi2_3_3_1 %f %f\n", d_pi2(3,3,1), d_pi2_3_3_1);
        std::printf("d_pi2_3_3_2 %f %f\n", d_pi2(3,3,2), d_pi2_3_3_2);
        std::printf("d_pi2_3_3_3 %f %f\n", d_pi2(3,3,3), d_pi2_3_3_3);
        std::printf("d_pi2_4_0_0 %f %f\n", d_pi2(4,0,0), d_pi2_4_0_0);
        std::printf("d_pi2_4_1_0 %f %f\n", d_pi2(4,1,0), d_pi2_4_1_0);
        std::printf("d_pi2_4_1_1 %f %f\n", d_pi2(4,1,1), d_pi2_4_1_1);
        std::printf("d_pi2_4_2_0 %f %f\n", d_pi2(4,2,0), d_pi2_4_2_0);
        std::printf("d_pi2_4_2_1 %f %f\n", d_pi2(4,2,1), d_pi2_4_2_1);
        std::printf("d_pi2_4_2_2 %f %f\n", d_pi2(4,2,2), d_pi2_4_2_2);
        std::printf("d_pi2_4_3_0 %f %f\n", d_pi2(4,3,0), d_pi2_4_3_0);
        std::printf("d_pi2_4_3_1 %f %f\n", d_pi2(4,3,1), d_pi2_4_3_1);
        std::printf("d_pi2_4_3_2 %f %f\n", d_pi2(4,3,2), d_pi2_4_3_2);
        std::printf("d_pi2_4_3_3 %f %f\n", d_pi2(4,3,3), d_pi2_4_3_3);
        std::printf("d_pi2_4_4_0 %f %f\n", d_pi2(4,4,0), d_pi2_4_4_0);
        std::printf("d_pi2_4_4_1 %f %f\n", d_pi2(4,4,1), d_pi2_4_4_1);
        std::printf("d_pi2_4_4_2 %f %f\n", d_pi2(4,4,2), d_pi2_4_4_2);
        std::printf("d_pi2_4_4_3 %f %f\n", d_pi2(4,4,3), d_pi2_4_4_3);
        std::printf("d_pi2_4_4_4 %f %f\n", d_pi2(4,4,4), d_pi2_4_4_4);
        return false;
    }
}

template <typename ExpansionSpanType>
bool test_rotation_completes()
{
    constexpr std::size_t order = 6;

    std::vector<std::array<double, 2>> buffer(ExpansionSpanType::Layout::size(order));
    
    zest::WignerdPiHalfCollection wigner_d_pi2(order);
    zest::Rotor rotor(order);
    ExpansionSpanType expansion(buffer, order);
    rotor.rotate(
            expansion, wigner_d_pi2, std::array<double, 3>{}, 
            zest::RotationType::OBJECT);

    return true;
}

bool test_sh_trivial_rotation_is_trivial_order_6()
{
    constexpr std::size_t order = 6;
    
    using ExpansionSpan = zest::st::RealSHExpansionSpan<std::array<double, 2>, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;

    std::vector<std::array<double, 2>> buffer(ExpansionSpan::Layout::size(order));
    
    ExpansionSpan expansion(buffer, order);

    constexpr double norm = 1.0*std::numbers::inv_sqrtpi/std::numbers::sqrt2;
    constexpr double norm2 = 0.5*std::numbers::inv_sqrtpi;
    for (std::size_t l = 0; l < order; ++l)
    {
        expansion(l,0) = {norm2*double(l)/3.0, 0.0};
        for (std::size_t m = 1; m <= l; ++m)
            expansion(l,m) = {
                norm*(double(l)/3.0 + double(m)/2.0), norm*(double(l)/10.0 - double(m)/5.0)
            };
    }

    std::vector<std::array<double, 2>> test_buffer(ExpansionSpan::Layout::size(order));
    std::ranges::copy(buffer, test_buffer.begin());
    ExpansionSpan test_expansion(test_buffer, order);

    zest::WignerdPiHalfCollection wigner_d_pi2(order);
    zest::Rotor rotor(order);
    rotor.rotate(
            expansion, wigner_d_pi2, std::array<double, 3>{}, 
            zest::RotationType::OBJECT);

    bool success = true;
    for (std::size_t l = 0; l < order; ++l)
    {
        for (std::size_t m = 0; m <= l; ++m)
            if (!is_close(expansion(l,m), test_expansion(l,m), 1.0e-13))
                success = false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf(
                        "%lu %lu {%f, %f} {%f, %f}\n", l, m,
                        expansion(l,m)[0]/norm, expansion(l,m)[1]/norm,
                        test_expansion(l,m)[0]/norm, test_expansion(l,m)[1]/norm);
        }
    }
    return success;
}

bool test_zernike_trivial_rotation_is_trivial_order_6()
{
    constexpr std::size_t order = 6;
    
    using ExpansionSpan = zest::zt::ZernikeExpansionSpan<std::array<double, 2>, zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;

    std::vector<std::array<double, 2>> buffer(ExpansionSpan::Layout::size(order));
    
    ExpansionSpan expansion(buffer, order);

    constexpr double norm = 1.0*std::numbers::inv_sqrtpi/std::numbers::sqrt2;
    constexpr double norm2 = 0.5*std::numbers::inv_sqrtpi;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            expansion(n,l,0) = {norm2*double(l)/3.0, 0.0};
            for (std::size_t m = 1; m <= l; ++m)
                expansion(n,l,m) = {
                    norm*(double(n)/7.0 + double(l)/3.0 + double(m)/2.0), norm*(double(n)/9.0 + double(l)/10.0 - double(m)/5.0)
                };
        }
    }

    std::vector<std::array<double, 2>> test_buffer(ExpansionSpan::Layout::size(order));
    std::ranges::copy(buffer, test_buffer.begin());
    ExpansionSpan test_expansion(test_buffer, order);

    zest::WignerdPiHalfCollection wigner_d_pi2(order);
    zest::Rotor rotor(order);
    rotor.rotate(
            expansion, wigner_d_pi2, std::array<double, 3>{}, 
            zest::RotationType::OBJECT);

    bool success = true;

    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                if (!is_close(expansion(n,l,m), test_expansion(n,l,m), 1.0e-13))
                    success = false;
        }
    }

    if (!success)
    {

        for (std::size_t n = 0; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                    std::printf(
                            "%lu %lu %lu {%f, %f} {%f, %f}\n", n, l, m,
                            expansion(n,l,m)[0]/norm, expansion(n,l,m)[1]/norm,
                            test_expansion(n,l,m)[0]/norm, test_expansion(n,l,m)[1]/norm);
            }
        }
    }
    return success;
}

bool test_sh_trivial_polar_rotation_is_trivial_order_6()
{
    constexpr std::size_t order = 6;
    
    using ExpansionSpan = zest::st::RealSHExpansionSpan<std::array<double, 2>, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;

    std::vector<std::array<double, 2>> buffer(ExpansionSpan::Layout::size(order));
    
    ExpansionSpan expansion(buffer, order);

    constexpr double norm = 1.0*std::numbers::inv_sqrtpi/std::numbers::sqrt2;
    constexpr double norm2 = 0.5*std::numbers::inv_sqrtpi;
    for (std::size_t l = 0; l < order; ++l)
    {
        expansion(l,0) = {norm2*double(l)/3.0, 0.0};
        for (std::size_t m = 1; m <= l; ++m)
            expansion(l,m) = {
                norm*(double(l)/3.0 + double(m)/2.0), norm*(double(l)/10.0 - double(m)/5.0)
            };
    }

    std::vector<std::array<double, 2>> test_buffer(ExpansionSpan::Layout::size(order));
    std::ranges::copy(buffer, test_buffer.begin());
    ExpansionSpan test_expansion(test_buffer, order);

    zest::WignerdPiHalfCollection wigner_d_pi2(order);
    zest::Rotor rotor(order);
    rotor.polar_rotate(expansion, 0.0, zest::RotationType::OBJECT);

    bool success = true;
    for (std::size_t l = 0; l < order; ++l)
    {
        for (std::size_t m = 0; m <= l; ++m)
            if (!is_close(expansion(l,m), test_expansion(l,m), 1.0e-13))
                success = false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf(
                        "%lu %lu {%f, %f} {%f, %f}\n", l, m,
                        expansion(l,m)[0]/norm, expansion(l,m)[1]/norm,
                        test_expansion(l,m)[0]/norm, test_expansion(l,m)[1]/norm);
        }
    }
    return success;
}

bool test_zernike_trivial_polar_rotation_is_trivial_order_6()
{
    constexpr std::size_t order = 6;
    
    using ExpansionSpan = zest::zt::ZernikeExpansionSpan<std::array<double, 2>, zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;

    std::vector<std::array<double, 2>> buffer(ExpansionSpan::Layout::size(order));
    
    ExpansionSpan expansion(buffer, order);

    constexpr double norm = 1.0*std::numbers::inv_sqrtpi/std::numbers::sqrt2;
    constexpr double norm2 = 0.5*std::numbers::inv_sqrtpi;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            expansion(n,l,0) = {norm2*double(l)/3.0, 0.0};
            for (std::size_t m = 1; m <= l; ++m)
                expansion(n,l,m) = {
                    norm*(double(n)/7.0 + double(l)/3.0 + double(m)/2.0), norm*(double(n)/9.0 + double(l)/10.0 - double(m)/5.0)
                };
        }
    }

    std::vector<std::array<double, 2>> test_buffer(ExpansionSpan::Layout::size(order));
    std::ranges::copy(buffer, test_buffer.begin());
    ExpansionSpan test_expansion(test_buffer, order);

    zest::WignerdPiHalfCollection wigner_d_pi2(order);
    zest::Rotor rotor(order);
    rotor.polar_rotate(expansion, 0.0, zest::RotationType::OBJECT);

    bool success = true;

    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                if (!is_close(expansion(n,l,m), test_expansion(n,l,m), 1.0e-13))
                    success = false;
        }
    }

    if (!success)
    {

        for (std::size_t n = 0; n < order; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                for (std::size_t m = 0; m <= l; ++m)
                    std::printf(
                            "%lu %lu %lu {%f, %f} {%f, %f}\n", n, l, m,
                            expansion(n,l,m)[0]/norm, expansion(n,l,m)[1]/norm,
                            test_expansion(n,l,m)[0]/norm, test_expansion(n,l,m)[1]/norm);
            }
        }
    }
    return success;
}

int main()
{
    assert(to_real_is_inverse_of_to_complex());

    assert(test_wigner_d_pi2_is_correct_to_order_5());

    using SHExpansionSpan = zest::st::RealSHExpansionSpan<std::array<double, 2>, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;
    assert(test_rotation_completes<SHExpansionSpan>());

    using ZernikeExpansionSpan = zest::zt::ZernikeExpansionSpan<std::array<double, 2>, zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;
    assert(test_rotation_completes<ZernikeExpansionSpan>());

    using ZernikeExpansionSHSpan = zest::zt::ZernikeExpansionSHSpan<std::array<double, 2>, zest::zt::ZernikeNorm::NORMED, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;
    assert(test_rotation_completes<ZernikeExpansionSHSpan>());

    assert(test_sh_trivial_rotation_is_trivial_order_6());
    assert(test_zernike_trivial_rotation_is_trivial_order_6());

    assert(test_sh_trivial_polar_rotation_is_trivial_order_6());
    assert(test_zernike_trivial_polar_rotation_is_trivial_order_6());
}