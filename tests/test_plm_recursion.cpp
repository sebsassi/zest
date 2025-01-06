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

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_plm_real_generates_real_correct_up_to_order_5(double z)
{
    constexpr std::size_t order = 5;
    constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
    constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
        0.5*std::numbers::inv_sqrtpi : 1.0;

    const double P00 = shnorm;

    const double P10 = shnorm*std::sqrt(3.0)*z;
    const double P11 = phase*shnorm*std::sqrt(3.0)*std::sqrt(1.0 - z*z);

    const double P20 = shnorm*std::sqrt(5.0/4.0)*(3.0*z*z - 1.0);
    const double P21 = phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z;
    const double P22 = shnorm*std::sqrt(15.0/4.0)*(1.0 - z*z);

    const double P30 = shnorm*std::sqrt(7.0/4.0)*(5.0*z*z - 3.0)*z;
    const double P31 = phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0);
    const double P32 = shnorm*std::sqrt(105.0/4.0)*(1.0 - z*z)*z;
    const double P33 = phase*shnorm*std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z);

    const double P40 = shnorm*std::sqrt(9.0/64.0)*((35.0*z*z - 30.0)*z*z + 3.0);
    const double P41 = phase*shnorm*std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z;
    const double P42 = shnorm*std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0);
    const double P43 = phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z;
    const double P44 = shnorm*std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z);

    zest::st::PlmRecursion recursion(order);

    std::vector<double> plm(zest::TriangleLayout::size(order));

    recursion.plm_real(z, zest::st::PlmSpan<double, sh_norm_param, sh_phase_param>(plm, recursion.max_order()));
    bool success = is_close(plm[0], P00, 1.0e-10)
            && is_close(plm[1], P10, 1.0e-10)
            && is_close(plm[2], P11, 1.0e-10)
            && is_close(plm[3], P20, 1.0e-10)
            && is_close(plm[4], P21, 1.0e-10)
            && is_close(plm[5], P22, 1.0e-10)
            && is_close(plm[6], P30, 1.0e-10)
            && is_close(plm[7], P31, 1.0e-10)
            && is_close(plm[8], P32, 1.0e-10)
            && is_close(plm[9], P33, 1.0e-10)
            && is_close(plm[10], P40, 1.0e-10)
            && is_close(plm[11], P41, 1.0e-10)
            && is_close(plm[12], P42, 1.0e-10)
            && is_close(plm[13], P43, 1.0e-10)
            && is_close(plm[14], P44, 1.0e-10);
    
    if (success)
        return true;
    else
    {
        std::printf("P00 %f %f\n", plm[0], P00);
        std::printf("P10 %f %f\n", plm[1], P10);
        std::printf("P11 %f %f\n", plm[2], P11);
        std::printf("P20 %f %f\n", plm[3], P20);
        std::printf("P21 %f %f\n", plm[4], P21);
        std::printf("P22 %f %f\n", plm[5], P22);
        std::printf("P30 %f %f\n", plm[6], P30);
        std::printf("P31 %f %f\n", plm[7], P31);
        std::printf("P32 %f %f\n", plm[8], P32);
        std::printf("P33 %f %f\n", plm[9], P33);
        std::printf("P40 %f %f\n", plm[10], P40);
        std::printf("P41 %f %f\n", plm[11], P41);
        std::printf("P42 %f %f\n", plm[12], P42);
        std::printf("P43 %f %f\n", plm[13], P43);
        std::printf("P44 %f %f\n", plm[14], P44);
        return false;
    }
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_plm_real_generates_real_vec_correct_up_to_order_5(double z)
{
    constexpr std::size_t order = 5;
    constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
    constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
        0.5*std::numbers::inv_sqrtpi : 1.0;

    const double P00 = shnorm;

    const double P10 = shnorm*std::sqrt(3.0)*z;
    const double P11 = phase*shnorm*std::sqrt(3.0)*std::sqrt(1.0 - z*z);

    const double P20 = shnorm*std::sqrt(5.0/4.0)*(3.0*z*z - 1.0);
    const double P21 = phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z;
    const double P22 = shnorm*std::sqrt(15.0/4.0)*(1.0 - z*z);

    const double P30 = shnorm*std::sqrt(7.0/4.0)*(5.0*z*z - 3.0)*z;
    const double P31 = phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0);
    const double P32 = shnorm*std::sqrt(105.0/4.0)*(1.0 - z*z)*z;
    const double P33 = phase*shnorm*std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z);

    const double P40 = shnorm*std::sqrt(9.0/64.0)*((35.0*z*z - 30.0)*z*z + 3.0);
    const double P41 = phase*shnorm*std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z;
    const double P42 = shnorm*std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0);
    const double P43 = phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z;
    const double P44 = shnorm*std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z);

    zest::st::PlmRecursion recursion(order);

    std::vector<double> plm(zest::TriangleLayout::size(order));

    recursion.plm_real(std::array<double, 1>{z}, zest::st::PlmVecSpan<double, sh_norm_param, sh_phase_param>(plm, recursion.max_order(), 1));
    bool success = is_close(plm[0], P00, 1.0e-10)
            && is_close(plm[1], P10, 1.0e-10)
            && is_close(plm[2], P11, 1.0e-10)
            && is_close(plm[3], P20, 1.0e-10)
            && is_close(plm[4], P21, 1.0e-10)
            && is_close(plm[5], P22, 1.0e-10)
            && is_close(plm[6], P30, 1.0e-10)
            && is_close(plm[7], P31, 1.0e-10)
            && is_close(plm[8], P32, 1.0e-10)
            && is_close(plm[9], P33, 1.0e-10)
            && is_close(plm[10], P40, 1.0e-10)
            && is_close(plm[11], P41, 1.0e-10)
            && is_close(plm[12], P42, 1.0e-10)
            && is_close(plm[13], P43, 1.0e-10)
            && is_close(plm[14], P44, 1.0e-10);
    
    if (success)
        return true;
    else
    {
        std::printf("P00 %f %f\n", plm[0], P00);
        std::printf("P10 %f %f\n", plm[1], P10);
        std::printf("P11 %f %f\n", plm[2], P11);
        std::printf("P20 %f %f\n", plm[3], P20);
        std::printf("P21 %f %f\n", plm[4], P21);
        std::printf("P22 %f %f\n", plm[5], P22);
        std::printf("P30 %f %f\n", plm[6], P30);
        std::printf("P31 %f %f\n", plm[7], P31);
        std::printf("P32 %f %f\n", plm[8], P32);
        std::printf("P33 %f %f\n", plm[9], P33);
        std::printf("P40 %f %f\n", plm[10], P40);
        std::printf("P41 %f %f\n", plm[11], P41);
        std::printf("P42 %f %f\n", plm[12], P42);
        std::printf("P43 %f %f\n", plm[13], P43);
        std::printf("P44 %f %f\n", plm[14], P44);
        return false;
    }
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
void test_plm_recursion()
{
    assert((test_plm_real_generates_real_correct_up_to_order_5<sh_norm_param, sh_phase_param>(1.0)));
    assert((test_plm_real_generates_real_correct_up_to_order_5<sh_norm_param, sh_phase_param>(-1.0)));
    assert((test_plm_real_generates_real_correct_up_to_order_5<sh_norm_param, sh_phase_param>(0.0)));
    assert((test_plm_real_generates_real_correct_up_to_order_5<sh_norm_param, sh_phase_param>(0.9741683087648949)));

    assert((test_plm_real_generates_real_vec_correct_up_to_order_5<sh_norm_param, sh_phase_param>(1.0)));
    assert((test_plm_real_generates_real_vec_correct_up_to_order_5<sh_norm_param, sh_phase_param>(-1.0)));
    assert((test_plm_real_generates_real_vec_correct_up_to_order_5<sh_norm_param, sh_phase_param>(0.0)));
    assert((test_plm_real_generates_real_vec_correct_up_to_order_5<sh_norm_param, sh_phase_param>(0.9741683087648949)));
}

int main()
{
    test_plm_recursion<zest::st::SHNorm::geo, zest::st::SHPhase::none>();
    test_plm_recursion<zest::st::SHNorm::geo, zest::st::SHPhase::cs>();
    test_plm_recursion<zest::st::SHNorm::qm, zest::st::SHPhase::none>();
    test_plm_recursion<zest::st::SHNorm::qm, zest::st::SHPhase::cs>();
}