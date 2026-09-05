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
#include "sh_generator.hpp"

#include <print>

namespace
{

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

template <zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_real_sh_generator_generates_correct_up_to_order_5_symmetric(
    double lon, double colat)
{
    constexpr std::size_t order = 5;
    constexpr double phase = (sh_phase == zest::st::SHPhase::none) ? -1.0 : 1.0;
    constexpr double shnorm = (sh_norm == zest::st::SHNorm::unit) ?
        0.5*std::numbers::inv_sqrtpi : 1.0;

    const double z = std::cos(colat);
    const double Y00 = shnorm;

    const double Y1m1 = phase*shnorm*std::numbers::sqrt3*std::sqrt(1.0 - z*z)*std::sin(lon);
    const double Y10 = shnorm*std::numbers::sqrt3*z;
    const double Y11 = phase*shnorm*std::numbers::sqrt3*std::sqrt(1.0 - z*z)*std::cos(lon);

    const double Y2m2 = shnorm*std::sqrt(15.0/4.0)*(1.0 - z*z)*std::sin(2.0*lon);
    const double Y2m1 = phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::sin(lon);
    const double Y20 = shnorm*std::sqrt(5.0/4.0)*(3.0*z*z - 1.0);
    const double Y21 = phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    const double Y22 = shnorm*std::sqrt(15.0/4.0)*(1.0 - z*z)*std::cos(2.0*lon);

    const double Y3m3 = phase*shnorm*std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*std::sin(3.0*lon);
    const double Y3m2 = shnorm*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    const double Y3m1 = phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::sin(lon);
    const double Y30 = shnorm*std::sqrt(7.0/4.0)*(5.0*z*z - 3.0)*z;
    const double Y31 = phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    const double Y32 = shnorm*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::cos(2.0*lon);
    const double Y33 = phase*shnorm*std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*std::cos(3.0*lon);

    const double Y4m4 = shnorm*std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z)*std::sin(4.0*lon);
    const double Y4m3 = phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    const double Y4m2 = shnorm*std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0)*std::sin(2.0*lon);
    const double Y4m1 = phase*shnorm*std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z*std::sin(lon);
    const double Y40 = shnorm*std::sqrt(9.0/64.0)*((35.0*z*z - 30.0)*z*z + 3.0);
    const double Y41 = phase*shnorm*std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z*std::cos(lon);
    const double Y42 = shnorm*std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0)*std::cos(2.0*lon);
    const double Y43 = phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::cos(3.0*lon);
    const double Y44 = shnorm*std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z)*std::cos(4.0*lon);

    auto expansion = zest::st::RealSHGenerator{}
        .generate<zest::IndexingMode::symmetric, sh_norm, sh_phase>(lon, colat, order);

    bool success = is_close(expansion[0, 0], Y00, 1.0e-10)
            && is_close(expansion[1, -1], Y1m1, 1.0e-10)
            && is_close(expansion[1, 0], Y10, 1.0e-10)
            && is_close(expansion[1, 1], Y11, 1.0e-10)
            && is_close(expansion[2, -2], Y2m2, 1.0e-10)
            && is_close(expansion[2, -1], Y2m1, 1.0e-10)
            && is_close(expansion[2, 0], Y20, 1.0e-10)
            && is_close(expansion[2, 1], Y21, 1.0e-10)
            && is_close(expansion[2, 2], Y22, 1.0e-10)
            && is_close(expansion[3, -3], Y3m3, 1.0e-10)
            && is_close(expansion[3, -2], Y3m2, 1.0e-10)
            && is_close(expansion[3, -1], Y3m1, 1.0e-10)
            && is_close(expansion[3, 0], Y30, 1.0e-10)
            && is_close(expansion[3, 1], Y31, 1.0e-10)
            && is_close(expansion[3, 2], Y32, 1.0e-10)
            && is_close(expansion[3, 3], Y33, 1.0e-10)
            && is_close(expansion[4, -4], Y4m4, 1.0e-10)
            && is_close(expansion[4, -3], Y4m3, 1.0e-10)
            && is_close(expansion[4, -2], Y4m2, 1.0e-10)
            && is_close(expansion[4, -1], Y4m1, 1.0e-10)
            && is_close(expansion[4, 0], Y40, 1.0e-10)
            && is_close(expansion[4, 1], Y41, 1.0e-10)
            && is_close(expansion[4, 2], Y42, 1.0e-10)
            && is_close(expansion[4, 3], Y43, 1.0e-10)
            && is_close(expansion[4, 4], Y44, 1.0e-10);

    if (success)
        return true;
    else
    {
        std::println("Y00 {} {}", expansion[0, 0], Y00);
        std::println("Y1m1 {} {}", expansion[1, -1], Y1m1);
        std::println("Y10 {} {}", expansion[1, 0], Y10);
        std::println("Y11 {} {}", expansion[1, 1], Y11);
        std::println("Y2m2 {} {}", expansion[2, -2], Y2m2);
        std::println("Y2m1 {} {}", expansion[2, -1], Y2m1);
        std::println("Y20 {} {}", expansion[2, 0], Y20);
        std::println("Y21 {} {}", expansion[2, 1], Y21);
        std::println("Y22 {} {}", expansion[2, 2], Y22);
        std::println("Y3m3 {} {}", expansion[3, -3], Y3m3);
        std::println("Y3m2 {} {}", expansion[3, -2], Y3m2);
        std::println("Y3m1 {} {}", expansion[3, -1], Y3m1);
        std::println("Y30 {} {}", expansion[3, 0], Y30);
        std::println("Y31 {} {}", expansion[3, 1], Y31);
        std::println("Y32 {} {}", expansion[3, 2], Y32);
        std::println("Y33 {} {}", expansion[3, 3], Y33);
        std::println("Y4m4 {} {}", expansion[4, -4], Y4m4);
        std::println("Y4m3 {} {}", expansion[4, -3], Y4m3);
        std::println("Y4m2 {} {}", expansion[4, -2], Y4m2);
        std::println("Y4m1 {} {}", expansion[4, -1], Y4m1);
        std::println("Y40 {} {}", expansion[4, 0], Y40);
        std::println("Y41 {} {}", expansion[4, 1], Y41);
        std::println("Y42 {} {}", expansion[4, 2], Y42);
        std::println("Y43 {} {}", expansion[4, 3], Y43);
        std::println("Y44 {} {}", expansion[4, 4], Y44);
        return false;
    }
}

template <zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_real_sh_generator_generates_correct_up_to_order_5_zero_based(
    double lon, double colat)
{
    constexpr std::size_t order = 5;
    constexpr double phase = (sh_phase == zest::st::SHPhase::none) ? -1.0 : 1.0;
    constexpr double shnorm = (sh_norm == zest::st::SHNorm::unit) ?
        0.5*std::numbers::inv_sqrtpi : 1.0;

    const double z = std::cos(colat);
    const double Y00 = shnorm;

    const double Y1m1 = phase*shnorm*std::numbers::sqrt3*std::sqrt(1.0 - z*z)*std::sin(lon);
    const double Y10 = shnorm*std::numbers::sqrt3*z;
    const double Y11 = phase*shnorm*std::numbers::sqrt3*std::sqrt(1.0 - z*z)*std::cos(lon);

    const double Y2m2 = shnorm*std::sqrt(15.0/4.0)*(1.0 - z*z)*std::sin(2.0*lon);
    const double Y2m1 = phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::sin(lon);
    const double Y20 = shnorm*std::sqrt(5.0/4.0)*(3.0*z*z - 1.0);
    const double Y21 = phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    const double Y22 = shnorm*std::sqrt(15.0/4.0)*(1.0 - z*z)*std::cos(2.0*lon);

    const double Y3m3 = phase*shnorm*std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*std::sin(3.0*lon);
    const double Y3m2 = shnorm*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    const double Y3m1 = phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::sin(lon);
    const double Y30 = shnorm*std::sqrt(7.0/4.0)*(5.0*z*z - 3.0)*z;
    const double Y31 = phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    const double Y32 = shnorm*std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::cos(2.0*lon);
    const double Y33 = phase*shnorm*std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*std::cos(3.0*lon);

    const double Y4m4 = shnorm*std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z)*std::sin(4.0*lon);
    const double Y4m3 = phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    const double Y4m2 = shnorm*std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0)*std::sin(2.0*lon);
    const double Y4m1 = phase*shnorm*std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z*std::sin(lon);
    const double Y40 = shnorm*std::sqrt(9.0/64.0)*((35.0*z*z - 30.0)*z*z + 3.0);
    const double Y41 = phase*shnorm*std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z*std::cos(lon);
    const double Y42 = shnorm*std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0)*std::cos(2.0*lon);
    const double Y43 = phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::cos(3.0*lon);
    const double Y44 = shnorm*std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z)*std::cos(4.0*lon);

    auto expansion = zest::st::RealSHGenerator{}
        .generate<zest::IndexingMode::zero_based, sh_norm, sh_phase>(lon, colat, order);

    bool success = is_close(expansion[0, 0, 0], Y00, 1.0e-10)
            && is_close(expansion[1, 1, 1], Y1m1, 1.0e-10)
            && is_close(expansion[1, 0, 0], Y10, 1.0e-10)
            && is_close(expansion[1, 1, 0], Y11, 1.0e-10)
            && is_close(expansion[2, 2, 1], Y2m2, 1.0e-10)
            && is_close(expansion[2, 1, 1], Y2m1, 1.0e-10)
            && is_close(expansion[2, 0, 0], Y20, 1.0e-10)
            && is_close(expansion[2, 1, 0], Y21, 1.0e-10)
            && is_close(expansion[2, 2, 0], Y22, 1.0e-10)
            && is_close(expansion[3, 3, 1], Y3m3, 1.0e-10)
            && is_close(expansion[3, 2, 1], Y3m2, 1.0e-10)
            && is_close(expansion[3, 1, 1], Y3m1, 1.0e-10)
            && is_close(expansion[3, 0, 0], Y30, 1.0e-10)
            && is_close(expansion[3, 1, 0], Y31, 1.0e-10)
            && is_close(expansion[3, 2, 0], Y32, 1.0e-10)
            && is_close(expansion[3, 3, 0], Y33, 1.0e-10)
            && is_close(expansion[4, 4, 1], Y4m4, 1.0e-10)
            && is_close(expansion[4, 3, 1], Y4m3, 1.0e-10)
            && is_close(expansion[4, 2, 1], Y4m2, 1.0e-10)
            && is_close(expansion[4, 1, 1], Y4m1, 1.0e-10)
            && is_close(expansion[4, 0, 0], Y40, 1.0e-10)
            && is_close(expansion[4, 1, 0], Y41, 1.0e-10)
            && is_close(expansion[4, 2, 0], Y42, 1.0e-10)
            && is_close(expansion[4, 3, 0], Y43, 1.0e-10)
            && is_close(expansion[4, 4, 0], Y44, 1.0e-10);

    if (success)
        return true;
    else
    {
        std::println("Y00 {} {}", expansion[0, 0, 0], Y00);
        std::println("Y1m1 {} {}", expansion[1, 1, 1], Y1m1);
        std::println("Y10 {} {}", expansion[1, 0, 0], Y10);
        std::println("Y11 {} {}", expansion[1, 1, 0], Y11);
        std::println("Y2m2 {} {}", expansion[2, 2, 1], Y2m2);
        std::println("Y2m1 {} {}", expansion[2, 1, 1], Y2m1);
        std::println("Y20 {} {}", expansion[2, 0, 0], Y20);
        std::println("Y21 {} {}", expansion[2, 1, 0], Y21);
        std::println("Y22 {} {}", expansion[2, 2, 0], Y22);
        std::println("Y3m3 {} {}", expansion[3, 3, 1], Y3m3);
        std::println("Y3m2 {} {}", expansion[3, 2, 1], Y3m2);
        std::println("Y3m1 {} {}", expansion[3, 1, 1], Y3m1);
        std::println("Y30 {} {}", expansion[3, 0, 0], Y30);
        std::println("Y31 {} {}", expansion[3, 1, 0], Y31);
        std::println("Y32 {} {}", expansion[3, 2, 0], Y32);
        std::println("Y33 {} {}", expansion[3, 3, 0], Y33);
        std::println("Y4m4 {} {}", expansion[4, 4, 1], Y4m4);
        std::println("Y4m3 {} {}", expansion[4, 3, 1], Y4m3);
        std::println("Y4m2 {} {}", expansion[4, 2, 1], Y4m2);
        std::println("Y4m1 {} {}", expansion[4, 1, 1], Y4m1);
        std::println("Y40 {} {}", expansion[4, 0, 0], Y40);
        std::println("Y41 {} {}", expansion[4, 1, 0], Y41);
        std::println("Y42 {} {}", expansion[4, 2, 0], Y42);
        std::println("Y43 {} {}", expansion[4, 3, 0], Y43);
        std::println("Y44 {} {}", expansion[4, 4, 0], Y44);
        return false;
    }
}

template <zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
void test_sh_generator()
{
    assert((test_real_sh_generator_generates_correct_up_to_order_5_symmetric<sh_norm, sh_phase>(0.0, 0.0)));
    assert((test_real_sh_generator_generates_correct_up_to_order_5_symmetric<sh_norm, sh_phase>(0.5*std::numbers::pi, 0.0)));
    assert((test_real_sh_generator_generates_correct_up_to_order_5_symmetric<sh_norm, sh_phase>(0.0, 0.5*std::numbers::pi)));
    assert((test_real_sh_generator_generates_correct_up_to_order_5_symmetric<sh_norm, sh_phase>(0.5*std::numbers::pi, 0.5*std::numbers::pi)));
    assert((test_real_sh_generator_generates_correct_up_to_order_5_symmetric<sh_norm, sh_phase>(1.0, 1.0)));

    assert((test_real_sh_generator_generates_correct_up_to_order_5_zero_based<sh_norm, sh_phase>(0.0, 0.0)));
    assert((test_real_sh_generator_generates_correct_up_to_order_5_zero_based<sh_norm, sh_phase>(0.5*std::numbers::pi, 0.0)));
    assert((test_real_sh_generator_generates_correct_up_to_order_5_zero_based<sh_norm, sh_phase>(0.0, 0.5*std::numbers::pi)));
    assert((test_real_sh_generator_generates_correct_up_to_order_5_zero_based<sh_norm, sh_phase>(0.5*std::numbers::pi, 0.5*std::numbers::pi)));
    assert((test_real_sh_generator_generates_correct_up_to_order_5_zero_based<sh_norm, sh_phase>(1.0, 1.0)));
}

} // namespace

int main()
{
    test_sh_generator<zest::st::SHNorm::four_pi, zest::st::SHPhase::none>();
    test_sh_generator<zest::st::SHNorm::four_pi, zest::st::SHPhase::cs>();
    test_sh_generator<zest::st::SHNorm::unit, zest::st::SHPhase::none>();
    test_sh_generator<zest::st::SHNorm::unit, zest::st::SHPhase::cs>();
}
