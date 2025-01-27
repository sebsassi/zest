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
#include "lsq_transformer.hpp"

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
bool test_real_sh_generator_generates_correct_up_to_order_5(
    double lon, double lat)
{
    constexpr std::size_t order = 5;
    constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
    constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
        0.5*std::numbers::inv_sqrtpi : 1.0;

    const double z = std::sin(lat);
    const double Y00 = shnorm;

    const double Y1m1 = phase*shnorm*std::sqrt(3.0)*std::sqrt(1.0 - z*z)*std::sin(lon);
    const double Y10 = shnorm*std::sqrt(3.0)*z;
    const double Y11 = phase*shnorm*std::sqrt(3.0)*std::sqrt(1.0 - z*z)*std::cos(lon);
    
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

    zest::st::RealSHGenerator generator(order);

    using YlmSpan = zest::st::RealSHSpan<double, sh_norm_param, sh_phase_param>;
    std::vector<double> ylm(YlmSpan::Layout::size(order));

    generator.generate(lon, lat, YlmSpan(ylm, order));

    bool success = is_close(ylm[0], Y00, 1.0e-10)
            && is_close(ylm[1], Y1m1, 1.0e-10)
            && is_close(ylm[2], Y10, 1.0e-10)
            && is_close(ylm[3], Y11, 1.0e-10)
            && is_close(ylm[4], Y2m2, 1.0e-10)
            && is_close(ylm[5], Y2m1, 1.0e-10)
            && is_close(ylm[6], Y20, 1.0e-10)
            && is_close(ylm[7], Y21, 1.0e-10)
            && is_close(ylm[8], Y22, 1.0e-10)
            && is_close(ylm[9], Y3m3, 1.0e-10)
            && is_close(ylm[10], Y3m2, 1.0e-10)
            && is_close(ylm[11], Y3m1, 1.0e-10)
            && is_close(ylm[12], Y30, 1.0e-10)
            && is_close(ylm[13], Y31, 1.0e-10)
            && is_close(ylm[14], Y32, 1.0e-10)
            && is_close(ylm[15], Y33, 1.0e-10)
            && is_close(ylm[16], Y4m4, 1.0e-10)
            && is_close(ylm[17], Y4m3, 1.0e-10)
            && is_close(ylm[18], Y4m2, 1.0e-10)
            && is_close(ylm[19], Y4m1, 1.0e-10)
            && is_close(ylm[20], Y40, 1.0e-10)
            && is_close(ylm[21], Y41, 1.0e-10)
            && is_close(ylm[22], Y42, 1.0e-10)
            && is_close(ylm[23], Y43, 1.0e-10)
            && is_close(ylm[24], Y44, 1.0e-10);
    
    if (success)
        return true;
    else
    {
        std::printf("Y00 %f %f\n", ylm[0], Y00);
        std::printf("Y1m1 %f %f\n", ylm[1], Y1m1);
        std::printf("Y10 %f %f\n", ylm[2], Y10);
        std::printf("Y11 %f %f\n", ylm[3], Y11);
        std::printf("Y2m2 %f %f\n", ylm[4], Y2m2);
        std::printf("Y2m1 %f %f\n", ylm[5], Y2m1);
        std::printf("Y20 %f %f\n", ylm[6], Y20);
        std::printf("Y21 %f %f\n", ylm[7], Y21);
        std::printf("Y22 %f %f\n", ylm[8], Y22);
        std::printf("Y3m3 %f %f\n", ylm[9], Y3m3);
        std::printf("Y3m2 %f %f\n", ylm[10], Y3m2);
        std::printf("Y3m1 %f %f\n", ylm[11], Y3m1);
        std::printf("Y30 %f %f\n", ylm[12], Y30);
        std::printf("Y31 %f %f\n", ylm[13], Y31);
        std::printf("Y32 %f %f\n", ylm[14], Y32);
        std::printf("Y33 %f %f\n", ylm[15], Y33);
        std::printf("Y4m4 %f %f\n", ylm[16], Y4m4);
        std::printf("Y4m3 %f %f\n", ylm[17], Y4m3);
        std::printf("Y4m2 %f %f\n", ylm[18], Y4m2);
        std::printf("Y4m1 %f %f\n", ylm[19], Y4m1);
        std::printf("Y40 %f %f\n", ylm[20], Y40);
        std::printf("Y41 %f %f\n", ylm[21], Y41);
        std::printf("Y42 %f %f\n", ylm[22], Y42);
        std::printf("Y43 %f %f\n", ylm[23], Y43);
        std::printf("Y44 %f %f\n", ylm[24], Y44);
        return false;
    }
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_lsq_geo_expansion_expands_Y00()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> lat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};


    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);
    
    for (std::size_t i = 0; i < num_points; ++i)
        lat[i] = std::numbers::pi*(dist(gen) - 0.5);
    
    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::sin(lat[i]));
    
    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<sh_norm_param, sh_phase_param>(values, lat, lon, order);

    const auto& coeffs = expansion.flatten();

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
        for (const auto& coeff : expansion.flatten())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_lsq_geo_expansion_expands_Y21()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> lat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};


    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);
    
    for (std::size_t i = 0; i < num_points; ++i)
        lat[i] = std::numbers::pi*(dist(gen) - 0.5);
    
    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::sin(lat[i]));
    
    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<sh_norm_param, sh_phase_param>(values, lat, lon, order);

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 4)
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
        for (const auto& coeff : expansion.flatten())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_lsq_geo_expansion_expands_Y31()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> lat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};


    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);
    
    for (std::size_t i = 0; i < num_points; ++i)
        lat[i] = std::numbers::pi*(dist(gen) - 0.5);
    
    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::sin(lat[i]));
    
    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<sh_norm_param, sh_phase_param>(values, lat, lon, order);

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 7)
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
        for (const auto& coeff : expansion.flatten())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_lsq_geo_expansion_expands_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> lat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};


    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);
    
    for (std::size_t i = 0; i < num_points; ++i)
        lat[i] = std::numbers::pi*(dist(gen) - 0.5);
    
    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::sin(lat[i]));
    
    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<sh_norm_param, sh_phase_param>(values, lat, lon, order);

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 13)
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
        for (const auto& coeff : expansion.flatten())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
bool test_lsq_geo_expansion_expands_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double phase = (sh_phase_param == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*(std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon));
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> lat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};


    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);
    
    for (std::size_t i = 0; i < num_points; ++i)
        lat[i] = std::numbers::pi*(dist(gen) - 0.5);
    
    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::sin(lat[i]));
    
    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<sh_norm_param, sh_phase_param>(values, lat, lon, order);

    const auto& coeffs = expansion.flatten();

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (std::size_t i = 0; i < coeffs.size(); ++i)
    {
        if (i == 7)
        {
            if (is_close(coeffs[i][0], 1.0, tol)
                    && is_close(coeffs[i][1], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
        else if (i == 13)
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
        for (const auto& coeff : expansion.flatten())
            std::printf("%f %f\n", coeff[0], coeff[1]);
    }
    return success;
}

template <zest::st::SHNorm sh_norm_param, zest::st::SHPhase sh_phase_param>
void test_lsq()
{
    assert((test_real_sh_generator_generates_correct_up_to_order_5<sh_norm_param, sh_phase_param>(0.0, 0.0)));

    assert((test_lsq_geo_expansion_expands_Y00<sh_norm_param, sh_phase_param>()));
    assert((test_lsq_geo_expansion_expands_Y21<sh_norm_param, sh_phase_param>()));
    assert((test_lsq_geo_expansion_expands_Y31<sh_norm_param, sh_phase_param>()));
    assert((test_lsq_geo_expansion_expands_Y4m3<sh_norm_param, sh_phase_param>()));
    assert((test_lsq_geo_expansion_expands_Y31_plus_Y4m3<sh_norm_param, sh_phase_param>()));
}

int main()
{
    test_lsq<zest::st::SHNorm::geo, zest::st::SHPhase::none>();
    test_lsq<zest::st::SHNorm::geo, zest::st::SHPhase::cs>();
    test_lsq<zest::st::SHNorm::qm, zest::st::SHPhase::none>();
    test_lsq<zest::st::SHNorm::qm, zest::st::SHPhase::cs>();
}