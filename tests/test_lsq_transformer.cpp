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

bool test_ylm_generator_generates_correct_up_to_order_5(double lon, double lat)
{
    constexpr std::size_t order = 5;

    const double z = std::sin(lat);
    const double Y00 = 1.0;

    const double Y1m1 = std::sqrt(3.0)*std::sqrt(1.0 - z*z)*std::sin(lon);
    const double Y10 = std::sqrt(3.0)*z;
    const double Y11 = std::sqrt(3.0)*std::sqrt(1.0 - z*z)*std::cos(lon);
    
    const double Y2m2 = std::sqrt(15.0/4.0)*(1.0 - z*z)*std::sin(2.0*lon);
    const double Y2m1 = std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::sin(lon);
    const double Y20 = std::sqrt(5.0/4.0)*(3.0*z*z - 1.0);
    const double Y21 = std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    const double Y22 = std::sqrt(15.0/4.0)*(1.0 - z*z)*std::cos(2.0*lon);

    const double Y3m3 = std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*std::sin(3.0*lon);
    const double Y3m2 = std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::sin(2.0*lon);
    const double Y3m1 = std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::sin(lon);
    const double Y30 = std::sqrt(7.0/4.0)*(5.0*z*z - 3.0)*z;
    const double Y31 = std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    const double Y32 = std::sqrt(105.0/4.0)*(1.0 - z*z)*z*std::cos(2.0*lon);
    const double Y33 = std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*std::cos(3.0*lon);

    const double Y4m4 = std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z)*std::sin(4.0*lon);
    const double Y4m3 = std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    const double Y4m2 = std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0)*std::sin(2.0*lon);
    const double Y4m1 = std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z*std::sin(lon);
    const double Y40 = std::sqrt(9.0/64.0)*((35.0*z*z - 30.0)*z*z + 3.0);
    const double Y41 = std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z*std::cos(lon);
    const double Y42 = std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0)*std::cos(2.0*lon);
    const double Y43 = std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::cos(3.0*lon);
    const double Y44 = std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z)*std::cos(4.0*lon);

    zest::st::RealYlmGenerator generator(order);

    std::vector<double> ylm(zest::DualTriangleLayout::size(order));

    generator.generate<zest::st::SequentialRealYlmPacking, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>(
            zest::st::RealYlmSpan<zest::st::SequentialRealYlmPacking, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>(ylm, order), lon, lat);

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

bool test_lsq_geo_expansion_expands_Y00()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        return 1.0;
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

    auto expansion = transformer.transform(values, lat, lon, order);

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

bool test_lsq_geo_expansion_expands_Y21()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        return std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
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

    auto expansion = transformer.transform(values, lat, lon, order);

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

bool test_lsq_geo_expansion_expands_Y31()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        return std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
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

    auto expansion = transformer.transform(values, lat, lon, order);

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

bool test_lsq_geo_expansion_expands_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        return std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
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

    auto expansion = transformer.transform(values, lat, lon, order);

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

bool test_lsq_geo_expansion_expands_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6; 

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        return std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
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

    auto expansion = transformer.transform(values, lat, lon, order);

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

int main()
{
    assert(test_ylm_generator_generates_correct_up_to_order_5(0.0, 0.0));

    assert(test_lsq_geo_expansion_expands_Y00());
    assert(test_lsq_geo_expansion_expands_Y21());
    assert(test_lsq_geo_expansion_expands_Y31());
    assert(test_lsq_geo_expansion_expands_Y4m3());
    assert(test_lsq_geo_expansion_expands_Y31_plus_Y4m3());
}