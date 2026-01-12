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
#include "lsq_transformer.hpp"
#include "sequence.hpp"
#include "sh_generator.hpp"

#include <cassert>
#include <cmath>
#include <print>
#include <random>

namespace
{

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

template <zest::IndexingMode indexing_mode, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_lsq_geo_expansion_expands_Y00()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> colat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen{1337};
    std::uniform_real_distribution dist{0.0, 1.0};

    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);

    for (std::size_t i = 0; i < num_points; ++i)
        colat[i] = std::numbers::pi*(dist(gen) - 0.5);

    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::cos(colat[i]));

    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<indexing_mode, sh_norm, sh_phase>(values, lon, colat, order);

    const double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        for (auto m : expansion_l.indices())
        {
            if constexpr (indexing_mode == zest::IndexingMode::zero_based)
            {
                if (l == 0 && m == 0)
                {
                    if (is_close(expansion_l[m, 0], reference_coeff, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m, 0], 0.0, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
            else
            {
                if (l == 0 && m == 0)
                {
                    if (is_close(expansion_l[m], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto l : expansion.indices())
        {
            auto expansion_l = expansion[l];
            for (auto m : expansion_l.indices())
            {
                if constexpr (indexing_mode == zest::IndexingMode::zero_based)
                    std::println("{} {} {} {}", l, m, expansion_l[m, 0], expansion_l[m, 1]);
                else
                    std::println("{} {} {}", l, m, expansion_l[m]);
            }
        }
    }
    return success;
}

template <zest::IndexingMode indexing_mode, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_lsq_geo_expansion_expands_Y21()
{
    constexpr std::size_t order = 6;

    auto function = [](double lon, double z)
    {
        constexpr double phase = (sh_phase == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> colat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen{1337};
    std::uniform_real_distribution dist{0.0, 1.0};

    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);

    for (std::size_t i = 0; i < num_points; ++i)
        colat[i] = std::numbers::pi*(dist(gen) - 0.5);

    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::cos(colat[i]));

    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<indexing_mode, sh_norm, sh_phase>(values, lon, colat, order);

    const double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        for (auto m : expansion_l.indices())
        {
            if constexpr (indexing_mode == zest::IndexingMode::zero_based)
            {
                if (l == 2 && m == 1)
                {
                    if (is_close(expansion_l[m, 0], reference_coeff, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m, 0], 0.0, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
            else
            {
                if (l == 2 && m == 1)
                {
                    if (is_close(expansion_l[m], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto l : expansion.indices())
        {
            auto expansion_l = expansion[l];
            for (auto m : expansion_l.indices())
            {
                if constexpr (indexing_mode == zest::IndexingMode::zero_based)
                    std::println("{} {} {} {}", l, m, expansion_l[m, 0], expansion_l[m, 1]);
                else
                    std::println("{} {} {}", l, m, expansion_l[m]);
            }
        }
    }
    return success;
}

template <zest::IndexingMode indexing_mode, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_lsq_geo_expansion_expands_Y31()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double phase = (sh_phase == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> colat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen{1337};
    std::uniform_real_distribution dist{0.0, 1.0};


    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);

    for (std::size_t i = 0; i < num_points; ++i)
        colat[i] = std::numbers::pi*(dist(gen) - 0.5);

    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::cos(colat[i]));

    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<indexing_mode, sh_norm, sh_phase>(values, lon, colat, order);

    const double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        for (auto m : expansion_l.indices())
        {
            if constexpr (indexing_mode == zest::IndexingMode::zero_based)
            {
                if (l == 3 && m == 1)
                {
                    if (is_close(expansion_l[m, 0], reference_coeff, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m, 0], 0.0, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
            else
            {
                if (l == 3 && m == 1)
                {
                    if (is_close(expansion_l[m], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto l : expansion.indices())
        {
            auto expansion_l = expansion[l];
            for (auto m : expansion_l.indices())
            {
                if constexpr (indexing_mode == zest::IndexingMode::zero_based)
                    std::println("{} {} {} {}", l, m, expansion_l[m, 0], expansion_l[m, 1]);
                else
                    std::println("{} {} {}", l, m, expansion_l[m]);
            }
        }
    }
    return success;
}

template <zest::IndexingMode indexing_mode, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_lsq_geo_expansion_expands_Y4m3()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double phase = (sh_phase == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> colat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen{1337};
    std::uniform_real_distribution dist{0.0, 1.0};


    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);

    for (std::size_t i = 0; i < num_points; ++i)
        colat[i] = std::numbers::pi*(dist(gen) - 0.5);

    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::cos(colat[i]));

    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<indexing_mode, sh_norm, sh_phase>(values, lon, colat, order);

    const double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        for (auto m : expansion_l.indices())
        {
            if constexpr (indexing_mode == zest::IndexingMode::zero_based)
            {
                if (l == 4 && m == 3)
                {
                    if (is_close(expansion_l[m, 0], 0.0, tol)
                            && is_close(expansion_l[m, 1], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m, 0], 0.0, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
            else
            {
                if (l == 4 && m == -3)
                {
                    if (is_close(expansion_l[m], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto l : expansion.indices())
        {
            auto expansion_l = expansion[l];
            for (auto m : expansion_l.indices())
            {
                if constexpr (indexing_mode == zest::IndexingMode::zero_based)
                    std::println("{} {} {} {}", l, m, expansion_l[m, 0], expansion_l[m, 1]);
                else
                    std::println("{} {} {}", l, m, expansion_l[m]);
            }
        }
    }
    return success;
}

template <zest::IndexingMode indexing_mode, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
bool test_lsq_geo_expansion_expands_Y31_plus_Y4m3()
{
    constexpr std::size_t order = 6;

    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double z)
    {
        constexpr double phase = (sh_phase == zest::st::SHPhase::none) ? -1.0 : 1.0;
        constexpr double shnorm = (sh_norm == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return phase*shnorm*(std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon) + std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon));
    };

    constexpr std::size_t num_points = 100;
    std::vector<double> lon(num_points);
    std::vector<double> colat(num_points);
    std::vector<double> values(num_points);

    std::mt19937 gen{1337};
    std::uniform_real_distribution dist{0.0, 1.0};


    for (std::size_t i = 0; i < num_points; ++i)
        lon[i] = 2.0*std::numbers::pi*dist(gen);

    for (std::size_t i = 0; i < num_points; ++i)
        colat[i] = std::numbers::pi*(dist(gen) - 0.5);

    for (std::size_t i = 0; i < num_points; ++i)
        values[i] = function(lon[i], std::cos(colat[i]));

    zest::st::LSQTransformer transformer(order);

    auto expansion = transformer.transform<indexing_mode, sh_norm, sh_phase>(values, lon, colat, order);

    const double reference_coeff = 1.0;

    constexpr double tol = 1.0e-10;

    bool success = true;
    for (auto l : expansion.indices())
    {
        auto expansion_l = expansion[l];
        for (auto m : expansion_l.indices())
        {
            if constexpr (indexing_mode == zest::IndexingMode::zero_based)
            {
                if (l == 3 && m == 1)
                {
                    if (is_close(expansion_l[m, 0], reference_coeff, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else if (l == 4 && m == 3)
                {
                    if (is_close(expansion_l[m, 0], 0.0, tol)
                            && is_close(expansion_l[m, 1], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m, 0], 0.0, tol)
                            && is_close(expansion_l[m, 1], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
            else
            {
                if (l == 3 && m == 1)
                {
                    if (is_close(expansion_l[m], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else if (l == 4 && m == -3)
                {
                    if (is_close(expansion_l[m], reference_coeff, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
                else
                {
                    if (is_close(expansion_l[m], 0.0, tol))
                        success = success && true;
                    else
                        success = success && false;
                }
            }
        }
    }

    if (!success)
    {
        for (auto l : expansion.indices())
        {
            auto expansion_l = expansion[l];
            for (auto m : expansion_l.indices())
            {
                if constexpr (indexing_mode == zest::IndexingMode::zero_based)
                    std::println("{} {} {} {}", l, m, expansion_l[m, 0], expansion_l[m, 1]);
                else
                    std::println("{} {} {}", l, m, expansion_l[m]);
            }
        }
    }
    return success;
}

template <zest::IndexingMode indexing_mode, zest::st::SHNorm sh_norm, zest::st::SHPhase sh_phase>
void test_lsq()
{
    assert((test_lsq_geo_expansion_expands_Y00<indexing_mode, sh_norm, sh_phase>()));
    assert((test_lsq_geo_expansion_expands_Y21<indexing_mode, sh_norm, sh_phase>()));
    assert((test_lsq_geo_expansion_expands_Y31<indexing_mode, sh_norm, sh_phase>()));
    assert((test_lsq_geo_expansion_expands_Y4m3<indexing_mode, sh_norm, sh_phase>()));
    assert((test_lsq_geo_expansion_expands_Y31_plus_Y4m3<indexing_mode, sh_norm, sh_phase>()));
}

} // namespace

int main()
{
    test_lsq<zest::IndexingMode::symmetric, zest::st::SHNorm::geo, zest::st::SHPhase::none>();
    test_lsq<zest::IndexingMode::symmetric, zest::st::SHNorm::geo, zest::st::SHPhase::cs>();
    test_lsq<zest::IndexingMode::symmetric, zest::st::SHNorm::qm, zest::st::SHPhase::none>();
    test_lsq<zest::IndexingMode::symmetric, zest::st::SHNorm::qm, zest::st::SHPhase::cs>();

    test_lsq<zest::IndexingMode::zero_based, zest::st::SHNorm::geo, zest::st::SHPhase::none>();
    test_lsq<zest::IndexingMode::zero_based, zest::st::SHNorm::geo, zest::st::SHPhase::cs>();
    test_lsq<zest::IndexingMode::zero_based, zest::st::SHNorm::qm, zest::st::SHPhase::none>();
    test_lsq<zest::IndexingMode::zero_based, zest::st::SHNorm::qm, zest::st::SHPhase::cs>();
}
