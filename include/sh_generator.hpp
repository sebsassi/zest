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
#pragma once

#include <cassert>
#include <span>
#include <vector>

#include "associated_legendre_recursion.hpp"
#include "md_span.hpp"
#include "sequence.hpp"
#include "sh_concepts.hpp"
#include "sh_conventions.hpp"
#include "sh_expansion.hpp"

namespace zest::st
{

/**
     @brief Generator of real spherical harmonics. 

    This class enables generation of collections of real spherical harmonics
    evaluated at a point using recursion formulae.
*/
class RealSHGenerator
{
public:
    RealSHGenerator() = default;
    explicit RealSHGenerator(std::size_t max_order);

    /**
        @brief Maximum order with cached recursion coefficients.
    */
    [[nodiscard]] std::size_t max_order() const noexcept
    {
        return m_recursion.max_order();
    }

    [[nodiscard]] std::size_t inner_size() const noexcept
    {
        return m_z.size();
    }

    /**
        @brief Increase the maximum order for which recursion coefficients are
        cached.

        @param max_order new maximum order
    */
    void expand(std::size_t max_order);

    void expand(std::size_t max_order, std::size_t inner_size);

    /**
        @brief Generate spherical harmonics at coordinates `lon`, `colat`.

        @tparam SHType type of spherical harmonic buffer

        @param lon longitude coordinate
        @param colat colatitude coordinate
        @param expansion buffer for spherical harmonic values
    */
    template <any_complete_sh_expansion ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
            && st::has_inner_rank<ExpansionType, 0>
    void generate(double lon, double colat, ExpansionType&& expansion)
    {
        assert(0.0 <= colat && colat <= std::numbers::pi);
        const std::size_t order = std::forward<ExpansionType>(expansion).order();
        expand(order);

        constexpr st::SHNorm sh_norm = sh_norm_of<ExpansionType>();
        constexpr st::SHPhase sh_phase = sh_phase_of<ExpansionType>();
        constexpr IndexingMode indexing_mode = indexing_mode_of<ExpansionType>();

        const double z = std::cos(colat);
        AssociatedLegendreSpan<double, sh_norm, sh_phase> ass_leg{m_ass_leg_poly, order};

        m_recursion.generate_real(z, ass_leg);

        for (std::size_t m = 0; m < order; ++m)
        {
            const double angle = double(m)*lon;
            m_cossin[m] = {std::cos(angle), std::sin(angle)};
        }

        for (auto l : ass_leg.indices())
        {
            const auto expansion_l = std::forward<ExpansionType>(expansion)[l];
            const auto ass_leg_l = ass_leg[l];
            if constexpr (indexing_mode == IndexingMode::symmetric)
                expansion_l[0] = ass_leg_l[0];
            else if constexpr (indexing_mode == IndexingMode::zero_based)
            {
                expansion_l[0, 0] = ass_leg_l[0];
                expansion_l[0, 1] = 0.0;
            }

            for (auto m : ass_leg_l.indices(1))
            {
                const double ass_leg_lm = ass_leg_l[m];

                if constexpr (indexing_mode == IndexingMode::symmetric)
                {
                    expansion_l[int(m)] = ass_leg_lm*m_cossin[m][0];
                    expansion_l[-int(m)] = ass_leg_lm*m_cossin[m][1];
                }
                else if constexpr (indexing_mode == IndexingMode::zero_based)
                {
                    expansion_l[m, 0] = ass_leg_lm*m_cossin[m][0];
                    expansion_l[m, 1] = ass_leg_lm*m_cossin[m][1];
                }
            }
        }
    }

    template <any_complete_sh_expansion ExpansionType>
        requires std::floating_point<value_type_of<ExpansionType>>
            && st::has_inner_rank<ExpansionType, 1>
    void generate(
        std::span<const double> lon, std::span<const double> colat, ExpansionType&& expansion)
    {
        assert(lon.size() == colat.size());
        for (double element : colat)
            assert(0.0 <= element && element <= std::numbers::pi);

        const std::size_t order = std::forward<ExpansionType>(expansion).order();
        expand(order, lon.size());

        constexpr st::SHNorm sh_norm = sh_norm_of<ExpansionType>();
        constexpr st::SHPhase sh_phase = sh_phase_of<ExpansionType>();
        constexpr IndexingMode indexing_mode = indexing_mode_of<ExpansionType>();

        for (std::size_t i = 0; i < colat.size(); ++i)
            m_z[i] = std::cos(colat[i]);

        AssociatedLegendreSpan<double, sh_norm, sh_phase, std::dynamic_extent>
        ass_leg{m_ass_leg_poly, order, m_z.size()};

        m_recursion.generate_real(m_z, ass_leg);

        DynamicMDSpan<std::array<double, 2>, 2> cossin{m_cossin.data(), order, lon.size()};

        for (std::size_t m = 0; m < order; ++m)
        {
            auto cossin_m = cossin[m];
            for (std::size_t i = 0; i < lon.size(); ++i)
            {
                const double angle = double(m)*lon[i];
                cossin_m[i] = {std::cos(angle), std::sin(angle)};
            }
        }

        for (auto l : ass_leg.indices())
        {
            const auto expansion_l = std::forward<ExpansionType>(expansion)[l];
            const auto ass_leg_l = ass_leg[l];

            auto ass_leg_l0 = ass_leg_l[0];
            if constexpr (indexing_mode == IndexingMode::symmetric)
            {
                auto expansion_l0 = expansion_l[0];
                for (std::size_t i = 0; i < colat.size(); ++i)
                    expansion_l0[i] = ass_leg_l0[i];
            }
            else if constexpr (indexing_mode == IndexingMode::zero_based)
            {
                auto expansion_l00 = expansion_l[0, 0];
                auto expansion_l01 = expansion_l[0, 1];
                for (std::size_t i = 0; i < colat.size(); ++i)
                    expansion_l00[i] = ass_leg_l0[i];
                for (std::size_t i = 0; i < colat.size(); ++i)
                    expansion_l01[i] = 0.0;
            }

            for (auto m : ass_leg_l.indices(1))
            {
                auto ass_leg_lm = ass_leg_l[m];
                auto cossin_m = cossin[m];

                if constexpr (indexing_mode == IndexingMode::symmetric)
                {
                    auto expansion_lpm = expansion_l[int(m)];
                    auto expansion_lmm = expansion_l[-int(m)];
                    for (std::size_t i = 0; i < lon.size(); ++i)
                        expansion_lpm = ass_leg_lm[i]*cossin_m[i][0];
                    for (std::size_t i = 0; i < lon.size(); ++i)
                        expansion_lmm = ass_leg_lm[i]*cossin_m[i][1];
                }
                else if constexpr (indexing_mode == IndexingMode::zero_based)
                {
                    auto expansion_lm0 = expansion_l[m, 0];
                    auto expansion_lm1 = expansion_l[m, 1];
                    for (std::size_t i = 0; i < lon.size(); ++i)
                        expansion_lm0[i] = ass_leg_lm[i]*cossin_m[i][0];
                    for (std::size_t i = 0; i < lon.size(); ++i)
                        expansion_lm1[i] = ass_leg_lm[i]*cossin_m[i][1];
                }
            }
        }
    }

    template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
    [[nodiscard]] SHExpansion<double, indexing_mode, sh_norm, sh_phase>
    generate(double lon, double colat, std::size_t order)
    {
        SHExpansion<double, indexing_mode, sh_norm, sh_phase> expansion{order};
        generate(lon, colat, expansion);
        return expansion;
    }

    template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
    [[nodiscard]] SHExpansion<double, indexing_mode, sh_norm, sh_phase, std::dynamic_extent>
    generate(std::span<const double> lon, std::span<const double> colat, std::size_t order)
    {
        SHExpansion<double, indexing_mode, sh_norm, sh_phase, std::dynamic_extent>
        expansion{order, lon.size()};

        generate(lon, colat, expansion);
        return expansion;
    }

private:
    AssociatedLegendreRecursion m_recursion;
    std::vector<double> m_ass_leg_poly;
    std::vector<double> m_z;
    std::vector<std::array<double, 2>> m_cossin;
};

} // namespace zest::st

