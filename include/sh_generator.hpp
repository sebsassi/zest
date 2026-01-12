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

#include <vector>

#include "associated_legendre_recursion.hpp"
#include "sh_expansion.hpp"

namespace zest::st
{

/**
     @brief Generator of real spherical harmonics. 

    This class enables generation of collections of real spherical harmonics evaluated at
    a point using recursion formulae.
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

    /**
        @brief Increase the maximum order for which recursion coefficients are cached.

        @param max_order new maximum order
    */
    void expand(std::size_t max_order);

    /**
        @brief Generate spherical harmonics at coordinates `lon`, `colat`.

        @tparam SHType type of spherical harmonic buffer

        @param lon longitude coordinate
        @param colat colatitude coordinate
        @param expansion buffer for spherical harmonic values
    */
    template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
    void generate(double lon, double colat, SHSpan<double, indexing_mode, sh_norm, sh_phase> expansion)
    {
        expand(expansion.order());

        const double z = std::cos(colat);
        AssociatedLegendreSpan<double, sh_norm, sh_phase> ass_leg(m_ass_leg_poly, expansion.order());
        m_recursion.generate_real(z, ass_leg);

        for (std::size_t m = 0; m < expansion.order(); ++m)
        {
            const double angle = double(m)*lon;
            m_cossin[m] = {std::cos(angle), std::sin(angle)};
        }

        for (auto l : ass_leg.indices())
        {
            const auto expansion_l = expansion[l];
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

    template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
    void generate(double lon, double colat, SHExpansion<double, indexing_mode, sh_norm, sh_phase>& expansion)
    {
        using ExpansionType = SHExpansion<double, indexing_mode, sh_norm, sh_phase>;
        generate<indexing_mode, sh_norm, sh_phase>(
                lon, colat, (typename ExpansionType::view)(expansion));
    }

    template <IndexingMode indexing_mode, SHNorm sh_norm, SHPhase sh_phase>
    [[nodiscard]] SHExpansion<double, indexing_mode, sh_norm, sh_phase>
    generate(double lon, double colat, std::size_t order)
    {
        using ExpansionType = SHExpansion<double, indexing_mode, sh_norm, sh_phase>;
        ExpansionType expansion{order};
        generate<indexing_mode, sh_norm, sh_phase>(
                lon, colat, (typename ExpansionType::view)(expansion));
        return expansion;
    }

private:
    AssociatedLegendreRecursion m_recursion;
    std::vector<double> m_ass_leg_poly;
    std::vector<std::array<double, 2>> m_cossin;
};

} // namespace zest::st

