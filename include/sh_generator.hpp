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
#pragma once

#include <vector>
#include <span>
#include <concepts>

#include "plm_recursion.hpp"
#include "layout.hpp"
#include "packing.hpp"
#include "real_sh_expansion.hpp"

namespace zest
{
namespace st
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
        @param ylm buffer for spherical harmonic values
    */
    template <real_sh_expansion SHType>
    void generate(double lon, double colat, SHType&& ylm)
    {
        constexpr SHNorm norm = std::remove_cvref_t<SHType>::norm;
        constexpr SHPhase phase = std::remove_cvref_t<SHType>::phase;
        using SHSpan = RealSHSpan<
                typename std::remove_cvref_t<SHType>::element_type, 
                norm, phase>;
        using index_type = SHSpan::index_type;
        expand(ylm.order());

        const double z = std::sin(colat);
        auto ass_leg = PlmSpan<double, norm, phase>(
                m_ass_leg_poly, ylm.order());
        m_recursion.plm_real(z, ass_leg);

        for (std::size_t m = 0; m < ylm.order(); ++m)
        {
            const double angle = double(m)*lon;
            m_cossin[m] = {std::cos(angle), std::sin(angle)};
        }

        constexpr IndexingMode indexing_mode = SHSpan::Layout::indexing_mode;
        for (auto l : ass_leg.indices())
        {
            auto ylm_l = ylm[index_type(l)];
            auto ass_leg_l = ass_leg[l];
            if constexpr (indexing_mode == IndexingMode::negative)
                ylm_l[0] = ass_leg_l[0];
            else if constexpr (indexing_mode == IndexingMode::nonnegative)
                ylm_l[0] = {ass_leg_l[0], 0.0};
            
            for (auto m : ass_leg_l.indices(1))
            {
                const double ass_leg_lm = ass_leg_l[m];
                
                if constexpr (indexing_mode == IndexingMode::negative)
                {
                    ylm_l[int(m)] = ass_leg_lm*m_cossin[m][0];
                    ylm_l[-int(m)] = ass_leg_lm*m_cossin[m][1];
                }
                else if constexpr (indexing_mode == IndexingMode::nonnegative)
                {
                    ylm_l[m] = {
                        ass_leg_lm*m_cossin[m][0],
                        ass_leg_lm*m_cossin[m][1]
                    };
                }
            }
        }
    }

private:
    PlmRecursion m_recursion;
    std::vector<double> m_ass_leg_poly;
    std::vector<std::array<double, 2>> m_cossin;
};

} // namespace st
} // namespace zest
