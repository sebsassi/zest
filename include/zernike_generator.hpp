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

#include <array>
#include <cstddef>

#include "zernike_conventions.hpp"
#include "associated_legendre_recursion.hpp"
#include "radial_zernike_recursion.hpp"
#include "sh_conventions.hpp"
#include "zernike_expansion.hpp"

namespace zest::zt
{

/**
     @brief Generator of real Zernike functions.

    This class enables generation of collections of real Zernike functions
    evaluated at a point using recursion formulae.
*/
class ZernikeGenerator
{
public:
    ZernikeGenerator() = default;
    explicit ZernikeGenerator(std::size_t max_order);

    /**
        @brief Maximum order with cached recursion coefficients.
    */
    [[nodiscard]] std::size_t max_order() const noexcept
    {
        return m_ass_leg_recursion.max_order();
    }

    /**
        @brief Increase the maximum order for which recursion coefficients are
        cached.

        @param max_order New maximum order.
    */
    void expand(std::size_t max_order);

    /**
        @brief Generate Zernike functions at coordinates `lon`, `colat`, `r`.

        @tparam ZernikeType type of Zernike function buffer

        @param lon Longitude coordinate.
        @param colat Colatitude coordinate.
        @param r Radial coordinate.
        @param expansion Buffer for Zernike function values.
    */
    template <
        Indexing indexing, ZernikeNorm zernike_norm,
        st::sh_convention Convention
    >
    void generate(
        double lon, double colat, double r,
        ZernikeSpan<double, indexing, zernike_norm, Convention>& expansion)
    {
        expand(expansion.order());

        const double z = std::cos(colat);
        auto ass_leg = st::AssociatedLegendreSpan<double, Convention>(
                m_ass_leg_poly, expansion.order());
        m_ass_leg_recursion.generate_real(z, ass_leg);

        auto radial_zernike = RadialZernikeSpan<double, zernike_norm>(
                m_radial_zernike, expansion.order());
        m_zernike_recursion.generate(r, radial_zernike);

        for (std::size_t m = 0; m < expansion.order(); ++m)
        {
            const double angle = double(m)*lon;
            m_cossin[m] = {std::cos(angle), std::sin(angle)};
        }

        for (auto n : radial_zernike.indices())
        {
            auto radial_zernike_n = radial_zernike[n];
            auto znlm_n = expansion[n];
            for (auto l : radial_zernike_n.indices())
            {
                const double radial_zernike_nl = radial_zernike_n[l];
                auto ass_leg_l = ass_leg[l];
                auto znlm_nl = znlm_n[l];
                if constexpr (indexing == Indexing::symmetric)
                    znlm_nl[0] = ass_leg_l[0];
                else if constexpr (indexing == Indexing::zero_based)
                {
                    znlm_nl[0, 0] = radial_zernike_nl*ass_leg_l[0];
                    znlm_nl[0, 1] = 0.0;
                }

                for (auto m : ass_leg_l.indices())
                {
                    const double ass_leg_lm = ass_leg_l[m];
                    const double prefactor = radial_zernike_nl*ass_leg_lm;
                    if constexpr (indexing == Indexing::symmetric)
                    {
                        znlm_nl[m] = prefactor*m_cossin[m][0];
                        znlm_n[-m] = prefactor*m_cossin[m][1];
                    }
                    else if constexpr (
                        indexing == Indexing::zero_based)
                    {
                        znlm_nl[m, 0] = prefactor*m_cossin[m][0];
                        znlm_nl[m, 1] = prefactor*m_cossin[m][1];
                    }
                }
            }
        }
    }

    template <
        Indexing indexing, ZernikeNorm zernike_norm,
        st::sh_convention Convention
    >
    void generate(
        double lon, double colat, double r,
        ZernikeExpansion<double, indexing, zernike_norm, Convention>& expansion)
    {
        using ExpansionType = ZernikeExpansion<double, indexing, zernike_norm, Convention>;
        generate<indexing, zernike_norm, Convention>(
                lon, colat, r, (typename ExpansionType::view)(expansion));
    }

    template <
        Indexing indexing, ZernikeNorm zernike_norm,
        st::sh_convention Convention
    >
    [[nodiscard]] ZernikeExpansion<double, indexing, zernike_norm, Convention>
    generate(double lon, double colat, double r, std::size_t order)
    {
        using ExpansionType = ZernikeExpansion<double, indexing, zernike_norm, Convention>;
        ExpansionType expansion{order};
        generate<indexing, zernike_norm, Convention>(
                lon, colat, r, (typename ExpansionType::view)(expansion));
        return expansion;
    }

private:
    st::AssociatedLegendreRecursion m_ass_leg_recursion;
    RadialZernikeRecursion m_zernike_recursion;
    std::vector<double> m_radial_zernike;
    std::vector<double> m_ass_leg_poly;
    std::vector<std::array<double, 2>> m_cossin;
};

} // namespace zest::zt

