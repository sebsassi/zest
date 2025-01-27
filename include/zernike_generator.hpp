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

#include <array>

#include "layout.hpp"
#include "zernike_conventions.hpp"
#include "sh_conventions.hpp"
#include "zernike_expansion.hpp"
#include "plm_recursion.hpp"
#include "radial_zernike_recursion.hpp"

namespace zest
{
namespace zt
{

template <
    ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param,
    st::SHPhase sh_phase_param>
using ZernikeSpan = RealZernikeSpan<
    std::array<double, 2>, zernike_norm_param, sh_norm_param, sh_phase_param>;

/**
     @brief Generation of real spherical harmonics based on recursion of associated Legendre polynomials.
*/
class ZernikeGenerator
{
public:
    ZernikeGenerator() = default;
    explicit ZernikeGenerator(std::size_t max_order);

    [[nodiscard]] std::size_t max_order() const noexcept
    {
        return m_plm_recursion.max_order();
    }

    void expand(std::size_t max_order);

    /**
        @brief Generate Zernike functions at radial, longitude and colatitude values `r`, `lon`, `colat`.
    */
    template <zernike_expansion ExpansionType>
    void generate(double r, double lon, double colat, ExpansionType&& znlm)
    {
        constexpr zt::ZernikeNorm zernike_norm
            = std::remove_cvref_t<ExpansionType>::zernike_norm;
        constexpr st::SHNorm sh_norm
            = std::remove_cvref_t<ExpansionType>::sh_norm;
        constexpr st::SHPhase sh_phase
            = std::remove_cvref_t<ExpansionType>::sh_phase;
        using ZernikeSpan = RealZernikeSpan<
                typename ExpansionType::element_type, 
                zernike_norm, sh_norm, sh_phase>;
        using index_type = typename ZernikeSpan::index_type;
        expand(znlm.order());

        const double z = std::cos(colat);
        auto ass_leg = st::PlmSpan<double, sh_norm, sh_phase>(
                m_ass_leg_poly, znlm.order());
        m_plm_recursion.plm_real(z, ass_leg);
        
        auto radial_zernike = RadialZernikeSpan<double, zernike_norm>(
                m_radial_zernike, znlm.order());
        m_zernike_recursion.zernike(r, radial_zernike);

        for (std::size_t m = 0; m < znlm.order(); ++m)
        {
            const double angle = double(m)*lon;
            m_cossin[m] = {std::cos(angle), std::sin(angle)};
        }

        constexpr IndexingMode indexing_mode
                = ZernikeSpan::Layout::indexing_mode;
        for (auto n : radial_zernike.indices())
        {
            auto radial_zernike_n = radial_zernike[n];
            auto znlm_n = znlm[index_type(n)];
            for (auto l : radial_zernike_n.indices())
            {
                const double radial_zernike_nl = radial_zernike_n[l];
                auto ass_leg_l = ass_leg[l];
                auto znlm_nl = znlm_n[index_type(l)];
                if constexpr (indexing_mode == IndexingMode::negative)
                    znlm_nl[0] = ass_leg_l[0];
                else if constexpr (indexing_mode == IndexingMode::nonnegative)
                    znlm_nl[0] = {radial_zernike_nl*ass_leg_l[0], 0.0};
                
                for (auto m : ass_leg_l.indices())
                {
                    const double ass_leg_lm = ass_leg_l[m];
                    const double prefactor = radial_zernike_nl*ass_leg_lm;
                    if constexpr (indexing_mode == IndexingMode::negative)
                    {
                        znlm_nl[index_type(m)] = prefactor*m_cossin[m][0];
                        znlm_n[-index_type(m)] = prefactor*m_cossin[m][1];
                    }
                    else if constexpr (
                        indexing_mode == IndexingMode::nonnegative)
                    {
                        znlm_n[index_type(m)] = {
                            prefactor*m_cossin[m][0], prefactor*m_cossin[m][1]
                        };
                    }
                }
            }
        }
    }

private:
    st::PlmRecursion m_plm_recursion{};
    RadialZernikeRecursion m_zernike_recursion{};
    std::vector<double> m_radial_zernike{};
    std::vector<double> m_ass_leg_poly{};
    std::vector<std::array<double, 2>> m_cossin{};
};

} // namespace zt
} // namespace zest