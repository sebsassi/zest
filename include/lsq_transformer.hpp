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

#include <span>
#include <vector>

#include "linearfit.hpp"
#include "real_sh_expansion.hpp"
#include "real_ylm.hpp"

namespace zest
{
namespace st
{

/**
    @brief Least-squares real spherical harmonic expansion fit on arbitrary real valued data on the sphere.
*/
class LSQTransformer
{
public:
    LSQTransformer() = default;
    explicit LSQTransformer(std::size_t order);

    [[nodiscard]] std::size_t order() const noexcept
    {
        return m_ylm_gen.max_order();
    }

    [[nodiscard]] const Matrix<double>& sh_values() const noexcept
    {
        return m_sh_values;
    }

    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    void transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon, RealSHExpansionSpan<std::array<double, 2>, sh_norm_param, sh_phase_param> expansion)
    {
        using Expansion = RealSHExpansionSpan<std::array<double, 2>, sh_norm_param, sh_phase_param>;

        m_ylm_gen.expand(expansion.order());

        m_sh_values.resize(
                data.size(), DualTriangleLayout::size(expansion.order()));

        for (size_t i = 0; i < data.size(); ++i)
        {
            RealYlmSpan<SequentialRealYlmPacking, sh_norm_param, sh_phase_param> ylm(m_sh_values.row(i), expansion.order());
            m_ylm_gen.generate<SequentialRealYlmPacking, sh_norm_param, sh_phase_param>(lon[i], lat[i], ylm);
        }

        m_coeffs.resize(m_sh_values.ncols());
        m_fitter.fit_parameters(m_sh_values, m_coeffs, data);

        std::span coeffs = expansion.flatten();
        for (std::size_t l = 0; l < expansion.order(); ++l)
        {
            coeffs[Expansion::Layout::idx(l, 0)] = {
                m_coeffs[DualTriangleLayout::idx(int(l), 0)],
                0.0
            };
            for (std::size_t m = 1; m <= l; ++m)
            {
                coeffs[Expansion::Layout::idx(l,m)] = {
                        m_coeffs[DualTriangleLayout::idx(int(l),int(m))],
                        m_coeffs[DualTriangleLayout::idx(int(l),-int(m))]
                    };
            }
        }
    }
    
    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    RealSHExpansion<sh_norm_param, sh_phase_param> transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon, std::size_t order)
    {
        RealSHExpansion<sh_norm_param, sh_phase_param> expansion(order);
        transform<sh_norm_param, sh_phase_param>(data, lat, lon, expansion);
        return expansion;
    }

private:
    RealYlmGenerator m_ylm_gen;
    Matrix<double> m_sh_values;
    std::vector<double> m_coeffs;
    detail::LinearMultifit m_fitter;
};

} // namespace st
} // namespace zest