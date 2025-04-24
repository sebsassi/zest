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

#include "md_span.hpp"
#include "md_array.hpp"
#include "linearfit.hpp"
#include "real_sh_expansion.hpp"
#include "sh_generator.hpp"
#include "zernike_generator.hpp"

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
        return m_sh_gen.max_order();
    }

    [[nodiscard]] MDSpan<const double, 2> sh_values() const noexcept
    {
        return m_sh_values;
    }

    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    void transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon,
        RealSHSpan<std::array<double, 2>, sh_norm_param, sh_phase_param> expansion)
    {
        using FitExpansion = RealSHSpan<double, sh_norm_param, sh_phase_param>;

        m_sh_gen.expand(expansion.order());

        m_sh_values.reshape(
                {data.size(), FitExpansion::Layout::size(expansion.order())});

        for (size_t i = 0; i < data.size(); ++i)
        {
            FitExpansion ylm(m_sh_values[i], expansion.order());
            m_sh_gen.generate(lon[i], lat[i], ylm);
        }

        m_coeffs.resize(m_sh_values.extent(1));
        m_fitter(m_sh_values, m_coeffs, data);

        FitExpansion coeffs(m_coeffs.data(), expansion.order());
        for (auto l : expansion.indices())
        {
            auto expansion_l = expansion[l];
            auto coeffs_l = coeffs[int(l)];
            expansion_l[0] = {coeffs_l[0], 0.0};
            for (auto m : expansion_l.indices(1))
                expansion_l[m] = {coeffs_l[int(m)], coeffs_l[-int(m)]};
        }
    }
    
    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    [[nodiscard]] auto transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon,
        std::size_t order)
    {
        RealSHExpansion<sh_norm_param, sh_phase_param> expansion(order);
        transform<sh_norm_param, sh_phase_param>(data, lat, lon, expansion);
        return expansion;
    }

private:
    RealSHGenerator m_sh_gen;
    MDArray<double, 2> m_sh_values;
    std::vector<double> m_coeffs;
    zest::detail::LinearMultifit m_fitter;
};

} // namespace st

namespace zt
{

/**
    @brief Least-squares Zernike expansion fit on arbitrary real valued data on the unit ball.
*/
class LSQTransformer
{
public:
    LSQTransformer() = default;
    explicit LSQTransformer(std::size_t order);

    [[nodiscard]] std::size_t order() const noexcept
    {
        return m_zernike_gen.max_order();
    }

    [[nodiscard]] MDSpan<const double, 2> zernike_values() const noexcept
    {
        return m_zernike_values;
    }

    template <
        ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param, st::SHPhase sh_phase_param>
    void transform(
        std::span<const double> data, std::span<const double> r, std::span<const double> lon,
        std::span<const double> colat,
        RealZernikeSpan<std::array<double, 2>, zernike_norm_param, sh_norm_param, sh_phase_param> expansion)
    {
        using FitExpansion = RealZernikeSpan<
                double, zernike_norm_param, sh_norm_param, sh_phase_param>;

        m_zernike_gen.expand(expansion.order());

        m_zernike_values.reshape(
                {data.size(), FitExpansion::Layout::size(expansion.order())});

        for (size_t i = 0; i < data.size(); ++i)
        {
            FitExpansion znlm(m_zernike_values[i], expansion.order());
            m_zernike_gen.generate(r[i], lon[i], colat[i], znlm);
        }

        m_coeffs.resize(m_zernike_values.extent(1));
        m_fitter(m_zernike_values, m_coeffs, data);

        FitExpansion coeffs(m_coeffs.data(), expansion.order());
        for (auto n : expansion.indices())
        {
            auto expansion_n = expansion[n];
            auto coeffs_n = coeffs[int(n)];
            for (auto l : expansion_n.indices())
            {
                auto expansion_nl = expansion_n[l];
                auto coeffs_nl = coeffs[int(l)];
                expansion_nl[0] = {coeffs_nl[0], 0.0};
                for (std::size_t m = 1; m <= l; ++m)
                    expansion_nl[m] = {coeffs_nl[int(m)], coeffs_nl[-int(m)]};
            }
        }
    }
    
    template <
        ZernikeNorm zernike_norm_param, st::SHNorm sh_norm_param,
        st::SHPhase sh_phase_param>
    [[nodiscard]] auto transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon,
        std::size_t order)
    {
        RealZernikeExpansion<zernike_norm_param, sh_norm_param, sh_phase_param> 
        expansion(order);
        transform<sh_norm_param, sh_phase_param>(data, lat, lon, expansion);
        return expansion;
    }

private:
    ZernikeGenerator m_zernike_gen;
    MDArray<double, 2> m_zernike_values;
    std::vector<double> m_coeffs;
    detail::LinearMultifit m_fitter;
};

} // namespace zt
} // namespace zest
