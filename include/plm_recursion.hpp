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

#include <cstddef>
#include <cassert>
#include <concepts>
#include <span>
#include <vector>
#include <numbers>
#include <type_traits>

#include "sh_conventions.hpp"
#include "layout.hpp"
#include "real_sh_expansion.hpp"

namespace zest
{
namespace st
{

using PlmLayout = TriangleLayout<IndexingMode::nonnegative>;

/**
    @brief Non-owfning view of associated Legendre polynomials.
*/
template <typename T, SHNorm sh_norm_param, SHPhase sh_phase_param>
    requires std::same_as<std::remove_const_t<T>, double>
using PlmSpan = SHLMSpan<T, PlmLayout, sh_norm_param, sh_phase_param>;

/**
    @brief Non-owfning view of vectors of associated Legendre polynomials.
*/
template <typename T, SHNorm sh_norm_param, SHPhase sh_phase_param>
    requires std::same_as<std::remove_const_t<T>, double>
using PlmVecSpan = SHLMVecSpan<T, PlmLayout, sh_norm_param, sh_phase_param>;

/**
    @brief Recursion of associated Legendre polynomials.

    @note The recursion described in (Holmes and Featherstone 2002, J. Geodesy, 76, 279-299).
*/
class PlmRecursion
{
public:
    PlmRecursion() = default;

    /*
        @brief Precompute cached recursion coefficients up to given order.

        @param max_order maximum order of coefficients
    */
    explicit PlmRecursion(std::size_t max_order);

    [[nodiscard]] std::size_t max_order() const noexcept { return m_max_order; }

    /**
        @brief Expand the number of cached recursion coefficients.

        @param max_order maximum order of coefficients
    */
    void expand(std::size_t max_order);

    void expand_vec(std::size_t vec_size);

    /**
        @brief Evaluate recursion of associated Legendre polynomials at a point.

        @param z point at which the polynomials are evaluated
        @param plm output buffer for the evaluated polynomials
    */
    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    void plm_real(double z, PlmSpan<double, sh_norm_param, sh_phase_param> plm)
    {
        return plm_impl(z, std::numbers::sqrt2, plm);
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at multiple points.

        @param z points at which the polynomials are evaluated
        @param plm output buffer for the evaluated polynomials
    */
    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    void plm_real(
        std::span<const double> z, PlmVecSpan<double, sh_norm_param, sh_phase_param> plm)
    {
        return plm_impl(z, std::numbers::sqrt2, plm);
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at a point.

        @param z point at which the polynomials are evaluated
        @param plm utput buffer for the evaluated polynomials
    */
    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    void plm_complex(double z, PlmSpan<double, sh_norm_param, sh_phase_param> plm)
    {
        return plm_impl(z, 1.0, plm);
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at multiple points.

        @param plm utput buffer for the evaluated polynomials
        @param z points at which the polynomials are evaluated
    */
    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    void plm_complex(
        std::span<const double> z, PlmVecSpan<double, sh_norm_param, sh_phase_param> plm)
    {
        return plm_impl(z, 1.0, plm);
    }

private:
    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    void plm_impl(
        double z, double complex_norm, PlmSpan<double, sh_norm_param, sh_phase_param> plm)
    {
        using PlmSpan_ = PlmSpan<double, sh_norm_param, sh_phase_param>;
        constexpr double inv_sqrt_4pi = 0.5*std::numbers::inv_sqrtpi;

        const std::size_t order = plm.order();
        if (order == 0) return;

        assert(std::fabs(z) <= 1.0);

        expand(order);
        
        const double u = std::sqrt((1.0 - z)*(1.0 + z));

        if constexpr (sh_norm_param == SHNorm::geo)
            plm(0, 0) = 1.0;
        else if constexpr (sh_norm_param == SHNorm::qm)
            plm(0, 0) = inv_sqrt_4pi;

        if (order == 1) return;

        if constexpr (sh_norm_param == SHNorm::geo)
            plm(1, 0) = m_sqrl[3]*z;
        else if constexpr (sh_norm_param == SHNorm::qm)
            plm(1, 0) = m_sqrl[3]*z*inv_sqrt_4pi;

        std::span<double> plm_flat = plm.flatten();

        // Calculate P(l,0)
        for (std::size_t l = 2; l < order; ++l)
        {
            const std::size_t ind = PlmSpan_::Layout::idx(l,0);
            plm_flat[ind] = m_alm[ind]*z*plm_flat[ind - l] - m_blm[ind]*plm_flat[ind - 2*l + 1];
        }

        constexpr double underflow_compensation = 1.0e-280;

        double pmm;
        if constexpr (sh_norm_param == SHNorm::geo)
            pmm = underflow_compensation*complex_norm;
        else if constexpr (sh_norm_param == SHNorm::qm)
            pmm = underflow_compensation*complex_norm*inv_sqrt_4pi;

        // This number is repeatedly multiplied by u < 1. To avoid underflow
        // at small values of u, we make it large. The rescaling is countered
        // by the presence of `underflow_compensation` in `pmm`.
        double u_scaled = 1.0/underflow_compensation;

        for (std::size_t m = 1; m < order - 1; ++m)
        {
            u_scaled *= u;

            // `P(m,m) = u*sqrt((2m + 1)/(2m))*P(m - 1,m - 1)`
            // NOTE: multiplication by `u` happens later
            pmm *= double(sh_phase_param)*m_sqrl[2*m + 1]/m_sqrl[2*m];
            plm(m, m) = pmm;

            // `P(m+1,m) = z*sqrt(2m + 3)*P(m,m)`
            plm(m + 1, m) = z*m_sqrl[2*m + 3]*pmm;

            for (std::size_t l = m + 2; l < order; ++l)
            {
                // P(l,m) = z*a(l,m)*P(l - 1,m) - b(l,m)*P(l - 2,m)
                const std::size_t ind = PlmSpan_::Layout::idx(l, m);
                plm_flat[ind] = z*m_alm[ind]*plm_flat[ind - l] - m_blm[ind]*plm_flat[ind - 2*l + 1];
                
                // Multiplication by `u` for `m <= l <= lmax - 2`
                plm_flat[ind - 2*l + 1] *= u_scaled;
            }

            // Multiplication by `u` for `l = lmax`
            plm(order - 1, m) *= u_scaled;

            // Multiplication by `u` for `l = lmax - 1`
            plm(order - 2, m) *= u_scaled;
        }

        u_scaled *= u;

        // P(lmax,lmax)
        plm(order - 1, order - 1)
                = double(sh_phase_param)*pmm*u_scaled*m_sqrl[2*order - 1]
                /m_sqrl[2*order - 2];
    }
    

    template <SHNorm sh_norm_param, SHPhase sh_phase_param>
    void plm_impl(
        std::span<const double> z, double complex_norm,
        PlmVecSpan<double, sh_norm_param, sh_phase_param> plm)
    {
        using PlmVecSpan_ = PlmVecSpan<double, sh_norm_param, sh_phase_param>;
        constexpr double inv_sqrt_4pi = 0.5*std::numbers::inv_sqrtpi;

        const std::size_t order = plm.order();
        if (order == 0) return;

        assert(z.size() == plm.vec_size());

        for ([[maybe_unused]] auto zi : z)
            assert(std::fabs(zi) <= 1.0);

        expand(order);
        expand_vec(z.size());
        
        for (std::size_t i = 0; i < z.size(); ++i)
            m_u[i] = std::sqrt((1.0 - z[i])*(1.0 + z[i]));

        auto plm_00 = plm(0, 0);
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if constexpr (sh_norm_param == SHNorm::geo)
                plm_00[i] = 1.0;
            else if constexpr (sh_norm_param == SHNorm::qm)
                plm_00[i] = inv_sqrt_4pi;
        }

        if (order == 1) return;

        auto plm_10 = plm(1, 0);
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if constexpr (sh_norm_param == SHNorm::geo)
                plm_10[i] = z[i]*m_sqrl[3];
            else if constexpr (sh_norm_param == SHNorm::qm)
                plm_10[i] = z[i]*(m_sqrl[3]*inv_sqrt_4pi);
        }

        auto plm_linear = plm.linear_view();
        // Calculate P(l,0) for l >= 2
        for (auto l : plm.indices(2))
        {
            const std::size_t ind = PlmVecSpan_::Layout::idx(l, 0);
            auto plm_l0 = plm_linear[ind];
            auto plm_lm10 = plm_linear[ind - l];
            auto plm_lm20 = plm_linear[ind - 2*l + 1];
            // P(l, 0) = z*a(l,m)*P(l - 1, 0) - b(l,m)*P(l - 2, 0)
            for (std::size_t i = 0; i < z.size(); ++i)
            {
                plm_l0[i] = m_alm[ind]*z[i]*plm_lm10[i]
                    - m_blm[ind]*plm_lm20[i];
            }
        }

        constexpr double underflow_compensation = 1.0e-280;

        double pmm;
        if constexpr (sh_norm_param == SHNorm::geo)
            pmm = underflow_compensation*complex_norm;
        else if constexpr (sh_norm_param == SHNorm::qm)
            pmm = underflow_compensation*complex_norm*inv_sqrt_4pi;

        // This number is repeatedly multiplied by u < 1. To avoid underflow
        // at small values of u, we make it large. The rescaling is countered
        // by the presence of `underflow_compensation` in `pmm`.
        for (std::size_t i = 0; i < z.size(); ++i)
            m_u_scaled[i] = 1.0/underflow_compensation;

        for (std::size_t m = 1; m < order - 1; ++m)
        {
            for (std::size_t i = 0; i < z.size(); ++i)
                m_u_scaled[i] *= m_u[i];

            // `P(m, m) = u*sqrt((2m + 1)/(2m))*P(m - 1, m - 1)`
            // NOTE: multiplication by `u` happens later
            pmm *= double(sh_phase_param)*m_sqrl[2*m + 1]/m_sqrl[2*m];
            auto plm_mm = plm(m, m);
            for (std::size_t i = 0; i < z.size(); ++i)
                plm_mm[i] = pmm;

            // `P(m+1, m) = z*sqrt(2m + 3)*P(m, m)`
            auto plm_mp1m = plm(m + 1, m);
            for (std::size_t i = 0; i < z.size(); ++i)
                plm_mp1m[i] = z[i]*(m_sqrl[2*m + 3]*pmm);

            for (std::size_t l = m + 2; l < order; ++l)
            {
                const std::size_t ind = PlmVecSpan_::Layout::idx(l, m);
                auto plm_lm = plm_linear[ind];
                auto plm_lm1m = plm_linear[ind - l];
                auto plm_lm2m = plm_linear[ind - 2*l + 1];
                // P(l, m) = z*a(l, m)*P(l - 1, m) - b(l, m)*P(l - 2, m)
                for (std::size_t i = 0; i < z.size(); ++i)
                    plm_lm[i] = z[i]*m_alm[ind]*plm_lm1m[i]
                        - m_blm[ind]*plm_lm2m[i];
                
                // Multiplication by `u` for `m <= l <= lmax - 2`
                for (std::size_t i = 0; i < z.size(); ++i)
                    plm_lm2m[i] *= m_u_scaled[i];
            }

            // Multiplication by `u` for `l = lmax`
            auto plm_om1m = plm(order - 1, m);
            for (std::size_t i = 0; i < z.size(); ++i)
                plm_om1m[i] *= m_u_scaled[i];

            // Multiplication by `u` for `l = lmax - 1`
            auto plm_om2m = plm(order - 2, m);
            for (std::size_t i = 0; i < z.size(); ++i)
                plm_om2m[i] *= m_u_scaled[i];
        }

        for (std::size_t i = 0; i < z.size(); ++i)
            m_u_scaled[i] *= m_u[i];

        // P(lmax,lmax)
        auto plm_om1om1 = plm(order - 1, order - 1);
        for (std::size_t i = 0; i < z.size(); ++i)
            plm_om1om1[i]
                = m_u_scaled[i]*(double(sh_phase_param)*pmm*m_sqrl[2*order - 1]
                /m_sqrl[2*order - 2]);
    }

    std::vector<double> m_sqrl{};
    std::vector<double> m_alm{};
    std::vector<double> m_blm{};
    std::vector<double> m_u_scaled{};
    std::vector<double> m_u{};
    std::size_t m_max_order{};
};

} // namespace st
} // namespace zest