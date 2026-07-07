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
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "sequence.hpp"
#include "sh_conventions.hpp"
#include "sh_expansion.hpp"
#include "triangle_spans.hpp"

namespace zest::st
{

/**
    @brief Recursion of associated Legendre polynomials.

    @note The recursion described in (Holmes and Featherstone 2002, J. Geodesy,
    76, 279-299).
*/
class AssociatedLegendreRecursion
{
public:
    AssociatedLegendreRecursion() = default;

    /**
        @brief Precompute cached recursion coefficients up to given order.

        @param max_order maximum order of coefficients
    */
    explicit AssociatedLegendreRecursion(std::size_t max_order);

    [[nodiscard]] std::size_t max_order() const noexcept { return m_max_order; }

    /**
        @brief Expand the number of cached recursion coefficients.

        @param max_order maximum order of coefficients
    */
    void expand(std::size_t max_order);

    /**
        @brief Expand the size of the vector segment the recursion can operate on.

        @param vec_size maximum segment size
    */
    void expand_vec(std::size_t vec_size);

    /**
        @brief Evaluate recursion of associated Legendre polynomials at a point.

        @param z point at which the polynomials are evaluated
        @param ass_leg output buffer for the evaluated polynomials
    */
    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_real(double z, AssociatedLegendreSpan<double, sh_norm, sh_phase> ass_leg)
    {
        return generate_impl(z, std::numbers::sqrt2, ass_leg);
    }

    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_real(
        double z, AssociatedLegendreExpansion<double, sh_norm, sh_phase>& ass_leg)
    {
        return generate_impl(
            z, std::numbers::sqrt2,
            AssociatedLegendreSpan<double, sh_norm, sh_phase>(ass_leg));
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at multiple points.

        @param z points at which the polynomials are evaluated
        @param ass_leg output buffer for the evaluated polynomials
    */
    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_real(
        std::span<const double> z,
        AssociatedLegendreSpan<double, sh_norm, sh_phase, std::dynamic_extent> ass_leg)
    {
        return generate_impl(z, std::numbers::sqrt2, ass_leg);
    }
    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_real(
        std::span<const double> z,
        AssociatedLegendreExpansion<double, sh_norm, sh_phase, std::dynamic_extent>& ass_leg)
    {
        return generate_impl(
            z, std::numbers::sqrt2,
            AssociatedLegendreSpan<double, sh_norm, sh_phase, std::dynamic_extent>(ass_leg));
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at a point.

        @param z point at which the polynomials are evaluated
        @param ass_leg utput buffer for the evaluated polynomials
    */
    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_complex(double z, AssociatedLegendreSpan<double, sh_norm, sh_phase> ass_leg)
    {
        return generate_impl(z, 1.0, ass_leg);
    }
    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_complex(
        double z, AssociatedLegendreExpansion<double, sh_norm, sh_phase>& ass_leg)
    {
        return generate_impl(
                z, 1.0, AssociatedLegendreSpan<double, sh_norm, sh_phase>(ass_leg));
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at multiple points.

        @param ass_leg utput buffer for the evaluated polynomials
        @param z points at which the polynomials are evaluated
    */
    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_complex(
        std::span<const double> z,
        AssociatedLegendreSpan<double, sh_norm, sh_phase, std::dynamic_extent> ass_leg)
    {
        return generate_impl(z, 1.0, ass_leg);
    }

    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_complex(
        std::span<const double> z,
        AssociatedLegendreExpansion<double, sh_norm, sh_phase, std::dynamic_extent>& ass_leg)
    {
        return generate_impl(
            z, 1.0, 
            AssociatedLegendreSpan<double, sh_norm, sh_phase, std::dynamic_extent>(ass_leg));
    }

private:
    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_impl(
        double z, double complex_norm,
        AssociatedLegendreSpan<double, sh_norm, sh_phase> ass_leg)
    {
        constexpr double inv_sqrt_4pi = 0.5*std::numbers::inv_sqrtpi;

        const std::size_t order = ass_leg.order();
        if (order == 0) return;

        assert(std::fabs(z) <= 1.0);

        expand(order);

        const double u = std::sqrt((1.0 - z)*(1.0 + z));

        if constexpr (sh_norm == SHNorm::geo)
            ass_leg[0, 0] = 1.0;
        else if constexpr (sh_norm == SHNorm::qm)
            ass_leg[0, 0] = inv_sqrt_4pi;

        if (order == 1) return;

        if constexpr (sh_norm == SHNorm::geo)
            ass_leg[1, 0] = m_sqrl[3]*z;
        else if constexpr (sh_norm == SHNorm::qm)
            ass_leg[1, 0] = m_sqrl[3]*z*inv_sqrt_4pi;

        std::span<double> ass_leg_flat = ass_leg.flatten();
        std::span<const double> alm_flat = m_alm.flatten();
        std::span<const double> blm_flat = m_blm.flatten();

        // Calculate P(l,0)
        for (std::size_t l = 2; l < order; ++l)
        {
            const std::size_t ind = ass_leg.shape()(l);
            ass_leg_flat[ind]
                = alm_flat[ind]*z*ass_leg_flat[ind - l]
                    - blm_flat[ind]*ass_leg_flat[ind - 2*l + 1];
        }

        constexpr double underflow_compensation = 1.0e-280;

        double pmm;
        if constexpr (sh_norm == SHNorm::geo)
            pmm = underflow_compensation*complex_norm;
        else if constexpr (sh_norm == SHNorm::qm)
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
            pmm *= double(sh_phase)*m_sqrl[2*m + 1]/m_sqrl[2*m];
            ass_leg[m, m] = pmm;

            // `P(m+1,m) = z*sqrt(2m + 3)*P(m,m)`
            ass_leg[m + 1, m] = z*m_sqrl[2*m + 3]*pmm;

            for (std::size_t l = m + 2; l < order; ++l)
            {
                // P(l,m) = z*a(l,m)*P(l - 1,m) - b(l,m)*P(l - 2,m)
                const std::size_t ind = ass_leg.shape()(l, m);
                ass_leg_flat[ind]
                    = z*alm_flat[ind]*ass_leg_flat[ind - l]
                        - blm_flat[ind]*ass_leg_flat[ind - 2*l + 1];

                // Multiplication by `u` for `m <= l <= lmax - 2`
                ass_leg_flat[ind - 2*l + 1] *= u_scaled;
            }

            // Multiplication by `u` for `l = lmax`
            ass_leg(order - 1, m) *= u_scaled;

            // Multiplication by `u` for `l = lmax - 1`
            ass_leg(order - 2, m) *= u_scaled;
        }

        u_scaled *= u;

        // P(lmax,lmax)
        ass_leg[order - 1, order - 1]
                = double(sh_phase)*pmm*u_scaled*m_sqrl[2*order - 1]/m_sqrl[2*order - 2];
    }

    template <SHNorm sh_norm, SHPhase sh_phase>
    void generate_impl(
        std::span<const double> z, double complex_norm,
        AssociatedLegendreSpan<double, sh_norm, sh_phase, std::dynamic_extent> ass_leg)
    {
        constexpr double inv_sqrt_4pi = 0.5*std::numbers::inv_sqrtpi;

        const std::size_t order = ass_leg.order();
        if (order == 0) return;

        assert((z.size() == ass_leg[0, 0].size()));

        for (double element : z)
            assert(std::fabs(element) <= 1.0);

        expand(order);
        expand_vec(z.size());

        for (std::size_t i = 0; i < z.size(); ++i)
            m_u[i] = std::sqrt((1.0 - z[i])*(1.0 + z[i]));

        auto ass_leg_00 = ass_leg[0, 0];
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if constexpr (sh_norm == SHNorm::geo)
                ass_leg_00[i] = 1.0;
            else if constexpr (sh_norm == SHNorm::qm)
                ass_leg_00[i] = inv_sqrt_4pi;
        }

        if (order == 1) return;

        auto ass_leg_10 = ass_leg[1, 0];
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if constexpr (sh_norm == SHNorm::geo)
                ass_leg_10[i] = z[i]*m_sqrl[3];
            else if constexpr (sh_norm == SHNorm::qm)
                ass_leg_10[i] = z[i]*(m_sqrl[3]*inv_sqrt_4pi);
        }

        // Calculate P(l,0) for l >= 2
        for (auto l : ass_leg.indices(2))
        {
            auto ass_leg_l0 = ass_leg[l, 0];
            auto ass_leg_lm10 = ass_leg[l - 1, 0];
            auto ass_leg_lm20 = ass_leg[l - 2, 0];
            const double alm_l0 = m_alm[l, 0];
            const double blm_l0 = m_blm[l, 0];
            // P(l, 0) = z*a(l,m)*P(l - 1, 0) - b(l,m)*P(l - 2, 0)
            for (std::size_t i = 0; i < z.size(); ++i)
            {
                ass_leg_l0[i] = alm_l0*z[i]*ass_leg_lm10[i]
                    - blm_l0*ass_leg_lm20[i];
            }
        }

        constexpr double underflow_compensation = 1.0e-280;

        double pmm;
        if constexpr (sh_norm == SHNorm::geo)
            pmm = underflow_compensation*complex_norm;
        else if constexpr (sh_norm == SHNorm::qm)
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
            pmm *= double(sh_phase)*m_sqrl[2*m + 1]/m_sqrl[2*m];
            auto ass_leg_mm = ass_leg(m, m);
            for (std::size_t i = 0; i < z.size(); ++i)
                ass_leg_mm[i] = pmm;

            // `P(m+1, m) = z*sqrt(2m + 3)*P(m, m)`
            auto ass_leg_mp1m = ass_leg(m + 1, m);
            for (std::size_t i = 0; i < z.size(); ++i)
                ass_leg_mp1m[i] = z[i]*(m_sqrl[2*m + 3]*pmm);

            for (std::size_t l = m + 2; l < order; ++l)
            {
                auto ass_leg_lm = ass_leg[l, m];
                auto ass_leg_lm1m = ass_leg[l - 1, m];
                auto ass_leg_lm2m = ass_leg[l - 2, m];
                const double alm_l0 = m_alm[l, m];
                const double blm_l0 = m_blm[l, m];
                // P(l, m) = z*a(l, m)*P(l - 1, m) - b(l, m)*P(l - 2, m)
                for (std::size_t i = 0; i < z.size(); ++i)
                    ass_leg_lm[i] = z[i]*alm_l0*ass_leg_lm1m[i]
                        - blm_l0*ass_leg_lm2m[i];

                // Multiplication by `u` for `m <= l <= lmax - 2`
                for (std::size_t i = 0; i < z.size(); ++i)
                    ass_leg_lm2m[i] *= m_u_scaled[i];
            }

            // Multiplication by `u` for `l = lmax`
            auto ass_leg_om1m = ass_leg(order - 1, m);
            for (std::size_t i = 0; i < z.size(); ++i)
                ass_leg_om1m[i] *= m_u_scaled[i];

            // Multiplication by `u` for `l = lmax - 1`
            auto ass_leg_om2m = ass_leg(order - 2, m);
            for (std::size_t i = 0; i < z.size(); ++i)
                ass_leg_om2m[i] *= m_u_scaled[i];
        }

        for (std::size_t i = 0; i < z.size(); ++i)
            m_u_scaled[i] *= m_u[i];

        // P(lmax,lmax)
        auto ass_leg_om1om1 = ass_leg(order - 1, order - 1);
        for (std::size_t i = 0; i < z.size(); ++i)
            ass_leg_om1om1[i]
                = m_u_scaled[i]*(double(sh_phase)*pmm*m_sqrl[2*order - 1]
                /m_sqrl[2*order - 2]);
    }

    std::vector<double> m_sqrl;
    TriangleArray<double, IndexingMode::zero_based> m_alm;
    TriangleArray<double, IndexingMode::zero_based> m_blm;
    std::vector<double> m_u_scaled;
    std::vector<double> m_u;
    std::size_t m_max_order{};
};

} // namespace zest::st

