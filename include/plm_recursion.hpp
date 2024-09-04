#pragma once

#include <cstddef>
#include <cassert>
#include <concepts>
#include <span>
#include <vector>

#include "sh_conventions.hpp"
#include "triangle_layout.hpp"
#include "real_sh_expansion.hpp"

namespace zest
{
namespace st
{

/**
    @brief Non-owfning view of associated Legendre polynomials.
*/
template <typename T, SHNorm NORM, SHPhase PHASE>
    requires std::same_as<std::remove_const_t<T>, double>
using PlmSpan = SHLMSpan<T, TriangleLayout, NORM, PHASE>;

/**
    @brief Non-owfning view of vectors of associated Legendre polynomials.
*/
template <typename T, SHNorm NORM, SHPhase PHASE>
    requires std::same_as<std::remove_const_t<T>, double>
using PlmVecSpan = SHLMVecSpan<T, TriangleLayout, NORM, PHASE>;

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
    template <SHNorm NORM, SHPhase PHASE>
    void plm_real(double z, PlmSpan<double, NORM, PHASE> plm)
    {
        return plm_impl(z, std::numbers::sqrt2, plm);
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at multiple points.

        @param z points at which the polynomials are evaluated
        @param plm output buffer for the evaluated polynomials
    */
    template <SHNorm NORM, SHPhase PHASE>
    void plm_real(
        std::span<const double> z, PlmVecSpan<double, NORM, PHASE> plm)
    {
        return plm_impl(z, std::numbers::sqrt2, plm);
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at a point.

        @param z point at which the polynomials are evaluated
        @param plm utput buffer for the evaluated polynomials
    */
    template <SHNorm NORM, SHPhase PHASE>
    void plm_complex(double z, PlmSpan<double, NORM, PHASE> plm)
    {
        return plm_impl(z, 1.0, plm);
    }

    /**
        @brief Evaluate recursion of associated Legendre polynomials at multiple points.

        @param plm utput buffer for the evaluated polynomials
        @param z points at which the polynomials are evaluated
    */
    template <SHNorm NORM, SHPhase PHASE>
    void plm_complex(
        std::span<const double> z, PlmVecSpan<double, NORM, PHASE> plm)
    {
        return plm_impl(z, 1.0, plm);
    }

private:
    template <SHNorm NORM, SHPhase PHASE>
    void plm_impl(
        double z, double norm, PlmSpan<double, NORM, PHASE> plm)
    {
        constexpr double inv_sqrt_4pi = 0.5*std::numbers::inv_sqrtpi;

        const std::size_t order = plm.order();
        if (order == 0) return;

        assert(std::fabs(z) <= 1.0);

        expand(order);
        
        const double u = std::sqrt((1.0 - z)*(1.0 + z));

        if constexpr (NORM == SHNorm::GEO)
            plm(0, 0) = 1.0;
        else if constexpr (NORM == SHNorm::QM)
            plm(0, 0) = inv_sqrt_4pi;

        if (order == 1) return;

        if constexpr (NORM == SHNorm::GEO)
            plm(1, 0) = m_sqrl[3]*z;
        else if constexpr (NORM == SHNorm::QM)
            plm(1, 0) = m_sqrl[3]*z*inv_sqrt_4pi;

        std::span<double> plm_flat = plm.flatten();

        // Calculate P(l,0)
        for (std::size_t l = 2; l < order; ++l)
        {
            const std::size_t ind = TriangleLayout::idx(l,0);
            plm_flat[ind] = m_alm[ind]*z*plm_flat[ind - l] - m_blm[ind]*plm_flat[ind - 2*l + 1];
        }

        constexpr double underflow_compensation = 1.0e-280;

        double pmm;
        if constexpr (NORM == SHNorm::GEO)
            pmm = underflow_compensation*norm;
        else if constexpr (NORM == SHNorm::QM)
            pmm = underflow_compensation*norm*inv_sqrt_4pi;

        // This number is repeatedly multiplied by u < 1. To avoid underflow
        // at small values of u, we make it large. The rescaling is countered
        // by the presence of `underflow_compensation` in `pmm`.
        double u_scaled = 1.0/underflow_compensation;

        for (std::size_t m = 1; m < order - 1; ++m)
        {
            u_scaled *= u;

            // `P(m,m) = u*sqrt((2m + 1)/(2m))*P(m - 1,m - 1)`
            // NOTE: multiplication by `u` happens later
            pmm *= double(PHASE)*m_sqrl[2*m + 1]/m_sqrl[2*m];
            plm(m, m) = pmm;

            // `P(m+1,m) = z*sqrt(2m + 3)*P(m,m)`
            plm(m + 1, m) = z*m_sqrl[2*m + 3]*pmm;

            for (std::size_t l = m + 2; l < order; ++l)
            {
                // P(l,m) = z*a(l,m)*P(l - 1,m) - b(l,m)*P(l - 2,m)
                const std::size_t ind = TriangleLayout::idx(l, m);
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
                = double(PHASE)*pmm*u_scaled*m_sqrl[2*order - 1]
                /m_sqrl[2*order - 2];
    }
    

    template <SHNorm NORM, SHPhase PHASE>
    void plm_impl(
        std::span<const double> z, double norm,
        PlmVecSpan<double, NORM, PHASE> plm)
    {
        constexpr double inv_sqrt_4pi = 0.5*std::numbers::inv_sqrtpi;

        const std::size_t order = plm.order();
        if (order == 0) return;

        assert(z.size() == plm.vec_size());

        for (auto zi : z)
            assert(std::fabs(zi) <= 1.0);

        expand(order);
        expand_vec(z.size());
        
        for (std::size_t i = 0; i < z.size(); ++i)
            m_u[i] = std::sqrt((1.0 - z[i])*(1.0 + z[i]));

        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if constexpr (NORM == SHNorm::GEO)
                plm[0][i] = 1.0;
            else if constexpr (NORM == SHNorm::QM)
                plm[0][i] = inv_sqrt_4pi;
        }

        if (order == 1) return;

        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if constexpr (NORM == SHNorm::GEO)
                plm[1][i] = z[i]*m_sqrl[3];
            else if constexpr (NORM == SHNorm::QM)
                plm[1][i] = z[i]*(m_sqrl[3]*inv_sqrt_4pi);
        }
        // Calculate P(l,0)
        for (std::size_t l = 2; l < order; ++l)
        {
            const std::size_t ind = TriangleLayout::idx(l,0);
            for (std::size_t i = 0; i < z.size(); ++i)
            {
                plm[ind][i] = m_alm[ind]*z[i]*plm[ind - l][i] - m_blm[ind]*plm[ind - 2*l + 1][i];
            }
        }

        constexpr double underflow_compensation = 1.0e-280;

        double pmm;
        if constexpr (NORM == SHNorm::GEO)
            pmm = underflow_compensation*norm;
        else if constexpr (NORM == SHNorm::QM)
            pmm = underflow_compensation*norm*inv_sqrt_4pi;

        // This number is repeatedly multiplied by u < 1. To avoid underflow
        // at small values of u, we make it large. The rescaling is countered
        // by the presence of `underflow_compensation` in `pmm`.
        for (std::size_t i = 0; i < z.size(); ++i)
            m_u_scaled[i] = 1.0/underflow_compensation;

        for (std::size_t m = 1; m < order - 1; ++m)
        {
            for (std::size_t i = 0; i < z.size(); ++i)
                m_u_scaled[i] *= m_u[i];

            // `P(m,m) = u*sqrt((2m + 1)/(2m))*P(m - 1,m - 1)`
            // NOTE: multiplication by `u` happens later
            pmm *= double(PHASE)*m_sqrl[2*m + 1]/m_sqrl[2*m];
            for (std::size_t i = 0; i < z.size(); ++i)
                plm[TriangleLayout::idx(m, m)][i] = pmm;

            // `P(m+1,m) = z*sqrt(2m + 3)*P(m,m)`
            for (std::size_t i = 0; i < z.size(); ++i)
                plm[TriangleLayout::idx(m + 1, m)][i] = z[i]*(m_sqrl[2*m + 3]*pmm);

            for (std::size_t l = m + 2; l < order; ++l)
            {
                // P(l,m) = z*a(l,m)*P(l - 1,m) - b(l,m)*P(l - 2,m)
                const std::size_t ind = TriangleLayout::idx(l, m);
                for (std::size_t i = 0; i < z.size(); ++i)
                    plm[ind][i] = z[i]*m_alm[ind]*plm[ind - l][i] - m_blm[ind]*plm[ind - 2*l + 1][i];
                
                // Multiplication by `u` for `m <= l <= lmax - 2`
                for (std::size_t i = 0; i < z.size(); ++i)
                    plm[ind - 2*l + 1][i] *= m_u_scaled[i];
            }

            // Multiplication by `u` for `l = lmax`
            for (std::size_t i = 0; i < z.size(); ++i)
                plm[TriangleLayout::idx(order - 1, m)][i] *= m_u_scaled[i];

            // Multiplication by `u` for `l = lmax - 1`
            for (std::size_t i = 0; i < z.size(); ++i)
                plm[TriangleLayout::idx(order - 2, m)][i] *= m_u_scaled[i];
        }

        for (std::size_t i = 0; i < z.size(); ++i)
            m_u_scaled[i] *= m_u[i];

        // P(lmax,lmax)
        for (std::size_t i = 0; i < z.size(); ++i)
            plm[TriangleLayout::idx(order - 1, order - 1)][i]
                = m_u_scaled[i]*(double(PHASE)*pmm*m_sqrl[2*order - 1]
                /m_sqrl[2*order - 2]);
    }

    std::vector<double> m_sqrl;
    std::vector<double> m_alm;
    std::vector<double> m_blm;
    std::vector<double> m_u_scaled;
    std::vector<double> m_u;
    std::size_t m_max_order;
};

} // namespace st
} // namespace zest