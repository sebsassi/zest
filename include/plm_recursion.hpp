#pragma once

#include <cstddef>
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

/*
Non-owfning view of associated Legendre polynomials.
*/
template <typename T, SHNorm NORM, SHPhase PHASE>
    requires std::same_as<std::remove_const_t<T>, double>
using PlmSpan = SHLMSpan<T, TriangleLayout, NORM, PHASE>;

/*
Non-owfning view of vectors of associated Legendre polynomials.
*/
template <typename T, SHNorm NORM, SHPhase PHASE>
    requires std::same_as<std::remove_const_t<T>, double>
using PlmVecSpan = SHLMVecSpan<T, TriangleLayout, NORM, PHASE>;

/*
Recursion of associated Legendre polynomials

Notes:
The recursion described in (Holmes and Featherstone 2002, J. Geodesy, 76, 279-299).
*/
class PlmRecursion
{
public:
    PlmRecursion(): PlmRecursion(0) {};
    explicit PlmRecursion(std::size_t lmax);

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }

    /*
    Expand the number of cached recursion coefficients up to order `lmax`.
    */
    void expand(std::size_t lmax);

    void expand_vec(std::size_t vec_size);

    /*
    Evaluate recursion of associated Legendre polynomials at a point.

    Parameters:
    `plm`: place to store the evaluated polynomials.
    `z`: point at which the polynomials are evaluated
    */
    template <SHNorm NORM, SHPhase PHASE>
    void plm_real(PlmSpan<double, NORM, PHASE> plm, double z)
    {
        return plm_impl(plm, z, std::sqrt(2.0));
    }

    /*
    Evaluate recursion of associated Legendre polynomials at multiple points.

    Parameters:
    `plm`: place to store the evaluated polynomials.
    `z`: points at which the polynomials are evaluated
    */
    template <SHNorm NORM, SHPhase PHASE>
    void plm_real(
        PlmVecSpan<double, NORM, PHASE> plm, std::span<const double> z)
    {
        return plm_impl(plm, z, std::sqrt(2.0));
    }

    /*
    Evaluate recursion of associated Legendre polynomials at a point.

    Parameters:
    `plm`: place to store the evaluated polynomials.
    `z`: point at which the polynomials are evaluated
    */
    template <SHNorm NORM, SHPhase PHASE>
    void plm_complex(PlmSpan<double, NORM, PHASE> plm, double z)
    {
        return plm_impl(plm, z, 1.0);
    }

    /*
    Evaluate recursion of associated Legendre polynomials at multiple points.

    Parameters:
    `plm`: place to store the evaluated polynomials.
    `z`: points at which the polynomials are evaluated
    */
    template <SHNorm NORM, SHPhase PHASE>
    void plm_complex(
        PlmVecSpan<double, NORM, PHASE> plm, std::span<const double> z)
    {
        return plm_impl(plm, z, 1.0);
    }

private:
    template <SHNorm NORM, SHPhase PHASE>
    void plm_impl(
        PlmSpan<double, NORM, PHASE> plm, double z, double norm)
    {
        const std::size_t lmax = plm.lmax();
        if (std::fabs(z) > 1.0)
            throw std::invalid_argument("z must be between -1 and 1");

        expand(lmax);
        
        const double u = std::sqrt((1.0 - z)*(1.0 + z));

        if constexpr (NORM == SHNorm::GEO)
            plm(0, 0) = 1.0;
        else if constexpr (NORM == SHNorm::QM)
            plm(0, 0) = 1.0/std::sqrt(4.0*std::numbers::pi);

        if (lmax == 0) return;

        if constexpr (NORM == SHNorm::GEO)
            plm(1, 0) = m_sqrl[2]*z;
        else if constexpr (NORM == SHNorm::QM)
            plm(1, 0) = m_sqrl[2]*z*(1.0/std::sqrt(4.0*std::numbers::pi));

        std::span<double> plm_flat = plm.flatten();

        // Calculate P(l,0)
        for (std::size_t l = 2; l <= lmax; ++l)
        {
            const std::size_t ind = TriangleLayout::idx(l,0);
            plm_flat[ind] = m_alm[ind]*z*plm_flat[ind - l] - m_blm[ind]*plm_flat[ind - 2*l + 1];
        }

        constexpr double underflow_compensation = 1.0e-280;

        double pmm;
        if constexpr (NORM == SHNorm::GEO)
            pmm = underflow_compensation*norm;
        else if constexpr (NORM == SHNorm::QM)
            pmm = underflow_compensation*norm*(1.0/std::sqrt(4.0*std::numbers::pi));

        // This number is repeatedly multiplied by u < 1. To avoid underflow
        // at small values of u, we make it large. The rescaling is countered
        // by the presence of `underflow_compensation` in `pmm`.
        double u_scaled = 1.0/underflow_compensation;

        for (std::size_t m = 1; m < lmax; ++m)
        {
            u_scaled *= u;

            // `P(m,m) = u*sqrt((2m + 1)/(2m))*P(m - 1,m - 1)`
            // NOTE: multiplication by `u` happens later
            pmm *= double(PHASE)*m_sqrl[2*m]/m_sqrl[2*m - 1];
            plm(m, m) = pmm;

            // `P(m+1,m) = z*sqrt(2m + 3)*P(m,m)`
            plm(m + 1, m) = z*m_sqrl[2*m + 2]*pmm;

            for (std::size_t l = m + 2; l <= lmax; ++l)
            {
                // P(l,m) = z*a(l,m)*P(l - 1,m) - b(l,m)*P(l - 2,m)
                const std::size_t ind = TriangleLayout::idx(l, m);
                plm_flat[ind] = z*m_alm[ind]*plm_flat[ind - l] - m_blm[ind]*plm_flat[ind - 2*l + 1];
                
                // Multiplication by `u` for `m <= l <= lmax - 2`
                plm_flat[ind - 2*l + 1] *= u_scaled;
            }

            // Multiplication by `u` for `l = lmax`
            plm(lmax, m) *= u_scaled;

            // Multiplication by `u` for `l = lmax - 1`
            plm(lmax - 1, m) *= u_scaled;
        }

        u_scaled *= u;

        // P(lmax,lmax)
        plm(lmax, lmax)
                = double(PHASE)*pmm*u_scaled*m_sqrl[2*lmax]
                /m_sqrl[2*lmax - 1];
    }
    

    template <SHNorm NORM, SHPhase PHASE>
    void plm_impl(
        PlmVecSpan<double, NORM, PHASE> plm, std::span<const double> z,
        double norm)
    {
        if (z.size() != plm.vec_size())
            throw std::invalid_argument(
                    "size of z is incompatible with size of plm");

        for (auto zi : z)
        {
            if (std::fabs(zi) > 1.0)
                throw std::invalid_argument("z must be between -1 and 1");
        }

        expand(plm.lmax());
        expand_vec(z.size());
        
        for (std::size_t i = 0; i < z.size(); ++i)
            m_u[i] = std::sqrt((1.0 - z[i])*(1.0 + z[i]));

        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if constexpr (NORM == SHNorm::GEO)
                plm[0][i] = 1.0;
            else if constexpr (NORM == SHNorm::QM)
                plm[0][i] = 1.0/std::sqrt(4.0*std::numbers::pi);
        }

        if (plm.lmax() == 0) return;

        for (std::size_t i = 0; i < z.size(); ++i)
        {
            if constexpr (NORM == SHNorm::GEO)
                plm[1][i] = z[i]*m_sqrl[2];
            else if constexpr (NORM == SHNorm::QM)
                plm[1][i] = z[i]*(m_sqrl[2]*(1.0/std::sqrt(4.0*std::numbers::pi)));
        }
        // Calculate P(l,0)
        for (std::size_t l = 2; l <= plm.lmax(); ++l)
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
            pmm = underflow_compensation*norm*(1.0/std::sqrt(4.0*std::numbers::pi));

        // This number is repeatedly multiplied by u < 1. To avoid underflow
        // at small values of u, we make it large. The rescaling is countered
        // by the presence of `underflow_compensation` in `pmm`.
        for (std::size_t i = 0; i < z.size(); ++i)
            m_u_scaled[i] = 1.0/underflow_compensation;

        for (std::size_t m = 1; m < plm.lmax(); ++m)
        {
            for (std::size_t i = 0; i < z.size(); ++i)
                m_u_scaled[i] *= m_u[i];

            // `P(m,m) = u*sqrt((2m + 1)/(2m))*P(m - 1,m - 1)`
            // NOTE: multiplication by `u` happens later
            pmm *= double(PHASE)*m_sqrl[2*m]/m_sqrl[2*m - 1];
            for (std::size_t i = 0; i < z.size(); ++i)
                plm[TriangleLayout::idx(m, m)][i] = pmm;

            // `P(m+1,m) = z*sqrt(2m + 3)*P(m,m)`
            for (std::size_t i = 0; i < z.size(); ++i)
                plm[TriangleLayout::idx(m + 1, m)][i] = z[i]*(m_sqrl[2*m + 2]*pmm);

            for (std::size_t l = m + 2; l <= plm.lmax(); ++l)
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
                plm[TriangleLayout::idx(plm.lmax(), m)][i] *= m_u_scaled[i];

            // Multiplication by `u` for `l = lmax - 1`
            for (std::size_t i = 0; i < z.size(); ++i)
                plm[TriangleLayout::idx(plm.lmax() - 1, m)][i] *= m_u_scaled[i];
        }

        for (std::size_t i = 0; i < z.size(); ++i)
            m_u_scaled[i] *= m_u[i];

        // P(lmax,lmax)
        for (std::size_t i = 0; i < z.size(); ++i)
            plm[TriangleLayout::idx(plm.lmax(), plm.lmax())][i]
                = m_u_scaled[i]*(double(PHASE)*pmm*m_sqrl[2*plm.lmax()]
                /m_sqrl[2*plm.lmax() - 1]);
    }

    std::vector<double> m_sqrl;
    std::vector<double> m_alm;
    std::vector<double> m_blm;
    std::vector<double> m_u_scaled;
    std::vector<double> m_u;
    std::size_t m_lmax;
};

}
}