#include "affine_legendre.hpp"

namespace zest
{

AffineLegendreRecursion::AffineLegendreRecursion(std::size_t lmax):
    m_a(lmax + 1), m_b(lmax + 1), m_c(lmax + 1), m_d(lmax + 1),
    m_lmax(lmax + 1)
{
    m_c[1] = 2.0/5.0;
    m_d[1] = 1.0;
    for (std::size_t n = 2; n <= lmax; ++n)
    {
        const double dn = double(n);
        const double inv_dn = 1.0/dn;
        m_a[n] = (2.0*dn - 1.0)*inv_dn;
        m_b[n] = (dn - 1.0)*inv_dn;
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0);
        m_d[n] = dn/(2.0*dn - 1.0);
    }
}

void AffineLegendreRecursion::expand(std::size_t lmax)
{
    if (lmax <= m_lmax) return;

    m_a.resize(lmax);
    m_b.resize(lmax);
    m_c.resize(lmax);
    m_d.resize(lmax);

    for (std::size_t n = m_lmax + 1; n <= lmax; ++n)
    {
        const double dn = double(n);
        const double inv_dn = 1.0/dn;
        m_a[n] = (2.0*dn - 1.0)*inv_dn;
        m_b[n] = (dn - 1.0)*inv_dn;
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0);
        m_d[n] = dn/(2.0*dn - 1.0);
    }

    m_lmax = lmax;
}

void AffineLegendreRecursion::evaluate_affine(
    TriangleSpan<double, TriangleLayout> expansion, double shift, double scale)
{
    expand(expansion.lmax());

    expansion(0, 0) = 1.0;
    expansion(1, 0) = shift;
    expansion(1, 1) = scale;
    for (std::size_t n = 2; n <= expansion.lmax(); ++n)
    {
        expansion(n, 0) = m_a[n]*shift*expansion(n - 1, 0)
                - m_b[n]*expansion(n - 2, 0)
                + m_a[n]*(1.0/3.0)*scale*expansion(n - 1, 1);
        for (std::size_t l = 1; l <= n - 2; ++l)
        {
            expansion(n, l) = m_a[n]*shift*expansion(n - 1, l)
                    - m_b[n]*expansion(n - 2, l)
                    + m_a[n]*m_c[l]*scale*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*scale*expansion(n - 1, l - 1);
        }

        expansion(n, n - 1) = m_a[n]*shift*expansion(n - 1, n - 1)
                + m_a[n]*m_d[n - 1]*scale*expansion(n - 1, n - 2);
        expansion(n, n) = scale*expansion(n - 1, n - 1);
    }
}

void AffineLegendreRecursion::evaluate_shifted(
    TriangleSpan<double, TriangleLayout> expansion, double shift)
{
    expand(expansion.lmax());

    expansion(0, 0) = 1.0;
    expansion(1, 0) = shift;
    expansion(1, 1) = 1.0;
    for (std::size_t n = 2; n <= expansion.lmax(); ++n)
    {
        expansion(n, 0) = m_a[n]*shift*expansion(n - 1, 0)
                - m_b[n]*expansion(n - 2, 0)
                + m_a[n]*(1.0/3.0)*expansion(n - 1, 1);
        for (std::size_t l = 1; l <= n - 2; ++l)
        {
            expansion(n, l) = m_a[n]*shift*expansion(n - 1, l)
                    - m_b[n]*expansion(n - 2, l)
                    + m_a[n]*m_c[l]*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*expansion(n - 1, l - 1);
        }

        expansion(n, n - 1) = m_a[n]*shift*expansion(n - 1, n - 1)
                + m_a[n]*m_d[n - 1]*expansion(n - 1, n - 2);
        expansion(n, n) = expansion(n - 1, n - 1);
    }
}

void AffineLegendreRecursion::evaluate_scaled(
    TriangleSpan<double, TriangleLayout> expansion, double scale)
{
    expand(expansion.lmax());

    std::ranges::fill(expansion.span(), 0.0);
    expansion(0, 0) = 1.0;
    expansion(1, 1) = scale;

    for (std::size_t n = 2; n <= expansion.lmax(); ++n)
    {
        expansion(n, 0) = m_a[n]*(1.0/3.0)*scale*expansion(n - 1, 1)
                - m_b[n]*expansion(n - 2, 0);
        
        const std::size_t lmin = 1 + ((n & 1) ^ 1);
        for (std::size_t l = lmin; l <= n - 2; l += 2)
        {
            expansion(n, l) = m_a[n]*m_c[l]*scale*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*scale*expansion(n - 1, l - 1)
                    - m_b[n]*expansion(n - 2, l);
        }

        expansion(n, n) = scale*expansion(n - 1, n - 1);
    }
}

}