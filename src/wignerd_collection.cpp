#include "wignerd_collection.hpp"

#include <cmath>

namespace zest
{
namespace detail
{

WignerdCollection::WignerdCollection(std::size_t lmax):
    m_matrices(((lmax + 1)*(lmax + 2)*(2*lmax + 3))/6), m_sqrtl_cache(2*lmax + 1), m_inv_sqrtl_cache(2*lmax + 1), m_lmax(lmax)
{
    for (std::size_t l = 1; l <= 2*lmax + 1; ++l)
        m_sqrtl_cache[l - 1] = std::sqrt(double(l));
    
    for (std::size_t i = 0; i < 2*lmax + 1; ++i)
        m_inv_sqrtl_cache[i] = 1.0/m_sqrtl_cache[i];

    m_matrices[idx(0,0,0)] = 1.0;
    if (lmax == 0) return;

    m_matrices[idx(1,0,0)] = 0.0;
    m_matrices[idx(1,1,0)] = -1.0/std::numbers::sqrt2;
    m_matrices[idx(1,0,1)] = 1.0/std::numbers::sqrt2;
    m_matrices[idx(1,1,1)] = 0.5;

    double d_l0 = -1.0/std::numbers::sqrt2;
    for (std::size_t l = 2; l <= lmax; ++l)
    {
        d_l0 *= -m_sqrtl_cache[2*l - 2]/m_sqrtl_cache[2*l - 1];

        // d^l_l,0 = sqrt((2l - 1)/2l)d^l-1_l-1,l
        m_matrices[idx(l,l,0)] = d_l0;

        // d^l_l-1,0 = 0
        m_matrices[idx(l,l - 1,0)] = 0.0;

        for (std::size_t i = 2; i <= l; i += 2)
        {
            std::size_t m2 = l - i;
            // d^l_m2,0 = -sqrt((l - m2 - 1)(l + m2 + 2)/(l - m2)(l + m2 - 1))d^l_m2+2,0
            m_matrices[idx(l,m2,0)] = -m_matrices[idx(l,m2 + 2,0)]
                    *m_sqrtl_cache[l - m2 - 2]*m_sqrtl_cache[l + m2 + 1]
                    *m_inv_sqrtl_cache[l - m2 - 1]*m_inv_sqrtl_cache[l + m2];
        }

        double d_lm = d_l0;

        for (std::size_t m1 = 1; m1 < l; ++m1)
        {
            // d^l_l,m1 = -sqrt((l - m + 1)/(l + m))d^l_l,m1-1
            d_lm *= -m_sqrtl_cache[l - m1]*m_inv_sqrtl_cache[l + m1 - 1];
            m_matrices[idx(l,l,m1)] = d_lm;

            // d^l_l-1,m1 = 2m1/sqrt(2l)*d^l_l,m1
            m_matrices[idx(l,l - 1,m1)] = m_matrices[idx(l,l,m1)]
                    *2.0*double(m1)*m_inv_sqrtl_cache[2*l - 1];

            for (std::size_t i = 2; i <= l - m1; ++i)
            {
                std::size_t m2 = l - i;
                m_matrices[idx(l,m2,m1)] = (2.0*double(m1)*m_matrices[idx(l,m2 + 1,m1)] - m_sqrtl_cache[l - m2 - 2]*m_sqrtl_cache[l + m2 + 1]*m_matrices[idx(l,m2 + 2,m1)])*m_inv_sqrtl_cache[l - m2 - 1]*m_inv_sqrtl_cache[l + m2];
            }
        }

        const double d_ll = -d_lm*m_inv_sqrtl_cache[2*l - 1];
        m_matrices[idx(l,l,l)] = d_ll;

        for (std::size_t m1 = 0; m1 <= l; ++m1)
        {
            for (std::size_t m2 = m1 + 1; m2 <= l; ++m2)
            {
                double sign = ((m1 ^ m2) & 1) ? -1.0 : 1.0;
                m_matrices[idx(l,m1,m2)] = sign*m_matrices[idx(l,m2,m1)];
            }
        }
    }
}

void WignerdCollection::resize(std::size_t lmax)
{
    if (lmax <= m_lmax) return;

    for (std::size_t l = 2*m_lmax + 2; l <= 2*lmax + 1; ++l)
        m_sqrtl_cache[l - 1] = std::sqrt(double(l));
    
    for (std::size_t i = 2*m_lmax + 1; i < 2*lmax + 1; ++i)
        m_inv_sqrtl_cache[i] = 1.0/m_sqrtl_cache[i];

    if (m_lmax == 0)
    {
        m_matrices[idx(1,0,0)] = 0.0;
        m_matrices[idx(1,1,0)] = -1.0/std::numbers::sqrt2;
        m_matrices[idx(1,0,1)] = 1.0/std::numbers::sqrt2;
        m_matrices[idx(1,1,1)] = 0.5;
    }

    double d_l0 = -1.0/std::numbers::sqrt2;
    for (std::size_t l = 2; l <= m_lmax; ++l)    
        d_l0 *= -m_sqrtl_cache[2*l - 2]/m_sqrtl_cache[2*l - 1];
    
    for (std::size_t l = m_lmax + 1; l <= lmax; ++l)
    {
        d_l0 *= -m_sqrtl_cache[2*l - 2]/m_sqrtl_cache[2*l - 1];

        // d^l_l,0 = sqrt((2l - 1)/2l)d^l-1_l-1,l
        m_matrices[idx(l,l,0)] = d_l0;

        // d^l_l-1,0 = 0
        m_matrices[idx(l,l - 1,0)] = 0.0;

        for (std::size_t i = 2; i <= l; i += 2)
        {
            std::size_t m2 = l - i;
            // d^l_m2,0 = -sqrt((l - m2 - 1)(l + m2 + 2)/(l - m2)(l + m2 - 1))d^l_m2+2,0
            m_matrices[idx(l,m2,0)] = -m_matrices[idx(l,m2 + 2,0)]
                    *m_sqrtl_cache[l - m2 - 2]*m_sqrtl_cache[l + m2 + 1]
                    *m_inv_sqrtl_cache[l - m2 - 1]*m_inv_sqrtl_cache[l + m2];
        }

        double d_lm = d_l0;

        for (std::size_t m1 = 1; m1 < l; ++m1)
        {
            // d^l_l,m1 = -sqrt((l - m + 1)/(l + m))d^l_l,m1-1
            d_lm *= -m_sqrtl_cache[l - m1]*m_inv_sqrtl_cache[l + m1 - 1];
            m_matrices[idx(l,l,m1)] = d_lm;

            // d^l_l-1,m1 = 2m1/sqrt(2l)*d^l_l,m1
            m_matrices[idx(l,l - 1,m1)] = m_matrices[idx(l,l,m1)]
                    *2.0*double(m1)*m_inv_sqrtl_cache[2*l - 1];

            for (std::size_t i = 2; i <= l - m1; ++i)
            {
                std::size_t m2 = l - i;
                m_matrices[idx(l,m2,m1)] = (2.0*double(m1)*m_matrices[idx(l,m2 + 1,m1)] - m_sqrtl_cache[l - m2 - 2]*m_sqrtl_cache[l + m2 + 1]*m_matrices[idx(l,m2 + 2,m1)])*m_inv_sqrtl_cache[l - m2 - 1]*m_inv_sqrtl_cache[l + m2];
            }
        }

        const double d_ll = -d_lm*m_inv_sqrtl_cache[2*l - 1];
        m_matrices[idx(l,l,l)] = d_ll;

        for (std::size_t m1 = 0; m1 <= l; ++m1)
        {
            for (std::size_t m2 = m1 + 1; m2 <= l; ++m2)
            {
                double sign = ((m1 ^ m2) & 1) ? -1.0 : 1.0;
                m_matrices[idx(l,m1,m2)] = sign*m_matrices[idx(l,m2,m1)];
            }
        }
    }

    m_lmax = lmax;
}

}    
}