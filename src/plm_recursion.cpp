#include "plm_recursion.hpp"

#include <cmath>

namespace zest
{
namespace st
{

PlmRecursion::PlmRecursion(std::size_t max_order):
    m_sqrl(2*max_order - std::min(1UL, max_order)), m_alm(max_order*max_order),
    m_blm(max_order*max_order), m_u_scaled(), m_u(), m_max_order(max_order)
{
    for (std::size_t l = 1; l <= m_sqrl.size(); ++l)
        m_sqrl[l - 1] = std::sqrt(double(l));

    for (std::size_t l = 2; l < m_max_order; ++l)
    {
        const std::size_t ind = TriangleLayout::idx(l,0);
        m_alm[ind] = m_sqrl[2*l - 2]*m_sqrl[2*l]/double(l);
        m_blm[ind] = (double(l) - 1.0)*m_sqrl[2*l]/(m_sqrl[2*l - 4]*double(l));

        for (std::size_t m = 1; m < l - 1; ++m)
        {
            const std::size_t ind = TriangleLayout::idx(l,m);
            // a(l,m) = sqrt((2l - 1)(2l + 1)/((l - m)(l + m)))
            m_alm[ind] = m_sqrl[2*l - 2]*m_sqrl[2*l]
                    /(m_sqrl[l - m - 1]*m_sqrl[l + m - 1]);
            // b(l,m) = sqrt((2l + 1)(l + m - 1)(l - m - 1)/((l - m)(l + m)(2l - 3)))
            m_blm[ind] = m_sqrl[2*l]*m_sqrl[l + m - 2]*m_sqrl[l - m - 2]
                    /(m_sqrl[l - m - 1]*m_sqrl[l + m - 1]*m_sqrl[2*l - 4]);
        }
    }
}

void PlmRecursion::expand(std::size_t max_order)
{
    if (max_order <= m_max_order) return;

    m_sqrl.resize(2*max_order - 1);
    m_alm.resize(max_order*max_order);
    m_blm.resize(max_order*max_order);
    
    for (std::size_t l = 2*m_max_order - 1; l <= m_sqrl.size(); ++l)
        m_sqrl[l - 1] = std::sqrt(double(l));
    
    for (std::size_t l = std::max(2UL, m_max_order - 1); l < max_order; ++l)
    {
        const std::size_t ind = TriangleLayout::idx(l,0);
        m_alm[ind] = m_sqrl[2*l - 2]*m_sqrl[2*l]/double(l);
        m_blm[ind] = (double(l) - 1.0)*m_sqrl[2*l]/(m_sqrl[2*l - 4]*double(l));

        for (std::size_t m = 1; m < l - 1; ++m)
        {
            const std::size_t ind = TriangleLayout::idx(l,m);
            // a(l,m) = sqrt((2l - 1)(2l + 1)/((l - m)(l + m)))
            m_alm[ind] = m_sqrl[2*l - 2]*m_sqrl[2*l]
                    /(m_sqrl[l - m - 1]*m_sqrl[l + m - 1]);
            // b(l,m) = sqrt((2l + 1)(l + m - 1)(l - m - 1)/((l - m)(l + m)(2l - 3)))
            m_blm[ind] = m_sqrl[2*l]*m_sqrl[l + m - 2]*m_sqrl[l - m - 2]
                    /(m_sqrl[l - m - 1]*m_sqrl[l + m - 1]*m_sqrl[2*l - 4]);
        }
    }

    m_max_order = max_order;
}

void PlmRecursion::expand_vec(std::size_t vec_size)
{
    if (m_u_scaled.size() <= vec_size)
    {
        m_u_scaled.resize(vec_size);
        m_u.resize(vec_size);
    }
}

}
}