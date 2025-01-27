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
#include "plm_recursion.hpp"

#include <cmath>

namespace zest
{
namespace st
{

PlmRecursion::PlmRecursion(std::size_t max_order):
    m_sqrl(2*max_order), m_alm(max_order*max_order),
    m_blm(max_order*max_order), m_u_scaled(), m_u(), m_max_order(max_order)
{
    for (std::size_t l = 1; l < m_sqrl.size(); ++l)
        m_sqrl[l] = std::sqrt(double(l));

    for (std::size_t l = 2; l < m_max_order; ++l)
    {
        const std::size_t ind = PlmLayout::idx(l, 0);
        m_alm[ind] = m_sqrl[2*l - 1]*m_sqrl[2*l + 1]/double(l);
        m_blm[ind] = (double(l) - 1.0)*m_sqrl[2*l + 1]/(m_sqrl[2*l - 3]*double(l));

        for (std::size_t m = 1; m < l - 1; ++m)
        {
            const std::size_t ind = PlmLayout::idx(l, m);
            // a(l,m) = sqrt((2l - 1)(2l + 1)/((l - m)(l + m)))
            m_alm[ind] = m_sqrl[2*l - 1]*m_sqrl[2*l + 1]
                    /(m_sqrl[l - m]*m_sqrl[l + m]);
            // b(l,m) = sqrt((2l + 1)(l + m - 1)(l - m - 1)/((l - m)(l + m)(2l - 3)))
            m_blm[ind] = m_sqrl[2*l + 1]*m_sqrl[l + m - 1]*m_sqrl[l - m - 1]
                    /(m_sqrl[l - m]*m_sqrl[l + m]*m_sqrl[2*l - 3]);
        }
    }
}

void PlmRecursion::expand(std::size_t max_order)
{
    if (max_order <= m_max_order) return;

    const std::size_t prev_sqrl_size = m_sqrl.size();
    m_sqrl.resize(2*max_order);
    m_alm.resize(max_order*max_order);
    m_blm.resize(max_order*max_order);
    
    for (std::size_t l = prev_sqrl_size; l < m_sqrl.size(); ++l)
        m_sqrl[l] = std::sqrt(double(l));
    
    for (std::size_t l = std::max(2UL, m_max_order); l < max_order; ++l)
    {
        const std::size_t ind = PlmLayout::idx(l, 0);
        m_alm[ind] = m_sqrl[2*l - 1]*m_sqrl[2*l + 1]/double(l);
        m_blm[ind] = (double(l) - 1.0)*m_sqrl[2*l + 1]/(m_sqrl[2*l - 3]*double(l));

        for (std::size_t m = 1; m < l - 1; ++m)
        {
            const std::size_t ind = PlmLayout::idx(l, m);
            // a(l,m) = sqrt((2l - 1)(2l + 1)/((l - m)(l + m)))
            m_alm[ind] = m_sqrl[2*l - 1]*m_sqrl[2*l + 1]
                    /(m_sqrl[l - m]*m_sqrl[l + m]);
            // b(l,m) = sqrt((2l + 1)(l + m - 1)(l - m - 1)/((l - m)(l + m)(2l - 3)))
            m_blm[ind] = m_sqrl[2*l + 1]*m_sqrl[l + m - 1]*m_sqrl[l - m - 1]
                    /(m_sqrl[l - m]*m_sqrl[l + m]*m_sqrl[2*l - 3]);
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

} // namespace st
} // namespace zest