#include "radial_zernike_recursion.hpp"

#include <cmath>

namespace zest
{
namespace zt
{

RadialZernikeRecursion::RadialZernikeRecursion(std::size_t max_order):
    m_norms(max_order),
    m_k1(RadialZernikeLayout::size(max_order)),
    m_k2(RadialZernikeLayout::size(max_order)),
    m_k3(RadialZernikeLayout::size(max_order)), m_max_order(max_order)
{
    for (std::size_t n = 0; n < max_order; ++n)
    {
        const double dn = double(n);
        m_norms[n] = std::sqrt(2.0*dn + 3.0);
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            const double dl = double(l);
            const double k0 = (dn - dl)*(dn + dl + 1.0)*(2.0*dn - 3.0);
            const double k1 = (2.0*dn - 1.0)*(2.0*dn + 1)*(2.0*dn - 3.0);
            const double k2 = -0.5*(2.0*dn - 1.0)*((2*dl + 1.0)*(2.0*dl + 1.0) + (2.0*dn + 1)*(2.0*dn - 3.0));
            const double k3 = -(dn - dl - 2.0)*(dn + dl - 1.0)*(2.0*dn + 1);
            
            const double k0_inv = 1.0/k0;
            const std::size_t idx = RadialZernikeLayout::idx(n,l);
            m_k1[idx] = k1*k0_inv;
            m_k2[idx] = k2*k0_inv;
            m_k3[idx] = k3*k0_inv;
        }
    }
}

void RadialZernikeRecursion::expand(std::size_t max_order)
{
    if (max_order <= m_max_order) return;

    m_norms.resize(max_order);
    m_k1.resize(RadialZernikeLayout::size(max_order));
    m_k2.resize(RadialZernikeLayout::size(max_order));
    m_k3.resize(RadialZernikeLayout::size(max_order));

    for (std::size_t n = m_max_order; n < max_order; ++n)
    {
        const double dn = double(n);
        m_norms[n] = std::sqrt(2.0*dn + 3.0);
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            const double dl = double(l);
            const double k0 = (dn - dl)*(dn + dl + 1.0)*(2.0*dn - 3.0);
            const double k1 = (2.0*dn - 1.0)*(2.0*dn + 1)*(2.0*dn - 3.0);
            const double k2 = -0.5*(2.0*dn - 1.0)*((2*dl + 1.0)*(2.0*dl + 1.0) + (2.0*dn + 1)*(2.0*dn - 3.0));
            const double k3 = -(dn - dl - 2.0)*(dn + dl - 1.0)*(2.0*dn + 1);
            
            const double k0_inv = 1.0/k0;
            const std::size_t idx = RadialZernikeLayout::idx(n,l);
            m_k1[idx] = k1*k0_inv;
            m_k2[idx] = k2*k0_inv;
            m_k3[idx] = k3*k0_inv;
        }
    }

    m_max_order = max_order;
}

} // namespace zt
} // namespace zest