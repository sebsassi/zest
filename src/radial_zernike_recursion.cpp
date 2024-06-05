#include "radial_zernike_recursion.hpp"

#include <cmath>

namespace zest
{
namespace zt
{

RadialZernikeRecursion::RadialZernikeRecursion(std::size_t lmax):
    m_norms(lmax + 1),
    m_k1(RadialZernikeLayout::size(lmax)),
    m_k2(RadialZernikeLayout::size(lmax)),
    m_k3(RadialZernikeLayout::size(lmax)), m_lmax(lmax)
{
    for (std::size_t n = 0; n <= lmax; ++n)
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

void RadialZernikeRecursion::expand(std::size_t lmax)
{
    if (lmax <= m_lmax) return;

    m_norms.resize(lmax + 1);
    m_k1.resize(RadialZernikeLayout::size(lmax));
    m_k2.resize(RadialZernikeLayout::size(lmax));
    m_k3.resize(RadialZernikeLayout::size(lmax));

    for (std::size_t n = m_lmax + 1; n <= lmax; ++n)
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

    m_lmax = lmax;
}

}
}