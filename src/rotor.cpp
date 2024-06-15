#include "rotor.hpp"

namespace zest
{

Rotor::Rotor(std::size_t lmax):
    m_wigner_d_pi2(lmax), m_temp(lmax + 1), m_exp_alpha(lmax + 1), m_exp_beta(lmax + 1), m_exp_gamma(lmax + 1) {}

void Rotor::resize(std::size_t lmax)
{
    if (lmax <= m_wigner_d_pi2.lmax()) return;

    m_wigner_d_pi2.resize(lmax);
    m_temp.resize(lmax + 1);
    m_exp_alpha.resize(lmax + 1);
    m_exp_beta.resize(lmax + 1);
    m_exp_gamma.resize(lmax + 1);
}

}