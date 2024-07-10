#include "rotor.hpp"

namespace zest
{

Rotor::Rotor(std::size_t max_order):
    m_wigner_d_pi2(max_order), m_temp(max_order), m_exp_alpha(max_order), m_exp_beta(max_order), m_exp_gamma(max_order) {}

void Rotor::expand(std::size_t max_order)
{
    if (max_order <= m_wigner_d_pi2.max_order()) return;

    m_wigner_d_pi2.expand(max_order);
    m_temp.resize(max_order);
    m_exp_alpha.resize(max_order);
    m_exp_beta.resize(max_order);
    m_exp_gamma.resize(max_order);
}

} // namespace zest