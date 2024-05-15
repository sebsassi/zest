#include "rotate.hpp"

namespace zest
{
namespace st
{

SHRotor::SHRotor(std::size_t lmax):
    m_wigner_d_pi2(lmax), m_temp(lmax + 1), m_exp_alpha(lmax + 1), m_exp_beta(lmax + 1), m_exp_gamma(lmax + 1), m_alternating(2*(lmax + 1)), m_lmax(lmax) {}

SHRotor::WignerdCollection::WignerdCollection(std::size_t lmax):
    m_matrices(((lmax + 1)*(lmax + 2)*(2*lmax + 3))/6), m_sqrtl_cache(2*lmax + 1), m_inv_sqrtl_cache(2*lmax + 1)
{
    for (std::size_t l = 1; l <= 2*lmax + 1; ++l)
        m_sqrtl_cache[l - 1] = std::sqrt(double(l));
    
    for (std::size_t i = 0; i < 2*lmax + 1; ++i)
        m_inv_sqrtl_cache[i] = 1.0/m_sqrtl_cache[i];

    m_matrices[idx(0,0,0)] = 1.0;
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

constexpr std::array<double, 3> convert(
    std::array<double, 3> euler_angles, RotationType convention)
{
    const auto& [alpha, beta, gamma] = euler_angles;

    const double alpha_rot = alpha - 0.5*std::numbers::pi;
    const double beta_rot = beta;
    const double gamma_rot = gamma + 0.5*std::numbers::pi;

    if (convention == RotationType::OBJECT)
        return {alpha_rot, beta_rot, gamma_rot};
    else
        return {-gamma_rot, -beta_rot, -alpha_rot};
}

void SHRotor::rotate(
        RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion,
        const std::array<double, 3>& euler_angles,
        RotationType convention)
{
    /*
    The rotation here is implemented using the ZXZXZ method, where the Y-rotation is decomposed into a 90 degree rotation about the X-axis, then a rotation by `beta` about the Z-axis, and then a -90 degree rotation about the X-axis.

    The rotations are expressed in the complex SH-basis, and therefore the coefficients are transformed to this basis and then back.
    */

    // The rotations are expressed in the complex SH-basis, and therefore the coefficients are transformed to this basis and then back.
    RealSHExpansionSpan<std::complex<double>, SHNorm::QM, SHPhase::CS> complex_expansion
            = to_complex_expansion<SHNorm::QM, SHPhase::CS>(expansion);

    const auto& [alpha_rot, beta_rot, gamma_rot]
            = convert(euler_angles, convention);

    for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
        m_exp_alpha[l] = std::polar(1.0, -double(l)*alpha_rot);

    for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
        m_exp_beta[l] = std::polar(1.0, -double(l)*beta_rot);

    for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
        m_exp_gamma[l] = std::polar(1.0, -double(l)*gamma_rot);

    for (std::size_t l = 1; l <= complex_expansion.lmax(); ++l)
    {
        const std::size_t l_is_odd = l & 1;

        WignerdSpan<const double> wigner_d = m_wigner_d_pi2(l);
        std::span<std::complex<double>> expansion_l = complex_expansion(l);

        // gamma rotation
        for (std::size_t m = 0; m <= l; ++m)
        {
            const std::complex<double> coeff = expansion_l[m];
            const std::complex<double> rot = m_exp_gamma[m];

            // Standard complex multiplication not used because it has additional logic for `Inf` components. We assume finite valuess
            expansion_l[m] = {
                coeff.real()*rot.real() - coeff.imag()*rot.imag(),
                coeff.real()*rot.imag() + coeff.imag()*rot.real()
            };
        }
        
        // 90 degree rotation
        if (l_is_odd)
        {
            for (std::size_t m = 0; m < l; m += 2)
            {
                std::array<double, 4> sum = {
                    0.5*wigner_d(m,0)*expansion_l[0].real(), 0.0,
                    0.5*wigner_d(m + 1,0)*expansion_l[0].real(), 0.0
                };

                for (std::size_t mp = 1; mp < l; mp += 2)
                {
                    const std::array<double, 4> wig = {
                        wigner_d(m,mp), wigner_d(m + 1,mp),
                        wigner_d(m + 1,mp + 1), wigner_d(m,mp + 1)
                    };
                    sum[0] += wig[0]*expansion_l[mp].real();
                    sum[1] += wig[1]*expansion_l[mp].imag();
                    sum[2] += wig[2]*expansion_l[mp + 1].real();
                    sum[3] += wig[3]*expansion_l[mp + 1].imag();
                }

                sum[0] += wigner_d(m,l)*expansion_l[l].real();
                sum[1] += wigner_d(m + 1,l)*expansion_l[l].imag();

                m_temp[m] = std::complex<double>{2.0*sum[0], 2.0*sum[3]};
                m_temp[m + 1] = std::complex<double>{2.0*sum[2], 2.0*sum[1]};
            }
        }
        else
        {
            for (std::size_t m = 0; m < l; m += 2)
            {
                std::array<double, 4> sum = {
                    0.5*wigner_d(m + 1,0)*expansion_l[0].real(), 0.0,
                    0.5*wigner_d(m,0)*expansion_l[0].real(), 0.0
                };

                for (std::size_t mp = 1; mp < l; mp += 2)
                {
                    sum[0] += wigner_d(m + 1,mp)*expansion_l[mp].real();
                    sum[1] += wigner_d(m,mp)*expansion_l[mp].imag();
                    sum[2] += wigner_d(m,mp + 1)*expansion_l[mp + 1].real();
                    sum[3] += wigner_d(m + 1,mp + 1)*expansion_l[mp + 1].imag();
                }

                m_temp[m] = std::complex<double>{2.0*sum[2], 2.0*sum[1]};
                m_temp[m + 1] = std::complex<double>{2.0*sum[0], 2.0*sum[3]};
            }

            std::array<double, 2> sum = {
                0.0, 0.5*wigner_d(l,0)*expansion_l[0].real()
            };

            for (std::size_t mp = 1; mp < l; mp += 2)
            {
                sum[1] += wigner_d(l,mp + 1)*expansion_l[mp + 1].real();
                sum[0] += wigner_d(l,mp)*expansion_l[mp].imag();
            }

            m_temp[l] = std::complex<double>{2.0*sum[1], 2.0*sum[0]};
        }

        // beta rotation
        for (std::size_t m = 0; m <= l; ++m)
        {
            const std::complex<double> coeff = m_temp[m];
            const std::complex<double> rot = m_exp_beta[m];

            // Standard complex multiplication not used because it has additional logic for `Inf` components. We assume finite values
            expansion_l[m] = {
                coeff.real()*rot.real() - coeff.imag()*rot.imag(),
                coeff.real()*rot.imag() + coeff.imag()*rot.real()
            };
        }
        
        // -90 degree rotation
        if (l_is_odd)
        {
            for (std::size_t m = 0; m < l; m += 2)
            {
                std::array<double, 4> sum = {
                    0.5*wigner_d(0,m)*expansion_l[0].real(), 0.0,
                    0.5*wigner_d(0,m + 1)*expansion_l[0].real(), 0.0
                };

                for (std::size_t mp = 1; mp < l; mp += 2)
                {
                    sum[0] += wigner_d(mp,m)*expansion_l[mp].real();
                    sum[1] += wigner_d(mp,m + 1)*expansion_l[mp].imag();
                    sum[2] += wigner_d(mp + 1,m + 1)*expansion_l[mp + 1].real();
                    sum[3] += wigner_d(mp + 1,m)*expansion_l[mp + 1].imag();
                }

                sum[0] += wigner_d(l,m)*expansion_l[l].real();
                sum[1] += wigner_d(l,m + 1)*expansion_l[l].imag();

                m_temp[m] = std::complex<double>{2.0*sum[0], 2.0*sum[3]};
                m_temp[m + 1] = std::complex<double>{2.0*sum[2], 2.0*sum[1]};
            }
        }
        else
        {
            for (std::size_t m = 0; m < l; m += 2)
            {
                std::array<double, 4> sum = {
                    0.5*wigner_d(0,m + 1)*expansion_l[0].real(), 0.0,
                    0.5*wigner_d(0,m)*expansion_l[0].real(), 0.0
                };

                for (std::size_t mp = 1; mp < l; mp += 2)
                {
                    sum[0] += wigner_d(mp,m + 1)*expansion_l[mp].real();
                    sum[1] += wigner_d(mp,m)*expansion_l[mp].imag();
                    sum[2] += wigner_d(mp + 1,m)*expansion_l[mp + 1].real();
                    sum[3] += wigner_d(mp + 1,m + 1)*expansion_l[mp + 1].imag();
                }

                m_temp[m] = std::complex<double>{2.0*sum[2], 2.0*sum[1]};
                m_temp[m + 1] = std::complex<double>{2.0*sum[0], 2.0*sum[3]};
            }

            std::array<double, 2> sum = {
                0.0, 0.5*wigner_d(l,0)*expansion_l[0].real()
            };

            for (std::size_t mp = 1; mp < l; mp += 2)
            {
                sum[1] += wigner_d(mp + 1,l)*expansion_l[mp + 1].real();
                sum[0] += wigner_d(mp,l)*expansion_l[mp].imag();
            }

            m_temp[l] = std::complex<double>{2.0*sum[1], 2.0*sum[0]};
        }

        // alpha rotation
        for (std::size_t m = 0; m <= l; ++m)
        {
            const std::complex<double> coeff = m_temp[m];
            const std::complex<double> rot = m_exp_alpha[m];

            // Standard complex multiplication not used because it has additional logic for `Inf` components. We assume finite values
            expansion_l[m] = {
                coeff.real()*rot.real() - coeff.imag()*rot.imag(),
                coeff.real()*rot.imag() + coeff.imag()*rot.real()
            };
        }
    }

    to_real_expansion<SHNorm::GEO, SHPhase::NONE>(complex_expansion);
}

}
}