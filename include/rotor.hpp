#pragma once

#include "real_sh_expansion.hpp"
#include "zernike_expansion.hpp"
#include "wignerd_collection.hpp"

namespace zest
{

/*
Switch for whether rotation applies to object or coordinate system
    `OBJECT`: object is rotated
    `COORDINATE`: coordinate system is rotated
*/
enum class RotationType { OBJECT, COORDINATE };

[[nodiscard]] constexpr std::array<double, 3> euler_angles_from_rotation_matrix(
    const std::array<std::array<double, 3>, 3>& rot) noexcept
{
    const double r22_sq = rot[2][2]*rot[2][2];
    const double beta_y = std::sqrt((1.0 + r22_sq)*(1.0 - r22_sq));
    return {
        std::atan2(rot[1][2], rot[0][2]),
        std::atan2(beta_y, rot[2][2]),
        std::atan2(rot[2][1], -rot[2][0])
    };
}

[[nodiscard]] constexpr std::array<double, 3> convert(
    std::array<double, 3> euler_angles, RotationType convention) noexcept
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

[[nodiscard]] constexpr double convert(
    double angle, RotationType convention) noexcept
{
    const double sign = (convention == RotationType::OBJECT) ? 1.0 : -1.0;
    return angle*sign;
}

/*
Rotations of spherical harmonic and Zernike coefficients.
*/
class Rotor
{
public:
    Rotor(): Rotor(0) {}
    explicit Rotor(std::size_t lmax);

    void resize(std::size_t lmax);

    [[nodiscard]] std::size_t
    lmax() const noexcept { return m_wigner_d_pi2.lmax(); }

    /*
    General rotation of a real spherical harmonic expansion via Wigner's D-matrix.

    The rotation uses the standard ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <st::real_sh_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion, const std::array<double, 3>& euler_angles,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr st::SHNorm NORM = std::remove_cvref_t<ExpansionType>::norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        /*
        The rotation here is implemented using the ZXZXZ method, where the Y-rotation is decomposed into a 90 degree rotation about the X-axis, then a rotation by `beta` about the Z-axis, and then a -90 degree rotation about the X-axis.

        The rotations are expressed in the complex SH-basis, and therefore the coefficients are transformed to this basis and then back.
        */

        // The rotations are expressed in the complex SH-basis, and therefore the coefficients are transformed to this basis and then back.
        st::RealSHExpansionSpan<std::complex<double>, NORM, PHASE>
        complex_expansion = to_complex_expansion<NORM, PHASE>(expansion);

        const auto& [alpha_rot, beta_rot, gamma_rot]
                = convert(euler_angles, convention);

        for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
            m_exp_alpha[l] = std::polar(1.0, -double(l)*alpha_rot);

        for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
            m_exp_beta[l] = std::polar(1.0, -double(l)*beta_rot);

        for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
            m_exp_gamma[l] = std::polar(1.0, -double(l)*gamma_rot);

        for (std::size_t l = 1; l <= complex_expansion.lmax(); ++l)
            zest::detail::rotate_l(
                complex_expansion[l], m_wigner_d_pi2(l),
                m_exp_gamma, m_exp_beta, m_exp_alpha, m_temp);

        to_real_expansion<NORM, PHASE>(complex_expansion);
    }

    /*
    General rotation of a real Zernike expansion via Wigner's D-matrix.

    The rotation uses the standard ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <zt::zernike_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion, const std::array<double, 3>& euler_angles,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr st::SHNorm NORM = std::remove_cvref_t<ExpansionType>::norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        /*
        The rotation here is implemented using the ZXZXZ method, where the Y-rotation is decomposed into a 90 degree rotation about the X-axis, then a rotation by `beta` about the Z-axis, and then a -90 degree rotation about the X-axis.

        The rotations are expressed in the complex SH-basis, and therefore the coefficients are transformed to this basis and then back.
        */

        // The rotations are expressed in the complex SH-basis, and therefore the coefficients are transformed to this basis and then back.
        zt::ZernikeExpansionSpan<std::complex<double>, NORM, PHASE> 
        complex_expansion = to_complex_expansion<NORM, PHASE>(expansion);

        const auto& [alpha_rot, beta_rot, gamma_rot]
                = convert(euler_angles, convention);

        for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
            m_exp_alpha[l] = std::polar(1.0, -double(l)*alpha_rot);

        for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
            m_exp_beta[l] = std::polar(1.0, -double(l)*beta_rot);

        for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
            m_exp_gamma[l] = std::polar(1.0, -double(l)*gamma_rot);

        for (std::size_t n = 1; n <= complex_expansion.lmax(); ++n)
        {
            auto expansion_n = complex_expansion[n];
            for (std::size_t l = n & 1; l <= n; ++l)
                zest::detail::rotate_l(
                        expansion_n[l], m_wigner_d_pi2(l),
                        m_exp_gamma, m_exp_beta, m_exp_alpha, m_temp);
        }

        to_real_expansion<NORM, PHASE>(complex_expansion);
    }

    /*
    General rotation of a real spherical harmonic expansion via Wigner's D-matrix.

    The rotation uses the standard ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <st::real_sh_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion,
        const std::array<std::array<double, 3>, 3>& matrix)
    {
        rotate(expansion, euler_angles_from_rotation_matrix(matrix));
    }

    /*
    General rotation of a real Zernike expansion via Wigner's D-matrix.

    The rotation uses the standard ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <zt::zernike_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion,
        const std::array<std::array<double, 3>, 3>& matrix)
    {
        rotate(expansion, euler_angles_from_rotation_matrix(matrix));
    }

    /*
    Rotation about the Z-axis of a real spherical harmonic expansion
    */
    template <st::real_sh_expansion ExpansionType>
    void polar_rotate(
        ExpansionType&& expansion, double angle,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr st::SHNorm NORM = std::remove_cvref_t<ExpansionType>::norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;

        st::RealSHExpansionSpan<std::complex<double>, NORM, PHASE> complex_expansion = to_complex_expansion<NORM, PHASE>(expansion);
        
        const double angle_rot = convert(angle, convention);
        for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
            m_exp_alpha[l] = std::polar(1.0, -double(l)*angle_rot);
        
        for (std::size_t l = 1; l <= complex_expansion.lmax(); ++l)
            zest::detail::polar_rotate_l(complex_expansion[l], m_exp_alpha);

        to_real_expansion<NORM, PHASE>(complex_expansion);
    }

    /*
    Rotation about the Z-axis of a real Zernike expansion
    */
    template <zt::zernike_expansion ExpansionType>
    void polar_rotate(
        ExpansionType&& expansion, double angle,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr st::SHNorm NORM = std::remove_cvref_t<ExpansionType>::norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;

        zt::ZernikeExpansionSpan<std::complex<double>, NORM, PHASE> 
        complex_expansion = to_complex_expansion<NORM, PHASE>(expansion);
        
        const double angle_rot = convert(angle, convention);
        for (std::size_t l = 0; l <= complex_expansion.lmax(); ++l)
            m_exp_alpha[l] = std::polar(1.0, -double(l)*angle_rot);
        
        for (std::size_t n = 1; n <= complex_expansion.lmax(); ++n)
        {
            auto expansion_n = complex_expansion[n];
            for (std::size_t l = n & 1; l <= n; ++l)
                zest::detail::polar_rotate_l(expansion_n[l], m_exp_alpha);
        }

        to_real_expansion<NORM, PHASE>(complex_expansion);
    }

private:
    // Wigner small d-matrices `d[l, m1, m2](pi/2)`
    zest::detail::WignerdCollection m_wigner_d_pi2;
    std::vector<std::complex<double>> m_temp;
    std::vector<std::complex<double>> m_exp_alpha;
    std::vector<std::complex<double>> m_exp_beta;
    std::vector<std::complex<double>> m_exp_gamma;
};

}