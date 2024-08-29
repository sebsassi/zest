#pragma once

#include <array>
#include <vector>

#include "real_sh_expansion.hpp"
#include "zernike_expansion.hpp"
#include "wignerd_collection.hpp"

namespace zest
{

/**
    @brief Describes whether rotation applies to object or coordinate system
*/
enum class RotationType
{
    /** object is rotated */
    OBJECT,
    /** coordinate system is rotated */
    COORDINATE
};

/**
    @brief Translate rotation matrix into corresponding Euler angles.

    @param rot rotation matrix

    @return Euler angles in order alpha, beta, gamma
*/
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

/**
    @brief Rotations of spherical harmonic and Zernike coefficients.
*/
class Rotor
{
public:
    Rotor() = default;
    explicit Rotor(std::size_t max_order);

    void expand(std::size_t max_order);

    [[nodiscard]] std::size_t
    max_order() const noexcept { return m_temp.size(); }

    /**
        @brief General rotation of a real spherical harmonic expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param convention rotation convention for interpreting the Euler angles

        @note The rotation uses the ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <st::real_sh_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<double, 3>& euler_angles,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr st::SHNorm SH_NORM = std::remove_cvref_t<ExpansionType>::sh_norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        st::RealSHExpansionSpan<std::complex<double>, SH_NORM, PHASE>
        complex_expansion = to_complex_expansion<SH_NORM, PHASE>(expansion);

        set_up_euler_rotations(euler_angles, convention, order);

        for (std::size_t l = 1; l < order; ++l)
            zest::rotate_l(
                complex_expansion[l], wigner_d_pi2(l),
                m_exp_gamma, m_exp_beta, m_exp_alpha, m_temp);

        to_real_expansion<SH_NORM, PHASE>(complex_expansion);
    }

    /**
        @brief General rotation of an even/odd real spherical harmonic expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param convention rotation convention for interpreting the Euler angles

        @note The rotation uses the ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <st::even_odd_real_sh_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<double, 3>& euler_angles,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr st::SHNorm SH_NORM = std::remove_cvref_t<ExpansionType>::sh_norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        auto complex_expansion
            = to_complex_expansion<SH_NORM, PHASE>(expansion);

        set_up_euler_rotations(euler_angles, convention, order);

        for (std::size_t l = std::size_t(expansion.parity()); l < order; l += 2)
            zest::rotate_l(
                complex_expansion[l], wigner_d_pi2(l),
                m_exp_gamma, m_exp_beta, m_exp_alpha, m_temp);

        to_real_expansion<SH_NORM, PHASE>(complex_expansion);
    }

    /**
        @brief General rotation of a real Zernike expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real Zernike expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param convention rotation convention for interpreting the Euler angles

        @note The rotation uses the ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <zt::zernike_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<double, 3>& euler_angles,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr zt::ZernikeNorm ZERNIKE_NORM
            = std::remove_cvref_t<ExpansionType>::zernike_norm;
        constexpr st::SHNorm SH_NORM
            = std::remove_cvref_t<ExpansionType>::sh_norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        zt::ZernikeExpansionSpan<std::complex<double>, ZERNIKE_NORM, SH_NORM, PHASE> 
        complex_expansion = to_complex_expansion<ZERNIKE_NORM, SH_NORM, PHASE>(expansion);

        set_up_euler_rotations(euler_angles, convention, order);

        for (std::size_t n = 1; n < order; ++n)
        {
            auto expansion_n = complex_expansion[n];
            for (std::size_t l = n & 1; l <= n; ++l)
                zest::rotate_l(
                        expansion_n[l], wigner_d_pi2(l),
                        m_exp_gamma, m_exp_beta, m_exp_alpha, m_temp);
        }

        to_real_expansion<ZERNIKE_NORM, SH_NORM, PHASE>(complex_expansion);
    }

    /**
        @brief General rotation of a real spherical harmonic expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param convention rotation convention for interpreting the Euler angles

        @note The rotation uses the ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <st::real_sh_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<std::array<double, 3>, 3>& matrix)
    {
        rotate(expansion, wigner_d_pi2, euler_angles_from_rotation_matrix(matrix));
    }

    /**
        @brief General rotation of a real Zernike expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real Zernike expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param convention rotation convention for interpreting the Euler angles

        @note The rotation uses the ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    template <zt::zernike_expansion ExpansionType>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<std::array<double, 3>, 3>& matrix)
    {
        rotate(expansion, wigner_d_pi2, euler_angles_from_rotation_matrix(matrix));
    }

    /**
        @brief Rotation about the Z-axis of a real spherical harmonic expansion.

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param angle polar rotation angle
        @param convention rotation convention for interpreting the angle
    */
    template <st::real_sh_expansion ExpansionType>
    void polar_rotate(
        ExpansionType&& expansion, double angle,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr st::SHNorm SH_NORM = std::remove_cvref_t<ExpansionType>::sh_norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        st::RealSHExpansionSpan<std::complex<double>, SH_NORM, PHASE> complex_expansion = to_complex_expansion<SH_NORM, PHASE>(expansion);
        
        const double angle_rot = convert(angle, convention);
        for (std::size_t l = 0; l < order; ++l)
            m_exp_alpha[l] = std::polar(1.0, -double(l)*angle_rot);
        
        for (std::size_t l = 1; l < order; ++l)
            zest::polar_rotate_l(complex_expansion[l], m_exp_alpha);

        to_real_expansion<SH_NORM, PHASE>(complex_expansion);
    }

    /**
        @brief Rotation about the Z-axis of a real Zernike expansion

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param angle polar rotation angle
        @param convention rotation convention for interpreting the angle
    */
    template <zt::zernike_expansion ExpansionType>
    void polar_rotate(
        ExpansionType&& expansion, double angle,
        RotationType convention = RotationType::OBJECT)
    {
        constexpr zt::ZernikeNorm ZERNIKE_NORM
            = std::remove_cvref_t<ExpansionType>::zernike_norm;
        constexpr st::SHNorm SH_NORM
            = std::remove_cvref_t<ExpansionType>::sh_norm;
        constexpr st::SHPhase PHASE = std::remove_cvref_t<ExpansionType>::phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        zt::ZernikeExpansionSpan<std::complex<double>, ZERNIKE_NORM, SH_NORM, PHASE> 
        complex_expansion = to_complex_expansion<ZERNIKE_NORM, SH_NORM, PHASE>(expansion);
        
        const double angle_rot = convert(angle, convention);
        for (std::size_t l = 0; l < order; ++l)
            m_exp_alpha[l] = std::polar(1.0, -double(l)*angle_rot);
        
        for (std::size_t n = 1; n < order; ++n)
        {
            auto expansion_n = complex_expansion[n];
            for (std::size_t l = n & 1; l <= n; ++l)
                zest::polar_rotate_l(expansion_n[l], m_exp_alpha);
        }

        to_real_expansion<ZERNIKE_NORM, SH_NORM, PHASE>(complex_expansion);
    }

private:
    void set_up_euler_rotations(
        const std::array<double, 3>& euler_angles, RotationType convention, std::size_t order)
    {
        const auto& [alpha_rot, beta_rot, gamma_rot]
                = convert(euler_angles, convention);

        for (std::size_t m = 0; m < order; ++m)
            m_exp_alpha[m] = std::polar(1.0, -double(m)*alpha_rot);

        for (std::size_t m = 0; m < order; ++m)
            m_exp_beta[m] = std::polar(1.0, -double(m)*beta_rot);

        for (std::size_t m = 0; m < order; ++m)
            m_exp_gamma[m] = std::polar(1.0, -double(m)*gamma_rot);
    }

    std::vector<std::complex<double>> m_temp;
    std::vector<std::complex<double>> m_exp_alpha;
    std::vector<std::complex<double>> m_exp_beta;
    std::vector<std::complex<double>> m_exp_gamma;
};

} // namespace zest