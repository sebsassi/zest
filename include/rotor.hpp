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
#pragma once

#include <array>
#include <vector>
#include <type_traits>

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
    object,
    /** coordinate system is rotated */
    coordinate
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

    if (convention == RotationType::object)
        return {alpha_rot, beta_rot, gamma_rot};
    else
        return {-gamma_rot, -beta_rot, -alpha_rot};
}

[[nodiscard]] constexpr double convert(
    double angle, RotationType convention) noexcept
{
    const double sign = (convention == RotationType::object) ? 1.0 : -1.0;
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
        @param type transformation type

        @note The rotation uses an intrinsic ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the new Y-axis, and the third angle rotates about the new Z-axis again. In summary, the convention is: right-handed, intrinsic, ZYZ
    */
    template <st::real_sh_expansion ExpansionType>
        requires std::same_as<
            typename std::remove_cvref_t<ExpansionType>::value_type, 
            std::array<double, 2>>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<double, 3>& euler_angles, RotationType type)
    {
        constexpr st::SHNorm sh_norm = std::remove_cvref_t<ExpansionType>::norm;
        constexpr st::SHPhase sh_phase
            = std::remove_cvref_t<ExpansionType>::phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        st::RealSHSpan<std::complex<double>, sh_norm, sh_phase>
        complex_expansion = to_complex_expansion<sh_norm, sh_phase>(expansion);

        set_up_euler_rotations(euler_angles, type, order);

        for (auto l : expansion.indices(1))
            zest::rotate_l(
                complex_expansion[l], wigner_d_pi2[l],
                m_exp_gamma, m_exp_beta, m_exp_alpha, m_temp);

        to_real_expansion<sh_norm, sh_phase>(complex_expansion);
    }

    /**
        @brief General rotation of an even/odd real spherical harmonic expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param type transformation type

        @note The rotation uses an intrinsic ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the new Y-axis, and the third angle rotates about the new Z-axis again. In summary, the convention is: right-handed, intrinsic, ZYZ
    */
    template <st::row_skipping_real_sh_expansion ExpansionType>
        requires std::same_as<
            typename std::remove_cvref_t<ExpansionType>::value_type, 
            std::array<double, 2>>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<double, 3>& euler_angles, RotationType type)
    {
        constexpr st::SHNorm sh_norm
            = std::remove_cvref_t<ExpansionType>::norm;
        constexpr st::SHPhase sh_phase
            = std::remove_cvref_t<ExpansionType>::phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        auto complex_expansion
            = to_complex_expansion<sh_norm, sh_phase>(expansion);

        set_up_euler_rotations(euler_angles, type, order);

        for (auto l : expansion.indices(1))
            zest::rotate_l(
                complex_expansion[l], wigner_d_pi2[l],
                m_exp_gamma, m_exp_beta, m_exp_alpha, m_temp);

        to_real_expansion<sh_norm, sh_phase>(complex_expansion);
    }

    /**
        @brief General rotation of a real Zernike expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real Zernike expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param type transformation type

        @note The rotation uses an intrinsic ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the new Y-axis, and the third angle rotates about the new Z-axis again. In summary, the convention is: right-handed, intrinsic, ZYZ
    */
    template <zt::zernike_expansion ExpansionType>
        requires std::same_as<
            typename std::remove_cvref_t<ExpansionType>::value_type, 
            std::array<double, 2>>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<double, 3>& euler_angles, RotationType type)
    {
        constexpr zt::ZernikeNorm zernike_norm
            = std::remove_cvref_t<ExpansionType>::zernike_norm;
        constexpr st::SHNorm sh_norm
            = std::remove_cvref_t<ExpansionType>::sh_norm;
        constexpr st::SHPhase sh_phase = std::remove_cvref_t<ExpansionType>::sh_phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        zt::RealZernikeSpan<std::complex<double>, zernike_norm, sh_norm, sh_phase> 
        complex_expansion = to_complex_expansion<zernike_norm, sh_norm, sh_phase>(expansion);

        set_up_euler_rotations(euler_angles, type, order);

        for (auto n : expansion.indices(1))
        {
            auto expansion_n = complex_expansion[n];
            for (auto l : expansion_n.indices(1))
                zest::rotate_l(
                        expansion_n[l], wigner_d_pi2[l],
                        m_exp_gamma, m_exp_beta, m_exp_alpha, m_temp);
        }

        to_real_expansion<zernike_norm, sh_norm, sh_phase>(complex_expansion);
    }

    /**
        @brief General rotation of a real spherical harmonic expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param type transformation type

        @note The rotation uses an intrinsic ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the new Y-axis, and the third angle rotates about the new Z-axis again. In summary, the convention is: right-handed, intrinsic, ZYZ
    */
    template <st::real_sh_expansion ExpansionType>
        requires std::same_as<
            typename std::remove_cvref_t<ExpansionType>::value_type, 
            std::array<double, 2>>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<std::array<double, 3>, 3>& matrix, RotationType type)
    {
        const std::array<double, 3> euler_angles
            = euler_angles_from_rotation_matrix(matrix);
        rotate(expansion, wigner_d_pi2, euler_angles, type);
    }

    /**
        @brief General rotation of a real Zernike expansion via Wigner's D-matrix.

        @tparam ExpansionType

        @param expansion real Zernike expansion
        @param wigner_d_pi2 Wigner d-matrices at pi/2
        @param euler_angles Euler angles defining the rotation
        @param type transformation type

        @note The rotation uses an intrinsic ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the new Y-axis, and the third angle rotates about the new Z-axis again. In summary, the convention is: right-handed, intrinsic, ZYZ
    */
    template <zt::zernike_expansion ExpansionType>
        requires std::same_as<
            typename std::remove_cvref_t<ExpansionType>::value_type, 
            std::array<double, 2>>
    void rotate(
        ExpansionType&& expansion,
        const zest::WignerdPiHalfCollection& wigner_d_pi2,
        const std::array<std::array<double, 3>, 3>& matrix, RotationType type)
    {
        const std::array<double, 3> euler_angles
            = euler_angles_from_rotation_matrix(matrix);
        rotate(expansion, wigner_d_pi2, euler_angles, type);
    }

    /**
        @brief Rotation about the Z-axis of a real spherical harmonic expansion.

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param angle polar rotation angle
        @param type transformation type
    */
    template <st::real_sh_expansion ExpansionType>
        requires std::same_as<
            typename std::remove_cvref_t<ExpansionType>::value_type, 
            std::array<double, 2>>
    void polar_rotate(
        ExpansionType&& expansion, double angle, RotationType type)
    {
        constexpr st::SHNorm sh_norm = std::remove_cvref_t<ExpansionType>::norm;
        constexpr st::SHPhase sh_phase = std::remove_cvref_t<ExpansionType>::phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        st::RealSHSpan<std::complex<double>, sh_norm, sh_phase> complex_expansion = to_complex_expansion<sh_norm, sh_phase>(expansion);
        
        const double angle_rot = convert(angle, type);
        for (std::size_t l = 0; l < order; ++l)
            m_exp_alpha[l] = std::polar(1.0, -double(l)*angle_rot);
        
        for (auto l : expansion.indices(1))
            zest::polar_rotate_l(complex_expansion[l], m_exp_alpha);

        to_real_expansion<sh_norm, sh_phase>(complex_expansion);
    }

    /**
        @brief Rotation about the Z-axis of a real Zernike expansion

        @tparam ExpansionType

        @param expansion real spherical harmonic expansion
        @param angle polar rotation angle
        @param type transformation type
    */
    template <zt::zernike_expansion ExpansionType>
        requires std::same_as<
            typename std::remove_cvref_t<ExpansionType>::value_type, 
            std::array<double, 2>>
    void polar_rotate(
        ExpansionType&& expansion, double angle, RotationType type)
    {
        constexpr zt::ZernikeNorm zernike_norm
            = std::remove_cvref_t<ExpansionType>::zernike_norm;
        constexpr st::SHNorm sh_norm
            = std::remove_cvref_t<ExpansionType>::sh_norm;
        constexpr st::SHPhase sh_phase
            = std::remove_cvref_t<ExpansionType>::sh_phase;
        
        const std::size_t order = expansion.order();
        expand(order);

        zt::RealZernikeSpan<std::complex<double>, zernike_norm, sh_norm, sh_phase> 
        complex_expansion = to_complex_expansion<zernike_norm, sh_norm, sh_phase>(expansion);
        
        const double angle_rot = convert(angle, type);
        for (std::size_t l = 0; l < order; ++l)
            m_exp_alpha[l] = std::polar(1.0, -double(l)*angle_rot);
        
        for (auto n : expansion.indices(1))
        {
            auto expansion_n = complex_expansion[n];
            for (auto l : expansion_n.indices(1))
                zest::polar_rotate_l(expansion_n[l], m_exp_alpha);
        }

        to_real_expansion<zernike_norm, sh_norm, sh_phase>(complex_expansion);
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