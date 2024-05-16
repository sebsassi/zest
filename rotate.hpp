#pragma once

#include "real_sh_expansion.hpp"

namespace zest
{
namespace st
{

/*
Switch for whether rotation applies to object or coordinate system
    OBJECT object is rotated
    COORDINATE coordinate system is rotated
*/
enum class RotationType { OBJECT, COORDINATE };

constexpr std::array<double, 3> euler_angles_from_rotation_matrix(
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

/*
Rotations of spherical harmonic coefficients.
*/
class SHRotor
{
private:
    class WignerdCollection;

public:
    SHRotor(): SHRotor(0) {}
    explicit SHRotor(std::size_t lmax);

    /*
    General rotation of a real spherical harmonic expansion via Wigner's D-matrix.

    The rotation uses the standard ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    void rotate(
        RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion,
        const std::array<double, 3>& euler_angles,
        RotationType convention = RotationType::OBJECT);

    /*
    General rotation of a real spherical harmonic expansion via Wigner's D-matrix.

    The rotation uses the standard ZYZ convention, where the first Euler angle rotates about the Z-axis, the second Euler angle rotates about the Y-axis, and the third angle rotates about the Z-axis again.
    */
    void rotate(
        RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, const std::array<std::array<double, 3>, 3>& matrix)
    {
        rotate(expansion, euler_angles_from_rotation_matrix(matrix));
    }

    /*
    Rotation about the Z-axis of a real spherical harmonic expansion
    */
    void polar_rotate(
        RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, double angle, RotationType convention);

    const WignerdCollection& wigner_d_pi2() const { return m_wigner_d_pi2; } 

private:
    template <typename T>
    class WignerdSpan
    {
    public:
        WignerdSpan(std::span<T> span, std::size_t idx, std::size_t l):
            m_span(span.begin() + idx, (l + 1)*(l + 1)), m_l(l) {}
        
        double operator()(std::size_t m1, std::size_t m2) const noexcept
        {
            return m_span[idx(m1, m2)];
        }

        std::size_t idx(std::size_t m1, std::size_t m2) const noexcept
        {
            return (m_l + 1)*m1 + m2;
        }

    private:
        std::span<T> m_span;
        std::size_t m_l;
    };

    /* Collection of Wigner (small) d-matrix elements at pi/2 */
    class WignerdCollection
    {
    public:
        WignerdCollection(): WignerdCollection(0) {}
        explicit WignerdCollection(std::size_t lmax);

        double
        operator()(std::size_t l, std::size_t m1, std::size_t m2) const noexcept
        {
            return m_matrices[idx(l, m1, m2)];
        }

        WignerdSpan<const double> operator()(std::size_t l)
        {
            return WignerdSpan<const double>(
                    m_matrices, (l*(l + 1)*(2*l + 1))/6, l);
        }

    private:
        static constexpr std::size_t idx(
            std::size_t l, std::size_t m1, std::size_t m2) noexcept
        {
            return (l*(l + 1)*(2*l + 1))/6 + (l + 1)*m1 + m2;
        }

        std::vector<double> m_matrices;
        std::vector<double> m_sqrtl_cache;
        std::vector<double> m_inv_sqrtl_cache;
    };

    // Wigner small d-matrices `d[l, m1, m2](pi/2)`
    WignerdCollection m_wigner_d_pi2;
    std::vector<std::complex<double>> m_temp;
    std::vector<std::complex<double>> m_exp_alpha;
    std::vector<std::complex<double>> m_exp_beta;
    std::vector<std::complex<double>> m_exp_gamma;
    std::vector<double> m_alternating;
    std::size_t m_lmax;
};

}
}