#pragma once

#include <span>
#include <vector>

#include "real_sh_expansion.hpp"
#include "zernike_expansion.hpp"
#include "sh_conventions.hpp"

namespace zest
{
namespace st
{

/**
    @brief Compute cross power spectrum of two spherical harmonic expansions.

    @param a spherical harmonic expansion
    @param b spherical harmonic expansion
    @param out output buffer for the cross power spectrum
*/
template <SHNorm NORM, SHPhase PHASE>
void cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> a,
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> b,
    std::span<double> out) noexcept
{
    constexpr double norm = normalization<NORM>();
    std::size_t min_order
            = std::min(std::min(a.order(), b.order()), out.size() - 1);
    
    for (std::size_t l = 0; l < min_order; ++l)
    {
        out[l] = a(l, 0)[0]*b(l, 0)[0];
        for (std::size_t m = 1; m <= l; ++m)
            out[l] += a(l, m)[0]*b(l, m)[0] + a(l, m)[1]*b(l, m)[1];
        out[l] *= norm;
    }
}

/**
    @brief Compute cross power spectrum of two spherical harmonic expansions.

    @param a spherical harmonic expansions
    @param b spherical harmonic expansions

    @return `std::vector` storing the the cross power spectrum
*/
template <SHNorm NORM, SHPhase PHASE>
[[nodiscard]] std::vector<double> cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> a,
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> b)
{
    std::size_t min_order = std::min(a.order(), b.order());
    std::vector<double> out(min_order);
    cross_power_spectrum(a, b, out);
    return out;
}

/**
    @brief Compute power spectrum of a spherical harmonic expansions.

    @param expansion spherical harmonic expansion
    @param out output buffer for the power spectrum
*/
template <SHNorm NORM, SHPhase PHASE>
void power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion,
    std::span<double> out) noexcept
{
    constexpr double norm = normalization<NORM>();
    std::size_t min_order = std::min(out.size(), expansion.order());
    
    for (std::size_t l = 0; l < expansion.order(); ++l)
    {
        out[l] = expansion(l, 0)[0]*expansion(l, 0)[0];
        for (std::size_t m = 1; m <= l; ++m)
            out[l] += expansion(l, m)[0]*expansion(l, m)[0]
                    + expansion(l, m)[1]*expansion(l, m)[1];
        out[l] *= norm;
    }
}

/**
    @brief Compute power spectrum of a spherical harmonic expansions.

    @param expansion spherical harmonic expansion

    @return `std::vector` storing the power spectrum
*/
template <SHNorm NORM, SHPhase PHASE>
[[nodiscard]] std::vector<double> power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion)
{
    std::vector<double> out(expansion.order());
    power_spectrum(expansion, out);
    return out;
}

} // namespace st

namespace zt
{

/**
    @brief Compute power spectrum of a Zernike expansion.

    @param expansion Zernike expansion.
    @param out place to store the power spectrum.
*/
template <ZernikeNorm ZERNIKE_NORM, st::SHNorm SH_NORM, st::SHPhase PHASE>
void power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion,
    RadialZernikeSpan<ZERNIKE_NORM, double> out) noexcept
{
    constexpr double norm = st::normalization<SH_NORM>();
    std::size_t min_order = std::min(out.order(), expansion.order());

    for (std::size_t n = 0; n < min_order; ++n)
    {
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            out(n, l) = expansion(n, l, 0)[0]*expansion(n, l, 0)[0];
            for (std::size_t m = 1; m <= l; ++m)
                out(n, l) += expansion(n, l, m)[0]*expansion(n, l, m)[0]
                        + expansion(n, l, m)[1]*expansion(n, l, m)[1];
            out(n, l) *= norm;
        }
    }
}

/**
    @brief Compute power spectrum of a Zernike expansions.

    @param expansion Zernike expansion

    @return `std::vector` storing the power spectrum.
*/
template <ZernikeNorm ZERNIKE_NORM, st::SHNorm SH_NORM, st::SHPhase PHASE>
[[nodiscard]] std::vector<double> power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>, ZERNIKE_NORM, SH_NORM, PHASE> expansion)
{
    std::vector<double> res(RadialZernikeLayout::size(expansion.order()));
    power_spectrum<ZERNIKE_NORM, SH_NORM, PHASE>(expansion, RadialZernikeSpan<ZERNIKE_NORM, double>(res, expansion.order()));
    return res;
}

} // namespace zt

} // namespace zest