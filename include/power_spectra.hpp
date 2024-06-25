#pragma once

#include "real_sh_expansion.hpp"
#include "zernike_expansion.hpp"
#include "sh_conventions.hpp"

namespace zest
{
namespace st
{

/*
Compute cross power spectrum of two spherical harmonic expansions.

Parameters:
´a´, `b`: spherical harmonic expansions.
`out`: place to store the cross power spectrum.
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

/*
Compute cross power spectrum of two spherical harmonic expansions.

Parameters:
´a´, `b`: spherical harmonic expansions.
`out`: place to store the cross power spectrum.

Returns:
`std::vector` storing the the cross power spectrum.
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

/*
Compute power spectrum of a spherical harmonic expansions.

Parameters:
´expansion`: spherical harmonic expansion.
`out`: place to store the cross power spectrum.
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


/*
Compute power spectrum of a spherical harmonic expansions.

Parameters:
´expansion`: spherical harmonic expansion.

Returns:
`std::vector` storing the power spectrum.
*/
template <SHNorm NORM, SHPhase PHASE>
[[nodiscard]] std::vector<double> power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion)
{
    std::vector<double> out(expansion.order());
    power_spectrum(expansion, out);
    return out;
}

}

namespace zt
{

template <st::SHNorm NORM, st::SHPhase PHASE>
void power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion,
    RadialZernikeSpan<double> out) noexcept
{
    constexpr double norm = st::normalization<NORM>();
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

template <st::SHNorm NORM, st::SHPhase PHASE>
[[nodiscard]] std::vector<double> power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion)
{
    std::vector<double> res(RadialZernikeLayout::size(expansion.order()));
    power_spectrum(expansion, RadialZernikeSpan<double>(res, expansion.order()));
    return res;
}

}

}