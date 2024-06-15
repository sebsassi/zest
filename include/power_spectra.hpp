#pragma once

#include "real_sh_expansion.hpp"
#include "zernike_expansion.hpp"
#include "sh_conventions.hpp"

namespace zest
{
namespace st
{

template <SHNorm NORM, SHPhase PHASE>
void cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> a,
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> b,
    std::span<double> out) noexcept
{
    constexpr double norm = normalization<NORM>();
    std::size_t min_lmax
            = std::min(std::min(a.lmax(), b.lmax()), out.size() - 1);
    
    for (std::size_t l = 0; l <= min_lmax; ++l)
    {
        out[l] = a(l, 0)[0]*b(l, 0)[0];
        for (std::size_t m = 1; m <= l; ++m)
            out[l] += a(l, m)[0]*b(l, m)[0] + a(l, m)[1]*b(l, m)[1];
        out[l] *= norm;
    }
}

template <SHNorm NORM, SHPhase PHASE>
[[nodiscard]] std::vector<double> cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> a,
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> b)
{
    std::size_t min_lmax = std::min(a.lmax(), b.lmax());
    std::vector<double> out(min_lmax + 1);
    cross_power_spectrum(a, b, out);
    return out;
}

template <SHNorm NORM, SHPhase PHASE>
void power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion,
    std::span<double> out) noexcept
{
    constexpr double norm = normalization<NORM>();
    std::size_t min_lmax = std::min(out.size() - 1, expansion.lmax());
    
    for (std::size_t l = 0; l <= expansion.lmax(); ++l)
    {
        out[l] = expansion(l, 0)[0]*expansion(l, 0)[0];
        for (std::size_t m = 1; m <= l; ++m)
            out[l] += expansion(l, m)[0]*expansion(l, m)[0]
                    + expansion(l, m)[1]*expansion(l, m)[1];
        out[l] *= norm;
    }
}

template <SHNorm NORM, SHPhase PHASE>
[[nodiscard]] std::vector<double> power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion)
{
    std::vector<double> out(expansion.lmax() + 1);
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
    std::size_t min_lmax = std::min(out.lmax(), expansion.lmax());

    for (std::size_t n = 0; n < min_lmax; ++n)
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
    std::vector<double> res(RadialZernikeLayout::size(expansion.lmax()));
    power_spectrum(expansion, RadialZernikeSpan<double>(res, expansion.lmax()));
    return res;
}

}

}