#include "power_spectra.hpp"

namespace zest
{
namespace st
{

void cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> a,
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> b,
    std::span<double> out)
{
    std::size_t min_lmax = std::min(a.lmax(), b.lmax());
    if (out.size() < min_lmax + 1)
        throw std::invalid_argument("not enough room to store power spectrum");
    
    for (std::size_t l = 0; l <= min_lmax; ++l)
    {
        out[l] = a(l, 0)[0]*b(l, 0)[0];
        for (std::size_t m = 1; m <= l; ++m)
            out[l] += a(l, m)[0]*b(l, m)[0] + a(l, m)[1]*b(l, m)[1];
    }
}

std::vector<double> cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> a,
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> b)
{
    std::size_t min_lmax = std::min(a.lmax(), b.lmax());
    std::vector<double> out(min_lmax + 1);
    cross_power_spectrum(a, b, out);
    return out;
}

void power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion,
    std::span<double> out)
{
    if (out.size() < expansion.lmax() + 1)
        throw std::invalid_argument("not enough room to store power spectrum");
    
    for (std::size_t l = 0; l <= expansion.lmax(); ++l)
    {
        out[l] = expansion(l, 0)[0]*expansion(l, 0)[0];
        for (std::size_t m = 1; m <= l; ++m)
            out[l] += expansion(l, m)[0]*expansion(l, m)[0]
                    + expansion(l, m)[1]*expansion(l, m)[1];
    }
}

std::vector<double> power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion)
{
    std::vector<double> out(expansion.lmax() + 1);
    power_spectrum(expansion, out);
    return out;
}

}

namespace zt
{

void power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion,
    RadialZernikeSpan<double> out)
{
    std::size_t min_lmax = std::min(out.lmax(), expansion.lmax());

    for (std::size_t n = 0; n < min_lmax; ++n)
    {
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            out(n, l) = expansion(n, l, 0)[0]*expansion(n, l, 0)[0];
            for (std::size_t m = 1; m <= l; ++m)
                out(n, l) += expansion(n, l, m)[0]*expansion(n, l, m)[0]
                        + expansion(n, l, m)[1]*expansion(n, l, m)[1];
        }
    }
}

std::vector<double> power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion)
{
    std::vector<double> res(RadialZernikeLayout::size(expansion.lmax()));
    power_spectrum(expansion, RadialZernikeSpan<double>(res, expansion.lmax()));
    return res;
}

}

}