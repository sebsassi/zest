#pragma once

#include "real_sh_expansion.hpp"

namespace zest
{
namespace st
{

/*
Cross power spectrum of two spherical harmonic expansions.
*/
void cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> a,
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> b,
    std::span<double> out);

/*
Cross power spectrum of two spherical harmonic expansions.
*/
std::vector<double> cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> a,
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> b);

/*
Power spectrum of a spherical harmonic expansion.
*/
void power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion,
    std::span<double> out);

/*
Power spectrum of a spherical harmonic expansion.
*/
std::vector<double> power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion);

}
}