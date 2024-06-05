#pragma once

#include "real_sh_expansion.hpp"
#include "zernike_expansion.hpp"

namespace zest
{
namespace st
{

void cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> a,
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> b,
    std::span<double> out);

std::vector<double> cross_power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> a,
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> b);

void power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion,
    std::span<double> out);

std::vector<double> power_spectrum(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion);

}

namespace zt
{

void power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion,
    std::span<double> out);

std::vector<double> power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion);

}

}