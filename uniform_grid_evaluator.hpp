#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <utility>
#include <stdexcept>
#include <ranges>
#include <algorithm>
#include <span>

#include <cassert>

#include "real_sh_expansion.hpp"
#include "plm_recursion.hpp"

namespace zest
{
namespace st
{

/*
Evaluate a spherical harmonic expansion on arbitrary uniform grids.
*/
class UniformGridEvaluator
{
public:
    UniformGridEvaluator(): UniformGridEvaluator(0, {}) {}
    explicit UniformGridEvaluator(
        std::size_t lmax, const std::array<std::size_t, 2>& shape);

    void resize(std::size_t lmax, const std::array<std::size_t, 2>& shape);

    std::array<std::vector<double>, 3> evaluate(
        RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, const std::array<std::size_t, 2>& shape);
private:
    PlmRecursion m_recursion;
    std::vector<double> m_plm;
    std::vector<std::complex<double>> m_ffts;
    std::vector<std::size_t> m_pocketfft_shape_grid = {0, 0};
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid = {0, 0};
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft = {0, 0};
    std::array<std::size_t, 2> m_shape;
    std::size_t m_lmax;
};

}
}