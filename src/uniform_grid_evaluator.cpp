#include "uniform_grid_evaluator.hpp"

#include <cmath>
#include <utility>
#include <stdexcept>
#include <ranges>
#include <algorithm>

#include "pocketfft.hpp"

namespace zest
{
namespace st
{

UniformGridEvaluator::UniformGridEvaluator(
    std::size_t lmax, const std::array<std::size_t, 2>& shape):
    m_recursion(lmax), m_plm(lmax),
    m_ffts(shape[0]*(((shape[1] - 1) >> 1) + 1)),
    m_pocketfft_shape_grid(2), m_pocketfft_stride_grid(2),
    m_pocketfft_stride_fft(2), m_shape(shape), m_lmax(lmax)
{
    m_pocketfft_shape_grid[0] = shape[0];
    m_pocketfft_shape_grid[1] = shape[1] - 1;

    m_pocketfft_stride_grid[0] = long(shape[1]*sizeof(double));
    m_pocketfft_stride_grid[1] = sizeof(double);

    const std::size_t num_fft = ((shape[1] - 1) >> 1) + 1;
    m_pocketfft_stride_fft[0] = long(num_fft*sizeof(std::complex<double>));
    m_pocketfft_stride_fft[1] = sizeof(std::complex<double>);
}

void UniformGridEvaluator::resize(
    std::size_t lmax, const std::array<std::size_t, 2>& shape)
{
    if (lmax != m_lmax)
    {
        m_recursion.expand(lmax);
        m_plm.resize(DualTriangleLayout::size(lmax));
        m_lmax = lmax;
    }

    if (shape != m_shape)
    {
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1] - 1;

        m_pocketfft_stride_grid[0] = long(shape[1]*sizeof(double));
        m_pocketfft_stride_grid[1] = sizeof(double);

        const std::size_t num_fft = ((shape[1] - 1) >> 1) + 1;
        m_pocketfft_stride_fft[0] = long(num_fft*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = sizeof(std::complex<double>);

        m_ffts.resize(shape[0]*num_fft);

        m_shape = shape;
    }

}

template <std::floating_point T>
std::vector<T> linspace(T start, T stop, std::size_t count)
{
    if (count == 0) return {};
    if (count == 1) return {start};

    std::vector<T> res(count);
    const T step = (stop - start)/T(count - 1);
    for (std::size_t i = 0; i < count - 1; ++i)
        res[i] = start + T(i)*step;
    
    res[count - 1] = stop;

    return res;
}

std::array<std::vector<double>, 3> UniformGridEvaluator::evaluate(
    RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, const std::array<std::size_t, 2>& shape)
{
    resize(expansion.lmax(), shape);

    const auto [num_lat, num_lon] = shape;
    const std::size_t num_fft = ((num_lon - 1) >> 1) + 1;

    std::vector<double> longitudes
            = linspace(0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes
            = linspace(0.0, std::numbers::pi, num_lat);
    std::vector<double> grid(num_lat*num_lon);
    
    std::ranges::fill(m_ffts, std::complex<double>{});
    for (std::size_t i = 0; i < num_lat; ++i)
    {
        std::span<std::complex<double>> fft(
                m_ffts.begin() + i*num_fft, num_fft);
        PlmSpan<double, SHNorm::GEO, SHPhase::NONE> plm(
                m_plm, expansion.lmax());
        
        const double z = std::cos(colatitudes[i]);
        m_recursion.plm_real(plm, z);
        for (std::size_t l = 0; l <= expansion.lmax(); ++l)
        {
            auto plm_l = plm[l];
            auto exp_l = expansion[l];
            fft[0] += plm_l[0]*std::complex<double>{exp_l[0][0], -exp_l[0][1]};
            for (std::size_t m = 1; m <= l; ++m)
                fft[m] += (0.5*plm_l[m])*std::complex<double>{exp_l[m][0], -exp_l[m][1]};
        }
    }

    constexpr std::size_t axis = 1;
    constexpr double prefactor = 1.0;
    pocketfft::c2r(
        m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid, axis, pocketfft::BACKWARD, m_ffts.data(),
        grid.data(), prefactor);
    
    for (std::size_t i = 0; i < num_lat; ++i)
        grid.data()[i*num_lon + num_lon - 1] = grid.data()[i*num_lon];
    
    return {longitudes, colatitudes, grid};
}

}
}