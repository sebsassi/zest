#pragma once

#include "linearfit.hpp"
#include "real_sh_expansion.hpp"
#include "real_ylm.hpp"

namespace zest
{
namespace st
{

/*
Least-squares real spherical harmonic expansion fit on arbitrary real valued data on the sphere.
*/
class LSQTransformer
{
public:
    LSQTransformer(): LSQTransformer(0) {}
    explicit LSQTransformer(std::size_t lmax);

    [[nodiscard]] std::size_t lmax() const noexcept
    {
        return m_ylm_gen.lmax();
    }

    [[nodiscard]] const Matrix<double>& sh_values() const noexcept
    {
        return m_sh_values;
    }

    void transform(
        RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, std::span<const double> data, std::span<const double> lat, std::span<const double> lon);
    
    RealSHExpansion<SHNorm::GEO, SHPhase::NONE> transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon, std::size_t lmax);

private:
    RealYlmGenerator m_ylm_gen;
    Matrix<double> m_sh_values;
    std::vector<double> m_coeffs;
    LinearMultifit m_fitter;
};

}
}