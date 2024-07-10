#pragma once

#include <span>
#include <vector>

#include "linearfit.hpp"
#include "real_sh_expansion.hpp"
#include "real_ylm.hpp"

namespace zest
{
namespace st
{

/**
    @brief Least-squares real spherical harmonic expansion fit on arbitrary real valued data on the sphere.
*/
class LSQTransformer
{
public:
    LSQTransformer() = default;
    explicit LSQTransformer(std::size_t order);

    [[nodiscard]] std::size_t order() const noexcept
    {
        return m_ylm_gen.max_order();
    }

    [[nodiscard]] const Matrix<double>& sh_values() const noexcept
    {
        return m_sh_values;
    }

    void transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon, RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion);
    
    RealSHExpansion<SHNorm::GEO, SHPhase::NONE> transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon, std::size_t order);

private:
    RealYlmGenerator m_ylm_gen;
    Matrix<double> m_sh_values;
    std::vector<double> m_coeffs;
    detail::LinearMultifit m_fitter;
};

} // namespace st
} // namespace zest