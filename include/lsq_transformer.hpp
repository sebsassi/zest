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

    template <SHNorm NORM, SHPhase PHASE>
    void transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon, RealSHExpansionSpan<std::array<double, 2>, NORM, PHASE> expansion)
    {
        using Expansion = RealSHExpansionSpan<std::array<double, 2>, NORM, PHASE>;

        m_ylm_gen.expand(expansion.order());

        m_sh_values.resize(
                data.size(), DualTriangleLayout::size(expansion.order()));

        for (size_t i = 0; i < data.size(); ++i)
        {
            RealYlmSpan<SequentialRealYlmPacking, NORM, PHASE> ylm(m_sh_values.row(i), expansion.order());
            m_ylm_gen.generate<SequentialRealYlmPacking, NORM, PHASE>(lon[i], lat[i], ylm);
        }

        m_coeffs.resize(m_sh_values.ncols());
        m_fitter.fit_parameters(m_sh_values, m_coeffs, data);

        std::span coeffs = expansion.flatten();
        for (std::size_t l = 0; l < expansion.order(); ++l)
        {
            coeffs[Expansion::Layout::idx(l, 0)] = {
                m_coeffs[DualTriangleLayout::idx(int(l), 0)],
                0.0
            };
            for (std::size_t m = 1; m <= l; ++m)
            {
                coeffs[Expansion::Layout::idx(l,m)] = {
                        m_coeffs[DualTriangleLayout::idx(int(l),int(m))],
                        m_coeffs[DualTriangleLayout::idx(int(l),-int(m))]
                    };
            }
        }
    }
    
    template <SHNorm NORM, SHPhase PHASE>
    RealSHExpansion<NORM, PHASE> transform(
        std::span<const double> data, std::span<const double> lat, std::span<const double> lon, std::size_t order)
    {
        RealSHExpansion<NORM, PHASE> expansion(order);
        transform<NORM, PHASE>(data, lat, lon, expansion);
        return expansion;
    }

private:
    RealYlmGenerator m_ylm_gen;
    Matrix<double> m_sh_values;
    std::vector<double> m_coeffs;
    detail::LinearMultifit m_fitter;
};

} // namespace st
} // namespace zest