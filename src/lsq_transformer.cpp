#include "lsq_transformer.hpp"

namespace zest
{
namespace st
{

LSQTransformer::LSQTransformer(std::size_t order):
    m_ylm_gen(order), m_sh_values(), m_fitter() {}

void LSQTransformer::transform(
    RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, std::span<const double> data, std::span<const double> lat, std::span<const double> lon)
{
    using Expansion = RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE>;

    m_ylm_gen.expand(expansion.order());

    m_sh_values.resize(
            data.size(), DualTriangleLayout::size(expansion.order()));

    for (size_t i = 0; i < data.size(); ++i)
    {
        RealYlmSpan<SequentialRealYlmPacking, SHNorm::GEO, SHPhase::NONE> ylm(m_sh_values.row(i), expansion.order());
        m_ylm_gen.generate<SequentialRealYlmPacking, SHNorm::GEO, SHPhase::NONE>(ylm, lon[i], lat[i]);
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

RealSHExpansion<SHNorm::GEO, SHPhase::NONE> LSQTransformer::transform(
    std::span<const double> data, std::span<const double> lat, std::span<const double> lon, std::size_t order)
{
    RealSHExpansion<SHNorm::GEO, SHPhase::NONE> expansion(order);
    transform(expansion, data, lat, lon);
    return expansion;
}

}
}