#pragma once

#include "linearfit/linearfit.hpp"
#include "plm_recursion.hpp"
#include "real_sh_expansion.hpp"

namespace zest
{
namespace st
{

/*
Pack real spherical harmonics in pairs `(Y(l,m), Y(l,-m))` indexed by `0 <= m <= l`.
*/
struct PairedRealYlmPacking
{
    using Layout = TriangleLayout;
    using Element = std::array<double, 2>;
};

/*
Pack real spherical harmonics sequentially indexed by `-l <= m <= l`.
*/
struct SequentialRealYlmPacking
{
    using Layout = DualTriangleLayout;
    using Element = double;
};

/*
Constrain to possible packings of real spherical harmonics.

Requires `T` to be one of:
`PairedRealYlmPacking`
`SequentialRealYlmPacking`
*/
template <typename T>
concept real_ylm_packing
        = std::same_as<T, PairedRealYlmPacking>
        || std::same_as<T, SequentialRealYlmPacking>;

/*
Non-owfning view of eral spherical harmonics.
*/
template <real_ylm_packing T, SHNorm NORM, SHPhase PHASE>
using RealYlmSpan = SHLMSpan<typename T::Element, typename T::Layout, NORM, PHASE>;

/*
Class for recursive generation of real spherical harmonics
*/
class RealYlmGenerator
{
public:
    RealYlmGenerator(): RealYlmGenerator(0) {}
    explicit RealYlmGenerator(std::size_t lmax);

    [[nodiscard]] std::size_t lmax() const noexcept
    {
        return m_recursion.lmax();
    }

    void expand(std::size_t lmax);

    template <real_ylm_packing T, SHNorm NORM, SHPhase PHASE>
    void generate(
        RealYlmSpan<T, NORM, PHASE> ylm, double lon, double lat)
    {
        using IndexType = RealYlmSpan<T, NORM, PHASE>::IndexType;
        expand(ylm.lmax());

        const double z = std::sin(lat);
        m_recursion.plm_real(
                PlmSpan<double, NORM, PHASE>(m_ass_leg_poly, ylm.lmax()), z);

        for (std::size_t m = 0; m <= ylm.lmax(); ++m)
        {
            const double angle = double(m)*lon;
            m_cossin[m] = {std::cos(angle), std::sin(angle)};
        }

        for (std::size_t l = 0; l <= ylm.lmax(); ++l)
        {
            const double ass_leg_poly
                    = m_ass_leg_poly[TriangleLayout::idx(l, 0)];
            if constexpr (std::is_same_v<T, SequentialRealYlmPacking>)
                ylm(IndexType(l), 0) = ass_leg_poly;
            else if constexpr (std::is_same_v<T, PairedRealYlmPacking>)
            {
                ylm(IndexType(l), 0)[0] = ass_leg_poly;
                ylm(IndexType(l), 0)[1] = 0.0;
            }
            
            for (std::size_t m = 1; m <= l; ++m)
            {
                const double ass_leg_poly
                        = m_ass_leg_poly[TriangleLayout::idx(l, m)];
                
                if constexpr (std::is_same_v<T, SequentialRealYlmPacking>)
                {
                    ylm(int(l), int(m)) = ass_leg_poly*m_cossin[m][0];
                    ylm(int(l), -int(m)) = ass_leg_poly*m_cossin[m][1];
                }
                else if constexpr (std::is_same_v<T, PairedRealYlmPacking>)
                {
                    ylm(l, m) = {
                        ass_leg_poly*m_cossin[m][0],
                        ass_leg_poly*m_cossin[m][1]
                    };
                }
            }
        }
    }

private:
    PlmRecursion m_recursion;
    std::vector<double> m_ass_leg_poly;
    std::vector<std::array<double, 2>> m_cossin;
};

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