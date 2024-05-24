#pragma once

#include "plm_recursion.hpp"
#include "triangle_layout.hpp"

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

template <
    SHNorm NORM, SHPhase PHASE, real_ylm_packing PackingType>
    requires std::same_as<typename PackingType::Element, std::array<double, 2>>
        || std::same_as<typename PackingType::Element, std::complex<double>>
class RealYlm
{
public:
    using Packing = PackingType;
    using Layout = typename PackingType::Layout;
    using Element = typename PackingType::Element;
    using IndexType = Layout::IndexType;
    using View = RealYlmSpan<Element, NORM, PHASE>;
    using ConstView = RealYlmSpan<const Element, NORM, PHASE>;

    RealYlm(): RealYlm(0) {}
    explicit RealYlm(std::size_t lmax):
        m_coeffs(Layout::size(lmax)), m_lmax(lmax) {}

    operator View()
    {
        return View(m_coeffs, m_lmax);
    };

    operator ConstView() const
    {
        return ConstView(m_coeffs, m_lmax);
    };
    
    [[nodiscard]] Element operator()(IndexType l, IndexType m) const noexcept
    {
        return m_coeffs[Layout::idx(l,m)];
    }
    Element& operator()(IndexType l, IndexType m)
    {
        return m_coeffs[Layout::idx(l,m)];
    }

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<const Element> coeffs() const noexcept
    {
        return m_coeffs;
    }

    std::span<Element> coeffs() noexcept { return m_coeffs; }

    void resize(std::size_t lmax)
    {
        m_lmax = lmax;
        m_coeffs.resize(Layout::size(lmax));
    }

private:
    std::vector<Element> m_coeffs;
    std::size_t m_lmax;
};

/*
Generation of real spherical harmonics based on recursion of associated Legendre polynomials.
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

    /*
    Generate spherical harmonics at longitude and latitude values `lon`, `lat`
    */
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

}
}