#pragma once

#include <span>
#include <complex>

#include "alignment.hpp"
#include "plm_recursion.hpp"
#include "gauss_legendre.hpp"

#include "pocketfft.hpp"

namespace zest
{
namespace st
{

/*
Longitudinally contiguous layout for storing a Gauss-Legendre quadrature grid.
*/
template <typename AlignmentType = CacheLineAlignment>
struct LatLonLayout
{
    using Alignment = AlignmentType;

    [[nodiscard]] static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return lat_size(lmax)*lon_size(lmax);
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    shape(std::size_t lmax) noexcept
    {
        return {lat_size(lmax), lon_size(lmax)};
    }

    [[nodiscard]] static constexpr std::size_t
    fft_size(std::size_t lmax) noexcept
    {
        return (lon_size(lmax) >> 1) + 1;
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    fft_stride(std::size_t lmax) noexcept
    {
        return {fft_size(lmax), 1};
    }

    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t lmax) noexcept
    {
        return lmax + 1UL;
    }

    [[nodiscard]] static constexpr std::size_t
    lon_size(std::size_t lmax) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = 2UL*lmax + 1UL;
        if constexpr (std::is_same_v<Alignment, NoAlignment>)
            return min_size;
        else
            return (min_size & (~(vector_size - 1UL))) + vector_size;
    }

    static constexpr std::size_t lat_axis = 0UL;
    static constexpr std::size_t lon_axis = 1UL;
};

/*
Latitudinally contiguous layout for storing a Gauss-Legendre quadrature grid.
*/
template <typename AlignmentType = CacheLineAlignment>
struct LonLatLayout
{
    using Alignment = AlignmentType;

    [[nodiscard]] static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return lat_size(lmax)*lon_size(lmax);
    }
    
    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    shape(std::size_t lmax) noexcept
    {
        return {lon_size(lmax), lat_size(lmax)};
    }

    [[nodiscard]] static constexpr std::size_t
    fft_size(std::size_t lmax) noexcept
    {
        return (lon_size(lmax) >> 1) + 1;
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 2>
    fft_stride(std::size_t lmax)
    {
        return {lat_size(lmax), 1};
    }

    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t lmax) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = lmax + 1UL;
        if constexpr (std::is_same_v<Alignment, NoAlignment>)
            return min_size;
        else
            return (min_size & (~(vector_size - 1UL))) + vector_size;
    }

    [[nodiscard]] static constexpr std::size_t
    lon_size(std::size_t lmax) noexcept
    {
        return 2UL*lmax + 1UL;
    }

    static constexpr std::size_t lat_axis = 1UL;
    static constexpr std::size_t lon_axis = 0UL;
};

using DefaultLayout = LonLatLayout<>;

/*
A non-owning view on data modeling a Gauss-Legendre quadrature grid on the sphere.
*/
template <typename Element, typename LayoutType = DefaultLayout>
    requires std::same_as<std::remove_const_t<Element>, double>
        || std::same_as<std::remove_const_t<Element>, std::complex<double>>
class SphereGLQGridSpan
{
public:
    using element_type = Element;
    using value_type = std::remove_cv_t<Element>;
    using size_type = std::size_t;
    using Layout = LayoutType;

    SphereGLQGridSpan(std::span<element_type> buffer, std::size_t lmax):
        m_values(buffer.begin(), Layout::size(lmax)), m_shape(Layout::shape(lmax)), m_lmax(lmax) {}
    SphereGLQGridSpan(std::span<element_type> buffer, std::size_t idx, std::size_t lmax):
        m_values(buffer.begin() + idx*Layout::size(lmax), Layout::size(lmax)), m_shape(Layout::shape(lmax)), m_lmax(lmax) {}

    [[nodiscard]] std::array<std::size_t, 2>
    shape() const noexcept { return m_shape; }

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }

    [[nodiscard]] std::span<const element_type>
    flatten() const noexcept { return m_values; }

    std::span<element_type> flatten() noexcept { return m_values; }

    [[nodiscard]] element_type
    operator()(std::size_t i, std::size_t j) const noexcept
    {
        return m_values[m_shape[1]*i + j];
    }

    element_type& operator()(std::size_t i, std::size_t j) noexcept
    {
        return m_values[m_shape[1]*i + j];
    }

private:
    std::span<element_type> m_values;
    std::array<std::size_t, 2> m_shape;
    std::size_t m_lmax;
};

/*
Container for Gauss-Legendre quadrature gridded data on the sphere.
*/
template <typename Element, typename LayoutType = DefaultLayout>
    requires std::same_as<std::remove_const_t<Element>, double>
        || std::same_as<std::remove_const_t<Element>, std::complex<double>>
class SphereGLQGrid
{
public:
    using element_type = Element;
    using Layout = LayoutType;
    using View = SphereGLQGridSpan<element_type, Layout>;
    using ConstView = SphereGLQGridSpan<const element_type, Layout>;

    SphereGLQGrid(): SphereGLQGrid(0) {}
    explicit SphereGLQGrid(std::size_t lmax):
        m_values(Layout::size(lmax)), m_shape(Layout::shape(lmax)), 
        m_lmax(lmax) {}

    [[nodiscard]] std::array<std::size_t, 2>
    shape() const noexcept { return m_shape; }

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }

    [[nodiscard]] std::span<const element_type>
    flatten() const noexcept { return m_values; }

    std::span<element_type> flatten() noexcept { return m_values; }

    operator View()
    {
        return View(m_values, m_lmax);
    };

    operator ConstView() const
    {
        return ConstView(m_values, m_lmax);
    };

    void resize(std::size_t lmax)
    {
        m_values.resize(Layout::size(lmax));
        m_shape = Layout::shape(lmax);
        m_lmax = lmax;
    }

    [[nodiscard]] element_type
    operator()(std::size_t i, std::size_t j) const noexcept
    {
        return m_values[m_shape[1]*i + j];
    }

    element_type& operator()(std::size_t i, std::size_t j) noexcept
    {
        return m_values[m_shape[1]*i + j];
    }

private:
    using Allocator = AlignedAllocator<element_type, LayoutType::Alignment::byte_alignment>;

    std::vector<element_type, Allocator> m_values;
    std::array<std::size_t, 2> m_shape;
    std::size_t m_lmax;
};

/*
Points defining a Gauss-Legendre quadrature grid on the sphere.
*/
class SphereGLQGridPoints
{
public:
    SphereGLQGridPoints() = default;

    [[nodiscard]] std::array<std::size_t, 2> shape() noexcept
    {
        return {m_glq_nodes.size(), m_longitudes.size()};
    }

    [[nodiscard]] std::span<const double> longitudes() const noexcept
    {
        return m_longitudes;
    }
    [[nodiscard]] std::span<const double> glq_nodes() const noexcept
    {
        return m_glq_nodes;
    }

    void resize(std::size_t num_lon, std::size_t num_lat)
    {
        if (num_lon != m_longitudes.size())
        {
            m_longitudes.resize(num_lon);
            const double dlon = (2.0*std::numbers::pi)/double(m_longitudes.size());
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
                m_longitudes[i] = dlon*double(i);
        }
        if (num_lat != m_glq_nodes.size())
        {
            m_glq_nodes.resize(num_lat);
            gl::gl_nodes<double, gl::GLLayout::UNPACKED, gl::GLNodeStyle::ANGLE>(m_glq_nodes, m_glq_nodes.size());
        }
    }

    template <typename LayoutType, typename FuncType>
    void generate_values(SphereGLQGridSpan<typename std::invoke_result<FuncType, double, double>::type, LayoutType> grid, FuncType&& f)
    {
        constexpr std::size_t lon_axis = decltype(grid)::Layout::lon_axis;
        constexpr std::size_t lat_axis = decltype(grid)::Layout::lat_axis;
        const auto shape = grid.shape();
        resize(shape[lon_axis], shape[lat_axis]);
        if constexpr (lon_axis == 1)
        {
            for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            {
                const double colatitude = m_glq_nodes[i];
                for (std::size_t j = 0; j < m_longitudes.size(); ++j)
                {
                    const double lon = m_longitudes[j];
                    grid(i,j) = f(lon, colatitude);
                }
            }
        }
        else if constexpr (lon_axis == 0)
        {
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
            {
                const double lon = m_longitudes[i];
                for (std::size_t j = 0; j < m_glq_nodes.size(); ++j)
                {
                    const double colatitude = m_glq_nodes[j];
                    grid(i,j) = f(lon, colatitude);
                }
            }
        }
    }

    template <typename Layout = DefaultLayout, typename FuncType>
    auto generate_values(FuncType&& f, std::size_t lmax)
    {
        using CodomainType = std::invoke_result<FuncType, double, double>::type;
        SphereGLQGrid<CodomainType, Layout> grid(lmax);
        generate_values<Layout, FuncType>(grid, f);
        return grid;
    }

private:
    std::vector<double> m_longitudes;
    std::vector<double> m_glq_nodes;
};

/*
Transformations between a Gauss-Legendre quadrature grid representation and spherical harmonic expansion representation of real data.
*/
template <typename GridLayoutType = DefaultLayout>
class GLQTransformer
{
public:
    using GridLayout = GridLayoutType;
    GLQTransformer(): GLQTransformer(0) {}
    explicit GLQTransformer(std::size_t lmax):
        m_recursion(lmax), m_glq_nodes(GridLayout::lat_size(lmax)),
        m_glq_weights(GridLayout::lat_size(lmax)),
        m_plm_grid(GridLayout::lat_size(lmax)*TriangleLayout::size(lmax)),
        m_ffts(GridLayout::lat_size(lmax)*GridLayout::fft_size(lmax)), m_symm_asymm(GridLayout::fft_size(lmax)*((GridLayout::lat_size(lmax) + 1) >> 1)*2),
        m_pocketfft_shape_grid(2), m_pocketfft_stride_grid(2), m_pocketfft_stride_fft(2),
        m_grids{SphereGLQGrid<double, GridLayout>(lmax), SphereGLQGrid<double, GridLayout>(lmax)},
        m_lmax(lmax)
    {
        gl::gl_nodes_and_weights<double, gl::GLLayout::PACKED, gl::GLNodeStyle::COS>(
                m_glq_nodes, m_glq_weights, GridLayout::lat_size(lmax));
        m_plm_grid.resize(m_glq_weights.size()*TriangleLayout::size(lmax));
        
        if constexpr (GridLayout::lon_axis == 1)
        {
            for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            {
                const double z = m_glq_nodes[i];
                PlmSpan<double, SHNorm::GEO, SHPhase::NONE> plm(m_plm_grid, i, m_lmax);
                m_recursion.plm_real(plm, z);
            }
        }
        else if constexpr (GridLayout::lon_axis == 0)
        {
            PlmVecSpan<double, SHNorm::GEO, SHPhase::NONE> plm(m_plm_grid, m_lmax, m_glq_nodes.size());
            m_recursion.plm_real(plm, m_glq_nodes);
        }

        auto shape = GridLayout::shape(lmax);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];

        m_pocketfft_stride_grid[0] = long(shape[1]*sizeof(double));
        m_pocketfft_stride_grid[1] = sizeof(double);

        auto fft_stride = GridLayout::fft_stride(lmax);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = fft_stride[1]*sizeof(std::complex<double>);
    }

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] SHPhase phase() const noexcept { return m_phase; }

    /*
    Resize transformer for specified expansion order `lmax`
    */
    void resize(std::size_t lmax)
    {
        if (lmax == this->lmax()) return;

        m_recursion.expand(lmax);

        gl::gl_nodes_and_weights<double, gl::GLLayout::PACKED, gl::GLNodeStyle::COS>(
                m_glq_nodes, m_glq_weights, lmax + 1);
        m_plm_grid.resize(m_glq_weights.size()*TriangleLayout::size(lmax));
        
        m_plm_grid.resize(m_glq_weights.size()*TriangleLayout::size(lmax));
        if constexpr (GridLayout::lon_axis == 1)
        {
            for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
            {
                const double z = m_glq_nodes[i];
                PlmSpan<double, SHNorm::GEO, SHPhase::NONE> plm(m_plm_grid, i, m_lmax);
                m_recursion.plm_real(plm, z);
            }
        }
        else if constexpr (GridLayout::lon_axis == 0)
        {
            PlmVecSpan<double, SHNorm::GEO, SHPhase::NONE> plm(m_plm_grid, m_lmax, m_glq_nodes.size());
            m_recursion.plm_real(plm, m_glq_nodes);
        }

        m_ffts.resize(GridLayout::lat_size(lmax)*GridLayout::fft_size(lmax));
        m_symm_asymm.resize(GridLayout::fft_size(lmax)*((GridLayout::lat_size(lmax) + 1) >> 1)*2);

        auto shape = GridLayout::shape(lmax);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];

        m_pocketfft_stride_grid[0] = long(shape[1]*sizeof(double));
        m_pocketfft_stride_grid[1] = sizeof(double);

        auto fft_stride = GridLayout::fft_stride(lmax);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = fft_stride[1]*sizeof(std::complex<double>);

        m_grids.first.resize(lmax);
        m_grids.second.resize(lmax);

        m_lmax = lmax;
    }

    /*
    Forward transform from Gauss-Legendre quadrature grid to spherical harmonic coefficients.

    Parameters:
    `values`: values on the spherical quadrature grid.
    `expansion`: coefficients of the expansion.
    */
    void forward_transform(
        SphereGLQGridSpan<const double, GridLayout> values,
        RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion)
    {
        resize(values.lmax());
        
        integrate_longitudinal(values);

        fft_to_symm_asymm();

        std::size_t min_lmax = std::min(expansion.lmax(), values.lmax());
        integrate_latitudinal(expansion, min_lmax);
    }

    /*
    Backward transform from spherical harmonic coefficients to Gauss-Legendre quadrature grid.

    Parameters:
    `values`: values on the spherical quadrature grid.
    `expansion`: coefficients of the expansion.
    */
    void backward_transform(
        RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion,
        SphereGLQGridSpan<double, GridLayout> values)
    {
        resize(values.lmax());

        std::size_t min_lmax = std::min(expansion.lmax(), values.lmax());
        
        sum_l(expansion, min_lmax);

        symm_asymm_to_fft();

        sum_m(values);
    }
    
    /*
    Forward transform from Gauss-Legendre quadrature grid to spherical harmonic coefficients.

    Parameters:
    `values`: values on the spherical quadrature grid.
    `lmax: order of expansion.
    */
    RealSHExpansion<SHNorm::GEO, SHPhase::NONE>
    forward_transform(
        SphereGLQGridSpan<const double, GridLayout> values, std::size_t lmax)
    {
        RealSHExpansion<SHNorm::GEO, SHPhase::NONE> expansion(lmax);
        forward_transform(values, expansion);
        return expansion;
    }

    /*
    Backward transform from spherical harmonic coefficients to Gauss-Legendre quadrature grid.

    Parameters:
    `values`: values on the spherical quadrature grid.
    `expansion`: coefficients of the expansion.
    */
    SphereGLQGrid<double, GridLayout> backward_transform(
        RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, std::size_t lmax)
    {
        SphereGLQGrid<double, GridLayout> grid(lmax);
        backward_transform(expansion, grid);
        return grid;
    }

    /*
    Compute coefficients of the product of two spherical harmonic expansions.
    */
    void multiply(
        RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> a,
        RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> b,
        RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> out)
    {
        if (out.lmax() != a.lmax() + b.lmax())
            throw std::invalid_argument(
                    "lmax of out is not equal to the sum of lmax of inputs");

        resize(a.lmax() + b.lmax());
        backward_transform(a, m_grids.first);
        backward_transform(b, m_grids.second);

        for (std::size_t i = 0; i < m_grids.first.values().size(); ++i)
            m_grids.first.values()[i] *= m_grids.second.values()[i];
        
        forward_transform(m_grids.first, out);
    }

private:
    void integrate_longitudinal(
        SphereGLQGridSpan<const double, GridLayout> values)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr double sh_normalization = 1.0/(4.0*std::numbers::pi);
        const double prefactor = sh_normalization*(2.0*std::numbers::pi)/double(values.shape()[lon_axis]);
        pocketfft::r2c(
            m_pocketfft_shape_grid, m_pocketfft_stride_grid, m_pocketfft_stride_fft, lon_axis, pocketfft::FORWARD, values.values().data(), m_ffts.data(), prefactor);
    }

    void apply_gl_weights() noexcept
    {
        const std::size_t num_lat = m_glq_weights.size();
        const std::size_t fft_order = m_glq_weights.size();
        if constexpr (GridLayout::lon_axis == 1)
        {
            for (std::size_t i = 0; i < num_lat; ++i)
            {
                const double weight = m_glq_weights[i];
                for (std::size_t m = 0; m < fft_order; ++m)
                {
                    std::complex<double>& x = m_ffts[fft_order*i + m];
                    x = {weight*x.real(), -weight*x.imag()};
                }
            }
        }
        else if constexpr (GridLayout::lon_axis == 0)
        {
            for (std::size_t m = 0; m < fft_order; ++m)
            {
                for (std::size_t i = 0; i < num_lat; ++i)
                {
                    const double weight = m_glq_weights[i];
                    std::complex<double>& x = m_ffts[num_lat*m + i];
                    x = {weight*x.real(), -weight*x.imag()};
                }
            }
        }
    }

    /*
    Apply Gauss-Legendre weights and divide Fourier transforms into symmetric and antisymmetric parts, `f(x) + f(-x)` and `f(x) - f(-x)`, where `x` are the Legendre nodes.

    This division is useful to reduce operation count in the latitudinal integration stage due to symmetry properties of the associated legendre functions. Namely, `P_lm(-x) = (-1)^(l+m)*P_lm(x)`. This means that for `l + m` even, the integration is over the symmetric parts, and for `l + m` odd, it is over the antisymmetric parts.

    The final symmetric and antisymmetric parts have a very specific memory layout. Given number of latitudes `num_lat`, there are `(num_lat + 1)/2` symmetric and antisymmetric components. These are denoted `+` and `-` for symmetric and antisymmetric, respectively. The layout differs in an expected way depending on whether `GridLayout` is latitude or longitude major. Namely, for latitude major order `+` and `-` refer to blocks of `(num_lat + 1)/2` complex numbers, whereas for longitude major order they refer to individual complex numbers.
    
    Given `m` denoting the order of the Fourier transform, these components are then stored in memory as two alternating sequences:
    `m 0 1 2 3 4 5 6 7 8 ...`
    `s + - + - + - + - + ...`
    `a - + - + - + - + - ...`
    In the latitudinal integration step, for even `l` the sequence `s` starting with the symmetric components is chosen, and for odd `l` the sequence `a` starting with the antisymmetric components is chosen. This leads to a moderately cache efficient access pattern in the latitudinal integration step.
    */
    void fft_to_symm_asymm() noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_lmax);
        const std::size_t num_lat = GridLayout::lat_size(m_lmax);
        const std::size_t central_offset = num_lat >> 1;
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t south_offset = num_unique_nodes - 1;
        const std::size_t north_offset = central_offset;

        if constexpr (GridLayout::lon_axis == 1)
        {
            for (std::size_t i = 0; i < num_unique_nodes; ++i)
            {
                const double weight = m_glq_weights[i];
                std::span<const std::complex<double>> fft_south(
                    m_ffts.begin() + (south_offset - i)*fft_order, fft_order);
                std::span<const std::complex<double>> fft_north(
                    m_ffts.begin() + (north_offset + i)*fft_order, fft_order);
                
                std::span<std::complex<double>> symm_asymm_i(
                    m_symm_asymm.begin() + 2*i*fft_order, fft_order);
                std::span<std::complex<double>> asymm_symm_i(
                    m_symm_asymm.begin() + (2*i + 1)*fft_order, fft_order);

                for (std::size_t m = 0; m < fft_order; ++m)
                {
                    const double sign = (m & 1) ? -1.0 : 1.0;
                    const std::complex<double> south = fft_south[m];
                    const std::complex<double> north = fft_north[m];
                    const std::complex<double> south_weighted = {
                        weight*south.real(), -weight*south.imag()
                    };
                    const std::complex<double> north_weighted = {
                        weight*north.real(), -weight*north.imag()
                    };
                    symm_asymm_i[m] = north_weighted + sign*south_weighted;
                    asymm_symm_i[m] = north_weighted - sign*south_weighted;
                }
            }
        }
        if constexpr (GridLayout::lon_axis == 0)
        {
            std::span<std::complex<double>> symm_asymm(
                m_symm_asymm.begin(), num_unique_nodes*fft_order);
            std::span<std::complex<double>> asymm_symm(
                m_symm_asymm.begin() + num_unique_nodes*fft_order, num_unique_nodes*fft_order);

            for (std::size_t m = 0; m < fft_order; ++m)
            {
                const double sign = (m & 1) ? -1.0 : 1.0;
                std::span<const std::complex<double>> fft_m(
                    m_ffts.begin() + num_lat*m, num_lat);

                std::span<std::complex<double>> symm_asymm_m(
                    symm_asymm.begin() + num_unique_nodes*m, num_unique_nodes);
                std::span<std::complex<double>> asymm_symm_m(
                    asymm_symm.begin() + num_unique_nodes*m, num_unique_nodes);
                    
                for (std::size_t i = 0; i < num_unique_nodes; ++i)
                {
                    const double weight = m_glq_weights[i];
                    const std::complex<double> south = fft_m[south_offset - i];
                    const std::complex<double> north = fft_m[north_offset + i];
                    const std::complex<double> south_weighted = {
                        weight*south.real(), -weight*south.imag()
                    };
                    const std::complex<double> north_weighted = {
                        weight*north.real(), -weight*north.imag()
                    };
                    symm_asymm_m[i] = north_weighted + sign*south_weighted;
                    asymm_symm_m[i] = north_weighted - sign*south_weighted;
                }
            }
        }
    }

    void integrate_latitudinal(
        RealSHExpansionSpan<std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, std::size_t min_lmax) noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_lmax);
        const std::size_t num_unique_nodes = m_glq_weights.size();

        std::span coeffs = expansion.flatten();
        std::ranges::fill(coeffs, std::array<double, 2>{});
        if constexpr (GridLayout::lon_axis == 1)
        {
            for (std::size_t i = 0; i < num_unique_nodes; ++i)
            {
                PlmSpan<const double, SHNorm::GEO, SHPhase::NONE> plm(m_plm_grid, i, lmax());
                std::span plm_flat = plm.flatten();
                for (std::size_t l = 0; l <= min_lmax; ++l)
                {
                    std::span<const std::complex<double>> fft(
                        m_symm_asymm.begin() + (2*i + (l & 1))*fft_order, fft_order);
                    for (std::size_t m = 0; m <= l; ++m)
                    {
                        const std::size_t ind = TriangleLayout::idx(l,m);
                        coeffs[ind][0] += plm_flat[ind]*fft[m].real();
                        coeffs[ind][1] += plm_flat[ind]*fft[m].imag();
                    }
                }
            }
        }
        else if constexpr (GridLayout::lon_axis == 0)
        {
            const std::size_t num_plm = num_unique_nodes;
            for (std::size_t l = 0; l <= min_lmax; ++l)
            {
                std::span<const std::complex<double>> ffts;
                ffts = std::span<const std::complex<double>>(
                    m_symm_asymm.begin() + (l & 1)*num_plm*fft_order, num_plm*fft_order);
                for (std::size_t m = 0; m <= l; ++m)
                {
                    const std::size_t ind = TriangleLayout::idx(l,m);
                    std::span<const double> plm(
                        m_plm_grid.begin() + ind*num_plm, num_plm);
                    std::span<const std::complex<double>> fft(
                        ffts.begin() + m*num_plm, num_plm);
                    
                    std::array<double, 2> coeff{};
                    switch (num_plm & 3)
                    {
                        case 1:
                            coeff[0] = plm[0]*fft[0].real();
                            coeff[1] = plm[0]*fft[0].imag();
                            break;
                        case 2:
                            coeff[0] = plm[0]*fft[0].real() + plm[1]*fft[1].real();
                            coeff[1] = plm[0]*fft[0].imag() + plm[1]*fft[1].imag();
                            break;
                        case 3:
                            coeff[0] = plm[0]*fft[0].real() + plm[1]*fft[1].real() + plm[2]*fft[2].real();
                            coeff[1] = plm[0]*fft[0].imag() + plm[1]*fft[1].imag() + plm[2]*fft[2].imag();
                            break;
                    }

                    /*for (std::size_t i = 0; i < (num_lat & 3); ++i)
                    {
                        coeff[0] += plm[i]*fft[i].real();
                        coeff[1] += plm[i]*fft[i].imag();
                    }*/

                    std::array<double, 8> partial_sum{};
                    for (std::size_t i = (num_plm & 3); i < num_plm; i += 4)
                    {
                        partial_sum[0] += plm[i]*fft[i].real();
                        partial_sum[1] += plm[i]*fft[i].imag();
                        partial_sum[2] += plm[i + 1]*fft[i + 1].real();
                        partial_sum[3] += plm[i + 1]*fft[i + 1].imag();
                        partial_sum[4] += plm[i + 2]*fft[i + 2].real();
                        partial_sum[5] += plm[i + 2]*fft[i + 2].imag();
                        partial_sum[6] += plm[i + 3]*fft[i + 3].real();
                        partial_sum[7] += plm[i + 3]*fft[i + 3].imag();
                    }

                    for (std::size_t i = 0; i < 8; i += 2)
                    {
                        coeff[0] += partial_sum[i];
                        coeff[1] += partial_sum[i + 1];
                    }

                    coeffs[ind] = coeff;
                }
            }
        }
    }

    void sum_l(
        RealSHExpansionSpan<const std::array<double, 2>, SHNorm::GEO, SHPhase::NONE> expansion, std::size_t min_lmax) noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_lmax);
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t num_plm = num_unique_nodes;

        std::span coeffs = expansion.flatten();
        std::ranges::fill(m_symm_asymm, std::complex<double>{});

        if constexpr (GridLayout::lon_axis == 1)
        {
            for (std::size_t i = 0; i < num_plm; ++i)
            {
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + 2*i*fft_order, 2*fft_order);
                PlmSpan<const double, SHNorm::GEO, SHPhase::NONE> plm(m_plm_grid, i, lmax());
                std::span plm_flat = plm.flatten();
                for (std::size_t l = 0; l <= min_lmax; ++l)
                {
                    const std::size_t ind = TriangleLayout::idx(l, 0);
                    symm_asymm[(l & 1)*fft_order] += std::complex<double>{
                        plm_flat[ind]*coeffs[ind][0],
                        -plm_flat[ind]*coeffs[ind][1]
                    };
                    for (std::size_t m = 1; m <= l; ++m)
                    {
                        const std::size_t ind = TriangleLayout::idx(l, m);
                        const double weight = 0.5*plm_flat[ind];
                        symm_asymm[(l & 1)*fft_order + m]
                            += std::complex<double>{
                                weight*coeffs[ind][0],
                                -weight*coeffs[ind][1]
                            };
                    }
                }
            }
        }
        else if constexpr (GridLayout::lon_axis == 0)
        {
            for (std::size_t l = 0; l <= min_lmax; ++l)
            {
                const std::size_t ind = TriangleLayout::idx(l, 0);
                const auto coeff = coeffs[ind];
                std::span<const double> plm(
                    m_plm_grid.begin() + ind*num_plm, num_plm);
                std::span<std::complex<double>> symm_asymm(
                    m_symm_asymm.begin() + (l & 1)*num_plm*fft_order, num_plm);
                for (std::size_t i = 0; i < num_plm; ++i)
                {
                    symm_asymm[i] += std::complex<double>{
                        plm[i]*coeff[0], -plm[i]*coeff[1]
                    };
                }

                for (std::size_t m = 1; m <= l; ++m)
                {
                    const std::size_t ind = TriangleLayout::idx(l, m);
                    const auto coeff = coeffs[ind];
                    std::span<const double> plm(
                        m_plm_grid.begin() + ind*num_plm, num_plm);
                    std::span<std::complex<double>> symm_asymm(
                        m_symm_asymm.begin() + ((l & 1)*fft_order + m)*num_plm, num_plm);
                    for (std::size_t i = 0; i < num_plm; ++i)
                    {
                        const double weight = 0.5*plm[i];
                        symm_asymm[i] += std::complex<double>{
                            weight*coeff[0], -weight*coeff[1]
                        };
                    }
                }
            }
        }
    }

    // Inverse of `fft_to_symm_asymm`
    void symm_asymm_to_fft() noexcept
    {
        const std::size_t fft_order = GridLayout::fft_size(m_lmax);
        const std::size_t num_lat = GridLayout::lat_size(m_lmax);
        const std::size_t central_offset = num_lat >> 1;
        const std::size_t num_unique_nodes = m_glq_weights.size();
        const std::size_t south_offset = num_unique_nodes - 1;
        const std::size_t north_offset = central_offset;

        if constexpr (GridLayout::lon_axis == 1)
        {
            for (std::size_t i = 0; i < num_unique_nodes; ++i)
            {
                std::span<const std::complex<double>> symm_asymm_i(
                    m_symm_asymm.begin() + 2*i*fft_order, fft_order);
                std::span<const std::complex<double>> asymm_symm_i(
                    m_symm_asymm.begin() + (2*i + 1)*fft_order, fft_order);

                std::span<std::complex<double>> fft_south(
                    m_ffts.begin() + (south_offset - i)*fft_order, fft_order);
                std::span<std::complex<double>> fft_north(
                    m_ffts.begin() + (north_offset + i)*fft_order, fft_order);

                for (std::size_t m = 0; m < fft_order; ++m)
                {
                    const double sign = (m & 1) ? -1.0 : 1.0;
                    fft_north[m] = symm_asymm_i[m] + asymm_symm_i[m];
                    fft_south[m] = sign*(symm_asymm_i[m] - asymm_symm_i[m]);
                }
            }
        }
        if constexpr (GridLayout::lon_axis == 0)
        {
            std::span<std::complex<double>> symm_asymm(
                m_symm_asymm.begin(), num_unique_nodes*fft_order);
            std::span<std::complex<double>> asymm_symm(
                m_symm_asymm.begin() + num_unique_nodes*fft_order, num_unique_nodes*fft_order);

            for (std::size_t m = 0; m < fft_order; ++m)
            {
                const double sign = (m & 1) ? -1.0 : 1.0;
                std::span<std::complex<double>> fft_m(
                    m_ffts.begin() + num_lat*m, num_lat);

                std::span<const std::complex<double>> symm_asymm_m(
                    symm_asymm.begin() + num_unique_nodes*m, num_unique_nodes);
                std::span<const std::complex<double>> asymm_symm_m(
                    asymm_symm.begin() + num_unique_nodes*m, num_unique_nodes);
                    
                for (std::size_t i = 0; i < num_unique_nodes; ++i)
                {
                    fft_m[north_offset + i] = symm_asymm_m[i] + asymm_symm_m[i];
                    fft_m[south_offset - i]
                        = sign*(symm_asymm_m[i] - asymm_symm_m[i]);
                }
            }
        }
    }

    void sum_m(SphereGLQGridSpan<double, GridLayout> values)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr double prefactor = 1.0;
        pocketfft::c2r(
            m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid, lon_axis, pocketfft::BACKWARD, m_ffts.data(), values.values().data(), prefactor);
    }

    PlmRecursion m_recursion;
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_plm_grid;
    std::vector<std::complex<double>> m_ffts;
    std::vector<std::complex<double>> m_symm_asymm;
    std::vector<std::size_t> m_pocketfft_shape_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft;
    std::pair<SphereGLQGrid<double, GridLayout>, SphereGLQGrid<double, GridLayout>> m_grids;
    SHPhase m_phase = SHPhase::NONE;
    std::size_t m_lmax;
};

}
}