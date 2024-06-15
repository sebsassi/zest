#pragma once

#include <vector>
#include <array>
#include <span>
#include <complex>

#include "gauss_legendre.hpp"
#include "pocketfft.hpp"
#include "plm_recursion.hpp"
#include "zernike_expansion.hpp"
#include "radial_zernike_recursion.hpp"
#include "alignment.hpp"
#include "md_span.hpp"

namespace zest
{
namespace zt
{

template <typename AlignmentType = CacheLineAlignment>
struct LonLatRadLayout
{
    using Alignment = AlignmentType;

    [[nodiscard]] static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return lat_size(lmax)*lon_size(lmax)*rad_size(lmax);
    }
    
    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    shape(std::size_t lmax) noexcept
    {
        return {lon_size(lmax), lat_size(lmax), rad_size(lmax)};
    }

    [[nodiscard]] static constexpr std::size_t
    fft_size(std::size_t lmax) noexcept
    {
        return (lon_size(lmax) >> 1) + 1;
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    fft_stride(std::size_t lmax) noexcept
    {
        return {lat_size(lmax)*rad_size(lmax), rad_size(lmax), 1};
    }

    [[nodiscard]] static constexpr std::size_t
    lat_size(std::size_t lmax) noexcept
    {
        return lmax + 2UL;
    }

    [[nodiscard]] static constexpr std::size_t
    rad_size(std::size_t lmax) noexcept
    {
        constexpr std::size_t vector_size
                = Alignment::template vector_size<double>();
        const std::size_t min_size = lmax + 2UL;
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
    static constexpr std::size_t rad_axis = 2UL;
};

using DefaultLayout = LonLatRadLayout<>;

/*
A non-owning view of gridded data in spherical coordinates in the unit ball.
*/
template <typename ElementType, typename LayoutType = DefaultLayout>
    requires std::same_as<std::remove_const_t<ElementType>, double>
class BallGLQGridSpan: public MDSpan<ElementType, 3>
{
public:
    using typename MDSpan<ElementType, 3>::element_type;
    using Layout = LayoutType;

    using MDSpan<ElementType, 3>::extents;
    using MDSpan<ElementType, 3>::data;
    using MDSpan<ElementType, 3>::size;

    [[nodiscard]] static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    shape(std::size_t lmax) noexcept
    {
        return Layout::shape(lmax);
    }

    constexpr BallGLQGridSpan(std::span<element_type> buffer, std::size_t lmax):
        MDSpan<ElementType, 3>(buffer.data(), Layout::shape(lmax)),
        m_lmax(lmax) {}
    constexpr BallGLQGridSpan(
        std::span<element_type> buffer, std::size_t idx, std::size_t lmax):
        MDSpan<ElementType, 3>(buffer.data() + idx*Layout::size(lmax), Layout::shape(lmax)), m_lmax(lmax) {}

    [[nodiscard]] constexpr std::size_t lmax() const noexcept { return m_lmax; }
    
    [[nodiscard]] constexpr const std::array<std::size_t, 3>&
    shape() const noexcept { return extents(); }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span<element_type>(data(), size()); }

private:
    std::size_t m_lmax;
};

/*
Container for gridded data in spherical coordinates in the unit ball.
*/
template <typename LayoutType = DefaultLayout>
class BallGLQGrid
{
public:
    using Layout = LayoutType;
    using View = BallGLQGridSpan<double>;
    using ConstView = BallGLQGridSpan<const double>;

    [[nodiscard]] static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    [[nodiscard]] static constexpr std::array<std::size_t, 3>
    shape(std::size_t lmax) noexcept
    {
        return Layout::shape(lmax);
    }

    BallGLQGrid(): BallGLQGrid(0) {}
    explicit BallGLQGrid(std::size_t lmax):
        m_values(Layout::size(lmax)), m_shape(Layout::shape(lmax)),
        m_lmax(lmax) {}

    [[nodiscard]] std::array<std::size_t, 3> shape() { return m_shape; }
    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<const double> values() const noexcept { return m_values; }
    std::span<double> values() noexcept { return m_values; }

    [[nodiscard]] operator View()
    {
        return View(m_values, m_lmax);
    };

    [[nodiscard]] operator ConstView() const
    {
        return ConstView(m_values, m_lmax);
    };

    void resize(std::size_t lmax)
    {
        m_values.resize(Layout::size(lmax));
        m_shape = Layout::shape(lmax);
        m_lmax = lmax;
    }

    [[nodiscard]] double operator()(
        std::size_t i, std::size_t j, std::size_t k) const noexcept
    {
        return m_values[m_shape[2]*(m_shape[1]*i + j) + k];
    }

    [[nodiscard]] double& operator()(
        std::size_t i, std::size_t j, std::size_t k) noexcept
    {
        return m_values[m_shape[2]*(m_shape[1]*i + j) + k];
    }
private:
    std::vector<double> m_values;
    std::array<std::size_t, 3> m_shape;
    std::size_t m_lmax;
};

/*
Points defining a grid in spherical coordinates in the unit ball.
*/
class BallGLQGridPoints
{
public:
    BallGLQGridPoints() = default;

    void resize(std::size_t num_lon, std::size_t num_lat, std::size_t num_rad);

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }

    [[nodiscard]] std::span<const double> longitudes() const noexcept
    {
        return m_longitudes;
    }

    [[nodiscard]] std::span<const double> rad_glq_nodes() const noexcept
    {
        return m_rad_glq_nodes;
    }
    
    [[nodiscard]] std::span<const double> lat_glq_nodes() const noexcept
    {
        return m_lat_glq_nodes;
    }

    template <typename LayoutType, typename FuncType>
    void generate_values(BallGLQGridSpan<double, LayoutType> grid, FuncType&& f)
    {
        constexpr std::size_t lon_axis = LayoutType::lon_axis;
        constexpr std::size_t lat_axis = LayoutType::lat_axis;
        constexpr std::size_t rad_axis = LayoutType::rad_axis;
        const auto shape = grid.shape();
        resize(shape[lon_axis], shape[lat_axis], shape[rad_axis]);
        
        if constexpr (std::same_as<LayoutType, LonLatRadLayout<typename LayoutType::Alignment>>)
        {
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
            {
                const double lon = m_longitudes[i];
                for (std::size_t j = 0; j < m_lat_glq_nodes.size(); ++j)
                {
                    const double colatitude = m_lat_glq_nodes[j];
                    for (std::size_t k = 0; k < m_rad_glq_nodes.size(); ++k)
                    {
                        const double r = m_rad_glq_nodes[k];
                        grid(i, j, k) = f(r, lon, colatitude);
                    }
                }
            }
        }
    }

    template <typename LayoutType = DefaultLayout, typename FuncType>
    [[nodiscard]] auto generate_values(FuncType&& f, std::size_t lmax)
    {
        BallGLQGrid<LayoutType> grid(lmax);
        generate_values<LayoutType, FuncType>(grid, f);
        return grid;
    }

#ifdef ZEST_USE_OMP
    template <typename LayoutType, typename FuncType>
    void generate_values(
        BallGLQGridSpan<double> grid, FuncType&& f, std::size_t num_threads)
    {
        constexpr std::size_t lon_axis = LayoutType::lon_axis;
        constexpr std::size_t lat_axis = LayoutType::lat_axis;
        constexpr std::size_t rad_axis = LayoutType::rad_axis;
        const auto shape = grid.shape();
        resize(shape[lon_axis], shape[lat_axis], shape[rad_axis]);

        std::size_t nthreads = (num_threads) ?
                num_threads : std::size_t(omp_get_max_threads());
        if constexpr (std::same_as<LayoutType, LonLatRadLayout<typename LayoutType::Alignment>>)
        {
            #pragma omp parallel for num_threads(nthreads) collapse(2)
            for (std::size_t i = 0; i < m_longitudes.size(); ++i)
            {
                for (std::size_t j = 0; j < m_lat_glq_nodes.size(); ++j)
                {
                    const double lon = m_longitudes[i];
                    const double colatitude = m_lat_glq_nodes[j];
                    for (std::size_t k = 0; k < m_rad_glq_nodes.size(); ++k)
                    {
                        const double r = m_rad_glq_nodes[k];
                        grid(i, j, k) = f(r, lon, colatitude);
                    }
                }
            }
        }
    }

    template <typename LayoutType = DefaultLayout, typename FuncType>
    [[nodiscard]] auto generate_values(
        FuncType&& f, std::size_t lmax, std::size_t num_threads)
    {
        BallGLQGrid<LayoutType> grid(lmax);
        generate_values<LayoutType, FuncType>(grid, f, num_threads);
        return grid;
    }
#endif

private:
    std::vector<double> m_rad_glq_nodes;
    std::vector<double> m_lat_glq_nodes;
    std::vector<double> m_longitudes;
    std::size_t m_lmax;
};

/*
Class for transforming between a Gauss-Legendre quadrature grid representation and Zernike polynomial expansion representation of data in the unit baal.
*/
template <st::SHNorm NORM, st::SHPhase PHASE, typename GridLayoutType = DefaultLayout>
class GLQTransformer
{
public:
    using GridLayout = GridLayoutType;
    explicit GLQTransformer(std::size_t lmax):
        m_zernike_recursion(lmax), m_plm_recursion(lmax),
        m_rad_glq_nodes(GridLayout::rad_size(lmax)),
        m_rad_glq_weights(GridLayout::rad_size(lmax)),
        m_lat_glq_nodes(GridLayout::lat_size(lmax)),
        m_lat_glq_weights(GridLayout::lat_size(lmax)),
        m_zernike_grid(GridLayout::rad_size(lmax)*RadialZernikeLayout::size(lmax)),
        m_plm_grid(GridLayout::lat_size(lmax)*TriangleLayout::size(lmax)),
        m_flm_grid(GridLayout::rad_size(lmax)*TriangleLayout::size(lmax)),
        m_ffts(GridLayout::rad_size(lmax)*GridLayout::lat_size(lmax)*GridLayout::fft_size(lmax)), m_pocketfft_shape_grid(3),
        m_pocketfft_stride_grid(3), m_pocketfft_stride_fft(3), m_lmax(lmax)
    {
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::COS>(
                m_rad_glq_nodes, m_rad_glq_weights,
                m_rad_glq_weights.size() & 1);
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::COS>(
                m_lat_glq_nodes, m_lat_glq_weights,
                m_lat_glq_weights.size() & 1);
        
        for (auto& node : m_rad_glq_nodes)
            node = 0.5*(1.0 + node);
        
        RadialZernikeVecSpan<double> zernike(
                m_zernike_grid, m_lmax, m_rad_glq_nodes.size());
        m_zernike_recursion.zernike<ZernikeNorm::NORMED>(
                zernike, m_rad_glq_nodes);

        st::PlmVecSpan<double, NORM, PHASE> plm(
                m_plm_grid, m_lmax, m_lat_glq_nodes.size());
        m_plm_recursion.plm_real(plm, m_lat_glq_nodes);

        auto shape = GridLayout::shape(lmax);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];
        m_pocketfft_shape_grid[2] = shape[2];

        m_pocketfft_stride_grid[0] = long(shape[1]*shape[2]*sizeof(double));
        m_pocketfft_stride_grid[1] = long(shape[2]*sizeof(double));
        m_pocketfft_stride_grid[2] = sizeof(double);

        auto fft_stride = GridLayout::fft_stride(lmax);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = long(fft_stride[1]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[2] = fft_stride[2]*sizeof(std::complex<double>);
    }

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] static constexpr st::SHNorm norm() noexcept { return NORM; }
    
    [[nodiscard]] static constexpr st::SHPhase
    phase() noexcept { return PHASE; }

    void resize(std::size_t lmax)
    {
        if (lmax == this->lmax()) return;

        m_plm_recursion.expand(lmax);
        m_zernike_recursion.expand(lmax);

        m_rad_glq_nodes.resize(GridLayout::rad_size(lmax));
        m_rad_glq_weights.resize(GridLayout::rad_size(lmax));
        m_lat_glq_nodes.resize(GridLayout::lat_size(lmax));
        m_lat_glq_weights.resize(GridLayout::lat_size(lmax));

        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::COS>(
                m_rad_glq_nodes, m_rad_glq_weights,
                m_rad_glq_weights.size() & 1);
        gl::gl_nodes_and_weights<gl::UnpackedLayout, gl::GLNodeStyle::COS>(
                m_lat_glq_nodes, m_lat_glq_weights,
                m_lat_glq_weights.size() & 1);
        
        for (auto& node : m_rad_glq_nodes)
            node = 0.5*(1.0 + node);
        
        m_zernike_grid.resize(GridLayout::rad_size(lmax)*RadialZernikeLayout::size(lmax));
        
        RadialZernikeVecSpan<double> zernike(
                m_zernike_grid, m_lmax, m_rad_glq_nodes.size());
        m_zernike_recursion.zernike<ZernikeNorm::NORMED>(
                zernike, m_rad_glq_nodes);
        
        m_plm_grid.resize(GridLayout::lat_size(lmax)*TriangleLayout::size(lmax));
        m_flm_grid.resize(GridLayout::rad_size(lmax)*TriangleLayout::size(lmax));

        st::PlmVecSpan<double, NORM, PHASE> plm(
                m_plm_grid, m_lmax, m_lat_glq_nodes.size());
        m_plm_recursion.plm_real(plm, m_lat_glq_nodes);

        m_ffts.resize(GridLayout::rad_size(lmax)*GridLayout::lat_size(lmax)*GridLayout::fft_size(lmax));
        std::array<std::size_t, 3> shape = GridLayout::shape(lmax);
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];
        m_pocketfft_shape_grid[2] = shape[2];

        m_pocketfft_stride_grid[0] = long(shape[1]*shape[2]*sizeof(double));
        m_pocketfft_stride_grid[1] = long(shape[2]*sizeof(double));
        m_pocketfft_stride_grid[2] = sizeof(double);

        std::array<std::size_t, 3> fft_stride = GridLayout::fft_stride(lmax);
        m_pocketfft_stride_fft[0] = long(fft_stride[0]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = long(fft_stride[1]*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[2] = fft_stride[2]*sizeof(std::complex<double>);

        m_lmax = lmax;
    }

    void forward_transform(
        BallGLQGridSpan<const double, GridLayout> values,
        ZernikeExpansionSpan<std::array<double, 2>, NORM, PHASE> expansion)
    {
        resize(values.lmax());

        integrate_longitudinal(values);
        apply_weights();

        std::size_t min_lmax = std::min(expansion.lmax(), values.lmax());
        integrate_latitudinal(min_lmax);
        integrate_radial(expansion, min_lmax);
    }

    void backward_transform(
        ZernikeExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion,
        BallGLQGridSpan<double, GridLayout> values)
    {
        resize(values.lmax());

        std::size_t min_lmax = std::min(expansion.lmax(), values.lmax());
        
        sum_n(expansion, min_lmax);
        sum_l(min_lmax);
        sum_m(values);
    }
    
    [[nodiscard]] ZernikeExpansion<NORM, PHASE> forward_transform(
        BallGLQGridSpan<const double, GridLayout> values, std::size_t lmax)
    {
        ZernikeExpansion<NORM, PHASE> expansion(lmax);
        forward_transform(values, expansion);
        return expansion;
    }

    [[nodiscard]] BallGLQGrid<GridLayout> backward_transform(
        ZernikeExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion, std::size_t lmax)
    {
        BallGLQGrid<GridLayout> grid(lmax);
        backward_transform(expansion, grid);
        return grid;
    }

private:
    void integrate_longitudinal(
        BallGLQGridSpan<const double, GridLayout> values)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr double radial_integral_norm = 0.5;
        constexpr double sh_norm = st::normalization<st::SHNorm::GEO>();
        const double fourier_norm = (2.0*std::numbers::pi)/double(values.shape()[lon_axis]);
        const double prefactor = sh_norm*radial_integral_norm*fourier_norm;
        pocketfft::r2c(
            m_pocketfft_shape_grid, m_pocketfft_stride_grid, m_pocketfft_stride_fft, lon_axis, pocketfft::FORWARD, values.flatten().data(), m_ffts.data(), prefactor);
    }

    void apply_weights() noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        const std::size_t lat_glq_size = m_lat_glq_weights.size();
        const std::size_t fft_order = GridLayout::fft_size(m_lmax);

        MDSpan<std::complex<double>, 3> fft(
                m_ffts.data(), {fft_order, lat_glq_size, rad_glq_size});

        if constexpr (std::same_as<GridLayout, LonLatRadLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t m = 0; m < fft_order; ++m)
            {
                MDSpan<std::complex<double>, 2> fft_m = fft[m];
                for (std::size_t i = 0; i < lat_glq_size; ++i)
                {
                    MDSpan<std::complex<double>, 1> fft_mi = fft_m[i];
                    const double lat_weight = m_lat_glq_weights[i];
                    for (std::size_t j = 0; j < rad_glq_size; ++j)
                    {
                        const double r = m_rad_glq_nodes[j];
                        const double weight
                                = lat_weight*r*r*m_rad_glq_weights[j];
                        std::complex<double>& x = fft_mi[j];
                        x = {weight*x.real(), -weight*x.imag()};
                    }
                }
            }
        }

        /*
        else if (std::same_as<GridLayout, RadLatLonLayout>)
        {
            for (std::size_t i = 0; i < rad_glq_size; ++i)
            {
                std::span<std::complex<double>> fft_i(
                        m_ffts.begin() + i*fft_order*lat_glq_size, 
                        fft_order*lat_glq_size);
                const double r = m_rad_glq_nodes[i];
                const double radial_weight = r*r*m_glq_weights[i];
                for (std::size_t j = 0; j < lat_glq_size; ++j)
                {
                    std::span<std::complex<double>> fft_ij((
                            fft_i.begin() + j*fft_order, fft_order);
                    const double weight = radial_weight*m_lat_glq_weights[j];
                    for (std::size_t m = 0; m < fft_order; ++m)
                    {
                        std::complex<double>& x = fft_ij[m];
                        x = {weight*x.real(), -weight*x.imag()};
                    }
                }
            }
        }
        */
    }

    void integrate_latitudinal(std::size_t min_lmax) noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        const std::size_t lat_glq_size = m_lat_glq_weights.size();
        const std::size_t fft_order = GridLayout::fft_size(m_lmax);
        std::ranges::fill(m_flm_grid, std::array<double, 2>{});

        TriangleVecSpan<std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, m_lmax, rad_glq_size);

        st::PlmVecSpan<const double, NORM, PHASE> ass_leg(
                m_plm_grid, m_lmax, m_lat_glq_nodes.size());

        MDSpan<const std::complex<double>, 3> fft(
                m_ffts.data(), {fft_order, lat_glq_size, rad_glq_size});
        if constexpr (std::same_as<GridLayout, LonLatRadLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t l = 0; l <= min_lmax; ++l)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::span<std::array<double, 2>> flm_lm = flm(l,m);
                    std::span<const double> ass_leg_lm = ass_leg(l,m);
                    MDSpan<const std::complex<double>, 2> fft_m = fft[m];
                    for (std::size_t i = 0; i < lat_glq_size; ++i)
                    {
                        const double ass_leg_lmi = ass_leg_lm[i];
                        MDSpan<const std::complex<double>, 1> fft_mi = fft_m[i];
                        for (std::size_t j = 0; j < rad_glq_size; ++j)
                        {
                            flm_lm[j][0] += ass_leg_lmi*fft_mi[j].real();
                            flm_lm[j][1] += ass_leg_lmi*fft_mi[j].imag();
                        }
                    }
                }
            }
        }
    }

    void integrate_radial(
        ZernikeExpansionSpan<std::array<double, 2>, NORM, PHASE> expansion, std::size_t min_lmax) noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        std::ranges::fill(expansion.span(), std::array<double, 2>{});

        TriangleVecSpan<const std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, m_lmax, rad_glq_size);

        RadialZernikeVecSpan<const double> zernike(
                m_zernike_grid, m_lmax, m_rad_glq_nodes.size());
        if constexpr (std::same_as<GridLayout, LonLatRadLayout<typename GridLayout::Alignment>>)
        {
            for (std::size_t n = 0; n <= min_lmax; ++n)
            {
                ZernikeExpansionLMSpan<std::array<double, 2>, NORM, PHASE> 
                expansion_n = expansion[n];
                
                for (std::size_t l = n & 1; l <= n; l += 2)
                {
                    std::span<std::array<double, 2>> expansion_nl
                        = expansion_n[l];
                    
                    std::span<const double> zernike_nl = zernike(n,l);
                    for (std::size_t m = 0; m <= l; ++m)
                    {
                        std::span<const std::array<double, 2>>
                        flm_lm = flm(l,m);

                        std::array<double, 2>& coeff = expansion_nl[m];
                        for (std::size_t i = 0; i < rad_glq_size; ++i)
                        {
                            coeff[0] += zernike_nl[i]*flm_lm[i][0];
                            coeff[1] += zernike_nl[i]*flm_lm[i][1];
                        }
                    }
                }
            }
        }
    }

    void sum_n(
        ZernikeExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion, 
        std::size_t min_lmax) noexcept
    {
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        std::ranges::fill(m_flm_grid, std::array<double, 2>{});

        RadialZernikeVecSpan<const double> zernike(
                m_zernike_grid, m_lmax, m_rad_glq_nodes.size());

        TriangleVecSpan<std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, m_lmax, rad_glq_size);
        for (std::size_t n = 0; n <= min_lmax; ++n)
        {
            auto expansion_n = expansion[n];
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                std::span<const double> zernike_nl = zernike(n,l);
                auto expansion_nl = expansion_n[l];
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::span<std::array<double, 2>> flm_lm = flm(l,m);
                    const std::array<double, 2> coeff = expansion_nl[m];
                    for (std::size_t i = 0; i < rad_glq_size; ++i)
                    {
                        flm_lm[i][0] += zernike_nl[i]*coeff[0];
                        flm_lm[i][1] += zernike_nl[i]*coeff[1];
                    }
                }
            }
        }
    }

    void sum_l(std::size_t min_lmax) noexcept
    {
        const std::size_t lat_glq_size = m_lat_glq_weights.size();
        const std::size_t rad_glq_size = m_rad_glq_weights.size();
        const std::size_t fft_order = GridLayout::fft_size(m_lmax);

        TriangleVecSpan<const std::array<double, 2>, TriangleLayout>
        flm(m_flm_grid, m_lmax, rad_glq_size);
        
        st::PlmVecSpan<const double, NORM, PHASE> ass_leg(
                m_plm_grid, m_lmax, m_lat_glq_nodes.size());

        std::ranges::fill(m_ffts, std::complex<double>{});
        MDSpan<std::complex<double>, 3> fft(
                m_ffts.data(), {fft_order, lat_glq_size, rad_glq_size});
        for (std::size_t l = 0; l <= min_lmax; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::span<const std::array<double, 2>> flm_lm = flm(l,m);
                std::span<const double> plm = ass_leg(l,m);
                const double m_factor = (m > 0) ? 0.5 : 1.0;
                
                MDSpan<std::complex<double>, 2> fft_m = fft[m];
                for (std::size_t i = 0; i < lat_glq_size; ++i)
                {
                    const double plm_i = plm[i];
                    const double weight = m_factor*plm_i;
                    MDSpan<std::complex<double>, 1> fft_mi = fft_m[i];
                    for (std::size_t j = 0; j < rad_glq_size; ++j)
                    {
                        fft_mi[j] += std::complex<double>{
                            weight*flm_lm[j][0], -weight*flm_lm[j][1]
                        };
                    }
                }
            }
        }
    }

    void sum_m(BallGLQGridSpan<double, GridLayout> values)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr double prefactor = 1.0;
        pocketfft::c2r(
            m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid, lon_axis, pocketfft::BACKWARD, m_ffts.data(), values.flatten().data(), prefactor);
    }
    /*
    void fft_fill(
        ZernikeExpansionSpan<const std::array<double, 2>, NORM, PHASE> expansion,
        std::size_t imin, std::size_t imax, std::size_t min_lmax)
    {
        std::ranges::fill(m_ffts, std::complex<double>{});
        const std::size_t glq_size = m_glq_weights.size();
        const std::size_t fft_size = glq_size - 1;
        for (std::size_t i = imin; i < imax; ++i)
        {
            RadialZernikeSpan<const double> zernike(m_zernike_grid, i, lmax());
            for (std::size_t j = 0; j < m_glq_weights.size(); ++j)
            {
                std::span<std::complex<double>> fft(
                        m_ffts.begin() + (i*glq_size + j)*fft_size, fft_size);
                st::PlmSpan<const double, st::SHNorm::GEO, st::SHPhase::NONE> plm(m_plm_grid, j, lmax());
                for (std::size_t n = 0; n <= min_lmax; ++n)
                {
                    for (std::size_t l = n & 1; l <= n; l += 2)
                    {
                        const double zer = zernike(n, l);
                        const auto coeff = expansion(n, l, 0);
                        fft[0] += (zer*plm(l, 0))*std::complex<double>{coeff[0], -coeff[1]};
                        for (std::size_t m = 1; m <= l; ++m)
                        {
                            const auto coeff = expansion(n, l, m);
                            fft[m] += (0.5*zer*plm(l, m))*std::complex<double>{coeff[0], -coeff[1]};
                        }
                    }
                }
            }
        }
    }*/

    RadialZernikeRecursion m_zernike_recursion;
    st::PlmRecursion m_plm_recursion;
    std::vector<double> m_rad_glq_nodes;
    std::vector<double> m_rad_glq_weights;
    std::vector<double> m_lat_glq_nodes;
    std::vector<double> m_lat_glq_weights;
    std::vector<double> m_zernike_grid;
    std::vector<double> m_plm_grid;
    std::vector<std::array<double, 2>> m_flm_grid;
    std::vector<std::complex<double>> m_ffts;
    std::vector<std::size_t> m_pocketfft_shape_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft;
    std::size_t m_lmax;
};

/*
Function concept taking Cartesian coordinates as inputs.
*/
template <typename Func>
concept cartesian_function = requires (Func f, std::array<double, 3> x)
{
    { f(x) } -> std::same_as<double>;
};

/*
Function concept taking spherical coordinates as inputs.
*/
template <typename Func>
concept spherical_function = requires (Func f, double r, double lon, double colat)
{
    { f(r, lon, colat) } -> std::same_as<double>;
};

/*
High-level interface for taking Zernike transforms of functions on balls of arbitrary radii.
*/
template <st::SHNorm NORM, st::SHPhase PHASE, typename GridLayoutType = DefaultLayout>
class ZernikeTransformer
{
public:
    using GridLayout = GridLayoutType;
    ZernikeTransformer(std::size_t lmax):
        m_grid(lmax), m_points(lmax), m_transformer(lmax) {}

    void resize(std::size_t lmax)
    {
        constexpr std::size_t lon_axis = GridLayout::lon_axis;
        constexpr std::size_t lat_axis = GridLayout::lat_axis;
        constexpr std::size_t rad_axis = GridLayout::rad_axis;
        const auto shape = m_grid.shape();
        m_points.resize(shape[lon_axis], shape[lat_axis], shape[rad_axis]);
        m_grid.resize(lmax);
        m_transformer.resize(lmax);
    }

    template <spherical_function Func>
    void transform(
        Func&& f, double radius,
        ZernikeExpansionSpan<std::array<double, 2>, NORM, PHASE> expansion)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            return f(r*radius, lon, colat);
        };
        resize(expansion.lmax());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.forward_transform(m_grid, expansion);
    }

    template <spherical_function Func>
    [[nodiscard]] ZernikeExpansion<NORM, PHASE> transform(
        Func&& f, double radius, std::size_t lmax)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            return f(r*radius, lon, colat);
        };
        resize(lmax);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.forward_transform(m_grid, lmax);
    }

    template <cartesian_function Func>
    void transform(
        Func&& f, double radius,
        ZernikeExpansionSpan<std::array<double, 2>, NORM, PHASE> expansion)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            const double rad = r*radius;
            const double scolat = std::sin(colat);
            const std::array<double, 3> x = {
                rad*scolat*std::cos(lon), rad*scolat*std::sin(lon),
                rad*std::cos(colat)
            };
            return f(x);
        };
        resize(expansion.lmax());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.forward_transform(m_grid, expansion);
    }

    template <cartesian_function Func>
    [[nodiscard]] ZernikeExpansion<NORM, PHASE> transform(
        Func&& f, double radius, std::size_t lmax)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            const double rad = r*radius;
            const double scolat = std::sin(colat);
            const std::array<double, 3> x = {
                rad*scolat*std::cos(lon), rad*scolat*std::sin(lon),
                rad*std::cos(colat)
            };
            return f(x);
        };
        resize(lmax);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.forward_transform(m_grid, lmax);
    }

private:
    BallGLQGrid<GridLayout> m_grid;
    BallGLQGridPoints m_points;
    GLQTransformer<NORM, PHASE, GridLayout> m_transformer;
};

/*
`GLQTransformer` with orthonormal spherical harmonics and no Condon-Shortley phase.
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerAcoustics
    = GLQTransformer<st::SHNorm::QM, st::SHPhase::NONE, GridLayout>;

/*
`GLQTransformer` with orthonormal spherical harmonics with Condon-Shortley phase.
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerQM
    = GLQTransformer<st::SHNorm::QM, st::SHPhase::CS, GridLayout>;

/*
`GLQTransformer` with 4-pi normal spherical harmonics and no Condon-Shortley phase.
*/
template <typename GridLayout = DefaultLayout>
using GLQTransformerGeo
    = GLQTransformer<st::SHNorm::GEO, st::SHPhase::NONE, GridLayout>;

}
}