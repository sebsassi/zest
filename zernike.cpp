#include "zernike.hpp"

#include "real_sh_expansion.hpp"

namespace zest
{
namespace zt
{

RadialZernikeRecursion::RadialZernikeRecursion(std::size_t lmax):
    m_norms(lmax + 1),
    m_k1(RadialZernikeLayout::size(lmax)),
    m_k2(RadialZernikeLayout::size(lmax)),
    m_k3(RadialZernikeLayout::size(lmax)), m_lmax(lmax)
{
    for (std::size_t n = 0; n <= lmax; ++n)
    {
        const double dn = double(n);
        m_norms[n] = std::sqrt(2.0*dn + 3.0);
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            const double dl = double(l);
            const double k0 = (dn - dl)*(dn + dl + 1.0)*(2.0*dn - 3.0);
            const double k1 = (2.0*dn - 1.0)*(2.0*dn + 1)*(2.0*dn - 3.0);
            const double k2 = -0.5*(2.0*dn - 1.0)*((2*dl + 1.0)*(2.0*dl + 1.0) + (2.0*dn + 1)*(2.0*dn - 3.0));
            const double k3 = -(dn - dl - 2.0)*(dn + dl - 1.0)*(2.0*dn + 1);
            
            const double k0_inv = 1.0/k0;
            const std::size_t idx = RadialZernikeLayout::idx(n,l);
            m_k1[idx] = k1*k0_inv;
            m_k2[idx] = k2*k0_inv;
            m_k3[idx] = k3*k0_inv;
        }
    }
}

void RadialZernikeRecursion::expand(std::size_t lmax)
{
    if (lmax <= m_lmax) return;

    m_norms.resize(lmax + 1);
    m_k1.resize(RadialZernikeLayout::size(lmax));
    m_k2.resize(RadialZernikeLayout::size(lmax));
    m_k3.resize(RadialZernikeLayout::size(lmax));

    for (std::size_t n = m_lmax + 1; n <= lmax; ++n)
    {
        const double dn = double(n);
        m_norms[n] = std::sqrt(2.0*dn + 3.0);
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            const double dl = double(l);
            const double k0 = (dn - dl)*(dn + dl + 1.0)*(2.0*dn - 3.0);
            const double k1 = (2.0*dn - 1.0)*(2.0*dn + 1)*(2.0*dn - 3.0);
            const double k2 = -0.5*(2.0*dn - 1.0)*((2*dl + 1.0)*(2.0*dl + 1.0) + (2.0*dn + 1)*(2.0*dn - 3.0));
            const double k3 = -(dn - dl - 2.0)*(dn + dl - 1.0)*(2.0*dn + 1);
            
            const double k0_inv = 1.0/k0;
            const std::size_t idx = RadialZernikeLayout::idx(n,l);
            m_k1[idx] = k1*k0_inv;
            m_k2[idx] = k2*k0_inv;
            m_k3[idx] = k3*k0_inv;
        }
    }

    m_lmax = lmax;
}

ZernikeExpansion::ZernikeExpansion(std::size_t lmax):
    m_coeffs(Layout::size(lmax)), m_lmax(lmax) {}

void power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion,
    RadialZernikeSpan<double> out)
{
    std::size_t min_lmax = std::min(out.lmax(), expansion.lmax());

    for (std::size_t n = 0; n < min_lmax; ++n)
    {
        for (std::size_t l = n & 1; l < n; l += 2)
        {
            out(n, l) = expansion(n, l, 0)[0]*expansion(n, l, 0)[0];
            for (std::size_t m = 1; m <= l; ++m)
                out(n, l) += expansion(n, l, m)[0]*expansion(n, l, m)[0]
                        + expansion(n, l, m)[1]*expansion(n, l, m)[1];
        }
    }
}

std::vector<double> power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion)
{
    std::vector<double> res(RadialZernikeLayout::size(expansion.lmax()));
    power_spectrum(expansion, RadialZernikeSpan<double>(res, expansion.lmax()));
    return res;
}

void BallGLQGrid::resize(std::size_t lmax)
{
    m_values.resize((lmax + 2)*(lmax + 2)*(2*lmax + 1));
    m_shape = {lmax + 2, lmax + 2, 2*lmax + 1};
    m_lmax = lmax;
}

BallGLQGridPoints::BallGLQGridPoints(std::size_t lmax):
    m_longitudes(2*lmax + 1), m_glq_nodes(lmax + 2), m_lmax(lmax)
{
    const double dlon = (2.0*std::numbers::pi)/double(m_longitudes.size());
    for (std::size_t i = 0; i < m_longitudes.size(); ++i)
        m_longitudes[i] = dlon*double(i);

    gl::gl_nodes<double, gl::GLLayout::UNPACKED, gl::GLNodeStyle::ANGLE>(m_glq_nodes, m_glq_nodes.size());
}

void BallGLQGridPoints::resize(std::size_t lmax)
{
    if (lmax == m_lmax) return;
    m_longitudes.resize(2*lmax + 1);
    m_glq_nodes.resize(lmax + 2);

    const double dlon = (2.0*std::numbers::pi)/double(m_longitudes.size());
    for (std::size_t i = 0; i < m_longitudes.size(); ++i)
        m_longitudes[i] = dlon*double(i);

    gl::gl_nodes<double, gl::GLLayout::UNPACKED, gl::GLNodeStyle::ANGLE>(m_glq_nodes, m_glq_nodes.size());
}

GLQTransformer::GLQTransformer(std::size_t lmax):
    m_zernike_recursion(lmax), m_plm_recursion(lmax), m_glq_nodes(lmax + 2),
    m_glq_weights(lmax + 2),
    m_zernike_grid((lmax + 2)*RadialZernikeLayout::size(lmax)),
    m_plm_grid((lmax + 2)*TriangleLayout::size(lmax)),
    m_flm_grid((lmax + 2)*TriangleLayout::size(lmax)),
    m_ffts((lmax + 2)*(lmax + 2)*(lmax + 1)), m_pocketfft_shape_grid(3),
    m_pocketfft_stride_grid(3), m_pocketfft_stride_fft(3), m_lmax(lmax)
{
    gl::gl_nodes_and_weights<double, gl::GLLayout::UNPACKED, gl::GLNodeStyle::COS>(
            m_glq_nodes, m_glq_weights, m_glq_weights.size());
    
    for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
    {
        const double r = 0.5*(1.0 + m_glq_nodes[i]);
        RadialZernikeSpan<double> zernike(m_zernike_grid, i, m_lmax);
        m_zernike_recursion.zernike<ZernikeNorm::NORMED>(zernike, r);
    }

    for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
    {
        const double z = m_glq_nodes[i];
        st::PlmSpan<double, st::SHNorm::GEO, st::SHPhase::NONE> plm(m_plm_grid, i, m_lmax);
        m_plm_recursion.plm_real(plm, z);
    }

    m_pocketfft_shape_grid[0] = lmax + 2;
    m_pocketfft_shape_grid[1] = lmax + 2;
    m_pocketfft_shape_grid[2] = 2*lmax + 1;

    m_pocketfft_stride_grid[0] = long((lmax + 2)*(2*lmax + 1)*sizeof(double));
    m_pocketfft_stride_grid[1] = long((2*lmax + 1)*sizeof(double));
    m_pocketfft_stride_grid[2] = sizeof(double);

    m_pocketfft_stride_fft[0] = long((lmax + 2)*(lmax + 1)*sizeof(std::complex<double>));
    m_pocketfft_stride_fft[1] = long((lmax + 1)*sizeof(std::complex<double>));
    m_pocketfft_stride_fft[2] = sizeof(std::complex<double>);
}

void GLQTransformer::resize(std::size_t lmax)
{
    if (lmax == this->lmax()) return;

    m_plm_recursion.expand(lmax);
    m_zernike_recursion.expand(lmax);

    gl::gl_nodes_and_weights<double, gl::GLLayout::UNPACKED, gl::GLNodeStyle::COS>(
            m_glq_nodes, m_glq_weights, m_glq_weights.size());
    
    m_zernike_grid.resize((lmax + 2)*RadialZernikeLayout::size(lmax));
    for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
    {
        const double r = 0.5*(1.0 + m_glq_nodes[i]);
        RadialZernikeSpan<double> zernike(m_zernike_grid, i, m_lmax);
        m_zernike_recursion.zernike<ZernikeNorm::NORMED>(zernike, r);
    }
    
    m_plm_grid.resize((lmax + 2)*TriangleLayout::size(lmax));
    m_flm_grid.resize((lmax + 2)*TriangleLayout::size(lmax));
    for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
    {
        const double z = m_glq_nodes[i];
        st::PlmSpan<double, st::SHNorm::GEO, st::SHPhase::NONE> plm(m_plm_grid, i, lmax);
        m_plm_recursion.plm_real(plm, z);
    }

    m_ffts.resize((lmax + 2)*(lmax + 2)*(lmax + 1));
    m_pocketfft_shape_grid[0] = lmax + 2;
    m_pocketfft_shape_grid[1] = lmax + 2;
    m_pocketfft_shape_grid[2] = 2*lmax + 1;

    m_pocketfft_stride_grid[0] = long((lmax + 2)*(2*lmax + 1)*sizeof(double));
    m_pocketfft_stride_grid[1] = long((2*lmax + 1)*sizeof(double));
    m_pocketfft_stride_grid[2] = sizeof(double);

    m_pocketfft_stride_fft[0] = long((lmax + 2)*(lmax + 1)*sizeof(std::complex<double>));
    m_pocketfft_stride_fft[1] = long((lmax + 1)*sizeof(std::complex<double>));
    m_pocketfft_stride_fft[2] = sizeof(std::complex<double>);

    m_lmax = lmax;
}

void GLQTransformer::transform(
    BallGLQGridSpan<const double> values,
    ZernikeExpansionSpan<std::array<double, 2>> expansion)
{
    resize(values.lmax());

    integrate_longitudinal(values);
    apply_weights();

    std::size_t min_lmax = std::min(expansion.lmax(), values.lmax());
    integrate_latitudinal(min_lmax);
    integrate_radial(expansion, min_lmax);
}

void GLQTransformer::integrate_longitudinal(
    BallGLQGridSpan<const double> values)
{
    constexpr std::size_t axis = 2;
    constexpr double radial_integral_norm = 0.5;
    constexpr double sh_norm = 1.0/(4.0*std::numbers::pi);
    const double fourier_norm = (2.0*std::numbers::pi)/double(values.shape()[2]);
    const double prefactor = sh_norm*radial_integral_norm*fourier_norm;
    pocketfft::r2c(
        m_pocketfft_shape_grid, m_pocketfft_stride_grid, m_pocketfft_stride_fft, axis, pocketfft::FORWARD, values.values().data(), m_ffts.data(), prefactor);
}

void GLQTransformer::apply_weights()
{
    const std::size_t glq_size = m_glq_weights.size();
    const std::size_t fft_size = glq_size - 1;
    for (std::size_t i = 0; i < glq_size; ++i)
    {
        const double r = 0.5*(1.0 + m_glq_nodes[i]);
        const double radial_weight = r*r*m_glq_weights[i];
        for (std::size_t j = 0; j < glq_size; ++j)
        {
            const double weight = radial_weight*m_glq_weights[j];
            for (std::size_t k = 0; k < fft_size; ++k)
            {
                std::complex<double>& x = m_ffts[k + (j + glq_size*i)*fft_size];
                x = {weight*x.real(), -weight*x.imag()};
            }
        }
    }
}

void GLQTransformer::integrate_latitudinal(std::size_t min_lmax)
{
    const std::size_t glq_size = m_glq_weights.size();
    const std::size_t fft_size = glq_size - 1;
    std::ranges::fill(m_flm_grid, std::array<double, 2>{});
    for (std::size_t i = 0; i < glq_size; ++i)
    {
        st::RealSHExpansionSpan<std::array<double, 2>, st::SHNorm::GEO, st::SHPhase::NONE> flm(m_flm_grid, i, lmax());
        for (std::size_t j = 0; j < glq_size; ++j)
        {
            std::span<const std::complex<double>> fft(
                    m_ffts.begin() + (i*glq_size + j)*fft_size, fft_size);
            st::PlmSpan<const double, st::SHNorm::GEO, st::SHPhase::NONE> plm(m_plm_grid, j, lmax());
            for (std::size_t l = 0; l <= min_lmax; ++l)
            {
                for (std::size_t m = 0; m <= l; ++m)
                {
                    const double plm_part = plm(l,m);
                    auto& coeff = flm(l,m);
                    coeff[0] += plm_part*fft[m].real();
                    coeff[1] += plm_part*fft[m].imag();
                }
            }
        }
    }
}

/*
void GLQTransformer::integrate_latitudinal_transpose(std::size_t min_lmax)
{
    for (std::size_t l = 0; l <= min_lmax; ++l)
    {
        for (std::size_t m = 0; m <= l; ++m)
        {
            std::span<std::array<double, 2>> flm = view_f(l,m);
            std::span<const std::array<double, 2>> plm = view_p(l,m);
            std::span<const std::complex<double>> fft = view_fft(m);
            for (std::size_t i = 0; i < glq_size; ++i)
            {
                std::array<double, 2>& flmi = flm[i];
                flmi = {};
                for (std::size_t j = 0; j < glq_size; ++j)
                {
                    flmi[0] += plm[j]*fft[j + i*glq_size][0];
                    flmi[1] += plm[j]*fft[j + i*glq_size][1];
                }
            }
        }
    }
}
*/

void GLQTransformer::integrate_radial(
    ZernikeExpansionSpan<std::array<double, 2>> expansion, std::size_t min_lmax)
{
    const std::size_t glq_size = m_glq_weights.size();
    std::ranges::fill(expansion.span(), std::array<double, 2>{});
    for (std::size_t i = 0; i < glq_size; ++i)
    {
        RadialZernikeSpan<const double> zernike(m_zernike_grid, i, lmax());
        st::RealSHExpansionSpan<std::array<double, 2>, st::SHNorm::GEO, st::SHPhase::NONE> flm(m_flm_grid, i, lmax());
        for (std::size_t n = 0; n <= min_lmax; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                const double radial_part = zernike(n,l);
                for (std::size_t m = 0; m <= l; ++m)
                {
                    const auto flm_part = flm(l,m);
                    auto& coeff = expansion(n,l,m);
                    coeff[0] += radial_part*flm_part[0];
                    coeff[1] += radial_part*flm_part[1];
                }
            }
        }
    }
}

/*
void GLQTransformer::integrate_radial_transpose(std::size_t min_lmax)
{
    const std::size_t glq_size = m_glq_weights.size();
    for (std::size_t n = 0; n <= min_lmax; ++n)
        {
            for (std::size_t l = n & 1; l <= n; l += 2)
            {
                std::span<const double> zernike = view_zernike(n,l);
                for (std::size_t m = 0; m <= l; ++m)
                {
                    std::span<const std::array<double, 2>> flm = view_flm(l,m);
                    std::array<double, 2>& coeff = expansion(n,l,m);
                    for (std::size_t i = 0; i < glq_size; ++i)
                    {
                        coeff[0] += zernike[i]*flm[i][0];
                        coeff[1] += zernike[i]*flm[i][1];
                    }
                }
            }
        }
}
*/

ZernikeExpansion GLQTransformer::transform(
    BallGLQGridSpan<const double> values, std::size_t lmax)
{
    ZernikeExpansion expansion(lmax);
    transform(values, expansion);
    return expansion;
}

void GLQTransformer::transform(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion,
    BallGLQGridSpan<double> values)
{
    resize(values.lmax());

    std::size_t min_lmax = std::min(expansion.lmax(), values.lmax());
    
    std::ranges::fill(m_ffts, std::complex<double>{});
    fft_fill(expansion, 0, m_glq_weights.size(), min_lmax);

    constexpr std::size_t axis = 2;
    constexpr double prefactor = 1.0;
    pocketfft::c2r(
        m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid, axis, pocketfft::BACKWARD, m_ffts.data(), values.values().data(), prefactor);
}

void GLQTransformer::fft_fill(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion,
    std::size_t imin, std::size_t imax, std::size_t min_lmax)
{
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
}

BallGLQGrid GLQTransformer::transform(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion,
    std::size_t lmax)
{
    BallGLQGrid grid(lmax);
    transform(expansion, grid);
    return grid;
}

UniformGridEvaluator::UniformGridEvaluator(
    std::size_t lmax, const std::array<std::size_t, 3>& shape):
    m_zernike_recursion(lmax), m_plm_recursion(lmax),
    m_zernike(RadialZernikeLayout::size(lmax)),
    m_plm(DualTriangleLayout::size(lmax)),
    m_ffts(shape[0]*shape[1]*(((shape[2] - 1) >> 1) + 1)),
    m_pocketfft_shape_grid(3), m_pocketfft_stride_grid(3), 
    m_pocketfft_stride_fft(3), m_shape(shape), m_lmax(lmax)
{
    m_pocketfft_shape_grid[0] = shape[0];
    m_pocketfft_shape_grid[1] = shape[1];
    m_pocketfft_shape_grid[2] = shape[2] - 1;

    m_pocketfft_stride_grid[0] = long(shape[1]*shape[2]*sizeof(double));
    m_pocketfft_stride_grid[1] = long(shape[2]*sizeof(double));
    m_pocketfft_stride_grid[2] = sizeof(double);

    const std::size_t num_fft = ((shape[2] - 1) >> 1) + 1;
    m_pocketfft_stride_fft[0] = long(shape[1]*num_fft*sizeof(std::complex<double>));
    m_pocketfft_stride_fft[1] = long(num_fft*sizeof(std::complex<double>));
    m_pocketfft_stride_fft[2] = sizeof(std::complex<double>);
}

void UniformGridEvaluator::resize(
    std::size_t lmax, const std::array<std::size_t, 3>& shape)
{
    if (lmax != m_lmax)
    {
        m_zernike_recursion.expand(lmax);
        m_plm_recursion.expand(lmax);
        m_zernike.resize(RadialZernikeLayout::size(lmax));
        m_plm.resize(DualTriangleLayout::size(lmax));
        m_lmax = lmax;
    }

    if (shape != m_shape)
    {
        m_pocketfft_shape_grid[0] = shape[0];
        m_pocketfft_shape_grid[1] = shape[1];
        m_pocketfft_shape_grid[2] = shape[2] - 1;

        m_pocketfft_stride_grid[0] = long(shape[1]*shape[2]*sizeof(double));
        m_pocketfft_stride_grid[1] = long(shape[2]*sizeof(double));
        m_pocketfft_stride_grid[2] = sizeof(double);

        const std::size_t num_fft = ((shape[2] - 1) >> 1) + 1;
        m_pocketfft_stride_fft[0] = long(shape[1]*num_fft*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[1] = long(num_fft*sizeof(std::complex<double>));
        m_pocketfft_stride_fft[2] = sizeof(std::complex<double>);

        m_ffts.resize(shape[0]*shape[1]*num_fft);

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

std::array<std::vector<double>, 4> UniformGridEvaluator::evaluate(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion, const std::array<std::size_t, 3>& shape)
{
    if (shape[2] < 2*expansion.lmax() + 1)
        throw std::invalid_argument(
            "last dimension in shape must be at least 2*lmax + 1");

    resize(expansion.lmax(), shape);

    const auto [num_rad, num_lat, num_lon] = shape;
    const std::size_t num_fft = ((num_lon - 1) >> 1) + 1;

    const std::vector<double> radii
            = linspace(0.0, 1.0, num_rad);
    const std::vector<double> longitudes
            = linspace(0.0, 2.0*std::numbers::pi, num_lon);
    const std::vector<double> colatitudes
            = linspace(0.0, std::numbers::pi, num_lat);
    std::vector<double> grid(num_rad*num_lat*num_lon);
    
    std::span coeffs = expansion.span();
    std::ranges::fill(m_ffts, std::complex<double>{});
    for (std::size_t i = 0; i < num_rad; ++i)
    {
        RadialZernikeSpan<double> zernike(m_zernike, expansion.lmax());
        const double r = radii[i];
        m_zernike_recursion.zernike<ZernikeNorm::NORMED>(zernike, r);
        for (std::size_t j = 0; j < num_lat; ++j)
        {
            std::span<std::complex<double>> fft(
                    m_ffts.begin() + (i*num_lat + j)*num_fft, num_fft);
            
            st::PlmSpan<double, st::SHNorm::GEO, st::SHPhase::NONE>
            plm(m_plm, expansion.lmax());
            
            const double z = std::cos(colatitudes[j]);
            m_plm_recursion.plm_real(plm, z);
            for (std::size_t n = 0; n <= expansion.lmax(); ++n)
            {
                for (std::size_t l = n & 1; l <= n; l += 2)
                {
                    const std::size_t zer_ind = RadialZernikeLayout::idx(n, l);
                    const std::size_t plm_ind = TriangleLayout::idx(l, 0);
                    const std::size_t exp_ind = ZernikeLayout::idx(n, l, 0);
                    const double zer = zernike[zer_ind];

                    const auto& coeff = coeffs[exp_ind];
                    fft[0] += (zer*plm[plm_ind])*std::complex<double>{coeff[0], -coeff[1]};
                    for (std::size_t m = 1; m <= l; ++m)
                    {
                        const std::size_t exp_ind = ZernikeLayout::idx(n, l, m);
                        const std::size_t plm_ind
                                = TriangleLayout::idx(l, m);

                        const auto& coeff = coeffs[exp_ind];
                        fft[m] += (0.5*zer*plm[plm_ind])*std::complex<double>{coeff[0], -coeff[1]};
                    }
                }
            }
        }
    }

    constexpr std::size_t axis = 2;
    constexpr double prefactor = 1.0;
    pocketfft::c2r(
        m_pocketfft_shape_grid, m_pocketfft_stride_fft, m_pocketfft_stride_grid, axis, pocketfft::BACKWARD, m_ffts.data(),
        grid.data(), prefactor);
    
    for (std::size_t i = 0; i < num_rad; ++i)
    {
        for (std::size_t j = 0; j < num_lat; ++j)
            grid.data()[(i*num_lat + j)*num_lon + num_lon - 1]
                = grid.data()[(i*num_lat + j)*num_lon];
    }
        
    
    return {radii, longitudes, colatitudes, grid};
}

}
}