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

namespace zest
{
namespace zt
{

/*
A non-owning view of gridded data in spherical coordinates in the unit ball.
*/
template <typename T>
    requires std::same_as<std::remove_const_t<T>, double>
class BallGLQGridSpan
{
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;

    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return (lmax + 2)*(lmax + 2)*(2*lmax + 1);
    }

    static constexpr std::array<std::size_t, 3> shape(std::size_t lmax) noexcept
    {
        return {lmax + 2, lmax + 2, 2*lmax + 1};
    }

    BallGLQGridSpan(std::span<T> buffer, std::size_t lmax):
        m_values(buffer.begin(), (lmax + 2)*(lmax + 2)*(2*lmax + 1)), m_shape{lmax + 2, lmax + 2, 2*lmax + 1}, m_lmax(lmax) {}
    BallGLQGridSpan(std::span<T> buffer, std::size_t idx, std::size_t lmax):
        m_values(buffer.begin() + idx*(lmax + 2)*(lmax + 2)*(2*lmax + 1), (lmax + 2)*(lmax + 2)*(2*lmax + 1)), m_shape{lmax + 2, lmax + 2, 2*lmax + 1}, m_lmax(lmax) {}

    [[nodiscard]] std::array<std::size_t, 3> shape() { return m_shape; }
    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<const T> values() const noexcept { return m_values; }
    std::span<T> values() noexcept { return m_values; }

    [[nodiscard]] T operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept
    {
        return m_values[m_shape[2]*(m_shape[1]*i + j) + k];
    }

    T& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept
    {
        return m_values[m_shape[2]*(m_shape[1]*i + j) + k];
    }

private:
    std::span<T> m_values;
    std::array<std::size_t, 3> m_shape;
    std::size_t m_lmax;
};

/*
Container for gridded data in spherical coordinates in the unit ball.
*/
class BallGLQGrid
{
public:
    using View = BallGLQGridSpan<double>;
    using ConstView = BallGLQGridSpan<const double>;

    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return (lmax + 2)*(lmax + 2)*(2*lmax + 1);
    }

    static constexpr std::array<std::size_t, 3> shape(std::size_t lmax) noexcept
    {
        return {lmax + 2, lmax + 2, 2*lmax + 1};
    }

    BallGLQGrid(): BallGLQGrid(0) {}
    explicit BallGLQGrid(std::size_t lmax):
        m_values((lmax + 2)*(lmax + 2)*(2*lmax + 1)),
        m_shape{lmax + 2, lmax + 2, 2*lmax + 1}, m_lmax(lmax) {}

    [[nodiscard]] std::array<std::size_t, 3> shape() { return m_shape; }
    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<const double> values() const noexcept { return m_values; }
    std::span<double> values() noexcept { return m_values; }

    operator View()
    {
        return View(m_values, m_lmax);
    };

    operator ConstView() const
    {
        return ConstView(m_values, m_lmax);
    };

    void resize(std::size_t lmax);

    [[nodiscard]] double operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept
    {
        return m_values[m_shape[2]*(m_shape[1]*i + j) + k];
    }

    double& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept
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
    BallGLQGridPoints(): BallGLQGridPoints(0) {};
    explicit BallGLQGridPoints(std::size_t lmax);

    void resize(std::size_t lmax);

    [[nodiscard]] std::array<std::size_t, 3> shape() const noexcept
    {
        return {m_glq_nodes.size(), m_glq_nodes.size(), m_longitudes.size()};
    }
    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }

    [[nodiscard]] std::span<const double> longitudes() const noexcept
    {
        return m_longitudes;
    }
    [[nodiscard]] std::span<const double> glq_nodes() const noexcept
    {
        return m_glq_nodes;
    }

    template <typename FuncType>
    void generate_values(BallGLQGridSpan<double> grid, FuncType&& f)
    {
        resize(grid.lmax());
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        {
            const double r = 0.5*(1.0 + std::cos(m_glq_nodes[i]));
            for (std::size_t j = 0; j < m_glq_nodes.size(); ++j)
            {
                const double colatitude = m_glq_nodes[j];
                for (std::size_t k = 0; k < m_longitudes.size(); ++k)
                {
                    const double lon = m_longitudes[k];
                    grid(i, j, k) = f(r, lon, colatitude);
                }
            }
        }
    }

    template <typename FuncType>
    BallGLQGrid generate_values(FuncType&& f, std::size_t lmax)
    {
        BallGLQGrid grid(lmax);
        generate_values(grid, f);
        return grid;
    }

    template <typename FuncType>
    BallGLQGrid generate_values(FuncType&& f)
    {
        BallGLQGrid grid(m_lmax);
        generate_values(grid, f);
        return grid;
    }

#ifdef ZEST_USE_OMP
    template <typename FuncType>
    void generate_values(BallGLQGridSpan<double> grid, FuncType&& f, std::size_t num_threads)
    {
        std::size_t nthreads = (num_threads) ?
                num_threads : std::size_t(omp_get_max_threads());
        resize(grid.lmax());
        #pragma omp parallel for num_threads(nthreads)
        for (std::size_t i = 0; i < m_glq_nodes.size(); ++i)
        {
            const double r = 0.5*(1.0 + std::cos(m_glq_nodes[i]));
            for (std::size_t j = 0; j < m_glq_nodes.size(); ++j)
            {
                const double colatitude = m_glq_nodes[j];
                for (std::size_t k = 0; k < m_longitudes.size(); ++k)
                {
                    const double lon = m_longitudes[k];
                    grid(i, j, k) = f(r, lon, colatitude);
                }
            }
        }
    }

    template <typename FuncType>
    BallGLQGrid generate_values(FuncType&& f, std::size_t lmax, std::size_t num_threads)
    {
        BallGLQGrid grid(lmax);
        generate_values(grid, f, num_threads);
        return grid;
    }

    template <typename FuncType>
    BallGLQGrid generate_values(FuncType&& f, std::size_t num_threads)
    {
        BallGLQGrid grid(m_lmax);
        generate_values(grid, f, num_threads);
        return grid;
    }
#endif

private:
    std::vector<double> m_longitudes;
    std::vector<double> m_glq_nodes;
    std::size_t m_lmax;
};

/*
Class for transforming between a Gauss-Legendre quadrature grid representation and Zernike polynomial expansion representation of data in the unit baal.
*/
class GLQTransformer
{
public:
    explicit GLQTransformer(std::size_t lmax);

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] st::SHPhase phase() const noexcept { return m_phase; }

    void resize(std::size_t lmax);

    void transform(
        BallGLQGridSpan<const double> values,
        ZernikeExpansionSpan<std::array<double, 2>> expansion);

    void transform(
        ZernikeExpansionSpan<const std::array<double, 2>> expansion,
        BallGLQGridSpan<double> values);
    
    ZernikeExpansion transform(BallGLQGridSpan<const double> values, std::size_t lmax);
    BallGLQGrid transform(ZernikeExpansionSpan<const std::array<double, 2>> expansion, std::size_t lmax);

private:
    void integrate_longitudinal(BallGLQGridSpan<const double> values);
    void apply_weights();
    void integrate_latitudinal(std::size_t min_lmax);
    void integrate_radial(
        ZernikeExpansionSpan<std::array<double, 2>> expansion, std::size_t min_lmax);
    
    void fft_fill(
        ZernikeExpansionSpan<const std::array<double, 2>> expansion,
        std::size_t imin, std::size_t imax, std::size_t min_lmax);

    RadialZernikeRecursion m_zernike_recursion;
    st::PlmRecursion m_plm_recursion;
    std::vector<double> m_glq_nodes;
    std::vector<double> m_glq_weights;
    std::vector<double> m_zernike_grid;
    std::vector<double> m_plm_grid;
    std::vector<std::array<double, 2>> m_flm_grid;
    std::vector<std::complex<double>> m_ffts;
    std::vector<std::size_t> m_pocketfft_shape_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid;
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft;
    st::SHPhase m_phase = st::SHPhase::NONE;
    std::size_t m_lmax;
};

class UniformGridEvaluator
{
public:
    UniformGridEvaluator(): UniformGridEvaluator(0, {}) {};
    explicit UniformGridEvaluator(
        std::size_t lmax, const std::array<std::size_t, 3>& shape);

    void resize(std::size_t lmax, const std::array<std::size_t, 3>& shape);

    std::array<std::vector<double>, 4> evaluate(
        ZernikeExpansionSpan<const std::array<double, 2>> expansion, const std::array<std::size_t, 3>& shape);
private:
    RadialZernikeRecursion m_zernike_recursion;
    st::PlmRecursion m_plm_recursion;
    std::vector<double> m_zernike;
    std::vector<double> m_plm;
    std::vector<std::complex<double>> m_ffts;
    std::vector<std::size_t> m_pocketfft_shape_grid = {0, 0, 0};
    std::vector<std::ptrdiff_t> m_pocketfft_stride_grid = {0, 0, 0};
    std::vector<std::ptrdiff_t> m_pocketfft_stride_fft = {0, 0, 0};
    std::array<std::size_t, 3> m_shape;
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
class ZernikeTransformer
{
public:
    ZernikeTransformer(std::size_t lmax);

    void resize(std::size_t lmax);

    template <spherical_function Func>
    void transform(
        Func&& f, double radius, ZernikeExpansionSpan<std::array<double, 2>> expansion)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            return f(r*radius, lon, colat);
        };
        resize(expansion.lmax());
        m_points.generate_values(m_grid, f_scaled);
        m_transformer.transform(m_grid, expansion);
    }

    template <spherical_function Func>
    ZernikeExpansion transform(
        Func&& f, double radius, std::size_t lmax)
    {
        auto f_scaled = [&](double r, double lon, double colat) {
            return f(r*radius, lon, colat);
        };
        resize(lmax);
        m_points.generate_values(m_grid, f_scaled);
        return m_transformer.transform(m_grid, lmax);
    }

    template <cartesian_function Func>
    void transform(
        Func&& f, double radius, ZernikeExpansionSpan<std::array<double, 2>> expansion)
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
        m_transformer.transform(m_grid, expansion);
    }

    template <cartesian_function Func>
    ZernikeExpansion transform(
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
        return m_transformer.transform(m_grid, lmax);
    }

private:
    BallGLQGrid m_grid;
    BallGLQGridPoints m_points;
    GLQTransformer m_transformer;
};

}
}