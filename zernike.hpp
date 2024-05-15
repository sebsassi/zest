#pragma once

#include <vector>
#include <array>
#include <span>
#include <complex>

#include "gauss_legendre.hpp"
#include "pocketfft.hpp"
#include "plm_recursion.hpp"

namespace zest
{
namespace zt
{

/* Zernike polynomial normalization switch */
enum class ZernikeNorm { NORMED, UNNORMED };

/*
Layout for storage of radial 3D Zernike polynomials.

The zernike polynomails are indexed by `(n,l)` with `n - l` even, and `0 <= l <= n`.

Given columns indexed by `n` and rows indexed by `l`, the rows are packed contiguous in memory in ascending order. That is, given an element indexed by `(n,l)`, the members are stored in memory as:
    `(0,0) (1,1) (2,0) (2,2) (3,1) (3,3) (4,0) (4,2) (4,4)...`
*/
struct RadialZernikeLayout
{
    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        // OEIS A002620
        return ((lmax + 2)*(lmax + 2)) >> 2; 
    }
    
    static constexpr std::size_t idx(std::size_t n, std::size_t l) noexcept
    {
         return (((n + 1)*(n + 1)) >> 2) + (l >> 1);
    }
};

/*
Layout for storage of 3D Zernike functions.

The zernike functions are indexed by `(n,l,m)` with `n - l` even, and `0 <= abs(m) <= l <= n`.

The values for `m` and `-m` are stored in pairs, indexed by positive `m`. Then the pairs indexed by `(n,l,m)` are contiguous in memory with the order:
    `(0,0,0) (1,1,0) (1,1,1) (2,0,0) (2,2,0) (2,2,1) (2,2,2)...`
That is, the rows of contiguous `m`, indexed by `(n,l)` have the order of `RadialZernikeLayout`
*/
struct ZernikeLayout
{
    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        // OEIS A002623
        return (lmax + 2)*(lmax + 4)*(2*lmax + 3)/24;
    }

    static constexpr std::size_t idx(
        std::size_t n, std::size_t l, std::size_t m) noexcept
    {
        return (n + 1)*(n + 3)*(2*n + 1)/24 + ((l*l) >> 2) + m;
    }
};

struct ZernikeLMLayout
{
    static constexpr std::size_t size(std::size_t n) noexcept
    {
        return ((n + 1)*(n + 1)) >> 2;
    }

    static constexpr std::size_t idx(std::size_t l, std::size_t m) noexcept
    {
        return ((l*l) >> 2) + m;
    }
};

/*
Non-owning view over values of radial 3D Zernike polynomials.
*/
template <typename T>
    requires std::same_as<std::remove_const_t<T>, double>
class RadialZernikeSpan
{
public:
    using Layout = RadialZernikeLayout;

    RadialZernikeSpan(std::span<T> buffer, std::size_t idx, std::size_t lmax):
        m_span(buffer.begin() + idx*Layout::size(lmax), Layout::size(lmax)), 
        m_lmax(lmax) {}
    RadialZernikeSpan(std::span<T> buffer, std::size_t lmax):
        m_span(buffer.begin(), Layout::size(lmax)), m_lmax(lmax) {}

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<T> span() const noexcept { return m_span; }
    [[nodiscard]] const T* data() const noexcept { return m_span.data(); }

    operator std::span<T>() noexcept { return m_span; }
    operator RadialZernikeSpan<const T>()
    {
        return RadialZernikeSpan<const T>(m_span, m_lmax);
    }

    [[nodiscard]] T operator()(std::size_t n, std::size_t l) const noexcept
    {
        return m_span[Layout::idx(n,l)];
    }
    T& operator()(std::size_t n, std::size_t l) noexcept
    {
        return m_span[Layout::idx(n,l)];
    }

    [[nodiscard]] T operator[](std::size_t idx) const noexcept
    {
        return m_span[idx];
    }

    T& operator[](std::size_t idx) noexcept { return m_span[idx]; }

private:
    std::span<T> m_span;
    std::size_t m_lmax;
};

/*
Non-owning view over vectors of values of radial 3D Zernike polynomials.
*/
template <typename T>
    requires std::same_as<std::remove_const_t<T>, double>
class RadialZernikeVecSpan
{
public:
    using Layout = RadialZernikeLayout;

    RadialZernikeVecSpan(
        std::span<T> buffer, std::size_t idx, std::size_t lmax,
        std::size_t vec_size):
        m_span(buffer.begin() + idx*Layout::size(lmax)*vec_size,
        Layout::size(lmax)*vec_size), m_lmax(lmax), m_vec_size(vec_size) {}
    RadialZernikeVecSpan(
        std::span<T> buffer, std::size_t lmax, std::size_t vec_size):
        m_span(buffer.begin(), Layout::size(lmax)*vec_size), m_lmax(lmax), 
        m_vec_size(vec_size) {}

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::size_t vec_size() const noexcept { return m_vec_size; }
    [[nodiscard]] std::span<T> span() const noexcept { return m_span; }
    [[nodiscard]] const T* data() const noexcept { return m_span.data(); }

    operator std::span<T>() noexcept { return m_span; }
    operator RadialZernikeVecSpan<const T>()
    {
        return RadialZernikeVecSpan<const T>(m_span, m_lmax, m_vec_size);
    }

    [[nodiscard]] std::span<const T> operator()(
        std::size_t n, std::size_t l) const noexcept
    {
        return std::span(
                m_span.begin() + Layout::idx(n,l)*m_vec_size, m_vec_size);
    }
    std::span<T> operator()(std::size_t n, std::size_t l) noexcept
    {
        return std::span(
                m_span.begin() + Layout::idx(n,l)*m_vec_size, m_vec_size);
    }

    [[nodiscard]] std::span<const T> operator[](std::size_t idx) const noexcept
    {
        return std::span(m_span.begin() + idx*m_vec_size, m_vec_size);
    }

    std::span<T> operator[](std::size_t idx) noexcept
    {
        return std::span(m_span.begin() + idx*m_vec_size, m_vec_size);
    }

private:
    std::span<T> m_span;
    std::size_t m_lmax;
    std::size_t m_vec_size;
};

/*
Class for recursive generation of radial 3D Zernike polynomials.
*/
class RadialZernikeRecursion
{
public:
    RadialZernikeRecursion(): RadialZernikeRecursion(0) {}
    explicit RadialZernikeRecursion(std::size_t lmax);

    void expand(std::size_t lmax);

    template <ZernikeNorm NORM>
    void zernike(RadialZernikeSpan<double> zernike, double r)
    {
        expand(zernike.lmax());

        const double r2 = r*r;

        zernike(0, 0) = 1.0;
        if (zernike.lmax() == 0)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
                zernike(0, 0) *= std::sqrt(3.0);
            return;
        }

        zernike(1, 1) = r;
        if (zernike.lmax() == 1)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                zernike(0, 0) *= std::sqrt(3.0);
                zernike(1, 1) *= std::sqrt(5.0);
            }
            return;
        }

        zernike(2, 0) = 2.5*r2 - 1.5;
        zernike(2, 2) = r2;
        if (zernike.lmax() == 2)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                zernike(0, 0) *= std::sqrt(3.0);
                zernike(1, 1) *= std::sqrt(5.0);
                zernike(2, 0) *= std::sqrt(7.0);
                zernike(2, 2) *= std::sqrt(7.0);
            }
            return;
        }

        zernike(3, 1) = (3.5*r2 - 2.5)*r;
        zernike(3, 3) = r2*r;

        for (std::size_t n = 4; n <= zernike.lmax(); ++n)
        {
            for (std::size_t l = n & 1; l <= n - 4; l += 2)
            {
                const std::size_t ind = RadialZernikeLayout::idx(n,l);
                zernike(n, l) = (m_k2[ind] + m_k1[ind]*r2)*zernike(n - 2, l) + m_k3[ind]*zernike(n - 4, l);

                if constexpr (NORM == ZernikeNorm::NORMED)
                    zernike(n - 4, l) *= m_norms[n - 4];
            }

            const double dn = double(n);
            zernike(n, n) = r*zernike(n - 1, n - 1);
            zernike(n, n - 2) = (dn + 0.5)*zernike(n, n)
                    - (dn - 0.5)*zernike(n - 2, n - 2);
        }

        if constexpr (NORM == ZernikeNorm::NORMED)
        {
            for (std::size_t n = zernike.lmax() - 3; n <= zernike.lmax(); ++n)
            {
                for (std::size_t l = n & 1; l <= n; l += 2)
                    zernike(n, l) *= m_norms[n];
            }
        }
    }

    template <ZernikeNorm NORM>
    void zernike(RadialZernikeVecSpan<double> zernike, std::span<const double> r)
    {
        if (r.size() != zernike.vec_size())
            throw std::invalid_argument(
                    "size of r is incompatible with size of zernike");
        
        expand(zernike.lmax());
        
        auto z_00 = zernike(0, 0);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_00[i] = 1.0;
        if (zernike.lmax() == 0)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::sqrt(3.0);
            }
            return;
        }

        auto z_11 = zernike(1, 1);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_11[i] = r[i];
        if (zernike.lmax() == 1)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::sqrt(3.0);
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_11[i] *= std::sqrt(5.0);
            }
            return;
        }

        auto z_22 = zernike(2, 2);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_22[i] = r[i]*r[i];

        auto z_20 = zernike(2, 0);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_20[i] = (2.5*z_22[i] - 1.5)*r[i];
        if (zernike.lmax() == 2)
        {
            if constexpr (NORM == ZernikeNorm::NORMED)
            {
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_00[i] *= std::sqrt(3.0);

                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_11[i] *= std::sqrt(5.0);
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_20[i] *= std::sqrt(7.0);
                
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_22[i] *= std::sqrt(7.0);
            }
            return;
        }

        auto z_31 = zernike(3, 1);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_31[i] = (3.5*z_22[i] - 2.5)*r[i];

        auto z_33 = zernike(3, 1);
        for (std::size_t i = 0; i < zernike.vec_size(); ++i)
            z_31[i] = z_22[i]*r[i];

        for (std::size_t n = 4; n <= zernike.lmax(); ++n)
        {
            for (std::size_t l = n & 1; l <= n - 4; l += 2)
            {
                const std::size_t ind = RadialZernikeLayout::idx(n,l);
                auto z_nl = zernike(n, l);
                auto z_nm2l = zernike(n - 2, l);
                auto z_nm4l = zernike(n - 4, l);
                for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                    z_nl[i] = (m_k2[ind] + m_k1[ind]*z_22[i])*z_nm2l[i] + m_k3[ind]*z_nm4l[i];

                if constexpr (NORM == ZernikeNorm::NORMED)
                {
                    for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                        z_nm4l[i] *= m_norms[n - 4];
                }
            }

            auto z_nn = zernike(n, n);
            auto z_nm1nm1 = zernike(n - 1, n - 1);
            for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                z_nn[i] = r[i]*z_nm1nm1[i];
            
            auto z_nm2nm2 = zernike(n - 2, n - 2);
            auto z_nnm2 = zernike(n, n - 2);

            const double dn = double(n);
            for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                z_nnm2[i] = (dn + 0.5)*z_nn[i]
                    - (dn - 0.5)*z_nm2nm2[i];
        }

        if constexpr (NORM == ZernikeNorm::NORMED)
        {
            for (std::size_t n = zernike.lmax() - 3; n <= zernike.lmax(); ++n)
            {
                for (std::size_t l = n & 1; l <= n - 4; l += 2)
                {
                    auto z_nl = zernike(n, l);
                    for (std::size_t i = 0; i < zernike.vec_size(); ++i)
                        z_nl[i] *= m_norms[n];
                }
            }
        }
    }


private:
    std::vector<double> m_norms;
    std::vector<double> m_k1;
    std::vector<double> m_k2;
    std::vector<double> m_k3;
    std::size_t m_lmax;
};

template <typename T>
    requires std::same_as<std::remove_const_t<T>, std::array<double, 2>>
class ZernikeExpansionSpan;

template <typename T>
    requires std::same_as<std::remove_const_t<T>, std::array<double, 2>>
class ZernikeExpansionLMSpan
{
public:
    using element_type = std::remove_cv_t<T>;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using Layout = ZernikeLMLayout;

    ZernikeExpansionLMSpan(ZernikeExpansionSpan<T> buffer, std::size_t offset, std::size_t n):
        m_span(buffer.begin() + offset, Layout::size(n)), m_n(n) {}
    
    [[nodiscard]] T operator()(std::size_t l, std::size_t m) const noexcept
    {
        return m_span[Layout::idx(l,m)];
    }
    T& operator()(std::size_t l, std::size_t m) noexcept
    {
        return m_span[Layout::idx(l,m)];
    }

private:
    std::span<T> m_span;
    std::size_t m_n;
};

/*
A non-owning view of a function expansion in the basis of real Zernike functions.
*/
template <typename T>
    requires std::same_as<std::remove_const_t<T>, std::array<double, 2>>
class ZernikeExpansionSpan
{
public:
    using element_type = std::remove_cv_t<T>;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using Layout = ZernikeLayout;

    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    ZernikeExpansionSpan(
        std::span<T> buffer, std::size_t idx, std::size_t lmax):
        m_span(buffer.begin() + idx*Layout::size(lmax), Layout::size(lmax)), 
        m_lmax(lmax) {}
    ZernikeExpansionSpan(std::span<T> buffer, std::size_t lmax):
        m_span(buffer.begin(), Layout::size(lmax)), m_lmax(lmax) {}

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<T> span() const noexcept { return m_span; }
    [[nodiscard]] const T* data() const noexcept { return m_span.data(); }

    operator std::span<T>() noexcept { return m_span; }
    operator ZernikeExpansionSpan<const T>()
    {
        return ZernikeExpansionSpan<const T>(m_span, m_lmax);
    }

    [[nodiscard]] T operator()(
        std::size_t n, std::size_t l, std::size_t m) const noexcept
    {
        return m_span[Layout::idx(n,l,m)];
    }
    T& operator()(std::size_t n, std::size_t l, std::size_t m) noexcept
    {
        return m_span[Layout::idx(n,l,m)];
    }

    [[nodiscard]] T operator[](std::size_t idx) const noexcept
    {
        return m_span[idx];
    }

    [[nodiscard]] ZernikeExpansionLMSpan<T> operator()(std::size_t n) noexcept
    {
        return ZernikeExpansionLMSpan<T>(m_span, Layout::idx(n, 0, 0), n);
    }

    T& operator[](std::size_t idx) { return m_span[idx]; }

private:
    std::span<T> m_span;
    std::size_t m_lmax;
};

/*
A container for a Zernike expansion of a real function.
*/
class ZernikeExpansion
{
public:
    using value_type = std::array<double, 2>;
    using size_type = std::size_t;
    using Element = std::array<double, 2>;
    using Layout = ZernikeLayout;
    using View = ZernikeExpansionSpan<Element>;
    using ConstView = ZernikeExpansionSpan<const Element>;

    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    ZernikeExpansion(): ZernikeExpansion(0) {}
    explicit ZernikeExpansion(std::size_t lmax);

    operator View()
    {
        return View(m_coeffs, m_lmax);
    };

    operator ConstView() const
    {
        return ConstView(m_coeffs, m_lmax);
    };
    
    [[nodiscard]] Element operator()(std::size_t n, std::size_t l, std::size_t m) const noexcept
    {
        return m_coeffs[Layout::idx(n,l,m)];
    }
    Element& operator()(std::size_t n, std::size_t l, std::size_t m) noexcept
    {
        return m_coeffs[Layout::idx(n,l,m)];
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
    std::vector<std::array<double, 2>> m_coeffs;
    std::size_t m_lmax;
};

void power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion,
    std::span<double> out);

std::vector<double> power_spectrum(
    ZernikeExpansionSpan<const std::array<double, 2>> expansion);

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

}
}