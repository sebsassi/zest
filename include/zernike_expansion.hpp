#pragma once

#include <concepts>
#include <vector>

#include "triangle_layout.hpp"

namespace zest
{
namespace zt
{

using RadialZernikeLayout = EvenDiagonalTriangleLayout;
using ZernikeLayout = EvenSemiDiagonalTetrahedralLayout;

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

    [[nodiscard]] EvenOddSpan<const T> operator()(std::size_t n) const noexcept
    {
        return EvenOddSpan(m_span.begin() + Layout::idx(n,0), Layout::line_length(n));
    }

    [[nodiscard]] EvenOddSpan<T> operator()(std::size_t n) noexcept
    {
        return EvenOddSpan(m_span.begin() + Layout::idx(n,0), Layout::line_length(n));
    }

    [[nodiscard]] EvenOddSpan<const T> operator[](std::size_t n) const noexcept
    {
        return EvenOddSpan(m_span.begin() + Layout::idx(n,0), Layout::line_length(n));
    }

    [[nodiscard]] EvenOddSpan<T> operator[](std::size_t n) noexcept
    {
        return EvenOddSpan(m_span.begin() + Layout::idx(n,0), Layout::line_length(n));
    }

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



template <typename T>
    requires std::same_as<std::remove_const_t<T>, std::array<double, 2>>
using ZernikeExpansionLMSpan = TriangleSpan<T, EvenPrimaryTriangleLayout>;

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
    using Layout = EvenSemiDiagonalTetrahedralLayout;

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

    [[nodiscard]] ZernikeExpansionLMSpan<const T> operator()(std::size_t n) const noexcept
    {
        return ZernikeExpansionLMSpan<const T>(m_span, Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] ZernikeExpansionLMSpan<T> operator()(std::size_t n) noexcept
    {
        return ZernikeExpansionLMSpan<T>(m_span, Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] ZernikeExpansionLMSpan<const T> operator[](std::size_t n) const noexcept
    {
        return ZernikeExpansionLMSpan<const T>(m_span, Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] ZernikeExpansionLMSpan<T> operator[](std::size_t n) noexcept
    {
        return ZernikeExpansionLMSpan<T>(m_span, Layout::idx(n, 0, 0), n);
    }

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
    using Layout = EvenSemiDiagonalTetrahedralLayout;
    using View = ZernikeExpansionSpan<Element>;
    using ConstView = ZernikeExpansionSpan<const Element>;

    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    ZernikeExpansion(): ZernikeExpansion(0) {}
    explicit ZernikeExpansion(std::size_t lmax):
        m_coeffs(Layout::size(lmax)), m_lmax(lmax) {}

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

    [[nodiscard]] auto operator()(std::size_t n) const noexcept
    {
        return ZernikeExpansionLMSpan<const Element>(m_coeffs, Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] auto operator()(std::size_t n) noexcept
    {
        return ZernikeExpansionLMSpan<Element>(m_coeffs, Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] auto operator[](std::size_t n) const noexcept
    {
        return ZernikeExpansionLMSpan<const Element>(m_coeffs, Layout::idx(n, 0, 0), n);
    }

    [[nodiscard]] auto operator[](std::size_t n) noexcept
    {
        return ZernikeExpansionLMSpan<Element>(m_coeffs, Layout::idx(n, 0, 0), n);
    }

private:
    std::vector<std::array<double, 2>> m_coeffs;
    std::size_t m_lmax;
};

}
}