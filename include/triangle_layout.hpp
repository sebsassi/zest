#pragma once

#include <span>

namespace zest
{

/*
Contiguous 2D layout with indexing
```
                     (0,0)
              (1,-1) (1,0) (1,1)
       (2,-2) (2,-1) (2,0) (2,1) (2,2)
(3,-3) (3,-2) (3,-1) (3,0) (3,1) (3,2) (3,3)
...
```
*/
struct DualTriangleLayout
{
    using index_type = int;
    [[nodiscard]] static constexpr
    std::size_t idx(index_type l, index_type m) noexcept
    {
        return std::size_t(l*(l + 1) + m);
    }

    [[nodiscard]] static constexpr
    std::size_t size(std::size_t lmax) noexcept
    {
        return (lmax + 1)*(lmax + 1);
    }

    [[nodiscard]] static constexpr
    std::size_t line_length(std::size_t l) noexcept
    {
        return 2*l + 1;
    }
};


/*
Contiguous 2D layout with indexing
```
(0,0)
(1,0) (1,1)
(2,0) (2,1) (2,2)
(3,0) (3,1) (3,2) (3,3)
...
```
*/
struct TriangleLayout
{
    using index_type = std::size_t;
    [[nodiscard]] static constexpr
    std::size_t idx(index_type l, index_type m) noexcept
    {
        return ((l*(l + 1)) >> 1) + m;
    }

    [[nodiscard]] static constexpr
    std::size_t size(std::size_t lmax) noexcept
    {
        return ((lmax + 1)*(lmax + 2)) >> 1;
    }

    [[nodiscard]] static constexpr
    std::size_t line_length(std::size_t l) noexcept
    {
        return l + 1;
    }
};

/*
Contiguous 2D layout with indexing
```
(0,0)
      (1,1)
(2,0)       (2,2)
      (3,1)       (3,3)
(4,0)       (4,2)       (4,4)
```
*/
struct EvenDiagonalTriangleLayout
{
    using index_type = std::size_t;
    [[nodiscard]] static constexpr
    std::size_t size(std::size_t lmax) noexcept
    {
        // OEIS A002620
        return ((lmax + 2)*(lmax + 2)) >> 2; 
    }
    
    [[nodiscard]] static constexpr std::size_t
    idx(std::size_t n, std::size_t l) noexcept
    {
         return (((n + 1)*(n + 1)) >> 2) + (l >> 1);
    }

    [[nodiscard]] static constexpr
    std::size_t line_length(std::size_t l) noexcept
    {
        return (l >> 1) + 1;
    }
};

/*
Contiguous 2D layout with indexing
```
(0,0)

(2,0) (2,1) (2,2)

(4,0) (4,1) (4,2) (4,3) (4,4)
```
*/
struct EvenPrimaryTriangleLayout
{
    using index_type = std::size_t;
    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return ((lmax + 2)*(lmax + 2)) >> 2;
    }

    static constexpr std::size_t idx(std::size_t l, std::size_t m) noexcept
    {
        return ((l*l) >> 2) + m;
    }

    [[nodiscard]] static constexpr
    std::size_t line_length(std::size_t l) noexcept
    {
        return l + 1;
    }
};

/*
Contiguous 3D layout with indexing
```
(0,0,0)

(1,1,0) (1,1,1)

(2,0,0)
(2,2,0) (2,2,1) (2,2,2)...
```
*/
struct EvenSemiDiagonalTetrahedralLayout
{
    using index_type = std::size_t;
    [[nodiscard]] static constexpr std::size_t
    size(std::size_t lmax) noexcept
    {
        // OEIS A002623
        return (lmax + 2)*(lmax + 4)*(2*lmax + 3)/24;
    }

    [[nodiscard]] static constexpr std::size_t
    idx(std::size_t n, std::size_t l, std::size_t m) noexcept
    {
        return (n + 1)*(n + 3)*(2*n + 1)/24 + ((l*l) >> 2) + m;
    }
};

/*
A non-owning view where adjacent even and odd indices refer to the same value.

Given index `i`, the corresponding offset in the underlying buffer is given by `i/2`.
*/
template <typename ElementType>
class EvenOddSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;

    constexpr EvenOddSpan(
        std::span<element_type> buffer, std::size_t idx, std::size_t size):
        m_span(buffer.begin() + idx*size, size), m_size(size) {}
    constexpr EvenOddSpan(std::span<element_type> buffer, std::size_t size):
        m_span(buffer.begin(), size), m_size(size) {}
    
    [[nodiscard]] constexpr std::size_t size() const noexcept { return m_size; }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return m_span;
    }

    [[nodiscard]] constexpr
    operator EvenOddSpan<const element_type>() const noexcept
    {
        return EvenOddSpan<const element_type>(m_span, m_size);
    }

    [[nodiscard]] constexpr element_type operator[](std::size_t i) const noexcept
    {
        return m_span[i >> 1];
    }
private:
    std::span<element_type> m_span;
    std::size_t m_size;
};

/*
A non-owning view modeling 2D data with triangular layout.
*/
template <typename ElementType, typename LayoutType>
class TriangleSpan
{
public:
    using Layout = LayoutType;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;

    static constexpr std::size_t size(std::size_t lmax) noexcept
    {
        return Layout::size(lmax);
    }

    constexpr TriangleSpan(std::span<element_type> buffer, std::size_t lmax):
        m_span(buffer.begin(), Layout::size(lmax)), m_lmax(lmax) {}
    constexpr TriangleSpan(element_type* data, std::size_t lmax):
        m_span(data, Layout::size(lmax)), m_lmax(lmax) {}

    [[nodiscard]] constexpr std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return m_span; }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_span.data(); }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return m_span;
    }

    [[nodiscard]] constexpr
    operator TriangleSpan<const element_type, LayoutType>() const noexcept
    {
        return TriangleSpan<const element_type, LayoutType>(m_span, m_lmax);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator()(index_type l) const noexcept
    {
        return std::span<element_type>(
                m_span.begin() + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator[](index_type l) const noexcept
    {
        return std::span<element_type>(
                m_span.begin() + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] constexpr element_type&
    operator()(index_type l, index_type m) const noexcept
    {
        return m_span[Layout::idx(l,m)];
    }

private:
    std::span<element_type> m_span;
    std::size_t m_lmax;
};

/*
A non-owning view modeling 3D data with triangular layout on the first two indices.
*/
template <typename ElementType, typename LayoutType>
class TriangleVecSpan
{
public:
    using Layout = LayoutType;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;

    constexpr TriangleVecSpan(
        std::span<element_type> buffer, std::size_t idx, std::size_t lmax,
        std::size_t vec_size):
        m_span(buffer.begin() + idx*Layout::size(lmax)*vec_size, Layout::size(lmax)*vec_size), m_lmax(lmax), m_vec_size(vec_size) {}
    constexpr TriangleVecSpan(std::span<element_type> buffer, std::size_t lmax, std::size_t vec_size):
        m_span(buffer.begin(), Layout::size(lmax)*vec_size), m_lmax(lmax), 
        m_vec_size(vec_size) {}

    [[nodiscard]] constexpr std::size_t
    lmax() const noexcept { return m_lmax; }

    [[nodiscard]] constexpr std::size_t
    vec_size() const noexcept { return m_vec_size; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return m_span; }

    [[nodiscard]] const element_type*
    data() const noexcept { return m_span.data(); }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept
    {
        return m_span;
    }

    [[nodiscard]] constexpr
    operator TriangleVecSpan<const element_type, LayoutType>() const noexcept
    {
        return TriangleVecSpan<const element_type, LayoutType>(m_span, m_lmax, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator()(index_type l, index_type m) const noexcept
    {
        return std::span(
                m_span.begin() + Layout::idx(l,m)*m_vec_size, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator[](std::size_t idx) const noexcept
    {
        return std::span(m_span.begin() + idx*m_vec_size, m_vec_size);
    }

private:
    std::span<element_type> m_span;
    std::size_t m_lmax;
    std::size_t m_vec_size;
};

}