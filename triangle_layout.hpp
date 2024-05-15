#pragma once

#include <span>

namespace zest
{

/*
2D layout with indexing
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
    using IndexType = int;
    [[nodiscard]] static constexpr
    std::size_t idx(IndexType l, IndexType m) noexcept
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
2D layout with indexing
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
    using IndexType = std::size_t;

    [[nodiscard]] static constexpr
    std::size_t idx(IndexType l, IndexType m) noexcept
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
A non-owning view modeling 2D data with triangular layout.
*/
template <typename T, typename LayoutType>
class TriangleSpan
{
public:
    using Layout = LayoutType;
    using IndexType = Layout::IndexType;
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;

    TriangleSpan(std::span<T> buffer, std::size_t idx, std::size_t lmax):
        m_span(buffer.begin() + idx*Layout::size(lmax), Layout::size(lmax)), 
        m_lmax(lmax) {}
    TriangleSpan(std::span<T> buffer, std::size_t lmax):
        m_span(buffer.begin(), Layout::size(lmax)), m_lmax(lmax) {}

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::span<T> span() const noexcept { return m_span; }
    [[nodiscard]] const T* data() const noexcept { return m_span.data(); }

    operator std::span<T>() { return m_span; }
    operator TriangleSpan<const T, LayoutType>()
    {
        return TriangleSpan<const T, LayoutType>(m_span, m_lmax);
    }

    [[nodiscard]] std::span<const T> operator()(IndexType l) const
    {
        return std::span<const T>(
                m_span.begin() + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] std::span<T> operator()(IndexType l)
    {
        return std::span<T>(
                m_span.begin() + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] T operator()(IndexType l, IndexType m) const
    {
        return m_span[Layout::idx(l,m)];
    }
    T& operator()(IndexType l, IndexType m)
    {
        return m_span[Layout::idx(l,m)];
    }

    [[nodiscard]] T operator[](std::size_t idx) const
    {
        return m_span[idx];
    }

    T& operator[](std::size_t idx) { return m_span[idx]; }

private:
    std::span<T> m_span;
    std::size_t m_lmax;
};

/*
A non-owning view modeling 3D data with triangular layout on the first two indices.
*/
template <typename T, typename LayoutType>
class TriangleVecSpan
{
public:
    using Layout = LayoutType;
    using IndexType = Layout::IndexType;
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;

    TriangleVecSpan(
        std::span<T> buffer, std::size_t idx, std::size_t lmax,
        std::size_t vec_size):
        m_span(buffer.begin() + idx*Layout::size(lmax)*vec_size,
        Layout::size(lmax)*vec_size), m_lmax(lmax), m_vec_size(vec_size) {}
    TriangleVecSpan(std::span<T> buffer, std::size_t lmax, std::size_t vec_size):
        m_span(buffer.begin(), Layout::size(lmax)*vec_size), m_lmax(lmax), 
        m_vec_size(vec_size) {}

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }
    [[nodiscard]] std::size_t vec_size() const noexcept { return m_vec_size; }
    [[nodiscard]] std::span<T> span() const noexcept { return m_span; }
    [[nodiscard]] const T* data() const noexcept { return m_span.data(); }

    operator std::span<T>() { return m_span; }
    operator TriangleVecSpan<const T, LayoutType>()
    {
        return TriangleVecSpan<const T, LayoutType>(m_span, m_lmax, m_vec_size);
    }

    [[nodiscard]] std::span<const T> operator()(IndexType l, IndexType m) const
    {
        return std::span(
                m_span.begin() + Layout::idx(l,m)*m_vec_size, m_vec_size);
    }
    std::span<T> operator()(IndexType l, IndexType m)
    {
        return std::span(
                m_span.begin() + Layout::idx(l,m)*m_vec_size, m_vec_size);
    }

    [[nodiscard]] std::span<const T> operator[](std::size_t idx) const
    {
        return std::span(m_span.begin() + idx*m_vec_size, m_vec_size);
    }

    std::span<T> operator[](std::size_t idx)
    {
        return std::span(m_span.begin() + idx*m_vec_size, m_vec_size);
    }

private:
    std::span<T> m_span;
    std::size_t m_lmax;
    std::size_t m_vec_size;
};

}