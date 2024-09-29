/*
Copyright (c) 2024 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/
#pragma once

#include <span>

namespace zest
{

/**
    @brief Contiguous 2D layout with indexing
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
    std::size_t size(std::size_t order) noexcept
    {
        return order*order;
    }

    [[nodiscard]] static constexpr
    std::size_t line_length(std::size_t l) noexcept
    {
        return 2*l + 1;
    }
};


/**
    @brief Contiguous 2D layout with indexing
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
    std::size_t size(std::size_t order) noexcept
    {
        return (order*(order + 1)) >> 1;
    }

    [[nodiscard]] static constexpr
    std::size_t line_length(std::size_t l) noexcept
    {
        return l + 1;
    }
};

/**
    @brief Contiguous 2D layout with indexing
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
    std::size_t size(std::size_t order) noexcept
    {
        // OEIS A002620
        return ((order + 1)*(order + 1)) >> 2; 
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

/**
    @brief Contiguous 2D layout with indexing
    ```
    (0,0)

    (2,0) (2,1) (2,2)

    (4,0) (4,1) (4,2) (4,3) (4,4)
    ```
    or
    ```
    (1,0) (1,1)

    (3,0) (3,1) (3,2) (3,3)

    (5,0) (5,1) (5,2) (5,3) (5,4) (5,5)
    ```
    @note In this layout the index obtained from a pair `(l,m)` is unique only for `l` of the same parity. Otherwise the index is not unique, e.g., `(0,0)` and `(1,0)` fall on the same index.
*/
struct EvenOddPrimaryTriangleLayout
{
    using index_type = std::size_t;
    static constexpr std::size_t size(std::size_t order) noexcept
    {
        return ((order + 1)*(order + 1)) >> 2;
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

/**
    @brief Contiguous 3D layout with indexing
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
    size(std::size_t order) noexcept
    {
        // OEIS A002623
        return (order + 1)*(order + 3)*(2*order + 1)/24;
    }

    [[nodiscard]] static constexpr std::size_t
    idx(std::size_t n, std::size_t l, std::size_t m) noexcept
    {
        return (n + 1)*(n + 3)*(2*n + 1)/24 + ((l*l) >> 2) + m;
    }
};

template <typename T>
concept triangular_layout
    = std::same_as<T, DualTriangleLayout>
    ||  std::same_as<T, TriangleLayout>
    ||  std::same_as<T, EvenDiagonalTriangleLayout>
    || std::same_as<T, EvenOddPrimaryTriangleLayout>;

/**
    @brief A non-owning view where adjacent even and odd indices refer to the same value. Given index `i`, the corresponding offset in the underlying buffer is given by `i/2`.
*/
template <typename ElementType>
class EvenOddSpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;

    constexpr EvenOddSpan() noexcept = default;
    constexpr EvenOddSpan(
        element_type* data, std::size_t size) noexcept:
        m_data(data), m_size(size), m_size(size) {}
    constexpr EvenOddSpan(
        std::span<element_type> buffer, std::size_t size) noexcept:
        m_data(buffer.begin()), m_size(size), m_size(size) {}

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }
    
    [[nodiscard]] constexpr std::size_t size() const noexcept { return m_size; }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return std::span(m_data, m_size);
    }

    [[nodiscard]] constexpr
    operator EvenOddSpan<const element_type>() const noexcept
    {
        return EvenOddSpan<const element_type>(m_data, m_size);
    }

    [[nodiscard]] constexpr element_type
    operator[](std::size_t i) const noexcept
    {
        return m_data[i >> 1];
    }
private:
    element_type* m_data;
    std::size_t m_size;
};

/**
    @brief A non-owning view modeling 2D data with triangular layout.

    @tparam ElementType type of elements in the view
    @tparam LayoutType layout of the elements
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
    using SubSpan = std::span<element_type>;
    using ConstView = TriangleSpan<const element_type, LayoutType>;

    static constexpr std::size_t size(std::size_t order) noexcept
    {
        return Layout::size(order);
    }

    constexpr TriangleSpan() noexcept = default;
    constexpr TriangleSpan(element_type* data, std::size_t order) noexcept:
        m_data(data), m_size(Layout::size(order)), m_order(order) {}
    constexpr TriangleSpan(
        std::span<element_type> buffer, std::size_t order) noexcept:
        m_data(buffer.data()), m_size(Layout::size(order)), m_order(order) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }
    
    [[nodiscard]] constexpr std::size_t size() const noexcept { return m_size; }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return std::span(m_data, m_size);
    }

    [[nodiscard]] constexpr
    operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_order);
    }

    [[nodiscard]] constexpr SubSpan
    operator()(index_type l) const noexcept
    {
        return SubSpan(m_data + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] constexpr SubSpan
    operator[](index_type l) const noexcept
    {
        return SubSpan(m_data + Layout::idx(l,0), Layout::line_length(l));
    }

    [[nodiscard]] constexpr element_type&
    operator()(index_type l, index_type m) const noexcept
    {
        return m_data[Layout::idx(l,m)];
    }

protected:
    friend TriangleSpan<std::remove_const_t<element_type>, LayoutType>;

    constexpr TriangleSpan(
        element_type* data, std::size_t size, std::size_t order) noexcept: 
        m_data(data), m_size(size), m_order(order) {}

private:
    element_type* m_data;
    std::size_t m_size;
    std::size_t m_order;
};

/**
    @brief A non-owning view modeling 3D data with triangular layout on the first two indices.

    @tparam ElementType type of elements in the view
    @tparam LayoutType layout of the elements
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
    using ConstView = TriangleVecSpan<const element_type, LayoutType>;

    constexpr TriangleVecSpan() noexcept = default;
    constexpr TriangleVecSpan(
        element_type* data, std::size_t order, std::size_t vec_size) noexcept:
        m_data(data), m_size(Layout::size(order)*vec_size), m_order(order),
        m_vec_size(vec_size) {}
    constexpr TriangleVecSpan(
        std::span<element_type> buffer, std::size_t order,
        std::size_t vec_size) noexcept:
        m_data(buffer.data()), m_size(Layout::size(order)*vec_size),
        m_order(order), m_vec_size(vec_size) {}

    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    [[nodiscard]] constexpr std::size_t
    vec_size() const noexcept { return m_vec_size; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }
    
    [[nodiscard]] constexpr std::size_t size() const noexcept { return m_size; }

    [[nodiscard]] constexpr
    operator std::span<element_type>() const noexcept
    {
        return std::span(m_data, m_size);
    }

    [[nodiscard]] constexpr
    operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_order, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator()(index_type l, index_type m) const noexcept
    {
        return std::span(m_data + Layout::idx(l,m)*m_vec_size, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator[](std::size_t idx) const noexcept
    {
        return std::span(m_data + idx*m_vec_size, m_vec_size);
    }

protected:
    friend TriangleVecSpan<std::remove_const_t<element_type>, LayoutType>;

    constexpr TriangleVecSpan(
        element_type* data, std::size_t size, std::size_t order,
        std::size_t vec_size) noexcept:
        m_data(data), m_size(size), m_order(order), m_vec_size(vec_size) {}

private:
    element_type* m_data;
    std::size_t m_size;
    std::size_t m_order;
    std::size_t m_vec_size;
};

enum class Parity { EVEN = 0, ODD = 1 };

template <typename T>
concept has_parity = requires (T x) { { x.parity() } -> std::same_as<Parity>; };

} // namespace zest