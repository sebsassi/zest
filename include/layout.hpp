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
#include <concepts>
#include <type_traits>

#include "indexing.hpp"

namespace zest
{

/**
    @brief Enum for tagging the `m` indexing style for spherical harmonics related things.
*/
enum class IndexingMode
{
    negative, // Index from `-l` to `l`
    nonnegative // Index from `0` to `l`
};

enum class Parity { even = 0, odd = 1 };

/**
    @brief Tag for easily differentiating various layouts
*/
enum class LayoutTag
{
    linear, // Linear 1d layouts
    triangular, // Triangular 2d layouts
    tetrahedral // Tetrahedral 3d layouts
};

template <typename T>
concept has_parity = requires (T x) { { x.parity() } -> std::same_as<Parity>; };

template <typename T>
concept one_dimensional_span
    = std::same_as<T, std::span<typename T::element_type>>
    || requires (T span, typename T::index_type i)
    {
        requires std::convertible_to<
                decltype(span[i]), typename T::element_type>;
    };

template <typename T>
concept two_dimensional_span
    = requires (T span, typename T::index_type i, typename T::index_type j)
    {
        requires std::convertible_to<
                decltype(span(i,j)), typename T::element_type>;
    };

template <typename T>
concept two_dimensional_subspannable
    = requires (T span, typename T::index_type i)
    {
        requires one_dimensional_span<std::remove_cvref_t<decltype(span[i])>>;
    };

/**
    @brief Contiguous 1d layout, which is indexed exactly as you think it is
    ```
    0 1 2 3 4 5...
    ```

    @tparam indexing_mode_param determines whether indexing may be negative
*/
template <IndexingMode indexing_mode_param>
struct StandardLinearLayout
{
private:
    /*
    Ugly hack to select appropriate index range. There is probably a cleaner 
    way.
    */
    template <std::integral type_param, IndexingMode mode>
    struct SelectIndexRange;

    template <std::signed_integral type_param, IndexingMode mode>
        requires (mode == IndexingMode::negative)
    struct SelectIndexRange<type_param, mode> 
    { using type = SymmetricIndexRange<type_param>; };

    template <std::integral type_param, IndexingMode mode>
        requires (mode == IndexingMode::nonnegative)
    struct SelectIndexRange<type_param, mode> 
    { using type = StandardIndexRange<type_param>; };
public:
    using index_type = std::conditional_t<
        indexing_mode_param == IndexingMode::negative, int, std::size_t>;
    using size_type = std::size_t;
    using IndexRange = SelectIndexRange<index_type, indexing_mode_param>::type;
    
    static constexpr LayoutTag layout_tag = LayoutTag::linear;

    /**
        @brief Number of elements in layout for size parameter `order`.

        @param order parameter presenting the size of the layout
    */
    [[nodiscard]] static constexpr
    std::size_t size(std::size_t order) noexcept
    {
        if constexpr (indexing_mode_param == IndexingMode::nonnegative)
            return order;
        else
            return 2*order - std::min(1UL, order);
    }

    /**
        @brief Linear index of an element in layout.
    */
    [[nodiscard]] static constexpr
    std::size_t idx(index_type l) noexcept
    {
        return l;
    }
};

/**
    @brief Contiguous 1d layout, with indexing according to certain parity
    ```
    0 2 4 6 8...
    ```
    or
    ```
    1 3 5 7 9...
    ```

    @note This indexing implies that adjacent even and odd indices map to the same memory slot. Indexing data with this layout mixing even and odd indices is an error.
*/
struct ParityLinearLayout
{
    using index_type = std::size_t;
    using size_type = std::size_t;
    using IndexRange = ParityIndexRange<index_type>;
    
    static constexpr LayoutTag layout_tag = LayoutTag::linear;

    /**
        @brief Number of elements in layout for size parameter `order`.

        @param order parameter presenting the size of the layout
    */
    [[nodiscard]] static constexpr
    std::size_t size(std::size_t order) noexcept
    {
        return (order + 1) >> 1;
    }

    /**
        @brief Linear index of an element in layout.
    */
    [[nodiscard]] static constexpr
    std::size_t idx(index_type l) noexcept
    {
        return l >> 1;
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
    or
    ```
                         (0,0)
                  (1,-1) (1,0) (1,1)
           (2,-2) (2,-1) (2,0) (2,1) (2,2)
    (3,-3) (3,-2) (3,-1) (3,0) (3,1) (3,2) (3,3)
    ...
    ```

    @tparam indexing_mode_param determines whether indexing may be negative
*/
template <IndexingMode indexing_mode_param>
struct TriangleLayout
{
    using SubLayout = StandardLinearLayout<indexing_mode_param>;
    using index_type = std::conditional_t<
        indexing_mode_param == IndexingMode::negative, int, std::size_t>;
    using size_type = std::size_t;
    using IndexRange = StandardIndexRange<index_type>;
    
    static constexpr LayoutTag layout_tag = LayoutTag::triangular;
    static constexpr IndexingMode indexing_mode = indexing_mode_param;

    /**
        @brief Number of elements in layout for size parameter `order`.

        @param order parameter presenting the size of the layout
    */
    [[nodiscard]] static constexpr
    std::size_t size(std::size_t order) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return (order*(order + 1)) >> 1;
        else
            return order*order;
    }

    /**
        @brief Linear index of an element in layout.
    */
    [[nodiscard]] static constexpr
    std::size_t idx(index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return ((l*(l + 1)) >> 1) + m;
        else
            return std::size_t(l*(l + 1) + m);
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
    ...
    ```

    @note This indexing implies that some index combinations are simply not valid. It is erroneous to access data using this layout with indices whose sum is an odd number.
*/
struct OddDiagonalSkippingTriangleLayout
{
    using SubLayout = ParityLinearLayout;
    using index_type = std::size_t;
    using size_type = std::size_t;
    using IndexRange = StandardIndexRange<index_type>;

    static constexpr LayoutTag layout_tag = LayoutTag::triangular;

    /**
        @brief Number of elements in layout for size parameter `order`.

        @param order parameter presenting the size of the layout
    */
    [[nodiscard]] static constexpr
    std::size_t size(std::size_t order) noexcept
    {
        // OEIS A002620
        return ((order + 1)*(order + 1)) >> 2; 
    }
    
    /**
        @brief Linear index of an element in layout.
    */
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
    ...
    ```
    or
    ```
    (1,0) (1,1)

    (3,0) (3,1) (3,2) (3,3)

    (5,0) (5,1) (5,2) (5,3) (5,4) (5,5)
    ...
    ```
    or alternatively
    ```
                                (0,0)

                  (2,-2) (2,-1) (2,0) (2,1) (2,2)

    (4,-4) (4,-3) (4,-2) (4,-1) (4,0) (4,1) (4,2) (4,3) (4,4)
    ...
    ```
    or
    ```
                                (1,-1) (1,0) (1,1)

                  (3,-3) (3,-2) (3,-1) (3,0) (3,1) (3,2) (3,3)

    (5,-5) (5,-4) (5,-3) (5,-2) (5,-1) (5,0) (5,1) (5,2) (5,3) (5,4) (5,5)
    ...
    ```

    @tparam indexing_mode_param determines whether indexing may be negative

    @note In this layout the index obtained from a pair `(l,m)` is unique only for `l` of the same parity. Otherwise the index is not unique, e.g., `(0,0)` and `(1,0)` fall on the same index.
*/
template <IndexingMode indexing_mode_param>
struct RowSkippingTriangleLayout
{
    using SubLayout = StandardLinearLayout<indexing_mode_param>;
    using index_type = std::conditional_t<
        indexing_mode_param == IndexingMode::negative, int, std::size_t>;
    using size_type = std::size_t;
    using IndexRange = ParityIndexRange<index_type>;

    static constexpr LayoutTag layout_tag = LayoutTag::triangular;
    static constexpr IndexingMode indexing_mode = indexing_mode_param;

    /**
        @brief Number of elements in layout for size parameter `order`.

        @param order parameter presenting the size of the layout
    */
    static constexpr std::size_t size(std::size_t order) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return ((order + 1)*(order + 1)) >> 2;
        else
            return (order*(order + 1)) >> 1;
    }

    /**
        @brief Linear index of an element in layout.
    */
    static constexpr std::size_t idx(index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return ((l*l) >> 2) + m;
        else
            return std::size_t(((l*(l + 1)) >> 1) + m);
    }
};

/**
    @brief Contiguous 3D layout with indexing
    ```
    (0,0,0)

    (1,1,0) (1,1,1)

    (2,0,0)
    (2,2,0) (2,2,1) (2,2,2)
    ...
    ```

    @tparam indexing_mode_param determines whether indexing may be negative
*/
template <IndexingMode indexing_mode_param>
struct ZernikeTetrahedralLayout
{
    using SubLayout = RowSkippingTriangleLayout<indexing_mode_param>;
    using index_type = std::conditional_t<
        indexing_mode_param == IndexingMode::negative, int, std::size_t>;
    using size_type = std::size_t;
    using IndexRange = StandardIndexRange<index_type>;
    
    static constexpr LayoutTag layout_tag = LayoutTag::triangular;
    static constexpr IndexingMode indexing_mode = indexing_mode_param;

    /**
        @brief Number of elements in layout for size parameter `order`.

        @param order parameter presenting the size of the layout
    */
    [[nodiscard]] static constexpr std::size_t
    size(std::size_t order) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return (order + 1)*(order + 3)*(2*order + 1)/24; // OEIS A002623
        else
            return order*(order + 1)*(order + 2)/6; // OEIS A000292
    }

    /**
        @brief Linear index of an element in layout.
    */
    [[nodiscard]] static constexpr std::size_t
    idx(index_type n, index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return (n + 1)*(n + 3)*(2*n + 1)/24 + ((l*l) >> 2) + m;
        else
            return std::size_t(n*(n + 1)*(n + 2)/6 + ((l*(l + 1)) >> 1) + m);
    }
};

template <typename T>
concept layout_2d = requires (
        T::size_type s, typename T::index_type l)
    {
        { T::size(s) } -> std::same_as<typename T::size_type>;
        { T::idx(l, l) } -> std::same_as<typename T::size_type>;
    };

template <typename T>
concept triangular_layout = layout_2d<T>
    && (T::layout_tag == LayoutTag::triangular);

/**
    @brief A non-owning one-dimensional view of data elements.

    @tparam ElementType type of data elements
    @tparam LayoutType type identifying the data layout
*/
template <typename ElementType, typename LayoutType>
class LinearSpan
{
public:
    using Layout = LayoutType;
    using IndexRange = typename Layout::IndexRange;
    using index_type = typename Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using ConstView = LinearSpan<const element_type, Layout>;

    /**
        @brief Number of data elements for size parameter `order`.

        @param order parameter presenting the size of the span
    */
    static constexpr std::size_t size(std::size_t order) noexcept
    {
        return Layout::size(order);
    }

    constexpr LinearSpan() noexcept = default;
    constexpr LinearSpan(element_type* data, std::size_t order) noexcept:
        m_data(data), m_size(Layout::size(order)), m_order(order) {}
    constexpr LinearSpan(
        std::span<element_type> buffer, std::size_t order) noexcept:
        m_data(buffer.data()), m_size(Layout::size(order)), m_order(order) {}

    /**
        @brief Order of data layout.
    */
    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] constexpr std::size_t
    size() const noexcept { return m_size; }

    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr IndexRange indices()
    {
        return IndexRange{
            index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr IndexRange indices(index_type begin)
    {
        return IndexRange{
            begin, index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return std::span(m_data, m_size);
    }

    [[nodiscard]] constexpr
    operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size);
    }

    [[nodiscard]] constexpr element_type&
    operator()(index_type i) const noexcept
    {
        return m_data[Layout::idx(i)];
    }

    [[nodiscard]] constexpr element_type&
    operator[](index_type i) const noexcept
    {
        return m_data[Layout::idx(i)];
    }

protected:
    friend LinearSpan<std::remove_const_t<element_type>, LayoutType>;

    constexpr LinearSpan(
        element_type* data, std::size_t size, std::size_t order) noexcept: 
        m_data(data), m_size(size), m_order(order) {}

private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
};

/**
    @brief A non-owning one-dimensional view of one-dimensional segments of data

    @tparam ElementType type of data elements
    @tparam LayoutType type identifying the data layout
*/
template <typename ElementType, typename LayoutType>
class LinearVecSpan
{
public:
    using Layout = LayoutType;
    using IndexRange = typename Layout::IndexRange;
    using index_type = typename Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using ConstView = LinearSpan<const element_type, Layout>;

    /**
        @brief Number of data elements for size parameter `order`.

        @param order parameter presenting the size of the span
        @param vec_size number of elements in a single data segment
    */
    static constexpr std::size_t size(
        std::size_t order, std::size_t vec_size) noexcept
    {
        return vec_size*Layout::size(order);
    }

    constexpr LinearVecSpan() noexcept = default;
    constexpr LinearVecSpan(element_type* data, std::size_t order, std::size_t vec_size) noexcept:
        m_data(data), m_size(vec_size*Layout::size(order)), m_order(order), 
        m_vec_size(vec_size) {}
    constexpr LinearVecSpan(
        std::span<element_type> buffer, std::size_t order, std::size_t vec_size) noexcept:
        m_data(buffer.data()), m_size(vec_size*Layout::size(order)),
        m_order(order), m_vec_size(vec_size) {}

    /**
        @brief Order of data layout.
    */
    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    /**
        @brief Size of a data segment.
    */
    [[nodiscard]] constexpr std::size_t
    vec_size() const noexcept { return m_vec_size; }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] constexpr std::size_t
    size() const noexcept { return m_size; }

    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr IndexRange indices()
    {
        return IndexRange{
            index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr IndexRange indices(index_type begin)
    {
        return IndexRange{
            begin, index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return std::span(m_data, m_size);
    }

    [[nodiscard]] constexpr
    operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator()(index_type i) const noexcept
    {
        return std::span<element_type>(
            m_data + m_vec_size*Layout::idx(i), m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator[](index_type i) const noexcept
    {
        return std::span<element_type>(
            m_data + m_vec_size*Layout::idx(i), m_vec_size);
    }

protected:
    friend LinearSpan<std::remove_const_t<element_type>, LayoutType>;

    constexpr LinearVecSpan(
        element_type* data, std::size_t size, std::size_t order,
        std::size_t vec_size) noexcept: 
        m_data(data), m_size(size), m_order(order), m_vec_size(vec_size) {}

private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
    std::size_t m_vec_size{};
};

/**
    @brief A non-owning view where adjacent even and odd indices refer to the same value. Given index `i`, the corresponding offset in the underlying buffer is given by `i/2`.
*/
template <typename ElementType>
class ParitySpan
{
public:
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;

    constexpr ParitySpan() noexcept = default;
    constexpr ParitySpan(
        element_type* data, std::size_t size) noexcept:
        m_data(data), m_size(size), m_size(size) {}
    constexpr ParitySpan(
        std::span<element_type> buffer, std::size_t size) noexcept:
        m_data(buffer.begin()), m_size(size), m_size(size) {}

    /**
        @brief Order of data layout.
    */
    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] constexpr std::size_t
    size() const noexcept { return m_size; }

    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return std::span(m_data, m_size);
    }

    [[nodiscard]] constexpr
    operator ParitySpan<const element_type>() const noexcept
    {
        return ParitySpan<const element_type>(m_data, m_size);
    }

    [[nodiscard]] constexpr element_type&
    operator[](std::size_t i) const noexcept
    {
        return m_data[i >> 1];
    }
private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
};

/**
    @brief A non-owning two-dimensional view of data elements with triangular layout.

    @tparam ElementType type of data elements
    @tparam LayoutType type identifying the data layout
*/
template <typename ElementType, typename LayoutType>
class TriangleSpan
{
public:
    using Layout = LayoutType;
    using IndexRange = typename Layout::IndexRange;
    using index_type = typename Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using SubSpan = LinearSpan<element_type, typename Layout::SubLayout>;
    using ConstView = TriangleSpan<const element_type, Layout>;

    /**
        @brief Number of data elements for size parameter `order`.

        @param order parameter presenting the size of the span
    */
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

    /**
        @brief Order of data layout.
    */
    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] constexpr std::size_t
    size() const noexcept { return m_size; }

    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr IndexRange indices()
    {
        return IndexRange{
            index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr IndexRange indices(index_type begin)
    {
        return IndexRange{
            begin, index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

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
        return SubSpan(m_data + Layout::idx(l, 0), l + 1);
    }

    [[nodiscard]] constexpr SubSpan
    operator[](index_type l) const noexcept
    {
        return SubSpan(m_data + Layout::idx(l, 0), l + 1);
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
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
};

/**
    @brief A non-owning two-dimensional view of one-dimensional segments of data with triangular layout.

    @tparam ElementType type of data elements
    @tparam LayoutType type identifying the data layout
*/
template <typename ElementType, typename LayoutType>
class TriangleVecSpan
{
public:
    using Layout = LayoutType;
    using IndexRange = typename Layout::IndexRange;
    using index_type = Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using SubSpan = LinearVecSpan<element_type, typename Layout::SubLayout>;
    using ConstView = TriangleVecSpan<const element_type, LayoutType>;
    using LinearView = LinearVecSpan<
        element_type, StandardLinearLayout<IndexingMode::nonnegative>>;

    /**
        @brief Number of data elements for size parameter `order`.

        @param order parameter presenting the size of the span
        @param vec_size number of elements in a single data segment
    */
    static constexpr std::size_t size(
        std::size_t order, std::size_t vec_size) noexcept
    {
        return vec_size*Layout::size(order);
    }

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

    /**
        @brief Order of data layout.
    */
    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }

    /**
        @brief Size of a data segment.
    */
    [[nodiscard]] constexpr std::size_t
    vec_size() const noexcept { return m_vec_size; }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] constexpr std::size_t
    size() const noexcept { return m_size; }

    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr LinearView linear_view() const noexcept
    {
        return LinearView(m_data, Layout::size(m_order), m_vec_size);
    }

    [[nodiscard]] constexpr IndexRange indices()
    {
        return IndexRange{
            index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr IndexRange indices(index_type begin)
    {
        return IndexRange{
            begin, index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

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

    [[nodiscard]] constexpr SubSpan
    operator()(index_type l) const noexcept
    {
        return SubSpan(
            m_data + m_vec_size*Layout::idx(l, 0), l + 1, m_vec_size);
    }

    [[nodiscard]] constexpr SubSpan
    operator[](index_type l) const noexcept
    {
        return SubSpan(
            m_data + m_vec_size*Layout::idx(l, 0), l + 1, m_vec_size);
    }

protected:
    friend TriangleVecSpan<std::remove_const_t<element_type>, LayoutType>;

    constexpr TriangleVecSpan(
        element_type* data, std::size_t size, std::size_t order,
        std::size_t vec_size) noexcept:
        m_data(data), m_size(size), m_order(order), m_vec_size(vec_size) {}

private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
    std::size_t m_vec_size{};
};

/**
    @brief A non-owning three-dimensional view of data elements with triangular layout.

    @tparam ElementType type of elements in the view
    @tparam LayoutType layout of the elements
*/
template <typename ElementType, typename LayoutType>
class TetrahedronSpan
{
public:
    using Layout = LayoutType;
    using IndexRange = typename Layout::IndexRange;
    using index_type = typename Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using SubSpan = TriangleSpan<element_type, typename Layout::SubLayout>;
    using ConstView = TetrahedronSpan<const element_type, LayoutType>;

    /**
        @brief Number of data elements for size parameter `order`.

        @param order parameter presenting the size of the span
    */
    static constexpr std::size_t size(std::size_t order) noexcept
    {
        return Layout::size(order);
    }

    constexpr TetrahedronSpan() noexcept = default;
    constexpr TetrahedronSpan(element_type* data, std::size_t order) noexcept:
        m_data(data), m_size(Layout::size(order)), m_order(order) {}
    constexpr TetrahedronSpan(
        std::span<element_type> buffer, std::size_t order) noexcept:
        m_data(buffer.data()), m_size(Layout::size(order)), m_order(order) {}

    /**
        @brief Order of data layout.
    */
    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] constexpr std::size_t size() const noexcept { return m_size; }

    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr IndexRange indices()
    {
        return IndexRange{
            index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr IndexRange indices(index_type begin)
    {
        return IndexRange{
            begin, index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

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
    operator()(index_type n) const noexcept
    {
        return SubSpan(m_data + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] constexpr SubSpan
    operator[](index_type n) const noexcept
    {
        return SubSpan(m_data + Layout::idx(n, 0, 0), n + 1);
    }

    [[nodiscard]] constexpr element_type&
    operator()(index_type n, index_type l, index_type m) const noexcept
    {
        return m_data[Layout::idx(n, l, m)];
    }

protected:
    friend TetrahedronSpan<std::remove_const_t<element_type>, LayoutType>;

    constexpr TetrahedronSpan(
        element_type* data, std::size_t size, std::size_t order) noexcept: 
        m_data(data), m_size(size), m_order(order) {}

private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
};

/**
    @brief A non-owning three-dimensional view of data elements with triangular layout.

    @tparam ElementType type of elements in the view
    @tparam LayoutType layout of the elements
*/
template <typename ElementType, typename LayoutType>
class TetrahedronVecSpan
{
public:
    using Layout = LayoutType;
    using IndexRange = typename Layout::IndexRange;
    using index_type = typename Layout::index_type;
    using element_type = ElementType;
    using value_type = std::remove_cv_t<ElementType>;
    using size_type = std::size_t;
    using SubSpan = TriangleVecSpan<element_type, typename Layout::SubLayout>;
    using ConstView = TetrahedronVecSpan<const element_type, LayoutType>;
    using LinearView = LinearVecSpan<
        element_type, StandardLinearLayout<IndexingMode::nonnegative>>;

    /**
        @brief Number of data elements for size parameter `order`.

        @param order parameter presenting the size of the span
        @param vec_size number of elements in a single data segment
    */
    static constexpr std::size_t size(
        std::size_t order, std::size_t vec_size) noexcept
    {
        return vec_size*Layout::size(order);
    }

    constexpr TetrahedronVecSpan() noexcept = default;
    constexpr TetrahedronVecSpan(
        element_type* data, std::size_t order, std::size_t vec_size) noexcept:
        m_data(data), m_size(vec_size*Layout::size(order)), m_order(order), 
        m_vec_size(vec_size) {}
    constexpr TetrahedronVecSpan(
        std::span<element_type> buffer, std::size_t order,
        std::size_t vec_size) noexcept:
        m_data(buffer.data()), m_size(vec_size*Layout::size(order)),
        m_order(order), m_vec_size(vec_size) {}

    /**
        @brief Order of data layout.
    */
    [[nodiscard]] constexpr std::size_t
    order() const noexcept { return m_order; }
    
    /**
        @brief Size of the underlying buffer.
    */
    [[nodiscard]] constexpr std::size_t size() const noexcept { return m_size; }

    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] constexpr element_type*
    data() const noexcept { return m_data; }

    [[nodiscard]] constexpr std::span<element_type>
    flatten() const noexcept { return std::span(m_data, m_size); }

    [[nodiscard]] constexpr LinearView linear_view() const noexcept
    {
        return LinearView(m_data, Layout::size(m_order), m_vec_size);
    }

    [[nodiscard]] constexpr IndexRange indices()
    {
        return IndexRange{
            index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr IndexRange indices(index_type begin)
    {
        return IndexRange{
            begin, index_type(m_order + (IndexRange::iterator::stride - 1))};
    }

    [[nodiscard]] constexpr operator std::span<element_type>() const noexcept
    {
        return std::span(m_data, m_size);
    }

    [[nodiscard]] constexpr
    operator ConstView() const noexcept
    {
        return ConstView(m_data, m_size, m_order, m_vec_size);
    }

    [[nodiscard]] constexpr SubSpan
    operator()(index_type n) const noexcept
    {
        return SubSpan(
            m_data + m_vec_size*Layout::idx(n, 0, 0), n + 1, m_vec_size);
    }

    [[nodiscard]] constexpr SubSpan
    operator[](index_type n) const noexcept
    {
        return SubSpan(
            m_data + m_vec_size*Layout::idx(n, 0, 0), n + 1, m_vec_size);
    }

    [[nodiscard]] constexpr std::span<element_type>
    operator()(index_type n, index_type l, index_type m) const noexcept
    {
        return std::span(m_data + Layout::idx(n, l, m)*m_vec_size, m_vec_size);
    }

protected:
    friend TetrahedronVecSpan<std::remove_const_t<element_type>, LayoutType>;

    constexpr TetrahedronVecSpan(
        element_type* data, std::size_t size, std::size_t order,
        std::size_t vec_size) noexcept: 
        m_data(data), m_size(size), m_order(order), m_vec_size(vec_size) {}

private:
    element_type* m_data{};
    std::size_t m_size{};
    std::size_t m_order{};
    std::size_t m_vec_size{};
};

} // namespace zest