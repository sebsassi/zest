
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

#include <concepts>
#include <span>
#include <type_traits>

#include "indexing.hpp"

namespace zest
{

/**
    @brief Enum for tagging the `m` indexing style for spherical harmonics
    related things.
*/
enum class IndexingMode
{
    negative, // Index from `-l` to `l`
    nonnegative // Index from `0` to `l`
};

enum class Parity { even = 0, odd = 1 };

template <typename T>
concept has_parity = requires (T x) { { x.parity() } -> std::same_as<Parity>; };

/**
    @brief Contiguous 1d layout, which is indexed exactly as you think it is
    ```
    0 1 2 3 4 5...
    ```

    @tparam indexing_mode_param determines whether indexing may be negative
*/
template <IndexingMode indexing_mode_param>
struct StandardLinearSequence
{
    using index_type = std::conditional_t<
        indexing_mode_param == IndexingMode::negative, int, std::size_t>;
    using size_type = std::size_t;
    using index_range = std::conditional_t<(indexing_mode_param == IndexingMode::negative),
        SymmetricIndexRange<index_type>, StandardIndexRange<index_type>>;

    static constexpr size_type rank = 1;

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
    std::size_t index(index_type l) noexcept
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

    @warning This indexing implies that adjacent even and odd indices map to
    the same memory slot. Indexing data with this layout mixing even and odd
    indices is an error.
*/
struct ParityLinearSequence
{
    using index_type = std::size_t;
    using size_type = std::size_t;
    using index_range = ParityIndexRange<index_type>;

    static constexpr size_type rank = 1;

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
    std::size_t index(index_type l) noexcept
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
struct TriangleSequence
{
    using index_type = std::conditional_t<indexing_mode_param == IndexingMode::negative,
        int, std::size_t>;
    using size_type = std::size_t;
    using index_range = StandardIndexRange<index_type>;

    static constexpr IndexingMode indexing_mode = indexing_mode_param;
    static constexpr size_type rank = 2;

private:
    template <std::size_t N> struct sublayout;

    template <> struct sublayout<1>
    {
        using type = StandardLinearSequence<indexing_mode_param>;
    };

public:
    template <std::size_t N>
        requires (N == 1)
    using sublayout_t = sublayout<N>::type;

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
    std::size_t index(index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return ((l*(l + 1)) >> 1) + m;
        else
            return std::size_t(l*(l + 1) + m);
    }
    /**
        @brief Linear index of an element in layout.
    */
    [[nodiscard]] static constexpr
    std::size_t index(index_type l) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return ((l*(l + 1)) >> 1);
        else
            return std::size_t(l*(l + 1));
    }

    [[nodiscard]] static constexpr
    std::size_t subextent(index_type l) noexcept { return l + 1; }
};

/**
    @brief Contiguous 2D sequence with indexing
    ```
    (0,0)
          (1,1)
    (2,0)       (2,2)
          (3,1)       (3,3)
    (4,0)       (4,2)       (4,4)
    ...
    ```

    @warning This indexing implies that some index combinations are simply not
    valid. It is erroneous to access data using this layout with indices whose
    sum is an odd number.
*/
struct EvenTriangleSequence
{
    using index_type = std::size_t;
    using size_type = std::size_t;
    using index_range = StandardIndexRange<index_type>;

    static constexpr size_type rank = 2;

private:
    template <std::size_t N> struct sublayout_helper;

    template <> struct sublayout_helper<1>
    {
        using type = ParityLinearSequence;
    };

public:
    template <std::size_t N>
        requires (N == 1)
    using sublayout_t = sublayout_helper<N>::type;

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
    index(std::size_t n, std::size_t l) noexcept
    {
         return (((n + 1)*(n + 1)) >> 2) + (l >> 1);
    }

    /**
        @brief Linear index of an element in layout.
    */
    [[nodiscard]] static constexpr std::size_t
    index(std::size_t n) noexcept
    {
         return (((n + 1)*(n + 1)) >> 2);
    }

    [[nodiscard]] static constexpr
    std::size_t subextent(index_type n) noexcept { return n + 1; }
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

    @note In this layout the index obtained from a pair `(l,m)` is unique only
    for `l` of the same parity. Otherwise the index is not unique, e.g., `(0,0)`
    and `(1,0)` fall on the same index.
*/
template <IndexingMode indexing_mode_param>
struct EvenRowTriangleSequence
{
    using SubLayout = StandardLinearSequence<indexing_mode_param>;
    using index_type = std::conditional_t<indexing_mode_param == IndexingMode::negative,
        int, std::size_t>;
    using size_type = std::size_t;
    using index_range = ParityIndexRange<index_type>;

    static constexpr IndexingMode indexing_mode = indexing_mode_param;
    static constexpr size_type rank = 2;

private:
    template <std::size_t N> struct sublayout_helper;

    template <> struct sublayout_helper<1>
    {
        using type = StandardLinearSequence<indexing_mode_param>;
    };

public:
    template <std::size_t N>
        requires (N == 1)
    using sublayout_type = sublayout_helper<N>::type;

    /**
        @brief Number of elements in layout for size parameter `order`.

        @param order parameter presenting the size of the layout
    */
    static constexpr std::size_t 
    size(std::size_t order) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return ((order + 1)*(order + 1)) >> 2;
        else
            return (order*(order + 1)) >> 1;
    }

    /**
        @brief Linear index of an element in layout.
    */
    static constexpr std::size_t 
    index(index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return ((l*l) >> 2) + m;
        else
            return std::size_t(((l*(l + 1)) >> 1) + m);
    }

    /**
        @brief Linear index of an element in layout.
    */
    static constexpr std::size_t index(index_type l) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return ((l*l) >> 2);
        else
            return std::size_t(((l*(l + 1)) >> 1));
    }

    [[nodiscard]] static constexpr std::size_t
    subextent(index_type l) noexcept { return l + 1; }
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
struct ZernikeTetrahedralSequence
{
    using SubLayout = EvenRowTriangleSequence<indexing_mode_param>;
    using index_type = std::conditional_t<indexing_mode_param == IndexingMode::negative,
        int, std::size_t>;
    using size_type = std::size_t;
    using index_range = StandardIndexRange<index_type>;

    static constexpr IndexingMode indexing_mode = indexing_mode_param;
    static constexpr size_type rank = 2;

private:
    template <std::size_t N> struct sublayout_helpler;

    template <> struct sublayout_helpler<1>
    {
        using type = EvenRowTriangleSequence<indexing_mode_param>;
    };
    template <> struct sublayout_helpler<2>
    {
        using type = StandardLinearSequence<indexing_mode_param>;
    };

public:
    template <std::size_t N>
        requires (N == 1 || N == 2)
    using sublayout_t = sublayout_helpler<N>::type;

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
    index(index_type n, index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return (n + 1)*(n + 3)*(2*n + 1)/24 + ((l*l) >> 2) + m;
        else
            return std::size_t(n*(n + 1)*(n + 2)/6 + ((l*(l + 1)) >> 1) + m);
    }

    /**
        @brief Linear index of an element in layout.
    */
    [[nodiscard]] static constexpr std::size_t
    index(index_type n, index_type l) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return (n + 1)*(n + 3)*(2*n + 1)/24 + ((l*l) >> 2);
        else
            return std::size_t(n*(n + 1)*(n + 2)/6 + ((l*(l + 1)) >> 1));
    }

    /**
        @brief Linear index of an element in layout.
    */
    [[nodiscard]] static constexpr std::size_t
    index(index_type n) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::nonnegative)
            return (n + 1)*(n + 3)*(2*n + 1)/24;
        else
            return std::size_t(n*(n + 1)*(n + 2)/6);
    }

    [[nodiscard]] static constexpr std::size_t
    subextent(index_type n) noexcept { return n + 1; }

    [[nodiscard]] static constexpr std::size_t
    subextent([[maybe_unused]] index_type n, index_type l) noexcept { return l + 1; }
};

} // namespace zest
