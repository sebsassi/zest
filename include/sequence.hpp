/*
Copyright (c) 2024-2026 Sebastian Sassi

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

#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "indexing.hpp"

namespace zest
{

/**
    @brief Enum for tagging index sequences as either symmetric or zero based.
*/
enum class IndexingMode
{
    symmetric, // Index from `-n` to `n`, inclusive.
    zero_based // Index from `0` to `n`, inclusive.
};

template <IndexingMode indexing_mode_param>
struct IndexingModeTag
{
    static constexpr IndexingMode indexing_mode = indexing_mode_param;
};

template <typename T>
concept indexing_mode_tagged = std::derived_from<
    std::remove_cvref_t<T>,
    IndexingModeTag<std::remove_cvref_t<T>::indexing_mode>>;

template <indexing_mode_tagged T>
consteval IndexingMode indexing_mode_of() { return std::remove_cvref_t<T>::indexing_mode; }

enum class Parity { even = 0, odd = 1 };

template <typename T>
concept has_parity = requires (T x) { { x.parity() } -> std::same_as<Parity>; };

/**
    @brief Contiguous 1d sequence, which is indexed exactly as you would expect.

    @tparam indexing_mode Determines if indexing is symmetric about zero
    or starts at zero.

    With zero based indexing, a sequence to order `n` is indexed as
    ```
    0 1 2 3 4 ... n - 1
    ```
    With symmetric indexing, the sequence is indexed as
    ```
    -n + 1 ... -2 -1 0 1 2 ... n - 1
    ```
*/
template <IndexingMode indexing_mode_param>
struct StandardLinearSequence
{
    using index_type = std::conditional_t<(indexing_mode_param == IndexingMode::symmetric),
        int, std::size_t>;
    using size_type = std::size_t;
    using index_range = std::conditional_t<(indexing_mode_param == IndexingMode::symmetric),
        SymmetricIndexRange<int>, StandardIndexRange<std::size_t>>;

    static constexpr IndexingMode indexing_mode = indexing_mode_param;
    static constexpr size_type rank = 1;

    /**
        @brief Number of elements in sequence at the given order.

        @param order Order at which the sequence is truncated.
    */
    [[nodiscard]] static constexpr size_type
    size(size_type order) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
            return order;
        else
            return 2*order - std::min(1UL, order);
    }

    /**
        @brief Linear index of an element in sequence.

        @param l
    */
    [[nodiscard]] static constexpr index_type
    index(index_type l) noexcept
    {
        return l;
    }
};

/**
    @brief Contiguous 1d sequence, with only even or odd elements.

    This sequence can present either one of the index sequences
    ```
    0 2 4 6 8 ...
    ```
    and
    ```
    1 3 5 7 9 ...
    ```

    @warning This indexing implies that adjacent even and odd indices map to
    the same element. Indexing data with this sequence mixing even and odd
    indices is an error.
*/
struct ParityLinearSequence
{
    using index_type = std::size_t;
    using size_type = std::size_t;
    using index_range = ParityIndexRange<index_type>;

    static constexpr size_type rank = 1;

    /**
        @brief Number of elements in sequence at the given order.

        @param order Order at which the sequence is truncated.
    */
    [[nodiscard]] static constexpr size_type
    size(size_type order) noexcept
    {
        return (order + 1) >> 1;
    }

    /**
        @brief Linear index of an element in sequence.

        @param l
    */
    [[nodiscard]] static constexpr index_type
    index(index_type l) noexcept
    {
        return l >> 1;
    }
};

/**
    @brief Contiguous 2D sequence for index pairs on a triangular grid.

    @tparam indexing_mode_param Determines if indexing is symmetric about zero
    or starts at zero.

    This sequence represents index pairs `(l,m)` subject to the condition
    `abs(m) <= l`. For zero based indexing this translates to to the sequence
    ```
    (0,0)
    (1,0) (1,1)
    (2,0) (2,1) (2,2)
    (3,0) (3,1) (3,2) (3,3)
    ...
    ```
    For symmetric indexing this translates to the sequence.
    ```
                         (0,0)
                  (1,-1) (1,0) (1,1)
           (2,-2) (2,-1) (2,0) (2,1) (2,2)
    (3,-3) (3,-2) (3,-1) (3,0) (3,1) (3,2) (3,3)
    ...
    ```
*/
template <IndexingMode indexing_mode_param>
struct TriangleSequence
{
    using index_type = std::conditional_t<indexing_mode_param == IndexingMode::symmetric,
        int, std::size_t>;
    using size_type = std::size_t;
    using index_range = std::conditional_t<(indexing_mode_param == IndexingMode::symmetric),
        SymmetricIndexRange<int>, StandardIndexRange<std::size_t>>;

    static constexpr IndexingMode indexing_mode = indexing_mode_param;
    static constexpr size_type rank = 2;

private:
    template <std::size_t N> struct subsequence_helper;

    template <std::size_t N>
        requires (N == 1)
    struct subsequence_helper<N>
    {
        using type = StandardLinearSequence<indexing_mode_param>;
    };

public:
    template <std::size_t N>
        requires (N == 1)
    using subsequence_type = subsequence_helper<N>::type;

    /**
        @brief Number of elements in sequence at the given order.

        @param order Order at which the sequence is truncated.
    */
    [[nodiscard]] static constexpr size_type
    size(size_type order) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
            return (order*(order + 1)) >> 1;
        else
            return order*order;
    }

    /**
        @brief Linear index of an element in sequence.

        @param l
        @param m
    */
    [[nodiscard]] static constexpr index_type
    index(index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
        {
            assert(m <= l);
            return ((l*(l + 1)) >> 1) + m;
        }
        else
        {
            assert(0 <= l && -l <= m && m <= l);
            return l*(l + 1) + m;
        }
    }
    /**
        @brief Linear index of an element in sequence.

        @param l
    */
    [[nodiscard]] static constexpr index_type
    index(index_type l) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
            return ((l*(l + 1)) >> 1);
        else
            return l*(l + 1);
    }

    /**
        @brief Order of a subsequence at given index `l`.

        @param l
    */
    [[nodiscard]] static constexpr size_type
    subextent(index_type l) noexcept { return size_type(l + 1); }
};

/**
    @brief Contiguous 2D sequence for index pairs with even sum on a triangular
    grid.

    This sequence represents index pairs `(n,l)` subject to the condition
    `l <= n` such that `l + n` is even. This translates to the sequence
    ```
    (0,0)
          (1,1)
    (2,0)       (2,2)
          (3,1)       (3,3)
    (4,0)       (4,2)       (4,4)
    ...
    ```

    @warning This indexing implies that some index combinations are simply not
    valid. It is erroneous to access data using this sequence with indices
    `(n,l)`.
*/
struct EvenTriangleSequence
{
    using index_type = std::size_t;
    using size_type = std::size_t;
    using index_range = StandardIndexRange<index_type>;

    static constexpr size_type rank = 2;

private:
    template <std::size_t N> struct subsequence_helper;

    template <std::size_t N>
        requires (N == 1)
    struct subsequence_helper<N>
    {
        using type = ParityLinearSequence;
    };

public:
    template <std::size_t N>
        requires (N == 1)
    using subsequence_type = subsequence_helper<N>::type;

    /**
        @brief Number of elements in sequence at the given order.

        @param order Order at which the sequence is truncated.
    */
    [[nodiscard]] static constexpr size_type
    size(size_type order) noexcept
    {
        // OEIS A002620
        return ((order + 1)*(order + 1)) >> 2; 
    }

    /**
        @brief Linear index of an element in sequence.

        @param n
        @param l
    */
    [[nodiscard]] static constexpr index_type
    index(index_type n, index_type l) noexcept
    {
        assert(l <= n && ((n - l) & 1) == 0);
        return (((n + 1)*(n + 1)) >> 2) + (l >> 1);
    }

    /**
        @brief Linear index of an element in sequence.

        @param n
    */
    [[nodiscard]] static constexpr index_type
    index(index_type n) noexcept
    {
         return (((n + 1)*(n + 1)) >> 2);
    }

    /**
        @brief Order of a subsequence at given index `n`.

        @param n
    */
    [[nodiscard]] static constexpr
    size_type subextent(index_type n) noexcept { return size_type(n + 1); }
};

/**
    @brief Contiguous 2D sequence for index pairs on a triangular grid with
    only even or odd indices in one direction.

    @tparam indexing_mode_param Determines if indexing is symmetric about zero
    or starts at zero.

    This sequence represents index pairs `(l,m)` subject to the condition
    `abs(m) <= l` such that `l` has definite parity. That is, for zero based
    indexing it either represents the sequence
    ```
    (0,0)

    (2,0) (2,1) (2,2)

    (4,0) (4,1) (4,2) (4,3) (4,4)
    ...
    ```
    or the sequence
    ```
    (1,0) (1,1)

    (3,0) (3,1) (3,2) (3,3)

    (5,0) (5,1) (5,2) (5,3) (5,4) (5,5)
    ...
    ```
    For symmetric indexing it alternatively represents
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

    @warning In this layout the index obtained from a pair `(l,m)` is unique
    only for `l` of the same parity. Otherwise the index is not unique, e.g.,
    `(0,0)` and `(1,0)` fall on the same index. It is therefore erroneous to
    mix even and odd `l` values when accessing data.
*/
template <IndexingMode indexing_mode_param>
struct ParityRowTriangleSequence
{
    using SubLayout = StandardLinearSequence<indexing_mode_param>;
    using index_type = std::conditional_t<indexing_mode_param == IndexingMode::symmetric,
        int, std::size_t>;
    using size_type = std::size_t;
    using index_range = ParityIndexRange<index_type>;

    static constexpr IndexingMode indexing_mode = indexing_mode_param;
    static constexpr size_type rank = 2;

private:
    template <std::size_t N> struct subsequence_helper;

    template <std::size_t N>
        requires (N == 1)
    struct subsequence_helper<N>
    {
        using type = StandardLinearSequence<indexing_mode_param>;
    };

public:
    template <std::size_t N>
        requires (N == 1)
    using subsequence_type = subsequence_helper<N>::type;

    /**
        @brief Number of elements in sequence at the given order.

        @param order Order at which the sequence is truncated.
    */
    static constexpr size_type
    size(size_type order) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
            return ((order + 1)*(order + 1)) >> 2;
        else
            return (order*(order + 1)) >> 1;
    }

    /**
        @brief Linear index of an element in sequence.

        @param l
        @param m
    */
    static constexpr index_type
    index(index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
        {
            assert(m <= l);
            return ((l*l) >> 2) + m;
        }
        else
        {
            assert(0 <= l && -l <= m && m <= l);
            return ((l*(l + 1)) >> 1) + m;
        }
    }

    /**
        @brief Linear index of an element in sequence.

        @param l
    */
    static constexpr index_type
    index(index_type l) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
            return ((l*l) >> 2);
        else
            return (l*(l + 1)) >> 1;
    }

    /**
        @brief Order of a subsequence at given index `l`.

        @param l
    */
    [[nodiscard]] static constexpr size_type
    subextent(index_type l) noexcept { return size_type(l + 1); }
};

/**
    @brief Contiguous 3D sequence for index triples laid out in a tetrahedral
    shape.

    @tparam indexing_mode_param Determines if indexing is symmetric about zero
    or starts at zero.

    This sequences represents index triples `(n,l,m)`, which are subject to the
    conditions `abs(m) <= l <= n` such that `n + l` is even. For zero based
    indexing this corresponds to the sequence
    ```
    (0,0,0)

    (1,1,0) (1,1,1)

    (2,0,0)
    (2,2,0) (2,2,1) (2,2,2)
    ...
    ```
    For symmetric indexing it corresponds to
    ```
                      (0,0,0)

             (1,1,-1) (1,1,0) (1,1,1)

                      (2,2,0)
    (2,2,-2) (2,2,-1) (2,2,0) (2,2,1) (2,2,2)
    ...
    ```

    @warning This indexing implies that some index combinations are simply not
    valid. It is erroneous to access data using this sequence with indices
    `(n,l)`.
*/
template <IndexingMode indexing_mode_param>
struct ZernikeTetrahedralSequence
{
    using SubLayout = ParityRowTriangleSequence<indexing_mode_param>;
    using index_type = std::conditional_t<indexing_mode_param == IndexingMode::symmetric,
        int, std::size_t>;
    using size_type = std::size_t;
    using index_range = StandardIndexRange<index_type>;

    static constexpr IndexingMode indexing_mode = indexing_mode_param;
    static constexpr size_type rank = 3;

private:
    template <std::size_t N> struct subsequence_helper;

    template <std::size_t N>
        requires (N == 1)
    struct subsequence_helper<N>
    {
        using type = ParityRowTriangleSequence<indexing_mode_param>;
    };

    template <std::size_t N>
        requires (N == 2)
    struct subsequence_helper<N>
    {
        using type = StandardLinearSequence<indexing_mode_param>;
    };

public:
    template <std::size_t N>
        requires (N == 1 || N == 2)
    using subsequence_type = subsequence_helper<N>::type;

    /**
        @brief Number of elements in sequence at the given order.

        @param order Order at which the sequence is truncated.
    */
    [[nodiscard]] static constexpr size_type
    size(size_type order) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
            return (order + 1)*(order + 3)*(2*order + 1)/24; // OEIS A002623
        else
            return order*(order + 1)*(order + 2)/6; // OEIS A000292
    }

    /**
        @brief Linear index of an element in sequence.

        @param n
        @param l
        @param m
    */
    [[nodiscard]] static constexpr index_type
    index(index_type n, index_type l, index_type m) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
        {
            assert(m <= l && l <= n && ((n - l) % 2) == 0);
            return (n + 1)*(n + 3)*(2*n + 1)/24 + ((l*l) >> 2) + m;
        }
        else
        {
            assert(0 <= l && 0 <= n && -l <= m && m <= l && l <= n && ((n - l) % 2) == 0);
            return n*(n + 1)*(n + 2)/6 + ((l*(l + 1)) >> 1) + m;
        }
    }

    /**
        @brief Linear index of an element in sequence.

        @param n
        @param l
    */
    [[nodiscard]] static constexpr index_type
    index(index_type n, index_type l) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
        {
            assert(l <= n && ((n - l) % 2) == 0);
            return (n + 1)*(n + 3)*(2*n + 1)/24 + ((l*l) >> 2);
        }
        else
        {
            assert(0 <= l && 0 <= n && l <= n && ((n - l) % 2) == 0);
            return n*(n + 1)*(n + 2)/6 + ((l*(l + 1)) >> 1);
        }
    }

    /**
        @brief Linear index of an element in sequence.

        @param n
    */
    [[nodiscard]] static constexpr index_type
    index(index_type n) noexcept
    {
        if constexpr (indexing_mode == IndexingMode::zero_based)
            return (n + 1)*(n + 3)*(2*n + 1)/24;
        else
            return n*(n + 1)*(n + 2)/6;
    }

    /**
        @brief Order of a subsequence at given index `n`.

        @param n
    */
    [[nodiscard]] static constexpr size_type
    subextent(index_type n) noexcept { return size_type(n + 1); }

    /**
        @brief Order of a subsequence at given index pair `(n,l)`.

        @param n
        @param l
    */
    [[nodiscard]] static constexpr size_type
    subextent([[maybe_unused]] index_type n, index_type l) noexcept { return size_type(l + 1); }
};

} // namespace zest
