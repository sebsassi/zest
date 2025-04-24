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

#include <vector>
#include <span>

#include "md_span.hpp"

namespace zest
{

/**
    @brief A non-owning view of a matrix.

    @tparam T type of elements of the matrix
*/
template <typename T>
class MatrixSpan: public MDSpan<T, 2>
{
public:
    using MDSpan<T, 2>::MDSpan;
    using MDSpan<T, 2>::extents;
    using MDSpan<T, 2>::extent;

    constexpr MatrixSpan(
        T* data, std::size_t nrows, std::size_t ncols) noexcept:
        MDSpan<T, 2>::MDSpan(data, {nrows, ncols}) {}

    /**
        @brief Number of rows in the matrix
    */
    [[nodiscard]] constexpr std::size_t
    extent() const noexcept { return extent(0); }

    /**
        @brief Number of columns in the matrix
    */
    [[nodiscard]] constexpr std::size_t
    extent() const noexcept { return extent(1); }
};

/*
    @brief A container representing a matrix.

    @tparam T type of elements of the matrix
*/
template <typename T>
class Matrix
{
public:
    using View = MatrixSpan<T>;
    using ConstView = MatrixSpan<const T>;

    Matrix() = default;
    Matrix(std::size_t p_nrows, std::size_t p_ncols):
        m_nrows(p_nrows), m_ncols(p_ncols), m_data(p_nrows*p_ncols) {}
    
    operator View()
    {
        return View(m_data.data(), m_nrows, m_ncols);
    }
    
    operator ConstView() const
    {
        return ConstView(m_data.data(), m_nrows, m_ncols);
    }

    [[nodiscard]] std::array<std::size_t, 2> shape() const noexcept
    {
        return {m_nrows, m_ncols};
    }

    /**
        @brief Number of rows in the matrix
    */
    [[nodiscard]] std::size_t nrows() const noexcept { return m_nrows; }

    /**
        @brief Number of columns in the matrix
    */
    [[nodiscard]] std::size_t ncols() const noexcept { return m_ncols; }

    /**
        @brief Pointer to underlying buffer.
    */
    [[nodiscard]] const T* data() const noexcept { return m_data; }

    /**
        @brief Pointer to underlying buffer.
    */
    T* data() noexcept { return m_data; }

    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const noexcept
    {
        return m_data[m_ncols*i + j];
    }

    T& operator()(std::size_t i, std::size_t j) noexcept
    {
        return m_data[m_ncols*i + j];
    }

    [[nodiscard]] std::span<const T> operator[](std::size_t i) const noexcept
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    std::span<T> operator[](std::size_t i) noexcept
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    [[nodiscard]] std::span<const T> row(std::size_t i) const noexcept
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    std::span<T> row(std::size_t i) noexcept
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    void resize(std::size_t p_nrows, std::size_t p_ncols)
    {
        m_nrows = p_nrows;
        m_ncols = p_ncols;
        m_data.resize(p_nrows*p_ncols);
    }

private:
    std::size_t m_nrows{};
    std::size_t m_ncols{};
    std::vector<T> m_data{};
};

} // namespace zest
