#pragma once

#include <vector>
#include <span>

template <typename T>
class MatrixSpan
{
public:
    MatrixSpan(std::span<T> span, std::size_t nrows, std::size_t ncols):
        m_nrows(nrows), m_ncols(ncols), m_data(span.begin(), nrows*ncols) {}

    [[nodiscard]] std::size_t nrows() const noexcept { return m_nrows; }
    [[nodiscard]] std::size_t ncols() const noexcept { return m_ncols; }
    [[nodiscard]] std::size_t size() const noexcept { return m_data.size(); }
    std::span<T> data() { return m_data; }

    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const
    {
        return m_data[m_ncols*i + j];
    }

    T& operator()(std::size_t i, std::size_t j)
    {
        return m_data[m_ncols*i + j];
    }

#if (__GNUC__ > 11)
    [[nodiscard]] const T& operator[](std::size_t i, std::size_t j) const
    {
        return m_data[m_ncols*i + j];
    }

    T& operator[](std::size_t i, std::size_t j)
    {
        return m_data[m_ncols*i + j];
    }
#endif

    [[nodiscard]] std::span<const T> operator[](std::size_t i) const
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    std::span<T> operator[](std::size_t i)
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    [[nodiscard]] std::span<const T> row(std::size_t i) const
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    std::span<T> row(std::size_t i)
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

#if (__GNUC__ > 11)
    std::views::stride column(std::size_t j)
    {
        return std::span<T>(m_data.begin() + j, m_data.end())
            | std::views::stride(m_ncols);
    }
#endif

private:
    std::size_t m_nrows;
    std::size_t m_ncols;
    std::span<T> m_data;
};

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
        return View(m_data, m_nrows, m_ncols);
    }
    
    operator ConstView() const
    {
        return ConstView(m_data, m_nrows, m_ncols);
    }

    [[nodiscard]] std::array<std::size_t, 2> shape() const noexcept
    {
        return {m_nrows, m_ncols};
    }

    [[nodiscard]] std::size_t nrows() const noexcept { return m_nrows; }
    [[nodiscard]] std::size_t ncols() const noexcept { return m_ncols; }
    [[nodiscard]] const std::vector<T>& data() const noexcept { return m_data; }
    std::vector<T>& data() noexcept { return m_data; }

    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const
    {
        return m_data[m_ncols*i + j];
    }

    T& operator()(std::size_t i, std::size_t j)
    {
        return m_data[m_ncols*i + j];
    }

#if (__GNUC__ > 11)
    [[nodiscard]] const T& operator[](std::size_t i, std::size_t j) const
    {
        return m_data[m_ncols*i + j];
    }

    T& operator[](std::size_t i, std::size_t j)
    {
        return m_data[m_ncols*i + j];
    }
#endif

    [[nodiscard]] std::span<const T> operator[](std::size_t i) const
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    std::span<T> operator[](std::size_t i)
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    [[nodiscard]] std::span<const T> row(std::size_t i) const
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

    std::span<T> row(std::size_t i)
    {
        return std::span<T>(m_data.begin() + i*m_ncols, m_ncols);
    }

#if (__GNUC__ > 11)
    std::views::stride column(std::size_t j)
    {
        return std::span<T>(m_data.begin() + j, m_data.end())
            | std::views::stride(m_ncols);
    }
#endif

    void resize(std::size_t p_nrows, std::size_t p_ncols)
    {
        m_nrows = p_nrows;
        m_ncols = p_ncols;
        m_data.resize(p_nrows*p_ncols);
    }

private:
    std::size_t m_nrows;
    std::size_t m_ncols;
    std::vector<T> m_data;
};