/*
Copyright (c) 2024, 2025 Sebastian Sassi

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
#include <vector>

#include "md_array.hpp"
#include "md_span.hpp"

namespace zest::detail
{

/**
    @brief Class for applying a linear least-squares fit on data.
*/
class LinearMultifit
{
public:
    LinearMultifit() = default;

    /**
        @brief Fit parameters to data

        @param model model matrix
        @param data data set to fit

        @returns `std::vector<double>` containing the fitted parameters
    */
    template <std::size_t N, std::size_t M>
    [[nodiscard]] std::vector<double> operator()(
        MDSpan<const double, N, M> model, std::span<const double> data)
    {
        std::vector<double> parameters = std::vector<double>(model.extent(1));
        operator()(model, parameters, data);
        return parameters;
    }

    template <std::size_t N, std::size_t M>
    [[nodiscard]] std::vector<double> operator()(
        MDSpan<double, N, M> model, std::span<const double> data)
    {
        return operator()(MDSpan<const double, N, M>(model), data);
    }

    template <std::size_t N, std::size_t M>
    [[nodiscard]] std::vector<double> operator()(
        MDArray<double, N, M> model, std::span<const double> data)
    {
        return operator()(MDSpan<const double, N, M>(model), data);
    }

    /**
        @brief Fit parameters to data

        @param model model matrix
        @param parameters fitted parameters
        @param data data set to fit
    */
    template <std::size_t N, std::size_t M>
    void operator()(
        MDSpan<const double, N, M> model, std::span<double> parameters, std::span<const double> data)
    {
        assert(model.extent(0) <= data.size());
        assert(model.extent(1) <= parameters.size());
        m_model_data.resize(model.extent(0)*model.extent(1));
        m_data.resize(std::max(model.extent(0), model.extent(1)));

        std::span<const double> data_view(data.begin(), model.extent(0));
        std::span<double> parameters_view(parameters.begin(), model.extent(1));

        // Copy because dgels_ will modify data
        std::ranges::copy(std::span<const double>(model), m_model_data.begin());
        std::ranges::copy(data_view, m_data.begin());

        dgels_wrapper(model.extents());

        std::copy_n(m_data.begin(), parameters_view.size(), parameters_view.begin());
    }

    template <std::size_t N, std::size_t M>
    void operator()(
        MDSpan<double, N, M> model, std::span<double> parameters, std::span<const double> data)
    {
        operator()(MDSpan<const double, N, M>(model), parameters, data);
    }

    template <std::size_t N, std::size_t M>
    void operator()(
        MDArray<double, N, M> model, std::span<double> parameters, std::span<const double> data)
    {
        operator()(MDSpan<const double, N, M>(model), parameters, data);
    }

private:
    void dgels_wrapper(std::array<std::size_t, 2> extents);

    std::vector<double> m_model_data;
    std::vector<double> m_data;
    std::vector<double> work;
};

} // namespace zest::detail

