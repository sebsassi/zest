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
namespace detail
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
    [[nodiscard]] std::vector<double> operator()(
        MDSpan<const double, 2> model, std::span<const double> data);

    /**
        @brief Fit parameters to data

        @param model model matrix
        @param parameters fitted parameters
        @param data data set to fit
    */
    void operator()(
        MDSpan<const double, 2> model, std::span<double> parameters, std::span<const double> data);

private:
    std::vector<double> m_model_data;
    std::vector<double> m_data;
    std::vector<double> work;
};

} // namespace detail
} // namespace zest
