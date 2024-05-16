#pragma once

#include <vector>

#include "matrix.hpp"

class LinearMultifit
{
public:
    LinearMultifit() = default;

    [[nodiscard]] std::vector<double> fit_parameters(
        MatrixSpan<const double> model, std::span<const double> data);

    void fit_parameters(
        MatrixSpan<const double> model, std::span<double> parameters, std::span<const double> data);

private:
    std::vector<double> m_model_data;
    std::vector<double> m_data;
    std::vector<double> work;
};