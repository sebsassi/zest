#include "linearfit.hpp"

#include <cassert>
#include <cmath>
#include <random>

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

bool test_linearfit_3_parameter_function()
{
    constexpr std::size_t data_size = 1000;

    auto f0 = [](double x) { return std::cos(x); };
    auto f1 = [](double x) { return std::sin(x); };
    auto f2 = [](double x) { return x; };

    auto fit_f = [&](double x)
    {
        return 2.0*f0(x) + 0.5*f1(x) + 0.7*f2(x);
    };

    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};

    std::vector<double> points(data_size);
    for (auto& point : points)
        point = dist(gen);

    std::vector<double> data(data_size);
    for (std::size_t i = 0; i < data_size; ++i)
        data[i] = fit_f(points[i]);
    
    Matrix<double> model(data_size, 3);
    for (std::size_t i = 0; i < model.nrows(); ++i)
    {
        model[i][0] = f0(points[i]);
        model[i][1] = f1(points[i]);
        model[i][2] = f2(points[i]);
    }

    std::vector<double> parameters = LinearMultifit().fit_parameters(model, data);

    bool success = is_close(parameters[0], 2.0, 1.0e-10)
            && is_close(parameters[1], 0.5, 1.0e-10)
            && is_close(parameters[2], 0.7, 1.0e-10);
    
    if (success)
        return true;
    else
    {
        std::printf("%f %f\n", parameters[0], 2.0);
        std::printf("%f %f\n", parameters[1], 0.5);
        std::printf("%f %f\n", parameters[2], 0.7);
        return false;
    }
}

int main()
{
    assert(test_linearfit_3_parameter_function());
}