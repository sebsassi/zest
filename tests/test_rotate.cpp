#include "../plm_recursion.hpp"
#include "../lsq_transformer.hpp"
#include "../glq_transformer.hpp"
#include "../rotate.hpp"
#include "../uniform_grid_evaluator.hpp"

#include <random>
#include <cmath>



constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::fabs(a[0] - b[0]) < tol && std::fabs(a[1] - b[1]) < tol;
}

bool to_real_is_inverse_of_to_complex()
{
    constexpr std::size_t lmax = 5;
    
    using ExpansionSpan = zest::st::RealSHExpansionSpan<std::array<double, 2>, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;

    std::vector<std::array<double, 2>> buffer(ExpansionSpan::Layout::size(lmax));
    
    ExpansionSpan expansion(buffer, lmax);
    for (std::size_t l = 0; l <= lmax; ++l)
    {
        for (std::size_t m = 1; m <= l; ++m)
            expansion(l,m) = {
                0.3*double(l) + 0.5*double(m), 0.1*double(l) - 0.2*double(m)
            };
    }

    std::vector<std::array<double, 2>> test_buffer(ExpansionSpan::Layout::size(lmax));
    std::ranges::copy(buffer, test_buffer.begin());
    ExpansionSpan test_expansion(test_buffer, lmax);

    zest::st::RealSHExpansionSpan<std::complex<double>, zest::st::SHNorm::QM, zest::st::SHPhase::CS> complex_expansion
            = zest::st::to_complex_expansion<zest::st::SHNorm::QM, zest::st::SHPhase::CS>(expansion);
    zest::st::to_real_expansion<zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>(complex_expansion);

    bool success = true;
    for (std::size_t l = 0; l <= lmax; ++l)
    {
        for (std::size_t m = 0; m <= l; ++m)
            if (!is_close(expansion(l,m), test_expansion(l,m), 1.0e-13))
                success = false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            for (std::size_t m = 0; m <= l; ++m)
                std::printf(
                        "%lu %lu {%f, %f} {%f, %f}\n", l, m,
                        expansion(l,m)[0], expansion(l,m)[1],
                        test_expansion(l,m)[0], test_expansion(l,m)[1]);
        }
    }
    return success;
}

bool test_wigner_d_pi2_is_correct_to_lmax_4()
{
    constexpr std::size_t lmax = 4;

    constexpr double d_pi2_0_0_0 = 1.0;

    constexpr double d_pi2_1_0_0 = 0.0;
    constexpr double d_pi2_1_0_1 = 1.0/std::sqrt(2.0);
    constexpr double d_pi2_1_1_0 = -1.0/std::sqrt(2.0);
    constexpr double d_pi2_1_1_1 = 1.0/2.0;

    constexpr double d_pi2_2_0_0 = -1.0/2.0;
    constexpr double d_pi2_2_0_1 = 0.0;
    constexpr double d_pi2_2_0_2 = std::sqrt(3.0)/(2.0*std::sqrt(2.0));
    constexpr double d_pi2_2_1_0 = 0.0;
    constexpr double d_pi2_2_1_1 = -1.0/2.0;
    constexpr double d_pi2_2_1_2 = 1.0/2.0;
    constexpr double d_pi2_2_2_0 = std::sqrt(3.0)/(2.0*std::sqrt(2.0));
    constexpr double d_pi2_2_2_1 = -1.0/2.0;
    constexpr double d_pi2_2_2_2 = 1.0/4.0;

    constexpr double d_pi2_3_0_0 = 0.0;
    constexpr double d_pi2_3_0_1 = -std::sqrt(3.0)/4.0;
    constexpr double d_pi2_3_0_2 = 0.0;
    constexpr double d_pi2_3_0_3 = std::sqrt(5.0)/4.0;
    constexpr double d_pi2_3_1_0 = std::sqrt(3.0)/4.0;
    constexpr double d_pi2_3_1_1 = -1.0/8.0;
    constexpr double d_pi2_3_1_2 = -std::sqrt(5.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_3_1_3 = std::sqrt(3.0*5.0)/8.0;
    constexpr double d_pi2_3_2_0 = 0.0;
    constexpr double d_pi2_3_2_1 = std::sqrt(5.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_3_2_2 = -1.0/2.0;
    constexpr double d_pi2_3_2_3 = std::sqrt(3.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_3_3_0 = -std::sqrt(5.0)/4.0;
    constexpr double d_pi2_3_3_1 = std::sqrt(3.0*5.0)/8.0;
    constexpr double d_pi2_3_3_2 = -std::sqrt(3.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_3_3_3 = 1.0/8.0;

    constexpr double d_pi2_4_0_0 = 3.0/8.0;
    constexpr double d_pi2_4_0_1 = 0.0;
    constexpr double d_pi2_4_0_2 = -std::sqrt(5.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_0_3 = 0.0;
    constexpr double d_pi2_4_0_4 = std::sqrt(5.0*7.0)/(8.0*std::sqrt(2.0));
    constexpr double d_pi2_4_1_0 = 0.0;
    constexpr double d_pi2_4_1_1 = 3.0/8.0;
    constexpr double d_pi2_4_1_2 = -1.0/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_1_3 = -std::sqrt(7.0)/8.0;
    constexpr double d_pi2_4_1_4 = std::sqrt(7.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_2_0 = -std::sqrt(5.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_2_1 = 1.0/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_2_2 = 1.0/4.0;
    constexpr double d_pi2_4_2_3 = -std::sqrt(7.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_2_4 = std::sqrt(7.0)/8.0;
    constexpr double d_pi2_4_3_0 = 0.0;
    constexpr double d_pi2_4_3_1 = -std::sqrt(7.0)/8.0;
    constexpr double d_pi2_4_3_2 = std::sqrt(7.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_3_3 = -3.0/8.0;
    constexpr double d_pi2_4_3_4 = 1.0/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_4_0 = std::sqrt(5.0*7.0)/(8.0*std::sqrt(2.0));
    constexpr double d_pi2_4_4_1 = -std::sqrt(7.0)/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_4_2 = std::sqrt(7.0)/8.0;
    constexpr double d_pi2_4_4_3 = -1.0/(4.0*std::sqrt(2.0));
    constexpr double d_pi2_4_4_4 = 1.0/16.0;


    zest::st::SHRotor rotor(lmax);

    const auto& d_pi2 = rotor.wigner_d_pi2();
    
    bool success = is_close(d_pi2(0,0,0), d_pi2_0_0_0, 1.0e-13)
            && is_close(d_pi2(1,0,0), d_pi2_1_0_0, 1.0e-13)
            && is_close(d_pi2(1,0,1), d_pi2_1_0_1, 1.0e-13)
            && is_close(d_pi2(1,1,0), d_pi2_1_1_0, 1.0e-13)
            && is_close(d_pi2(1,1,1), d_pi2_1_1_1, 1.0e-13)
            && is_close(d_pi2(2,0,0), d_pi2_2_0_0, 1.0e-13)
            && is_close(d_pi2(2,0,1), d_pi2_2_0_1, 1.0e-13)
            && is_close(d_pi2(2,0,2), d_pi2_2_0_2, 1.0e-13)
            && is_close(d_pi2(2,1,0), d_pi2_2_1_0, 1.0e-13)
            && is_close(d_pi2(2,1,1), d_pi2_2_1_1, 1.0e-13)
            && is_close(d_pi2(2,1,2), d_pi2_2_1_2, 1.0e-13)
            && is_close(d_pi2(2,2,0), d_pi2_2_2_0, 1.0e-13)
            && is_close(d_pi2(2,2,1), d_pi2_2_2_1, 1.0e-13)
            && is_close(d_pi2(2,2,2), d_pi2_2_2_2, 1.0e-13)
            && is_close(d_pi2(3,0,0), d_pi2_3_0_0, 1.0e-13)
            && is_close(d_pi2(3,0,1), d_pi2_3_0_1, 1.0e-13)
            && is_close(d_pi2(3,0,2), d_pi2_3_0_2, 1.0e-13)
            && is_close(d_pi2(3,0,3), d_pi2_3_0_3, 1.0e-13)
            && is_close(d_pi2(3,1,0), d_pi2_3_1_0, 1.0e-13)
            && is_close(d_pi2(3,1,1), d_pi2_3_1_1, 1.0e-13)
            && is_close(d_pi2(3,1,2), d_pi2_3_1_2, 1.0e-13)
            && is_close(d_pi2(3,1,3), d_pi2_3_1_3, 1.0e-13)
            && is_close(d_pi2(3,2,0), d_pi2_3_2_0, 1.0e-13)
            && is_close(d_pi2(3,2,1), d_pi2_3_2_1, 1.0e-13)
            && is_close(d_pi2(3,2,2), d_pi2_3_2_2, 1.0e-13)
            && is_close(d_pi2(3,2,3), d_pi2_3_2_3, 1.0e-13)
            && is_close(d_pi2(3,3,0), d_pi2_3_3_0, 1.0e-13)
            && is_close(d_pi2(3,3,1), d_pi2_3_3_1, 1.0e-13)
            && is_close(d_pi2(3,3,2), d_pi2_3_3_2, 1.0e-13)
            && is_close(d_pi2(3,3,3), d_pi2_3_3_3, 1.0e-13)
            && is_close(d_pi2(4,0,0), d_pi2_4_0_0, 1.0e-13)
            && is_close(d_pi2(4,0,1), d_pi2_4_0_1, 1.0e-13)
            && is_close(d_pi2(4,0,2), d_pi2_4_0_2, 1.0e-13)
            && is_close(d_pi2(4,0,3), d_pi2_4_0_3, 1.0e-13)
            && is_close(d_pi2(4,0,4), d_pi2_4_0_4, 1.0e-13)
            && is_close(d_pi2(4,1,0), d_pi2_4_1_0, 1.0e-13)
            && is_close(d_pi2(4,1,1), d_pi2_4_1_1, 1.0e-13)
            && is_close(d_pi2(4,1,2), d_pi2_4_1_2, 1.0e-13)
            && is_close(d_pi2(4,1,3), d_pi2_4_1_3, 1.0e-13)
            && is_close(d_pi2(4,1,4), d_pi2_4_1_4, 1.0e-13)
            && is_close(d_pi2(4,2,0), d_pi2_4_2_0, 1.0e-13)
            && is_close(d_pi2(4,2,1), d_pi2_4_2_1, 1.0e-13)
            && is_close(d_pi2(4,2,2), d_pi2_4_2_2, 1.0e-13)
            && is_close(d_pi2(4,2,3), d_pi2_4_2_3, 1.0e-13)
            && is_close(d_pi2(4,2,4), d_pi2_4_2_4, 1.0e-13)
            && is_close(d_pi2(4,3,0), d_pi2_4_3_0, 1.0e-13)
            && is_close(d_pi2(4,3,1), d_pi2_4_3_1, 1.0e-13)
            && is_close(d_pi2(4,3,2), d_pi2_4_3_2, 1.0e-13)
            && is_close(d_pi2(4,3,3), d_pi2_4_3_3, 1.0e-13)
            && is_close(d_pi2(4,3,4), d_pi2_4_3_4, 1.0e-13)
            && is_close(d_pi2(4,4,0), d_pi2_4_4_0, 1.0e-13)
            && is_close(d_pi2(4,4,1), d_pi2_4_4_1, 1.0e-13)
            && is_close(d_pi2(4,4,2), d_pi2_4_4_2, 1.0e-13)
            && is_close(d_pi2(4,4,3), d_pi2_4_4_3, 1.0e-13)
            && is_close(d_pi2(4,4,4), d_pi2_4_4_4, 1.0e-13);
    
    if (success)
        return true;
    else
    {
        std::printf("d_pi2_0_0_0 %f %f\n", d_pi2(0,0,0), d_pi2_0_0_0);
        std::printf("d_pi2_1_0_0 %f %f\n", d_pi2(1,0,0), d_pi2_1_0_0);
        std::printf("d_pi2_1_1_0 %f %f\n", d_pi2(1,1,0), d_pi2_1_1_0);
        std::printf("d_pi2_1_1_1 %f %f\n", d_pi2(1,1,1), d_pi2_1_1_1);
        std::printf("d_pi2_2_0_0 %f %f\n", d_pi2(2,0,0), d_pi2_2_0_0);
        std::printf("d_pi2_2_1_0 %f %f\n", d_pi2(2,1,0), d_pi2_2_1_0);
        std::printf("d_pi2_2_1_1 %f %f\n", d_pi2(2,1,1), d_pi2_2_1_1);
        std::printf("d_pi2_2_2_0 %f %f\n", d_pi2(2,2,0), d_pi2_2_2_0);
        std::printf("d_pi2_2_2_1 %f %f\n", d_pi2(2,2,1), d_pi2_2_2_1);
        std::printf("d_pi2_2_2_2 %f %f\n", d_pi2(2,2,2), d_pi2_2_2_2);
        std::printf("d_pi2_3_0_0 %f %f\n", d_pi2(3,0,0), d_pi2_3_0_0);
        std::printf("d_pi2_3_1_0 %f %f\n", d_pi2(3,1,0), d_pi2_3_1_0);
        std::printf("d_pi2_3_1_1 %f %f\n", d_pi2(3,1,1), d_pi2_3_1_1);
        std::printf("d_pi2_3_2_0 %f %f\n", d_pi2(3,2,0), d_pi2_3_2_0);
        std::printf("d_pi2_3_2_1 %f %f\n", d_pi2(3,2,1), d_pi2_3_2_1);
        std::printf("d_pi2_3_2_2 %f %f\n", d_pi2(3,2,2), d_pi2_3_2_2);
        std::printf("d_pi2_3_3_0 %f %f\n", d_pi2(3,3,0), d_pi2_3_3_0);
        std::printf("d_pi2_3_3_1 %f %f\n", d_pi2(3,3,1), d_pi2_3_3_1);
        std::printf("d_pi2_3_3_2 %f %f\n", d_pi2(3,3,2), d_pi2_3_3_2);
        std::printf("d_pi2_3_3_3 %f %f\n", d_pi2(3,3,3), d_pi2_3_3_3);
        std::printf("d_pi2_4_0_0 %f %f\n", d_pi2(4,0,0), d_pi2_4_0_0);
        std::printf("d_pi2_4_1_0 %f %f\n", d_pi2(4,1,0), d_pi2_4_1_0);
        std::printf("d_pi2_4_1_1 %f %f\n", d_pi2(4,1,1), d_pi2_4_1_1);
        std::printf("d_pi2_4_2_0 %f %f\n", d_pi2(4,2,0), d_pi2_4_2_0);
        std::printf("d_pi2_4_2_1 %f %f\n", d_pi2(4,2,1), d_pi2_4_2_1);
        std::printf("d_pi2_4_2_2 %f %f\n", d_pi2(4,2,2), d_pi2_4_2_2);
        std::printf("d_pi2_4_3_0 %f %f\n", d_pi2(4,3,0), d_pi2_4_3_0);
        std::printf("d_pi2_4_3_1 %f %f\n", d_pi2(4,3,1), d_pi2_4_3_1);
        std::printf("d_pi2_4_3_2 %f %f\n", d_pi2(4,3,2), d_pi2_4_3_2);
        std::printf("d_pi2_4_3_3 %f %f\n", d_pi2(4,3,3), d_pi2_4_3_3);
        std::printf("d_pi2_4_4_0 %f %f\n", d_pi2(4,4,0), d_pi2_4_4_0);
        std::printf("d_pi2_4_4_1 %f %f\n", d_pi2(4,4,1), d_pi2_4_4_1);
        std::printf("d_pi2_4_4_2 %f %f\n", d_pi2(4,4,2), d_pi2_4_4_2);
        std::printf("d_pi2_4_4_3 %f %f\n", d_pi2(4,4,3), d_pi2_4_4_3);
        std::printf("d_pi2_4_4_4 %f %f\n", d_pi2(4,4,4), d_pi2_4_4_4);
        return false;
    }
}

bool test_trivial_rotation_is_trivial_max_5()
{
    constexpr std::size_t lmax = 5;
    
    using ExpansionSpan = zest::st::RealSHExpansionSpan<std::array<double, 2>, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>;

    std::vector<std::array<double, 2>> buffer(ExpansionSpan::Layout::size(lmax));
    
    ExpansionSpan expansion(buffer, lmax);

    constexpr double norm = 1.0/std::sqrt(2.0*std::numbers::pi);
    constexpr double norm2 = 1.0/std::sqrt(4.0*std::numbers::pi);
    for (std::size_t l = 0; l <= lmax; ++l)
    {
        expansion(l,0) = {norm2*double(l)/3.0, 0.0};
        for (std::size_t m = 1; m <= l; ++m)
            expansion(l,m) = {
                norm*(double(l)/3.0 + double(m)/2.0), norm*(double(l)/10.0 - double(m)/5.0)
            };
    }

    std::vector<std::array<double, 2>> test_buffer(ExpansionSpan::Layout::size(lmax));
    std::ranges::copy(buffer, test_buffer.begin());
    ExpansionSpan test_expansion(test_buffer, lmax);

    zest::st::SHRotor rotor(lmax);
    rotor.rotate(expansion, std::array<double, 3>{});

    bool success = true;
    for (std::size_t l = 0; l <= lmax; ++l)
    {
        for (std::size_t m = 0; m <= l; ++m)
            if (!is_close(expansion(l,m), test_expansion(l,m), 1.0e-13))
                success = false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l <= lmax; ++l)
        {
            std::printf(
                    "%lu %lu {%f, %f} {%f, %f}\n", l, 0UL,
                    expansion(l,0)[0]/norm2, expansion(l,0)[1]/norm2,
                    test_expansion(l,0)[0]/norm2, test_expansion(l,0)[1]/norm2);
            for (std::size_t m = 1; m <= l; ++m)
                std::printf(
                        "%lu %lu {%f, %f} {%f, %f}\n", l, m,
                        expansion(l,m)[0]/norm, expansion(l,m)[1]/norm,
                        test_expansion(l,m)[0]/norm, test_expansion(l,m)[1]/norm);
        }
    }
    return success;
}

int main()
{
    assert(to_real_is_inverse_of_to_complex());

    assert(test_wigner_d_pi2_is_correct_to_lmax_4());
    assert(test_trivial_rotation_is_trivial_max_5());
}