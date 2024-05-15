#include "../plm_recursion.hpp"

#include <random>
#include <cmath>
#include <cassert>

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::fabs(a[0] - b[0]) < tol && std::fabs(a[1] - b[1]) < tol;
}

bool test_plm_real_generates_real_correct_up_to_lmax_4(double z)
{
    constexpr std::size_t lmax = 4;

    const double P00 = 1.0;

    const double P10 = std::sqrt(3.0)*z;
    const double P11 = std::sqrt(3.0)*std::sqrt(1.0 - z*z);

    const double P20 = std::sqrt(5.0/4.0)*(3.0*z*z - 1.0);
    const double P21 = std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z;
    const double P22 = std::sqrt(15.0/4.0)*(1.0 - z*z);

    const double P30 = std::sqrt(7.0/4.0)*(5.0*z*z - 3.0)*z;
    const double P31 = std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0);
    const double P32 = std::sqrt(105.0/4.0)*(1.0 - z*z)*z;
    const double P33 = std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z);

    const double P40 = std::sqrt(9.0/64.0)*((35.0*z*z - 30.0)*z*z + 3.0);
    const double P41 = std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z;
    const double P42 = std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0);
    const double P43 = std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z;
    const double P44 = std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z);

    zest::st::PlmRecursion recursion(lmax);

    std::vector<double> plm(zest::TriangleLayout::size(lmax));

    recursion.plm_real(zest::st::PlmSpan<double, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>(plm, recursion.lmax()), z);
    bool success = is_close(plm[0], P00, 1.0e-10)
            && is_close(plm[1], P10, 1.0e-10)
            && is_close(plm[2], P11, 1.0e-10)
            && is_close(plm[3], P20, 1.0e-10)
            && is_close(plm[4], P21, 1.0e-10)
            && is_close(plm[5], P22, 1.0e-10)
            && is_close(plm[6], P30, 1.0e-10)
            && is_close(plm[7], P31, 1.0e-10)
            && is_close(plm[8], P32, 1.0e-10)
            && is_close(plm[9], P33, 1.0e-10)
            && is_close(plm[10], P40, 1.0e-10)
            && is_close(plm[11], P41, 1.0e-10)
            && is_close(plm[12], P42, 1.0e-10)
            && is_close(plm[13], P43, 1.0e-10)
            && is_close(plm[14], P44, 1.0e-10);
    
    if (success)
        return true;
    else
    {
        std::printf("P00 %f %f\n", plm[0], P00);
        std::printf("P10 %f %f\n", plm[1], P10);
        std::printf("P11 %f %f\n", plm[2], P11);
        std::printf("P20 %f %f\n", plm[3], P20);
        std::printf("P21 %f %f\n", plm[4], P21);
        std::printf("P22 %f %f\n", plm[5], P22);
        std::printf("P30 %f %f\n", plm[6], P30);
        std::printf("P31 %f %f\n", plm[7], P31);
        std::printf("P32 %f %f\n", plm[8], P32);
        std::printf("P33 %f %f\n", plm[9], P33);
        std::printf("P40 %f %f\n", plm[10], P40);
        std::printf("P41 %f %f\n", plm[11], P41);
        std::printf("P42 %f %f\n", plm[12], P42);
        std::printf("P43 %f %f\n", plm[13], P43);
        std::printf("P44 %f %f\n", plm[14], P44);
        return false;
    }
}

bool test_plm_real_generates_real_vec_correct_up_to_lmax_4(double z)
{
    constexpr std::size_t lmax = 4;

    const double P00 = 1.0;

    const double P10 = std::sqrt(3.0)*z;
    const double P11 = std::sqrt(3.0)*std::sqrt(1.0 - z*z);

    const double P20 = std::sqrt(5.0/4.0)*(3.0*z*z - 1.0);
    const double P21 = std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z;
    const double P22 = std::sqrt(15.0/4.0)*(1.0 - z*z);

    const double P30 = std::sqrt(7.0/4.0)*(5.0*z*z - 3.0)*z;
    const double P31 = std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0);
    const double P32 = std::sqrt(105.0/4.0)*(1.0 - z*z)*z;
    const double P33 = std::sqrt(35.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z);

    const double P40 = std::sqrt(9.0/64.0)*((35.0*z*z - 30.0)*z*z + 3.0);
    const double P41 = std::sqrt(45.0/8.0)*std::sqrt(1.0 - z*z)*(7.0*z*z - 3.0)*z;
    const double P42 = std::sqrt(45.0/16.0)*(1.0 - z*z)*(7.0*z*z - 1.0);
    const double P43 = std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z;
    const double P44 = std::sqrt(315.0/64.0)*(1.0 - z*z)*(1.0 - z*z);

    zest::st::PlmRecursion recursion(lmax);

    std::vector<double> plm(zest::TriangleLayout::size(lmax));

    recursion.plm_real(zest::st::PlmVecSpan<double, zest::st::SHNorm::GEO, zest::st::SHPhase::NONE>(plm, recursion.lmax(), 1), std::array<double, 1>{z});
    bool success = is_close(plm[0], P00, 1.0e-10)
            && is_close(plm[1], P10, 1.0e-10)
            && is_close(plm[2], P11, 1.0e-10)
            && is_close(plm[3], P20, 1.0e-10)
            && is_close(plm[4], P21, 1.0e-10)
            && is_close(plm[5], P22, 1.0e-10)
            && is_close(plm[6], P30, 1.0e-10)
            && is_close(plm[7], P31, 1.0e-10)
            && is_close(plm[8], P32, 1.0e-10)
            && is_close(plm[9], P33, 1.0e-10)
            && is_close(plm[10], P40, 1.0e-10)
            && is_close(plm[11], P41, 1.0e-10)
            && is_close(plm[12], P42, 1.0e-10)
            && is_close(plm[13], P43, 1.0e-10)
            && is_close(plm[14], P44, 1.0e-10);
    
    if (success)
        return true;
    else
    {
        std::printf("P00 %f %f\n", plm[0], P00);
        std::printf("P10 %f %f\n", plm[1], P10);
        std::printf("P11 %f %f\n", plm[2], P11);
        std::printf("P20 %f %f\n", plm[3], P20);
        std::printf("P21 %f %f\n", plm[4], P21);
        std::printf("P22 %f %f\n", plm[5], P22);
        std::printf("P30 %f %f\n", plm[6], P30);
        std::printf("P31 %f %f\n", plm[7], P31);
        std::printf("P32 %f %f\n", plm[8], P32);
        std::printf("P33 %f %f\n", plm[9], P33);
        std::printf("P40 %f %f\n", plm[10], P40);
        std::printf("P41 %f %f\n", plm[11], P41);
        std::printf("P42 %f %f\n", plm[12], P42);
        std::printf("P43 %f %f\n", plm[13], P43);
        std::printf("P44 %f %f\n", plm[14], P44);
        return false;
    }
}

int main()
{
    assert(test_plm_real_generates_real_correct_up_to_lmax_4(1.0));
    assert(test_plm_real_generates_real_correct_up_to_lmax_4(-1.0));
    assert(test_plm_real_generates_real_correct_up_to_lmax_4(0.0));
    assert(test_plm_real_generates_real_correct_up_to_lmax_4(0.9741683087648949));

    assert(test_plm_real_generates_real_vec_correct_up_to_lmax_4(1.0));
    assert(test_plm_real_generates_real_vec_correct_up_to_lmax_4(-1.0));
    assert(test_plm_real_generates_real_vec_correct_up_to_lmax_4(0.0));
    assert(test_plm_real_generates_real_vec_correct_up_to_lmax_4(0.9741683087648949));
}