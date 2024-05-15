#pragma once

#include <vector>
#include <algorithm>

#include "triangle_layout.hpp"

namespace zest
{

/*
Recursion for expanding a Legendre function with affine transformed argument in terms of Legendre functions.

A Legendre polynomial `P_n(a + bx)` with an affinely transformed argument is a polynomial in `x`, and can therefore be expanded in terms of Legendre polynomials in `x`:
```txt
P_n(a + bx) = A_n0(a, b)P_0(x) + A_n1(a, b)P_1(x) + ... + A_nn(a, b)P_n(x)
```
This recursion generates the coefficients `A_nl(a, b)` for `0 <= l <= n <= lmax` for given parameters `a` and `b`.
*/
class AffineLegendreRecursion
{
public:
    AffineLegendreRecursion(): AffineLegendreRecursion(0) {}
    explicit AffineLegendreRecursion(std::size_t lmax);

    [[nodiscard]] std::size_t lmax() const noexcept { return m_lmax; }

    void expand(std::size_t lmax);

    void evaluate_affine(
        TriangleSpan<double, TriangleLayout> expansion, double shift, double scale);

    void evaluate_shifted(
        TriangleSpan<double, TriangleLayout> expansion, double shift);

    void evaluate_scaled(
        TriangleSpan<double, TriangleLayout> expansion, double scale);

private:
    std::vector<double> m_a;
    std::vector<double> m_b;
    std::vector<double> m_c;
    std::vector<double> m_d;
    std::size_t m_lmax;
};

}