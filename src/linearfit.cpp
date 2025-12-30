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
#include "linearfit.hpp"

#include <cassert>

namespace zest
{

extern "C" 
{
int dgels_(
    char* trans, long int* m, long int* n, long int* nrhs, double* a,
    long int* lda, double* b, long int* ldb, double* work, long int* lwork, 
    long int* info);
}

namespace detail
{

void LinearMultifit::dgels_wrapper(std::array<std::size_t, 2> extents)
{
    // Have to interpret as transpose of Fortran order
    auto nrows_f = (long int)(extents[1]);
    auto ncols_f = (long int)(extents[0]);
    char trans = 'T';

    long int lda = nrows_f;
    long int ldb = std::max(nrows_f, ncols_f);

    long int lwork = -1;
    long int info = 0;
    long int nrhs = 1;

    // Must be double because LAPACK is stupidly designed
    double work_size = -1.0; 

    // All this to get the optimal work_size because LAPACK is stupidly designed
    dgels_(&trans, &nrows_f, &ncols_f, &nrhs, m_model_data.data(), &lda, m_data.data(), &ldb, &work_size, &lwork, &info);

    work.resize(std::size_t(work_size));

    lwork = (long int)(work_size);
    dgels_(&trans, &nrows_f, &ncols_f, &nrhs, m_model_data.data(), &lda, m_data.data(), &ldb, work.data(), &lwork, &info);

    // Did I ever tell you how much I hate the interfaces of these old Fortran libraries?
}

} // namespace detail

} // namespace zest
