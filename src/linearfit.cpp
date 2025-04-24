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
#include "linearfit.hpp"

#include <cassert>
#include <algorithm>

#include "md_span.hpp"

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

[[nodiscard]] std::vector<double> LinearMultifit::operator()(
    MDSpan<const double, 2> model, std::span<const double> data)
{
    std::vector<double> parameters = std::vector<double>(model.extent(1));
    operator()(model, parameters, data);
    return parameters;
}

void LinearMultifit::operator()(
    MDSpan<const double, 2> model, std::span<double> parameters, std::span<const double> data)
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
    
    // Have to interpret as transpose of Fortran order
    long int nrows_f = (long int)(model.extent(1));
    long int ncols_f = (long int)(model.extent(0));
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

    std::copy_n(m_data.begin(), parameters_view.size(), parameters_view.begin());

    // Everything above hurts me physically but at least we're out
}

} // namespace detail

} // namespace zest
