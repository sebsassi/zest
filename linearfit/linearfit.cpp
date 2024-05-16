#include <cassert>

#include "linearfit.hpp"

extern "C" 
{
int dgels_(
    char* trans, long int* m, long int* n, long int* nrhs, double* a,
    long int* lda, double* b, long int* ldb, double* work, long int* lwork, 
    long int* info);
}

[[nodiscard]] std::vector<double> LinearMultifit::fit_parameters(
    MatrixSpan<const double> model, std::span<const double> data)
{
    std::vector<double> parameters = std::vector<double>(model.ncols());
    fit_parameters(model, parameters, data);
    return parameters;
}

void LinearMultifit::fit_parameters(
    MatrixSpan<const double> model, std::span<double> parameters, std::span<const double> data)
{
    m_model_data.resize(model.nrows()*model.ncols());
    m_data.resize(std::max(model.nrows(), model.ncols()));

    std::span<double> parameters_view(parameters.begin(), model.ncols());
    std::span<const double> data_view(data.begin(), model.nrows());

    // Copy because dgels_ will modify data
    std::span<const double> mat_data = model.data(); 
    std::copy(mat_data.begin(), mat_data.end(), m_model_data.begin());
    std::copy_n(data_view.begin(), data_view.size(), m_data.begin());
    
    // Have to interpret as transpose of Fortran order
    long int nrows_f = (long int)(model.ncols());
    long int ncols_f = (long int)(model.nrows());
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