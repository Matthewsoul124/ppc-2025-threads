#include "all/kholin_k_multidimensional_integrals_rectangle/include/ops_all.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"
#endif

#include <mpi.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#include <omp.h>

#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <vector>

double kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::Integrate(
    const Function& f, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
    const std::vector<double>& h, std::vector<double>& f_values, int curr_index_dim, int dim, double n) {
  if (curr_index_dim == dim) {
    return f(f_values);
  }

  double sum = 0.0;
  const double l_limit = l_limits[curr_index_dim];
  const double step = h[curr_index_dim];

  for (int i = 0; i < static_cast<int>(n); ++i) {
    f_values[curr_index_dim] = l_limit + (static_cast<double>(i) + 0.5) * step;
    sum += Integrate(f, l_limits, u_limits, h, f_values, curr_index_dim + 1, dim, n);
  }
  return sum * h[curr_index_dim];
}

double kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::IntegrateWithRectangleMethod(
    const Function& f, std::vector<double>& f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, int dim, double n) {
  std::vector<double> h(dim);
#pragma omp parallel for
  for (int i = 0; i < dim; ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }
  return Integrate(f, l_limits, u_limits, h, f_values, 0, dim, n);
}

double kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::RunMultistepSchemeMethodRectangle(
    const Function& f, std::vector<double>& f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, int dim, double n) {
  int rank = 0;
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank >= 0) {
    local_l_limits_ = std::vector<double>(dim);
    local_u_limits_ = std::vector<double>(dim);
  }
  for (int i = 0; i < dim_; ++i) {
    double range = u_limits[i] - l_limits[i];
    local_l_limits_[i] = l_limits[i] + (rank * (range / size));
    local_u_limits_[i] = l_limits[i] + ((rank + 1) * (range / size));
  }
  double local_result = IntegrateWithRectangleMethod(f, f_values, local_l_limits_, local_u_limits_, dim, n);
  if (dim_ > 1) {
    local_result = local_result * std::pow(size, dim - 1);
  }
  MPI_Reduce(&local_result, &I_2n_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  return I_2n_;
}

bool kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    // Init value for input and output
    sz_values_ = static_cast<int>(task_data->inputs_count[0]);
    sz_lower_limits_ = static_cast<int>(task_data->inputs_count[1]);
    sz_upper_limits_ = static_cast<int>(task_data->inputs_count[2]);

    auto* ptr_dim = reinterpret_cast<int*>(task_data->inputs[0]);
    dim_ = *ptr_dim;

    auto* ptr_f_values = reinterpret_cast<double*>(task_data->inputs[1]);
    f_values_.assign(ptr_f_values, ptr_f_values + sz_values_);

    auto* ptr_lower_limits = reinterpret_cast<double*>(task_data->inputs[2]);
    lower_limits_.assign(ptr_lower_limits, ptr_lower_limits + sz_lower_limits_);

    auto* ptr_upper_limits = reinterpret_cast<double*>(task_data->inputs[3]);
    upper_limits_.assign(ptr_upper_limits, ptr_upper_limits + sz_upper_limits_);

    auto* ptr_start_n = reinterpret_cast<double*>(task_data->inputs[4]);
    start_n_ = *ptr_start_n;
  }
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return task_data->inputs_count[1] > 0U && task_data->inputs_count[2] > 0U;
  }
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&sz_values_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sz_lower_limits_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sz_upper_limits_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dim_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&start_n_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank > 0) {
    f_values_ = std::vector<double>(sz_values_);
    lower_limits_ = std::vector<double>(sz_lower_limits_);
    upper_limits_ = std::vector<double>(sz_upper_limits_);
  }
  MPI_Bcast(lower_limits_.data(), sz_lower_limits_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(upper_limits_.data(), sz_upper_limits_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(f_values_.data(), sz_values_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  RunMultistepSchemeMethodRectangle(f_, f_values_, lower_limits_, upper_limits_, dim_, start_n_);
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = I_2n_;
  }
  return true;
}
