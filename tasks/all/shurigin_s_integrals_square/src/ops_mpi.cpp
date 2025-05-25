#include "all/shurigin_s_integrals_square/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"
#endif
#include <mpi.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <omp.h>

namespace shurigin_s_integrals_square_mpi {

Integral::Integral(const std::shared_ptr<ppc::core::TaskData>& task_data_param)
    : Task(task_data_param), result_(0.0), func_(nullptr), dimensions_(0), mpi_rank_(0), mpi_world_size_(1) {}

void Integral::SetFunction(const std::function<double(double)>& func) {
  if (!func) {
    throw std::invalid_argument("SetFunction (1D): Function is null.");
  }
  func_ = [func](const std::vector<double>& point) {
    if (point.empty()) {
      throw std::runtime_error("1D Wrapper: Point vector is empty.");
    }
    return func(point[0]);
  };
  dimensions_ = 1;
  down_limits_.assign(1, 0.0);
  up_limits_.assign(1, 1.0);
  counts_.assign(1, 100);
}

void Integral::SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions) {
  if (!func) {
    throw std::invalid_argument("SetFunction (ND): Function is null.");
  }
  if (dimensions <= 0) {
    throw std::invalid_argument("SetFunction (ND): Dimensions must be positive.");
  }
  func_ = func;
  dimensions_ = dimensions;
  down_limits_.assign(dimensions_, 0.0);
  up_limits_.assign(dimensions_, 1.0);
  counts_.assign(dimensions_, 100);
}

bool Integral::PreProcessingImpl() {
  try {
    if (!this->task_data) {
      std::cerr << "Rank " << mpi_rank_ << " PreProc Error: TaskData is null.\n";
      return false;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size_);

    if (mpi_rank_ == 0) {
      if (dimensions_ <= 0) {
        std::cerr << "Rank 0 PreProc Error: Dimensions not set or invalid.\n";
        return false;
      }
      if (this->task_data->inputs.empty() || this->task_data->inputs_count.empty() ||
          this->task_data->inputs[0] == nullptr) {
        std::cerr << "Rank 0 PreProc Error: Invalid inputs or inputs_count.\n";
        return false;
      }

      auto expected_elements = static_cast<size_t>(dimensions_) * 3;
      auto actual_elements = this->task_data->inputs_count[0] / sizeof(double);

      if (actual_elements != expected_elements) {
        std::cerr << "Rank 0 PreProc Error: Input size mismatch. Expected " << expected_elements << " doubles, got "
                  << actual_elements << " doubles.\n";
        return false;
      }

      down_limits_.resize(dimensions_);
      up_limits_.resize(dimensions_);
      counts_.resize(dimensions_);

      auto* inputs_ptr = reinterpret_cast<double*>(this->task_data->inputs[0]);
      for (int i = 0; i < dimensions_; ++i) {
        down_limits_[i] = inputs_ptr[i];
        up_limits_[i] = inputs_ptr[i + dimensions_];
        counts_[i] = static_cast<int>(inputs_ptr[i + (2 * dimensions_)]);

        if (counts_[i] <= 0) {
          std::cerr << "Rank 0 PreProc Error: Counts must be positive for dim " << i << "\n";
          return false;
        }
        if (up_limits_[i] <= down_limits_[i]) {
          std::cerr << "Rank 0 PreProc Error: Upper limit <= lower for dim " << i << "\n";
          return false;
        }
      }
    }

    MPI_Bcast(&dimensions_, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (dimensions_ <= 0 && mpi_rank_ != 0) {
      std::cerr << "Rank " << mpi_rank_ << " PreProc Error: Invalid dimensions after Bcast.\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return false;
    }

    if (mpi_rank_ != 0) {
      down_limits_.resize(dimensions_);
      up_limits_.resize(dimensions_);
      counts_.resize(dimensions_);
    }

    MPI_Bcast(down_limits_.data(), dimensions_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(up_limits_.data(), dimensions_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(counts_.data(), dimensions_, MPI_INT, 0, MPI_COMM_WORLD);

    result_ = 0.0;
    return true;

  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " PreProc Exception: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool Integral::ValidationImpl() {
  try {
    if (!this->task_data) {
      if (mpi_rank_ == 0) {
        std::cerr << "Rank 0 Validation Error: TaskData is null.\n";
      }
      return false;
    }

    if (dimensions_ <= 0) {
      std::cerr << "Rank " << mpi_rank_ << " Validation Error: dimensions_ not positive.\n";
      return false;
    }

    int root_validation_ok = 1;
    if (mpi_rank_ == 0) {
      if (this->task_data->inputs_count.empty() || this->task_data->outputs_count.empty()) {
        std::cerr << "Rank 0 Validation Error: inputs_count or outputs_count empty.\n";
        root_validation_ok = 0;
      } else {
        auto expected_elements_input = static_cast<size_t>(dimensions_) * 3;
        if (this->task_data->inputs_count[0] != expected_elements_input * sizeof(double)) {
          std::cerr << "Rank 0 Validation Error: Input size mismatch.\n";
          root_validation_ok = 0;
        }
        if (this->task_data->outputs_count[0] != sizeof(double)) {
          std::cerr << "Rank 0 Validation Error: Output size mismatch.\n";
          root_validation_ok = 0;
        }
      }
    }

    MPI_Bcast(&root_validation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (root_validation_ok == 0) {
      if (mpi_rank_ != 0) {
        std::cerr << "Rank " << mpi_rank_ << " Validation Error: Root validation failed.\n";
      }
      return false;
    }

    int local_validation_ok_flag = 1;
    if (!func_) {
      std::cerr << "Rank " << mpi_rank_ << " Validation Error: Function not set.\n";
      local_validation_ok_flag = 0;
    }
    if (static_cast<int>(down_limits_.size()) != dimensions_ || static_cast<int>(up_limits_.size()) != dimensions_ ||
        static_cast<int>(counts_.size()) != dimensions_) {
      std::cerr << "Rank " << mpi_rank_ << " Validation Error: Vector sizes mismatch.\n";
      local_validation_ok_flag = 0;
    }

    int final_status = 0;
    MPI_Allreduce(&local_validation_ok_flag, &final_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    return (final_status == 1);

  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " Validation Exception: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool Integral::RunImpl() {
  try {
    if (!func_ || dimensions_ <= 0 || counts_.empty() || down_limits_.empty() || up_limits_.empty() ||
        static_cast<int>(counts_.size()) != dimensions_ || static_cast<int>(down_limits_.size()) != dimensions_ ||
        static_cast<int>(up_limits_.size()) != dimensions_) {
      std::cerr << "Rank " << mpi_rank_ << " RunImpl Error: Preconditions not met (func, dims, vectors).\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return false;
    }

    double local_integral_sum = 0.0;
    int n0_total = counts_[0];

    if (n0_total > 0) {
      double a0_global = down_limits_[0];
      double b0_global = up_limits_[0];

      int chunk_size = n0_total / mpi_world_size_;
      int remainder = n0_total % mpi_world_size_;

      int n0_local_start_index = (mpi_rank_ * chunk_size) + std::min(mpi_rank_, remainder);
      int n0_local_count = chunk_size + (mpi_rank_ < remainder ? 1 : 0);

      if (n0_local_count > 0) {
        double h0_global_step = (b0_global - a0_global) / n0_total;
        double a0_local = a0_global + (static_cast<double>(n0_local_start_index) * h0_global_step);
        double b0_local = a0_global + (static_cast<double>(n0_local_start_index + n0_local_count) * h0_global_step);

        if (mpi_rank_ == mpi_world_size_ - 1) {
          b0_local = b0_global;
        }
        a0_local = std::max(a0_global, std::min(b0_global, a0_local));
        b0_local = std::min(b0_global, std::max(a0_global, b0_local));

        if (a0_local >= b0_local) {
          n0_local_count = 0;
        }

        if (n0_local_count > 0) {
          if (dimensions_ == 1) {
            local_integral_sum = ComputeOneDimensionalOMP(func_, a0_local, b0_local, n0_local_count);
          } else {
            local_integral_sum = ComputeOuterParallelInnerSequential(func_, a0_local, b0_local, n0_local_count,
                                                                     down_limits_, up_limits_, counts_, dimensions_);
          }
        }
      }
    }
    MPI_Reduce(&local_integral_sum, &result_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " RunImpl Exception: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

double Integral::ComputeOneDimensionalOMP(const std::function<double(const std::vector<double>&)>& f, double a_local,
                                          double b_local, int n_local) {
  if (n_local <= 0 || a_local >= b_local) {
    return 0.0;
  }
  const double step = (b_local - a_local) / n_local;
  double total_sum_omp = 0.0;

#pragma omp parallel
  {
    std::vector<double> point(1);
#pragma omp for schedule(static) reduction(+ : total_sum_omp)
    for (int i = 0; i < n_local; ++i) {
      point[0] = a_local + ((static_cast<double>(i) + 0.5) * step);
      total_sum_omp += f(point);
    }
  }
  return total_sum_omp * step;
}

double Integral::ComputeOuterParallelInnerSequential(const std::function<double(const std::vector<double>&)>& f,
                                                     double a0_local_mpi, double b0_local_mpi, int n0_local_mpi,
                                                     const std::vector<double>& full_a,
                                                     const std::vector<double>& full_b, const std::vector<int>& full_n,
                                                     int total_dims) {
  if (n0_local_mpi <= 0 || a0_local_mpi >= b0_local_mpi) {
    return 0.0;
  }
  const double h0_local_step = (b0_local_mpi - a0_local_mpi) / n0_local_mpi;
  double outer_integral_sum_omp = 0.0;

#pragma omp parallel
  {
    std::vector<double> current_point(static_cast<size_t>(total_dims));
#pragma omp for schedule(static) reduction(+ : outer_integral_sum_omp)
    for (int i = 0; i < n0_local_mpi; ++i) {
      current_point[0] = a0_local_mpi + ((static_cast<double>(i) + 0.5) * h0_local_step);
      outer_integral_sum_omp += ComputeSequentialRecursive(f, full_a, full_b, full_n, total_dims, current_point, 1);
    }
  }
  return outer_integral_sum_omp * h0_local_step;
}

double Integral::ComputeSequentialRecursive(const std::function<double(const std::vector<double>&)>& f,
                                            const std::vector<double>& a_all_dims,
                                            const std::vector<double>& b_all_dims, const std::vector<int>& n_all_dims,
                                            int total_dims, std::vector<double>& current_eval_point,
                                            int current_dim_index) {
  if (current_dim_index == total_dims) {
    return f(current_eval_point);
  }
  if (current_dim_index < 0 || static_cast<size_t>(current_dim_index) >= n_all_dims.size() ||
      static_cast<size_t>(current_dim_index) >= a_all_dims.size() ||
      static_cast<size_t>(current_dim_index) >= b_all_dims.size() ||
      static_cast<size_t>(current_dim_index) >= current_eval_point.size()) {
    throw std::out_of_range("Recursive Dim index out of bounds: " + std::to_string(current_dim_index));
  }
  const int n_for_current_dim = n_all_dims[static_cast<size_t>(current_dim_index)];
  const double a_for_current_dim = a_all_dims[static_cast<size_t>(current_dim_index)];
  const double b_for_current_dim = b_all_dims[static_cast<size_t>(current_dim_index)];
  if (n_for_current_dim <= 0) {
    std::cerr << "Recursive Error: N <= 0 for dim " << current_dim_index << "\n";
    return 0.0;
  }
  if (a_for_current_dim >= b_for_current_dim) {
    return 0.0;
  }
  const double h_step_for_current_dim = (b_for_current_dim - a_for_current_dim) / n_for_current_dim;
  double sum_for_this_dimension = 0.0;
  for (int i = 0; i < n_for_current_dim; ++i) {
    current_eval_point[static_cast<size_t>(current_dim_index)] =
        a_for_current_dim + ((static_cast<double>(i) + 0.5) * h_step_for_current_dim);
    sum_for_this_dimension += ComputeSequentialRecursive(f, a_all_dims, b_all_dims, n_all_dims, total_dims,
                                                         current_eval_point, current_dim_index + 1);
  }
  return sum_for_this_dimension * h_step_for_current_dim;
}

bool Integral::PostProcessingImpl() {
  try {
    if (mpi_rank_ == 0) {
      if (!this->task_data) {
        std::cerr << "Rank 0 PostProc Error: TaskData is null.\n";
        return false;
      }
      if (this->task_data->outputs.empty() || this->task_data->outputs[0] == nullptr ||
          this->task_data->outputs_count.empty() || this->task_data->outputs_count[0] != sizeof(double)) {
        std::cerr << "Rank 0 PostProc Error: Invalid outputs or outputs_count.\n";
        return false;
      }
      auto* outputs_ptr = reinterpret_cast<double*>(this->task_data->outputs[0]);
      outputs_ptr[0] = result_;
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank_ << " PostProc Exception: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

}  // namespace shurigin_s_integrals_square_mpi