#include "stl/anufriev_d_integrals_simpson/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {

int SimpsonCoeff(int i, int n) {
  if (i == 0 || i == n) {
    return 1;
  }
  if (i % 2 != 0) {
    return 4;
  }
  return 2;
}
}  // namespace

namespace anufriev_d_integrals_simpson_stl {

double IntegralsSimpsonSTL::FunctionN(const std::vector<double>& coords) const {
  switch (func_code_) {
    case 0: {
      double s = 0.0;
      for (double c : coords) {
        s += c * c;
      }
      return s;
    }
    case 1: {
      double val = 1.0;
      for (size_t i = 0; i < coords.size(); i++) {
        if (i % 2 == 0) {
          val *= std::sin(coords[i]);
        } else {
          val *= std::cos(coords[i]);
        }
      }
      return val;
    }
    default:
      return 0.0;
  }
}

double IntegralsSimpsonSTL::RecursiveSimpsonSum(int dim_index, std::vector<int>& idx,
                                                const std::vector<double>& steps) const {
  if (dim_index == dimension_) {
    double coeff = 1.0;
    std::vector<double> coords(dimension_);
    for (int d = 0; d < dimension_; ++d) {
      coords[d] = a_[d] + idx[d] * steps[d];
      coeff *= SimpsonCoeff(idx[d], n_[d]);
    }
    return coeff * FunctionN(coords);
  }
  double sum = 0.0;
  for (int i_rec = 0; i_rec <= n_[dim_index]; ++i_rec) {
    idx[dim_index] = i_rec;
    sum += RecursiveSimpsonSum(dim_index + 1, idx, steps);
  }
  return sum;
}

bool IntegralsSimpsonSTL::PreProcessingImpl() {
  if (task_data->inputs.empty()) {
    return false;
  }
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t in_size_bytes = task_data->inputs_count[0];
  size_t num_doubles = in_size_bytes / sizeof(double);
  if (num_doubles < 1) {
    return false;
  }
  int d = static_cast<int>(in_ptr[0]);
  if (d < 1) {
    return false;
  }
  size_t needed_count = static_cast<size_t>(3 * d) + 2;
  if (num_doubles < needed_count) {
    return false;
  }
  dimension_ = d;
  a_.resize(dimension_);
  b_.resize(dimension_);
  n_.resize(dimension_);
  int idx_ptr = 1;
  for (int i = 0; i < dimension_; i++) {
    a_[i] = in_ptr[idx_ptr++];
    b_[i] = in_ptr[idx_ptr++];
    n_[i] = static_cast<int>(in_ptr[idx_ptr++]);
    if (n_[i] <= 0 || (n_[i] % 2) != 0) {
      return false;
    }
  }
  func_code_ = static_cast<int>(in_ptr[idx_ptr]);
  result_ = 0.0;
  return true;
}

bool IntegralsSimpsonSTL::ValidationImpl() {
  if (task_data->outputs.empty()) {
    return false;
  }
  if (task_data->outputs_count.empty() || task_data->outputs_count[0] < 1) {
    return false;
  }
  return true;
}

void IntegralsSimpsonSTL::ThreadTaskRunner(int start_idx, int end_idx, const std::vector<double>& steps,
                                           double* partial_sum_output) {
  double local_partial_sum = 0.0;
  *partial_sum_output = 0.0;

  for (int i = start_idx; i < end_idx; ++i) {
    if (this->dimension_ < 1) {
      return;
    }
    std::vector<int> local_idx(dimension_);
    local_idx[0] = i;
    local_partial_sum += RecursiveSimpsonSum(1, local_idx, steps);
  }
  *partial_sum_output = local_partial_sum;
}

unsigned int IntegralsSimpsonSTL::DetermineNumThreads(int total_iterations) {
  unsigned int num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads == 0) {
    num_threads = 2;
  }
  if (total_iterations <= 0) {
    return 0;
  }
  num_threads = std::min(num_threads, static_cast<unsigned int>(total_iterations));

  if (num_threads == 0) {
    num_threads = 1;
  }
  return num_threads;
}

std::vector<IntegralsSimpsonSTL::IterationRange> IntegralsSimpsonSTL::DistributeIterations(int total_iterations,
                                                                                           unsigned int num_threads) {
  std::vector<IterationRange> ranges;
  if (num_threads == 0 || total_iterations == 0) {
    return ranges;
  }
  auto total_iter_unsigned = static_cast<unsigned int>(total_iterations);
  unsigned int iterations_per_thread_unsigned = total_iter_unsigned / num_threads;
  unsigned int remaining_iterations_unsigned = total_iter_unsigned % num_threads;

  int current_start_idx = 0;
  for (unsigned int i = 0; i < num_threads; ++i) {
    unsigned int current_chunk_size = iterations_per_thread_unsigned;
    if (remaining_iterations_unsigned > 0) {
      current_chunk_size++;
      remaining_iterations_unsigned--;
    }

    if (current_chunk_size == 0) {
      break;
    }

    int current_end_idx = current_start_idx + static_cast<int>(current_chunk_size);
    current_end_idx = std::min(current_end_idx, total_iterations);

    if (current_start_idx < current_end_idx) {
      ranges.push_back({current_start_idx, current_end_idx});
    }
    current_start_idx = current_end_idx;
    if (current_start_idx >= total_iterations) {
      break;
    }
  }
  return ranges;
}

bool IntegralsSimpsonSTL::RunImpl() {
  if (dimension_ < 1) {
    return false;
  }
  if (n_.empty() || n_[0] < 0) {
    return false;
  }

  std::vector<double> steps(dimension_);
  for (int i = 0; i < dimension_; i++) {
    if (n_[i] <= 0) {
      return false;
    }
    steps[i] = (b_[i] - a_[i]) / n_[i];
  }

  const int total_iterations_dim0 = n_[0] + 1;

  unsigned int num_actual_threads = DetermineNumThreads(total_iterations_dim0);

  if (num_actual_threads == 0 && total_iterations_dim0 > 0) {
    num_actual_threads = 1;
  }

  std::vector<double> partial_sums(num_actual_threads, 0.0);
  std::vector<std::thread> threads;

  if (num_actual_threads == 1) {
    if (!partial_sums.empty()) {
      ThreadTaskRunner(0, total_iterations_dim0, steps, partial_sums.data());
    }
  } else if (num_actual_threads != 0) {
    std::vector<IterationRange> ranges = DistributeIterations(total_iterations_dim0, num_actual_threads);

    if (ranges.size() < num_actual_threads && !ranges.empty()) {
      partial_sums.resize(ranges.size());
    } else if (ranges.empty() && num_actual_threads > 0) {
      partial_sums.clear();
    }

    for (size_t i = 0; i < ranges.size(); ++i) {
      const auto& range = ranges[i];
      threads.emplace_back(&IntegralsSimpsonSTL::ThreadTaskRunner, this, range.start, range.end, std::cref(steps),
                           &partial_sums[i]);
    }
  }

  for (auto& th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  double total_sum = 0.0;
  for (double p_sum : partial_sums) {
    total_sum += p_sum;
  }

  double coeff = 1.0;
  for (int i = 0; i < dimension_; i++) {
    coeff *= steps[i] / 3.0;
  }

  result_ = coeff * total_sum;
  return true;
}

bool IntegralsSimpsonSTL::PostProcessingImpl() {
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr || task_data->outputs_count.empty() ||
      task_data->outputs_count[0] < sizeof(double)) {
    return false;
  }
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  out_ptr[0] = result_;
  return true;
}

}  // namespace anufriev_d_integrals_simpson_stl