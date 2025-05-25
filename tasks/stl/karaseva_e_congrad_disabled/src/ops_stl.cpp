#include "stl/karaseva_e_congrad/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

using namespace karaseva_a_test_task_stl;

bool TestTaskSTL::PreProcessingImpl() {
  // Set system size from input data
  size_ = task_data->inputs_count[1];
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  // Initialize matrix A and vectors b, x
  A_ = std::vector<double>(a_ptr, a_ptr + (size_ * size_));
  b_ = std::vector<double>(b_ptr, b_ptr + size_);
  x_ = std::vector<double>(size_, 0.0);  // Initial guess

  return true;
}

bool TestTaskSTL::ValidationImpl() {
  // Validate matrix and vector dimensions
  const bool valid_input = task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1];
  const bool valid_output = task_data->outputs_count[0] == task_data->inputs_count[1];
  return valid_input && valid_output;
}

// Helper functions to reduce cognitive complexity
namespace {
void ParallelInit(std::vector<double>& r, std::vector<double>& p, const std::vector<double>& b, size_t size) {
  int thread_count = ppc::util::GetPPCNumThreads();
  const int max_hardware = static_cast<int>(std::thread::hardware_concurrency());
  if (thread_count <= 0 || thread_count > max_hardware) {
    thread_count = max_hardware > 0 ? max_hardware : 1;
  }
  const size_t num_threads = static_cast<size_t>(std::max(1, thread_count));

  std::vector<std::thread> threads;
  const size_t chunk_size = (size + num_threads - 1) / num_threads;

  for (size_t t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = std::min(start + chunk_size, size);
    threads.emplace_back([start, end, &r, &p, &b]() {
      for (size_t i = start; i < end; ++i) {
        r[i] = b[i];
        p[i] = r[i];
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

double ParallelDotProduct(const std::vector<double>& a, const std::vector<double>& b, size_t size) {
  int thread_count = ppc::util::GetPPCNumThreads();
  const int max_hardware = static_cast<int>(std::thread::hardware_concurrency());
  if (thread_count <= 0 || thread_count > max_hardware) {
    thread_count = max_hardware > 0 ? max_hardware : 1;
  }
  const size_t num_threads = static_cast<size_t>(std::max(1, thread_count));

  std::vector<std::thread> threads(num_threads);
  std::vector<double> partial_sums(num_threads, 0.0);
  const size_t chunk_size = (size + num_threads - 1) / num_threads;

  for (size_t t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = std::min(start + chunk_size, size);
    threads[t] = std::thread([start, end, &a, &b, &partial_sums, t]() {
      double sum = 0.0;
      for (size_t i = start; i < end; ++i) {
        sum += a[i] * b[i];
      }
      partial_sums[t] = sum;
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
}

void MatrixVectorMultiply(const std::vector<double>& a, std::vector<double>& ap, const std::vector<double>& p,
                          size_t size) {
  int thread_count = ppc::util::GetPPCNumThreads();
  const int max_hardware = static_cast<int>(std::thread::hardware_concurrency());
  if (thread_count <= 0 || thread_count > max_hardware) {
    thread_count = max_hardware > 0 ? max_hardware : 1;
  }
  const size_t num_threads = static_cast<size_t>(std::max(1, thread_count));

  std::vector<std::thread> threads;
  const size_t chunk_size = (size + num_threads - 1) / num_threads;

  for (size_t t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = std::min(start + chunk_size, size);
    threads.emplace_back([start, end, &a, &ap, &p, size]() {
      for (size_t i = start; i < end; ++i) {
        ap[i] = 0.0;
        for (size_t j = 0; j < size; ++j) {
          ap[i] += a[(i * size) + j] * p[j];
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

void ParallelVectorUpdate(std::vector<double>& x, std::vector<double>& r, const std::vector<double>& p,
                          const std::vector<double>& ap, double alpha, size_t size) {
  int thread_count = ppc::util::GetPPCNumThreads();
  const int max_hardware = static_cast<int>(std::thread::hardware_concurrency());
  if (thread_count <= 0 || thread_count > max_hardware) {
    thread_count = max_hardware > 0 ? max_hardware : 1;
  }
  const size_t num_threads = static_cast<size_t>(std::max(1, thread_count));

  std::vector<std::thread> threads;
  const size_t chunk_size = (size + num_threads - 1) / num_threads;

  for (size_t t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = std::min(start + chunk_size, size);
    threads.emplace_back([start, end, &x, &r, &p, &ap, alpha]() {
      for (size_t i = start; i < end; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

void UpdateSearchDirection(std::vector<double>& p, const std::vector<double>& r, double beta, size_t size) {
  int thread_count = ppc::util::GetPPCNumThreads();
  const int max_hardware = static_cast<int>(std::thread::hardware_concurrency());
  if (thread_count <= 0 || thread_count > max_hardware) {
    thread_count = max_hardware > 0 ? max_hardware : 1;
  }
  const size_t num_threads = static_cast<size_t>(std::max(1, thread_count));

  std::vector<std::thread> threads;
  const size_t chunk_size = (size + num_threads - 1) / num_threads;

  for (size_t t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk_size;
    const size_t end = std::min(start + chunk_size, size);
    threads.emplace_back([start, end, &p, &r, beta]() {
      for (size_t i = start; i < end; ++i) {
        p[i] = r[i] + beta * p[i];
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}
}  // namespace

bool TestTaskSTL::RunImpl() {
  std::vector<double> r(size_);
  std::vector<double> p(size_);
  std::vector<double> ap(size_);

  // 1. Initialize residual and search direction vectors
  ParallelInit(r, p, b_, size_);

  // 2. Compute initial residual norm
  double rs_old = ParallelDotProduct(r, r, size_);

  const double tolerance = 1e-10;
  const size_t max_iterations = size_;

  // 3. Main CG loop
  for (size_t k = 0; k < max_iterations; ++k) {
    // 3.1 Matrix-vector multiplication
    MatrixVectorMultiply(A_, ap, p, size_);

    // 3.2 Compute dot product
    const double p_ap = ParallelDotProduct(p, ap, size_);
    if (std::fabs(p_ap) < 1e-15) {
      break;
    }

    // 3.3 Compute alpha and update vectors
    const double alpha = rs_old / p_ap;
    ParallelVectorUpdate(x_, r, p, ap, alpha, size_);

    // 3.4 Check convergence
    const double rs_new = ParallelDotProduct(r, r, size_);
    if (rs_new < tolerance * tolerance) {
      break;
    }

    // 3.5 Update search direction
    const double beta = rs_new / rs_old;
    UpdateSearchDirection(p, r, beta, size_);
    rs_old = rs_new;
  }

  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < x_.size(); ++i) {
    x_ptr[i] = x_[i];
  }
  return true;
}