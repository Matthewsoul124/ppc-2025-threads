#include "stl/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <future>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool zolotareva_a_sle_gradient_method_stl::TestTaskSTL::PreProcessingImpl() {
  n_ = static_cast<int>(task_data->inputs_count[1]);
  a_.resize(n_ * n_);
  b_.resize(n_);
  x_.resize(n_, 0.0);
  const auto* input_matrix = reinterpret_cast<const double*>(task_data->inputs[0]);
  const auto* input_vector = reinterpret_cast<const double*>(task_data->inputs[1]);

  for (int i = 0; i < n_; ++i) {
    b_[i] = input_vector[i];
    for (int j = 0; j < n_; ++j) {
      a_[(i * n_) + j] = input_matrix[(i * n_) + j];
    }
  }

  return true;
}

bool zolotareva_a_sle_gradient_method_stl::TestTaskSTL::ValidationImpl() {
  if (static_cast<int>(task_data->inputs_count[0]) < 0 || static_cast<int>(task_data->inputs_count[1]) < 0 ||
      static_cast<int>(task_data->outputs_count[0]) < 0) {
    return false;
  }
  if (task_data->inputs_count.size() < 2 || task_data->inputs.size() < 2 || task_data->outputs.empty()) {
    return false;
  }

  if (static_cast<int>(task_data->inputs_count[0]) !=
      (static_cast<int>(task_data->inputs_count[1]) * static_cast<int>(task_data->inputs_count[1]))) {
    return false;
  }
  if (task_data->outputs_count[0] != task_data->inputs_count[1]) {
    return false;
  }

  // проверка симметрии и положительной определённости
  const auto* a = reinterpret_cast<const double*>(task_data->inputs[0]);

  return IsPositiveAndSimm(a, static_cast<int>(task_data->inputs_count[1]));
}

bool zolotareva_a_sle_gradient_method_stl::TestTaskSTL::RunImpl() {
  ConjugateGradient(a_, b_, x_, n_);
  return true;
}

bool zolotareva_a_sle_gradient_method_stl::TestTaskSTL::PostProcessingImpl() {
  auto* output_raw = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(x_.begin(), x_.end(), output_raw);
  return true;
}

void zolotareva_a_sle_gradient_method_stl::TestTaskSTL::ConjugateGradient(const std::vector<double>& a,
                                                                          const std::vector<double>& b,
                                                                          std::vector<double>& x, int n) {
  std::vector<double> r = b;
  std::vector<double> p = r;
  double rs_old = DotProduct(r, r, n);

  double initial_norm = std::sqrt(DotProduct(b, b, n));
  double threshold = initial_norm == 0.0 ? 1e-12 : initial_norm * 1e-12;

  for (int iter = 0; iter <= n; ++iter) {
    // ap = A*p
    std::vector<double> ap(n, 0.0);
    MatrixVectorMult(a, p, ap, n);

    double p_ap = DotProduct(p, ap, n);
    if (p_ap == 0.0) {
      break;
    }
    double alpha = rs_old / p_ap;

    // x += alpha*p ; r -= alpha*ap (параллельно)
    ParallelFor(0, n, [&](int i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    });

    double rs_new = DotProduct(r, r, n);
    if (rs_new < threshold) {
      break;
    }

    double beta = rs_new / rs_old;
    // p = r + beta*p
    ParallelFor(0, n, [&](int i) { p[i] = r[i] + beta * p[i]; });

    rs_old = rs_new;
  }
}

double zolotareva_a_sle_gradient_method_stl::TestTaskSTL::DotProduct(const std::vector<double>& vec1,
                                                                     const std::vector<double>& vec2, int n) {
  int num_threads = ppc::util::GetPPCNumThreads();
  int block = (n + num_threads - 1) / num_threads;
  std::vector<std::future<double>> futures;
  for (int t = 0; t < num_threads; ++t) {
    int s = t * block;
    int e = std::min(n, s + block);
    if (s >= e) {
      break;
    }
    futures.emplace_back(std::async(std::launch::async, [=, &vec1, &vec2] {
      double sum = 0;
      for (int i = s; i < e; ++i) {
        sum += vec1[i] * vec2[i];
      }
      return sum;
    }));
  }
  double total = 0;
  for (auto& f : futures) {
    total += f.get();
  }
  return total;
}

void zolotareva_a_sle_gradient_method_stl::TestTaskSTL::MatrixVectorMult(const std::vector<double>& matrix,
                                                                         const std::vector<double>& vector,
                                                                         std::vector<double>& result, int n) {
  ParallelFor(0, n, [&](int i) {
    double sum = 0;
    for (int j = 0; j < n; ++j) {
      sum += matrix[(i * n) + j] * vector[j];
    }
    result[i] = sum;
  });
}

bool zolotareva_a_sle_gradient_method_stl::TestTaskSTL::IsPositiveAndSimm(const double* a, int n) {
  std::vector<double> m;
  m.assign(a, a + (n * n));
  // симметрия
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (m[(i * n) + j] != m[(j * n) + i]) {
        return false;
      }
    }
  }
  // проверка положительной определённости (холу разложение)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = m[(i * n) + j];
      for (int k = 0; k < j; ++k) {
        sum -= m[(i * n) + k] * m[(j * n) + k];
      }
      if (i == j) {
        if (sum <= 1e-15) {
          return false;
        }
        m[(i * n) + j] = std::sqrt(sum);
      } else {
        m[(i * n) + j] = sum / m[(j * n) + j];
      }
    }
  }
  return true;
}
void zolotareva_a_sle_gradient_method_stl::TestTaskSTL::ParallelFor(int start, int end,
                                                                    const std::function<void(int)>& f) {
  int num_threads = ppc::util::GetPPCNumThreads();
  int total = end - start;
  int block = (total + num_threads - 1) / num_threads;
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    int s = start + (t * block);
    int e = std::min(end, s + block);
    if (s >= e) {
      break;
    }
    threads.emplace_back([s, e, &f] {
      for (int i = s; i < e; ++i) {
        f(i);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}