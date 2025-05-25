#include "all/poroshin_v_multi_integral_with_trapez_method/include/ops_all.hpp"

#include <omp.h>

#include <boost/serialization/utility.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/reduce.hpp"

void poroshin_v_multi_integral_with_trapez_method_all::TestTaskALL::CountMultiIntegralTrapezMethodAll(double &res) {
  const int dimensions = static_cast<int>(limits_.size());
  std::vector<double> h(dimensions);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < dimensions; ++i) {
    h[i] = (limits_[i].second - limits_[i].first) / n_[i];
  }

  std::vector<std::vector<double>> weights(dimensions);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < dimensions; ++i) {
    weights[i].resize(n_[i] + 1);
    for (int j = 0; j <= n_[i]; ++j) {
      weights[i][j] = (j == 0 || j == n_[i]) ? 0.5 : 1.0;
    }
  }

  double integral = 0.0;

#pragma omp parallel reduction(+ : integral)
  {
    std::vector<double> vars(dimensions);
    std::vector<int> indices(dimensions, 0);

    int total_points = 1;
    for (int n : n_) {
      total_points *= (n + 1);
    }

#pragma omp for schedule(static)
    for (int linear_idx = 0; linear_idx < total_points; ++linear_idx) {
      int idx = linear_idx;
      for (int dim = dimensions - 1; dim >= 0; --dim) {
        indices[dim] = idx % (n_[dim] + 1);
        idx /= (n_[dim] + 1);
      }

      double weight = 1.0;
      for (int dim = 0; dim < dimensions; ++dim) {
        vars[dim] = limits_[dim].first + indices[dim] * h[dim];
        weight *= weights[dim][indices[dim]];
      }

      integral += func_(vars) * weight;
    }
  }

  double volume = 1.0;
#pragma omp parallel for reduction(* : volume)
  for (int i = 0; i < dimensions; ++i) {
    volume *= h[i];
  }

  res = integral * volume;
}

bool poroshin_v_multi_integral_with_trapez_method_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    dim_ = task_data->inputs_count[0];
    n_.resize(dim_);
    limits_.resize(dim_);
    for (size_t i = 0; i < dim_; i++) {
      n_[i] = reinterpret_cast<int *>(task_data->inputs[0])[i];
      limits_[i].first = reinterpret_cast<double *>(task_data->inputs[1])[i];
      limits_[i].second = reinterpret_cast<double *>(task_data->inputs[2])[i];
    }
    res_ = 0;
  }
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1);
  }
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_all::TestTaskALL::RunImpl() {
  size_t len = 0;
  double step = 0;
  int delta = 0;
  int last = 0;
  double res = 0;
  if (world_.rank() == 0) {
    if ((n_[0] % world_.size()) == 0) {
      delta = n_[0] / world_.size();
      last = delta;
    } else {
      delta = n_[0] / world_.size();
      last = delta + n_[0] % world_.size();
    }
    step = (limits_[0].second - limits_[0].first) / static_cast<double>(n_[0]);
    len = dim_;
  }
  boost::mpi::broadcast(world_, step, 0);
  boost::mpi::broadcast(world_, delta, 0);
  boost::mpi::broadcast(world_, last, 0);
  boost::mpi::broadcast(world_, len, 0);
  if (world_.rank() != 0) {
    limits_.resize(len);
    n_.resize(len);
  }
  boost::mpi::broadcast(world_, limits_.data(), static_cast<int>(len), 0);
  boost::mpi::broadcast(world_, n_.data(), static_cast<int>(len), 0);
  if (world_.rank() == world_.size() - 1) {
    n_[0] = last;
    limits_[0].first += world_.rank() * step * delta;
  } else {
    n_[0] = delta;
    limits_[0].first += world_.rank() * step * delta;
    limits_[0].second = limits_[0].first + step * delta;
  }
  CountMultiIntegralTrapezMethodAll(res);
  boost::mpi::reduce(world_, res, res_, std::plus<>(), 0);
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  }
  return true;
}