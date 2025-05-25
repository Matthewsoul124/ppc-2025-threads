#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <future>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace kazunin_n_montecarlo_stl {

template <std::size_t N, typename F>
class MonteCarloStl : public ppc::core::Task {
 public:
  explicit MonteCarloStl(ppc::core::TaskDataPtr task_data, F f) : Task(std::move(task_data)), f_(f) {}

  bool ValidationImpl() override {
    return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == N && task_data->outputs_count[0] == 1 &&
           (*reinterpret_cast<std::size_t*>(task_data->inputs[0]) > 0);
  }

  bool PreProcessingImpl() override {
    precision_ = *reinterpret_cast<std::size_t*>(task_data->inputs[0]);
    limits_ = *reinterpret_cast<decltype(limits_)*>(task_data->inputs[1]);
    std::ranges::generate(generators_, [&, i = 0]() mutable { return MakeGenerator(limits_[i++]); });
    total_space_ = std::accumulate(
        limits_.begin(), limits_.end(), 1.0,
        [](const double acc, const std::pair<double, double>& limit) { return acc * (limit.second - limit.first); });
    return true;
  }
  bool RunImpl() override {
    const auto numthreads = static_cast<std::size_t>(ppc::util::GetPPCNumThreads());
    std::vector<std::future<double>> futures(numthreads);
    std::vector<std::thread> threads(numthreads);

    const auto delta = precision_ / numthreads;
    const auto offset = precision_ % numthreads;

    for (std::size_t i = 0; i < numthreads; i++) {
      std::promise<double> promise;
      futures[i] = promise.get_future();
      threads[i] = std::thread(
          [&](std::size_t iters, std::promise<double>&& promise) {
            double sum = 0.0;
            std::array<double, N> random_args;
            const auto generators = generators_;
            for (std::size_t iter = 0; iter < iters; ++iter) {
              std::ranges::generate(random_args, [&generators, j = 0]() mutable { return generators[j++](); });
              sum += f_(random_args);
            }
            promise.set_value(sum);
          },
          delta + static_cast<std::size_t>(i < offset), std::move(promise));
    }

    double total_sum = 0.0;
    for (std::size_t i = 0; i < numthreads; i++) {
      threads[i].join();
      total_sum += futures[i].get();
    }

    result_ = (total_space_ * total_sum) / precision_;

    return true;
  }
  bool PostProcessingImpl() override {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
    return true;
  }

 private:
  std::function<double()> MakeGenerator(const std::pair<double, double>& interval) {
    std::uniform_real_distribution<double> distrib(interval.first, interval.second);
    std::mt19937 engine(seedgen_());
    return [distrib, engine]() mutable { return distrib(engine); };
  }

  F f_;
  std::size_t precision_;
  double total_space_;
  std::array<std::pair<double, double>, N> limits_;
  std::array<std::function<double()>, N> generators_;
  std::random_device seedgen_;
  double result_;
};

}  // namespace kazunin_n_montecarlo_stl

namespace kazunin_n_montecarlo_seq {

template <std::size_t N, typename F>
class MonteCarloSeq : public ppc::core::Task {
 public:
  explicit MonteCarloSeq(ppc::core::TaskDataPtr task_data, F f) : Task(std::move(task_data)), f_(f) {}

  bool ValidationImpl() override {
    return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] == N && task_data->outputs_count[0] == 1 &&
           (*reinterpret_cast<std::size_t*>(task_data->inputs[0]) > 0);
  }

  bool PreProcessingImpl() override {
    precision_ = *reinterpret_cast<std::size_t*>(task_data->inputs[0]);
    limits_ = *reinterpret_cast<decltype(limits_)*>(task_data->inputs[1]);
    std::ranges::generate(generators_, [&, i = 0]() mutable { return MakeGenerator(limits_[i++]); });
    total_space_ = std::accumulate(
        limits_.begin(), limits_.end(), 1.0,
        [](const double acc, const std::pair<double, double>& limit) { return acc * (limit.second - limit.first); });
    return true;
  }
  bool RunImpl() override {
    double sum = 0.0;

    std::array<double, N> random_args;
    for (std::size_t iter = 0; iter < precision_; ++iter) {
      std::ranges::generate(random_args, [this, j = 0]() mutable { return generators_[j++](); });
      sum += f_(random_args);
    }

    result_ = (total_space_ * sum) / precision_;
    return true;
  }
  bool PostProcessingImpl() override {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
    return true;
  }

 private:
  std::function<double()> MakeGenerator(const std::pair<double, double>& interval) {
    std::uniform_real_distribution<double> distrib(interval.first, interval.second);
    std::mt19937 engine(seedgen_());
    return [distrib, engine]() mutable { return distrib(engine); };
  }

  F f_;
  std::size_t precision_;
  double total_space_;
  std::array<std::pair<double, double>, N> limits_;
  std::array<std::function<double()>, N> generators_;
  std::random_device seedgen_;
  double result_;
};

}  // namespace kazunin_n_montecarlo_seq