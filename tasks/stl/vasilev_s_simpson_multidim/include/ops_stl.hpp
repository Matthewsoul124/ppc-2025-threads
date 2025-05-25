#pragma once

#include <cstddef>
#include <future>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasilev_s_simpson_multidim {

using Coordinate = std::span<double>;
using IntegrandFunction = double (*)(const Coordinate&);

struct Bound {
  double lo, hi;
};

class SimpsonTaskStl : public ppc::core::Task {
 public:
  explicit SimpsonTaskStl(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void ComputeThreadSum(std::pair<std::size_t, std::size_t> range, std::promise<double>&& promise);

  IntegrandFunction func_;
  std::size_t arity_;
  std::size_t approxs_;
  std::vector<Bound> bounds_;

  std::size_t gridcap_;
  std::vector<double> steps_;
  double scale_;

  double result_;
};

}  // namespace vasilev_s_simpson_multidim
