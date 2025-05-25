#pragma once

#include <boost/mpi/communicator.hpp>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lopatin_i_monte_carlo_all {

std::vector<double> GenerateBounds(double min_val, double max_val, int dimensions);

class TestTaskAll : public ppc::core::Task {
 public:
  using IntegrandFunction = double(const std::vector<double>&);
  explicit TestTaskAll(ppc::core::TaskDataPtr task_data, std::function<IntegrandFunction> func)
      : Task(std::move(task_data)), integrand_(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> integrationBounds_;
  std::function<IntegrandFunction> integrand_;
  double result_{};
  int iterations_;

  boost::mpi::communicator world_;
};

}  // namespace lopatin_i_monte_carlo_all