#pragma once
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_multi_dim_monte_carlo_all {
class MultiDimMonteCarloTask : public ppc::core::Task {
 public:
  explicit MultiDimMonteCarloTask(std::shared_ptr<ppc::core::TaskData> task_data,
                                  std::function<double(const std::vector<double>&)> integrating_function)
      : Task(std::move(task_data)), integrating_function_(std::move(integrating_function)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> boundaries_;
  boost::mpi::communicator world_;
  int number_of_iterations_ = 0;
  double final_result_ = 0.0;
  std::function<double(const std::vector<double>&)> integrating_function_;
};
}  // namespace sharamygina_i_multi_dim_monte_carlo_all