#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace poroshin_v_multi_integral_with_trapez_method_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(std::shared_ptr<ppc::core::TaskData> &task_data,
                       std::function<double(std::vector<double> &args)> func)
      : Task(std::move(task_data)), func_(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void CountMultiIntegralTrapezMethodAll(double &res);
  std::vector<std::pair<double, double>> limits_;
  size_t dim_{};
  std::function<double(std::vector<double> &args)> func_;
  std::vector<int> n_;
  double res_{};
  boost::mpi::communicator world_;
};

}  // namespace poroshin_v_multi_integral_with_trapez_method_all