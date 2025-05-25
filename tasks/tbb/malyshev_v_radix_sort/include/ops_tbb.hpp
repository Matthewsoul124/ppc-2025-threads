#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_radix_sort_tbb {

class SortTBB : public ppc::core::Task {
 public:
  explicit SortTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
};

}  // namespace malyshev_v_radix_sort_tbb