#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace milovankin_m_histogram_stretching_all {

class TestTaskAll : public ppc::core::Task {
 public:
  explicit TestTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> img_;
  boost::mpi::communicator world_;
};

}  // namespace milovankin_m_histogram_stretching_all