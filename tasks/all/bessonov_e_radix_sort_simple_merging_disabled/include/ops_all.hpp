#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_radix_sort_simple_merging_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  boost::mpi::communicator world_;

  void HandleSingleProcess();
  void HandleParallelProcess();
  static void MergeChunks(std::deque<std::vector<double>>& chunks);

  static void ConvertDoubleToBits(const std::vector<double>& input, std::vector<uint64_t>& bits, size_t start,
                                  size_t end);
  static void ConvertBitsToDouble(const std::vector<uint64_t>& bits, std::vector<double>& output, size_t start,
                                  size_t end);
  static void RadixSortPass(std::vector<uint64_t>& bits, std::vector<uint64_t>& temp, int shift);
  static std::vector<double> Merge(const std::vector<double>& left, const std::vector<double>& right);
};
}  // namespace bessonov_e_radix_sort_simple_merging_all