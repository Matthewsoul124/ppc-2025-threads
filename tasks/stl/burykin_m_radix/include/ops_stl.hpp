#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_radix_stl {

class RadixSTL : public ppc::core::Task {
 public:
  explicit RadixSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::array<int, 256> ComputeFrequency(const std::vector<int>& a, int shift);
  static std::array<int, 256> ComputeFrequencyParallel(const std::vector<int>& a, int shift, int num_threads);
  static std::array<int, 256> ComputeIndices(const std::array<int, 256>& count);
  static void DistributeElements(const std::vector<int>& a, std::vector<int>& b, std::array<int, 256> index, int shift);
  static void DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                         const std::array<int, 256>& global_index, int shift, int num_threads);
  static void ComputeThreadCounts(const std::vector<int>& a, std::vector<std::array<int, 256>>& thread_counts,
                                  int shift, int num_threads, size_t chunk_size);
  static void ComputeThreadIndices(std::vector<std::array<int, 256>>& thread_indices,
                                   const std::vector<std::array<int, 256>>& thread_counts,
                                   const std::array<int, 256>& global_index, int num_threads);

 private:
  std::vector<int> input_, output_;
};

}  // namespace burykin_m_radix_stl