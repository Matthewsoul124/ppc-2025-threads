#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anufriev_d_integrals_simpson_stl {

class IntegralsSimpsonSTL : public ppc::core::Task {
 public:
  explicit IntegralsSimpsonSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  struct IterationRange {
    int start;
    int end;
  };

  int dimension_{};
  std::vector<double> a_, b_;
  std::vector<int> n_;
  int func_code_{};
  double result_{};

  [[nodiscard]] double FunctionN(const std::vector<double>& coords) const;
  double RecursiveSimpsonSum(int dim_index, std::vector<int>& idx, const std::vector<double>& steps) const;
  void ThreadTaskRunner(int start_idx, int end_idx, const std::vector<double>& steps, double* partial_sum_output);

  [[nodiscard]] static unsigned int DetermineNumThreads(int total_iterations);
  [[nodiscard]] static std::vector<IterationRange> DistributeIterations(int total_iterations, unsigned int num_threads);
};

}  // namespace anufriev_d_integrals_simpson_stl