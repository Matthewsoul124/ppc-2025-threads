#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kolokolova_d_integral_simpson_method_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data, std::function<double(std::vector<double>)> func)
      : Task(std::move(task_data)), func_(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<double> FindFunctionValue(const std::vector<std::vector<double>>& coordinates,
                                        const std::function<double(std::vector<double>)>& f);
  void GeneratePointsAndEvaluate(const std::vector<std::vector<double>>& coordinates, int index,
                                 std::vector<double>& current, std::vector<double>& results,
                                 const std::function<double(const std::vector<double>)>& f);
  static std::vector<double> FindCoeff(int count_step);
  static bool CheckBorders(std::vector<int> vec);
  void CalculateStepSizes();
  void CreatePointsVector();
  void PrepareCoefficientsAndResults();
  void ApplyCoefficientIteration();

 private:
  double result_output_ = 0;
  int nums_variables_ = 0;
  double local_results_output_ = 0;
  std::vector<double> results_func_;
  std::vector<double> local_results_func_;
  int size_local_results_func_ = 0;
  std::vector<double> coeff_;
  std::vector<double> vec_coeff_;
  std::vector<double> local_coeff_;
  int size_local_coeff_ = 0;
  std::vector<int> steps_;
  std::vector<int> borders_;
  std::vector<double> size_step_;
  std::vector<double> local_size_step_;
  int size_local_size_step_ = 0;
  std::vector<std::vector<double>> points_;
  std::function<double(std::vector<double>)> func_;
  boost::mpi::communicator world_;
};

}  // namespace kolokolova_d_integral_simpson_method_all