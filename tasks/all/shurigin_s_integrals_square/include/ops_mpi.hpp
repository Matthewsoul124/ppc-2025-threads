#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace shurigin_s_integrals_square_mpi {

class Integral : public ppc::core::Task {
 public:
  explicit Integral(const std::shared_ptr<ppc::core::TaskData>& task_data_param);

  void SetFunction(const std::function<double(double)>& func);
  void SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions);

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> down_limits_;
  std::vector<double> up_limits_;
  std::vector<int> counts_;

  double result_;

  std::function<double(const std::vector<double>&)> func_;
  int dimensions_;

  int mpi_rank_;
  int mpi_world_size_;

  static double ComputeOneDimensionalOMP(const std::function<double(const std::vector<double>&)>& f, double a_local,
                                         double b_local, int n_local);

  static double ComputeOuterParallelInnerSequential(const std::function<double(const std::vector<double>&)>& f,
                                                    double a0_local_mpi, double b0_local_mpi, int n0_local_mpi,
                                                    const std::vector<double>& full_a,
                                                    const std::vector<double>& full_b, const std::vector<int>& full_n,
                                                    int total_dims);

  static double ComputeSequentialRecursive(const std::function<double(const std::vector<double>&)>& f,
                                           const std::vector<double>& a_all_dims, const std::vector<double>& b_all_dims,
                                           const std::vector<int>& n_all_dims, int total_dims,
                                           std::vector<double>& current_eval_point, int current_dim_index);
};

}  // namespace shurigin_s_integrals_square_mpi