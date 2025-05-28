#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace zolotareva_a_sle_gradient_method_all {
void GenerateSle(std::vector<double>& a, std::vector<double>& b, int n);
class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  inline static bool IsPositiveAndSimm(const double* a, int n);
  inline void DistributeData(int world_size, int base_rows);
  inline void CalculateIterations(double threshold, std::vector<double>& p, std::vector<double>& global_p,
                                  std::vector<double>& r, double rs_global_old, std::vector<int>& recvcounts,
                                  std::vector<int>& displs, std::vector<double>& ap);

 private:
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> x_;
  std::vector<double> local_a_;
  std::vector<double> local_b_;
  std::vector<double> local_x_;
  int n_{0};
  int local_rows_{0};
  boost::mpi::communicator world_;
};

}  // namespace zolotareva_a_sle_gradient_method_all