#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vasilev_s_simpson_multidim {

using Coordinate = std::span<double>;
using IntegrandFunction = double (*)(const Coordinate&);

struct Bound {
  double lo, hi;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {  // NOLINT
    ar & lo & hi;
  }
};

class SimpsonTaskAll : public ppc::core::Task {
 public:
  explicit SimpsonTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void CalcSteps();

  boost::mpi::communicator world_;

  IntegrandFunction func_;
  std::size_t arity_;
  std::size_t approxs_;
  std::vector<Bound> bounds_;

  std::size_t gridcap_;
  std::vector<double> steps_;
  double scale_;

  double result_;
};

}  // namespace vasilev_s_simpson_multidim
