#pragma once

#include <boost/mpi/communicator.hpp>
#include <boost/serialization/access.hpp>
#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kondratev_ya_ccs_complex_multiplication_all {

constexpr double kEpsilon = 1e-10;
constexpr double kEpsilonForZero = kEpsilon * kEpsilon;

bool IsZero(const std::complex<double>& value);
bool IsEqual(const std::complex<double>& a, const std::complex<double>& b);

class CCSMatrix {
 public:
  friend class boost::serialization::access;
  friend class TestTaskALL;

  // clang-format off
  // NOLINTBEGIN(*)
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {
    ar& rows;
    ar& cols;
    ar& values;
    ar& row_index;
    ar& col_ptrs;
  }
  // NOLINTEND(*)
  // clang-format on

  int rows;
  int cols;
  std::vector<std::complex<double>> values;
  std::vector<int> row_index;
  std::vector<int> col_ptrs;

  [[nodiscard]] std::vector<std::vector<std::pair<int, std::complex<double>>>> ComputeColumns(
      const CCSMatrix& other) const;

  CCSMatrix() : rows(0), cols(0) {}
  explicit CCSMatrix(std::pair<int, int> size) : rows(size.first), cols(size.second) { col_ptrs.resize(cols + 1, 0); }

  CCSMatrix operator*(const CCSMatrix& other) const;
};

struct ColumnUpdateData {
  int global_col_idx;
  std::vector<std::complex<double>> values_to_insert;
  std::vector<int> row_indices_to_insert;
};

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  TestTaskALL(const TestTaskALL&) = delete;
  TestTaskALL& operator=(const TestTaskALL&) = delete;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  CCSMatrix a_, b_, c_;
  boost::mpi::communicator world_;

  [[nodiscard]] static std::pair<int, int> GetLocalColumnRange(int rank, int total_cols);
  void CreateLocalMatrix(int rank, int size, const std::vector<std::pair<int, int>>& all_local_cols, CCSMatrix& out);
  void MergeResults(const std::vector<CCSMatrix>& all_results, const std::vector<std::pair<int, int>>& all_local_cols);
  static void ProcessBlocksInParallel(const std::vector<CCSMatrix>& all_results,
                                      const std::vector<std::pair<int, int>>& all_local_cols,
                                      std::vector<std::vector<ColumnUpdateData>>& thread_local_updates, int size);
  void UpdateResultMatrix(const std::vector<ColumnUpdateData>& all_column_updates);
};

}  // namespace kondratev_ya_ccs_complex_multiplication_all