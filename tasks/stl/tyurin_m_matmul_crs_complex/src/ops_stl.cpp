#include "stl/tyurin_m_matmul_crs_complex/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <tuple>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
MatrixCRS TransposeMatrixCRS(const MatrixCRS &crs) {
  const auto new_cols = crs.GetRows();

  MatrixCRS res;
  res.cols_count = new_cols;
  res.rowptr.resize(crs.GetCols() + 2);
  res.colind.resize(crs.colind.size(), 0);
  res.data.resize(crs.data.size(), 0);

  for (uint32_t i = 0; i < crs.data.size(); ++i) {
    ++res.rowptr[crs.colind[i] + 2];
  }
  for (uint32_t i = 2; i < res.rowptr.size(); ++i) {
    res.rowptr[i] += res.rowptr[i - 1];
  }
  for (uint32_t i = 0; i < new_cols; ++i) {
    for (uint32_t j = crs.rowptr[i]; j < crs.rowptr[i + 1]; ++j) {
      const auto new_index = res.rowptr[crs.colind[j] + 1]++;
      res.data[new_index] = crs.data[j];
      res.colind[new_index] = i;
    }
  }
  res.rowptr.pop_back();

  return res;
}
}  // namespace

bool tyurin_m_matmul_crs_complex_stl::TestTaskStl::ValidationImpl() {
  const bool left_cols_equal_right_rows = task_data->inputs_count[1] == task_data->inputs_count[2];
  const bool there_are_rows_and_cols =
      task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
  return left_cols_equal_right_rows && there_are_rows_and_cols && task_data->outputs_count[0] == 1;
}

bool tyurin_m_matmul_crs_complex_stl::TestTaskStl::PreProcessingImpl() {
  lhs_ = *reinterpret_cast<MatrixCRS *>(task_data->inputs[0]);
  rhs_ = TransposeMatrixCRS(*reinterpret_cast<MatrixCRS *>(task_data->inputs[1]));
  res_ = {};
  res_.rowptr.resize(lhs_.GetRows() + 1);
  res_.cols_count = rhs_.GetRows();
  return true;
}

bool tyurin_m_matmul_crs_complex_stl::TestTaskStl::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  const auto rows = lhs_.GetRows();
  const auto cols = rhs_.GetRows();

  std::vector<std::vector<std::tuple<std::complex<double>, uint32_t>>> buf(rows);

  const std::size_t nthreads = ppc::util::GetPPCNumThreads();
  const std::size_t avg = rows / nthreads;
  const std::size_t nextra = rows % nthreads;

  std::vector<std::thread> threads(nthreads);
  uint32_t cur = 0;
  for (std::size_t t = 0; t < nthreads; t++) {
    uint32_t forthread = avg + ((t < nextra) ? 1 : 0);
    threads[t] = std::thread(
        [&](uint32_t thread_rows_begin, uint32_t thread_rows_end) {
          for (uint32_t i = thread_rows_begin; i < thread_rows_end; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
              auto ii = lhs_.rowptr[i];
              auto ij = rhs_.rowptr[j];
              std::complex<double> summul = 0.0;
              while (ii < lhs_.rowptr[i + 1] && ij < rhs_.rowptr[j + 1]) {
                if (lhs_.colind[ii] < rhs_.colind[ij]) {
                  ++ii;
                } else if (lhs_.colind[ii] > rhs_.colind[ij]) {
                  ++ij;
                } else {
                  summul += lhs_.data[ii++] * rhs_.data[ij++];
                }
              }
              if (summul != 0.0) {
                buf[i].emplace_back(summul, j);
              }
            }
          }
        },
        cur, cur + forthread);
    cur += forthread;
  }
  std::ranges::for_each(threads, [](auto &thread) { thread.join(); });

  for (uint32_t i = 0; i < rows; i++) {
    res_.rowptr[i + 1] = res_.rowptr[i];
    for (const auto &[summul, j] : buf[i]) {
      res_.data.push_back(summul);
      res_.colind.push_back(j);
      ++res_.rowptr[i + 1];
    }
  }

  return true;
}

bool tyurin_m_matmul_crs_complex_stl::TestTaskStl::PostProcessingImpl() {
  *reinterpret_cast<MatrixCRS *>(task_data->outputs[0]) = res_;
  return true;
}
