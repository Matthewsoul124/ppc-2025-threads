#include "all/tyurin_m_matmul_crs_complex/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "mpi.h"

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

bool tyurin_m_matmul_crs_complex_all::TestTaskAll::ValidationImpl() {
  if (rank_ != 0) {
    return true;
  }
  const bool left_cols_equal_right_rows = task_data->inputs_count[1] == task_data->inputs_count[2];
  const bool there_are_rows_and_cols =
      task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
  return left_cols_equal_right_rows && there_are_rows_and_cols && task_data->outputs_count[0] == 1;
}

bool tyurin_m_matmul_crs_complex_all::TestTaskAll::PreProcessingImpl() {
  if (rank_ != 0) {
    return true;
  }
  global_lhs_ = *reinterpret_cast<MatrixCRS *>(task_data->inputs[0]);
  rhs_ = TransposeMatrixCRS(*reinterpret_cast<MatrixCRS *>(task_data->inputs[1]));
  return true;
}

bool tyurin_m_matmul_crs_complex_all::TestTaskAll::RunImpl() {
  int row_offset{};
  int idx_offset{};
  auto local_lhs = Scatter(row_offset, idx_offset);

  MatrixCRS local_res{};
  local_res.rowptr.resize(local_lhs.GetRows() + 1);
  local_res.cols_count = rhs_.GetRows();

  const auto rows = local_lhs.GetRows();
  const auto cols = rhs_.GetRows();

  std::vector<std::vector<std::tuple<std::complex<double>, uint32_t>>> buf(rows);

  const std::size_t nthreads = ppc::util::GetPPCNumThreads();
  const std::size_t avg = rows / nthreads;
  const std::size_t nextra = rows % nthreads;

  auto mulrow = [&](uint32_t i) {
    for (uint32_t j = 0; j < cols; ++j) {
      auto ii = local_lhs.rowptr[i - row_offset];
      auto ij = rhs_.rowptr[j];
      std::complex<double> summul = 0.0;
      while (ii < local_lhs.rowptr[i + 1 - row_offset] && ij < rhs_.rowptr[j + 1]) {
        if (local_lhs.colind[ii - idx_offset] < rhs_.colind[ij]) {
          ++ii;
        } else if (local_lhs.colind[ii - idx_offset] > rhs_.colind[ij]) {
          ++ij;
        } else {
          summul += local_lhs.data[ii++ - idx_offset] * rhs_.data[ij++];
        }
      }
      if (summul != 0.0) {
        buf[i - row_offset].emplace_back(summul, j);
      }
    }
  };
  auto thexec = [&](uint32_t thread_rows_begin, uint32_t thread_rows_end) {
    for (uint32_t i = row_offset + thread_rows_begin; i < row_offset + thread_rows_end; ++i) {
      mulrow(i);
    }
  };

  std::vector<std::thread> threads(nthreads);
  uint32_t cur = 0;
  for (std::size_t t = 0; t < nthreads; t++) {
    uint32_t forthread = avg + ((t < nextra) ? 1 : 0);
    threads[t] = std::thread(thexec, cur, cur + forthread);
    cur += forthread;
  }
  std::ranges::for_each(threads, [](auto &thread) { thread.join(); });

  for (uint32_t i = 0; i < rows; i++) {
    local_res.rowptr[i + 1] = local_res.rowptr[i];
    for (const auto &[summul, j] : buf[i]) {
      local_res.data.push_back(summul);
      local_res.colind.push_back(j);
      ++local_res.rowptr[i + 1];
    }
  }

  procres_ = Gather(std::move(local_res));

  return true;
}

bool tyurin_m_matmul_crs_complex_all::TestTaskAll::PostProcessingImpl() {
  if (rank_ != 0) {
    return true;
  }

  const auto size = std::accumulate(procres_.begin(), procres_.end(), size_t(0),
                                    [](size_t acc, MatrixCRS &part) { return acc + part.data.size(); });

  MatrixCRS global_res{};
  global_res.rowptr.resize(global_lhs_.GetRows() + 1);
  global_res.cols_count = rhs_.GetRows();
  global_res.colind.resize(size);
  global_res.data.resize(size);

  auto rowptr_iter = global_res.rowptr.begin() + 1;
  auto colind_iter = global_res.colind.begin();
  auto data_iter = global_res.data.begin();
  //
  size_t nz = 0;
  for (const auto &part : procres_) {
    auto prev_it_rowp = rowptr_iter;

    rowptr_iter = std::copy(part.rowptr.begin() + 1, part.rowptr.end(), rowptr_iter);
    colind_iter = std::ranges::copy(part.colind, colind_iter).out;
    data_iter = std::ranges::copy(part.data, data_iter).out;

    std::ranges::for_each(std::ranges::subrange{prev_it_rowp, rowptr_iter}, [&](auto &rp) { rp += nz; });
    nz = *(rowptr_iter - 1);
  }

  *reinterpret_cast<MatrixCRS *>(task_data->outputs[0]) = global_res;
  return true;
}

MatrixCRS tyurin_m_matmul_crs_complex_all::TestTaskAll::Scatter(int &row_offset, int &idx_offset) {
  rhs_.Bcast(MPI_COMM_WORLD, 0);

  int offsets_buf[2]{};

  if (rank_ == 0) {
    int workers{};
    MPI_Comm_size(MPI_COMM_WORLD, &workers);

    std::vector<std::size_t> dist(workers, global_lhs_.GetRows() / workers);
    std::for_each(dist.begin(), dist.begin() + (global_lhs_.GetRows() % workers), [](auto &e) { ++e; });

    std::size_t row_range_begin = 0;
    std::size_t row_range_end = dist[0];

    row_offset = idx_offset = 0;
    auto w0m = global_lhs_.ExtractPart({row_range_begin, row_range_end});

    for (int p = 1; p < workers; ++p) {
      row_range_begin = row_range_end;
      row_range_end += dist[p];

      const MatrixCRS::TSubrange subrange{row_range_begin, row_range_end};
      global_lhs_.GetOffsets(subrange, offsets_buf);
      MPI_Send(offsets_buf, 2, MPI_INT, p, 0, MPI_COMM_WORLD);

      global_lhs_.SendPart(MPI_COMM_WORLD, p, subrange);
    }

    return w0m;
  }

  MPI_Recv(offsets_buf, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  row_offset = offsets_buf[0];
  idx_offset = offsets_buf[1];
  return MatrixCRS::Recv(MPI_COMM_WORLD, 0);
}

std::vector<MatrixCRS> tyurin_m_matmul_crs_complex_all::TestTaskAll::Gather(MatrixCRS &&local_res) const {
  if (rank_ != 0) {
    local_res.Send(MPI_COMM_WORLD, 0);
    return {};
  }

  int workers{};
  MPI_Comm_size(MPI_COMM_WORLD, &workers);
  workers = std::max(1, workers);  // Wnull-dereference

  std::vector<MatrixCRS> v(workers);

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
  v[0] = std::move(local_res);
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif

  for (int p = 1; p < workers; ++p) {
    v[p] = MatrixCRS::Recv(MPI_COMM_WORLD, p);
  }

  return v;
}