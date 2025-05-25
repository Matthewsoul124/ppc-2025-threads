#include "all/lavrentiev_A_CCS/include/ops_stl.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::ConvertToSparse(std::pair<int, int> size,
                                                                           const std::vector<double> &values) {
  auto [nsize, elements, rows, columns_sum] = Sparse();
  columns_sum.resize(size.second);
  for (int i = 0; i < size.second; ++i) {
    for (int j = 0; j < size.first; ++j) {
      if (values[i + (size.second * j)] != 0) {
        elements.emplace_back(values[i + (size.second * j)]);
        rows.emplace_back(j);
        columns_sum[i] += 1;
      }
    }
    if (i != size.second - 1) {
      columns_sum[i + 1] = columns_sum[i];
    }
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::Transpose(const Sparse &sparse) {
  auto [size, elements, rows, columns_sum] = Sparse();
  size.first = sparse.size.second;
  size.second = sparse.size.first;
  int need_size = std::max(sparse.size.first, sparse.size.second);
  std::vector<std::vector<std::pair<double, int>>> new_elements_and_rows(need_size);
  int counter = 0;
  for (int i = 0; i < static_cast<int>(sparse.columnsSum.size()); ++i) {
    for (int j = 0; j < GetElementsCount(i, sparse.columnsSum); ++j) {
      new_elements_and_rows[sparse.rows[counter]].emplace_back(sparse.elements[counter], i);
      counter++;
    }
  }
  elements.reserve(counter);
  rows.reserve(counter);
  for (int i = 0; i < static_cast<int>(new_elements_and_rows.size()); ++i) {
    for (int j = 0; j < static_cast<int>(new_elements_and_rows[i].size()); ++j) {
      elements.emplace_back(new_elements_and_rows[i][j].first);
      rows.emplace_back(new_elements_and_rows[i][j].second);
    }
    i > 0 ? columns_sum.emplace_back(new_elements_and_rows[i].size() + columns_sum[i - 1])
          : columns_sum.emplace_back(new_elements_and_rows[i].size());
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

int lavrentiev_a_ccs_all::CCSALL::CalculateStartIndex(int index, const std::vector<int> &columns_sum) {
  return index == 0 ? 0 : columns_sum[index] - GetElementsCount(index, columns_sum);
}

lavrentiev_a_ccs_all::Sparse lavrentiev_a_ccs_all::CCSALL::MatMul(const Sparse &matrix1, const Sparse &matrix2,
                                                                  int interval_begin, int interval_end) {
  Sparse temporary_matrix;
  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads());
  temporary_matrix.columnsSum.resize(matrix2.size.second);
  temporary_matrix.elements.resize((matrix2.columnsSum.size() * matrix1.columnsSum.size()) +
                                   std::max(matrix1.columnsSum.size(), matrix2.columnsSum.size()));
  temporary_matrix.rows.resize((matrix2.columnsSum.size() * matrix1.columnsSum.size()) +
                               std::max(matrix1.columnsSum.size(), matrix2.columnsSum.size()));
  auto accumulate = [&](int i_index, int j_index) {
    double sum = 0.0;
    for (int x = 0; x < GetElementsCount(j_index, matrix1.columnsSum); x++) {
      for (int y = 0; y < GetElementsCount(i_index, matrix2.columnsSum); y++) {
        if (matrix1.rows[CalculateStartIndex(j_index, matrix1.columnsSum) + x] ==
            matrix2.rows[CalculateStartIndex(i_index, matrix2.columnsSum) + y]) {
          sum += matrix1.elements[x + CalculateStartIndex(j_index, matrix1.columnsSum)] *
                 matrix2.elements[y + CalculateStartIndex(i_index, matrix2.columnsSum)];
        }
      }
    }
    return sum;
  };
  auto matrix_multiplicator = [&](int begin, int end) {
    for (int i = begin; i != end; ++i) {
      if (temporary_matrix.columnsSum[i] == 0) {
        temporary_matrix.columnsSum[i]++;
      }
      for (int j = 0; j < static_cast<int>(matrix1.columnsSum.size()); ++j) {
        double s = accumulate(i, j);
        if (s != 0) {
          temporary_matrix.elements[(i * matrix2.size.second) + j] = s;
          temporary_matrix.rows[(i * matrix2.size.second) + j] = j + 1;
          temporary_matrix.columnsSum[i]++;
        }
      }
    }
  };
  int thread_data_amount = (interval_end - interval_begin) / ppc::util::GetPPCNumThreads();
  for (size_t i = 0; i < threads.size(); ++i) {
    if (i != threads.size() - 1) {
      threads[i] = std::thread(matrix_multiplicator, interval_begin + (i * thread_data_amount),
                               interval_begin + ((i + 1) * thread_data_amount));
    } else {
      threads[i] = std::thread(matrix_multiplicator, interval_begin + (i * thread_data_amount),
                               interval_begin + ((i + 1) * thread_data_amount) +
                                   ((interval_end - interval_begin) % ppc::util::GetPPCNumThreads()));
    }
  }
  std::ranges::for_each(threads, [&](std::thread &thread) { thread.join(); });
  std::erase_if(temporary_matrix.elements, [](auto &current_element) { return current_element == 0.0; });
  std::erase_if(temporary_matrix.rows, [](auto &current_element) { return current_element == 0; });
  return {.size = temporary_matrix.size,
          .elements = temporary_matrix.elements,
          .rows = temporary_matrix.rows,
          .columnsSum = temporary_matrix.columnsSum};
}

int lavrentiev_a_ccs_all::CCSALL::GetElementsCount(int index, const std::vector<int> &columns_sum) {
  if (index == 0) {
    return columns_sum[index];
  }
  return columns_sum[index] - columns_sum[index - 1];
}

std::vector<double> lavrentiev_a_ccs_all::CCSALL::ConvertFromSparse(const Sparse &matrix) {
  std::vector<double> nmatrix(matrix.size.first * matrix.size.second);
  int counter = 0;
  for (size_t i = 0; i < matrix.columnsSum.size(); ++i) {
    for (int j = 0; j < GetElementsCount(static_cast<int>(i), matrix.columnsSum); ++j) {
      nmatrix[i + (matrix.size.second * matrix.rows[counter])] = matrix.elements[counter];
      counter++;
    }
  }
  return nmatrix;
}

void lavrentiev_a_ccs_all::CCSALL::GetDisplacements() {
  displ_.resize(world_.size());
  int amount = static_cast<int>(B_.columnsSum.size()) % world_.size();
  int count = static_cast<int>(B_.columnsSum.size()) / world_.size();
  if (static_cast<int>(B_.columnsSum.size()) == 0) {
    return;
  }
  for (int i = 0; i < world_.size(); ++i) {
    if (i != 0) {
      displ_[i] = count + displ_[i - 1];
      if (amount != 0) {
        displ_[i]++;
        amount--;
      }
    }
  }
  displ_.emplace_back(static_cast<int>(B_.columnsSum.size()));
}

bool lavrentiev_a_ccs_all::CCSALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    A_.size = {static_cast<int>(task_data->inputs_count[0]), static_cast<int>(task_data->inputs_count[1])};
    B_.size = {static_cast<int>(task_data->inputs_count[2]), static_cast<int>(task_data->inputs_count[3])};
    if (IsEmpty()) {
      return true;
    }
    auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    auto am = std::vector<double>(in_ptr, in_ptr + (A_.size.first * A_.size.second));
    A_ = ConvertToSparse(A_.size, am);
    A_ = Transpose(A_);
    auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
    auto bm = std::vector<double>(in_ptr2, in_ptr2 + (B_.size.first * B_.size.second));
    B_ = ConvertToSparse(B_.size, bm);
    GetDisplacements();
  }
  return true;
}

bool lavrentiev_a_ccs_all::CCSALL::IsEmpty() const {
  return A_.size.first * A_.size.second == 0 || B_.size.first * B_.size.second == 0;
}

void lavrentiev_a_ccs_all::CCSALL::CollectSizes() {
  if (world_.rank() != 0) {
    world_.send(0, 0, static_cast<int>(Process_data_.elements.size()));
    world_.send(0, 1, static_cast<int>(Process_data_.columnsSum.size()));
  } else {
    sum_sizes_.resize(world_.size(), static_cast<int>(Process_data_.columnsSum.size()));
    elements_sizes_.resize(world_.size(), static_cast<int>(Process_data_.elements.size()));
    for (int i = 1; i < world_.size(); ++i) {
      world_.recv(i, 0, elements_sizes_[i]);
      world_.recv(i, 1, sum_sizes_[i]);
    }
  }
}
bool lavrentiev_a_ccs_all::CCSALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0] &&
           task_data->inputs_count[0] == task_data->inputs_count[3] &&
           task_data->inputs_count[1] == task_data->inputs_count[2];
  }
  return true;
}

bool lavrentiev_a_ccs_all::CCSALL::RunImpl() {
  boost::mpi::broadcast(world_, displ_, 0);
  boost::mpi::broadcast(world_, A_, 0);
  boost::mpi::broadcast(world_, B_, 0);
  if (displ_.empty() || IsEmpty()) {
    return true;
  }
  Process_data_ = MatMul(A_, B_, displ_[world_.rank()], displ_[world_.rank() + 1]);
  CollectSizes();
  if (world_.rank() == 0) {
    Sparse data_collector;
    std::vector<int> columns_nums_collector(B_.columnsSum.size() * world_.size());
    auto size = std::accumulate(elements_sizes_.begin(), elements_sizes_.end(), 0);
    data_collector.elements.resize(size);
    data_collector.rows.resize(size);
    boost::mpi::gatherv(world_, Process_data_.elements, data_collector.elements.data(), elements_sizes_, 0);
    boost::mpi::gatherv(world_, Process_data_.rows, data_collector.rows.data(), elements_sizes_, 0);
    boost::mpi::gatherv(world_, Process_data_.columnsSum, columns_nums_collector.data(), sum_sizes_, 0);
    for (auto &element : columns_nums_collector) {
      if (element != 0) {
        data_collector.columnsSum.emplace_back(--element);
      }
    }
    for (size_t i = 1; i < data_collector.columnsSum.size(); ++i) {
      data_collector.columnsSum[i] = data_collector.columnsSum[i] + data_collector.columnsSum[i - 1];
    }
    std::ranges::for_each(data_collector.rows, [&](auto &row) { row--; });
    Answer_ = std::move(data_collector);
    Answer_.size.first = B_.size.second;
    Answer_.size.second = B_.size.second;
  } else {
    boost::mpi::gatherv(world_, Process_data_.elements, 0);
    boost::mpi::gatherv(world_, Process_data_.rows, 0);
    boost::mpi::gatherv(world_, Process_data_.columnsSum, 0);
  }
  return true;
}

bool lavrentiev_a_ccs_all::CCSALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto answer = ConvertFromSparse(Answer_);
    if (!answer.empty()) {
      std::ranges::copy(answer, reinterpret_cast<double *>(task_data->outputs[0]));
    }
  }
  return true;
}