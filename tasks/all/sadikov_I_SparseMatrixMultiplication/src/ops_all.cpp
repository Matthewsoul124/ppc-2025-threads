#include "all/sadikov_I_SparseMatrixMultiplication/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cstddef>
#include <optional>
#include <vector>

#include "all/sadikov_I_SparseMatrixMultiplication/include/SparseMatrix.hpp"

void sadikov_i_sparse_matrix_multiplication_task_all::CCSMatrixALL::CalculateDisplacements() {
  int n = static_cast<int>(m_sMatrix_.GetElementsSum().size()) % m_world_.size();
  int count = static_cast<int>(m_sMatrix_.GetElementsSum().size()) / m_world_.size();
  if (m_sMatrix_.GetElementsSum().empty()) {
    return;
  }
  m_displacements_.resize(m_world_.size());
  for (int i = 0; i < m_world_.size(); ++i) {
    if (i != 0) {
      m_displacements_[i] = count + m_displacements_[i - 1];
      if (i < n) {
        m_displacements_[i]++;
      }
    }
  }
}

bool sadikov_i_sparse_matrix_multiplication_task_all::CCSMatrixALL::PreProcessingImpl() {
  if (m_world_.rank() == 0) {
    auto fmatrix_rows_count = static_cast<int>(task_data->inputs_count[0]);
    auto fmatrxix_columns_count = static_cast<int>(task_data->inputs_count[1]);
    auto smatrix_rows_count = static_cast<int>(task_data->inputs_count[2]);
    auto smatrix_columns_count = static_cast<int>(task_data->inputs_count[3]);
    if (fmatrix_rows_count == 0 || fmatrxix_columns_count == 0 || smatrix_rows_count == 0 ||
        smatrix_columns_count == 0) {
      return true;
    }
    auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    auto fmatrix = std::vector<double>(in_ptr, in_ptr + (fmatrix_rows_count * fmatrxix_columns_count));
    auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
    auto smatrix = std::vector<double>(in_ptr2, in_ptr2 + (smatrix_columns_count * smatrix_rows_count));
    m_fMatrix_ = SparseMatrix::MatrixToSparse(fmatrix_rows_count, fmatrxix_columns_count, fmatrix);
    m_sMatrix_ = SparseMatrix::MatrixToSparse(smatrix_rows_count, smatrix_columns_count, smatrix);
    m_fMatrix_ = SparseMatrix::Transpose(m_fMatrix_);
    m_sizes_.first.resize(m_world_.size());
    m_sizes_.second.resize(m_world_.size());
    CalculateDisplacements();
  }
  return true;
}

bool sadikov_i_sparse_matrix_multiplication_task_all::CCSMatrixALL::ValidationImpl() {
  if (m_world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->inputs_count[3] &&
           task_data->inputs_count[1] == task_data->inputs_count[2] &&
           task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0];
  }
  return true;
}

bool sadikov_i_sparse_matrix_multiplication_task_all::CCSMatrixALL::RunImpl() {
  boost::mpi::broadcast(m_world_, m_fMatrix_, 0);
  boost::mpi::broadcast(m_world_, m_sMatrix_, 0);
  boost::mpi::broadcast(m_world_, m_displacements_, 0);
  if (m_displacements_.empty()) {
    return true;
  }
  m_intermediate_data_ = SparseMatrix::Multiplicate(m_fMatrix_, m_sMatrix_, m_displacements_[m_world_.rank()],
                                                    m_world_.rank() == m_world_.size() - 1
                                                        ? static_cast<int>(m_sMatrix_.GetElementsSum().size())
                                                        : static_cast<int>(m_displacements_[m_world_.rank() + 1]));
  if (m_world_.rank() != 0) {
    m_world_.send(0, 0, static_cast<int>(m_intermediate_data_.m_values.size()));
    m_world_.send(0, 1, static_cast<int>(m_intermediate_data_.m_elementsSum.size()));
  } else {
    m_sizes_.first[0] = static_cast<int>(m_intermediate_data_.m_values.size());
    m_sizes_.second[0] = static_cast<int>(m_intermediate_data_.m_elementsSum.size());

    for (int i = 1; i < m_world_.size(); ++i) {
      m_world_.recv(i, 0, m_sizes_.first[i]);
      m_world_.recv(i, 1, m_sizes_.second[i]);
    }
  }
  if (m_world_.rank() == 0) {
    MatrixComponents component;
    component.Resize(m_fMatrix_.GetElementsSum().size() * m_sMatrix_.GetElementsSum().size() * m_world_.size(),
                     std::nullopt);
    MatrixComponents intermediate_component;
    intermediate_component.Resize(
        m_fMatrix_.GetElementsSum().size() * m_sMatrix_.GetElementsSum().size() * m_world_.size(),
        m_sMatrix_.GetElementsSum().size() * m_world_.size());
    boost::mpi::gatherv(m_world_, m_intermediate_data_.m_values, intermediate_component.m_values.data(), m_sizes_.first,
                        0);
    boost::mpi::gatherv(m_world_, m_intermediate_data_.m_rows, intermediate_component.m_rows.data(), m_sizes_.first, 0);
    boost::mpi::gatherv(m_world_, m_intermediate_data_.m_elementsSum, intermediate_component.m_elementsSum.data(),
                        m_sizes_.second, 0);
    for (size_t i = 0; i < intermediate_component.m_rows.size(); ++i) {
      if (intermediate_component.m_values[i] != 0.0) {
        component.m_values[i] = intermediate_component.m_values[i];
        component.m_rows[i] = intermediate_component.m_rows[i];
      }
    }
    std::ranges::for_each(intermediate_component.m_elementsSum, [&](auto &element) {
      if (element != 0) {
        component.m_elementsSum.emplace_back(element - 1);
      }
    });
    std::erase_if(component.m_values, [&](auto &value) { return value == 0.0; });
    std::erase_if(component.m_rows, [&](auto &row) { return row == 0; });
    std::ranges::for_each(component.m_rows, [&](auto &row) { row--; });
    for (size_t i = 1; i < component.m_elementsSum.size(); ++i) {
      component.m_elementsSum[i] = component.m_elementsSum[i] + component.m_elementsSum[i - 1];
    }
    m_answerMatrix_ = SparseMatrix(m_sMatrix_.GetColumnsCount(), m_sMatrix_.GetColumnsCount(), component);
  } else {
    boost::mpi::gatherv(m_world_, m_intermediate_data_.m_values, 0);
    boost::mpi::gatherv(m_world_, m_intermediate_data_.m_rows, 0);
    boost::mpi::gatherv(m_world_, m_intermediate_data_.m_elementsSum, 0);
  }
  return true;
}

bool sadikov_i_sparse_matrix_multiplication_task_all::CCSMatrixALL::PostProcessingImpl() {
  if (m_world_.rank() == 0) {
    auto answer = FromSparseMatrix(m_answerMatrix_);
    for (size_t i = 0; i < answer.size(); ++i) {
      reinterpret_cast<double *>(task_data->outputs[0])[i] = answer[i];
    }
  }
  return true;
}
