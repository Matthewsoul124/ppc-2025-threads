#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/yasakova_t_sparse_matrix_multiplication/include/ops_stl.hpp"

TEST(yasakova_t_sparse_matrix_multiplication_stl, TestRealNumbersMultiplication) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(3, 3);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(1, 0), 0);
  left_matrix.InsertElement(0, std::complex<double>(2, 0), 2);
  left_matrix.InsertElement(1, std::complex<double>(3, 0), 1);
  left_matrix.InsertElement(2, std::complex<double>(4, 0), 0);
  left_matrix.InsertElement(2, std::complex<double>(5, 0), 1);

  right_matrix.InsertElement(0, std::complex<double>(6, 0), 1);
  right_matrix.InsertElement(1, std::complex<double>(7, 0), 0);
  right_matrix.InsertElement(2, std::complex<double>(8, 0), 2);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  expected_result.InsertElement(0, std::complex<double>(6, 0), 1);
  expected_result.InsertElement(0, std::complex<double>(16, 0), 2);
  expected_result.InsertElement(1, std::complex<double>(21, 0), 0);
  expected_result.InsertElement(2, std::complex<double>(24, 0), 1);
  expected_result.InsertElement(2, std::complex<double>(35, 0), 0);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), true);
  multiplication_task.PreProcessing();
  multiplication_task.Run();
  multiplication_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, TestIncompatibleDimensions) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(5, 3);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(1, 0), 0);
  left_matrix.InsertElement(0, std::complex<double>(2, 0), 2);
  left_matrix.InsertElement(1, std::complex<double>(3, 0), 1);
  left_matrix.InsertElement(2, std::complex<double>(4, 0), 0);
  left_matrix.InsertElement(2, std::complex<double>(5, 0), 1);

  right_matrix.InsertElement(0, std::complex<double>(6, 0), 1);
  right_matrix.InsertElement(1, std::complex<double>(7, 0), 0);
  right_matrix.InsertElement(2, std::complex<double>(8, 0), 2);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), false);
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, TestComplexNumbersMultiplication) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(3, 3);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(1, 1), 0);
  left_matrix.InsertElement(0, std::complex<double>(2, 2), 2);
  left_matrix.InsertElement(1, std::complex<double>(3, 3), 1);
  left_matrix.InsertElement(2, std::complex<double>(4, 4), 0);
  left_matrix.InsertElement(2, std::complex<double>(5, 5), 1);

  right_matrix.InsertElement(0, std::complex<double>(6, 6), 1);
  right_matrix.InsertElement(1, std::complex<double>(7, 7), 0);
  right_matrix.InsertElement(2, std::complex<double>(8, 8), 2);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  expected_result.InsertElement(0, std::complex<double>(0, 12), 1);
  expected_result.InsertElement(0, std::complex<double>(0, 32), 2);
  expected_result.InsertElement(1, std::complex<double>(0, 42), 0);
  expected_result.InsertElement(2, std::complex<double>(0, 48), 1);
  expected_result.InsertElement(2, std::complex<double>(0, 70), 0);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), true);
  multiplication_task.PreProcessing();
  multiplication_task.Run();
  multiplication_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, test_matmul_rectangular_matrix) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(2, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(3, 4);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(2, 4);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(1, 0), 1);
  left_matrix.InsertElement(0, std::complex<double>(2, 0), 2);
  left_matrix.InsertElement(1, std::complex<double>(3, 0), 1);

  right_matrix.InsertElement(0, std::complex<double>(3, 0), 2);
  right_matrix.InsertElement(1, std::complex<double>(5, 0), 0);
  right_matrix.InsertElement(1, std::complex<double>(4, 0), 3);
  right_matrix.InsertElement(2, std::complex<double>(7, 0), 0);
  right_matrix.InsertElement(2, std::complex<double>(8, 0), 1);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  expected_result.InsertElement(0, std::complex<double>(19, 0), 0);
  expected_result.InsertElement(0, std::complex<double>(4, 0), 3);
  expected_result.InsertElement(0, std::complex<double>(16, 0), 1);
  expected_result.InsertElement(1, std::complex<double>(15, 0), 0);
  expected_result.InsertElement(1, std::complex<double>(12, 0), 3);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), true);
  multiplication_task.PreProcessing();
  multiplication_task.Run();
  multiplication_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, test_matmul_with_negative_elems) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(2, 2);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(-1, -1), 0);
  left_matrix.InsertElement(1, std::complex<double>(3, 3), 1);

  right_matrix.InsertElement(0, std::complex<double>(6, 6), 1);
  right_matrix.InsertElement(1, std::complex<double>(-7, -7), 0);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  expected_result.InsertElement(0, std::complex<double>(0, -12), 1);
  expected_result.InsertElement(1, std::complex<double>(0, -42), 0);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), true);
  multiplication_task.PreProcessing();
  multiplication_task.Run();
  multiplication_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, test_matmul_with_double_elems) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(2, 2);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(2, 2);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(-1.7, -1.5), 0);
  left_matrix.InsertElement(1, std::complex<double>(3.7, 3.1), 1);

  right_matrix.InsertElement(0, std::complex<double>(6.3, 6.1), 1);
  right_matrix.InsertElement(1, std::complex<double>(-7.4, -7.7), 0);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  expected_result.InsertElement(0, std::complex<double>(-1.56, -19.82), 1);
  expected_result.InsertElement(1, std::complex<double>(-3.51, -51.43), 0);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), true);
  multiplication_task.PreProcessing();
  multiplication_task.Run();
  multiplication_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, test_matmul_row_by_col) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(1, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(3, 1);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(1, 1);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(-1, 0), 0);
  left_matrix.InsertElement(0, std::complex<double>(-2, 0), 1);
  left_matrix.InsertElement(0, std::complex<double>(-3, 0), 2);

  right_matrix.InsertElement(0, std::complex<double>(1, 0), 0);
  right_matrix.InsertElement(1, std::complex<double>(2, 0), 0);
  right_matrix.InsertElement(2, std::complex<double>(3, 0), 0);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  expected_result.InsertElement(0, std::complex<double>(-14, 0), 0);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), true);
  multiplication_task.PreProcessing();
  multiplication_task.Run();
  multiplication_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, test_matmul_diag_matrix) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(3, 3);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(-1, 0), 0);
  left_matrix.InsertElement(1, std::complex<double>(-2, 0), 1);
  left_matrix.InsertElement(2, std::complex<double>(-3, 0), 2);

  right_matrix.InsertElement(0, std::complex<double>(1, 0), 0);
  right_matrix.InsertElement(1, std::complex<double>(2, 0), 1);
  right_matrix.InsertElement(2, std::complex<double>(3, 0), 2);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  expected_result.InsertElement(0, std::complex<double>(-1, 0), 0);
  expected_result.InsertElement(1, std::complex<double>(-4, 0), 1);
  expected_result.InsertElement(2, std::complex<double>(-9, 0), 2);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), true);
  multiplication_task.PreProcessing();
  multiplication_task.Run();
  multiplication_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, test_matmul_only_imag) {
  // Create data
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(3, 3);
  std::vector<std::complex<double>> input_data = {};
  std::vector<std::complex<double>> left_matrix_data;
  std::vector<std::complex<double>> right_matrix_data;
  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  left_matrix.InsertElement(0, std::complex<double>(0, 1), 0);
  left_matrix.InsertElement(0, std::complex<double>(0, 2), 2);
  left_matrix.InsertElement(1, std::complex<double>(0, 3), 1);
  left_matrix.InsertElement(2, std::complex<double>(0, 4), 0);
  left_matrix.InsertElement(2, std::complex<double>(0, 5), 1);

  right_matrix.InsertElement(0, std::complex<double>(0, 6), 1);
  right_matrix.InsertElement(1, std::complex<double>(0, 7), 0);
  right_matrix.InsertElement(2, std::complex<double>(0, 8), 2);
  left_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  right_matrix_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  input_data.reserve(left_matrix_data.size() + right_matrix_data.size());
  for (unsigned int i = 0; i < left_matrix_data.size(); i++) {
    input_data.emplace_back(left_matrix_data[i]);
  }
  for (unsigned int i = 0; i < right_matrix_data.size(); i++) {
    input_data.emplace_back(right_matrix_data[i]);
  }
  expected_result.InsertElement(0, std::complex<double>(-6, 0), 1);
  expected_result.InsertElement(0, std::complex<double>(-16, 0), 2);
  expected_result.InsertElement(1, std::complex<double>(-21, 0), 0);
  expected_result.InsertElement(2, std::complex<double>(-24, 0), 1);
  expected_result.InsertElement(2, std::complex<double>(-35, 0), 0);
  // Create task_data
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data_stl->outputs_count.emplace_back(output_buffer.size());

  // Create Task
  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask multiplication_task(task_data_stl);
  ASSERT_EQ(multiplication_task.Validation(), true);
  multiplication_task.PreProcessing();
  multiplication_task.Run();
  multiplication_task.PostProcessing();
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, TestIdentityMatrixMultiplication) {
  // Test multiplication with identity matrix
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(3, 3);

  // Create identity matrix
  for (int i = 0; i < 3; ++i) {
    right_matrix.InsertElement(i, std::complex<double>(1, 0), i);
  }

  // Fill left matrix with some values
  left_matrix.InsertElement(0, std::complex<double>(1, 2), 0);
  left_matrix.InsertElement(0, std::complex<double>(3, 4), 1);
  left_matrix.InsertElement(1, std::complex<double>(5, 6), 1);
  left_matrix.InsertElement(2, std::complex<double>(7, 8), 2);

  // Expected result should be equal to left matrix
  expected_result = left_matrix;

  // Prepare input data
  auto left_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  auto right_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  std::vector<std::complex<double>> input_data;
  input_data.insert(input_data.end(), left_data.begin(), left_data.end());
  input_data.insert(input_data.end(), right_data.begin(), right_data.end());

  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data->outputs_count.emplace_back(output_buffer.size());

  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask task(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  auto actual_result = yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, TestLargeSparseMatrixMultiplication) {
  // Test with large sparse matrices
  const int size = 100;
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(size, size);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(size, size);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(size, size);

  // Fill matrices with diagonal pattern
  for (int i = 0; i < size; ++i) {
    left_matrix.InsertElement(i, std::complex<double>(i + 1, 0), i);
    right_matrix.InsertElement(i, std::complex<double>(size - i, 0), i);
    expected_result.InsertElement(i, std::complex<double>((i + 1) * (size - i), 0), i);
  }

  // Prepare input data
  auto left_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  auto right_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  std::vector<std::complex<double>> input_data;
  input_data.insert(input_data.end(), left_data.begin(), left_data.end());
  input_data.insert(input_data.end(), right_data.begin(), right_data.end());

  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data->outputs_count.emplace_back(output_buffer.size());

  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask task(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  auto actual_result = yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, TestEmptyMatricesMultiplication) {
  // Test multiplication of empty matrices
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage left_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage right_matrix(3, 3);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage expected_result(3, 3);

  // No elements inserted, matrices are empty

  // Prepare input data
  auto left_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(left_matrix);
  auto right_data = yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(right_matrix);
  std::vector<std::complex<double>> input_data;
  input_data.insert(input_data.end(), left_data.begin(), left_data.end());
  input_data.insert(input_data.end(), right_data.begin(), right_data.end());

  std::vector<std::complex<double>> output_buffer(left_matrix.columnCount * right_matrix.rowCount * 100, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->inputs_count.emplace_back(input_data.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_buffer.data()));
  task_data->outputs_count.emplace_back(output_buffer.size());

  yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask task(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  auto actual_result = yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_buffer);
  ASSERT_TRUE(yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(actual_result, expected_result));
}
