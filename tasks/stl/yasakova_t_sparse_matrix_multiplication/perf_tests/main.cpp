#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/yasakova_t_sparse_matrix_multiplication/include/ops_stl.hpp"

TEST(yasakova_t_sparse_matrix_multiplication_stl, test_pipeline_run) {
  const int matrix_size = 1000;
  const int non_zero_elements = 10000;
  const int num_runs = 100;
  std::srand(std::time(nullptr));

  // Создаем разреженные матрицы с заданным числом ненулевых элементов
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage first_matrix(matrix_size, matrix_size);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage second_matrix(matrix_size, matrix_size);

  // Заполняем first_matrix случайными элементами
  for (int i = 0; i < non_zero_elements; ++i) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    std::complex<double> value(-50 + (rand() % 100), -50 + (rand() % 100));
    first_matrix.InsertElement(row, value, col);
  }

  // Заполняем second_matrix случайными элементами
  for (int i = 0; i < non_zero_elements; ++i) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    std::complex<double> value(-50 + (rand() % 100), -50 + (rand() % 100));
    second_matrix.InsertElement(row, value, col);
  }

  // Конвертируем матрицы в векторы для передачи в задачу
  std::vector<std::complex<double>> first_matrix_data =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(first_matrix);
  std::vector<std::complex<double>> second_matrix_data =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(second_matrix);

  // Подготавливаем входные данные (конкатенация двух векторов)
  std::vector<std::complex<double>> input_data;
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  input_data.insert(input_data.end(), first_matrix_data.begin(), first_matrix_data.end());
  input_data.insert(input_data.end(), second_matrix_data.begin(), second_matrix_data.end());

  // Подготавливаем вектор для результата
  std::vector<std::complex<double>> output_data(matrix_size * matrix_size, std::complex<double>(0, 0));

  // Создаем данные задачи
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_stl->outputs_count.emplace_back(output_data.size());

  // Создаем задачу
  auto test_task_stl =
      std::make_shared<yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask>(task_data_stl);

  // Создаем атрибуты для измерения производительности
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = num_runs;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Создаем объект для сбора результатов производительности
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Запускаем анализ производительности
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Конвертируем результат обратно в матрицу
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_data);

  // Проверяем размеры результата
  ASSERT_EQ(actual_result.rowCount, matrix_size);
  ASSERT_EQ(actual_result.columnCount, matrix_size);

  // Проверяем, что результат не нулевой
  bool is_result_non_zero = false;
  for (const auto &elem : output_data) {
    if (elem != std::complex<double>(0, 0)) {
      is_result_non_zero = true;
      break;
    }
  }
  ASSERT_TRUE(is_result_non_zero);
}

TEST(yasakova_t_sparse_matrix_multiplication_stl, test_task_run) {
  const int matrix_size = 1000;
  const int non_zero_elements = 10000;
  const int num_runs = 100;
  std::srand(std::time(nullptr));
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage first_matrix(matrix_size, matrix_size);
  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage second_matrix(matrix_size, matrix_size);

  for (int i = 0; i < non_zero_elements; ++i) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    std::complex<double> value(-50 + (rand() % 100), -50 + (rand() % 100));
    first_matrix.InsertElement(row, value, col);
  }

  for (int i = 0; i < non_zero_elements; ++i) {
    int row = rand() % matrix_size;
    int col = rand() % matrix_size;
    std::complex<double> value(-50 + (rand() % 100), -50 + (rand() % 100));
    second_matrix.InsertElement(row, value, col);
  }

  std::vector<std::complex<double>> first_matrix_data =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(first_matrix);
  std::vector<std::complex<double>> second_matrix_data =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(second_matrix);

  std::vector<std::complex<double>> input_data;
  input_data.reserve(first_matrix_data.size() + second_matrix_data.size());
  input_data.insert(input_data.end(), first_matrix_data.begin(), first_matrix_data.end());
  input_data.insert(input_data.end(), second_matrix_data.begin(), second_matrix_data.end());

  std::vector<std::complex<double>> output_data(matrix_size * matrix_size, std::complex<double>(0, 0));

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data_stl->inputs_count.emplace_back(input_data.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data_stl->outputs_count.emplace_back(output_data.size());

  auto test_task_stl =
      std::make_shared<yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask>(task_data_stl);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = num_runs;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage actual_result =
      yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(output_data);

  ASSERT_EQ(actual_result.rowCount, matrix_size);
  ASSERT_EQ(actual_result.columnCount, matrix_size);

  bool is_result_non_zero = false;
  for (const auto &elem : output_data) {
    if (elem != std::complex<double>(0, 0)) {
      is_result_non_zero = true;
      break;
    }
  }
  ASSERT_TRUE(is_result_non_zero);
}
