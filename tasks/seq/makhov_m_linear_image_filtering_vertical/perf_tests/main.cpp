#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/makhov_m_linear_image_filtering_vertical/include/ops_seq.hpp"

namespace {
std::vector<uint8_t> GenerateRandomImage(int height, int width) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  int size = height * width * 3;
  std::vector<uint8_t> image(size);

  for (int i = 0; i < size; ++i) {
    image[i] = dis(gen);
  }

  return image;
}
}  // namespace

TEST(makhov_m_linear_image_filtering_vertical_seq, test_pipeline_run) {
  int width = 1000;
  int height = 1000;
  std::vector<uint8_t> input_image = GenerateRandomImage(height, width);
  std::vector<uint8_t> output_image(width * height * 3, 0);
  std::vector<uint8_t> reference_image(input_image);
  // Gauss blur imitation for RGB
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int sum_r = 0;
      int sum_g = 0;
      int sum_b = 0;

      // Проход по окрестности 3x3
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          // Индекс пикселя с учетом RGB
          std::size_t idx = ((y + ky) * width + (x + kx)) * 3;

          sum_r += reference_image[idx];      // Красный канал
          sum_g += reference_image[idx + 1];  // Зеленый канал
          sum_b += reference_image[idx + 2];  // Синий канал
        }
      }

      // Индекс для записи результата
      std::size_t ref_idx = (y * width + x) * 3;

      // Усреднение и запись
      reference_image[ref_idx] = static_cast<uint8_t>(sum_r / 9);
      reference_image[ref_idx + 1] = static_cast<uint8_t>(sum_g / 9);
      reference_image[ref_idx + 2] = static_cast<uint8_t>(sum_b / 9);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input_image.data());
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(output_image.data());
  task_data_seq->outputs_count.emplace_back(output_image.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<makhov_m_linear_image_filtering_vertical_seq::TaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < output_image.size(); ++i) {
    EXPECT_GE(output_image[i], 0);
    EXPECT_LE(output_image[i], 255);
  }
}

TEST(makhov_m_linear_image_filtering_vertical_seq, test_task_run) {
  int width = 1000;
  int height = 1000;
  std::vector<uint8_t> input_image = GenerateRandomImage(height, width);
  std::vector<uint8_t> output_image(width * height * 3, 0);
  std::vector<uint8_t> reference_image(input_image);
  // Gauss blur imitation for RGB
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int sum_r = 0;
      int sum_g = 0;
      int sum_b = 0;

      // Проход по окрестности 3x3
      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          // Индекс пикселя с учетом RGB
          std::size_t idx = ((y + ky) * width + (x + kx)) * 3;

          sum_r += reference_image[idx];      // Красный канал
          sum_g += reference_image[idx + 1];  // Зеленый канал
          sum_b += reference_image[idx + 2];  // Синий канал
        }
      }

      // Индекс для записи результата
      std::size_t ref_idx = (y * width + x) * 3;

      // Усреднение и запись
      reference_image[ref_idx] = static_cast<uint8_t>(sum_r / 9);
      reference_image[ref_idx + 1] = static_cast<uint8_t>(sum_g / 9);
      reference_image[ref_idx + 2] = static_cast<uint8_t>(sum_b / 9);
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input_image.data());
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(output_image.data());
  task_data_seq->outputs_count.emplace_back(output_image.size());
  // Create Task
  auto test_task_sequential =
      std::make_shared<makhov_m_linear_image_filtering_vertical_seq::TaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (size_t i = 0; i < output_image.size(); ++i) {
    EXPECT_GE(output_image[i], 0);
    EXPECT_LE(output_image[i], 255);
  }
}
