#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
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

#ifndef _WIN32
TEST(makhov_m_linear_image_filtering_vertical_seq, test_opencv_image_validation) {
  cv::Mat input_image = cv::imread(
      ppc::util::GetAbsolutePath("seq/makhov_m_linear_image_filtering_vertical/data/10x10_orig.jpg"), cv::IMREAD_COLOR);
  ASSERT_FALSE(input_image.empty());
}

TEST(makhov_m_linear_image_filtering_vertical_seq, test_opencv_10x10_image) {
  int width = 10;
  int height = 10;

  cv::Mat input_image = cv::imread(
      ppc::util::GetAbsolutePath("seq/makhov_m_linear_image_filtering_vertical/data/10x10_orig.jpg"), cv::IMREAD_COLOR);

  cv::Mat reference_image =
      cv::imread(ppc::util::GetAbsolutePath("seq/makhov_m_linear_image_filtering_vertical/data/10x10_redacted.jpg"),
                 cv::IMREAD_COLOR);

  cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
  cv::cvtColor(reference_image, reference_image, cv::COLOR_BGR2RGB);

  std::vector<uint8_t> input_vector;
  std::vector<uint8_t> output_vector;
  std::vector<uint8_t> expected_output_vector;

  if (input_image.type() == CV_8UC3) {
    input_vector.assign(input_image.data, input_image.data + (input_image.total() * 3));
    output_vector.assign(input_image.data, input_image.data + (input_image.total() * 3));
  } else {
    std::cerr << "Format error" << '\n';
  }

  if (reference_image.type() == CV_8UC3) {
    expected_output_vector.assign(reference_image.data, reference_image.data + (reference_image.total() * 3));
  } else {
    std::cerr << "Format error" << '\n';
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
  task_data_seq->outputs_count.emplace_back(output_vector.size());

  makhov_m_linear_image_filtering_vertical_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  cv::Mat result_image(input_image.rows, input_image.cols, CV_8UC3, output_vector.data());
  double mse = cv::norm(result_image, reference_image, cv::NORM_L2) / (result_image.rows * result_image.cols);
  double psnr = 10.0 * log10((255.0 * 255.0) / mse);
  EXPECT_GT(psnr, 40.0);
}

#endif

TEST(makhov_m_linear_image_filtering_vertical_seq, test_synthetic_image_3x3) {
  int width = 3;
  int height = 3;
  std::vector<uint8_t> input_image = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0,  0,
                                      0,   100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
  std::vector<uint8_t> output_image(width * height * 3);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input_image.data());
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(output_image.data());
  task_data_seq->outputs_count.emplace_back(output_image.size());

  makhov_m_linear_image_filtering_vertical_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(int(output_image[12]), 75);
  EXPECT_EQ(int(output_image[13]), 75);
  EXPECT_EQ(int(output_image[14]), 75);
}

TEST(makhov_m_linear_image_filtering_vertical_seq, test_synthetic_image_10x10) {
  int width = 10;
  int height = 10;
  std::vector<uint8_t> input_image(width * height * 3, 0);
  std::vector<uint8_t> output_image(width * height * 3, 0);

  // White square 5x5 in middle
  for (std::size_t y = 3; y < 8; ++y) {
    for (std::size_t x = 3; x < 8; ++x) {
      std::size_t base_index = (y * width + x) * 3;

      input_image[base_index] = 255;
      input_image[base_index + 1] = 255;
      input_image[base_index + 2] = 255;
    }
  }

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

  makhov_m_linear_image_filtering_vertical_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (int i = 0; i < width * height * 3; ++i) {
    EXPECT_NEAR(output_image[i], reference_image[i], 50);
  }
}

TEST(makhov_m_linear_image_filtering_vertical_seq, test_random_image_10x10) {
  int width = 10;
  int height = 10;
  std::vector<uint8_t> input_image = GenerateRandomImage(height, width);
  std::vector<uint8_t> output_image(width * height * 3, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input_image.data());
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(output_image.data());
  task_data_seq->outputs_count.emplace_back(output_image.size());

  makhov_m_linear_image_filtering_vertical_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (std::size_t i = 0; i < output_image.size(); i++) {
    EXPECT_GE(output_image[i], 0);
    EXPECT_LE(output_image[i], 255);
  }
}

TEST(makhov_m_linear_image_filtering_vertical_seq, test_random_image_100x100) {
  int width = 100;
  int height = 100;
  std::vector<uint8_t> input_image = GenerateRandomImage(height, width);
  std::vector<uint8_t> output_image(width * height * 3, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input_image.data());
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(output_image.data());
  task_data_seq->outputs_count.emplace_back(output_image.size());

  makhov_m_linear_image_filtering_vertical_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (std::size_t i = 0; i < output_image.size(); i++) {
    EXPECT_GE(output_image[i], 0);
    EXPECT_LE(output_image[i], 255);
  }
}

TEST(makhov_m_linear_image_filtering_vertical_seq, test_random_image_500x500) {
  int width = 500;
  int height = 500;
  std::vector<uint8_t> input_image = GenerateRandomImage(height, width);
  std::vector<uint8_t> output_image(width * height * 3, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input_image.data());
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(output_image.data());
  task_data_seq->outputs_count.emplace_back(output_image.size());

  makhov_m_linear_image_filtering_vertical_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (std::size_t i = 0; i < output_image.size(); i++) {
    EXPECT_GE(output_image[i], 0);
    EXPECT_LE(output_image[i], 255);
  }
}

TEST(makhov_m_linear_image_filtering_vertical_seq, validation_test1) {
  int width = 3;
  int height = 1;
  std::vector<uint8_t> input_image = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> output_image(width * height * 3);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input_image.data());
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(output_image.data());
  task_data_seq->outputs_count.emplace_back(output_image.size());

  makhov_m_linear_image_filtering_vertical_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(makhov_m_linear_image_filtering_vertical_seq, validation_test2) {
  int width = 3;
  int height = 1;
  std::vector<uint8_t> input_image = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> output_image(2);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(input_image.data());
  task_data_seq->inputs_count.push_back(width);
  task_data_seq->inputs_count.push_back(height);
  task_data_seq->outputs.emplace_back(output_image.data());
  task_data_seq->outputs_count.emplace_back(output_image.size());

  makhov_m_linear_image_filtering_vertical_seq::TaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}