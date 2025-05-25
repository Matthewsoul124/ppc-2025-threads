// Golovkin Maksim
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/golovkin_contrast_stretching/include/ops_stl.hpp"

TEST(golovkin_contrast_stretching_stl, test_contrast_basic) {
  constexpr size_t kSize = 8;
  std::vector<uint8_t> in = {30, 60, 90, 120, 150, 180, 210, 240};
  std::vector<uint8_t> out(kSize, 0);
  std::vector<uint8_t> expected = {0, 36, 72, 109, 145, 182, 218, 255};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_contrast_flat_image) {
  constexpr size_t kSize = 10;
  std::vector<uint8_t> in(kSize, 100);
  std::vector<uint8_t> out(kSize, 0);
  std::vector<uint8_t> expected(kSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_all_maximum) {
  constexpr size_t kSize = 16;
  std::vector<uint8_t> in(kSize, 255);
  std::vector<uint8_t> out(kSize, 0);
  std::vector<uint8_t> expected(kSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_gradient_image) {
  constexpr size_t kSize = 256;
  std::vector<uint8_t> in(kSize);
  std::vector<uint8_t> out(kSize, 0);

  for (size_t i = 0; i < kSize; ++i) {
    in[i] = static_cast<uint8_t>(i);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, in);
}

TEST(golovkin_contrast_stretching_stl, test_small_range) {
  std::vector<uint8_t> in = {100, 101, 102, 103, 104, 105};
  std::vector<uint8_t> out(in.size(), 0);
  std::vector<uint8_t> expected = {0, 51, 102, 153, 204, 255};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_extreme_values_only) {
  std::vector<uint8_t> in = {0, 255, 0, 255, 0, 255};
  std::vector<uint8_t> out(in.size(), 0);
  std::vector<uint8_t> expected = {0, 255, 0, 255, 0, 255};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_min_max_near_extremes) {
  std::vector<uint8_t> in = {1, 254, 1, 254};
  std::vector<uint8_t> out(in.size(), 0);
  std::vector<uint8_t> expected = {0, 255, 0, 255};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_alternating_values) {
  std::vector<uint8_t> in = {10, 20, 10, 20, 10, 20};
  std::vector<uint8_t> out(in.size(), 0);
  std::vector<uint8_t> expected = {0, 255, 0, 255, 0, 255};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_all_zeros) {
  std::vector<uint8_t> in(32, 0);
  std::vector<uint8_t> out(in.size(), 123);
  std::vector<uint8_t> expected(32, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_random_mid_range_values) {
  std::vector<uint8_t> in = {50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
  std::vector<uint8_t> out(in.size(), 0);

  std::vector<uint8_t> expected;
  expected.reserve(in.size());
  for (auto val : in) {
    expected.push_back(static_cast<uint8_t>((val - 50) * 255 / (150 - 50)));
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}

TEST(golovkin_contrast_stretching_stl, test_empty_input) {
  std::vector<uint8_t> in;
  std::vector<uint8_t> out;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_TRUE(out.empty());
}

TEST(golovkin_contrast_stretching_stl, test_uint16_pixels) {
  std::vector<uint16_t> in = {1000, 2000, 3000, 4000, 5000};
  std::vector<uint16_t> out(in.size(), 0);
  std::vector<uint16_t> expected = {0, 63, 127, 191, 255};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size() * sizeof(uint16_t));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size() * sizeof(uint16_t));

  golovkin_contrast_stretching::ContrastStretchingSTL<uint16_t> task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(out, expected);
}