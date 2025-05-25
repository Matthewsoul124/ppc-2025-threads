// Golovkin Maksim
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/golovkin_contrast_stretching/include/ops_stl.hpp"

TEST(golovkin_contrast_stretching_stl, test_pipeline_run) {
  constexpr size_t kCount = 1'000'000;

  std::vector<uint8_t> in(kCount);
  std::vector<uint8_t> out(kCount, 0);

  for (size_t i = 0; i < kCount; ++i) {
    in[i] = static_cast<uint8_t>(i % 256);
  }

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data_stl);

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(out.size(), in.size());
  EXPECT_GE(out[100], out[50]);
}

TEST(golovkin_contrast_stretching_stl, test_task_run) {
  constexpr size_t kCount = 1'000'000;

  std::vector<uint8_t> in(kCount);
  std::vector<uint8_t> out(kCount, 0);

  for (size_t i = 0; i < kCount; ++i) {
    in[i] = static_cast<uint8_t>(i % 256);
  }

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  golovkin_contrast_stretching::ContrastStretchingSTL task(task_data_stl);

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(out.size(), in.size());
  EXPECT_GE(out[100], out[50]);
}