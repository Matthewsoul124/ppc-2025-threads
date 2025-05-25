#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/khovansky_d_double_radix_batcher/include/ops_stl.hpp"

TEST(khovansky_d_double_radix_batcher_stl, invalid_input) {
  std::vector<double> in{1.0};
  std::vector<double> out(1);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  khovansky_d_double_radix_batcher_stl::RadixSTL test_task_stl(task_data_stl);
  EXPECT_EQ(test_task_stl.ValidationImpl(), false);
}

TEST(khovansky_d_double_radix_batcher_stl, negative_values) {
  std::vector<double> in{-3.14, -1.0, -100.5, -0.1, -999.99};
  std::vector<double> exp_out{-999.99, -100.5, -3.14, -1.0, -0.1};
  std::vector<double> out(5);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  khovansky_d_double_radix_batcher_stl::RadixSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.ValidationImpl(), true);
  test_task_stl.PreProcessingImpl();
  test_task_stl.RunImpl();
  test_task_stl.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(khovansky_d_double_radix_batcher_stl, positive_values) {
  std::vector<double> in{3.14, 1.0, 100.5, 0.1, 999.99};
  std::vector<double> exp_out{0.1, 1.0, 3.14, 100.5, 999.99};
  std::vector<double> out(5);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  khovansky_d_double_radix_batcher_stl::RadixSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.ValidationImpl(), true);
  test_task_stl.PreProcessingImpl();
  test_task_stl.RunImpl();
  test_task_stl.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(khovansky_d_double_radix_batcher_stl, mixed_values) {
  std::vector<double> in{0.0, -2.5, 3.3, -1.1, 2.2};
  std::vector<double> exp_out{-2.5, -1.1, 0.0, 2.2, 3.3};
  std::vector<double> out(5);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  khovansky_d_double_radix_batcher_stl::RadixSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.ValidationImpl(), true);
  test_task_stl.PreProcessingImpl();
  test_task_stl.RunImpl();
  test_task_stl.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(khovansky_d_double_radix_batcher_stl, duplicate_values) {
  std::vector<double> in{5.5, 2.2, 5.5, 3.3, 2.2};
  std::vector<double> exp_out{2.2, 2.2, 3.3, 5.5, 5.5};
  std::vector<double> out(5);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  khovansky_d_double_radix_batcher_stl::RadixSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.ValidationImpl(), true);
  test_task_stl.PreProcessingImpl();
  test_task_stl.RunImpl();
  test_task_stl.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(khovansky_d_double_radix_batcher_stl, sorted_input) {
  std::vector<double> in{-2.2, -1.1, 0.0, 1.1, 2.2};
  std::vector<double> exp_out{-2.2, -1.1, 0.0, 1.1, 2.2};
  std::vector<double> out(5);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  khovansky_d_double_radix_batcher_stl::RadixSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.ValidationImpl(), true);
  test_task_stl.PreProcessingImpl();
  test_task_stl.RunImpl();
  test_task_stl.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(khovansky_d_double_radix_batcher_stl, large_numbers) {
  std::vector<double> in{1e308, -1e308, 1e307, -1e307, 0.0};
  std::vector<double> exp_out{-1e308, -1e307, 0.0, 1e307, 1e308};
  std::vector<double> out(5);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  khovansky_d_double_radix_batcher_stl::RadixSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.ValidationImpl(), true);
  test_task_stl.PreProcessingImpl();
  test_task_stl.RunImpl();
  test_task_stl.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(khovansky_d_double_radix_batcher_stl, large_array) {
  constexpr size_t kSize = 1000000;
  std::vector<double> in(kSize);
  std::vector<double> exp_out(kSize);

  for (size_t i = 0; i < kSize; ++i) {
    in[i] = static_cast<double>(kSize - i);
    exp_out[i] = static_cast<double>(i + 1);
  }

  std::vector<double> out(kSize);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  khovansky_d_double_radix_batcher_stl::RadixSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.ValidationImpl(), true);
  test_task_stl.PreProcessingImpl();
  test_task_stl.RunImpl();
  test_task_stl.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}