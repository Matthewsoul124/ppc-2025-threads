#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/khovansky_d_double_radix_batcher/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(khovansky_d_double_radix_batcher_all, invalid_input) {
  boost::mpi::communicator world;
  std::vector<double> in{1.0};
  std::vector<double> out(1);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  khovansky_d_double_radix_batcher_all::RadixAll test_task_all(task_data_all);
  if (world.rank() == 0) {
    EXPECT_EQ(test_task_all.ValidationImpl(), false);
  }
}

TEST(khovansky_d_double_radix_batcher_all, negative_values) {
  boost::mpi::communicator world;
  std::vector<double> in{-3.14, -1.0, -100.5, -0.1, -999.99};
  std::vector<double> exp_out{-999.99, -100.5, -3.14, -1.0, -0.1};
  std::vector<double> out(5);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  khovansky_d_double_radix_batcher_all::RadixAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.ValidationImpl(), true);
  test_task_all.PreProcessingImpl();
  test_task_all.RunImpl();
  test_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(khovansky_d_double_radix_batcher_all, positive_values) {
  std::vector<double> in{3.14, 1.0, 100.5, 0.1, 999.99};
  std::vector<double> exp_out{0.1, 1.0, 3.14, 100.5, 999.99};
  std::vector<double> out(5);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  khovansky_d_double_radix_batcher_all::RadixAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.ValidationImpl(), true);
  test_task_all.PreProcessingImpl();
  test_task_all.RunImpl();
  test_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(khovansky_d_double_radix_batcher_all, mixed_values) {
  std::vector<double> in{0.0, -2.5, 3.3, -1.1, 2.2};
  std::vector<double> exp_out{-2.5, -1.1, 0.0, 2.2, 3.3};
  std::vector<double> out(5);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  khovansky_d_double_radix_batcher_all::RadixAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.ValidationImpl(), true);
  test_task_all.PreProcessingImpl();
  test_task_all.RunImpl();
  test_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(khovansky_d_double_radix_batcher_all, duplicate_values) {
  std::vector<double> in{5.5, 2.2, 5.5, 3.3, 2.2};
  std::vector<double> exp_out{2.2, 2.2, 3.3, 5.5, 5.5};
  std::vector<double> out(5);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  khovansky_d_double_radix_batcher_all::RadixAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.ValidationImpl(), true);
  test_task_all.PreProcessingImpl();
  test_task_all.RunImpl();
  test_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(khovansky_d_double_radix_batcher_all, sorted_input) {
  std::vector<double> in{-2.2, -1.1, 0.0, 1.1, 2.2};
  std::vector<double> exp_out{-2.2, -1.1, 0.0, 1.1, 2.2};
  std::vector<double> out(5);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  khovansky_d_double_radix_batcher_all::RadixAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.ValidationImpl(), true);
  test_task_all.PreProcessingImpl();
  test_task_all.RunImpl();
  test_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(khovansky_d_double_radix_batcher_all, large_numbers) {
  std::vector<double> in{1e308, -1e308, 1e307, -1e307, 0.0};
  std::vector<double> exp_out{-1e308, -1e307, 0.0, 1e307, 1e308};
  std::vector<double> out(5);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  khovansky_d_double_radix_batcher_all::RadixAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.ValidationImpl(), true);
  test_task_all.PreProcessingImpl();
  test_task_all.RunImpl();
  test_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(khovansky_d_double_radix_batcher_all, large_array) {
  constexpr size_t kSize = 1000000;
  std::vector<double> in(kSize);
  std::vector<double> exp_out(kSize);

  for (size_t i = 0; i < kSize; ++i) {
    in[i] = static_cast<double>(kSize - i);
    exp_out[i] = static_cast<double>(i + 1);
  }

  std::vector<double> out(kSize);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  khovansky_d_double_radix_batcher_all::RadixAll test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.ValidationImpl(), true);
  test_task_all.PreProcessingImpl();
  test_task_all.RunImpl();
  test_task_all.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}