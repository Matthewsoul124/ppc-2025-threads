#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/tsatsyn_a_radix_sort_simple_merge/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> GetRandomVector(int sz, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(a, b);
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dis(gen);
  }
  return vec;
}
}  // namespace

TEST(tsatsyn_a_radix_sort_simple_merge_all, negative_validation) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize / 2, 0);
  in = GetRandomVector(arrsize, -100, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, negative_double_10) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, negative_double_100) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, negative_double_1000) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, negative_double_10000) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}

TEST(tsatsyn_a_radix_sort_simple_merge_all, pozitive_double_10) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, pozitive_double_100) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, pozitive_double_1000) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, pozitive_double_10000) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}

TEST(tsatsyn_a_radix_sort_simple_merge_all, mix_double_10) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, mix_double_100) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, mix_double_1000) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, mix_double_10000) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = GetRandomVector(arrsize, -100, 100);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}
TEST(tsatsyn_a_radix_sort_simple_merge_all, pozitive_double_reverse_10) {
  // Create data
  boost::mpi::communicator world;
  int arrsize = 10;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);
  in = {5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    std::ranges::sort(in);
    EXPECT_EQ(in, out);
  }
}