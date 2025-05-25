#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/kalyakina_a_Shell_with_simple_merge/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<int> CreateReverseSortedVector(unsigned int size, const int left) {
  std::vector<int> result;
  while (size-- != 0) {
    result.push_back(left + static_cast<int>(size));
  }
  return result;
}

std::vector<int> CreateRandomVector(unsigned int size, const int left, const int right) {
  std::vector<int> result;
  std::random_device dev;
  std::mt19937 gen(dev());
  while (size-- != 0) {
    result.push_back(static_cast<int>(gen() % static_cast<int>(right - left)) + left);
  }
  return result;
}

std::vector<int> CreateSortedVector(unsigned int size, const int left) {
  std::vector<int> result;
  unsigned int i = 0;
  while (i < size) {
    result.push_back(left + static_cast<int>(i++));
  }
  return result;
}

void TestOfFunction(std::vector<int> &in) {
  boost::mpi::communicator world;

  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    out = std::vector<int>(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_shell_with_simple_merge_all::ShellSortALL task_all(task_data_all);

  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_TRUE(std::ranges::is_sorted(out.begin(), out.end()));
  }
}
}  // namespace

TEST(kalyakina_a_shell_with_simple_merge_all, test_of_Validation1) {
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{2, 9, 5, 1, 4, 8, 3};
    out.resize(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(0);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_shell_with_simple_merge_all::ShellSortALL task_all(task_data_all);

  if (world.rank() == 0) {
    ASSERT_EQ(task_all.Validation(), false);
  }
}

TEST(kalyakina_a_shell_with_simple_merge_all, test_of_Validation2) {
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{2, 9, 5, 1, 4, 8, 3};
    out.resize(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(0);
  }

  kalyakina_a_shell_with_simple_merge_all::ShellSortALL task_all(task_data_all);

  if (world.rank() == 0) {
    ASSERT_EQ(task_all.Validation(), false);
  }
}

TEST(kalyakina_a_shell_with_simple_merge_all, test_of_Validation3) {
  boost::mpi::communicator world;
  std::vector<int> in;
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<int>{2, 9, 5, 1, 4, 8, 3};
    out.resize(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size() + 1);
  }

  kalyakina_a_shell_with_simple_merge_all::ShellSortALL task_all(task_data_all);

  if (world.rank() == 0) {
    ASSERT_EQ(task_all.Validation(), false);
  }
}

TEST(kalyakina_a_shell_with_simple_merge_all, small_fixed_vector) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3, 2};
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, medium_fixed_vector) {
  std::vector<int> in = {2,  9,  5,  1,  4,  8,  3,  11, 34, 12, 6,  29,  13,   7,     56,     32,      88,
                         90, 94, 78, 54, 47, 37, 77, 22, 44, 55, 66, 123, 1234, 12345, 123456, 1234567, 0};
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, reverse_sorted_vector_50) {
  std::vector<int> in = CreateReverseSortedVector(50, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, reverse_sorted_vector_100) {
  std::vector<int> in = CreateReverseSortedVector(100, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, reverse_sorted_vector_1000) {
  std::vector<int> in = CreateReverseSortedVector(1000, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, reverse_sorted_vector_10000) {
  std::vector<int> in = CreateReverseSortedVector(10000, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, random_vector_50) {
  std::vector<int> in = CreateRandomVector(50, -7000, 7000);
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, random_vector_100) {
  std::vector<int> in = CreateRandomVector(100, -7000, 7000);
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, random_vector_1000) {
  std::vector<int> in = CreateRandomVector(1000, -7000, 7000);
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, random_vector_10000) {
  std::vector<int> in = CreateRandomVector(10000, -7000, 7000);
  TestOfFunction(in);
}

TEST(kalyakina_a_shell_with_simple_merge_all, simple_merge_sort_100) {
  std::vector<int> vec1 = CreateSortedVector(100, -50);
  std::vector<int> vec2 = CreateSortedVector(100, 0);

  std::vector<int> res = kalyakina_a_shell_with_simple_merge_all::ShellSortALL::SimpleMergeSort(vec1, vec2);
  ASSERT_TRUE(std::ranges::is_sorted(res.begin(), res.end()));
}

TEST(kalyakina_a_shell_with_simple_merge_all, simple_merge_sort_1000) {
  std::vector<int> vec1 = CreateSortedVector(1000, 0);
  std::vector<int> vec2 = CreateSortedVector(1000, -500);

  std::vector<int> res = kalyakina_a_shell_with_simple_merge_all::ShellSortALL::SimpleMergeSort(vec1, vec2);
  ASSERT_TRUE(std::ranges::is_sorted(res.begin(), res.end()));
}

TEST(kalyakina_a_shell_with_simple_merge_all, simple_merge_sort_10000) {
  std::vector<int> vec1 = CreateSortedVector(10000, -3000);
  std::vector<int> vec2 = CreateSortedVector(10000, -1000);

  std::vector<int> res = kalyakina_a_shell_with_simple_merge_all::ShellSortALL::SimpleMergeSort(vec1, vec2);
  ASSERT_TRUE(std::ranges::is_sorted(res.begin(), res.end()));
}
