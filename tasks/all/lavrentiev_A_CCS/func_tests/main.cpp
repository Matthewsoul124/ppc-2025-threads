#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "all/lavrentiev_A_CCS/include/ops_stl.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> GenerateRandomMatrix(int size, int sparse_size) {
  std::vector<double> data(size);
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<> random_element(-500, 500);
  size = size / sparse_size;
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<double>(random_element(generator));
  }
  std::ranges::shuffle(data, generator);
  return data;
}

std::vector<double> GenerateSingleMatrix(int size) {
  std::vector<double> test_data(size, 0.0);
  int sqrt = static_cast<int>(std::sqrt(size));
  for (int i = 0; i < sqrt; ++i) {
    for (int j = 0; j < sqrt; ++j) {
      if (i == j) {
        test_data[(sqrt * i) + j] = 1.0;
      }
    }
  }
  return test_data;
}
constexpr auto kEpsilon = 0.000001;
struct TestData {
  std::vector<double> random_data;
  std::vector<double> single_matrix;
  std::vector<double> result;
  std::shared_ptr<ppc::core::TaskData> task_data_all;
  TestData(std::pair<int, int> matrix1_size, std::pair<int, int> matrix2_size, int sparse_size,
           boost::mpi::communicator &world);
  [[nodiscard]] lavrentiev_a_ccs_all::CCSALL CreateTask() const;
};

TestData::TestData(std::pair<int, int> matrix1_size, std::pair<int, int> matrix2_size, int sparse_size,
                   boost::mpi::communicator &world) {
  random_data = GenerateRandomMatrix(matrix1_size.first * matrix1_size.second, sparse_size);
  single_matrix = GenerateSingleMatrix(matrix2_size.first * matrix2_size.second);
  result.resize(matrix1_size.first * matrix2_size.second);
  task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(random_data.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
    task_data_all->inputs_count.emplace_back(matrix1_size.first);
    task_data_all->inputs_count.emplace_back(matrix1_size.second);
    task_data_all->inputs_count.emplace_back(matrix2_size.first);
    task_data_all->inputs_count.emplace_back(matrix2_size.second);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    task_data_all->outputs_count.emplace_back(result.size());
  }
}

lavrentiev_a_ccs_all::CCSALL TestData::CreateTask() const { return lavrentiev_a_ccs_all::CCSALL(task_data_all); }
}  // namespace

TEST(lavrentiev_a_ccs_all, test_0x0_matrix) {
  boost::mpi::communicator world;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> result;
  std::vector<double> test_result;
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    for (int i = 0; i < 4; ++i) {
      task_data_all->inputs_count.emplace_back(0);
    }
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    task_data_all->outputs_count.emplace_back(result.size());
  }
  lavrentiev_a_ccs_all::CCSALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_NEAR(result[i], test_result[i], kEpsilon);
    }
  }
}

TEST(lavrentiev_a_ccs_all, test_3x2_matrix) {
  boost::mpi::communicator world;
  std::vector<double> a{2.0, 0.0, 0.0, 4.0, 0.0, 1.0};
  std::vector<double> b{6.0, 0.0, 0.0, 0.0, 0.0, 9.0};
  std::vector<double> result(3 * 3, 0.0);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_all->inputs_count.emplace_back(3);
    task_data_all->inputs_count.emplace_back(2);
    task_data_all->inputs_count.emplace_back(2);
    task_data_all->inputs_count.emplace_back(3);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    task_data_all->outputs_count.emplace_back(result.size());
  }
  lavrentiev_a_ccs_all::CCSALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  std::vector<double> test_result{12.0, 0.0, 0.0, 0.0, 0.0, 36.0, 0.0, 0.0, 9.0};
  if (world.rank() == 0) {
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_NEAR(result[i], test_result[i], kEpsilon);
    }
  }
}

TEST(lavrentiev_a_ccs_all, test_3x3_matrixes) {
  boost::mpi::communicator world;
  std::vector<double> a{2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 1.0, 6.0};
  std::vector<double> b{6.0, 0.0, 0.0, 0.0, 0.0, 9.0, 7.0, 2.0, 0.0};
  std::vector<double> result(3 * 3, 0.0);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    for (auto i = 0; i < 4; ++i) {
      task_data_all->inputs_count.emplace_back(3);
    }
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    task_data_all->outputs_count.emplace_back(result.size());
  }
  lavrentiev_a_ccs_all::CCSALL test_task_all(task_data_all);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  std::vector<double> test_result{12.0, 0.0, 0.0, 7.0, 2.0, 36.0, 42.0, 12.0, 9.0};
  if (world.rank() == 0) {
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_NEAR(result[i], test_result[i], kEpsilon);
    }
  }
}

TEST(lavrentiev_a_ccs_all, test_12x12_matrix) {
  boost::mpi::communicator world;
  auto task = TestData({12, 12}, {12, 12}, 1, world);
  auto test_task_all = task.CreateTask();
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < task.result.size(); ++i) {
      EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
    }
  }
}

TEST(lavrentiev_a_ccs_all, test_25x25_matrix) {
  boost::mpi::communicator world;
  auto task = TestData({25, 25}, {25, 25}, 5, world);
  auto test_task_all = task.CreateTask();
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < task.result.size(); ++i) {
      EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
    }
  }
}

TEST(lavrentiev_a_ccs_all, test_10x0_matrix) {
  boost::mpi::communicator world;
  auto task = TestData({0, 10}, {10, 0}, 5, world);
  auto test_task_all = task.CreateTask();
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (world.rank() == 0) {
    for (size_t i = 0; i < task.result.size(); ++i) {
      EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
    }
  }
}
