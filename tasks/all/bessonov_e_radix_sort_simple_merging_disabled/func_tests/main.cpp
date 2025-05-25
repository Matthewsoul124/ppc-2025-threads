#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "all/bessonov_e_radix_sort_simple_merging/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> GenerateVector(std::size_t n, double first, double last) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(first, last);

  std::vector<double> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = dist(gen);
  }
  return result;
}
}  // namespace

TEST(bessonov_e_radix_sort_simple_merging_all, BasicSortingTest) {
  std::vector<double> input_vector = {3.4, 1.2, 0.5, 7.8, 2.3, 4.5, 6.7, 8.9, 1.0, 0.2, 5.6, 4.3, 9.1, 1.5, 3.0};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, SingleElementTest) {
  std::vector<double> input_vector = {42.0};
  std::vector<double> output_vector(1, 0.0);
  std::vector<double> result_vector = {42.0};

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, NegativeAndPositiveTest) {
  std::vector<double> input_vector = {-3.2, 1.1, -7.5, 0.0, 4.4, -2.2, 3.3};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {-7.5, -3.2, -2.2, 0.0, 1.1, 3.3, 4.4};

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, RandomVectorTest) {
  const std::size_t n = 1000;
  std::vector<double> input_vector = GenerateVector(n, -1000.0, 1000.0);
  std::vector<double> output_vector(n, 0.0);
  std::vector<double> result_vector = input_vector;
  std::ranges::sort(result_vector);

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, AllSameElementsTest) {
  std::vector<double> input_vector = {3.14, 3.14, 3.14, 3.14};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {3.14, 3.14, 3.14, 3.14};

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, ExtremeValuesTest) {
  std::vector<double> input_vector = {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 0.0,
                                      -42.5, 100.0};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {std::numeric_limits<double>::lowest(), -42.5, 0.0, 100.0,
                                       std::numeric_limits<double>::max()};

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, TinyNumbersTest) {
  std::vector<double> input_vector = {1e-10, -1e-10, 1e-20, -1e-20};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {-1e-10, -1e-20, 1e-20, 1e-10};

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, DenormalNumbersTest) {
  std::vector<double> input_vector = {1e-310, -1e-310, 0.0};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {-1e-310, 0.0, 1e-310};

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, ReverseOrderTest) {
  std::vector<double> input_vector = {9.1, 8.9, 7.8, 6.7, 5.6, 4.5, 4.3, 3.4, 3.0, 2.3, 1.5, 1.2, 1.0, 0.5, 0.2};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, Validation_NullInput) {
  std::vector<double> output(100);
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(nullptr);
    task_data->inputs_count.emplace_back(100);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_all, Validation_NullOutput) {
  std::vector<double> input(100);
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(nullptr);
    task_data->outputs_count.emplace_back(100);
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_all, Validation_SizeMismatch) {
  std::vector<double> input(100);
  std::vector<double> output(50);
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_all, Validation_SizeOverflow) {
  auto huge_size = static_cast<size_t>(INT_MAX) + 1;
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(nullptr);
    task_data->inputs_count.emplace_back(huge_size);
    task_data->outputs.emplace_back(nullptr);
    task_data->outputs_count.emplace_back(huge_size);
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_all, Validation_EmptyCounts) {
  std::vector<double> input(100);
  std::vector<double> output(100);
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_all, Validation_ZeroSize) {
  std::vector<double> input(0);
  std::vector<double> output(0);
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    task_data->inputs_count.emplace_back(0);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
    task_data->outputs_count.emplace_back(0);
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_all, Validation_MaxSize) {
  auto max_size = static_cast<size_t>(INT_MAX);
  boost::mpi::communicator world;
  auto* dummy_input = new uint8_t[1];
  auto* dummy_output = new uint8_t[1];
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(dummy_input);
    task_data->inputs_count.emplace_back(max_size);
    task_data->outputs.emplace_back(dummy_output);
    task_data->outputs_count.emplace_back(max_size);
  }

  bessonov_e_radix_sort_simple_merging_all::TestTaskALL task(task_data);

  delete[] dummy_input;
  delete[] dummy_output;

  ASSERT_TRUE(task.Validation());
}