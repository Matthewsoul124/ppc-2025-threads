#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/kolokolova_d_integral_simpson_method/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

TEST(kolokolova_d_integral_simpson_method_all, test_easy_func) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {2, 4, 3, 6};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 81.0;
  double error = 0.1;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_func_two_value1) {
  auto func = [](std::vector<double> vec) { return 3 * vec[0] * vec[0] * vec[1] * vec[1]; };
  std::vector<int> step = {10, 10};
  std::vector<int> bord = {4, 6, 3, 6};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 9576.0;
  double error = 0.1;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_func_two_value2) {
  auto func = [](std::vector<double> vec) { return 4 * vec[0] * 2 * vec[1]; };
  std::vector<int> step = {4, 4};
  std::vector<int> bord = {0, 2, 1, 4};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 120.0;
  double error = 0.1;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_func_two_value3) {
  auto func = [](std::vector<double> vec) { return (vec[0] * vec[1] / 6) + (2 * vec[0]); };
  std::vector<int> step = {8, 8};
  std::vector<int> bord = {3, 8, 1, 5};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 275.0;
  double error = 0.1;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_func_three_value) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1] * vec[2]; };
  std::vector<int> step = {20, 20, 20};
  std::vector<int> bord = {0, 2, 3, 6, 4, 8};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 662.0;
  double error = 0.1;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_difficult_func1) {
  auto func = [](std::vector<double> vec) { return (std::cos(vec[0]) * std::sin(vec[1])); };
  std::vector<int> step = {20, 20};
  std::vector<int> bord = {0, 1, 0, 3};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 1.6745;
  double error = 1.0;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_difficult_func2) {
  auto func = [](std::vector<double> vec) { return (std::cos(vec[0]) * std::sin(vec[1])); };
  std::vector<int> step = {20, 20};
  std::vector<int> bord = {0, 1, 0, 3};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  ASSERT_EQ(test_task_all.Validation(), true);
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  double ans = 1.6745;
  double error = 1.0;
  if (world.rank() == 0) {
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_validation1) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step;
  std::vector<int> bord = {0, 1, 0, 3};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_validation2) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord;
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_validation3) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {10, 0, 20, 5};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_validation4) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2, 3};
  std::vector<int> bord = {0, 4, 0, 5};
  double func_result = 0.0;
  boost::mpi::communicator world;

  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
    task_data_all->inputs_count.emplace_back(step.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
    task_data_all->inputs_count.emplace_back(bord.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
    task_data_all->outputs_count.emplace_back(1);
  }

  // Create Task
  kolokolova_d_integral_simpson_method_all::TestTaskALL test_task_all(task_data_all, func);
  if (world.rank() == 0) {
    ASSERT_EQ(test_task_all.Validation(), false);
  }
}