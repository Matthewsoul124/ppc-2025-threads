#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "all/sharamygina_i_multi_dim_monte_carlo/include/ops_all.h"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> GetBoundaries(double left, double right, unsigned int dimension) {
  std::vector<double> v(dimension * 2);
  for (unsigned int i = 0; i < dimension; i++) {
    v[i * 2] = left;
    v[(i * 2) + 1] = right;
  }
  return v;
}
}  // namespace

TEST(sharamygina_i_multi_dim_monte_carlo_all, WrongInputCountValidationTest) {
  boost::mpi::communicator world;

  int iterations = 30000;
  double result = 0.0;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(3);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }

  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 1);
    return std::sin(values[0]);
  };
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);

  if (world.rank() == 0) {
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, WrongOutputCountValidationTest) {
  boost::mpi::communicator world;

  int iterations = 30000;
  double result = 0.0;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(0);
  }

  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 1);
    return std::sin(values[0]);
  };
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);

  if (world.rank() == 0) {
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, WrongBoundariesValidationTest) {
  boost::mpi::communicator world;

  int iterations = 30000;
  double result = 0.0;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size() - 1);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }

  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 1);
    return std::sin(values[0]);
  };
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);

  if (world.rank() == 0) {
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, EmptyOutputValidationTest) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::sin(values[0]);
  };

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs_count.emplace_back(0);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, EmptyInputValidationTest) {
  boost::mpi::communicator world;

  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(2);
    task_data->inputs_count.emplace_back(1);
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(0);
  }
  auto test_function = [](const std::vector<double>& values) { return 0.0; };
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 1DSinFunction) {
  boost::mpi::communicator world;

  int iterations = 30000;
  double result = 0.0;

  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 1);
    return std::sin(values[0]);
  };
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 0.45969769413186;
    double tol = 0.03 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 2DFunction) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 2);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return values[0] + values[1];
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 1;
    double tol = 0.03 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 3DFunction) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 3.0, 3);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return 1.0;
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 27;
    double tol = 0.03 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 3DFunctionPrime) {
  boost::mpi::communicator world;

  int iterations = 30011;
  std::vector<double> boundaries = GetBoundaries(0.0, 3.0, 3);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return 1.0;
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 27;
    double tol = 0.03 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 4DFunction) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(-1.0, 5.0, 4);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 4);
    return (values[0] * values[3]) + values[2] - (0.3 * values[1]);
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 6998.4;
    double tol = 0.07 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 2DExpFunction) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(1.0, 1.5, 2);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::exp(values[0] + values[1]);
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 3.109605100711371;
    double tol = 0.03 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 2DExpFunction12345) {
  boost::mpi::communicator world;

  int iterations = 12345;
  std::vector<double> boundaries = GetBoundaries(1.0, 1.5, 2);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::exp(values[0] + values[1]);
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 3.109605100711371;
    double tol = 0.07 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 10DFunction) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 10);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 10);
    return 5 + values[0];
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 5.5;
    double tol = 0.03 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 3DFunctionWithDifferentBoundaries) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = {1.0, 2.0, 1.3, 4.4, 0.5, 0.98};
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return 1.0;
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 1.488;
    double tol = 0.03 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 2DFunctionWithDifferentBoundaries) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = {3.0, 7.1, 1.3, 4.4};
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return -(values[0] * std::cos(values[1]));
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 39.65339;
    double tol = 0.04 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}

TEST(sharamygina_i_multi_dim_monte_carlo_all, 3DSinFunctionWithDifferentBoundaries) {
  boost::mpi::communicator world;

  int iterations = 30000;
  std::vector<double> boundaries = {1.0, 2.0, 1.3, 4.4, 0.5, 0.98};
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return std::sin(values[0]) + std::cos(values[1]) + std::exp(values[2]);
  };
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
    task_data->inputs_count.emplace_back(boundaries.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
    task_data->inputs_count.emplace_back(1);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.emplace_back(1);
  }
  sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask test_task(task_data, test_function);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());
  if (world.rank() == 0) {
    double expected = 3.652697810257515;
    double tol = 0.03 * expected;
    EXPECT_NEAR(result, expected, tol);
  }
}
