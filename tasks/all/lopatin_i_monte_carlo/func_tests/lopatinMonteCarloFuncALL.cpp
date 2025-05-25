#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "all/lopatin_i_monte_carlo/include/lopatinMonteCarloALL.hpp"
#include "core/task/include/task.hpp"

namespace lopatin_i_monte_carlo_all {

std::vector<double> GenerateBounds(double min_val, double max_val, int dimensions) {
  std::vector<double> bounds;
  for (int i = 0; i < dimensions; ++i) {
    bounds.push_back(min_val);
    bounds.push_back(max_val);
  }
  return bounds;
}
}  // namespace lopatin_i_monte_carlo_all

TEST(lopatin_i_monte_carlo_all, validationInvalidInputOddBoundsCount) {
  boost::mpi::communicator world;

  std::vector<double> bounds = {0.0, 1.0, 2.0};  // even num of bounds
  const int iterations = 10;
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());  // incorrect num of inputs
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, [](const std::vector<double>&) { return 1.0; });
  if (world.rank() == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(lopatin_i_monte_carlo_all, validationMissingOutputData) {
  boost::mpi::communicator world;

  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, 1.0, 2);
  const int iterations = 10;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(4);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);
  }

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, [](const std::vector<double>&) { return 1.0; });
  if (world.rank() == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(lopatin_i_monte_carlo_all, validationZeroIterations) {
  boost::mpi::communicator world;

  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, 1.0, 2);
  const int iterations = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(4);
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);
  }

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, [](const std::vector<double>&) { return 1.0; });
  if (world.rank() == 0) {
    ASSERT_FALSE(task.Validation());
  }
}

TEST(lopatin_i_monte_carlo_all, highDimensionalIntegration) {
  boost::mpi::communicator world;

  const int dimensions = 7;
  const int iterations = 40000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(-1.0, 1.0, dimensions);
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, [](const std::vector<double>&) {
    return 1.0;  // const
  });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    const double expected = std::pow(2.0, dimensions);  // hypercube
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 1DConstantFunction) {
  boost::mpi::communicator world;

  const int dimensions = 1;
  const int iterations = 100000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(2.0, 5.0, dimensions);  // [2, 5]
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, [](const std::vector<double>& x) {
    return 1.0;  // f(x) = 1
  });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    const double expected = 3.0;
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 3DExponentialFunction) {
  boost::mpi::communicator world;

  const int dimensions = 3;
  const int iterations = 70000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, 1.0, dimensions);  // [0,1]^3
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }
  lopatin_i_monte_carlo_all::TestTaskAll task(
      task_data, [](const std::vector<double>& x) { return std::pow(std::numbers::e, x[0] + x[1] + x[2]); });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    const double expected = std::pow(std::numbers::e - 1, 3);  // = 5.073
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 2DLinearFunction) {
  boost::mpi::communicator world;

  const int dimensions = 2;
  const int iterations = 70000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, 1.0, dimensions);
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }

  auto function = [](const std::vector<double>& x) {
    assert(x.size() == 2);
    return x[0] + x[1];
  };

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    const double expected = 1.0;
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 3DProductFunction) {
  boost::mpi::communicator world;

  const int dimensions = 3;
  const int iterations = 50000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, 1.0, dimensions);
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data,
                                              [](const std::vector<double>& x) { return x[0] * x[1] * x[2]; });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    const double expected = 0.125;
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 4DQuadraticFunction) {
  boost::mpi::communicator world;

  const int dimensions = 4;
  const int iterations = 70000;  // increase for 4D
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, 1.0, dimensions);
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }

  // (x1 + x2 + x3 + x4)^2
  auto function = [](const std::vector<double>& x) {
    double sum = x[0] + x[1] + x[2] + x[3];
    return sum * sum;
  };

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    // anal 13/3 = 4.33333
    const double expected = 13.0 / 3.0;
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 2DCosineFunction) {
  boost::mpi::communicator world;

  const int dimensions = 2;
  const int iterations = 100000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, std::numbers::pi / 2, dimensions);
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }

  // cos(x + y)
  auto function = [](const std::vector<double>& x) { return std::cos(x[0] + x[1]); };

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    // analytical = 0
    const double expected = 0.0;
    const double tolerance = 0.05;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 2DSqrtFunction) {
  boost::mpi::communicator world;

  const int dimensions = 2;
  const int iterations = 40000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, 1.0, dimensions);
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }

  // sqrt(x + y)
  auto function = [](const std::vector<double>& x) { return std::sqrt(x[0] + x[1]); };

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    // analytical = 0.975
    const double expected = 0.975;
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 3DSinFunction) {
  boost::mpi::communicator world;

  const int dimensions = 3;
  const int iterations = 40000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(0.0, std::numbers::pi / 6, dimensions);
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }
  // sin(x + y + z)
  auto function = [](const std::vector<double>& x) { return std::sin(x[0] + x[1] + x[2]); };

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    // analytical = 0.098
    const double expected = 0.098;
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}

TEST(lopatin_i_monte_carlo_all, 4DLogFunction) {
  boost::mpi::communicator world;

  const int dimensions = 4;
  const int iterations = 70000;
  std::vector<double> bounds = lopatin_i_monte_carlo_all::GenerateBounds(1.0, std::numbers::e, dimensions);
  double result = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
    task_data->inputs_count.push_back(bounds.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
    task_data->inputs_count.push_back(1);

    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
    task_data->outputs_count.push_back(1);
  }
  // ln(x1 + x2 + x3 + x4)
  auto function = [](const std::vector<double>& x) { return std::log(x[0] + x[1] + x[2] + x[3]); };

  lopatin_i_monte_carlo_all::TestTaskAll task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  if (world.rank() == 0) {
    // analytical = 17.4108
    const double expected = 17.4108;
    const double tolerance = 0.05 * expected;
    EXPECT_NEAR(result, expected, tolerance);  // error 5%
  }
}
