#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "all/filateva_e_simpson/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace {
void RunTest(size_t mer, size_t steps, std::vector<double> &a, std::vector<double> &b, filateva_e_simpson_all::Func f,
             double ans) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> res(1, 0);

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.SetFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_NEAR(res[0], ans, 0.01);
  }
}

void RunTest(size_t mer, size_t steps, std::vector<double> &a, std::vector<double> &b, filateva_e_simpson_all::Func f,
             filateva_e_simpson_all::Func p_f) {
  double ans = p_f(b) - p_f(a);
  RunTest(mer, steps, a, b, f, ans);
}

void RunTestError(size_t mer, size_t steps, std::vector<double> &a, std::vector<double> &b) {
  boost::mpi::communicator world;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> res(1, 0);

  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  ASSERT_FALSE(simpson.Validation());
}

}  // namespace

TEST(filateva_e_simpson_all, test_x_pow_2) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {10};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };
  filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0] * x[0] / 3;
  };
  RunTest(mer, steps, a, b, f, integral_f);
}

TEST(filateva_e_simpson_all, test_x_pow_2_negative) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {-10};
  std::vector<double> b = {10};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };
  filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0] * x[0] / 3;
  };
  RunTest(mer, steps, a, b, f, integral_f);
}

TEST(filateva_e_simpson_all, test_x) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {10};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0];
  };
  filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0] / 2;
  };
  RunTest(mer, steps, a, b, f, integral_f);
}

TEST(filateva_e_simpson_all, test_x_pow_3) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {100};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0] * x[0];
  };
  filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return std::pow(x[0], 4) / 4;
  };
  RunTest(mer, steps, a, b, f, integral_f);
}

TEST(filateva_e_simpson_all, test_x_del) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {10};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return 1 / x[0];
  };
  filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return std::log(x[0]);
  };
  RunTest(mer, steps, a, b, f, integral_f);
}

TEST(filateva_e_simpson_all, test_x_sin) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {std::numbers::pi};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return std::sin(x[0]);
  };
  filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return -std::cos(x[0]);
  };
  RunTest(mer, steps, a, b, f, integral_f);
}

TEST(filateva_e_simpson_all, test_x_cos) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {std::numbers::pi / 2};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return std::cos(x[0]);
  };
  filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return std::sin(x[0]);
  };
  RunTest(mer, steps, a, b, f, integral_f);
}

TEST(filateva_e_simpson_all, test_gausa) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {0};
  std::vector<double> b = {1};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return pow(std::numbers::e, -pow(x[0], 2));
  };
  RunTest(mer, steps, a, b, f, 0.746824);
}

TEST(filateva_e_simpson_all, test_sum_integral) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {0};
  std::vector<double> b = {10};
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return pow(x[0], 3) + pow(x[0], 2) + x[0];
  };
  filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return (pow(x[0], 4) / 4) + (pow(x[0], 3) / 3) + (pow(x[0], 2) / 2);
  };
  RunTest(mer, steps, a, b, f, integral_f);
}

TEST(filateva_e_simpson_all, test_error_1) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {10};
  std::vector<double> b = {0};
  RunTestError(mer, steps, a, b);
}

TEST(filateva_e_simpson_all, test_error_n_mer) {
  size_t mer = 3;
  size_t steps = 100;
  std::vector<double> a = {0, 0, 10};
  std::vector<double> b = {10, 10, 0};
  RunTestError(mer, steps, a, b);
}

TEST(filateva_e_simpson_all, test_error_2) {
  size_t mer = 1;
  size_t steps = 101;
  std::vector<double> a = {0};
  std::vector<double> b = {10};
  RunTestError(mer, steps, a, b);
}

TEST(filateva_e_simpson_all, test_x_y_pow_2) {
  size_t mer = 2;
  size_t steps = 100;
  std::vector<double> a = {0, 0};
  std::vector<double> b = {1, 1};
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return (param[0] * param[0]) + (param[1] * param[1]);
  };
  RunTest(mer, steps, a, b, f, 0.66666);
}

TEST(filateva_e_simpson_all, test_x_y) {
  size_t mer = 2;
  size_t steps = 100;
  std::vector<double> a = {0, 0};
  std::vector<double> b = {10, 10};
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return param[0] + param[1];
  };
  RunTest(mer, steps, a, b, f, 1000);
}

TEST(filateva_e_simpson_all, test_sin_x_cos_y) {
  size_t mer = 2;
  size_t steps = 100;
  std::vector<double> a = {0, 0};
  std::vector<double> b = {std::numbers::pi, std::numbers::pi / 2};
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return std::sin(param[0]) * std::cos(param[1]);
  };
  RunTest(mer, steps, a, b, f, 2);
}

TEST(filateva_e_simpson_all, test_sum_integral_x_y) {
  size_t mer = 2;
  size_t steps = 100;
  std::vector<double> a = {0, 0};
  std::vector<double> b = {10, 10};
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return pow(param[0], 3) + pow(param[1], 2) + param[0];
  };
  RunTest(mer, steps, a, b, f, 28833.33);
}

TEST(filateva_e_simpson_all, test_x_y_negative) {
  size_t mer = 2;
  size_t steps = 100;
  std::vector<double> a = {-10, -10};
  std::vector<double> b = {10, 10};
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return param[0] + param[1];
  };
  RunTest(mer, steps, a, b, f, 0.0);
}
