#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "all/kazunin_n_montecarlo/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

using std::sin;

namespace {
template <std::size_t N, typename F>
void MonteCarloTest(F f, std::size_t precision, std::array<std::pair<double, double>, N> limits) {
  boost::mpi::communicator world;

  double out = 0.0;
  double ref = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs = {reinterpret_cast<uint8_t *>(&precision), reinterpret_cast<uint8_t *>(&limits)};
    task_data->inputs_count = {1, N};
    task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
    task_data->outputs_count = {1};
  }

  // Create Task
  kazunin_n_montecarlo_all::MonteCarloAll<N, F> task_all(task_data, f);
  ASSERT_TRUE(task_all.Validation());
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();

  if (world.rank() == 0) {
    task_data->outputs = {reinterpret_cast<uint8_t *>(&ref)};

    kazunin_n_montecarlo_seq::MonteCarloSeq<N, F> task_seq(task_data, f);
    ASSERT_TRUE(task_seq.Validation());
    task_seq.PreProcessing();
    task_seq.Run();
    task_seq.PostProcessing();

    EXPECT_NEAR(out, ref, 0.2);
  }
}
template <std::size_t N, typename F>
void InvalidMonteCarloTest(F f, std::size_t precision, std::array<std::pair<double, double>, N> limits) {
  boost::mpi::communicator world;

  double out = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs = {reinterpret_cast<uint8_t *>(&precision), reinterpret_cast<uint8_t *>(&limits)};
    task_data->inputs_count = {1, N};
    task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
    task_data->outputs_count = {1};
  }

  // Create Task
  kazunin_n_montecarlo_all::MonteCarloAll<N, F> task(task_data, f);
  if (world.rank() == 0) {
    EXPECT_FALSE(task.Validation());
  }
}
}  // namespace

TEST(kazunin_n_montecarlo_all, zero_precision) {
  const std::size_t n = 2;
  InvalidMonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      0, {{{0.0, 1.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, no_bounds) {
  const std::size_t n = 2;
  InvalidMonteCarloTest<n>([](const std::array<double, n> &args) { return -1; }, 0, {});
}

TEST(kazunin_n_montecarlo_all, sin_prod_2d) {
  const std::size_t n = 2;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      5000, {{{0.0, 1.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, sin_sum_2d) {
  const std::size_t n = 2;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 0.0,
                               [](const double acc, double component) { return acc + sin(component); });
      },
      7000, {{{0.0, 1.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, sin_prod_3d) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      5000, {{{0.0, 1.0}, {-1.0, 0.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, sin_sum_3d) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 0.0,
                               [](const double acc, double component) { return acc + sin(component); });
      },
      5000, {{{0.0, 1.0}, {-1.0, 0.0}, {-1.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, product_3d) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
}

TEST(kazunin_n_montecarlo_all, sum_3d) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
}

TEST(kazunin_n_montecarlo_all, sin_prod_2d_degenerate) {
  const std::size_t n = 2;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, sin_sum_2d_degenerate) {
  const std::size_t n = 2;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 0.0,
                               [](const double acc, double component) { return acc + sin(component); });
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, sin_prod_3d_degenerate) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0,
                               [](const double acc, double component) { return acc * sin(component); });
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, sin_sum_3d_degenerate) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 0.0,
                               [](const double acc, double component) { return acc + sin(component); });
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, product_3d_degenerate) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, sum_3d_degenerate) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}});
}

TEST(kazunin_n_montecarlo_all, sum_3d_coinciding_bounds) {
  const std::size_t n = 3;
  MonteCarloTest<n>(
      [](const std::array<double, n> &args) {
        return std::accumulate(args.begin(), args.end(), 1.0, std::multiplies<>());
      },
      5000, {{{1.0, 1.0}, {5.0, 5.0}, {9.0, 9.0}}});
}
