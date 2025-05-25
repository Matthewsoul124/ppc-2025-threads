#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "all/shulpin_i_jarvis_passage/include/ops_all.hpp"
#include "all/shulpin_i_jarvis_passage/include/test_modules.hpp"

TEST(shulpin_i_jarvis_all, square_with_point) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
    expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
  }

  shulpin_all_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_all, ox_line) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    input = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
    expected = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  }

  shulpin_all_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_all, octagone) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    input = {{1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}, {0, 1}};
    expected = {{0, 1}, {1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}};
  }

  shulpin_all_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_all, repeated_points) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 0}, {0, 0}};
    expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
  }

  shulpin_all_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_all, real_case) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    input = {{1, 1}, {3, 2}, {5, 1}, {4, 3}, {2, 4}, {1, 3}, {3, 3}};
    expected = {{1, 1}, {5, 1}, {4, 3}, {2, 4}, {1, 3}};
  }

  shulpin_all_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_all, star_case) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    // clang-format off
    input = {
        {0.0, 3.0},
        {1.0, 1.0},
        {3.0, 1.0},
        {1.5, -0.5},
        {2.5, -3.0},
        {0.0, -1.5},
        {-2.5, -3.0},
        {-1.5, -0.5},
        {-3.0, 1.0},
        {-1.0, 1.0},
        {0.0, 3.0}};
    expected = {
        {1, 1},
        {5, 1},
        {4, 3},
        {2, 4},
        {1, 3}};
    // clang-format on
  }

  shulpin_all_test_module::MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_all, one_point_validation_false) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    input = {{0, 0}};
    expected = {{0, 0}};
  }

  shulpin_all_test_module::TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_all, three_points_validation_false) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    input = {{1, 1}, {2, 2}};
    expected = {{1, 1}, {2, 2}};
  }

  shulpin_all_test_module::TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_all, zero_points_validation_false) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  if (rank == 0) {
    input = {};
    expected = {};
  }

  shulpin_all_test_module::TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_all, circle_r10_p10) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  size_t num_points = 10;

  if (rank == 0) {
    shulpin_i_jarvis_all::Point center{0, 0};

    double radius = 10.0;

    input = shulpin_all_test_module::GeneratePointsInCircle(num_points, center, radius);
    expected = input;
  }
  shulpin_all_test_module::TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_all, circle_r10_p20) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  size_t num_points = 20;

  if (rank == 0) {
    shulpin_i_jarvis_all::Point center{0, 0};

    double radius = 10.0;

    input = shulpin_all_test_module::GeneratePointsInCircle(num_points, center, radius);
    expected = input;
  }
  shulpin_all_test_module::TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_all, random_10_points) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  size_t num_points = 10;

  if (rank == 0) {
    input = shulpin_all_test_module::GenerateRandomPoints(num_points);
    expected = shulpin_all_test_module::ComputeConvexHull(input);
  }

  shulpin_all_test_module::RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_all, random_50_points) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  size_t num_points = 50;

  if (rank == 0) {
    input = shulpin_all_test_module::GenerateRandomPoints(num_points);
    expected = shulpin_all_test_module::ComputeConvexHull(input);
  }

  shulpin_all_test_module::RandomTestBody(input, expected);
}

TEST(shulpin_i_jarvis_all, random_100_points) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<shulpin_i_jarvis_all::Point> input;
  std::vector<shulpin_i_jarvis_all::Point> expected;

  size_t num_points = 100;

  if (rank == 0) {
    input = shulpin_all_test_module::GenerateRandomPoints(num_points);
    expected = shulpin_all_test_module::ComputeConvexHull(input);
  }

  shulpin_all_test_module::RandomTestBody(input, expected);
}