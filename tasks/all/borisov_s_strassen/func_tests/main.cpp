#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/borisov_s_strassen/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<double> MultiplyNaiveDouble(const std::vector<double>& a, const std::vector<double>& b, int rows_a,
                                        int cols_a, int cols_b) {
  std::vector<double> c(rows_a * cols_b, 0.0);
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      double sum = 0.0;
      for (int k = 0; k < cols_a; ++k) {
        sum += a[(i * cols_a) + k] * b[(k * cols_b) + j];
      }
      c[(i * cols_b) + j] = sum;
    }
  }
  return c;
}

std::vector<double> GenerateRandomMatrix(int rows, int cols, int seed, double min_val = -50.0, double max_val = 50.0) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(min_val, max_val);
  std::vector<double> m(rows * cols);
  for (double& v : m) {
    v = dist(rng);
  }
  return m;
}

namespace test_utils {

constexpr double kTol = 1e-9;

bool IsMaster() {
  static boost::mpi::communicator world;
  return world.rank() == 0;
}

using UniqueBuf = std::unique_ptr<double[]>;

UniqueBuf ExecuteTask(const std::vector<double>& in, std::size_t out_count) {
  auto data = std::make_shared<ppc::core::TaskData>();

  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  data->inputs_count.emplace_back(in.size());

  UniqueBuf out(new double[out_count]());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.get()));
  data->outputs_count.emplace_back(out_count);

  borisov_s_strassen_all::ParallelStrassenMpiStl task(data);
  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  return out;
}

void ExpectVectorNear(const std::vector<double>& expected, const double* actual, std::size_t offset,
                      double tol = kTol) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[offset + i], tol);
  }
}

}  // namespace test_utils
}  // namespace

TEST(borisov_s_strassen_all, OneByOne) {
  std::vector<double> in = {1.0, 1.0, 1.0, 1.0, 7.5, 2.5};
  auto out = test_utils::ExecuteTask(in, 3);

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], 1.0);
    EXPECT_DOUBLE_EQ(out[1], 1.0);
    EXPECT_DOUBLE_EQ(out[2], 18.75);
  }
}

TEST(borisov_s_strassen_all, TwoByTwo) {
  const std::vector<double> a = {1.0, 2.5, 3.0, 4.0};
  const std::vector<double> b = {1.5, 2.0, 0.5, 3.5};
  const std::vector<double> c_exp = {2.75, 10.75, 6.5, 20.0};

  std::vector<double> in = {2.0, 2.0, 2.0, 2.0};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 6);

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], 2.0);
    EXPECT_DOUBLE_EQ(out[1], 2.0);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Rectangular2x3_3x4) {
  const std::vector<double> a = {1.0, 2.5, 3.0, 4.0, 5.5, 6.0};
  const std::vector<double> b = {0.5, 1.0, 2.0, 1.5, 2.0, 0.5, 1.0, 3.0, 4.0, 2.5, 0.5, 1.0};
  const auto c_exp = MultiplyNaiveDouble(a, b, 2, 3, 4);

  std::vector<double> in = {2.0, 3.0, 3.0, 4.0};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 10);

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], 2.0);
    EXPECT_DOUBLE_EQ(out[1], 4.0);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square5x5_Random) {
  const int n = 5;
  auto a = GenerateRandomMatrix(n, n, 7777);
  auto b = GenerateRandomMatrix(n, n, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in = {double(n), double(n), double(n), double(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square20x20_Random) {
  const int n = 20;
  auto a = GenerateRandomMatrix(n, n, 7777);
  auto b = GenerateRandomMatrix(n, n, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                            static_cast<double>(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square32x32_Random) {
  const int n = 32;
  auto a = GenerateRandomMatrix(n, n, 7777);
  auto b = GenerateRandomMatrix(n, n, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                            static_cast<double>(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square128x128_Random) {
  const int n = 128;
  auto a = GenerateRandomMatrix(n, n, 7777);
  auto b = GenerateRandomMatrix(n, n, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                            static_cast<double>(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square128x128_IdentityMatrix) {
  const int n = 128;
  auto a = GenerateRandomMatrix(n, n, 7777);
  std::vector<double> e(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    e[(i * n) + i] = 1.0;
  }

  std::vector<double> in = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                            static_cast<double>(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), e.begin(), e.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(a, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square129x129_Random) {
  const int n = 129;
  auto a = GenerateRandomMatrix(n, n, 7777);
  auto b = GenerateRandomMatrix(n, n, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                            static_cast<double>(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square240x240_Random) {
  const int n = 240;
  auto a = GenerateRandomMatrix(n, n, 7777);
  auto b = GenerateRandomMatrix(n, n, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                            static_cast<double>(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square512x512_Random) {
  const int n = 512;
  auto a = GenerateRandomMatrix(n, n, 7777);
  auto b = GenerateRandomMatrix(n, n, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                            static_cast<double>(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Square600x600_Random) {
  const int n = 600;
  auto a = GenerateRandomMatrix(n, n, 7777);
  auto b = GenerateRandomMatrix(n, n, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in = {double(n), double(n), double(n), double(n)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (n * n));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], n);
    EXPECT_DOUBLE_EQ(out[1], n);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, ValidCase) {
  std::vector<double> in = {2.0, 3.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  data->inputs_count.emplace_back(in.size());
  data->outputs.emplace_back(nullptr);
  data->outputs_count.emplace_back(0);

  borisov_s_strassen_all::ParallelStrassenMpiStl task(data);
  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
}

TEST(borisov_s_strassen_all, MismatchCase) {
  std::vector<double> in = {2.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  data->inputs_count.emplace_back(in.size());
  data->outputs.emplace_back(nullptr);
  data->outputs_count.emplace_back(0);

  borisov_s_strassen_all::ParallelStrassenMpiStl task(data);
  task.PreProcessingImpl();
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(borisov_s_strassen_all, NotEnoughDataCase) {
  std::vector<double> in = {2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  data->inputs_count.emplace_back(in.size());
  data->outputs.emplace_back(nullptr);
  data->outputs_count.emplace_back(0);

  borisov_s_strassen_all::ParallelStrassenMpiStl task(data);
  task.PreProcessingImpl();
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(borisov_s_strassen_all, Rectangular16x17_Random) {
  const int r = 16;
  const int c = 17;
  const int cb = 18;
  auto a = GenerateRandomMatrix(r, c, 7777);
  auto b = GenerateRandomMatrix(c, cb, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, r, c, cb);

  std::vector<double> in = {static_cast<double>(r), static_cast<double>(c), static_cast<double>(c),
                            static_cast<double>(cb)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (r * cb));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], r);
    EXPECT_DOUBLE_EQ(out[1], cb);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Rectangular19x23_Random) {
  const int r = 19;
  const int c = 23;
  const int cb = 21;
  auto a = GenerateRandomMatrix(r, c, 7777);
  auto b = GenerateRandomMatrix(c, cb, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, r, c, cb);

  std::vector<double> in = {static_cast<double>(r), static_cast<double>(c), static_cast<double>(c),
                            static_cast<double>(cb)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (r * cb));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], r);
    EXPECT_DOUBLE_EQ(out[1], cb);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}

TEST(borisov_s_strassen_all, Rectangular32x64_Random) {
  const int r = 32;
  const int c = 64;
  const int cb = 32;
  auto a = GenerateRandomMatrix(r, c, 7777);
  auto b = GenerateRandomMatrix(c, cb, 7777);
  auto c_exp = MultiplyNaiveDouble(a, b, r, c, cb);

  std::vector<double> in = {static_cast<double>(r), static_cast<double>(c), static_cast<double>(c),
                            static_cast<double>(cb)};
  in.insert(in.end(), a.begin(), a.end());
  in.insert(in.end(), b.begin(), b.end());

  auto out = test_utils::ExecuteTask(in, 2 + (r * cb));

  if (test_utils::IsMaster()) {
    EXPECT_DOUBLE_EQ(out[0], r);
    EXPECT_DOUBLE_EQ(out[1], cb);
    test_utils::ExpectVectorNear(c_exp, out.get(), 2);
  }
}
