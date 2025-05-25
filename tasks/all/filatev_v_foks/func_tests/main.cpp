#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/filatev_v_foks/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<double> GeneratMatrix(filatev_v_foks_all::MatrixSize size) {
  std::vector<double> matrix(size.n * size.m);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (auto& el : matrix) {
    el = dist(gen);
  }

  return matrix;
}

std::vector<double> IdentityMatrix(size_t size) {
  std::vector<double> matrix(size * size, 0);

  for (size_t i = 0; i < size; i++) {
    matrix[(i * size) + i] = 1;
  }

  return matrix;
}

void RunTest(size_t size_block, std::vector<size_t>& size, std::vector<double>& matrix_a, std::vector<double>& matrix_b,
             std::vector<double>& ans) {
  boost::mpi::communicator world;

  filatev_v_foks_all::MatrixSize size_a(size[0], size[1]);
  filatev_v_foks_all::MatrixSize size_b(size[2], size[3]);
  filatev_v_foks_all::MatrixSize size_c(size[4], size[5]);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(size_a.n);
    task_data->inputs_count.emplace_back(size_a.m);
    task_data->inputs_count.emplace_back(size_b.n);
    task_data->inputs_count.emplace_back(size_b.m);
    task_data->inputs_count.emplace_back(size_block);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_b.data()));

    task_data->outputs_count.emplace_back(size_c.n);
    task_data->outputs_count.emplace_back(size_c.m);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  }

  filatev_v_foks_all::Focks focks(task_data);
  ASSERT_TRUE(focks.Validation());
  focks.PreProcessing();
  focks.Run();
  focks.PostProcessing();

  if (world.rank() == 0) {
    EXPECT_EQ(ans, matrix_c);
  }
}

void TestError(size_t size_block, std::vector<size_t>& size, std::vector<double>& matrix_a,
               std::vector<double>& matrix_b) {
  boost::mpi::communicator world;

  filatev_v_foks_all::MatrixSize size_a(size[0], size[1]);
  filatev_v_foks_all::MatrixSize size_b(size[2], size[3]);
  filatev_v_foks_all::MatrixSize size_c(size[4], size[5]);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> matrix_c(size_c.n * size_c.m, 0.0);

  if (world.rank() == 0) {
    task_data->inputs_count.emplace_back(size_a.n);
    task_data->inputs_count.emplace_back(size_a.m);
    task_data->inputs_count.emplace_back(size_b.n);
    task_data->inputs_count.emplace_back(size_b.m);
    task_data->inputs_count.emplace_back(size_block);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_b.data()));

    task_data->outputs_count.emplace_back(size_c.n);
    task_data->outputs_count.emplace_back(size_c.m);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  }

  filatev_v_foks_all::Focks focks(task_data);
  ASSERT_FALSE(focks.Validation());
}

}  // namespace

TEST(filatev_v_foks_all, test_matrix_4_4_block_2) {
  std::vector<size_t> size(6, 4);
  size_t size_block = 2;
  std::vector<double> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b = {8, 1, 1, 5, 4, 1, 0, 6, 5, 3, 1, 1, 1, 4, 2, 9};
  std::vector<double> matrix_ans = {35, 28, 12, 56, 107, 64, 28, 140, 179, 100, 44, 224, 251, 136, 60, 308};

  RunTest(size_block, size, matrix_a, matrix_b, matrix_ans);
}

TEST(filatev_v_foks_all, test_matrix_1_1_block_2) {
  std::vector<size_t> size(6, 1);
  size_t size_block = 2;
  std::vector<double> matrix_a = {4};
  std::vector<double> matrix_b = {6};
  std::vector<double> matrix_ans = {24};

  RunTest(size_block, size, matrix_a, matrix_b, matrix_ans);
}

TEST(filatev_v_foks_all, test_matrix_4_4_block_3) {
  std::vector<size_t> size(6, 4);
  size_t size_block = 3;

  std::vector<double> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b = {8, 1, 1, 5, 4, 1, 0, 6, 5, 3, 1, 1, 1, 4, 2, 9};
  std::vector<double> matrix_ans = {35, 28, 12, 56, 107, 64, 28, 140, 179, 100, 44, 224, 251, 136, 60, 308};

  RunTest(size_block, size, matrix_a, matrix_b, matrix_ans);
}

TEST(filatev_v_foks_all, test_matrix_4_4_block_2_IdentityMatrix) {
  std::vector<size_t> size(6, 4);
  size_t size_block = 2;
  std::vector<double> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b = IdentityMatrix(4);

  RunTest(size_block, size, matrix_a, matrix_b, matrix_a);
}

TEST(filatev_v_foks_all, test_matrix_10_10_block_2_IdentityMatrix) {
  std::vector<size_t> size(6, 10);
  size_t size_block = 2;
  std::vector<double> matrix_a = GeneratMatrix({10, 10});
  std::vector<double> matrix_b = IdentityMatrix(10);

  RunTest(size_block, size, matrix_a, matrix_b, matrix_a);
}

TEST(filatev_v_foks_all, test_matrix_10_10_block_2_IdentityMatrix_revert) {
  std::vector<size_t> size(6, 10);
  size_t size_block = 2;
  std::vector<double> matrix_a = IdentityMatrix(10);
  std::vector<double> matrix_b = GeneratMatrix({10, 10});

  RunTest(size_block, size, matrix_a, matrix_b, matrix_b);
}

TEST(filatev_v_foks_all, test_matrix_10_10_block_5_IdentityMatrix) {
  std::vector<size_t> size(6, 10);
  size_t size_block = 5;
  std::vector<double> matrix_a = GeneratMatrix({10, 10});
  std::vector<double> matrix_b = IdentityMatrix(10);

  RunTest(size_block, size, matrix_a, matrix_b, matrix_a);
}

TEST(filatev_v_foks_all, test_matrix_4_3_block_2) {
  std::vector<size_t> size = {3, 4, 4, 3, 4, 4};
  size_t size_block = 2;
  std::vector<double> matrix_a = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15};
  std::vector<double> matrix_b = {8, 1, 1, 5, 4, 1, 0, 6, 5, 3, 1, 1};
  std::vector<double> matrix_ans = {31, 12, 4, 20, 99, 32, 12, 68, 167, 52, 20, 116, 235, 72, 28, 164};

  RunTest(size_block, size, matrix_a, matrix_b, matrix_ans);
}

TEST(filatev_v_foks_all, test_matrix_4_1_block_2) {
  std::vector<size_t> size = {1, 4, 4, 1, 4, 4};
  size_t size_block = 2;
  std::vector<double> matrix_a = {1, 5, 9, 13};
  std::vector<double> matrix_b = {8, 1, 1, 5};
  std::vector<double> matrix_ans = {8, 1, 1, 5, 40, 5, 5, 25, 72, 9, 9, 45, 104, 13, 13, 65};

  RunTest(size_block, size, matrix_a, matrix_b, matrix_ans);
}

TEST(filatev_v_foks_all, test_error_matrix_size_b) {
  std::vector<size_t> size = {1, 4, 1, 4, 4, 4};
  size_t size_block = 2;
  std::vector<double> matrix_a = {1, 5, 9, 13};
  std::vector<double> matrix_b = {8, 1, 1, 5};

  TestError(size_block, size, matrix_a, matrix_b);
}

TEST(filatev_v_foks_all, test_error_matrix_size_c) {
  std::vector<size_t> size = {1, 4, 4, 1, 1, 1};
  size_t size_block = 2;
  std::vector<double> matrix_a = {1, 5, 9, 13};
  std::vector<double> matrix_b = {8, 1, 1, 5};

  TestError(size_block, size, matrix_a, matrix_b);
}

TEST(filatev_v_foks_all, test_error_size_block) {
  std::vector<size_t> size = {1, 4, 4, 1, 4, 4};
  size_t size_block = 0;
  std::vector<double> matrix_a = {1, 5, 9, 13};
  std::vector<double> matrix_b = {8, 1, 1, 5};

  TestError(size_block, size, matrix_a, matrix_b);
}