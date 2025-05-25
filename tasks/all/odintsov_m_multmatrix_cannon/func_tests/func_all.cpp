#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/odintsov_m_multmatrix_cannon/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace odintsov_m_mulmatrix_cannon_all {
namespace {
static std::vector<double> GenerateMatrix(int sz);
static std::vector<double> MultiplyMatrices(const std::vector<double> &a, const std::vector<double> &b, int n);

std::vector<double> GenerateMatrix(int sz) {
  std::vector<double> matrix(sz * sz);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);
  for (int i = 0; i < sz; ++i) {
    for (int j = 0; j < sz; ++j) {
      matrix[(i * sz) + j] = dis(gen);
    }
  }
  return matrix;
}

std::vector<double> MultiplyMatrices(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      c[(i * n) + j] = sum;
    }
  }
  return c;
}
}  // namespace
}  // namespace odintsov_m_mulmatrix_cannon_all

TEST(odintsov_m_mulmatrix_cannon_all, test_matrix_4) {
  boost::mpi::communicator com;
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_all::GenerateMatrix(4);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_all::GenerateMatrix(4);
  std::vector<double> out_all(16, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_all::MultiplyMatrices(matrix_a, matrix_b, 4);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->inputs_count.emplace_back(matrix_b.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  }

  odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);

  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();
  if (com.rank() == 0) {
    ASSERT_EQ(out_ans.size(), out_all.size());
    for (size_t i = 0; i < out_ans.size(); ++i) {
      EXPECT_NEAR(out_ans[i], out_all[i], 0.00001);
    }
  }
}
TEST(odintsov_m_mulmatrix_cannon_all, not_random_matrix) {
  boost::mpi::communicator com;
  std::vector<double> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> out_all(16, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_all::MultiplyMatrices(matrix_a, matrix_b, 4);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->inputs_count.emplace_back(matrix_b.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  }
  odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);

  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (com.rank() == 0) {
    ASSERT_EQ(out_ans.size(), out_all.size());
    for (size_t i = 0; i < out_ans.size(); ++i) {
      EXPECT_NEAR(out_ans[i], out_all[i], 0.00001);
    }
  }
}

TEST(odintsov_m_mulmatrix_cannon_all, test_matrix_100) {
  boost::mpi::communicator com;
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_all::GenerateMatrix(10);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_all::GenerateMatrix(10);
  std::vector<double> out_all(100, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_all::MultiplyMatrices(matrix_a, matrix_b, 10);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->inputs_count.emplace_back(matrix_b.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  }
  odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);

  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (com.rank() == 0) {
    ASSERT_EQ(out_ans.size(), out_all.size());
    for (size_t i = 0; i < out_ans.size(); ++i) {
      EXPECT_NEAR(out_ans[i], out_all[i], 0.00001);
    }
  }
}

TEST(odintsov_m_mulmatrix_cannon_all, test_matrix_900) {
  boost::mpi::communicator com;
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_all::GenerateMatrix(30);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_all::GenerateMatrix(30);
  std::vector<double> out_all(900, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_all::MultiplyMatrices(matrix_a, matrix_b, 30);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->inputs_count.emplace_back(matrix_b.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  }
  odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);

  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (com.rank() == 0) {
    ASSERT_EQ(out_ans.size(), out_all.size());
    for (size_t i = 0; i < out_ans.size(); ++i) {
      EXPECT_NEAR(out_ans[i], out_all[i], 0.00001);
    }
  }
}

TEST(odintsov_m_mulmatrix_cannon_all, test_block_sz_1) {
  boost::mpi::communicator com;
  std::vector<double> matrix_a = odintsov_m_mulmatrix_cannon_all::GenerateMatrix(3);
  std::vector<double> matrix_b = odintsov_m_mulmatrix_cannon_all::GenerateMatrix(3);
  std::vector<double> out_all(9, 0);
  std::vector<double> out_ans = odintsov_m_mulmatrix_cannon_all::MultiplyMatrices(matrix_a, matrix_b, 3);

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->inputs_count.emplace_back(matrix_b.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_all.data()));
  }
  odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL test_task_all(task_data_all);

  ASSERT_EQ(test_task_all.Validation(), true);

  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  if (com.rank() == 0) {
    ASSERT_EQ(out_ans.size(), out_all.size());
    for (size_t i = 0; i < out_ans.size(); ++i) {
      EXPECT_NEAR(out_ans[i], out_all[i], 0.00001);
    }
  }
}
