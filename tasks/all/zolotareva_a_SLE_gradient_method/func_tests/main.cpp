#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"
#include "core/task/include/task.hpp"

void zolotareva_a_sle_gradient_method_all::GenerateSle(std::vector<double> &a, std::vector<double> &b, int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (int i = 0; i < n; ++i) {
    b[i] = dist(gen);
    for (int j = i; j < n; ++j) {
      double value = dist(gen);
      a[(i * n) + j] = value;
      a[(j * n) + i] = value;
    }
  }

  for (int i = 0; i < n; ++i) {
    a[(i * n) + i] += n * 100.0;
  }
}

namespace {
void Form(int n) {
  boost::mpi::communicator world;
  std::vector<double> a(n * n);
  std::vector<double> b(n);
  std::vector<double> x(n);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    zolotareva_a_sle_gradient_method_all::GenerateSle(a, b, n);
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(n);
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(x.size());
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += a[(i * n) + j] * x[j];
      }
      EXPECT_NEAR(sum, b[i], 1e-4);
    }
  }
}
}  // namespace

TEST(zolotareva_a_sle_gradient_method_all, invalid_input_sizes) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> a = {2, -1, -1, 2};
  std::vector<double> b = {1, 3, 4};  // Неправильный размер b
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(b.size());
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(n);
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.ValidationImpl());
  } else {
    ASSERT_TRUE(task.ValidationImpl());
  }
}

TEST(zolotareva_a_sle_gradient_method_all, non_symmetric_matrix) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> a = {2, -1, 0, 2};  // a[0][1] != a[1][0]
  std::vector<double> b = {1, 3};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(n);
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(n);
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.ValidationImpl());
  } else {
    ASSERT_TRUE(task.ValidationImpl());
  }
}

TEST(zolotareva_a_sle_gradient_method_all, not_positive_definite_matrix) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> a = {0, 0, 0, 0};
  std::vector<double> b = {0, 0};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(n);
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(n);
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.ValidationImpl());
  } else {
    ASSERT_TRUE(task.ValidationImpl());
  }
}

TEST(zolotareva_a_sle_gradient_method_all, negative_definite_matrix) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> a = {-1, 0, 0, -2};
  std::vector<double> b = {1, 1};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(n);
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(n);
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.ValidationImpl());
  } else {
    ASSERT_TRUE(task.ValidationImpl());
  }
}

TEST(zolotareva_a_sle_gradient_method_all, zero_dimension) {
  boost::mpi::communicator world;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> x;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs_count.push_back(0);
    task_data_seq->inputs_count.push_back(0);
    task_data_seq->outputs_count.push_back(0);
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.ValidationImpl());
  } else {
    ASSERT_TRUE(task.ValidationImpl());
  }
}

TEST(zolotareva_a_sle_gradient_method_all, singular_matrix) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> a = {1, 1, 1, 1};  // Сингулярная матрица
  std::vector<double> b = {2, 2};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(n);
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(n);
  }

  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  if (world.rank() == 0) {
    ASSERT_FALSE(task.ValidationImpl());
  } else {
    ASSERT_TRUE(task.ValidationImpl());
  }
}

TEST(zolotareva_a_sle_gradient_method_all, zero_vector_solution) {
  boost::mpi::communicator world;
  int n = 2;
  std::vector<double> a = {1, 0, 0, 1};
  std::vector<double> b = {0, 0};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(n);
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(n);
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(x[i], 0.0, 1e-2);  // Ожидаем нулевой вектор решения
    }
  }
}

TEST(zolotareva_a_sle_gradient_method_all, n_equals_one) {
  boost::mpi::communicator world;
  int n = 1;
  std::vector<double> a = {2};
  std::vector<double> b = {4};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(n);
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(n);
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_NEAR(x[0], 2.0, 1e-1);  // Ожидаемое решение x = 2
  }
}

TEST(zolotareva_a_sle_gradient_method_all, test_correct_answer1) {
  boost::mpi::communicator world;
  int n = 3;
  std::vector<double> a = {4, -1, 2, -1, 6, -2, 2, -2, 5};
  std::vector<double> b = {-1, 9, -10};
  std::vector<double> x;
  x.resize(n);
  std::vector<double> ref_x = {1, 1, -2};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data_seq->inputs_count.push_back(n * n);
    task_data_seq->inputs_count.push_back(n);
    task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    task_data_seq->outputs_count.push_back(x.size());
  }
  zolotareva_a_sle_gradient_method_all::TestTaskALL task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(x[i], ref_x[i], 1e-12);
    }
  }
}
TEST(zolotareva_a_sle_gradient_method_all, Test_Image_random_n_3) { Form(3); };
TEST(zolotareva_a_sle_gradient_method_all, Test_Image_random_n_10) { Form(5); };
TEST(zolotareva_a_sle_gradient_method_all, Test_Image_random_n_200) { Form(200); };
TEST(zolotareva_a_sle_gradient_method_all, Test_Image_random_n_591) { Form(591); };
