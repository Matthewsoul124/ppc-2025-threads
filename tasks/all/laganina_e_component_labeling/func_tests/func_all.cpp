#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/laganina_e_component_labeling/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(laganina_e_component_labeling_all, validation_test1) {
  boost::mpi::communicator world;
  int m = 0;
  int n = 1;
  // Create data
  std::vector<uint32_t> in(m * n, 0);
  std::vector<uint32_t> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
    // Create Task..
    laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
    ASSERT_EQ(test_task_omp.ValidationImpl(), false);
  }
}

TEST(laganina_e_component_labeling_all, validation_test4) {
  boost::mpi::communicator world;
  int m = 1;
  int n = 0;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  // Create task_data 2
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
    task_data_omp->outputs_count.emplace_back(n);
    // Create Task
    laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
    ASSERT_EQ(test_task_omp.ValidationImpl(), false);
  }
}

TEST(laganina_e_component_labeling_all, validation_test2) {
  boost::mpi::communicator world;
  int m = 0;
  int n = 0;
  // Create data
  std::vector<int> in(m * n, 3);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);

    // Create Task
    laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
    ASSERT_EQ(test_task_omp.ValidationImpl(), false);
  }
}

TEST(laganina_e_component_labeling_all, validation_test3) {
  boost::mpi::communicator world;
  int m = 3;
  int n = 1;
  // Create data
  std::vector<int> in(m * n, 3);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);

    // Create Task
    laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
    ASSERT_EQ(test_task_omp.ValidationImpl(), false);
  }
}

TEST(laganina_e_component_labeling_all, Find_test) {
  boost::mpi::communicator world;
  int m = 3;
  int n = 3;
  // Create data
  std::vector<int> in = {1, 0, 1, 1, 1, 0, 0, 1, 1};
  std::vector<int> out(m * n);
  std::vector<int> exp_out = {1, 0, 2, 1, 1, 0, 0, 1, 1};
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }
  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out, exp_out);
  }
}

TEST(laganina_e_component_labeling_all, all_one) {
  boost::mpi::communicator world;
  int m = 3;
  int n = 2;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  std::vector<int> res(m * n, 1);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }
  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(out, res);
  }
}

TEST(laganina_e_component_labeling_all, all_one_large) {
  boost::mpi::communicator world;
  int m = 100;
  int n = 1000;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  std::vector<int> res(m * n, 1);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }
  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(out, res);
  }
}

TEST(laganina_e_component_labeling_all, all_zero) {
  boost::mpi::communicator world;
  int m = 3;
  int n = 2;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }
  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(in, out);
  }
}

TEST(laganina_e_component_labeling_all, test1) {
  boost::mpi::communicator world;
  int m = 3;
  int n = 3;
  // Create data
  std::vector<int> in = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<int> exp_out = {1, 0, 2, 0, 3, 0, 4, 0, 5};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, test2) {
  boost::mpi::communicator world;
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }
  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, test3) {
  boost::mpi::communicator world;
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 0, 1, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }
  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, test6) {
  boost::mpi::communicator world;
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 0, 1, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, test7) {
  boost::mpi::communicator world;
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 0, 0, 2, 1, 1, 1, 0, 2, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, test4) {
  boost::mpi::communicator world;
  int m = 4;
  int n = 5;
  // Create data
  std::vector<int> in = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<int> exp_out = {1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }
  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, test5) {
  boost::mpi::communicator world;
  int m = 3;
  int n = 3;
  // Create data
  std::vector<int> in = {1, 1, 1, 1, 0, 1, 1, 1, 1};
  std::vector<int> exp_out = {1, 1, 1, 1, 0, 1, 1, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, all_one_100) {
  boost::mpi::communicator world;
  int m = 100;
  int n = 100;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 1);

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }
  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, all_one_300) {
  boost::mpi::communicator world;
  int m = 300;
  int n = 300;
  // Create data
  std::vector<int> in(m * n, 1);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 1);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, simple_rectangles_100) {
  boost::mpi::communicator world;
  int m = 100;
  int n = 100;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 0);
  for (int i = 10; i < 40; ++i) {
    for (int j = 20; j < 60; ++j) {
      in[(i * n) + j] = 1;
    }
  }

  for (int i = 60; i < 90; ++i) {
    for (int j = 50; j < 80; ++j) {
      in[(i * n) + j] = 1;
    }
  }

  for (int i = 10; i < 40; ++i) {
    for (int j = 20; j < 60; ++j) {
      exp_out[(i * n) + j] = 1;
    }
  }

  for (int i = 60; i < 90; ++i) {
    for (int j = 50; j < 80; ++j) {
      exp_out[(i * n) + j] = 2;
    }
  }

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, diagonal_line_100) {
  boost::mpi::communicator world;
  int m = 100;
  int n = 100;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 0);
  for (int i = 0; i < n; i += 2) {
    in[(i * n) + i] = 1;
  }
  for (int i = 0; i < n; i += 2) {
    exp_out[(i * n) + i] = 1 + (i / 2);
  }

  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, u_shaped_shape_100) {
  boost::mpi::communicator world;
  int m = 100;
  int n = 100;
  // Create data data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 0);
  for (int i = 10; i < 90; ++i) {
    for (int j = 10; j < 90; ++j) {
      in[(i * n) + j] = 1;
    }
  }
  for (int i = 40; i < 60; ++i) {
    for (int j = 40; j < 60; ++j) {
      in[(i * n) + j] = 0;
    }
  }

  for (int i = 10; i < 90; ++i) {
    for (int j = 10; j < 90; ++j) {
      exp_out[(i * n) + j] = 1;
    }
  }

  for (int i = 40; i < 60; ++i) {
    for (int j = 40; j < 60; ++j) {
      exp_out[(i * n) + j] = 0;
    }
  }
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(laganina_e_component_labeling_all, spiral_pattern) {
  boost::mpi::communicator world;
  int m = 10;
  int n = 10;
  std::vector<int> in = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1,
                         1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> out(m * n, 0);
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(in, out);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST(laganina_e_component_labeling_all, ring_with_a_hole_100) {
  boost::mpi::communicator world;
  int m = 100;
  int n = 100;
  // Create data
  std::vector<int> in(m * n, 0);
  std::vector<int> out(m * n, 0);
  std::vector<int> exp_out(m * n, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i < 10 || i >= 90 || j < 10 || j >= 90) {
        in[(i * n) + j] = 1;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i < 10 || i >= 90 || j < 10 || j >= 90) {
        exp_out[(i * n) + j] = 1;
      }
    }
  }
  // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_omp->inputs_count.emplace_back(m);
    task_data_omp->inputs_count.emplace_back(n);
    task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_omp->outputs_count.emplace_back(m);
    task_data_omp->outputs_count.emplace_back(n);
  }

  // Create Task
  laganina_e_component_labeling_all::TestTaskALL test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}