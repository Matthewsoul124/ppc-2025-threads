#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "all/komshina_d_image_filtering_vertical_gaussian/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(komshina_d_image_filtering_vertical_gaussian_all, EmptyImage) {
  std::size_t width = 0;
  std::size_t height = 0;
  std::vector<unsigned char> in = {};
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> expected = {};
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, ZeroWidthImage) {
  std::size_t width = 0;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> expected = {};
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, ValidationInvalidKernelSize) {
  std::size_t width = 3;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1};
  std::vector<unsigned char> out(9);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, ValidationInvalidOutputSize) {
  std::size_t width = 3;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> out(5);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, RandomImage) {
  std::size_t width = 5;
  std::size_t height = 5;
  std::vector<unsigned char> in(width * height * 3);
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> out(in.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);

  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_GE(out[i], 0);
    EXPECT_LE(out[i], 255);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, RandomImage2) {
  std::size_t width = 17;
  std::size_t height = 23;
  std::vector<unsigned char> in(width * height * 3);
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> out(in.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);

  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_GE(out[i], 0);
    EXPECT_LE(out[i], 255);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, EdgeHandling) {
  std::size_t width = 1;
  std::size_t height = 5;
  std::vector<unsigned char> in = {10, 10, 10, 50, 50, 50, 100, 100, 100, 50, 50, 50, 10, 10, 10};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> out(in.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (auto pixel : out) {
    EXPECT_GE(pixel, 0);
    EXPECT_LE(pixel, 255);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, MPI_Scatter_PartitionsCorrect) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::size_t width = 3;
  std::size_t height = 10;
  std::vector<unsigned char> in(width * height * 3, 100);
  std::vector<float> kernel(9, 1.0F);
  std::vector<unsigned char> out(in.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    for (auto val : out) {
      EXPECT_GE(val, 0);
      EXPECT_LE(val, 255);
    }
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, MPI_SingleProcess) {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    GTEST_SKIP() << "Single process test";
  }

  std::size_t width = 4;
  std::size_t height = 4;
  std::vector<unsigned char> in(width * height * 3, 128);
  std::vector<float> kernel(9, 1.0F);
  std::vector<unsigned char> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (auto val : out) {
    EXPECT_GE(val, 0);
    EXPECT_LE(val, 255);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, MPI_MoreProcessesThanRows) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 5) {
    GTEST_SKIP() << "Test requires at least 5 MPI processes";
  }

  std::size_t width = 3;
  std::size_t height = 3;
  std::vector<unsigned char> in(width * height * 3, 200);
  std::vector<float> kernel(9, 1.0F);
  std::vector<unsigned char> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (rank == 0) {
    for (auto val : out) {
      EXPECT_GE(val, 0);
      EXPECT_LE(val, 255);
    }
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, MPI_DataDistributionBranches) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "Test requires at least 2 MPI processes";
  }

  std::size_t width = 2;
  std::size_t height = 5;

  std::vector<unsigned char> in(width * height * 3);
  std::iota(in.begin(), in.end(), 0);

  std::vector<float> kernel(9, 1.0F);
  std::vector<unsigned char> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  MPI_Barrier(MPI_COMM_WORLD);
  for (auto val : out) {
    EXPECT_GE(val, 0);
    EXPECT_LE(val, 255);
  }
}