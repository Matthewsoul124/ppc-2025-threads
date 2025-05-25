#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "all/kozlova_e_contrast_enhancement/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<uint8_t> GenerateVector(int length) {
  std::vector<uint8_t> vec(length);
  for (int i = 0; i < length; ++i) {
    vec[i] = rand() % 256;
  }
  return vec;
}

std::shared_ptr<ppc::core::TaskData> CreateTaskData(std::vector<uint8_t>& in, std::vector<uint8_t>& out, size_t width,
                                                    size_t height) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(in.size());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  return task_data;
}

void TestRun(std::vector<uint8_t> in, std::vector<uint8_t> out, size_t width, size_t height,
             const boost::mpi::communicator& world) {
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all = CreateTaskData(in, out, width, height);
  }

  kozlova_e_contrast_enhancement_all::TestTaskAll test_task_all(task_data_all);

  ASSERT_TRUE(test_task_all.Validation());
  test_task_all.PreProcessing();
  test_task_all.Run();
  test_task_all.PostProcessing();

  uint8_t min_value = *std::ranges::min_element(in);
  uint8_t max_value = *std::ranges::max_element(in);
  if (world.rank() == 0) {
    for (size_t i = 0; i < in.size(); ++i) {
      uint8_t expected = (max_value == min_value)
                             ? in[i]
                             : static_cast<uint8_t>(((in[i] - min_value) / double(max_value - min_value)) * 255);
      EXPECT_EQ(out[i], expected);
    }
  }
}

void TestValidation(std::vector<uint8_t> in, std::vector<uint8_t> out, size_t width, size_t height,
                    const boost::mpi::communicator& world) {
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all = CreateTaskData(in, out, width, height);
  }
  kozlova_e_contrast_enhancement_all::TestTaskAll test_task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_FALSE(test_task_all.Validation());
  }
}

}  // namespace

TEST(kozlova_e_contrast_enhancement_all, test_1st_image) {
  boost::mpi::communicator world;
  std::vector<uint8_t> in{10, 0, 50, 100, 200, 34};
  std::vector<uint8_t> out(6, 0);
  TestRun(in, out, 2, 3, world);
}

TEST(kozlova_e_contrast_enhancement_all, test_large_image) {
  boost::mpi::communicator world;
  std::vector<uint8_t> in = GenerateVector(400);
  std::vector<uint8_t> out(400, 0);
  TestRun(in, out, 10, 40, world);
}

TEST(kozlova_e_contrast_enhancement_all, test_empty_input) {
  boost::mpi::communicator world;
  TestValidation({}, {}, 0, 0, world);
}

TEST(kozlova_e_contrast_enhancement_all, test_same_values_input) {
  boost::mpi::communicator world;
  std::vector<uint8_t> in(6, 100);
  std::vector<uint8_t> out(6, 0);
  TestRun(in, out, 2, 3, world);
}

TEST(kozlova_e_contrast_enhancement_all, test_difference_input) {
  boost::mpi::communicator world;
  std::vector<uint8_t> in{10, 20, 30, 100, 200, 250};
  std::vector<uint8_t> out(6, 0);
  TestRun(in, out, 2, 3, world);
}

TEST(kozlova_e_contrast_enhancement_all, test_incorrect_input_size) {
  boost::mpi::communicator world;
  TestValidation({3, 3, 3}, {}, 3, 1, world);
}

TEST(kozlova_e_contrast_enhancement_all, test_incorrect_input_width) {
  boost::mpi::communicator world;
  TestValidation({3, 3, 3, 3}, {0, 0, 0, 0}, 3, 1, world);
}