#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "all/titov_s_ImageFilter_HorizGaussian3x3/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace {
constexpr size_t kSmallWidth = 10;
constexpr size_t kSmallHeight = 10;
constexpr size_t kLargeWidth = 100;
constexpr size_t kLargeHeight = 100;

struct TestData {
  std::vector<double> input;
  std::vector<double> expected;
  std::vector<double> output;
  std::vector<int> kernel = {1, 2, 1};
};

TestData CreateUniformTestData(double value = 1.0) {
  TestData data;
  data.input.resize(kSmallWidth * kSmallHeight, value);
  data.expected.resize(kSmallWidth * kSmallHeight, value);
  data.output.resize(kSmallWidth * kSmallHeight);

  for (size_t i = 0; i < kSmallHeight; ++i) {
    const size_t row_start = i * kSmallWidth;
    const size_t row_end = ((i + 1) * kSmallWidth) - 1;
    data.expected[row_start] = data.expected[row_end] = value * 0.75;

    for (size_t j = 1; j < kSmallWidth - 1; ++j) {
      data.expected[row_start + j] = value;
    }
  }
  return data;
}

TestData CreateVerticalLinesTestData() {
  TestData data;
  data.input.resize(kSmallWidth * kSmallHeight);
  data.expected.resize(kSmallWidth * kSmallHeight);
  data.output.resize(kSmallWidth * kSmallHeight);

  for (size_t i = 0; i < kSmallHeight; ++i) {
    const size_t row_start = i * kSmallWidth;
    data.input[row_start + 2] = 1.0;
    data.input[row_start + 7] = 1.0;

    data.expected[row_start + 1] = 0.25;
    data.expected[row_start + 2] = 0.5;
    data.expected[row_start + 3] = 0.25;
    data.expected[row_start + 6] = 0.25;
    data.expected[row_start + 7] = 0.5;
    data.expected[row_start + 8] = 0.25;
  }
  return data;
}

TestData CreateSharpTransitionTestData() {
  TestData data;
  data.input.resize(kSmallWidth * kSmallHeight);
  data.expected.resize(kSmallWidth * kSmallHeight);
  data.output.resize(kSmallWidth * kSmallHeight);

  for (size_t i = 0; i < kSmallHeight; ++i) {
    const size_t row_start = i * kSmallWidth;
    for (size_t j = 0; j < kSmallWidth / 2; ++j) {
      data.input[row_start + j] = 0.0;
    }
    for (size_t j = kSmallWidth / 2; j < kSmallWidth; ++j) {
      data.input[row_start + j] = 1.0;
    }

    data.expected[row_start + 4] = 0.25;
    data.expected[row_start + 5] = 0.75;
    for (size_t j = 6; j < 9; ++j) {
      data.expected[row_start + j] = 1.0;
    }
    data.expected[row_start + 9] = 0.75;
  }
  return data;
}

void RunTestAndVerify(TestData& data, const boost::mpi::communicator& world) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.input.data()));
  task_data->inputs_count.emplace_back(data.input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.kernel.data()));
  task_data->inputs_count.emplace_back(data.kernel.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(data.output.data()));
  task_data->outputs_count.emplace_back(data.output.size());

  titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL filter(task_data);
  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < data.input.size(); ++i) {
      ASSERT_NEAR(data.output[i], data.expected[i], 1e-5);
    }
  }
}
}  // namespace

TEST(titov_s_image_filter_horiz_gaussian3x3_all, test_uniform) {
  boost::mpi::communicator world;
  auto test_data = CreateUniformTestData();
  RunTestAndVerify(test_data, world);
}

TEST(titov_s_image_filter_horiz_gaussian3x3_all, test_vertical_lines) {
  boost::mpi::communicator world;
  auto test_data = CreateVerticalLinesTestData();
  RunTestAndVerify(test_data, world);
}

TEST(titov_s_image_filter_horiz_gaussian3x3_all, test_sharp_transitions) {
  boost::mpi::communicator world;
  auto test_data = CreateSharpTransitionTestData();
  RunTestAndVerify(test_data, world);
}

TEST(titov_s_image_filter_horiz_gaussian3x3_all, test_empty_image) {
  boost::mpi::communicator world;
  TestData data;
  data.input.resize(kSmallWidth * kSmallHeight);
  data.expected.resize(kSmallWidth * kSmallHeight);
  data.output.resize(kSmallWidth * kSmallHeight);
  RunTestAndVerify(data, world);
}

TEST(titov_s_image_filter_horiz_gaussian3x3_all, test_random_invariant_mean) {
  boost::mpi::communicator world;
  TestData data;
  data.input.resize(kLargeWidth * kLargeHeight);
  data.output.resize(kLargeWidth * kLargeHeight);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 255.0);
  for (auto& val : data.input) {
    val = dis(gen);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.input.data()));
  task_data->inputs_count.emplace_back(data.input.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.kernel.data()));
  task_data->inputs_count.emplace_back(data.kernel.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(data.output.data()));
  task_data->outputs_count.emplace_back(data.output.size());

  titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL filter(task_data);
  ASSERT_TRUE(filter.Validation());

  filter.PreProcessing();
  filter.Run();
  filter.PostProcessing();

  const auto input_size = static_cast<double>(data.input.size());
  const auto output_size = static_cast<double>(data.output.size());
  double avg_input = std::accumulate(data.input.begin(), data.input.end(), 0.0) / input_size;
  double avg_output = std::accumulate(data.output.begin(), data.output.end(), 0.0) / output_size;

  if (world.rank() == 0) {
    ASSERT_NEAR(avg_input, avg_output, 1);
  }
}