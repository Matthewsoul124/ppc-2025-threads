#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/zinoviev_a_convex_hull_components/include/ops_stl.hpp"

namespace {
void SetupTaskData(const std::vector<int>& input, int width, int height, size_t output_size,
                   std::shared_ptr<ppc::core::TaskData>& task_data) {
  task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(input.data())));
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->outputs.emplace_back(
      reinterpret_cast<uint8_t*>(new zinoviev_a_convex_hull_components_stl::Point[output_size]));
  task_data->outputs_count.emplace_back(output_size);
}

void VerifyResult(const std::vector<zinoviev_a_convex_hull_components_stl::Point>& actual,
                  const std::vector<zinoviev_a_convex_hull_components_stl::Point>& expected) {
  auto sorted_actual = actual;
  auto sorted_expected = expected;

  auto point_comparator = [](const zinoviev_a_convex_hull_components_stl::Point& a,
                             const zinoviev_a_convex_hull_components_stl::Point& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  };

  std::ranges::sort(sorted_actual, point_comparator);
  std::ranges::sort(sorted_expected, point_comparator);

  ASSERT_EQ(sorted_actual.size(), sorted_expected.size());
  for (size_t i = 0; i < sorted_actual.size(); ++i) {
    ASSERT_EQ(sorted_actual[i].x, sorted_expected[i].x);
    ASSERT_EQ(sorted_actual[i].y, sorted_expected[i].y);
  }
}

void RunAndValidate(const std::vector<int>& input,
                    const std::vector<zinoviev_a_convex_hull_components_stl::Point>& expected, int width, int height) {
  std::shared_ptr<ppc::core::TaskData> task_data;
  SetupTaskData(input, width, height, expected.size(), task_data);

  zinoviev_a_convex_hull_components_stl::ConvexHullSTL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto* output = reinterpret_cast<zinoviev_a_convex_hull_components_stl::Point*>(task_data->outputs[0]);
  std::vector<zinoviev_a_convex_hull_components_stl::Point> actual(output, output + expected.size());

  VerifyResult(actual, expected);
  delete[] reinterpret_cast<zinoviev_a_convex_hull_components_stl::Point*>(task_data->outputs[0]);
}
}  // namespace

TEST(zinoviev_a_convex_hull_components_stl, SquareShape) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  const std::vector<int> input = {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_stl::Point> expected = {
      {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 3, .y = 4}, {.x = 0, .y = 4}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_stl, TriangleShape) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  const std::vector<int> input = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_stl::Point> expected = {
      {.x = 0, .y = 0}, {.x = 2, .y = 2}, {.x = 0, .y = 4}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_stl, SinglePixel) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  std::vector<int> input(25, 0);
  input[12] = 1;
  const std::vector<zinoviev_a_convex_hull_components_stl::Point> expected{{.x = 2, .y = 2}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_stl, EmptyImageTest) {
  constexpr int kWidth = 10;
  constexpr int kHeight = 10;
  const std::vector<int> input(100, 0);
  const std::vector<zinoviev_a_convex_hull_components_stl::Point> expected;
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_stl, CrossShapeTest) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  const std::vector<int> input = {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_stl::Point> expected{
      {.x = 0, .y = 2}, {.x = 4, .y = 2}, {.x = 2, .y = 0}, {.x = 2, .y = 4}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_stl, SmallRectangleTest) {
  constexpr int kWidth = 3;
  constexpr int kHeight = 3;
  const std::vector<int> input = {1, 1, 1, 1, 0, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_stl::Point> expected{
      {.x = 0, .y = 0}, {.x = 2, .y = 0}, {.x = 1, .y = 2}, {.x = 0, .y = 2}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_stl, DiamondShapeTest) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  const std::vector<int> input = {0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0};
  const std::vector<zinoviev_a_convex_hull_components_stl::Point> expected{
      {.x = 0, .y = 2}, {.x = 4, .y = 2}, {.x = 2, .y = 0}, {.x = 2, .y = 4}};
  RunAndValidate(input, expected, kWidth, kHeight);
}

TEST(zinoviev_a_convex_hull_components_stl, RightTriangleTest) {
  constexpr int kWidth = 4;
  constexpr int kHeight = 4;
  const std::vector<int> input = {1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1};
  const std::vector<zinoviev_a_convex_hull_components_stl::Point> expected{
      {.x = 0, .y = 0}, {.x = 0, .y = 3}, {.x = 3, .y = 3}};
  RunAndValidate(input, expected, kWidth, kHeight);
}