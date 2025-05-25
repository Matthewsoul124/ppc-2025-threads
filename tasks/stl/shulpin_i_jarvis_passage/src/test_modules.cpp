#include "stl/shulpin_i_jarvis_passage/include/test_modules.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <unordered_set>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/shulpin_i_jarvis_passage/include/ops_stl.hpp"

void shulpin_stl_test_module::VerifyResults(const std::vector<shulpin_i_jarvis_stl::Point> &expected,
                                            const std::vector<shulpin_i_jarvis_stl::Point> &result_tbb) {
  for (const auto &p : result_tbb) {
    bool found = false;
    for (const auto &q : expected) {
      if (std::fabs(p.x - q.x) < 1e-9 && std::fabs(p.y - q.y) < 1e-9) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }
}

void shulpin_stl_test_module::MainTestBody(std::vector<shulpin_i_jarvis_stl::Point> &input,
                                           std::vector<shulpin_i_jarvis_stl::Point> &expected) {
  std::vector<shulpin_i_jarvis_stl::Point> result_seq(expected.size());
  std::vector<shulpin_i_jarvis_stl::Point> result_tbb(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result_seq.size()));

  shulpin_i_jarvis_stl::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_tbb.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_tbb.size()));

  shulpin_i_jarvis_stl::JarvisSTLParallel stl_task(task_data_par);
  ASSERT_EQ(stl_task.Validation(), true);
  stl_task.PreProcessing();
  stl_task.Run();
  stl_task.PostProcessing();

  shulpin_stl_test_module::VerifyResults(result_seq, result_tbb);
}

std::vector<shulpin_i_jarvis_stl::Point> shulpin_stl_test_module::GeneratePointsInCircle(
    size_t num_points, const shulpin_i_jarvis_stl::Point &center, double radius) {
  std::vector<shulpin_i_jarvis_stl::Point> points;
  for (size_t i = 0; i < num_points; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points);
    double x = center.x + (radius * std::cos(angle));
    double y = center.y + (radius * std::sin(angle));
    points.emplace_back(x, y);
  }
  return points;
}

void shulpin_stl_test_module::TestBodyRandomCircle(std::vector<shulpin_i_jarvis_stl::Point> &input,
                                                   std::vector<shulpin_i_jarvis_stl::Point> &expected,
                                                   size_t &num_points) {
  std::vector<shulpin_i_jarvis_stl::Point> result_seq(expected.size());
  std::vector<shulpin_i_jarvis_stl::Point> result_tbb(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_seq.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result_seq.size()));

  shulpin_i_jarvis_stl::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_tbb.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_tbb.size()));

  shulpin_i_jarvis_stl::JarvisSTLParallel stl_task(task_data_par);
  ASSERT_EQ(stl_task.Validation(), true);
  stl_task.PreProcessing();
  stl_task.Run();
  stl_task.PostProcessing();

  shulpin_stl_test_module::VerifyResults(result_seq, result_tbb);
}

void shulpin_stl_test_module::TestBodyFalse(std::vector<shulpin_i_jarvis_stl::Point> &input,
                                            std::vector<shulpin_i_jarvis_stl::Point> &expected) {
  std::vector<shulpin_i_jarvis_stl::Point> result_tbb(expected.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_tbb.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_tbb.size()));

  shulpin_i_jarvis_stl::JarvisSTLParallel stl_task(task_data_par);
  ASSERT_EQ(stl_task.Validation(), false);
}

std::vector<shulpin_i_jarvis_stl::Point> shulpin_stl_test_module::ComputeConvexHull(
    std::vector<shulpin_i_jarvis_stl::Point> raw_points) {
  std::vector<shulpin_i_jarvis_stl::Point> convex_shell{};
  std::unordered_set<shulpin_i_jarvis_stl::Point, shulpin_i_jarvis_stl::PointHash, shulpin_i_jarvis_stl::PointEqual>
      unique_points;

  size_t most_left = 0;
  for (size_t i = 1; i < raw_points.size(); ++i) {
    if (raw_points[i].x < raw_points[most_left].x ||
        (raw_points[i].x == raw_points[most_left].x && raw_points[i].y < raw_points[most_left].y)) {
      most_left = i;
    }
  }

  const shulpin_i_jarvis_stl::Point &min_point = raw_points[most_left];
  shulpin_i_jarvis_stl::Point prev_point = min_point;
  shulpin_i_jarvis_stl::Point next_point;

  convex_shell.push_back(min_point);
  unique_points.insert(min_point);

  do {
    next_point = raw_points[0];

    for (const auto &point : raw_points) {
      if (point == prev_point) {
        continue;
      }

      double cross_product = ((point.y - prev_point.y) * (next_point.x - prev_point.x)) -
                             ((point.x - prev_point.x) * (next_point.y - prev_point.y));
      double dist1 = std::pow(point.x - prev_point.x, 2) + std::pow(point.y - prev_point.y, 2);
      double dist2 = std::pow(next_point.x - prev_point.x, 2) + std::pow(next_point.y - prev_point.y, 2);

      if (cross_product > 0 || (cross_product == 0 && dist1 > dist2)) {
        next_point = point;
      }
    }

    if (unique_points.find(next_point) == unique_points.end()) {
      convex_shell.push_back(next_point);
      unique_points.insert(next_point);
    }

    prev_point = next_point;

  } while (next_point != min_point);
  return convex_shell;
}

std::vector<shulpin_i_jarvis_stl::Point> shulpin_stl_test_module::GenerateRandomPoints(size_t num_points) {
  std::vector<shulpin_i_jarvis_stl::Point> points;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-10000, 10000);

  for (size_t i = 0; i < num_points; ++i) {
    int x = dist(gen);
    int y = dist(gen);
    points.emplace_back(static_cast<double>(x), static_cast<double>(y));
  }

  return points;
}

void shulpin_stl_test_module::RandomTestBody(std::vector<shulpin_i_jarvis_stl::Point> &input,
                                             std::vector<shulpin_i_jarvis_stl::Point> &expected) {
  std::vector<shulpin_i_jarvis_stl::Point> result_tbb(expected.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_par->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(result_tbb.data()));
  task_data_par->outputs_count.emplace_back(static_cast<uint32_t>(result_tbb.size()));

  shulpin_i_jarvis_stl::JarvisSTLParallel stl_task(task_data_par);
  ASSERT_EQ(stl_task.Validation(), true);
  stl_task.PreProcessing();
  stl_task.Run();
  stl_task.PostProcessing();

  shulpin_stl_test_module::VerifyResults(expected, result_tbb);
}

std::vector<shulpin_i_jarvis_stl::Point> shulpin_stl_test_module::PerfRandomGenerator(size_t num_points, int from,
                                                                                      int to) {
  std::vector<shulpin_i_jarvis_stl::Point> points;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(from, to);

  for (size_t i = 0; i < num_points; ++i) {
    int x = dist(gen);
    int y = dist(gen);
    points.emplace_back(static_cast<double>(x), static_cast<double>(y));
  }

  return points;
}
