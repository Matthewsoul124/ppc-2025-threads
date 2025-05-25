#include <cstddef>
#include <vector>

#include "stl/shulpin_i_jarvis_passage/include/ops_stl.hpp"

namespace shulpin_stl_test_module {
void VerifyResults(const std::vector<shulpin_i_jarvis_stl::Point> &expected,
                   const std::vector<shulpin_i_jarvis_stl::Point> &result_tbb);

void MainTestBody(std::vector<shulpin_i_jarvis_stl::Point> &input, std::vector<shulpin_i_jarvis_stl::Point> &expected);

std::vector<shulpin_i_jarvis_stl::Point> GeneratePointsInCircle(size_t num_points,
                                                                const shulpin_i_jarvis_stl::Point &center,
                                                                double radius);

void TestBodyRandomCircle(std::vector<shulpin_i_jarvis_stl::Point> &input,
                          std::vector<shulpin_i_jarvis_stl::Point> &expected, size_t &num_points);

void TestBodyFalse(std::vector<shulpin_i_jarvis_stl::Point> &input, std::vector<shulpin_i_jarvis_stl::Point> &expected);

int Orientation(const shulpin_i_jarvis_stl::Point &p, const shulpin_i_jarvis_stl::Point &q,
                const shulpin_i_jarvis_stl::Point &r);

std::vector<shulpin_i_jarvis_stl::Point> ComputeConvexHull(std::vector<shulpin_i_jarvis_stl::Point> raw_points);

std::vector<shulpin_i_jarvis_stl::Point> GenerateRandomPoints(size_t num_points);

void RandomTestBody(std::vector<shulpin_i_jarvis_stl::Point> &input,
                    std::vector<shulpin_i_jarvis_stl::Point> &expected);

std::vector<shulpin_i_jarvis_stl::Point> PerfRandomGenerator(size_t num_points, int from, int to);

}  // namespace shulpin_stl_test_module