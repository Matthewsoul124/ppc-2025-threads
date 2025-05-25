#include <cstddef>
#include <vector>

#include "all/shulpin_i_jarvis_passage/include/ops_all.hpp"

namespace shulpin_all_test_module {
void VerifyResults(const std::vector<shulpin_i_jarvis_all::Point> &expected,
                   const std::vector<shulpin_i_jarvis_all::Point> &result_tbb);

void MainTestBody(std::vector<shulpin_i_jarvis_all::Point> &input, std::vector<shulpin_i_jarvis_all::Point> &expected);

std::vector<shulpin_i_jarvis_all::Point> GeneratePointsInCircle(size_t num_points,
                                                                const shulpin_i_jarvis_all::Point &center,
                                                                double radius);

void TestBodyRandomCircle(std::vector<shulpin_i_jarvis_all::Point> &input,
                          std::vector<shulpin_i_jarvis_all::Point> &expected, size_t &num_points);

void TestBodyFalse(std::vector<shulpin_i_jarvis_all::Point> &input, std::vector<shulpin_i_jarvis_all::Point> &expected);

int Orientation(const shulpin_i_jarvis_all::Point &p, const shulpin_i_jarvis_all::Point &q,
                const shulpin_i_jarvis_all::Point &r);

std::vector<shulpin_i_jarvis_all::Point> ComputeConvexHull(std::vector<shulpin_i_jarvis_all::Point> raw_points);

std::vector<shulpin_i_jarvis_all::Point> GenerateRandomPoints(size_t num_points);

void RandomTestBody(std::vector<shulpin_i_jarvis_all::Point> &input,
                    std::vector<shulpin_i_jarvis_all::Point> &expected);

std::vector<shulpin_i_jarvis_all::Point> PerfRandomGenerator(size_t num_points, int from, int to);

}  // namespace shulpin_all_test_module