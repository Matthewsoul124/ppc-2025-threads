#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace makhov_m_linear_image_filtering_vertical_seq {

class TaskSequential : public ppc::core::Task {
 public:
  explicit TaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void ApplyHorizontalGaussian(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int width, int height,
                                      const std::vector<float> &kernel);
  static void ApplyVerticalGaussian(const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int width, int height,
                                    const std::vector<float> &kernel);

 private:
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
  std::vector<float> kernel_;
  int height_;
  int width_;
  int input_size_;
};

}  // namespace makhov_m_linear_image_filtering_vertical_seq