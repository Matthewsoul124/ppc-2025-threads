#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace frolova_e_sobel_filter_all {

struct RGB {
  int R{};
  int G{};
  int B{};
};

std::vector<int> ToGrayScaleImg(std::vector<RGB>& color_img, size_t width, size_t height);

int GetPixelSafe(const std::vector<int>& img, size_t x, size_t y, size_t width, size_t height);

void ApplySobelKernel(const std::vector<int>& local_image, std::vector<int>& local_result, int width, int extended_rows,
                      int has_top, int local_rows);

void InitWorkArea(int active_processes, int rows_per_proc, int remainder, int rank, int& y_start, int& local_rows,
                  int& has_top, int& has_bottom, int& extended_rows);

void InitProcWorkArea(int proc, int active_processes, int rows_per_proc, int remainder, int& y_start, int& local_rows,
                      int& has_top, int& has_bottom, int& extended_rows);

class SobelFilterALL : public ppc::core::Task {
 public:
  explicit SobelFilterALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void CollectWorkerResults(const std::vector<int>& local_result, int rows_per_proc, int remainder,
                            int active_processes);
  static void CopyOrZeroLine(std::vector<int>& chunk, const std::vector<int>& gray, int i, int proc_y_start, int top,
                             int width, int height);

 private:
  std::vector<RGB> picture_;
  std::vector<int> grayscale_image_;
  size_t width_{};
  size_t height_{};
  std::vector<int> res_image_;
  boost::mpi::communicator world_;

  std::vector<int> local_image_;
  std::vector<int> local_result_;

  int y_start_;
  int local_rows_;
  int has_top_;
  int has_bottom_;
  int extended_rows_;
};

}  // namespace frolova_e_sobel_filter_all
