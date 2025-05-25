#include "all/frolova_e_Sobel_filter/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

std::vector<int> frolova_e_sobel_filter_all::ToGrayScaleImg(std::vector<frolova_e_sobel_filter_all::RGB>& color_img,
                                                            size_t width, size_t height) {
  std::vector<int> gray_scale_image(width * height);
  for (size_t i = 0; i < width * height; i++) {
    gray_scale_image[i] =
        static_cast<int>((0.299 * color_img[i].R) + (0.587 * color_img[i].G) + (0.114 * color_img[i].B));
  }

  return gray_scale_image;
}

int frolova_e_sobel_filter_all::GetPixelSafe(const std::vector<int>& img, size_t x, size_t y, size_t width,
                                             size_t height) {
  if (x >= width || y >= height) {
    return 0;
  }
  return img[(y * width) + x];
}

void frolova_e_sobel_filter_all::ApplySobelKernel(const std::vector<int>& local_image, std::vector<int>& local_result,
                                                  int width, int extended_rows, int has_top, int local_rows) {
  tbb::parallel_for(tbb::blocked_range<int>(has_top, local_rows + has_top), [&](const tbb::blocked_range<int>& range) {
    for (int y = range.begin(); y < range.end(); ++y) {
      for (int x = 0; x < width; ++x) {
        size_t base_idx = (y - has_top) * width;

        int p00 = GetPixelSafe(local_image, x - 1, y - 1, width, extended_rows);
        int p01 = GetPixelSafe(local_image, x, y - 1, width, extended_rows);
        int p02 = GetPixelSafe(local_image, x + 1, y - 1, width, extended_rows);
        int p10 = GetPixelSafe(local_image, x - 1, y, width, extended_rows);
        int p11 = GetPixelSafe(local_image, x, y, width, extended_rows);
        int p12 = GetPixelSafe(local_image, x + 1, y, width, extended_rows);
        int p20 = GetPixelSafe(local_image, x - 1, y + 1, width, extended_rows);
        int p21 = GetPixelSafe(local_image, x, y + 1, width, extended_rows);
        int p22 = GetPixelSafe(local_image, x + 1, y + 1, width, extended_rows);

        int res_x = (-1 * p00) + (0 * p01) + (1 * p02) + (-2 * p10) + (0 * p11) + (2 * p12) + (-1 * p20) + (0 * p21) +
                    (1 * p22);

        int res_y = (-1 * p00) + (-2 * p01) + (-1 * p02) + (0 * p10) + (0 * p11) + (0 * p12) + (1 * p20) + (2 * p21) +
                    (1 * p22);

        int gradient = static_cast<int>(std::sqrt((res_x * res_x) + (res_y * res_y)));
        local_result[base_idx + x] = std::clamp(gradient, 0, 255);
      }
    }
  });
}

void frolova_e_sobel_filter_all::InitWorkArea(int active_processes, int rows_per_proc, int remainder, int rank,
                                              int& y_start, int& local_rows, int& has_top, int& has_bottom,
                                              int& extended_rows) {
  y_start = (rank * rows_per_proc) + std::min(rank, remainder);
  int y_end = y_start + rows_per_proc + (rank < remainder ? 1 : 0);
  local_rows = y_end - y_start;

  has_top = (rank > 0) ? 1 : 0;
  has_bottom = (rank < active_processes - 1) ? 1 : 0;
  extended_rows = local_rows + has_top + has_bottom;
}

void frolova_e_sobel_filter_all::InitProcWorkArea(int proc, int active_processes, int rows_per_proc, int remainder,
                                                  int& y_start, int& local_rows, int& has_top, int& has_bottom,
                                                  int& extended_rows) {
  y_start = (proc * rows_per_proc) + std::min(proc, remainder);
  int y_end = y_start + rows_per_proc + (proc < remainder ? 1 : 0);
  local_rows = y_end - y_start;

  has_top = (proc > 0) ? 1 : 0;
  has_bottom = (proc < active_processes - 1) ? 1 : 0;
  extended_rows = local_rows + has_top + has_bottom;
}

void frolova_e_sobel_filter_all::SobelFilterALL::CollectWorkerResults(const std::vector<int>& local_result,
                                                                      int rows_per_proc, int remainder,
                                                                      int active_processes) {
  std::ranges::copy(local_result,
                    res_image_.begin() + static_cast<std::vector<int>::difference_type>(y_start_ * width_));

  for (int proc = 1; proc < active_processes; ++proc) {
    int proc_y_start = (proc * rows_per_proc) + std::min(proc, remainder);
    int proc_y_end = proc_y_start + rows_per_proc + (proc < remainder ? 1 : 0);
    int proc_local_rows = proc_y_end - proc_y_start;

    std::vector<int> proc_result(proc_local_rows * width_);
    world_.recv(proc, 1, proc_result);

    using DiffT = std::vector<int>::difference_type;
    std::ranges::copy(proc_result, res_image_.begin() + static_cast<DiffT>(proc_y_start * width_));
  }
}

void frolova_e_sobel_filter_all::SobelFilterALL::CopyOrZeroLine(std::vector<int>& chunk, const std::vector<int>& gray,
                                                                int i, int proc_y_start, int top, int width,
                                                                int height) {
  int src_y = proc_y_start - top + i;
  auto dst_offset = static_cast<std::vector<uint8_t>::difference_type>(width) * i;

  if (src_y >= 0 && src_y < height) {
    auto src_offset = static_cast<std::vector<uint8_t>::difference_type>(width) * src_y;
    std::copy_n(gray.begin() + src_offset, width, chunk.begin() + dst_offset);
  } else {
    std::fill_n(chunk.begin() + dst_offset, width, 0);
  }
}

bool frolova_e_sobel_filter_all::SobelFilterALL::PreProcessingImpl() {
  // Init value for input and output
  if (world_.rank() == 0) {
    int* value_1 = reinterpret_cast<int*>(task_data->inputs[0]);
    width_ = static_cast<size_t>(value_1[0]);
    height_ = static_cast<size_t>(value_1[1]);

    int* value_2 = reinterpret_cast<int*>(task_data->inputs[1]);
    std::vector<int> picture_vector;
    picture_vector.assign(value_2, value_2 + task_data->inputs_count[1]);

    for (size_t i = 0; i < picture_vector.size(); i += 3) {
      RGB pixel;
      pixel.R = picture_vector[i];
      pixel.G = picture_vector[i + 1];
      pixel.B = picture_vector[i + 2];
      picture_.push_back(pixel);
    }
    grayscale_image_ = frolova_e_sobel_filter_all::ToGrayScaleImg(picture_, width_, height_);

    res_image_.resize(width_ * height_);
  }

  return true;
}

bool frolova_e_sobel_filter_all::SobelFilterALL::ValidationImpl() {
  // Check equality of counts elements
  if (world_.rank() == 0) {
    int* value_1 = reinterpret_cast<int*>(task_data->inputs[0]);

    if (task_data->inputs_count[0] != 2) {
      return false;
    }

    if (value_1[0] <= 0 || value_1[1] <= 0) {
      return false;
    }

    auto width_1 = static_cast<size_t>(value_1[0]);
    auto height_1 = static_cast<size_t>(value_1[1]);

    int* value_2 = reinterpret_cast<int*>(task_data->inputs[1]);
    std::vector<int> picture_vector(value_2, value_2 + task_data->inputs_count[1]);
    if (task_data->inputs_count[1] != width_1 * height_1 * 3) {
      return false;
    }

    for (size_t i = 0; i < picture_vector.size(); i++) {
      if (picture_vector[i] < 0 || picture_vector[i] > 255) {
        return false;
      }
    }
  }
  return true;
}

bool frolova_e_sobel_filter_all::SobelFilterALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  broadcast(world_, height_, 0);
  broadcast(world_, width_, 0);

  int active_processes = std::min(size, static_cast<int>(height_));

  if (rank < active_processes) {
    int rows_per_proc = static_cast<int>(height_ / active_processes);
    int remainder = static_cast<int>(height_ % active_processes);

    InitWorkArea(active_processes, rows_per_proc, remainder, rank, y_start_, local_rows_, has_top_, has_bottom_,
                 extended_rows_);

    local_image_.resize(extended_rows_ * width_);
    std::vector<int> local_result(local_rows_ * width_);

    if (rank == 0) {
      std::vector<int> gray = grayscale_image_;
      for (int proc = 0; proc < active_processes; proc++) {
        int proc_y_start = 0;
        int proc_local_rows = 0;
        int top = 0;
        int bottom = 0;
        int ext_rows = 0;

        InitProcWorkArea(proc, active_processes, rows_per_proc, remainder, proc_y_start, proc_local_rows, top, bottom,
                         ext_rows);

        std::vector<int> chunk(ext_rows * width_, 0);
        for (int i = 0; i < ext_rows; ++i) {
          CopyOrZeroLine(chunk, gray, i, proc_y_start, top, static_cast<int>(width_), static_cast<int>(height_));
        }

        if (proc == 0) {
          local_image_ = chunk;
        } else {
          world_.send(proc, 0, chunk);
        }
      }
    } else {
      world_.recv(0, 0, local_image_);
    }

    frolova_e_sobel_filter_all::ApplySobelKernel(local_image_, local_result, static_cast<int>(width_), extended_rows_,
                                                 has_top_, local_rows_);

    if (world_.rank() == 0) {
      CollectWorkerResults(local_result, rows_per_proc, remainder, active_processes);
    } else {
      world_.send(0, 1, local_result);
    }
  }

  return true;
}

bool frolova_e_sobel_filter_all::SobelFilterALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < width_ * height_; i++) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = res_image_[i];
    }
  }

  return true;
}
