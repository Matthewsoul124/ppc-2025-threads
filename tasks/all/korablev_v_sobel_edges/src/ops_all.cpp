#include "all/korablev_v_sobel_edges/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/communicator.hpp"

// clang-format off
constexpr int8_t kSobelKernelX[3][3] = {
  {-1, 0, 1},
  {-2, 0, 2},
  {-1, 0, 1}
};
constexpr int8_t kSobelKernelY[3][3] = {
  {-1, -2, -1},
  { 0,  0,  0},
  { 1,  2,  1}
};
// clang-format on

void korablev_v_sobel_edges_all::Image::SetDimensions(std::size_t w,  // NOLINT(bugprone-easily-swappable-parameters)
                                                      std::size_t h) {
  width = w;
  height = h;
  data.resize(width * height * kPixelChannels);
}
void korablev_v_sobel_edges_all::Image::CopyFrom(const uint8_t* buf) {
  data.assign(buf, buf + (width * height * kPixelChannels));
}

bool korablev_v_sobel_edges_all::TestTask::ValidationImpl() {
  return world_.rank() != 0 || (task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 &&
                                task_data->outputs_count[0] ==
                                    (task_data->inputs_count[0] * task_data->inputs_count[1] * Image::kPixelChannels));
}

bool korablev_v_sobel_edges_all::TestTask::PreProcessingImpl() {
  if (world_.rank() == 0) {
    in_.SetDimensions(task_data->inputs_count[0], task_data->inputs_count[1]);
    in_.CopyFrom(task_data->inputs[0]);
    out_.SetDimensions(task_data->inputs_count[0], task_data->inputs_count[1]);
  }
  return true;
}

bool korablev_v_sobel_edges_all::TestTask::RunImpl() {  // NOLINT(readability-function-cognitive-complexity)
  int width = static_cast<int>(in_.width);
  int total_height = static_cast<int>(in_.height);

  boost::mpi::broadcast(world_, width, 0);
  boost::mpi::broadcast(world_, total_height, 0);

  const auto elements_in_row = width * Image::kPixelChannels;

  const int numproc = std::min(total_height, world_.size());
  if (world_.rank() >= numproc) {
    world_.split(1);
  }
  world_.split(0);

  const int rows_per_process = total_height / numproc;
  const int unacc_rows = total_height % numproc;

  std::vector<int> sendcounts(numproc, 0);
  std::vector<int> senddispls(numproc, 0);
  std::vector<int> recvcounts(numproc, 0);
  std::vector<int> recvdispls(numproc, 0);

  for (int i = 0; i < numproc; i++) {
    const int prows = rows_per_process + ((i < unacc_rows) ? 1 : 0);
    sendcounts[i] = prows;
    recvcounts[i] = prows;
  }
  if (numproc > 1) {
    std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, senddispls.begin() + 1, std::plus<>());
    std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, recvdispls.begin() + 1, std::plus<>());
  }
  {
    for (int i = 1; i < numproc - 1; i++) {
      sendcounts[i] += 2;
    }
    if (numproc > 1) {
      sendcounts.front() += 1;
      sendcounts.back() += 1;
    }

    for (int i = 1; i < numproc; i++) {
      senddispls[i] -= 1;
    }
  }

  const auto height = sendcounts[world_.rank()];

  for (int i = 0; i < numproc; i++) {
    sendcounts[i] *= elements_in_row;
    senddispls[i] *= elements_in_row;
    recvcounts[i] *= elements_in_row;
    recvdispls[i] *= elements_in_row;
  }

  std::vector<uint8_t> partial_image(sendcounts[world_.rank()]);
  std::vector<uint8_t> partial_out(sendcounts[world_.rank()]);
  boost::mpi::scatterv(world_, in_.data, sendcounts, senddispls, partial_image.data(), int(partial_image.size()), 0);

#pragma omp parallel for
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      std::array<int32_t, 3> sum_x{0};
      std::array<int32_t, 3> sum_y{0};

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int idx = ((y + ky) * width + (x + kx)) * 3;  // NOLINT(bugprone-narrowing-conversions)
          for (int j = 0; j < 3; j++) {
            const uint8_t part_image = partial_image[idx + j];
            sum_x[j] += kSobelKernelX[ky + 1][kx + 1] * part_image;
            sum_y[j] += kSobelKernelY[ky + 1][kx + 1] * part_image;
          }
        }
      }

      for (int i = 0; i < 3; ++i) {
        partial_out[((y * width + x) * 3) + i] = static_cast<uint8_t>(
            std::min(static_cast<int32_t>(std::sqrt((sum_x[i] * sum_x[i]) + (sum_y[i] * sum_y[i]))), 255));
      }
    }
  }

  boost::mpi::gatherv(world_, partial_out.data() + (world_.rank() == 0 ? 0 : elements_in_row),
                      recvcounts[world_.rank()], out_.data.data(), recvcounts, recvdispls, 0);

  return true;
}

bool korablev_v_sobel_edges_all::TestTask::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(out_.data, task_data->outputs[0]);
  }
  return true;
}
