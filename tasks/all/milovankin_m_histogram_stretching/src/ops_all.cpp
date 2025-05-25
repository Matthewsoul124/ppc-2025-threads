#include "../include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_reduce.h"

namespace milovankin_m_histogram_stretching_all {

struct MinMaxPair;
namespace {
MinMaxPair CalculateLocalMinMax(const std::vector<uint8_t>& img, std::size_t start_idx, std::size_t end_idx);
void ApplyStretchingLocal(std::vector<uint8_t>& img, std::size_t start_idx, std::size_t end_idx, uint8_t global_min,
                          uint8_t global_max);
void GatherResults(boost::mpi::communicator& world, std::vector<uint8_t>& img, std::size_t img_size,
                   std::size_t chunk_size, std::size_t remainder);
}  // namespace

struct MinMaxPair {
  uint8_t min_val;
  uint8_t max_val;

  MinMaxPair() : min_val(std::numeric_limits<uint8_t>::max()), max_val(0) {}

  MinMaxPair(uint8_t min, uint8_t max) : min_val(min), max_val(max) {}

  template <class Archive>
  void serialize(Archive& ar, const unsigned int) {  // NOLINT
    ar & min_val;
    ar & max_val;
  }
};

namespace {
MinMaxPair CalculateLocalMinMax(const std::vector<uint8_t>& img, std::size_t start_idx, std::size_t end_idx) {
  const std::size_t grain_size = std::max(std::size_t(16), (end_idx - start_idx) / 16);
  return tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(start_idx, end_idx, grain_size), MinMaxPair(),
      [&img](const tbb::blocked_range<std::size_t>& range, MinMaxPair init) -> MinMaxPair {
        uint8_t local_min = init.min_val;
        uint8_t local_max = init.max_val;
        for (std::size_t i = range.begin(); i != range.end(); ++i) {
          uint8_t val = img[i];
          local_min = std::min(val, local_min);
          local_max = std::max(val, local_max);
        }
        return {local_min, local_max};
      },
      [](MinMaxPair a, MinMaxPair b) -> MinMaxPair {
        return {std::min(a.min_val, b.min_val), std::max(a.max_val, b.max_val)};
      });
}

void ApplyStretchingLocal(std::vector<uint8_t>& img, std::size_t start_idx, std::size_t end_idx, uint8_t global_min,
                          uint8_t global_max) {
  if (global_min == global_max) {
    return;  // No stretching needed
  }

  const int delta = global_max - global_min;
  const std::size_t grain_size = std::max(std::size_t(16), (end_idx - start_idx) / 16);

  tbb::parallel_for(tbb::blocked_range<std::size_t>(start_idx, end_idx, grain_size),
                    [global_min, delta, &img](const tbb::blocked_range<std::size_t>& range) {
                      for (std::size_t i = range.begin(); i != range.end(); ++i) {
                        img[i] = static_cast<uint8_t>(((img[i] - global_min) * 255 + delta / 2) / delta);
                      }
                    });
}

void GatherResults(boost::mpi::communicator& world, std::vector<uint8_t>& img, std::size_t img_size,
                   std::size_t chunk_size, std::size_t remainder) {
  int rank = world.rank();
  int size = world.size();

  if (size <= 1) {
    return;
  }

  if (rank == 0) {
    for (int src_rank = 1; src_rank < size; ++src_rank) {
      std::size_t src_start = src_rank * chunk_size;
      std::size_t src_size = (src_rank == size - 1) ? (chunk_size + remainder) : chunk_size;

      if (src_size > 0 && src_start < img_size) {
        src_size = std::min(src_size, img_size - src_start);
        world.recv(src_rank, 0, img.data() + src_start, static_cast<int>(src_size));
      }
    }
  } else {
    std::size_t start_idx = rank * chunk_size;
    std::size_t my_chunk_size = (rank == size - 1) ? (chunk_size + remainder) : chunk_size;

    if (my_chunk_size > 0 && start_idx < img_size) {
      my_chunk_size = std::min(my_chunk_size, img_size - start_idx);
      world.send(0, 0, img.data() + start_idx, static_cast<int>(my_chunk_size));
    }
  }
}
}  // namespace

bool TestTaskAll::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->inputs_count.empty() && task_data->inputs_count[0] != 0 &&
         !task_data->outputs.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskAll::PreProcessingImpl() {
  const uint8_t* input_data = task_data->inputs.front();
  const uint32_t input_size = task_data->inputs_count.front();

  img_.assign(input_data, input_data + input_size);
  return true;
}

bool TestTaskAll::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  bool is_small_dataset = img_.size() <= 100;
  if (is_small_dataset) {
    if (rank == 0) {
      MinMaxPair minmax = CalculateLocalMinMax(img_, 0, img_.size());
      ApplyStretchingLocal(img_, 0, img_.size(), minmax.min_val, minmax.max_val);
    }
    world_.barrier();
    return true;
  }

  std::size_t img_size = img_.size();
  boost::mpi::broadcast(world_, img_size, 0);  // Ensure all processes know the size

  // Chunk sizes
  std::size_t chunk_size = img_size / size;
  std::size_t remainder = img_size % size;
  std::size_t start_idx = rank * chunk_size;
  std::size_t end_idx = (rank == size - 1) ? (start_idx + chunk_size + remainder) : (start_idx + chunk_size);

  // Process might get no data
  if (start_idx >= img_size || start_idx >= end_idx) {
    MinMaxPair local_minmax(std::numeric_limits<uint8_t>::max(), 0);  // Participate with neutral values
    MinMaxPair global_minmax;
    boost::mpi::all_reduce(world_, local_minmax, global_minmax, [](const MinMaxPair& a, const MinMaxPair& b) {
      return MinMaxPair{std::min(a.min_val, b.min_val), std::max(a.max_val, b.max_val)};
    });
  } else {
    MinMaxPair local_minmax = CalculateLocalMinMax(img_, start_idx, end_idx);

    MinMaxPair global_minmax;
    boost::mpi::all_reduce(world_, local_minmax, global_minmax, [](const MinMaxPair& a, const MinMaxPair& b) {
      return MinMaxPair{std::min(a.min_val, b.min_val), std::max(a.max_val, b.max_val)};
    });

    ApplyStretchingLocal(img_, start_idx, end_idx, global_minmax.min_val, global_minmax.max_val);
  }

  GatherResults(world_, img_, img_size, chunk_size, remainder);

  return true;
}

bool TestTaskAll::PostProcessingImpl() {
  uint8_t* output_data = task_data->outputs[0];
  const uint32_t output_size = task_data->outputs_count[0];
  const uint32_t copy_size = std::min(output_size, static_cast<uint32_t>(img_.size()));

  if (world_.rank() == 0) {
    std::copy_n(img_.cbegin(), copy_size, output_data);
  }

  return true;
}

}  // namespace milovankin_m_histogram_stretching_all