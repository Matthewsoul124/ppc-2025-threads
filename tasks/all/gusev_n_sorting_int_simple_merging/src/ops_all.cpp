#include "all/gusev_n_sorting_int_simple_merging/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>  // IWYU pragma: keep
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>  // IWYU pragma: keep
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>  // IWYU pragma: keep
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // IWYU pragma: keep
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::SplitBySign(const std::vector<int>& arr,
                                                                                     std::vector<int>& negatives,
                                                                                     std::vector<int>& positives) {
  for (int num : arr) {
    if (num < 0) {
      negatives.push_back(std::abs(num));
    } else {
      positives.push_back(num);
    }
  }
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::MergeResults(
    std::vector<int>& arr, const std::vector<int>& negatives, const std::vector<int>& positives) {
  arr.clear();
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), positives.begin(), positives.end());
}

std::vector<std::vector<int>> gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::DistributeArray(
    const std::vector<int>& arr, int num_procs) {
  std::vector<std::vector<int>> chunks(num_procs);

  if (arr.empty()) {
    return chunks;
  }

  size_t chunk_size = arr.size() / num_procs;
  size_t remainder = arr.size() % num_procs;

  size_t start = 0;
  for (size_t i = 0; i < static_cast<size_t>(num_procs); ++i) {
    size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);

    if (current_chunk_size > 0) {
      size_t end = start + current_chunk_size;
      chunks[i].insert(chunks[i].end(), arr.begin() + static_cast<std::ptrdiff_t>(start),
                       arr.begin() + static_cast<std::ptrdiff_t>(end));
      start = end;
    }
  }

  return chunks;
}

std::vector<int> gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::MergeSortedArrays(
    const std::vector<std::vector<int>>& arrays) {
  std::vector<int> result;

  size_t total_size = 0;
  for (const auto& arr : arrays) {
    total_size += arr.size();
  }
  result.reserve(total_size);

  for (const auto& arr : arrays) {
    if (arr.empty()) {
      continue;
    }

    if (result.empty()) {
      result = arr;
    } else {
      std::vector<int> merged(result.size() + arr.size());
      std::ranges::merge(result, arr, merged.begin());
      result = std::move(merged);
    }
  }

  return result;
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::DistributeAndSortChunk(
    std::vector<int>& chunk, std::vector<std::vector<int>>& chunks, int rank, int size, int tag) {
  boost::mpi::communicator world;

  std::vector<int> my_chunk;
  if (size > 1) {
    if (rank == 0) {
      boost::mpi::scatter(world, chunks, my_chunk, 0);
    } else {
      boost::mpi::scatter(world, my_chunk, 0);
    }
  } else {
    my_chunk = chunk;
  }

  if (!my_chunk.empty()) {
    RadixSortForNonNegative(my_chunk);
  }

  chunk = std::move(my_chunk);
}

std::vector<int> gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::GatherSortedChunks(
    const std::vector<int>& my_chunk, int rank, int size, int tag) {
  boost::mpi::communicator world;

  if (size <= 1) {
    return my_chunk;
  }

  std::vector<std::vector<int>> gathered_chunks;
  boost::mpi::gather(world, my_chunk, gathered_chunks, 0);

  if (rank == 0) {
    return MergeSortedArrays(gathered_chunks);
  }

  return {};
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::ProcessNegativeNumbers(
    std::vector<int>& negatives, std::vector<int>& sorted_negatives, int rank, int size) {
  boost::mpi::communicator world;

  if (negatives.empty()) {
    return;
  }

  std::vector<std::vector<int>> neg_chunks;
  if (rank == 0) {
    neg_chunks = DistributeArray(negatives, size);
  }

  std::vector<int> my_neg_chunk;
  DistributeAndSortChunk(negatives, neg_chunks, rank, size, 0);
  my_neg_chunk = std::move(negatives);

  sorted_negatives = GatherSortedChunks(my_neg_chunk, rank, size, 0);

  if (rank == 0 && !sorted_negatives.empty()) {
    std::ranges::reverse(sorted_negatives);
    std::ranges::transform(sorted_negatives, sorted_negatives.begin(), std::negate{});
  }
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::ProcessPositiveNumbers(
    std::vector<int>& positives, std::vector<int>& sorted_positives, int rank, int size) {
  boost::mpi::communicator world;

  if (positives.empty()) {
    return;
  }

  std::vector<std::vector<int>> pos_chunks;
  if (rank == 0) {
    pos_chunks = DistributeArray(positives, size);
  }

  std::vector<int> my_pos_chunk;
  DistributeAndSortChunk(positives, pos_chunks, rank, size, 1);
  my_pos_chunk = std::move(positives);

  sorted_positives = GatherSortedChunks(my_pos_chunk, rank, size, 1);
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::RadixSort(std::vector<int>& arr) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (arr.empty()) {
    return;
  }

  std::vector<int> negatives;
  std::vector<int> positives;

  if (rank == 0) {
    SplitBySign(arr, negatives, positives);
  }

  size_t negatives_size = negatives.size();
  size_t positives_size = positives.size();
  boost::mpi::broadcast(world, negatives_size, 0);
  boost::mpi::broadcast(world, positives_size, 0);

  std::vector<int> sorted_negatives;
  std::vector<int> sorted_positives;

  if (negatives_size > 0) {
    if (rank != 0) {
      negatives.resize(negatives_size);
    }
    boost::mpi::broadcast(world, negatives, 0);

    ProcessNegativeNumbers(negatives, sorted_negatives, rank, size);
  }

  if (positives_size > 0) {
    if (rank != 0) {
      positives.resize(positives_size);
    }
    boost::mpi::broadcast(world, positives, 0);

    ProcessPositiveNumbers(positives, sorted_positives, rank, size);
  }

  if (rank == 0) {
    MergeResults(arr, sorted_negatives, sorted_positives);
  }
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::RadixSortForNonNegative(
    std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  int max = *std::ranges::max_element(arr);
  for (int exp = 1; max / exp > 0; exp *= 10) {
    CountingSort(arr, exp);
  }
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::CountingSort(std::vector<int>& arr, int exp) {
  boost::mpi::communicator world;

  std::vector<int> output(arr.size());
  std::vector<int> count(10, 0);

  oneapi::tbb::enumerable_thread_specific<std::vector<int>> tl_counts([&] { return std::vector<int>(10, 0); });

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, arr.size()),
                            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                              auto& local_counts = tl_counts.local();
                              for (size_t i = r.begin(); i < r.end(); ++i) {
                                int digit = (arr[i] / exp) % 10;
                                local_counts[digit]++;
                              }
                            });

  for (const auto& lc : tl_counts) {
    for (int d = 0; d < 10; ++d) {
      count[d] += lc[d];
    }
  }

  std::partial_sum(count.begin(), count.end(), count.begin());

  for (auto i = arr.size(); i > 0; --i) {
    int digit = (arr[i - 1] / exp) % 10;
    output[--count[digit]] = arr[i - 1];
  }

  arr = output;
}

bool gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::PreProcessingImpl() {
  input_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                            reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  output_.resize(input_.size());
  return true;
}

bool gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::ValidationImpl() {
  if (task_data->inputs_count[0] == 0) {
    return true;
  }

  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::RunImpl() {
  RadixSort(input_);
  return true;
}

bool gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
