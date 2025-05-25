#include "../include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cstddef>
#include <functional>
#include <queue>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

bool nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    vect_size_ = task_data->inputs_count[0];
    auto *vect_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    vect_ = std::vector<double>(vect_ptr, vect_ptr + vect_size_);
  }
  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] != 0 && task_data->inputs[0] != nullptr &&
           task_data->outputs[0] != nullptr && task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::RunImpl() {
  int rank = world_.rank();
  int comm_size = world_.size();

  size_t total_size = BroadcastTotalSize();
  std::vector<double> local_vect = DistributeVector(total_size, rank, comm_size);
  LocalSort(local_vect);
  GlobalMerge(rank, local_vect);

  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < vect_size_; i++) {
      reinterpret_cast<double *>(task_data->outputs[0])[i] = vect_[i];
    }
  }
  return true;
}

void nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::QuickSort(std::vector<double> &vec, size_t low,
                                                                                size_t high) {
  if (low >= high) {
    return;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(static_cast<int>(low), static_cast<int>(high));
  size_t random_pivot_index = dist(gen);
  double pivot = vec[random_pivot_index];
  std::swap(vec[random_pivot_index], vec[low]);

  size_t i = low + 1;
  for (size_t j = low + 1; j <= high; ++j) {
    if (vec[j] < pivot) {
      std::swap(vec[i], vec[j]);
      ++i;
    }
  }
  std::swap(vec[low], vec[i - 1]);
  size_t pivot_pos = i - 1;
  if (pivot_pos > low) {
    QuickSort(vec, low, pivot_pos - 1);
  }
  QuickSort(vec, pivot_pos + 1, high);
}

size_t nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::BroadcastTotalSize() {
  size_t total_size = vect_size_;
  boost::mpi::broadcast(world_, total_size, 0);
  return total_size;
}

std::vector<double> nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::DistributeVector(size_t total_size,
                                                                                                      int rank,
                                                                                                      int comm_size) {
  size_t base_chunk = total_size / comm_size;
  size_t remainder = total_size % comm_size;
  size_t local_count = base_chunk + (static_cast<size_t>(rank) < remainder ? 1 : 0);

  std::vector<double> local_vect(local_count);
  std::vector<int> counts(comm_size);
  std::vector<int> displs(comm_size);
  size_t offset = 0;

  for (int i = 0; i < comm_size; ++i) {
    counts[i] = static_cast<int>(base_chunk + (i < static_cast<int>(remainder) ? 1 : 0));
    displs[i] = static_cast<int>(offset);
    offset += counts[i];
  }

  boost::mpi::scatterv(world_, vect_, counts, displs, local_vect.data(), static_cast<int>(local_vect.size()), 0);
  return local_vect;
}

void nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::LocalSort(std::vector<double> &local_vect) {
  size_t num_threads = ppc::util::GetPPCNumThreads();
  num_threads = std::min(num_threads, local_vect.size());

  size_t seg_size = !local_vect.empty() ? local_vect.size() / num_threads : 0;
  size_t seg_remainder = !local_vect.empty() ? local_vect.size() % num_threads : 0;
  std::vector<std::pair<size_t, size_t>> segments;

  size_t start = 0;
  for (size_t i = 0; i < num_threads; ++i) {
    size_t curr_size = seg_size + (i < seg_remainder ? 1 : 0);
    if (curr_size > 0) {
      segments.emplace_back(start, start + curr_size - 1);
    }
    start += curr_size;
  }

  oneapi::tbb::task_arena arena(static_cast<int>(num_threads));
  arena.execute([&]() {
    oneapi::tbb::task_group tg;
    for (const auto &seg : segments) {
      tg.run([&, seg]() { QuickSort(local_vect, seg.first, seg.second); });
    }
    tg.wait();
  });

  if (!segments.empty()) {
    size_t merged_end = segments[0].second;
    for (size_t i = 1; i < segments.size(); ++i) {
      std::inplace_merge(local_vect.begin(), local_vect.begin() + static_cast<std::ptrdiff_t>(merged_end + 1),
                         local_vect.begin() + static_cast<std::ptrdiff_t>(segments[i].second + 1));
      merged_end = segments[i].second;
    }
  }
}

void nikolaev_r_hoare_sort_simple_merge_all::HoareSortSimpleMergeALL::GlobalMerge(
    int rank, const std::vector<double> &local_vect) {
  std::vector<std::vector<double>> gathered;
  boost::mpi::gather(world_, local_vect, gathered, 0);

  if (rank == 0) {
    std::vector<double> global_sorted;
    using HeapElem = std::tuple<double, int, size_t>;
    std::priority_queue<HeapElem, std::vector<HeapElem>, std::greater<>> min_heap;

    for (int i = 0; i < static_cast<int>(gathered.size()); ++i) {
      if (!gathered[i].empty()) {
        min_heap.emplace(gathered[i][0], i, 0);
      }
    }

    while (!min_heap.empty()) {
      auto [value, list_index, elem_index] = min_heap.top();
      min_heap.pop();
      global_sorted.push_back(value);
      if (elem_index + 1 < gathered[list_index].size()) {
        min_heap.emplace(gathered[list_index][elem_index + 1], list_index, elem_index + 1);
      }
    }
    vect_ = std::move(global_sorted);
  }

  boost::mpi::broadcast(world_, vect_, 0);
}