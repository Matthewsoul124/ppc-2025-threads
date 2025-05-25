#include "all/trubin_a_algorithm_dijkstra/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <atomic>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/operations.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/util/include/util.hpp"

bool trubin_a_algorithm_dijkstra_all::TestTaskALL::PreProcessingImpl() {
  if (!validation_passed_) {
    return false;
  }

  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::vector<int> graph_data(in_ptr, in_ptr + input_size);

  num_vertices_ = task_data->outputs_count[0];

  if (num_vertices_ == 0 || input_size == 0) {
    return true;
  }

  adjacency_list_.assign(num_vertices_, {});
  distances_.assign(num_vertices_, std::numeric_limits<int>::max());

  if (!BuildAdjacencyList(graph_data)) {
    return false;
  }

  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    int* ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    if (*ptr < 0 || static_cast<size_t>(*ptr) >= num_vertices_) {
      return false;
    }
    start_vertex_ = static_cast<size_t>(*ptr);
  } else {
    start_vertex_ = 0;
  }

  distances_[start_vertex_] = 0;
  return true;
}

bool trubin_a_algorithm_dijkstra_all::TestTaskALL::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty() || task_data->inputs_count.empty() ||
      task_data->outputs_count.empty()) {
    return false;
  }

  if (task_data->outputs_count[0] == 0 && task_data->inputs_count[0] == 0) {
    validation_passed_ = true;
    return true;
  }

  if (task_data->outputs[0] == nullptr || task_data->outputs_count[0] == 0) {
    return false;
  }

  validation_passed_ = true;
  return true;
}

bool trubin_a_algorithm_dijkstra_all::TestTaskALL::RunImpl() {
  if (num_vertices_ == 0) {
    return true;
  }
  if (start_vertex_ >= num_vertices_) {
    return false;
  }

  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  RunAlgorithm(world, rank, size);
  return true;
}

void trubin_a_algorithm_dijkstra_all::TestTaskALL::RunAlgorithm(boost::mpi::communicator& world, int rank, int size) {
  const int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);

  arena.execute([&] {
    std::vector<std::atomic<int>> distances_atomic(num_vertices_);
    InitializeAtomicDistances(distances_atomic);

    const size_t block_size = (num_vertices_ + size - 1) / size;
    const size_t local_start = rank * block_size;
    const size_t local_end = std::min(local_start + block_size, num_vertices_);

    bool global_changed = true;

    while (global_changed) {
      std::atomic<bool> local_changed = false;

      tbb::parallel_for(tbb::blocked_range<size_t>(local_start, local_end, 512),
                        [&](const tbb::blocked_range<size_t>& r) { ProcessBlock(r, local_changed, distances_atomic); });

      const int local_flag = local_changed.load() ? 1 : 0;
      int global_flag = 0;
      boost::mpi::all_reduce(world, local_flag, global_flag, std::plus<>());
      global_changed = (global_flag != 0);

      SyncGlobalDistances(world, distances_atomic);
    }

    FinalizeDistances(distances_atomic);
  });
}

void trubin_a_algorithm_dijkstra_all::TestTaskALL::InitializeAtomicDistances(
    std::vector<std::atomic<int>>& distances_atomic) const {
  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_atomic[i] = std::numeric_limits<int>::max();
  }
  distances_atomic[start_vertex_] = 0;
}

void trubin_a_algorithm_dijkstra_all::TestTaskALL::ProcessBlock(const tbb::blocked_range<size_t>& r,
                                                                std::atomic<bool>& local_changed,
                                                                std::vector<std::atomic<int>>& distances_atomic) const {
  for (size_t u = r.begin(); u < r.end(); ++u) {
    const int u_dist = distances_atomic[u];
    if (u_dist == std::numeric_limits<int>::max()) {
      continue;
    }

    for (const auto& edge : adjacency_list_[u]) {
      const int new_dist = u_dist + edge.weight;
      int old_dist = distances_atomic[edge.to].load();
      while (new_dist < old_dist) {
        if (distances_atomic[edge.to].compare_exchange_strong(old_dist, new_dist)) {
          local_changed.store(true, std::memory_order_relaxed);
          break;
        }
      }
    }
  }
}

void trubin_a_algorithm_dijkstra_all::TestTaskALL::SyncGlobalDistances(
    boost::mpi::communicator& world, std::vector<std::atomic<int>>& distances_atomic) const {
  std::vector<int> snapshot(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    snapshot[i] = distances_atomic[i].load();
  }

  std::vector<int> global_result(num_vertices_);

  if (num_vertices_ > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Too many vertices for MPI call");
  }

  int count = static_cast<int>(num_vertices_);
  boost::mpi::all_reduce(world, snapshot.data(), count, global_result.data(), boost::mpi::minimum<int>());

  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_atomic[i] = global_result[i];
  }
}

void trubin_a_algorithm_dijkstra_all::TestTaskALL::FinalizeDistances(
    const std::vector<std::atomic<int>>& distances_atomic) {
  distances_.resize(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    const int d = distances_atomic[i];
    distances_[i] = (d == std::numeric_limits<int>::max()) ? -1 : d;
  }
}

bool trubin_a_algorithm_dijkstra_all::TestTaskALL::PostProcessingImpl() {
  if (num_vertices_ == 0) {
    return true;
  }

  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < num_vertices_; ++i) {
    out_ptr[i] = (distances_[i] == std::numeric_limits<int>::max()) ? -1 : distances_[i];
  }
  return true;
}

bool trubin_a_algorithm_dijkstra_all::TestTaskALL::BuildAdjacencyList(const std::vector<int>& graph_data) {
  size_t current_vertex = 0;
  size_t i = 0;

  while (i < graph_data.size()) {
    if (graph_data[i] == kEndOfVertexList) {
      if (current_vertex >= num_vertices_) {
        return false;
      }
      current_vertex++;
      i++;
      continue;
    }

    if (i + 1 >= graph_data.size()) {
      return false;
    }

    auto to = static_cast<size_t>(graph_data[i]);
    int weight = graph_data[i + 1];

    if (to >= num_vertices_ || weight < 0) {
      return false;
    }

    adjacency_list_[current_vertex].emplace_back(to, weight);
    i += 2;
  }

  return true;
}
