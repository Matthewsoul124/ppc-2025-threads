#include "stl/trubin_a_algorithm_dijkstra/include/ops_stl.hpp"

#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::PreProcessingImpl() {
  if (!this->validation_passed_) {
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

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  if (task_data->outputs_count[0] == 0 && task_data->inputs_count[0] == 0) {
    this->validation_passed_ = true;
    return true;
  }

  if (task_data->outputs[0] == nullptr || task_data->outputs_count[0] == 0) {
    return false;
  }

  this->validation_passed_ = true;
  return true;
}

void trubin_a_algorithm_dijkstra_stl::TestTaskSTL::WorkerThread(
    std::priority_queue<std::pair<int, size_t>, std::vector<std::pair<int, size_t>>, std::greater<>>& pq,
    std::mutex& pq_mutex, std::condition_variable& cv, std::atomic<bool>& terminate_flag,
    std::atomic<int>& workers_running, std::vector<std::atomic<int>>& atomic_distances) {
  while (!terminate_flag.load(std::memory_order_acquire)) {
    std::pair<int, size_t> current;
    {
      std::unique_lock<std::mutex> lock(pq_mutex);
      if (pq.empty()) {
        if (workers_running == 0) {
          terminate_flag = true;
          cv.notify_all();
          return;
        }
        cv.wait(lock, [&] { return !pq.empty() || terminate_flag.load(); });
        if (terminate_flag.load()) {
          return;
        }
      }
      current = pq.top();
      pq.pop();
      ++workers_running;
    }

    size_t u = current.second;
    int dist_u = current.first;
    int known = atomic_distances[u].load(std::memory_order_relaxed);
    if (dist_u > known) {
      --workers_running;
      continue;
    }

    for (const auto& edge : adjacency_list_[u]) {
      size_t v = edge.to;
      int weight = edge.weight;

      int candidate = dist_u + weight;
      int current_dist = atomic_distances[v].load(std::memory_order_relaxed);

      while (candidate < current_dist) {
        if (atomic_distances[v].compare_exchange_weak(current_dist, candidate, std::memory_order_relaxed)) {
          std::lock_guard<std::mutex> lock(pq_mutex);
          pq.emplace(candidate, v);
          cv.notify_one();
          break;
        }
      }
    }

    --workers_running;
    std::lock_guard<std::mutex> lock(pq_mutex);
    cv.notify_all();
  }
}

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::RunImpl() {
  if (num_vertices_ == 0) {
    return true;
  }
  if (start_vertex_ >= num_vertices_) {
    return false;
  }

  using QueueElement = std::pair<int, size_t>;
  std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<>> pq;
  std::mutex pq_mutex;
  std::condition_variable cv;
  std::atomic<bool> terminate_flag{false};
  std::atomic<int> workers_running{0};

  std::vector<std::atomic<int>> atomic_distances(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    atomic_distances[i].store(distances_[i], std::memory_order_relaxed);
  }

  {
    std::lock_guard<std::mutex> lock(pq_mutex);
    pq.emplace(0, start_vertex_);
  }

  int thread_count = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> thread_pool;
  thread_pool.reserve(thread_count);

  for (int i = 0; i < thread_count; ++i) {
    thread_pool.emplace_back(&TestTaskSTL::WorkerThread, this, std::ref(pq), std::ref(pq_mutex), std::ref(cv),
                             std::ref(terminate_flag), std::ref(workers_running), std::ref(atomic_distances));
  }
  for (auto& t : thread_pool) {
    t.join();
  }

  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_[i] = atomic_distances[i].load(std::memory_order_relaxed);
  }

  return true;
}

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::PostProcessingImpl() {
  if (num_vertices_ == 0) {
    return true;
  }
  if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < num_vertices_; ++i) {
    int d = distances_[i];
    out_ptr[i] = (d == std::numeric_limits<int>::max()) ? -1 : d;
  }
  return true;
}

bool trubin_a_algorithm_dijkstra_stl::TestTaskSTL::BuildAdjacencyList(const std::vector<int>& graph_data) {
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
