#pragma once

#include <oneapi/tbb/blocked_range.h>

#include <atomic>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace trubin_a_algorithm_dijkstra_all {

struct Edge {
  size_t to;
  int weight;

  Edge(size_t to, int weight) : to(to), weight(weight) {}
};

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  bool BuildAdjacencyList(const std::vector<int>& graph_data);
  void RunAlgorithm(boost::mpi::communicator& world, int rank, int size);

  void InitializeAtomicDistances(std::vector<std::atomic<int>>& distances_atomic) const;
  void ProcessBlock(const tbb::blocked_range<size_t>& r, std::atomic<bool>& local_changed,
                    std::vector<std::atomic<int>>& distances_atomic) const;
  void SyncGlobalDistances(boost::mpi::communicator& world, std::vector<std::atomic<int>>& distances_atomic) const;
  void FinalizeDistances(const std::vector<std::atomic<int>>& distances_atomic);

  std::vector<std::vector<Edge>> adjacency_list_;
  std::vector<int> distances_;
  size_t start_vertex_ = 0;
  size_t num_vertices_ = 0;

  bool validation_passed_ = false;

  static constexpr int kEndOfVertexList = -1;
};

}  // namespace trubin_a_algorithm_dijkstra_all
