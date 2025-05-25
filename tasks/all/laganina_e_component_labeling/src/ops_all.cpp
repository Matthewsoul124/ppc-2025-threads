#include "all/laganina_e_component_labeling/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cassert>
#include <queue>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

bool laganina_e_component_labeling_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data == nullptr || task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
      return false;
    }

    const auto size = static_cast<int>(task_data->inputs_count[0] * task_data->inputs_count[1]);
    const int* input = reinterpret_cast<const int*>(task_data->inputs[0]);

    for (int i = 0; i < size; ++i) {
      if (input[i] != 0 && input[i] != 1) {
        return false;
      }
    }
  }
  return true;
}

bool laganina_e_component_labeling_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    m_ = static_cast<int>(task_data->inputs_count[0]);
    n_ = static_cast<int>(task_data->inputs_count[1]);
    binary_.resize(m_ * n_);
    const int* input = reinterpret_cast<const int*>(task_data->inputs[0]);
    std::copy_n(input, m_ * n_, binary_.begin());
  }
  return true;
}

bool laganina_e_component_labeling_all::TestTaskALL::RunImpl() {
  int size_of_sizes = 0;
  std::vector<int> sizes;
  int col = 0;
  if (world_.rank() == 0) {
    col = n_;
    int delta = (m_ + world_.size() - 1) / world_.size();
    int last = m_ - (delta * (world_.size() - 1));
    delta *= n_;
    last *= n_;
    int i = 0;
    sizes.resize(world_.size(), 0);
    for (; i < n_ * m_ - last; i += delta) {
      sizes[i / delta] = delta;
    }
    sizes[i / delta] = last;
    size_of_sizes = static_cast<int>(sizes.size());
  }

  boost::mpi::broadcast(world_, size_of_sizes, 0);

  if (world_.rank() != 0) {
    sizes.resize(size_of_sizes);
  }
  boost::mpi::broadcast(world_, sizes.data(), size_of_sizes, 0);
  boost::mpi::broadcast(world_, col, 0);
  local_bin_.resize(sizes[world_.rank()]);

  if (world_.rank() == 0) {
    int start = 0;
    int end = sizes[0];
    std::copy(binary_.begin() + start, binary_.begin() + end, local_bin_.begin());
    for (int i = 1; i < size_of_sizes; i++) {
      if (sizes[i] != 0) {
        start += sizes[i - 1];
        end += sizes[i];
        std::vector<int> tmp(sizes[i]);
        std::copy(binary_.begin() + start, binary_.begin() + end, tmp.begin());
        world_.send(i, 0, tmp);
      }
    }
  } else {
    if (sizes[world_.rank()] != 0) {
      world_.recv(0, 0, local_bin_);
    }
  }

  if (sizes[world_.rank()] != 0) {
    local_n_ = col;
    local_m_ = sizes[world_.rank()] / col;
    LabelConnectedComponents();
    NormalizeLabels(local_bin_);
  }

  if (world_.rank() == 0) {
    binary_ = local_bin_;
    for (int i = 1; i < size_of_sizes; i++) {
      if (sizes[i] != 0) {
        std::vector<int> tmp(sizes[i]);
        world_.recv(i, 0, tmp);
        binary_.insert(binary_.end(), tmp.begin(), tmp.end());
      }
    }
  } else {
    if (sizes[world_.rank()] != 0) {
      world_.send(0, 0, local_bin_);
    }
  }

  world_.barrier();

  if (world_.rank() == 0) {
    RelabelImage();
  }

  world_.barrier();

  return true;
}

bool laganina_e_component_labeling_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    int* output = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(binary_.cbegin(), binary_.cend(), output);
  }
  return true;
}

void laganina_e_component_labeling_all::TestTaskALL::InitializeParents(std::vector<int>& parent) {
  int size = 0;
  size = local_m_ * local_n_;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    parent[i] = (local_bin_[i] != 0) ? i : -1;
  }
}

void laganina_e_component_labeling_all::TestTaskALL::ProcessSweep(bool reverse, std::vector<int>& parent,
                                                                  bool& changed) const {
  bool local_changed = false;

#pragma omp parallel for reduction(|| : local_changed) schedule(static)
  for (int row_idx = 0; row_idx < local_m_; ++row_idx) {
    local_changed |= ProcessRow(row_idx, reverse, parent);
  }

  changed = local_changed;
}

bool laganina_e_component_labeling_all::TestTaskALL::ProcessRow(int row_idx, bool reverse,
                                                                std::vector<int>& parent) const {
  const int row = reverse ? local_m_ - 1 - row_idx : row_idx;
  bool row_changed = false;

  for (int col_idx = 0; col_idx < local_n_; ++col_idx) {
    const int col = reverse ? local_n_ - 1 - col_idx : col_idx;
    const int current = (row * local_n_) + col;

    if (parent[current] == -1) {
      continue;
    }

    const int vert_neighbor_row = row - (reverse ? -1 : 1);
    const int vert_neighbor = (vert_neighbor_row * local_n_) + col;
    if (vert_neighbor_row >= 0 && vert_neighbor_row < local_m_ && parent[vert_neighbor] != -1) {
      row_changed |= UnionNodes(current, vert_neighbor, parent);
    }

    const int horz_neighbor_col = col - (reverse ? -1 : 1);
    const int horz_neighbor = (row * local_n_) + horz_neighbor_col;
    if (horz_neighbor_col >= 0 && horz_neighbor_col < local_n_ && parent[horz_neighbor] != -1) {
      row_changed |= UnionNodes(current, horz_neighbor, parent);
    }
  }

  return row_changed;
}

bool laganina_e_component_labeling_all::TestTaskALL::CheckNeighbor(int nr, int nc, int current,
                                                                   std::vector<int>& parent) const {
  if (nr >= 0 && nr < local_m_ && nc >= 0 && nc < local_n_) {
    const int neighbor = (nr * local_n_) + nc;
    if (parent[neighbor] != -1) {
      return UnionNodes(current, neighbor, parent);
    }
  }
  return false;
}

int laganina_e_component_labeling_all::TestTaskALL::FindRoot(std::vector<int>& parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

bool laganina_e_component_labeling_all::TestTaskALL::UnionNodes(int a, int b, std::vector<int>& parent) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);

  if (root_a != root_b) {
    if (root_b < root_a) {
      std::swap(root_a, root_b);
    }

    parent[root_b] = root_a;

    return true;
  }
  return false;
}

void laganina_e_component_labeling_all::TestTaskALL::FinalizeRoots(std::vector<int>& parent) const {
  int size = 0;
  size = local_m_ * local_n_;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    if (parent[i] != -1) {
      parent[i] = FindRoot(parent, i);
    }
  }
}

void laganina_e_component_labeling_all::TestTaskALL::AssignLabels(std::vector<int>& parent) {
  std::vector<int> labels((local_m_ * local_n_) + 1, 0);
  int current_label = 1;

#pragma omp parallel
  {
    std::vector<int> local_roots;
#pragma omp for nowait
    for (int i = 0; i < local_m_ * local_n_; ++i) {
      if (parent[i] != -1 && parent[i] == i) {
        local_roots.push_back(i);
      }
    }

#pragma omp critical
    {
      for (int root : local_roots) {
        if (labels[root] == 0) {
          labels[root] = current_label++;
        }
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < local_m_ * local_n_; ++i) {
    local_bin_[i] = parent[i] != -1 ? labels[parent[i]] : 0;
  }
}

void laganina_e_component_labeling_all::TestTaskALL::LabelConnectedComponents() {
  std::vector<int> parent(local_m_ * local_n_);
  InitializeParents(parent);

  constexpr int kMaxIterations = 100;
  bool changed = false;
  int iterations = 0;

  do {
    changed = false;
    ProcessSweep(false, parent, changed);
    ProcessSweep(true, parent, changed);
  } while (changed && ++iterations < kMaxIterations);

  FinalizeRoots(parent);
  AssignLabels(parent);
}

void laganina_e_component_labeling_all::TestTaskALL::RelabelImage() {
  if (binary_.empty() || m_ == 0 || n_ == 0) {
    return;
  }

  int current_label = -1;
  const std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  for (int i = 0; i < m_; ++i) {
    for (int j = 0; j < n_; ++j) {
      const int idx = (i * n_) + j;
      if (binary_[idx] > 0) {
        ProcessConnectedComponent(i, j, current_label, directions);
        current_label--;
      }
    }
  }

  Normalize();
}

[[nodiscard]] bool laganina_e_component_labeling_all::TestTaskALL::IsValidCoordinate(int x, int y) const {
  return (x >= 0) && (x < m_) && (y >= 0) && (y < n_);
}

void laganina_e_component_labeling_all::TestTaskALL::ProcessConnectedComponent(
    int start_x, int start_y, int label, const std::vector<std::pair<int, int>>& directions) {
  std::queue<std::pair<int, int>> queue;
  queue.emplace(start_x, start_y);
  binary_[(start_x * n_) + start_y] = label;

  while (!queue.empty()) {
    const auto [x, y] = queue.front();
    queue.pop();

    for (const auto& dir : directions) {
      const int nx = x + dir.first;
      const int ny = y + dir.second;

      if (IsValidCoordinate(nx, ny)) {
        const int neighbor_idx = (nx * n_) + ny;
        if (binary_[neighbor_idx] > 0) {
          binary_[neighbor_idx] = label;
          queue.emplace(nx, ny);
        }
      }
    }
  }
}

void laganina_e_component_labeling_all::TestTaskALL::Normalize() {
  for (auto& val : binary_) {
    if (val < 0) {
      val = -val;
    }
  }
}
