#include "all/solovyev_d_shell_sort_simple/include/ops_all.hpp"

#include <algorithm>
#include <barrier>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool solovyev_d_shell_sort_simple_all::TaskALL::PreProcessingImpl() {
  size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  return true;
}

bool solovyev_d_shell_sort_simple_all::TaskALL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void solovyev_d_shell_sort_simple_all::TaskALL::ShellSort(std::vector<int>& data) const {
  std::barrier sync_point(num_threads_);
  std::vector<std::thread> threads(num_threads_);
  for (int t = 0; t < num_threads_; ++t) {
    threads[t] = std::thread([&, t] {
      for (size_t gap = data.size() / 2; gap > 0; gap /= 2) {
        sync_point.arrive_and_wait();
        for (size_t i = t; i < gap; i += num_threads_) {
          for (size_t f = i + gap; f < data.size(); f += gap) {
            int val = data[f];
            size_t j = f;
            while (j >= gap && data[j - gap] > val) {
              data[j] = data[j - gap];
              j -= gap;
            }
            data[j] = val;
          }
        }
      }
    });
  }
  for (auto& th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }
}
namespace {
void FinalMerge(std::vector<int>& data, const std::vector<int>& send_counts, const std::vector<int>& displs) {
  struct Block {
    int start;
    int end;
    int index;
  };
  std::vector<Block> blocks;
  for (size_t i = 0; i < send_counts.size(); ++i) {
    int start = displs[i];
    int end = start + send_counts[i];
    if (start < end) {
      blocks.push_back({start, end, start});
    }
  }
  std::vector<int> result;
  result.reserve(data.size());
  while (!blocks.empty()) {
    int min_val = data[blocks[0].index];
    size_t min_block = 0;
    for (size_t i = 1; i < blocks.size(); ++i) {
      if (data[blocks[i].index] < min_val) {
        min_val = data[blocks[i].index];
        min_block = i;
      }
    }
    result.push_back(min_val);
    blocks[min_block].index++;
    if (blocks[min_block].index >= blocks[min_block].end) {
      blocks.erase(blocks.begin() + static_cast<long>(min_block));
    }
  }
  data = std::move(result);
}
}  // namespace
bool solovyev_d_shell_sort_simple_all::TaskALL::RunImpl() {
  num_threads_ = std::max(1, ppc::util::GetPPCNumThreads());
  int rank = world_.rank();
  int size = world_.size();
  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  int base_size = static_cast<int>(input_.size()) / size;
  int remainder = static_cast<int>(input_.size()) % size;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : (displs[i - 1] + send_counts[i - 1]);
  }
  int local_size = send_counts[rank];
  std::vector<int> local_data(local_size);
  boost::mpi::scatterv(world_, input_, send_counts, displs, local_data.data(), local_size, 0);
  solovyev_d_shell_sort_simple_all::TaskALL::ShellSort(local_data);
  boost::mpi::gatherv(world_, local_data, input_.data(), send_counts, displs, 0);
  if (rank == 0) {
    FinalMerge(input_, send_counts, displs);
  }
  return true;
}

bool solovyev_d_shell_sort_simple_all::TaskALL::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
