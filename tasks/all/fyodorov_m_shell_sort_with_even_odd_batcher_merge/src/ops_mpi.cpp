#include "all/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cstddef>
#include <vector>

#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/collectives/scatterv.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi {

boost::mpi::communicator world_;

bool TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = 0;
  if (world_.rank() == 0) {
    input_size = task_data->inputs_count[0];
  }
  boost::mpi::broadcast(world_, input_size, 0);

  input_.resize(input_size);
  if (world_.rank() == 0) {
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(in_ptr, in_ptr + input_size, input_.begin());
  }
  boost::mpi::broadcast(world_, input_, 0);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  std::cout << "rank " << world_.rank() << " input_ (first 10): ";
  for (size_t i = 0; i < std::min<size_t>(10, input_size); ++i) std::cout << input_[i] << " ";
  std::cout << std::endl;

  return true;
}

bool TestTaskMPI::ValidationImpl() {
  return ((task_data->inputs_count[0] == task_data->outputs_count[0]) &&
          (task_data->outputs.size() == task_data->outputs_count.size()));
}

bool TestTaskMPI::RunImpl() {
  int n = static_cast<int>(input_.size());
  int rank = world_.rank();
  int size = world_.size();
  std::vector<int> local_data;
  int local_n = n / size;
  int remainder = n % size;
  std::vector<int> sendcounts, displs;
  int local_size = 0;
  if (rank == 0) {
    sendcounts.resize(size, local_n);
    displs.resize(size, 0);
    for (int i = 0; i < remainder; ++i) {
      sendcounts[i]++;
    }
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    local_size = sendcounts[rank];
    for (int i = 1; i < size; ++i) {
      world_.send(i, 0, sendcounts[i]);
    }
  } else {
    world_.recv(0, 0, local_size);
  }
  local_data.resize(local_size);

  if (rank == 0) {
    int offset = 0;
    for (int i = 0; i < size; ++i) {
      int count = sendcounts[i];
      if (i == 0) {
        local_data.assign(input_.begin() + offset, input_.begin() + offset + count);
      } else {
        world_.send(i, 0, std::vector<int>(input_.begin() + offset, input_.begin() + offset + count));
      }
      offset += count;
    }
  } else {
    world_.recv(0, 0, local_data);
  }

  std::cout << "[rank " << rank << "] local_data после распределения (first 10): ";
  for (int i = 0; i < std::min(10, (int)local_data.size()); ++i) {
    std::cout << local_data[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "[rank " << rank << "] sendcounts: ";
  for (auto v : sendcounts) std::cout << v << " ";
  std::cout << "\ndispls: ";
  for (auto v : displs) std::cout << v << " ";
  std::cout << std::endl;

  std::cout << "[rank " << rank << "] local_data перед ShellSort (first 10): ";
  for (int i = 0; i < std::min(10, (int)local_data.size()); ++i) std::cout << local_data[i] << " ";
  std::cout << std::endl;

  ShellSort(local_data);

  std::cout << "[rank " << rank << "] local_data после ShellSort (first 10): ";
  for (int i = 0; i < std::min(10, (int)local_data.size()); ++i) std::cout << local_data[i] << " ";
  std::cout << std::endl;

  std::vector<int> gathered;
  if (rank == 0) {
    gathered.resize(n);
  }

  boost::mpi::broadcast(world_, sendcounts, 0);
  boost::mpi::broadcast(world_, displs, 0);

  boost::mpi::gatherv(world_, local_data.data(), sendcounts[rank], gathered.data(), sendcounts, displs, 0);

  if (rank == 0) {
    std::cout << "gathered (first 10): ";
    for (int i = 0; i < std::min(10, (int)gathered.size()); ++i) {
      std::cout << gathered[i] << " ";
    }
    std::cout << std::endl;
    std::vector<std::vector<int>> blocks(size);
    for (int i = 0, pos = 0; i < size; ++i) {
      if (sendcounts[i] > 0) {
        blocks[i] = std::vector<int>(gathered.begin() + pos, gathered.begin() + pos + sendcounts[i]);
      } else {
        blocks[i] = std::vector<int>();
      }
      pos += sendcounts[i];
    }
    std::vector<int> merged = blocks[0];
    for (int i = 1; i < size; ++i) {
      std::vector<int> temp(merged.size() + blocks[i].size());
      BatcherMerge(merged, blocks[i], temp);
      merged = temp;
    }
    output_ = merged;
    if (rank == 0) {
      std::cout << "output_ (first 10): ";
      for (int i = 0; i < std::min(10, (int)output_.size()); ++i) {
        std::cout << output_[i] << " ";
      }
      std::cout << std::endl;
    }
    for (int i = 1; i < size; ++i) {
      world_.send(i, 0, output_);
    }
  } else {
    world_.recv(0, 0, output_);
  }

  unsigned int output_size = task_data->outputs_count[0];
  if (output_.size() != output_size) {
    output_.resize(output_size, 0);
  }

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  unsigned int output_size = task_data->outputs_count[0];
  if (output_.size() == output_size) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
    }
  } else {
    for (size_t i = 0; i < output_size; ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = 0;
    }
  }
  return true;
}

void TestTaskMPI::ShellSort(std::vector<int>& arr) {
  int n = static_cast<int>(arr.size());
  std::vector<int> gaps;
  for (int k = 1; (1 << k) - 1 < n; ++k) {
    gaps.push_back((1 << k) - 1);
  }
  for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
    int gap = *it;

#pragma omp parallel for default(none) shared(arr, n, gap)
    for (int offset = 0; offset < gap; ++offset) {
      for (int i = offset + gap; i < n; i += gap) {
        int temp = arr[i];
        int j = i;
        while (j >= gap && arr[j - gap] > temp) {
          arr[j] = arr[j - gap];
          j -= gap;
        }
        arr[j] = temp;
      }
    }
  }
}

void TestTaskMPI::BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result) {
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result[k++] = left[i++];
    } else {
      result[k++] = right[j++];
    }
  }
  while (i < left.size()) {
    result[k++] = left[i++];
  }
  while (j < right.size()) {
    result[k++] = right[j++];
  }
}

}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi
