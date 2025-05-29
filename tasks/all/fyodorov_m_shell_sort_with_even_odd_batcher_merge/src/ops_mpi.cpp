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

bool TestTaskMPI::RunImpl() {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  // Явное обнуление output_, чтобы избежать null-dereference при последующих вызовах
  output_.clear();
  output_.shrink_to_fit();

  // Синхронизация входных данных
  if (rank != 0) {
    input_.clear();
  }
  boost::mpi::broadcast(world, input_, 0);

  int n = static_cast<int>(input_.size());
  if (rank == 0) {
    std::cout << "input_ (first 10): ";
    for (int i = 0; i < std::min(10, n); ++i) {
      std::cout << input_[i] << " ";
    }
    std::cout << std::endl;
  }

  std::vector<int> local_data;
  int local_size = 0;

  if (n > 0) {
    int local_n = n / size;
    int remainder = n % size;

    std::vector<int> sendcounts(size, local_n);
    std::vector<int> displs(size, 0);

    for (int i = 0; i < remainder; ++i) {
      sendcounts[i]++;
    }

    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    local_size = sendcounts[rank];
    local_data.resize(local_size);

    int* send_ptr = (rank == 0 && n > 0) ? input_.data() : nullptr;
    int* recv_ptr = (local_size > 0) ? local_data.data() : nullptr;

    boost::mpi::scatterv(world, send_ptr, sendcounts, displs, recv_ptr, local_size, 0);

    if (local_size > 0) {
      ShellSort(local_data);

      std::cout << "rank " << rank << " local_data (first 10): ";
      for (int i = 0; i < std::min(10, local_size); ++i) {
        std::cout << local_data[i] << " ";
      }
      std::cout << std::endl;
    }

    std::vector<int> gathered;
    if (rank == 0) {
      gathered.resize(n);
    }

    int* send_ptr_g = (local_size > 0) ? local_data.data() : nullptr;
    int* recv_ptr_g = (rank == 0 && n > 0) ? gathered.data() : nullptr;

    boost::mpi::gatherv(world, send_ptr_g, local_size, recv_ptr_g, sendcounts, displs, 0);

    if (rank == 0) {
      std::cout << "gathered (first 10): ";
      for (int i = 0; i < std::min(10, static_cast<int>(gathered.size())); ++i) {
        std::cout << gathered[i] << " ";
      }
      std::cout << std::endl;

      std::vector<std::vector<int>> blocks(size);
      int pos = 0;
      for (int i = 0; i < size; ++i) {
        if (sendcounts[i] > 0 && pos + sendcounts[i] <= static_cast<int>(gathered.size())) {
          auto first = gathered.begin() + pos;
          auto last = first + sendcounts[i];
          blocks[i].assign(first, last);  // <-- безопасное копирование
        } else {
          blocks[i].clear();
        }
        pos += sendcounts[i];
      }

      std::vector<int> merged;
      if (!blocks[0].empty()) {
        merged = blocks[0];
      }
      for (int i = 1; i < size; ++i) {
        if (!blocks[i].empty()) {
          std::vector<int> temp(merged.size() + blocks[i].size());
          BatcherMerge(merged, blocks[i], temp);
          merged.assign(temp.begin(), temp.end());
        }
      }

      // Присвоение без копирующего оператора
      output_.assign(merged.begin(), merged.end());

      // Рассылаем результат другим процессам
      for (int dest = 1; dest < size; ++dest) {
        world.send(dest, 0, output_);
      }
    } else {
      world.recv(0, 0, output_);
    }
  }

  // Гарантированно задаём нужный размер output_
  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size, 0);

  // Вывод только если есть данные и это главный процесс
  if (rank == 0 && !output_.empty()) {
    std::cout << "output_ (first 10): ";
    for (int i = 0; i < std::min(10, static_cast<int>(output_.size())); ++i) {
      std::cout << output_[i] << " ";
    }
    std::cout << std::endl;
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
  if (arr.empty()) return;
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