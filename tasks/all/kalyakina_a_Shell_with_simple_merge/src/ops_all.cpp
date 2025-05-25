#include "all/kalyakina_a_Shell_with_simple_merge/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

std::vector<unsigned int> kalyakina_a_shell_with_simple_merge_all::ShellSortALL::CalculationOfGapLengths(
    unsigned int size) {
  std::vector<unsigned int> result;
  unsigned int local_res = 1;
  for (unsigned int i = 1; (local_res * 3 <= size) || (local_res == 1); i++) {
    result.push_back(local_res);
    if (i % 2 != 0) {
      local_res = static_cast<unsigned int>((8 * pow(2, i)) - (6 * pow(2, static_cast<float>(i + 1) / 2)) + 1);
    } else {
      local_res = static_cast<unsigned int>((9 * pow(2, i)) - (9 * pow(2, static_cast<float>(i) / 2)) + 1);
    }
  }
  return result;
}

void kalyakina_a_shell_with_simple_merge_all::ShellSortALL::ShellSort(std::vector<int> &vec) {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  for (unsigned int k = Sedgwick_sequence_.size(); k > 0;) {
    unsigned int gap = Sedgwick_sequence_[--k];
    arena.execute([&] {
      oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<unsigned int>(0, gap),
                                [&](const oneapi::tbb::blocked_range<unsigned int> &r) {
                                  for (unsigned int i = r.begin(); i != r.end(); i++) {
                                    for (unsigned int j = i; j < vec.size(); j += gap) {
                                      unsigned int index = j;
                                      int tmp = vec[index];
                                      while ((index >= i + gap) && (tmp < vec[index - gap])) {
                                        vec[index] = vec[index - gap];
                                        index -= gap;
                                      }
                                      vec[index] = tmp;
                                    }
                                  }
                                });
    });
  }
}

std::vector<int> kalyakina_a_shell_with_simple_merge_all::ShellSortALL::SimpleMergeSort(const std::vector<int> &vec1,
                                                                                        const std::vector<int> &vec2) {
  std::vector<int> result(vec1.size() + vec2.size());
  unsigned int first = 0;
  unsigned int second = 0;
  unsigned int res = 0;
  for (; (first < vec1.size()) && (second < vec2.size()); res++) {
    result[res] = (vec1[first] < vec2[second]) ? vec1[first++] : vec2[second++];
  }
  while (first < vec1.size()) {
    result[res++] = vec1[first++];
  }
  while (second < vec2.size()) {
    result[res++] = vec2[second++];
  }
  return result;
}

bool kalyakina_a_shell_with_simple_merge_all::ShellSortALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_ = std::vector<int>(task_data->inputs_count[0]);
    auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    std::ranges::copy(in_ptr, in_ptr + task_data->inputs_count[0], input_.begin());

    output_ = std::vector<int>(task_data->inputs_count[0]);
  }

  return true;
}

bool kalyakina_a_shell_with_simple_merge_all::ShellSortALL::ValidationImpl() {
  return (world_.rank() != 0) || ((task_data->inputs_count[0] > 0) && (task_data->outputs_count[0] > 0) &&
                                  (task_data->inputs_count[0] == task_data->outputs_count[0]));
}

bool kalyakina_a_shell_with_simple_merge_all::ShellSortALL::RunImpl() {
  unsigned int num = 0;
  std::vector<int> distr(static_cast<unsigned int>(world_.size()), 0);
  std::vector<int> displ(static_cast<unsigned int>(world_.size()), 0);

  if (world_.rank() == 0) {
    num = (static_cast<unsigned int>(world_.size()) > input_.size()) ? input_.size()
                                                                     : static_cast<unsigned int>(world_.size());
    unsigned int part = input_.size() / num;
    unsigned int reminder = input_.size() % num;

    for (unsigned int i = 0; i < num; i++) {
      distr[i] = (i < reminder) ? static_cast<int>(part + 1) : static_cast<int>(part);
      displ[i] = (i == 0) ? 0 : static_cast<int>(displ[i - 1] + distr[i - 1]);
    }
  }

  boost::mpi::broadcast(world_, num, 0);
  boost::mpi::broadcast(world_, distr, 0);
  boost::mpi::broadcast(world_, displ, 0);
  std::vector<int> local_res(distr[world_.rank()]);
  Sedgwick_sequence_ = CalculationOfGapLengths(distr[world_.rank()]);

  boost::mpi::scatterv(world_, input_.data(), distr, displ, local_res.data(), distr[world_.rank()], 0);

  ShellSort(local_res);

  unsigned int step = 1;
  while (step <= num) {
    step *= 2;
    if (((world_.rank() - step / 2) % step) == 0) {
      world_.send(static_cast<int>(world_.rank() - (step / 2)), 0, local_res);
    } else if ((world_.rank() % step == 0) && (static_cast<unsigned int>(world_.size()) > world_.rank() + step / 2)) {
      std::vector<int> message;
      world_.recv(static_cast<int>(world_.rank() + (step / 2)), 0, message);
      local_res = SimpleMergeSort(local_res, message);
    }
  }

  if (world_.rank() == 0) {
    output_ = local_res;
  }

  return true;
}

bool kalyakina_a_shell_with_simple_merge_all::ShellSortALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(output_, reinterpret_cast<int *>(task_data->outputs[0]));
  }

  return true;
}
