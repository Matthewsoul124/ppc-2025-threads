#include "all/kharin_m_multidimensional_integral_calc/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cstddef>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/reduce.hpp"
#include "core/util/include/util.hpp"

bool kharin_m_multidimensional_integral_calc_all::TaskALL::ValidationImpl() {
  bool is_valid = true;
  if (world_.rank() == 0) {
    // Объединяем условия с одинаковым телом в одно составное условие
    if (task_data->inputs.size() != 3 || task_data->outputs.size() != 1 ||
        task_data->inputs_count[1] != task_data->inputs_count[2] || task_data->outputs_count[0] != 1) {
      is_valid = false;
    }
  }
  boost::mpi::broadcast(world_, is_valid, 0);
  return is_valid;
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* input_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    size_t input_size = task_data->inputs_count[0];
    input_ = std::vector<double>(input_ptr, input_ptr + input_size);
    auto* sizes_ptr = reinterpret_cast<size_t*>(task_data->inputs[1]);
    size_t d = task_data->inputs_count[1];
    grid_sizes_ = std::vector<size_t>(sizes_ptr, sizes_ptr + d);
    auto* steps_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
    step_sizes_ = std::vector<double>(steps_ptr, steps_ptr + d);
  }
  return true;
}

double kharin_m_multidimensional_integral_calc_all::TaskALL::ComputeLocalSum() {
  if (local_input_.empty()) {
    return 0.0;
  }

  // Определение количества потоков
  num_threads_ = std::min(static_cast<size_t>(ppc::util::GetPPCNumThreads()), local_input_.size());
  if (num_threads_ == 0) {
    return 0.0;  // Дополнительная проверка
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads_);
  std::vector<double> partial_sums(num_threads_, 0.0);

  // Распределение работы между потоками
  auto input_chunk_size = local_input_.size() / num_threads_;
  auto remainder = local_input_.size() % num_threads_;

  auto chunk_plus = [&](std::vector<double>::iterator it_begin, size_t size, double& result_location) {
    double local = 0.0;
    for (size_t i = 0; i < size; ++i) {
      local += *(it_begin + static_cast<std::vector<double>::difference_type>(i));
    }
    result_location = local;
  };

  size_t current_start_index = 0;
  for (size_t i = 0; i < num_threads_; ++i) {
    size_t size = (i < remainder) ? (input_chunk_size + 1) : input_chunk_size;
    auto it_begin = local_input_.begin() + static_cast<std::vector<double>::difference_type>(current_start_index);
    std::thread th(chunk_plus, it_begin, size, std::ref(partial_sums[i]));
    threads.push_back(std::move(th));
    current_start_index += size;
  }

  // Ожидание завершения потоков
  for (auto& th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  // Суммирование частичных результатов
  double local_sum = 0.0;
  for (const auto& partial : partial_sums) {
    local_sum += partial;
  }

  return local_sum;
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::RunImpl() {
  // Рассылка grid_sizes_ и step_sizes_ всем процессам
  boost::mpi::broadcast(world_, grid_sizes_, 0);
  boost::mpi::broadcast(world_, step_sizes_, 0);
  // Проверка шагов
  bool local_steps_valid = std::ranges::all_of(step_sizes_, [](double h) { return h > 0.0; });
  bool all_steps_valid = false;
  boost::mpi::all_reduce(world_, local_steps_valid, all_steps_valid, std::logical_and<>());
  if (!all_steps_valid) {
    if (world_.rank() == 0) {
      output_result_ = 0.0;
    }
    return false;
  }

  // Распределение данных
  size_t total_size = 1;
  for (auto n : grid_sizes_) {
    total_size *= n;
  }
  size_t p = world_.size();
  size_t chunk_size = total_size / p;
  size_t remainder = total_size % p;
  size_t rank = world_.rank();
  size_t local_size = (rank < remainder) ? chunk_size + 1 : chunk_size;
  local_input_.resize(local_size);

  if (world_.rank() == 0) {
    std::vector<int> send_counts(p);
    std::vector<int> displacements(p);
    size_t offset = 0;
    for (size_t i = 0; i < p; ++i) {
      size_t size = (i < remainder) ? chunk_size + 1 : chunk_size;
      send_counts[i] = static_cast<int>(size);
      displacements[i] = static_cast<int>(offset);
      offset += size;
    }
    boost::mpi::scatterv(world_, input_, send_counts, displacements, local_input_.data(),
                         static_cast<int>(local_input_.size()), 0);
  } else {
    boost::mpi::scatterv(world_, local_input_.data(), static_cast<int>(local_input_.size()), 0);
  }

  double local_sum = ComputeLocalSum();
  double total_sum = 0.0;
  boost::mpi::reduce(world_, local_sum, total_sum, std::plus<>(), 0);

  if (world_.rank() == 0) {
    double volume_element = 1.0;
    for (const auto& h : step_sizes_) {
      volume_element *= h;
    }
    output_result_ = total_sum * volume_element;
  }
  return true;
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::PostProcessingImpl() {
  // Отправка результата всем процессам
  boost::mpi::broadcast(world_, output_result_, 0);

  // Запись результата в выходные данные
  if (!task_data->outputs.empty() && !task_data->outputs_count.empty() && task_data->outputs_count[0] > 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = output_result_;
  }

  return true;
}