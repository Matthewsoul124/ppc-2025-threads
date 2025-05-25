#include "all/kolokolova_d_integral_simpson_method/include/ops_all.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    nums_variables_ = int(task_data->inputs_count[0]);

    steps_ = std::vector<int>(task_data->inputs_count[0]);
    auto* input_steps = reinterpret_cast<int*>(task_data->inputs[0]);
    for (unsigned i = 0; i < task_data->inputs_count[0]; i++) {
      steps_[i] = input_steps[i];
    }

    borders_ = std::vector<int>(task_data->inputs_count[1]);
    auto* input_borders = reinterpret_cast<int*>(task_data->inputs[1]);
    for (unsigned i = 0; i < task_data->inputs_count[1]; i++) {
      borders_[i] = input_borders[i];
    }

    result_output_ = 0;

    // Find size of step
    size_step_.resize(nums_variables_);
    CalculateStepSizes();

    //  Create vector of points
    points_.resize(nums_variables_);
    CreatePointsVector();

    results_func_.resize(int(points_.size()));
    coeff_.resize(steps_[0]);
    results_func_ = FindFunctionValue(points_, func_);
    coeff_ = FindCoeff(steps_[0]);

    PrepareCoefficientsAndResults();

    size_local_results_func_ = int(results_func_.size()) / world_.size();
    size_local_coeff_ = int(vec_coeff_.size()) / world_.size();
    size_local_size_step_ = int(size_step_.size());
  }
  return true;
}

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::ValidationImpl() {
  // Check inputs and outputs
  if (world_.rank() == 0) {
    std::vector<int> bord = std::vector<int>(task_data->inputs_count[1]);
    auto* input_bord = reinterpret_cast<int*>(task_data->inputs[1]);
    for (unsigned i = 0; i < task_data->inputs_count[1]; i++) {
      bord[i] = input_bord[i];
    }
    int num_var = int(task_data->inputs_count[0]);
    int num_bord = int(task_data->inputs_count[1]) / 2;
    return (task_data->inputs_count[0] != 0 && task_data->inputs_count[1] != 0 && task_data->outputs_count[0] != 0 &&
            CheckBorders(bord) && num_var == num_bord);
  }
  return true;
}

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  // Send vector sizes to all processes
  boost::mpi::broadcast(world_, size_local_results_func_, 0);
  boost::mpi::broadcast(world_, nums_variables_, 0);
  boost::mpi::broadcast(world_, size_local_coeff_, 0);
  boost::mpi::broadcast(world_, size_local_size_step_, 0);

  local_results_func_ = std::vector<double>(size_local_results_func_);
  local_coeff_ = std::vector<double>(size_local_coeff_);

  // Sending vectors with coefficients and function values
  if (rank == 0) {
    for (int proc = 1; proc < size; proc++) {
      world_.communicator::send(proc, 0, results_func_.data() + (proc * size_local_results_func_),
                                size_local_results_func_);
    }
    local_results_func_ = std::vector<double>(results_func_.begin(), results_func_.begin() + size_local_results_func_);
  } else {
    world_.communicator::recv(0, 0, local_results_func_.data(), size_local_results_func_);
  }

  if (rank == 0) {
    for (int proc = 1; proc < size; proc++) {
      world_.communicator::send(proc, 0, vec_coeff_.data() + (proc * size_local_coeff_), size_local_coeff_);
    }
    local_coeff_ = std::vector<double>(vec_coeff_.begin(), vec_coeff_.begin() + size_local_coeff_);
  } else {
    world_.communicator::recv(0, 0, local_coeff_.data(), size_local_coeff_);
  }

  int function_vec_size = int(local_results_func_.size());

  // Multiplication by coefficients
  tbb::parallel_for(0, function_vec_size,
                    [&](int i) { local_results_func_[i] *= local_coeff_[i % local_coeff_.size()]; });

  boost::mpi::gather(world_, local_results_func_.data(), size_local_results_func_, results_func_, 0);

  if (rank == 0) {
    ApplyCoefficientIteration();
  }

  local_results_func_ = std::vector<double>(size_local_results_func_);
  local_size_step_ = std::vector<double>(size_local_size_step_);

  if (rank == 0) {
    for (int proc = 1; proc < size; proc++) {
      world_.communicator::send(proc, 0, results_func_.data() + (proc * size_local_results_func_),
                                size_local_results_func_);
    }
    local_results_func_ = std::vector<double>(results_func_.begin(), results_func_.begin() + size_local_results_func_);
  } else {
    world_.communicator::recv(0, 0, local_results_func_.data(), size_local_results_func_);
  }

  if (rank == 0) {
    for (int proc = 1; proc < size; proc++) {
      world_.communicator::send(proc, 0, size_step_.data(), size_local_size_step_);
    }
    local_size_step_ = std::vector<double>(size_step_.begin(), size_step_.begin() + size_local_size_step_);
  } else {
    world_.communicator::recv(0, 0, local_size_step_.data(), size_local_size_step_);
  }

  // Formation of the result
  double sum = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, local_results_func_.size()), 0.0,
      [&](const tbb::blocked_range<size_t>& r, double init) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          init += local_results_func_[i];
        }
        return init;
      },
      std::plus<>());
  local_results_output_ += sum;

  for (size_t i = 0; i < local_size_step_.size(); i++) {
    local_results_output_ *= local_size_step_[i];
  }

  boost::mpi::reduce(world_, local_results_output_, result_output_, std::plus(), 0);

  if (rank == 0) {
    // divided by 3 to the power
    result_output_ /= pow(3, nums_variables_);
  }
  return true;
}

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result_output_;
  }
  return true;
}

std::vector<double> kolokolova_d_integral_simpson_method_all::TestTaskALL::FindFunctionValue(
    const std::vector<std::vector<double>>& coordinates, const std::function<double(std::vector<double>)>& f) {
  std::vector<double> results;                                     // result of function
  std::vector<double> current;                                     // current point
  GeneratePointsAndEvaluate(coordinates, 0, current, results, f);  // recursive function
  return results;
}

void kolokolova_d_integral_simpson_method_all::TestTaskALL::GeneratePointsAndEvaluate(
    const std::vector<std::vector<double>>& coordinates, int index, std::vector<double>& current,
    std::vector<double>& results, const std::function<double(const std::vector<double>)>& f) {
  // if it the end of vector
  if (index == int(coordinates.size())) {
    double result = f(current);  // find value of function
    results.push_back(result);   // save result
    return;
  }

  // sort through the coordinates
  for (double coord : coordinates[index]) {
    current.push_back(coord);
    GeneratePointsAndEvaluate(coordinates, index + 1, current, results, func_);  // recursive
    current.pop_back();                                                          // delete for next coordinat
  }
}

std::vector<double> kolokolova_d_integral_simpson_method_all::TestTaskALL::FindCoeff(int count_step) {
  std::vector<double> result_coeff(1, 1.0);  // first coeff is always 1
  for (int i = 1; i < count_step; i++) {
    if (i % 2 != 0) {
      result_coeff.push_back(4.0);  // odd coeff is 4
    } else {
      result_coeff.push_back(2.0);  // even coeff is 2
    }
  }
  result_coeff.push_back(1.0);  // last coeff is always 1
  return result_coeff;
}

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::CheckBorders(std::vector<int> vec) {
  size_t i = 0;
  while (i < vec.size()) {
    if (vec[i] > vec[i + 1]) {
      return false;
    }
    i += 2;
  }
  return true;
}

void kolokolova_d_integral_simpson_method_all::TestTaskALL::CalculateStepSizes() {
  tbb::parallel_for(tbb::blocked_range<int>(0, nums_variables_), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      double a = (double((borders_[(2 * i) + 1] - borders_[2 * i])) / double(steps_[i]));
      size_step_[i] = a;
    }
  });
}

void kolokolova_d_integral_simpson_method_all::TestTaskALL::CreatePointsVector() {
  tbb::parallel_for(tbb::blocked_range<int>(0, nums_variables_), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      std::vector<double> vec;
      for (int j = 0; j < steps_[i] + 1; ++j) {
        auto num = double(borders_[2 * i] + (double(j) * size_step_[i]));
        vec.push_back(num);
      }
      points_[i] = vec;
    }
  });
}

void kolokolova_d_integral_simpson_method_all::TestTaskALL::PrepareCoefficientsAndResults() {
  vec_coeff_ = std::vector<double>(int(results_func_.size()));
  for (int i = 0; i < int(vec_coeff_.size()); ++i) {
    vec_coeff_[i] = coeff_[i % int(coeff_.size())];
  }
  while (int(results_func_.size()) % world_.size() != 0) {
    results_func_.push_back(0.0);
  }
  while (int(vec_coeff_.size()) % world_.size() != 0) {
    vec_coeff_.push_back(0.0);
  }
}

void kolokolova_d_integral_simpson_method_all::TestTaskALL::ApplyCoefficientIteration() {
  for (int iteration = 1; iteration < nums_variables_; ++iteration) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, results_func_.size()), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        int block_size = iteration * int(coeff_.size());
        int current_n_index = (int(i) / block_size) % int(coeff_.size());
        results_func_[i] *= coeff_[current_n_index];
      }
    });
  }
}
