#include "stl/oturin_a_gift_wrapping/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

double oturin_a_gift_wrapping_stl::ABTP(Coord a, Coord b, Coord c) {
  Coord ab = {.x = b.x - a.x, .y = b.y - a.y};
  Coord cb = {.x = b.x - c.x, .y = b.y - c.y};
  double dot = ((ab.x * cb.x) + (ab.y * cb.y));
  double cross = ((ab.x * cb.y) - (ab.y * cb.x));
  return fabs(atan2(cross, dot));
}

double oturin_a_gift_wrapping_stl::ABTP(Coord a, Coord c) {
  Coord b{.x = a.x, .y = (a.y - 1)};
  return ABTP(b, a, c);
}

double oturin_a_gift_wrapping_stl::Distance(Coord a, Coord b) {
  int t1 = a.x - b.x;
  int t2 = a.y - b.y;
  return sqrt((t1 * t1) + (t2 * t2));
}

bool oturin_a_gift_wrapping_stl::TestTaskSTL::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<Coord *>(task_data->inputs[0]);
  input_ = std::vector<Coord>(in_ptr, in_ptr + input_size);
  n_ = int(input_.size());
  output_ = std::vector<Coord>();
  output_.reserve(n_);

  // check if all points are same
  auto are_same = [&](const auto &p) { return p == input_[0]; };
  return !std::ranges::all_of(input_.begin(), input_.end(), are_same);
}

bool oturin_a_gift_wrapping_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] >= 3;  // task requires 3 or more points to wrap
}

#ifdef __linux__
bool oturin_a_gift_wrapping_stl::TestTaskSTL::RunImpl() {
  if (!output_.empty()) {
    output_.clear();
  }
  // this .clear() used ONLY for perftest TaskRun. for some reason output_ has something in it

  // find most left point (priority to top)
  int start_index = FindMostLeft();
  output_.push_back(input_[start_index]);

  int search_index = 0;
  FindSecondPoint(start_index, search_index);

  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads() - 1);  // -1 because main thread will not be here
  std::vector<std::pair<int, double>> thread_results(threads.size());
  std::size_t thread_block = 0;
  if (!threads.empty()) {
    thread_block = input_.size() / (threads.size() + 1);
    for (int i = 0; i < (int)threads.size(); i++) {
      thread_results[i].first = 10;
      threads[i] = std::thread(
          [&](const int i) { SearchThreadP(i * thread_block, (i + 1) * thread_block, thread_results[i]); }, i);
    }
  }

  // main loop
  do {
    output_.push_back(input_[search_index]);

    for (auto &i : thread_results) {
      i.first = -1;  // start work
    }

    std::pair<int, double> main_thread_result = SearchThread(thread_block * threads.size(), input_.size());

    double reference_abtp = main_thread_result.second;
    double reference_distance = Distance(output_.back(), input_[main_thread_result.first]);
    search_index = main_thread_result.first;

    double secondary_abtp = NAN;
    double secondary_distance = NAN;

    while (true) {
      if (std::ranges::all_of(thread_results.cbegin(), thread_results.cend(), [](auto &i) { return i.first >= 0; })) {
        break;
      }
      std::this_thread::yield();
    }

    for (int i = 0; i < (int)threads.size(); i++) {
      secondary_abtp = thread_results[i].second;
      if (secondary_abtp > reference_abtp) {
        reference_abtp = secondary_abtp;
        reference_distance = Distance(output_.back(), input_[thread_results[i].first]);
        search_index = thread_results[i].first;
      } else if (reference_abtp == secondary_abtp) {
        secondary_distance = Distance(output_.back(), input_[thread_results[i].first]);
        if (reference_distance > secondary_distance) {
          reference_abtp = secondary_abtp;
          reference_distance = secondary_distance;
          search_index = thread_results[i].first;
        }
      }
    }

  } while (search_index != start_index);

  for (int i = 0; i < (int)threads.size(); i++) {
    thread_results[i].first = -2;
    threads[i].join();
  }
  return true;
}

#else
bool oturin_a_gift_wrapping_stl::TestTaskSTL::RunImpl() {
  if (!output_.empty()) {
    output_.clear();
  }
  // this .clear() used ONLY for perftest TaskRun. for some reason output_ has something in it

  // find most left point (priority to top)
  int start_index = FindMostLeft();
  output_.push_back(input_[start_index]);

  int search_index = 0;
  FindSecondPoint(start_index, search_index);

  std::vector<std::thread> threads(ppc::util::GetPPCNumThreads() - 1);  // -1 because main thread will not be here
  std::vector<std::pair<int, double>> thread_results(threads.size());
  std::size_t thread_block = 0;
  if (!threads.empty()) {
    thread_block = input_.size() / (threads.size() + 1);
  }

  // main loop
  do {
    output_.push_back(input_[search_index]);

    for (int i = 0; i < (int)threads.size(); i++) {
      threads[i] = std::thread(
          [&](const int i) { thread_results[i] = SearchThread(i * thread_block, (i + 1) * thread_block); }, i);
    }

    std::pair<int, double> main_thread_result = SearchThread(thread_block * threads.size(), input_.size());

    double reference_abtp = main_thread_result.second;
    double reference_distance = Distance(output_.back(), input_[main_thread_result.first]);
    search_index = main_thread_result.first;

    double secondary_abtp = NAN;
    double secondary_distance = NAN;

    for (int i = 0; i < (int)threads.size(); i++) {
      threads[i].join();
      secondary_abtp = thread_results[i].second;
      if (secondary_abtp > reference_abtp) {
        reference_abtp = secondary_abtp;
        reference_distance = Distance(output_.back(), input_[thread_results[i].first]);
        search_index = thread_results[i].first;
      } else if (reference_abtp == secondary_abtp) {
        secondary_distance = Distance(output_.back(), input_[thread_results[i].first]);
        if (reference_distance > secondary_distance) {
          reference_abtp = secondary_abtp;
          reference_distance = secondary_distance;
          search_index = thread_results[i].first;
        }
      }
    }

  } while (search_index != start_index);

  return true;
}
#endif

void oturin_a_gift_wrapping_stl::TestTaskSTL::FindSecondPoint(int start_index, int &search_index) {
  double line_angle = -std::numbers::pi;
  for (int i = 0; i < n_; i++) {
    if (i == start_index) {
      continue;
    }
    double t = ABTP(input_[start_index], input_[i]);
    if (t > line_angle || (t == line_angle && Distance(input_[start_index], input_[i]) <
                                                  Distance(input_[start_index], input_[search_index]))) {
      search_index = i;
      line_angle = t;
    }
  }
}

#ifdef __linux__
void oturin_a_gift_wrapping_stl::TestTaskSTL::SearchThreadP(std::size_t start, std::size_t end,
                                                            std::pair<int, double> &thread_result) {
  while (thread_result.first != -2) {
    std::this_thread::yield();
    if (thread_result.first != -1) {
      continue;
    }
    thread_result = SearchThread(start, end);
  }
}
#endif

std::pair<int, double> oturin_a_gift_wrapping_stl::TestTaskSTL::SearchThread(std::size_t start, std::size_t end) {
  double line_angle = -std::numbers::pi;
  std::size_t search_index = start;
  for (std::size_t i = start; i < end; i++) {
    double t = ABTP(output_[output_.size() - 2], output_.back(), input_[i]);
    PointSearch(t, line_angle, search_index, i);
  }

  return {(int)search_index, line_angle};
}

bool oturin_a_gift_wrapping_stl::TestTaskSTL::PostProcessingImpl() {
  auto *result_ptr = reinterpret_cast<Coord *>(task_data->outputs[0]);
  std::ranges::copy(output_.begin(), output_.end(), result_ptr);
  return true;
}

int oturin_a_gift_wrapping_stl::TestTaskSTL::FindMostLeft() {
  Coord most_left = input_[0];
  int start_index = 0;
  for (int i = 1; i < n_; i++) {
    if (input_[i].x < most_left.x || (input_[i].x == most_left.x && input_[i].y > most_left.y)) {
      start_index = i;
      most_left = input_[i];
    }
  }
  return start_index;
}

void oturin_a_gift_wrapping_stl::TestTaskSTL::PointSearch(const double t, double &line_angle, std::size_t &search_index,
                                                          const std::size_t i) {
  if (t < line_angle) {
    return;
  }
  if (output_.back() != input_[i] && output_[output_.size() - 2] != input_[i]) {
    if (t == line_angle && Distance(output_.back(), input_[i]) >= Distance(output_.back(), input_[search_index])) {
      return;
    }
    search_index = i;
    line_angle = t;
  }
}
