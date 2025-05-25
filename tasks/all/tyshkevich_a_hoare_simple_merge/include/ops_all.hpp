#pragma once

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>  // NOLINT
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace tyshkevich_a_hoare_simple_merge_all {

template <typename T>
struct ArrayPiece {
  T* first;
  T* last;

  [[nodiscard]] std::size_t Size() const noexcept { return last - first; }

  operator std::span<T>() const noexcept { return std::span<T>(first, last); }

  void Concat(const ArrayPiece& other) { last = other.last; }

  void Send(boost::mpi::communicator& comm, int dest) {
    comm.send(dest, 0, Size());
    comm.send(dest, 0, first, Size());
  }
  static void Recv(boost::mpi::communicator& comm, int source, std::vector<T>& out) {
    decltype(ArrayPiece{}.Size()) size{};
    comm.recv(0, 0, size);

    const auto divsz = out.size();
    out.resize(divsz + size);

    comm.recv(source, 0, out.data() + divsz, size);
  }

  static std::vector<ArrayPiece> Partition(std::span<T> arr, std::size_t pieces) {
    const std::size_t delta = arr.size() / pieces;
    const std::size_t extra = arr.size() % pieces;

    std::vector<ArrayPiece> v(pieces);
    auto* cur = arr.data();
    for (std::size_t i = 0; i < pieces; i++) {
      const std::size_t sz = delta + ((i < extra) ? 1 : 0);
      v[i] = {.first = cur, .last = cur + sz};
      cur += sz;
    }

    return v;
  }
};

template <typename T, typename Comparator>
class HoareSortTask : public ppc::core::Task {
 public:
  explicit HoareSortTask(ppc::core::TaskDataPtr task_data, Comparator cmp) : Task(std::move(task_data)), cmp_(cmp) {}

  bool ValidationImpl() override {
    return world_.rank() != 0 || (task_data->inputs_count[0] == task_data->outputs_count[0]);
  }

  bool PreProcessingImpl() override {
    if (world_.rank() == 0) {
      input_ = {reinterpret_cast<const T*>(task_data->inputs[0]), task_data->inputs_count[0]};
      output_ = {reinterpret_cast<T*>(task_data->outputs[0]), task_data->outputs_count[0]};
    }
    return true;
  }

  void ParallelSort(std::span<T> arr) {
    const std::size_t concurrency = std::min(arr.size(), std::size_t(ppc::util::GetPPCNumThreads()));
    if (concurrency == 0) {
      return;
    }
    tbb::task_arena arena(concurrency);

    auto pieces = ArrayPiece<T>::Partition(arr, concurrency);

    arena.execute([&] {
      oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, concurrency, 1), [&](const auto& r) {
        for (std::size_t tnum = r.begin(); tnum < r.end(); tnum++) {
          auto& piece = pieces[tnum];
          HoareSort(piece, 0, piece.Size() - 1);
        }
      });
    });

    std::size_t leap = 1;
    for (std::size_t reduc = concurrency; reduc > 1; reduc /= 2) {
      const auto dl = 2 * leap;
      const auto nextreduc = reduc / 2;

      arena.execute([&] {
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<std::size_t>(0, nextreduc,
                                                    (pieces[0].Size() > kParallelizeThreshold)
                                                        ? nextreduc / tbb::this_task_arena::max_concurrency()
                                                        : nextreduc),
            [&](const tbb::blocked_range<std::size_t>& r) {
              for (std::size_t g = r.begin(); g < r.end(); ++g) {
                auto& head = pieces[dl * g];
                auto& tail = pieces[(dl * g) + leap];
                std::inplace_merge(head.first, tail.first, tail.last, cmp_);
                head.Concat(tail);
              }
            });
      });
      if (nextreduc % 2 != 0) {
        ArrayPiece<T>* head = nullptr;
        ArrayPiece<T>* tail = nullptr;
        if (nextreduc == 1) {
          head = &pieces.front();
          tail = &pieces.back();
        } else {
          head = &pieces[dl * (nextreduc - 2)];
          tail = &pieces[dl * (nextreduc - 1)];
        }
        std::inplace_merge(head->first, tail->first, tail->last, cmp_);
        head->Concat(*tail);
      }

      leap *= 2;
    }
  }

  bool RunImpl() override {
    std::size_t overall_size{};

    if (world_.rank() == 0) {
      overall_size = input_.size();
      std::copy(input_.begin(), input_.end(), output_.begin());
    }
    boost::mpi::broadcast(world_, overall_size, 0);  // NOLINT(misc-include-cleaner)

    const int procs = std::min(int(overall_size), world_.size());
    const int rank = world_.rank();

    if (rank >= procs) {
      world_.split(0);
      return true;
    }
    auto comm = world_.split(1);

    piece_.clear();
    if (comm.rank() == 0) {
      auto pieces = ArrayPiece<T>::Partition(output_, procs);
      if (pieces.size() == 0) {
        return true;
      }
      for (std::size_t i = 1; i < pieces.size(); i++) {
        pieces[i].Send(comm, i);
      }
      piece_.assign(pieces[0].first, pieces[0].last);
    } else {
      ArrayPiece<T>::Recv(comm, 0, piece_);
    }

    ParallelSort(piece_);

    for (int i = 1; i < procs; i *= 2) {
      if (comm.rank() % (2 * i) == 0) {
        const auto tail = comm.rank() + i;
        if (tail >= procs) {
          continue;
        }
        int size{};
        comm.recv(int(tail), 0, size);

        const auto divsz = piece_.size();
        piece_.resize(divsz + size);
        comm.recv(int(tail), 0, piece_.data() + divsz, size);

        std::inplace_merge(piece_.begin(), piece_.begin() + std::int64_t(divsz), piece_.end(), cmp_);
      } else if ((comm.rank() % i) == 0) {
        const auto size = int(piece_.size());
        const auto head = comm.rank() - i;
        comm.send(int(head), 0, size);
        comm.send(int(head), 0, piece_.data(), size);
        break;
      }
    }

    return true;
  }

  bool PostProcessingImpl() override {
    if (world_.rank() == 0) {
      std::ranges::copy(piece_, reinterpret_cast<T*>(task_data->outputs[0]));
    }
    return true;
  }

 private:
  void HoareSort(std::span<T> arr, int64_t low, int64_t high) {
    const auto partition = [&cmp = this->cmp_](std::span<T> region, int64_t plo, int64_t phi) -> int64_t {
      const auto& pivot = region[phi];
      int64_t e = plo - 1;
      for (int64_t j = plo; j <= phi - 1; j++) {
        if (cmp(region[j], pivot)) {
          std::swap(region[++e], region[j]);
        }
      }
      std::swap(region[e + 1], region[phi]);
      return e + 1;
    };

    if (low < high) {
      int64_t p = partition(arr, low, high);
      HoareSort(arr, low, p - 1);
      HoareSort(arr, p + 1, high);
    }
  };

  Comparator cmp_;

  std::span<const T> input_;
  std::span<T> output_;
  std::vector<T> piece_;

  boost::mpi::communicator world_;

  static constexpr std::size_t kParallelizeThreshold = 32;
};

template <typename T, typename Comparator>
HoareSortTask<T, Comparator> CreateHoareTestTask(ppc::core::TaskDataPtr task_data, Comparator cmp) {
  return HoareSortTask<T, Comparator>(std::move(task_data), cmp);
}

}  // namespace tyshkevich_a_hoare_simple_merge_all