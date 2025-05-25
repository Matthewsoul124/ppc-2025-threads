#include "all/odintsov_m_multmatrix_cannon/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>  // для boost::mpi::broadcast
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

using namespace std;
void odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::ShiftRow(std::vector<double>& matrix, int root, int row,
                                                                   int shift) {
  shift = shift % root;
  std::vector<double> tmp(root);
  for (int j = 0; j < root; j++) {
    tmp[j] = matrix[(row * root) + ((j + shift) % root)];
  }
  for (int j = 0; j < root; j++) {
    matrix[(row * root) + j] = tmp[j];
  }
}

void odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::ShiftColumn(std::vector<double>& matrix, int root, int col,
                                                                      int shift) {
  shift = shift % root;
  std::vector<double> tmp(root);

  for (int i = 0; i < root; i++) {
    tmp[i] = matrix[(((i + shift) % root) * root) + col];
  }
  for (int i = 0; i < root; i++) {
    matrix[(i * root) + col] = tmp[i];
  }
}
void odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::ShiftBlocksUp(std::vector<double>& matrix, int root,
                                                                        int sz) const {
  int p = root / block_sz_;
  for (int bj = 0; bj < p; bj++) {
    std::vector<double> first_block(block_sz_ * block_sz_);

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        first_block[(i * block_sz_) + j] = matrix[(i * root) + ((bj * block_sz_) + j)];
      }
    }

    for (int bi = 0; bi < (p - 1); bi++) {
      for (int i = 0; i < block_sz_; i++) {
        for (int j = 0; j < block_sz_; j++) {
          matrix[((bi * block_sz_ + i) * root) + (bj * block_sz_) + j] =
              matrix[(((bi + 1) * block_sz_) * root) + (i * root) + ((bj * block_sz_) + j)];
        }
      }
    }

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        matrix[((((p - 1) * block_sz_) * root) + (i * root) + ((bj * block_sz_) + j))] =
            first_block[(i * block_sz_) + j];
      }
    }
  }
}

void odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::ShiftBlocksLeft(std::vector<double>& matrix, int root,
                                                                          int sz) const {
  int p = root / block_sz_;
  for (int bi = 0; bi < p; bi++) {
    std::vector<double> first_block(block_sz_ * block_sz_);

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        first_block[(i * block_sz_) + j] = matrix[((bi * block_sz_ + i) * root) + j];
      }
    }

    for (int bj = 0; bj < (p - 1); bj++) {
      for (int i = 0; i < block_sz_; i++) {
        for (int j = 0; j < block_sz_; j++) {
          matrix[((bi * block_sz_ + i) * root) + (bj * block_sz_) + j] =
              matrix[((bi * block_sz_ + i) * root) + (((bj + 1) * block_sz_) + j)];
        }
      }
    }

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        matrix[((bi * block_sz_ + i) * root) + (((p - 1) * block_sz_) + j)] = first_block[(i * block_sz_) + j];
      }
    }
  }
}

bool odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::IsSquere(unsigned int num) {
  auto root = static_cast<unsigned int>(std::sqrt(num));
  return (root * root) == num;
}

int odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::GetBlockSize(int n) {
  for (int k = (n / 2); k >= 2; k--) {
    if ((n % k) == 0) {
      return k;
    }
  }
  return 1;
}

void odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::InitializeShift(std::vector<double>& matrix, int root,
                                                                          int grid_size, int block_sz,
                                                                          bool is_row_shift) {
  for (int b = 0; b < grid_size; ++b) {
    for (int index = b * block_sz; index < (b + 1) * block_sz; ++index) {
      for (int shift = 0; shift < b; ++shift) {
        if (is_row_shift) {
          ShiftRow(matrix, root, index, block_sz);
        } else {
          ShiftColumn(matrix, root, index, block_sz);
        }
      }
    }
  }
}

void odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::ProcessBlockMul(int bi, int bj_start, int bj_end, int root,
                                                                          int block_sz,
                                                                          const std::vector<double>& matrix_a,
                                                                          const std::vector<double>& matrix_b,
                                                                          std::vector<double>& local_c) {
  std::vector<double> a_block(block_sz * block_sz);
  std::vector<double> b_block(block_sz * block_sz);

  for (int bj = bj_start; bj < bj_end; ++bj) {
    // Копирование подблока
    for (int i = 0; i < block_sz; ++i) {
      for (int j = 0; j < block_sz; ++j) {
        int row = (bi * block_sz) + i;
        int col = (bj * block_sz) + j;
        a_block[(i * block_sz) + j] = matrix_a[(row * root) + col];
        b_block[(i * block_sz) + j] = matrix_b[(row * root) + col];
      }
    }

    // Перемножение и накопление
    for (int i = 0; i < block_sz; ++i) {
      for (int k = 0; k < block_sz; ++k) {
        double a_ik = a_block[(i * block_sz) + k];
        int base = (((bi * block_sz) + i) * root) + (bj * block_sz);
        for (int j = 0; j < block_sz; ++j) {
          local_c[base + j] += a_ik * b_block[(k * block_sz) + j];
        }
      }
    }
  }
}

void odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::ProcessBlockSTL(int bi, int num_blocks, int root,
                                                                          int block_sz,
                                                                          const std::vector<double>& matrix_a,
                                                                          const std::vector<double>& matrix_b,
                                                                          std::vector<double>& local_c) {
  int max_threads = ppc::util::GetPPCNumThreads();
  int tcount = std::min(max_threads, num_blocks);
  std::vector<thread> threads;
  threads.reserve(tcount);

  for (int t = 0; t < tcount; ++t) {
    int bj_start = (num_blocks * t) / tcount;
    int bj_end = (num_blocks * (t + 1)) / tcount;
    threads.emplace_back(&MulMatrixCannonALL::ProcessBlockMul, bi, bj_start, bj_end, root, block_sz,
                         std::cref(matrix_a), std::cref(matrix_b), std::ref(local_c));
  }

  for (auto& th : threads) {
    th.join();
  }
}

bool odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::PreProcessingImpl() {
  if (com_.rank() == 0) {
    szA_ = task_data->inputs_count[0];
    szB_ = task_data->inputs_count[1];
    matrixA_.assign(reinterpret_cast<double*>(task_data->inputs[0]),
                    reinterpret_cast<double*>(task_data->inputs[0]) + szA_);
    matrixB_.assign(reinterpret_cast<double*>(task_data->inputs[1]),
                    reinterpret_cast<double*>(task_data->inputs[1]) + szB_);
    matrixC_.assign(szA_, 0);

    block_sz_ = GetBlockSize(static_cast<int>(sqrt(szA_)));
  }
  return true;
}

bool odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::ValidationImpl() {
  if (com_.rank() == 0) {
    if (task_data->inputs_count[0] != task_data->inputs_count[1]) {
      return false;
    }

    if ((!(IsSquere(task_data->inputs_count[0]))) || (!(IsSquere(task_data->inputs_count[1])))) {
      return false;
    }
  }
  return true;
}

bool odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::RunImpl() {
  int rank = com_.rank();
  int size = com_.size();

  // Считываем параметры на всех рангах
  boost::mpi::broadcast(com_, szA_, /*root=*/0);
  boost::mpi::broadcast(com_, block_sz_, /*root=*/0);

  int count = static_cast<int>(szA_);
  if (rank != 0) {
    matrixA_.resize(count);
    matrixB_.resize(count);
    matrixC_.resize(count);
  }
  // Вычисляем размер матрицы (root × root) и параметры блоков
  int root = static_cast<int>(std::round(std::sqrt(szA_)));
  int num_blocks = std::max(1, root / block_sz_);
  int steps = num_blocks;

  // 1) Начальная инициализация сдвигов на rank 0
  if (rank == 0) {
    InitializeShift(matrixA_, root, steps, block_sz_, /*is_row_shift=*/true);
    InitializeShift(matrixB_, root, steps, block_sz_, /*is_row_shift=*/false);
  }

  // 2) Рассылка первоначально сдвинутых A и B всем процессам
  boost::mpi::broadcast(com_, matrixA_.data(), count, /*root=*/0);
  boost::mpi::broadcast(com_, matrixB_.data(), count, /*root=*/0);

  // 3) Локальный накопитель результата за все шаги
  std::vector<double> local_c_acc(root * root, 0.0);

  for (int step = 0; step < steps; ++step) {
    // 3.1) Локальный вектор для этого шага
    std::vector<double> local_c(root * root, 0.0);

    // 3.2) Каждый процесс обрабатывает блоки bi = rank, rank+size, …
    for (int bi = rank; bi < num_blocks; bi += size) {
      ProcessBlockSTL(bi, num_blocks, root, block_sz_, matrixA_, matrixB_, local_c);
    }

    // 3.3) Аккумулируем во внешний local_c_acc
    for (int i = 0; i < root * root; ++i) {
      local_c_acc[i] += local_c[i];
    }

    // 3.4) Сдвиги перед следующим шагом — только на rank 0
    if (rank == 0) {
      ShiftBlocksLeft(matrixA_, root, block_sz_);
      ShiftBlocksUp(matrixB_, root, block_sz_);
    }

    // 3.5) Рассылка обновлённых A и B
    boost::mpi::broadcast(com_, matrixA_.data(), count, /*root=*/0);
    boost::mpi::broadcast(com_, matrixB_.data(), count, /*root=*/0);
  }

  // 4) Подготавливаем matrixC_ только на rank 0
  if (rank == 0) {
    matrixC_.assign(root * root, 0.0);
  }

  // 5) Сводим накопленные данные в matrixC_ на rank 0
  boost::mpi::reduce(com_, local_c_acc.data(), root * root, matrixC_.data(), std::plus<>(), 0);

  return true;
}

bool odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL::PostProcessingImpl() {
  if (com_.rank() == 0) {
    std::size_t sz_c = matrixC_.size();
    for (std::size_t i = 0; i < sz_c; i++) {
      reinterpret_cast<double*>(task_data->outputs[0])[i] = matrixC_[i];
    }
  }
  return true;
}
