#include "tbb/odintsov_m_multmatrix_cannon/include/ops_tbb.hpp"

#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

using namespace std;
void odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::ShiftRow(std::vector<double>& matrix, int root, int row,
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

void odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::ShiftColumn(std::vector<double>& matrix, int root, int col,
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
void odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::ShiftBlocksUp(std::vector<double>& matrix, int root,
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

void odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::ShiftBlocksLeft(std::vector<double>& matrix, int root,
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

bool odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::IsSquere(unsigned int num) {
  auto root = static_cast<unsigned int>(std::sqrt(num));
  return (root * root) == num;
}

int odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::GetBlockSize(int n) {
  for (int k = (n / 2); k >= 2; k--) {
    if ((n % k) == 0) {
      return k;
    }
  }
  return 1;
}
void odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::CopyBlock(const std::vector<double>& matrix,
                                                                    std::vector<double>& block, int start, int root,
                                                                    int block_sz) {
  for (int i = 0; i < block_sz; i++) {
    for (int j = 0; j < block_sz; j++) {
      int index = start + (i * root) + j;
      block[(i * block_sz) + j] = matrix[index];
    }
  }
}
void odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::InitializeShift(std::vector<double>& matrix, int root,
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
bool odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::PreProcessingImpl() {
  szA_ = task_data->inputs_count[0];
  szB_ = task_data->inputs_count[1];
  matrixA_.assign(reinterpret_cast<double*>(task_data->inputs[0]),
                  reinterpret_cast<double*>(task_data->inputs[0]) + szA_);
  matrixB_.assign(reinterpret_cast<double*>(task_data->inputs[1]),
                  reinterpret_cast<double*>(task_data->inputs[1]) + szB_);
  matrixC_.assign(szA_, 0);

  block_sz_ = GetBlockSize(static_cast<int>(sqrt(szA_)));
  return true;
}
void odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::MultiplyBlock(int bi, int bj, int root) {
  // Локальные буферы для A, B и результата
  std::vector<double> local_a(block_sz_ * block_sz_);
  std::vector<double> local_b(block_sz_ * block_sz_);
  std::vector<double> local_c(block_sz_ * block_sz_, 0.0);

  // Начальный индекс блока в развернутой матрице
  int start = ((bi * block_sz_) * root) + (bj * block_sz_);

  // Копируем блоки из глобальных матриц
  CopyBlock(matrixA_, local_a, start, root, block_sz_);
  CopyBlock(matrixB_, local_b, start, root, block_sz_);

  // Собственно умножение блока A на блок B → local_c
  for (int i = 0; i < block_sz_; ++i) {
    for (int k = 0; k < block_sz_; ++k) {
      double a = local_a[(i * block_sz_) + k];
      for (int j = 0; j < block_sz_; ++j) {
        local_c[(i * block_sz_) + j] += a * local_b[(k * block_sz_) + j];
      }
    }
  }

  // Аккумуляция результата в глобальную матрицу C
  for (int i = 0; i < block_sz_; ++i) {
    for (int j = 0; j < block_sz_; ++j) {
      int idx = (((bi * block_sz_) + i) * root) + ((bj * block_sz_) + j);
      matrixC_[idx] += local_c[(i * block_sz_) + j];
    }
  }
}

bool odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1]) {
    return false;
  }

  if ((!(IsSquere(task_data->inputs_count[0]))) || (!(IsSquere(task_data->inputs_count[1])))) {
    return false;
  }
  return true;
}

bool odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::RunImpl() {
  int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, num_threads);

  int root = static_cast<int>(std::sqrt(szA_));
  int num_blocks = std::max(1, root / block_sz_);
  int grid_size = num_blocks;

  // Начальные сдвиги
  InitializeShift(matrixA_, root, grid_size, block_sz_, true);
  InitializeShift(matrixB_, root, grid_size, block_sz_, false);

  for (int step = 0; step < grid_size; ++step) {
    tbb::parallel_for(tbb::blocked_range2d<int>(0, num_blocks, 0, num_blocks), [&](const tbb::blocked_range2d<int>& r) {
      for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
        for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
          MultiplyBlock(bi, bj, root);
        }
      }
    });

    // Пост-шафты
    ShiftBlocksLeft(matrixA_, root, block_sz_);
    ShiftBlocksUp(matrixB_, root, block_sz_);
  }

  return true;
}

bool odintsov_m_mulmatrix_cannon_tbb::MulMatrixCannonTBB::PostProcessingImpl() {
  std::size_t sz_c = matrixC_.size();
  for (std::size_t i = 0; i < sz_c; i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = matrixC_[i];
  }
  return true;
}