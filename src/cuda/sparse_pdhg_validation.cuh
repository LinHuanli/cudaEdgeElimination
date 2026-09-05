#pragma once

#include "../fgpu/sparse_pdhg.hpp"
#include <cuda_runtime.h>

namespace cudaee::detail::sparse_pdhg_validation {

static __global__ void OffsetsKernel(const SparsePdhgDeviceModel model, std::int32_t* invalid) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index == 0 && (model.row_offsets[0] != 0 || model.column_offsets[0] != 0 ||
                     model.row_offsets[model.rows] != model.nonzeros ||
                     model.column_offsets[model.columns] != model.nonzeros)) {
    atomicExch(invalid, 1);
  }
  if (index < model.rows &&
      (model.row_offsets[index] < 0 || model.row_offsets[index] > model.row_offsets[index + 1] ||
       model.row_offsets[index + 1] > model.nonzeros || model.equality[index] > 1U)) {
    atomicExch(invalid, 1);
  }
  if (index < model.columns && (model.column_offsets[index] < 0 ||
                                model.column_offsets[index] > model.column_offsets[index + 1] ||
                                model.column_offsets[index + 1] > model.nonzeros)) {
    atomicExch(invalid, 1);
  }
}

// CSR/CSC 都严格按索引排序且没有重复；逐 CSR entry 查 CSC 的精确
// 整数系数。同样的 nnz 与单射匹配保证两份矩阵完全一致，无哈希碰撞假设。
static __global__ void MatrixKernel(const SparsePdhgDeviceModel model, std::int32_t* status) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < model.rows) {
    for (std::int64_t p = model.row_offsets[index]; p < model.row_offsets[index + 1]; ++p) {
      const std::int32_t column = model.column_ids[p];
      if (column < 0 || column >= model.columns ||
          (p > model.row_offsets[index] && model.column_ids[p - 1] >= column)) {
        atomicExch(status, 1);
        continue;
      }
      std::int64_t lo = model.column_offsets[column], hi = model.column_offsets[column + 1];
      while (lo < hi) {
        const std::int64_t middle = lo + (hi - lo) / 2;
        if (model.row_ids[middle] < index) {
          lo = middle + 1;
        } else {
          hi = middle;
        }
      }
      if (lo >= model.column_offsets[column + 1] || model.row_ids[lo] != index ||
          model.column_values[lo] != model.row_values[p]) {
        atomicExch(status, 1);
      }
    }
  }
  if (index < model.columns) {
    for (std::int64_t p = model.column_offsets[index]; p < model.column_offsets[index + 1]; ++p) {
      if (model.row_ids[p] < 0 || model.row_ids[p] >= model.rows ||
          (p > model.column_offsets[index] && model.row_ids[p - 1] >= model.row_ids[p])) {
        atomicExch(status, 1);
      }
    }
    if (model.active_ids != nullptr && (model.active == nullptr || model.active[index] != 0U)) {
      atomicAdd(status + 1, 1);
    }
  }
  if (model.active_ids != nullptr && index < model.active_count) {
    const std::int32_t column = model.active_ids[index];
    if (column < 0 || column >= model.columns ||
        (index > 0 && model.active_ids[index - 1] >= column) ||
        (column >= 0 && column < model.columns && model.active != nullptr &&
         model.active[column] == 0U)) {
      atomicExch(status, 1);
    }
  }
}

} // namespace cudaee::detail::sparse_pdhg_validation
