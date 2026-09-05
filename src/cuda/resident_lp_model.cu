#include "../fgpu/resident_lp_model.hpp"
#include "device_workspace.cuh"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cudaee::detail {
namespace {

__global__ void CountColumnsKernel(const std::int32_t columns, const std::uint8_t* active,
                                   const std::uint8_t* incidence_count, const std::int32_t stride,
                                   std::int64_t* counts, std::int32_t* invalid) {
  const std::int32_t column = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (column == columns) {
    counts[column] = 0;
  }
  if (column >= columns) {
    return;
  }
  const std::int32_t count = active[column] != 0U ? incidence_count[column] : 0;
  if (count > stride) {
    atomicExch(invalid, 1);
  }
  counts[column] = active[column] != 0U ? 2LL + count : 0;
}

__global__ void BuildColumnsKernel(const quick_hs::GraphView graph, const std::int32_t count,
                                   const std::int32_t* ids, const std::int32_t cuts,
                                   const std::uint8_t* incidence_count,
                                   const std::int32_t* incidence_ids, const std::int32_t stride,
                                   const std::int64_t* offsets, std::int32_t* rows,
                                   std::int64_t* values, unsigned long long* row_counts,
                                   std::int32_t* invalid) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= count) {
    return;
  }
  const std::int32_t edge = ids[work];
  const std::int64_t begin = offsets[edge];
  const std::int64_t end = offsets[edge + 1];
  rows[begin] = graph.edge_u[edge];
  rows[begin + 1] = graph.edge_v[edge];
  for (std::int32_t index = 0; index < incidence_count[edge]; ++index) {
    const std::int32_t cut = incidence_ids[static_cast<std::int64_t>(edge) * stride + index];
    if (cut < 0 || cut >= cuts) {
      atomicExch(invalid, 1);
      return;
    }
    rows[begin + 2 + index] = graph.dimension + cut;
  }
  // 每列仅现有 incidence 的大小；规范排序后浮点归约顺序与 CTA 调度无关。
  for (std::int64_t index = begin + 1; index < end; ++index) {
    const std::int32_t value = rows[index];
    std::int64_t place = index;
    while (place > begin && rows[place - 1] > value) {
      rows[place] = rows[place - 1];
      --place;
    }
    rows[place] = value;
  }
  for (std::int64_t index = begin; index < end; ++index) {
    if (index > begin && rows[index] == rows[index - 1]) {
      atomicExch(invalid, 1);
    }
    values[index] = 1;
    atomicAdd(row_counts + rows[index], 1ULL);
  }
}

__global__ void ScatterRowsKernel(const std::int32_t count, const std::int32_t* ids,
                                  const std::int64_t* offsets, const std::int32_t* rows,
                                  unsigned long long* cursor, std::int32_t* columns) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= count) {
    return;
  }
  const std::int32_t edge = ids[work];
  for (std::int64_t p = offsets[edge]; p < offsets[edge + 1]; ++p) {
    columns[atomicAdd(cursor + rows[p], 1ULL)] = edge;
  }
}

__global__ void SetRowsKernel(const std::int32_t dimension, const std::int32_t rows,
                              const std::int32_t static_cuts, const std::uint8_t* valid_cuts,
                              std::int64_t* rhs, std::uint8_t* equality) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < rows) {
    equality[row] = row < dimension ? 1U : 0U;
    rhs[row] = row < dimension || row - dimension < static_cuts || valid_cuts[row - dimension] != 0U
                   ? 2
                   : 0;
  }
}

__global__ void SetValuesKernel(const std::int64_t count, std::int64_t* values) {
  const std::int64_t begin = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t index = begin; index < count; index += stride) {
    values[index] = 1;
  }
}

} // namespace

class ResidentSecModelCuda::Impl {
public:
  explicit Impl(const int device)
      : selected(device), column_counts(device), column_offsets(device), row_offsets(device),
        row_values(device), column_values(device), rhs(device), row_counts(device), cursor(device),
        row_ids(device), columns(device), unsorted_columns(device), invalid(device),
        equality(device), scratch(device) {}
  int selected;
  CudaWorkspace<std::int64_t> column_counts, column_offsets, row_offsets, row_values, column_values,
      rhs;
  CudaWorkspace<unsigned long long> row_counts, cursor;
  CudaWorkspace<std::int32_t> row_ids, columns, unsorted_columns, invalid;
  CudaWorkspace<std::uint8_t> equality, scratch;
};

ResidentSecModelCuda::ResidentSecModelCuda(const int device)
    : impl_(std::make_unique<Impl>(device)) {}
ResidentSecModelCuda::~ResidentSecModelCuda() = default;

SparsePdhgDeviceModel
ResidentSecModelCuda::Build(const quick_hs::GraphView graph, const std::int64_t* costs,
                            const std::int32_t active_count, const std::int32_t* active_ids,
                            const std::int32_t cut_count, const std::int32_t static_cut_count,
                            const std::uint8_t* valid_cuts, const std::uint8_t* incidence_count,
                            const std::int32_t* incidence_ids, const std::int32_t incidence_stride,
                            const std::uint64_t version) {
  if (graph.dimension <= 0 || graph.edge_count <= 0 || graph.edge_count >= INT32_MAX ||
      cut_count < 0 || static_cut_count < 0 || static_cut_count > cut_count ||
      static_cast<std::int64_t>(cut_count) + graph.dimension >= INT32_MAX ||
      incidence_stride <= 0 || active_count <= 0 || active_count > graph.edge_count ||
      costs == nullptr || active_ids == nullptr || incidence_count == nullptr ||
      incidence_ids == nullptr || (cut_count > static_cut_count && valid_cuts == nullptr)) {
    throw std::invalid_argument("resident SEC 稀疏模型参数非法");
  }
  CheckWorkspaceCuda(cudaSetDevice(impl_->selected));
  const std::int32_t columns = static_cast<std::int32_t>(graph.edge_count);
  const std::int32_t rows = graph.dimension + cut_count;
  auto& s = *impl_;
  s.column_counts.Reserve(static_cast<std::uint64_t>(columns) + 1U);
  s.column_offsets.Reserve(static_cast<std::uint64_t>(columns) + 1U);
  s.row_counts.Reserve(static_cast<std::uint64_t>(rows) + 1U);
  s.row_offsets.Reserve(static_cast<std::uint64_t>(rows) + 1U);
  s.cursor.Reserve(rows);
  s.rhs.Reserve(rows);
  s.equality.Reserve(rows);
  s.invalid.Reserve(1U);
  CheckWorkspaceCuda(cudaMemset(s.invalid.get(), 0, sizeof(std::int32_t)));
  CountColumnsKernel<<<(columns + 128) / 128, 128>>>(columns, graph.edge_active, incidence_count,
                                                     incidence_stride, s.column_counts.get(),
                                                     s.invalid.get());
  std::size_t temporary_bytes = 0U;
  CheckWorkspaceCuda(cub::DeviceScan::ExclusiveSum(nullptr, temporary_bytes, s.column_counts.get(),
                                                   s.column_offsets.get(),
                                                   static_cast<std::int64_t>(columns) + 1));
  s.scratch.Reserve(temporary_bytes);
  CheckWorkspaceCuda(cub::DeviceScan::ExclusiveSum(s.scratch.get(), temporary_bytes,
                                                   s.column_counts.get(), s.column_offsets.get(),
                                                   static_cast<std::int64_t>(columns) + 1));
  std::int64_t nonzeros = 0;
  std::int32_t invalid = 0;
  CheckWorkspaceCuda(cudaMemcpy(&nonzeros, s.column_offsets.get() + columns, sizeof(nonzeros),
                                cudaMemcpyDeviceToHost));
  CheckWorkspaceCuda(
      cudaMemcpy(&invalid, s.invalid.get(), sizeof(invalid), cudaMemcpyDeviceToHost));
  if (invalid != 0 || nonzeros < 2LL * active_count) {
    throw std::runtime_error("SEC incidence 计数无效");
  }
  s.row_ids.Reserve(nonzeros);
  s.columns.Reserve(nonzeros);
  s.unsorted_columns.Reserve(nonzeros);
  s.column_values.Reserve(nonzeros);
  s.row_values.Reserve(nonzeros);
  CheckWorkspaceCuda(cudaMemset(
      s.row_counts.get(), 0, (static_cast<std::size_t>(rows) + 1U) * sizeof(unsigned long long)));
  BuildColumnsKernel<<<(active_count + 127) / 128, 128>>>(
      graph, active_count, active_ids, cut_count, incidence_count, incidence_ids, incidence_stride,
      s.column_offsets.get(), s.row_ids.get(), s.column_values.get(), s.row_counts.get(),
      s.invalid.get());
  CheckWorkspaceCuda(
      cudaMemcpy(&invalid, s.invalid.get(), sizeof(invalid), cudaMemcpyDeviceToHost));
  if (invalid != 0) {
    throw std::runtime_error("SEC incidence 引用非法或重复 cut");
  }
  temporary_bytes = 0U;
  CheckWorkspaceCuda(cub::DeviceScan::ExclusiveSum(nullptr, temporary_bytes, s.row_counts.get(),
                                                   s.row_offsets.get(), rows + 1));
  s.scratch.Reserve(temporary_bytes);
  CheckWorkspaceCuda(cub::DeviceScan::ExclusiveSum(
      s.scratch.get(), temporary_bytes, s.row_counts.get(), s.row_offsets.get(), rows + 1));
  CheckWorkspaceCuda(cudaMemcpy(s.cursor.get(), s.row_offsets.get(),
                                static_cast<std::size_t>(rows) * sizeof(std::int64_t),
                                cudaMemcpyDeviceToDevice));
  ScatterRowsKernel<<<(active_count + 127) / 128, 128>>>(active_count, active_ids,
                                                         s.column_offsets.get(), s.row_ids.get(),
                                                         s.cursor.get(), s.unsorted_columns.get());
  temporary_bytes = 0U;
  CheckWorkspaceCuda(cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr, temporary_bytes, s.unsorted_columns.get(), s.columns.get(), nonzeros, rows,
      s.row_offsets.get(), s.row_offsets.get() + 1));
  s.scratch.Reserve(temporary_bytes);
  CheckWorkspaceCuda(cub::DeviceSegmentedRadixSort::SortKeys(
      s.scratch.get(), temporary_bytes, s.unsorted_columns.get(), s.columns.get(), nonzeros, rows,
      s.row_offsets.get(), s.row_offsets.get() + 1));
  SetRowsKernel<<<(rows + 127) / 128, 128>>>(graph.dimension, rows, static_cut_count, valid_cuts,
                                             s.rhs.get(), s.equality.get());
  SetValuesKernel<<<static_cast<int>(std::min<std::int64_t>(65535, (nonzeros + 127) / 128)), 128>>>(
      nonzeros, s.row_values.get());
  CheckWorkspaceCuda(cudaGetLastError());
  return {rows,
          columns,
          nonzeros,
          s.row_offsets.get(),
          s.columns.get(),
          s.row_values.get(),
          s.column_offsets.get(),
          s.row_ids.get(),
          s.column_values.get(),
          s.rhs.get(),
          s.equality.get(),
          costs,
          graph.edge_active,
          version,
          active_ids,
          active_count};
}

std::uint64_t ResidentSecModelCuda::workspace_bytes() const {
  const auto& s = *impl_;
  return s.column_counts.bytes() + s.column_offsets.bytes() + s.row_offsets.bytes() +
         s.row_values.bytes() + s.column_values.bytes() + s.rhs.bytes() + s.row_counts.bytes() +
         s.cursor.bytes() + s.row_ids.bytes() + s.columns.bytes() + s.unsorted_columns.bytes() +
         s.invalid.bytes() + s.equality.bytes() + s.scratch.bytes();
}

} // namespace cudaee::detail
