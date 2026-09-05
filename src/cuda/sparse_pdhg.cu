#include "../fgpu/sparse_pdhg.hpp"
#include "sparse_pdhg_validation.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaee::detail {
namespace {

void Check(const cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("sparse PDHG: ") + cudaGetErrorString(status));
  }
}

template <typename T> class Buffer {
public:
  ~Buffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
    }
  }
  void Resize(const std::size_t count) {
    if (count <= size_) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("sparse PDHG workspace 溢出");
    }
    T* next = nullptr;
    Check(cudaMalloc(&next, count * sizeof(T)));
    try {
      Check(cudaMemset(next, 0, count * sizeof(T)));
      if (data_ != nullptr) {
        Check(cudaMemcpy(next, data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
        Check(cudaFree(data_));
      }
    } catch (...) {
      static_cast<void>(cudaFree(next));
      throw;
    }
    data_ = next;
    size_ = count;
  }
  T* get() const { return data_; }
  std::uint64_t bytes() const { return size_ * sizeof(T); }

private:
  T* data_{};
  std::size_t size_{};
};

struct State {
  double* x;
  double* extrapolated;
  double* y;
  double* average_x;
  double* average_y;
  double* tau;
  double* sigma;
  unsigned long long* iteration;
};

__global__ void PrepareKernel(const SparsePdhgDeviceModel model, const State state) {
  const std::int32_t index = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < model.columns) {
    double sum = 0.0;
    for (std::int64_t p = model.column_offsets[index]; p < model.column_offsets[index + 1]; ++p) {
      sum += fabs(static_cast<double>(model.column_values[p]));
    }
    state.tau[index] = 0.9 / fmax(1.0, sum);
    if (model.active != nullptr && model.active[index] == 0U) {
      state.x[index] = 0.0;
    }
    state.extrapolated[index] = state.x[index];
    state.average_x[index] = 0.0;
  }
  if (index < model.rows) {
    double sum = 0.0;
    for (std::int64_t p = model.row_offsets[index]; p < model.row_offsets[index + 1]; ++p) {
      sum += fabs(static_cast<double>(model.row_values[p]));
    }
    state.sigma[index] = 0.9 / fmax(1.0, sum);
    state.average_y[index] = 0.0;
  }
}

__global__ void AdvanceIterationKernel(unsigned long long* const iteration) { ++*iteration; }

__global__ void DualStepKernel(const SparsePdhgDeviceModel model, const State state) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= model.rows) {
    return;
  }
  double value = 0.0;
  for (std::int64_t p = model.row_offsets[row]; p < model.row_offsets[row + 1]; ++p) {
    value += static_cast<double>(model.row_values[p]) * state.extrapolated[model.column_ids[p]];
  }
  const double next =
      state.y[row] + state.sigma[row] * (static_cast<double>(model.rhs[row]) - value);
  state.y[row] = model.equality[row] != 0U ? next : fmax(0.0, next);
  state.average_y[row] +=
      (state.y[row] - state.average_y[row]) / static_cast<double>(*state.iteration);
}

__global__ void PrimalStepKernel(const SparsePdhgDeviceModel model, const State state,
                                 const double scale) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t count = model.active_ids == nullptr ? model.columns : model.active_count;
  if (work >= count) {
    return;
  }
  const std::int32_t column = model.active_ids == nullptr ? work : model.active_ids[work];
  double transpose = 0.0;
  for (std::int64_t p = model.column_offsets[column]; p < model.column_offsets[column + 1]; ++p) {
    transpose += static_cast<double>(model.column_values[p]) * state.y[model.row_ids[p]];
  }
  const double old = state.x[column];
  const double next =
      model.active != nullptr && model.active[column] == 0U
          ? 0.0
          : fmin(1.0, fmax(0.0, old + state.tau[column] *
                                          (transpose -
                                           static_cast<double>(model.objective[column]) / scale)));
  state.x[column] = next;
  state.extrapolated[column] = 2.0 * next - old;
  state.average_x[column] +=
      (next - state.average_x[column]) / static_cast<double>(*state.iteration);
}

// 诊断只影响后续调度。真正的删边仍需整数系数、量化 dual 的精确重算。
__global__ void DiagnosticsKernel(const SparsePdhgDeviceModel model, const State state,
                                  const double scale, double* const diagnostics) {
  __shared__ double primal[128], dual[128], violation[128];
  double p_value = 0.0;
  double d_value = 0.0;
  double max_violation = 0.0;
  for (std::int32_t column = static_cast<std::int32_t>(threadIdx.x); column < model.columns;
       column += 128) {
    if (model.active != nullptr && model.active[column] == 0U) {
      continue;
    }
    p_value += static_cast<double>(model.objective[column]) * state.average_x[column];
    double reduced = static_cast<double>(model.objective[column]);
    for (std::int64_t offset = model.column_offsets[column];
         offset < model.column_offsets[column + 1]; ++offset) {
      reduced -= static_cast<double>(model.column_values[offset]) *
                 state.average_y[model.row_ids[offset]] * scale;
    }
    d_value += fmin(0.0, reduced);
  }
  for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < model.rows; row += 128) {
    d_value += static_cast<double>(model.rhs[row]) * state.average_y[row] * scale;
    double activity = 0.0;
    for (std::int64_t offset = model.row_offsets[row]; offset < model.row_offsets[row + 1];
         ++offset) {
      activity +=
          static_cast<double>(model.row_values[offset]) * state.average_x[model.column_ids[offset]];
    }
    const double residual = static_cast<double>(model.rhs[row]) - activity;
    max_violation =
        fmax(max_violation, model.equality[row] != 0U ? fabs(residual) : fmax(0.0, residual));
  }
  primal[threadIdx.x] = p_value;
  dual[threadIdx.x] = d_value;
  violation[threadIdx.x] = max_violation;
  __syncthreads();
  for (std::int32_t stride = 64; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      primal[threadIdx.x] += primal[threadIdx.x + stride];
      dual[threadIdx.x] += dual[threadIdx.x + stride];
      violation[threadIdx.x] = fmax(violation[threadIdx.x], violation[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    diagnostics[0] = primal[0];
    diagnostics[1] = dual[0];
    diagnostics[2] = violation[0];
  }
}

} // namespace

class SparsePdhgCuda::Impl {
public:
  explicit Impl(const int selected) : device(selected) { Check(cudaSetDevice(device)); }
  ~Impl() {
    static_cast<void>(cudaSetDevice(device));
    if (execution != nullptr) {
      static_cast<void>(cudaGraphExecDestroy(execution));
    }
  }
  State state() {
    return {x.get(),         extrapolated.get(), y.get(),     average_x.get(),
            average_y.get(), tau.get(),          sigma.get(), iteration.get()};
  }
  void Prepare(const SparsePdhgDeviceModel& model, const double next_scale) {
    // 两次 Iterate 之间允许调用者更换设备；不依赖隐式 current device。
    Check(cudaSetDevice(device));
    if (prepared && execution != nullptr && model == bound_model && scale == next_scale) {
      return;
    }
    if (execution != nullptr) {
      Check(cudaGraphExecDestroy(execution));
      execution = nullptr;
    }
    invalid.Resize(2U);
    Check(cudaMemset(invalid.get(), 0, 2U * sizeof(std::int32_t)));
    const int validation_blocks = static_cast<int>(
        (static_cast<std::int64_t>(std::max(model.rows, model.columns)) + 127) / 128);
    sparse_pdhg_validation::OffsetsKernel<<<validation_blocks, 128>>>(model, invalid.get());
    std::int32_t model_status[2]{};
    Check(cudaMemcpy(model_status, invalid.get(), sizeof(model_status), cudaMemcpyDeviceToHost));
    if (model_status[0] != 0) {
      throw std::invalid_argument("sparse PDHG CSR/CSC offsets 无效");
    }
    sparse_pdhg_validation::MatrixKernel<<<validation_blocks, 128>>>(model, invalid.get());
    Check(cudaMemcpy(model_status, invalid.get(), sizeof(model_status), cudaMemcpyDeviceToHost));
    if (model_status[0] != 0 ||
        (model.active_ids != nullptr && model_status[1] != model.active_count)) {
      throw std::invalid_argument("sparse PDHG 矩阵转置或活动列集合不一致");
    }
    x.Resize(model.columns);
    extrapolated.Resize(model.columns);
    y.Resize(model.rows);
    average_x.Resize(model.columns);
    average_y.Resize(model.rows);
    tau.Resize(model.columns);
    sigma.Resize(model.rows);
    iteration.Resize(1U);
    diagnostics.Resize(3U);
    // 目标尺度变化时清零 dual，避免把旧尺度 warm start 当成新问题的值。
    if (!prepared || scale != next_scale) {
      Check(cudaMemset(y.get(), 0, y.bytes()));
    }
    Check(cudaMemset(iteration.get(), 0, sizeof(unsigned long long)));
    const int blocks = (std::max(model.rows, model.columns) + 127) / 128;
    PrepareKernel<<<blocks, 128>>>(model, state());
    Check(cudaGetLastError());
    rows = model.rows;
    columns = model.columns;
    bound_model = model;
    scale = next_scale;
    prepared = true;
    // 一次执行图包含 64 次 PDHG 更新，消除逐迭代的 CPU launch 开销。
    cudaStream_t capture = nullptr;
    cudaGraph_t graph = nullptr;
    Check(cudaStreamCreateWithFlags(&capture, cudaStreamNonBlocking));
    try {
      Check(cudaStreamBeginCapture(capture, cudaStreamCaptureModeThreadLocal));
      for (int step = 0; step < 64; ++step) {
        AdvanceIterationKernel<<<1, 1, 0, capture>>>(iteration.get());
        DualStepKernel<<<(rows + 127) / 128, 128, 0, capture>>>(model, state());
        const std::int32_t active_columns =
            model.active_ids == nullptr ? columns : model.active_count;
        PrimalStepKernel<<<(active_columns + 127) / 128, 128, 0, capture>>>(model, state(), scale);
      }
      Check(cudaStreamEndCapture(capture, &graph));
      Check(cudaGraphInstantiate(&execution, graph, nullptr, nullptr, 0));
      Check(cudaGraphDestroy(graph));
      graph = nullptr;
      Check(cudaStreamDestroy(capture));
    } catch (...) {
      static_cast<void>(cudaStreamEndCapture(capture, &graph));
      if (graph != nullptr) {
        static_cast<void>(cudaGraphDestroy(graph));
      }
      static_cast<void>(cudaStreamDestroy(capture));
      throw;
    }
  }
  int device;
  Buffer<double> x, extrapolated, y, average_x, average_y, tau, sigma, diagnostics;
  Buffer<unsigned long long> iteration;
  Buffer<std::int32_t> invalid;
  cudaGraphExec_t execution{};
  std::int32_t rows{}, columns{};
  SparsePdhgDeviceModel bound_model{};
  double scale{};
  bool prepared{};
};

SparsePdhgCuda::SparsePdhgCuda(const int device) : impl_(std::make_unique<Impl>(device)) {}
SparsePdhgCuda::~SparsePdhgCuda() = default;

SparsePdhgDiagnostics SparsePdhgCuda::Iterate(const SparsePdhgDeviceModel& model,
                                              const double objective_scale,
                                              const std::uint32_t iterations) {
  if (model.rows <= 0 || model.columns <= 0 || model.nonzeros < 0 || iterations == 0U ||
      (model.active_ids != nullptr &&
       (model.active_count <= 0 || model.active_count > model.columns)) ||
      !std::isfinite(objective_scale) || objective_scale <= 0.0 || model.row_offsets == nullptr ||
      model.column_offsets == nullptr || model.rhs == nullptr || model.equality == nullptr ||
      model.objective == nullptr ||
      (model.nonzeros > 0 && (model.column_ids == nullptr || model.row_values == nullptr ||
                              model.row_ids == nullptr || model.column_values == nullptr))) {
    throw std::invalid_argument("sparse PDHG 输入模型无效");
  }
  const auto begin = std::chrono::steady_clock::now();
  impl_->Prepare(model, objective_scale);
  const std::uint32_t batches = iterations / 64U;
  for (std::uint32_t batch = 0U; batch < batches; ++batch) {
    Check(cudaGraphLaunch(impl_->execution, nullptr));
  }
  for (std::uint32_t tail = batches * 64U; tail < iterations; ++tail) {
    AdvanceIterationKernel<<<1, 1>>>(impl_->iteration.get());
    DualStepKernel<<<(model.rows + 127) / 128, 128>>>(model, impl_->state());
    const std::int32_t active_columns =
        model.active_ids == nullptr ? model.columns : model.active_count;
    PrimalStepKernel<<<(active_columns + 127) / 128, 128>>>(model, impl_->state(), objective_scale);
  }
  DiagnosticsKernel<<<1, 128>>>(model, impl_->state(), objective_scale, impl_->diagnostics.get());
  Check(cudaGetLastError());
  double values[3]{};
  unsigned long long steps = 0U;
  Check(cudaMemcpy(values, impl_->diagnostics.get(), sizeof(values), cudaMemcpyDeviceToHost));
  Check(cudaMemcpy(&steps, impl_->iteration.get(), sizeof(steps), cudaMemcpyDeviceToHost));
  for (const double value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("sparse PDHG 产生非有限诊断，禁止授权");
    }
  }
  const double gap =
      fabs(values[0] - values[1]) / std::max({1.0, fabs(values[0]), fabs(values[1])});
  return {
      steps,
      values[2],
      gap,
      values[0],
      values[1],
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(),
      values[2] <= 1.0e-6 && gap <= 1.0e-6};
}

const double* SparsePdhgCuda::primal() const { return impl_->average_x.get(); }
const double* SparsePdhgCuda::dual() const { return impl_->average_y.get(); }
std::uint64_t SparsePdhgCuda::workspace_bytes() const {
  return impl_->x.bytes() + impl_->extrapolated.bytes() + impl_->y.bytes() +
         impl_->average_x.bytes() + impl_->average_y.bytes() + impl_->tau.bytes() +
         impl_->sigma.bytes() + impl_->iteration.bytes() + impl_->diagnostics.bytes() +
         impl_->invalid.bytes();
}

} // namespace cudaee::detail
