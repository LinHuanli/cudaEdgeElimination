#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaee::detail {
namespace {

void CheckCuda(const cudaError_t status, const char* const operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    if (count_ != 0) {
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(HT c,d)");
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() { return data_; }
  [[nodiscard]] const T* get() const { return data_; }

  void CopyFromHost(const T* const source) {
    if (count_ != 0) {
      CheckCuda(cudaMemcpy(data_, source, sizeof(T) * count_, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D(HT c,d)");
    }
  }

  void CopyToHost(T* const destination) const {
    if (count_ != 0) {
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(HT c,d)");
    }
  }

private:
  T* data_{nullptr};
  std::size_t count_{};
};

int SelectDevice(std::string* const reason) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    if (reason != nullptr) {
      *reason =
          count_status == cudaSuccess ? "没有可见 CUDA 设备" : cudaGetErrorString(count_status);
    }
    return -1;
  }
  int best_device = -1;
  std::size_t best_free_bytes = 0;
  for (int device = 0; device < device_count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        (best_device < 0 || free_bytes > best_free_bytes)) {
      best_device = device;
      best_free_bytes = free_bytes;
    }
  }
  if (best_device < 0) {
    if (reason != nullptr) {
      *reason = "所有可见 GPU 均无法查询显存";
    }
    return -1;
  }
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice(HT c,d)");
  return best_device;
}

__device__ std::uint64_t IntegerSqrtFloorDevice(const std::uint64_t value) {
  std::uint64_t remainder = value;
  std::uint64_t root = 0;
  std::uint64_t bit = std::uint64_t{1} << 62;
  while (bit > remainder) {
    bit >>= 2;
  }
  while (bit != 0) {
    if (remainder >= root + bit) {
      remainder -= root + bit;
      root = (root >> 1) + bit;
    } else {
      root >>= 1;
    }
    bit >>= 2;
  }
  return root;
}

__device__ std::uint64_t AbsoluteDifference(const std::int64_t first, const std::int64_t second) {
  return first >= second ? static_cast<std::uint64_t>(first - second)
                         : static_cast<std::uint64_t>(second - first);
}

__device__ std::uint64_t ExactDistanceDevice(const std::int32_t first, const std::int32_t second,
                                             const std::int64_t* const x,
                                             const std::int64_t* const y,
                                             const std::uint8_t distance_type) {
  const std::uint64_t dx = AbsoluteDifference(x[first], x[second]);
  const std::uint64_t dy = AbsoluteDifference(y[first], y[second]);
  const std::uint64_t squared = dx * dx + dy * dy;
  const std::uint64_t root = IntegerSqrtFloorDevice(squared);
  std::uint64_t rounded = root;
  if (distance_type == static_cast<std::uint8_t>(DistanceType::kEuc2D)) {
    rounded += squared - root * root > root ? 1U : 0U;
  } else {
    rounded += root * root != squared ? 1U : 0U;
  }
  return rounded;
}

__global__ void ScreenHtCdKernel(const NodeEdge target, const HtCdScreenTask* const tasks,
                                 const std::size_t task_count, const std::int64_t* const x,
                                 const std::int64_t* const y, const std::uint8_t distance_type,
                                 const std::uint8_t mode, std::uint8_t* const flags) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= task_count) {
    return;
  }
  const HtCdScreenTask task = tasks[index];
  const std::uint64_t original = ExactDistanceDevice(target.u, target.v, x, y, distance_type) +
                                 ExactDistanceDevice(task.c, task.d, x, y, distance_type);
  const std::uint64_t orientation0 = ExactDistanceDevice(target.u, task.d, x, y, distance_type) +
                                     ExactDistanceDevice(target.v, task.c, x, y, distance_type);
  const std::uint64_t orientation1 = ExactDistanceDevice(target.u, task.c, x, y, distance_type) +
                                     ExactDistanceDevice(target.v, task.d, x, y, distance_type);
  const bool incompatible = orientation0 < original && orientation1 < original;
  if (mode == static_cast<std::uint8_t>(HtCdMode::kActiveIncompatible)) {
    flags[index] = static_cast<std::uint8_t>(task.active != 0 && incompatible);
  } else {
    flags[index] = static_cast<std::uint8_t>(task.active == 0 || incompatible);
  }
}

__device__ void AppendCompletedState(const std::uint32_t state_index,
                                     std::uint32_t* const next_frontier,
                                     std::uint32_t* const next_count,
                                     const std::uint32_t state_count) {
  const std::uint32_t slot = atomicAdd(next_count, 1U);
  if (slot < state_count) {
    next_frontier[slot] = state_index;
  }
}

__global__ void PropagateHtContinuationsKernel(
    const HtWavefrontStateTask* const states, const HtWavefrontMoveTask* const moves,
    const std::uint32_t* const frontier, const std::uint32_t frontier_count,
    std::int32_t* const status, std::uint32_t* const remaining_moves,
    std::uint32_t* const remaining_children, std::uint32_t* const move_failed,
    std::uint32_t* const next_frontier, std::uint32_t* const next_count,
    const std::uint32_t state_count) {
  const std::size_t frontier_offset =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (frontier_offset >= frontier_count) {
    return;
  }
  const std::uint32_t state_index = frontier[frontier_offset];
  const std::uint32_t parent_move = states[state_index].parent_move;
  if (parent_move == kNoHtChild) {
    return;
  }
  if (status[state_index] == 0) {
    atomicExch(&move_failed[parent_move], 1U);
  }
  __threadfence();
  const std::uint32_t children_before = atomicSub(&remaining_children[parent_move], 1U);
  if (children_before != 1U) {
    return;
  }

  const std::uint32_t parent_state = moves[parent_move].parent_state;
  if (atomicAdd(&move_failed[parent_move], 0U) == 0U) {
    if (atomicCAS(&status[parent_state], -1, 1) == -1) {
      AppendCompletedState(parent_state, next_frontier, next_count, state_count);
    }
    return;
  }

  const std::uint32_t moves_before = atomicSub(&remaining_moves[parent_state], 1U);
  if (moves_before == 1U && atomicCAS(&status[parent_state], -1, 0) == -1) {
    AppendCompletedState(parent_state, next_frontier, next_count, state_count);
  }
}

} // namespace

bool HtCdCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<std::uint8_t> ScreenHtCdCandidatesCuda(const GraphSnapshot& graph,
                                                   const NodeEdge target_edge,
                                                   const std::vector<HtCdScreenTask>& tasks,
                                                   const HtCdMode mode,
                                                   int* const selected_device) {
  if (target_edge.u < 0 || target_edge.v >= graph.dimension || target_edge.u >= target_edge.v ||
      (mode != HtCdMode::kActiveIncompatible && mode != HtCdMode::kMissingOrIncompatible)) {
    throw std::invalid_argument("CUDA HT c,d 输入非法");
  }
  for (const HtCdScreenTask& task : tasks) {
    if (task.c < 0 || task.d >= graph.dimension || task.c >= task.d || task.c == target_edge.u ||
        task.c == target_edge.v || task.d == target_edge.u || task.d == target_edge.v ||
        task.active > 1 || (task.active != 0) != graph.HasActiveEdge(task.c, task.d)) {
      throw std::invalid_argument("CUDA HT c,d task 非法或活动位不一致");
    }
  }

  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA HT c,d 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  if (tasks.empty()) {
    return {};
  }
  constexpr std::size_t kThreads = 256;
  const std::size_t blocks = (tasks.size() + kThreads - 1U) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA HT c,d 网格过大");
  }

  std::vector<std::int64_t> host_x(graph.points.size());
  std::vector<std::int64_t> host_y(graph.points.size());
  for (std::size_t index = 0; index < graph.points.size(); ++index) {
    host_x[index] = graph.points[index].integer_x;
    host_y[index] = graph.points[index].integer_y;
  }
  DeviceBuffer<HtCdScreenTask> device_tasks(tasks.size());
  DeviceBuffer<std::int64_t> device_x(host_x.size());
  DeviceBuffer<std::int64_t> device_y(host_y.size());
  DeviceBuffer<std::uint8_t> device_flags(tasks.size());
  device_tasks.CopyFromHost(tasks.data());
  device_x.CopyFromHost(host_x.data());
  device_y.CopyFromHost(host_y.data());

  ScreenHtCdKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      target_edge, device_tasks.get(), tasks.size(), device_x.get(), device_y.get(),
      static_cast<std::uint8_t>(graph.distance_type), static_cast<std::uint8_t>(mode),
      device_flags.get());
  CheckCuda(cudaGetLastError(), "ScreenHtCdKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "ScreenHtCdKernel synchronize");

  std::vector<std::uint8_t> flags(tasks.size());
  device_flags.CopyToHost(flags.data());
  return flags;
}

bool HtWavefrontCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<std::uint8_t> EvaluateHtWavefrontCuda(const std::vector<HtWavefrontStateTask>& states,
                                                  const std::vector<HtWavefrontMoveTask>& moves,
                                                  const std::vector<HtWavefrontReplyTask>& replies,
                                                  const std::vector<std::uint32_t>& level_offsets,
                                                  int* const selected_device) {
  if (states.empty() || states.size() > std::numeric_limits<std::uint32_t>::max() ||
      moves.size() > std::numeric_limits<std::uint32_t>::max() ||
      replies.size() > std::numeric_limits<std::uint32_t>::max() || level_offsets.size() < 2U ||
      level_offsets.front() != 0U || level_offsets.back() != states.size()) {
    throw std::invalid_argument("CUDA HT wavefront 数组规模或层边界非法");
  }
  for (std::size_t level = 1; level < level_offsets.size(); ++level) {
    if (level_offsets[level - 1U] >= level_offsets[level]) {
      throw std::invalid_argument("CUDA HT wavefront 层边界不是严格递增序列");
    }
  }

  std::size_t expected_move = 0;
  std::size_t expected_reply = 0;
  std::size_t current_level = 0;
  std::vector<bool> child_seen(states.size(), false);
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    while (state_index >= level_offsets[current_level + 1U]) {
      ++current_level;
    }
    const std::uint32_t next_level_begin = level_offsets[current_level + 1U];
    const HtWavefrontStateTask& state = states[state_index];
    if ((state_index == 0U && state.parent_move != kNoHtChild) ||
        (state_index != 0U && state.parent_move >= moves.size()) || state.leaf_proven > 1U ||
        state.move_begin != expected_move || state.move_count > moves.size() - expected_move ||
        (state.leaf_proven != 0U && state.move_count != 0U)) {
      throw std::invalid_argument("CUDA HT wavefront state task 非法");
    }
    for (std::uint32_t move_offset = 0; move_offset < state.move_count; ++move_offset) {
      const std::size_t move_index = expected_move++;
      const HtWavefrontMoveTask& move = moves[move_index];
      if (move.parent_state != state_index || move.reply_begin != expected_reply ||
          move.reply_count > replies.size() - expected_reply) {
        throw std::invalid_argument("CUDA HT wavefront move task 非法");
      }
      std::uint32_t child_count = 0;
      for (std::uint32_t reply_offset = 0; reply_offset < move.reply_count; ++reply_offset) {
        const HtWavefrontReplyTask& reply = replies[expected_reply++];
        if (reply.path_infeasible > 1U ||
            (reply.path_infeasible != 0U && reply.child_index != kNoHtChild) ||
            (reply.path_infeasible == 0U &&
             (reply.child_index < next_level_begin || reply.child_index >= states.size()))) {
          throw std::invalid_argument("CUDA HT wavefront reply task 非法");
        }
        if (reply.path_infeasible == 0U) {
          if (child_seen[reply.child_index] ||
              states[reply.child_index].parent_move != move_index) {
            throw std::invalid_argument("CUDA HT wavefront child parent 映射非法");
          }
          child_seen[reply.child_index] = true;
          ++child_count;
        }
      }
      if (move.child_count != child_count) {
        throw std::invalid_argument("CUDA HT wavefront move child_count 非法");
      }
    }
  }
  if (expected_move != moves.size() || expected_reply != replies.size()) {
    throw std::invalid_argument("CUDA HT wavefront task 数组含未引用记录");
  }
  if (std::find(child_seen.begin() + 1, child_seen.end(), false) != child_seen.end()) {
    throw std::invalid_argument("CUDA HT wavefront 含未引用非根状态");
  }

  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA HT wavefront 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }

  DeviceBuffer<HtWavefrontStateTask> device_states(states.size());
  DeviceBuffer<HtWavefrontMoveTask> device_moves(moves.size());
  DeviceBuffer<std::int32_t> device_status(states.size());
  DeviceBuffer<std::uint32_t> device_remaining_moves(states.size());
  DeviceBuffer<std::uint32_t> device_remaining_children(moves.size());
  DeviceBuffer<std::uint32_t> device_move_failed(moves.size());
  DeviceBuffer<std::uint32_t> device_frontier_first(states.size());
  DeviceBuffer<std::uint32_t> device_frontier_second(states.size());
  DeviceBuffer<std::uint32_t> device_next_count(1U);

  std::vector<std::int32_t> host_status(states.size(), -1);
  std::vector<std::uint32_t> host_remaining_moves(states.size());
  std::vector<std::uint32_t> host_remaining_children(moves.size());
  std::vector<std::uint32_t> initial_frontier;
  initial_frontier.reserve(states.size());
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    const HtWavefrontStateTask& state = states[state_index];
    host_remaining_moves[state_index] = state.move_count;
    bool has_vacuous_move = false;
    for (std::uint32_t offset = 0; offset < state.move_count; ++offset) {
      has_vacuous_move = has_vacuous_move || moves[state.move_begin + offset].child_count == 0U;
    }
    if (state.leaf_proven != 0U || has_vacuous_move) {
      host_status[state_index] = 1;
    } else if (state.move_count == 0U) {
      host_status[state_index] = 0;
    }
    if (host_status[state_index] != -1) {
      initial_frontier.push_back(static_cast<std::uint32_t>(state_index));
    }
  }
  for (std::size_t move_index = 0; move_index < moves.size(); ++move_index) {
    host_remaining_children[move_index] = moves[move_index].child_count;
  }

  device_states.CopyFromHost(states.data());
  device_moves.CopyFromHost(moves.data());
  device_status.CopyFromHost(host_status.data());
  device_remaining_moves.CopyFromHost(host_remaining_moves.data());
  device_remaining_children.CopyFromHost(host_remaining_children.data());
  if (!moves.empty()) {
    CheckCuda(cudaMemset(device_move_failed.get(), 0, moves.size() * sizeof(std::uint32_t)),
              "cudaMemset(HT wavefront move_failed)");
  }
  device_frontier_first.CopyFromHost(initial_frontier.data());

  constexpr std::uint32_t kThreads = 256U;
  auto* current_frontier = device_frontier_first.get();
  auto* next_frontier = device_frontier_second.get();
  std::uint32_t current_count = static_cast<std::uint32_t>(initial_frontier.size());
  std::size_t propagation_rounds = 0;
  while (current_count != 0U) {
    if (++propagation_rounds > level_offsets.size() - 1U) {
      throw std::runtime_error("CUDA HT continuation propagation 超出层数上限");
    }
    CheckCuda(cudaMemset(device_next_count.get(), 0, sizeof(std::uint32_t)),
              "cudaMemset(HT wavefront next_count)");
    const std::uint32_t blocks = (current_count + kThreads - 1U) / kThreads;
    PropagateHtContinuationsKernel<<<blocks, kThreads>>>(
        device_states.get(), device_moves.get(), current_frontier, current_count,
        device_status.get(), device_remaining_moves.get(), device_remaining_children.get(),
        device_move_failed.get(), next_frontier, device_next_count.get(),
        static_cast<std::uint32_t>(states.size()));
    CheckCuda(cudaGetLastError(), "PropagateHtContinuationsKernel launch");
    std::uint32_t next_count = 0;
    device_next_count.CopyToHost(&next_count);
    if (next_count > states.size()) {
      throw std::runtime_error("CUDA HT continuation queue 溢出");
    }
    current_count = next_count;
    std::swap(current_frontier, next_frontier);
  }

  device_status.CopyToHost(host_status.data());
  std::vector<std::uint8_t> status(states.size());
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    if (host_status[state_index] != 0 && host_status[state_index] != 1) {
      throw std::runtime_error("CUDA HT continuation 存在未完成状态");
    }
    status[state_index] = static_cast<std::uint8_t>(host_status[state_index]);
  }
  return status;
}

} // namespace cudaee::detail
