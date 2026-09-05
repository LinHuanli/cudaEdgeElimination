#include "../fgpu/gpu_bootstrap.hpp"
#include "../fgpu/permutation_catalog.hpp"
#include "../fgpu/quick_hs_predicate.hpp"
#include "device_workspace.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>
#include <stdexcept>

namespace cudaee::detail {
namespace {
using Clock = std::chrono::steady_clock;
constexpr int kThreads = 128;
double Milliseconds(const Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int ChooseDevice(int requested) {
  int count = 0;
  CheckWorkspaceCuda(cudaGetDeviceCount(&count));
  if (requested < -1 || requested >= count || count == 0) {
    throw std::invalid_argument("GPU bootstrap 设备不可用");
  }
  if (requested < 0) {
    std::size_t largest = 0;
    for (int device = 0; device < count; ++device) {
      CheckWorkspaceCuda(cudaSetDevice(device));
      std::size_t free = 0, total = 0;
      CheckWorkspaceCuda(cudaMemGetInfo(&free, &total));
      if (requested < 0 || free > largest) {
        requested = device;
        largest = free;
      }
    }
  }
  CheckWorkspaceCuda(cudaSetDevice(requested));
  return requested;
}

// 核心与重放使用不同解码方式：生成按顶点行展开，重放按线性 edge id 二分。
__global__ void BuildDistancesKernel(const quick_hs::GraphView graph, std::int64_t* out) {
  const std::int64_t u = blockIdx.x;
  const std::int64_t begin = u * (2LL * graph.dimension - u - 1) / 2;
  for (int v = static_cast<int>(u) + 1 + threadIdx.x; v < graph.dimension; v += blockDim.x) {
    out[begin + v - u - 1] = quick_hs::Distance(graph, static_cast<int>(u), v);
  }
}

__global__ void ReplayDistancesKernel(const quick_hs::GraphView graph, const std::int64_t count,
                                      const std::int64_t* distances, int* invalid, Edge* edges) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  std::int64_t low = 0, high = graph.dimension - 2;
  while (low < high) {
    const auto mid = (low + high + 1) / 2;
    if (mid * (2LL * graph.dimension - mid - 1) / 2 <= index)
      low = mid;
    else
      high = mid - 1;
  }
  const int u = static_cast<int>(low);
  const int v = static_cast<int>(index - low * (2LL * graph.dimension - low - 1) / 2 + low + 1);
  // 独立逐 bit 平方根，不消费 proposal 的 sqrt 或缓存。
  const auto dx = graph.coordinate_x[u] - graph.coordinate_x[v];
  const auto dy = graph.coordinate_y[u] - graph.coordinate_y[v];
  const auto ax = static_cast<std::uint64_t>(dx < 0 ? -dx : dx);
  const auto ay = static_cast<std::uint64_t>(dy < 0 ? -dy : dy);
  const auto squared = ax * ax + ay * ay;
  std::uint64_t value = squared, root = 0, bit = 1ULL << 62;
  while (bit > value)
    bit >>= 2;
  while (bit != 0) {
    if (value >= root + bit) {
      value -= root + bit;
      root = (root >> 1) + bit;
    } else
      root >>= 1;
    bit >>= 2;
  }
  const auto whole = root / graph.coordinate_denominator;
  const auto scaled_whole = whole * graph.coordinate_denominator;
  const auto expected =
      graph.coordinate_denominator == 1U
          ? static_cast<std::int64_t>(root + (graph.distance_type == 0
                                                  ? squared - root * root > root
                                                  : squared != root * root))
          : static_cast<std::int64_t>(whole + (graph.distance_type == 0
                                                   ? root > scaled_whole
                                                   : squared != scaled_whole * scaled_whole));
  // 保证整个 tour 成本、局部改进差值以及既有证明谓词都不溢出。
  const auto divisor = graph.dimension > 64 ? graph.dimension : 64;
  if (expected != distances[index] || expected > INT64_MAX / divisor)
    atomicExch(invalid, 1);
  if (edges != nullptr)
    edges[index] = Edge{u, v, expected, true};
}

struct Choice {
  std::int64_t delta;
  std::int64_t ordinal;
};

__global__ void BuildPermutationKernel(const int nodes, std::uint8_t* const catalog) {
  const int rank = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int rows = permutation_catalog::Factorial(nodes);
  if (rank >= rows)
    return;
  int remaining_rank = rank;
  unsigned available = (1U << nodes) - 1U;
  const auto offset = permutation_catalog::Offset(nodes);
  for (int position = 0; position < nodes; ++position) {
    const int factorial = permutation_catalog::Factorial(nodes - position - 1);
    int choice = remaining_rank / factorial;
    remaining_rank %= factorial;
    unsigned selected = available;
    while (choice-- > 0)
      selected &= selected - 1U;
    const int vertex = __ffs(static_cast<int>(selected)) - 1;
    catalog[offset + static_cast<std::size_t>(position) * rows + rank] = vertex;
    available &= ~(1U << vertex);
  }
}

__global__ void ReplayPermutationKernel(const int nodes, const std::uint8_t* const catalog,
                                        int* const invalid) {
  const int expected_rank = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int rows = permutation_catalog::Factorial(nodes);
  if (expected_rank >= rows)
    return;
  const auto offset = permutation_catalog::Offset(nodes);
  int rank = 0;
  unsigned seen = 0U;
  for (int position = 0; position < nodes; ++position) {
    const unsigned vertex =
        catalog[offset + static_cast<std::size_t>(position) * rows + expected_rank];
    if (vertex >= static_cast<unsigned>(nodes) || (seen & (1U << vertex)) != 0U) {
      atomicExch(invalid, 1);
      return;
    }
    // 从输出排列重建 Lehmer rank，独立于候选的逐位选择算法。
    int smaller = 0;
    for (int next = position + 1; next < nodes; ++next)
      smaller += catalog[offset + static_cast<std::size_t>(next) * rows + expected_rank] < vertex;
    rank = rank * (nodes - position) + smaller;
    seen |= 1U << vertex;
  }
  if (rank != expected_rank || seen != (1U << nodes) - 1U)
    atomicExch(invalid, 1);
}

__device__ Choice Best(Choice local, Choice* shared) {
  const int tid = threadIdx.x;
  shared[tid] = local;
  __syncthreads();
  for (int stride = kThreads / 2; stride != 0; stride /= 2) {
    if (tid < stride) {
      const auto other = shared[tid + stride];
      if (other.delta < shared[tid].delta ||
          (other.delta == shared[tid].delta && other.ordinal < shared[tid].ordinal)) {
        shared[tid] = other;
      }
    }
    __syncthreads();
  }
  const Choice result = shared[0];
  __syncthreads(); // 下次归约不能在其他线程读结果前覆写共享内存。
  return result;
}

__global__ void IncumbentKernel(const quick_hs::GraphView graph, int* tours, int* scratch,
                                std::uint8_t* visited, std::int64_t* costs,
                                unsigned long long* improvements) {
  __shared__ Choice shared[kThreads];
  __shared__ std::int64_t current_cost;
  __shared__ unsigned long long changes;
  const int n = graph.dimension;
  const int tid = threadIdx.x;
  auto* tour = tours + static_cast<std::int64_t>(blockIdx.x) * n;
  auto* temporary = scratch + static_cast<std::int64_t>(blockIdx.x) * n;
  auto* used = visited + static_cast<std::int64_t>(blockIdx.x) * n;
  for (int v = tid; v < n; v += blockDim.x)
    used[v] = 0;
  if (tid == 0) {
    tour[0] = static_cast<int>(static_cast<std::int64_t>(blockIdx.x) * n / gridDim.x);
    current_cost = 0;
    changes = 0;
  }
  __syncthreads();
  if (tid == 0)
    used[tour[0]] = 1;
  __syncthreads();
  for (int position = 1; position < n; ++position) {
    Choice local{INT64_MAX, INT64_MAX};
    for (int v = tid; v < n; v += blockDim.x) {
      if (used[v])
        continue;
      const auto cost = quick_hs::Distance(graph, tour[position - 1], v);
      if (cost < local.delta || (cost == local.delta && v < local.ordinal))
        local = {cost, v};
    }
    const auto selected = Best(local, shared);
    if (tid == 0) {
      tour[position] = static_cast<int>(selected.ordinal);
      used[selected.ordinal] = 1;
      current_cost += selected.delta;
    }
    __syncthreads();
  }
  if (tid == 0)
    current_cost += quick_hs::Distance(graph, tour[n - 1], tour[0]);
  __syncthreads();

  // 每次严格降低整数 tour 成本，故无需迭代预算；平局按 move ordinal 确定选择。
  for (;;) {
    Choice local{0, INT64_MAX};
    const auto pairs = static_cast<std::int64_t>(n) * n;
    for (auto key = static_cast<std::int64_t>(tid); key < pairs; key += blockDim.x) {
      const int i = static_cast<int>(key / n), j = static_cast<int>(key % n);
      if (j <= i + 1 || (i == 0 && j == n - 1))
        continue;
      const auto delta = quick_hs::Distance(graph, tour[i], tour[j]) +
                         quick_hs::Distance(graph, tour[i + 1], tour[(j + 1) % n]) -
                         quick_hs::Distance(graph, tour[i], tour[i + 1]) -
                         quick_hs::Distance(graph, tour[j], tour[(j + 1) % n]);
      if (delta < local.delta || (delta == local.delta && key < local.ordinal))
        local = {delta, key};
    }
    auto selected = Best(local, shared);
    if (selected.delta < 0) {
      const int left = static_cast<int>(selected.ordinal / n) + 1;
      const int right = static_cast<int>(selected.ordinal % n);
      for (int offset = tid; offset < (right - left + 1) / 2; offset += blockDim.x) {
        const int saved = tour[left + offset];
        tour[left + offset] = tour[right - offset];
        tour[right - offset] = saved;
      }
      if (tid == 0) {
        current_cost += selected.delta;
        ++changes;
      }
      __syncthreads();
      continue;
    }
    local = {0, INT64_MAX};
    // Or-opt 长度 1/2/3 的完整循环邻域，允许移动跨越 tour 数组边界的片段。
    for (auto key = static_cast<std::int64_t>(tid); key < 3 * pairs; key += blockDim.x) {
      const int length = static_cast<int>(key / pairs) + 1;
      const int i = static_cast<int>((key % pairs) / n), j = static_cast<int>(key % n);
      if (length >= n - 1 || j == (i + n - 1) % n || (j - i + n) % n < length)
        continue;
      const int before = tour[(i + n - 1) % n], first = tour[i];
      const int last = tour[(i + length - 1) % n], after = tour[(i + length) % n];
      const int to = tour[j], next = tour[(j + 1) % n];
      const auto delta =
          quick_hs::Distance(graph, before, after) + quick_hs::Distance(graph, to, first) +
          quick_hs::Distance(graph, last, next) - quick_hs::Distance(graph, before, first) -
          quick_hs::Distance(graph, last, after) - quick_hs::Distance(graph, to, next);
      if (delta < local.delta || (delta == local.delta && key < local.ordinal))
        local = {delta, key};
    }
    selected = Best(local, shared);
    if (selected.delta >= 0)
      break;
    if (tid == 0) {
      const int length = static_cast<int>(selected.ordinal / pairs) + 1;
      const int i = static_cast<int>((selected.ordinal % pairs) / n);
      const int j = static_cast<int>(selected.ordinal % n);
      int out = 0;
      for (int position = 0; position < n; ++position) {
        if ((position - i + n) % n < length)
          continue;
        temporary[out++] = tour[position];
        if (position == j)
          for (int k = 0; k < length; ++k)
            temporary[out++] = tour[(i + k) % n];
      }
      current_cost += selected.delta;
      ++changes;
    }
    __syncthreads();
    for (int position = tid; position < n; position += blockDim.x)
      tour[position] = temporary[position];
    __syncthreads();
  }
  if (tid == 0) {
    costs[blockIdx.x] = current_cost;
    improvements[blockIdx.x] = changes;
  }
}

// 与候选搜索完全分离：重新检验排列、独立坐标成本；失败不提供任何 UB。
__global__ void ReplayIncumbentKernel(quick_hs::GraphView graph, const int* tours,
                                      const std::int64_t* costs, int* seen, int* invalid) {
  __shared__ std::int64_t partial[kThreads];
  graph.triangular_distance = nullptr;
  const int n = graph.dimension;
  const auto* tour = tours + static_cast<std::int64_t>(blockIdx.x) * n;
  auto* marks = seen + static_cast<std::int64_t>(blockIdx.x) * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    marks[i] = 0;
  __syncthreads();
  std::int64_t sum = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int a = tour[i], b = tour[(i + 1) % n];
    if (a < 0 || a >= n || b < 0 || b >= n) {
      atomicExch(invalid, 1);
      continue;
    }
    if (atomicExch(marks + a, 1) != 0)
      atomicExch(invalid, 1);
    sum += quick_hs::Distance(graph, a, b);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0 && partial[0] != costs[blockIdx.x])
    atomicExch(invalid, 1);
}

__global__ void SelectIncumbentKernel(const int starts, const std::int64_t* costs, int* best) {
  if (threadIdx.x != 0)
    return;
  int selected = 0;
  for (int i = 1; i < starts; ++i)
    if (costs[i] < costs[selected])
      selected = i;
  *best = selected;
}
} // namespace

struct GpuBootstrap::Impl {
  int device;
  quick_hs::GraphView view{};
  CudaWorkspace<std::int64_t> x, y, distance;
  CudaWorkspace<std::uint8_t> permutations;
  GpuBootstrapMetrics metrics;
  std::vector<int> tour;
  Clock::time_point begin{Clock::now()};
  Impl(const GraphSnapshot& graph, int selected)
      : device(selected), x(selected), y(selected), distance(selected), permutations(selected) {
    if (graph.dimension < 3 || graph.points.size() != static_cast<std::size_t>(graph.dimension) ||
        (!graph.integer_coordinates && graph.integer_coordinate_denominator != 2U) ||
        !graph.integer_distance_safe ||
        (graph.integer_coordinate_denominator != 1U &&
         graph.integer_coordinate_denominator != 2U) ||
        (graph.distance_type != DistanceType::kEuc2D &&
         graph.distance_type != DistanceType::kCeil2D)) {
      throw std::invalid_argument("GPU bootstrap 需要 n>=3 的安全整数 EUC_2D/CEIL_2D 坐标");
    }
    const auto count = static_cast<std::int64_t>(graph.dimension) * (graph.dimension - 1) / 2;
    if (count > INT32_MAX)
      throw std::overflow_error("GPU 完整图超出 stable edge id 范围");
    x.Reserve(graph.dimension);
    y.Reserve(graph.dimension);
    distance.Reserve(count);
    std::vector<std::int64_t> hx(graph.dimension), hy(graph.dimension);
    for (int i = 0; i < graph.dimension; ++i) {
      hx[i] = graph.points[i].integer_x;
      hy[i] = graph.points[i].integer_y;
    }
    CheckWorkspaceCuda(cudaMemcpy(x.get(), hx.data(), x.bytes(), cudaMemcpyHostToDevice));
    CheckWorkspaceCuda(cudaMemcpy(y.get(), hy.data(), y.bytes(), cudaMemcpyHostToDevice));
    view.dimension = graph.dimension;
    view.coordinate_x = x.get();
    view.coordinate_y = y.get();
    view.distance_type = static_cast<std::uint8_t>(graph.distance_type);
    view.coordinate_denominator = graph.integer_coordinate_denominator;
    CudaWorkspace<int> invalid(device);
    invalid.Reserve(1);
    CheckWorkspaceCuda(cudaMemset(invalid.get(), 0, sizeof(int)));
    const auto start = Clock::now();
    BuildDistancesKernel<<<graph.dimension, kThreads>>>(view, distance.get());
    ReplayDistancesKernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads>>>(
        view, count, distance.get(), invalid.get(), nullptr);
    CheckWorkspaceCuda(cudaGetLastError());
    int status = 0;
    CheckWorkspaceCuda(cudaMemcpy(&status, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost));
    if (status != 0)
      throw std::runtime_error("GPU 距离缓存精确 replay 失败或成本溢出");
    view.triangular_distance = distance.get();
    metrics.distance_bytes = distance.bytes();
    metrics.distance_ms = Milliseconds(start);
    metrics.total_ms = Milliseconds(begin);
  }
};

GpuBootstrap::GpuBootstrap(const GraphSnapshot& graph, int device)
    : impl_(std::make_unique<Impl>(graph, ChooseDevice(device))) {}
GpuBootstrap::~GpuBootstrap() = default;
const std::int64_t* GpuBootstrap::distances() const { return impl_->distance.get(); }
const std::uint8_t* GpuBootstrap::permutations() const { return impl_->permutations.get(); }
int GpuBootstrap::device() const { return impl_->device; }
const GpuBootstrapMetrics& GpuBootstrap::metrics() const { return impl_->metrics; }
const std::vector<std::int32_t>& GpuBootstrap::tour() const { return impl_->tour; }

void GpuBootstrap::BuildPermutationCatalog() {
  CheckWorkspaceCuda(cudaSetDevice(impl_->device));
  const auto start = Clock::now();
  impl_->permutations.Reserve(permutation_catalog::kBytes);
  CudaWorkspace<int> invalid(impl_->device);
  invalid.Reserve(1);
  CheckWorkspaceCuda(cudaMemset(invalid.get(), 0, sizeof(int)));
  for (int nodes = 1; nodes <= permutation_catalog::kMaximumNodes; ++nodes) {
    const int blocks = (permutation_catalog::Factorial(nodes) + kThreads - 1) / kThreads;
    BuildPermutationKernel<<<blocks, kThreads>>>(nodes, impl_->permutations.get());
    ReplayPermutationKernel<<<blocks, kThreads>>>(nodes, impl_->permutations.get(), invalid.get());
    CheckWorkspaceCuda(cudaGetLastError());
  }
  int status{};
  CheckWorkspaceCuda(cudaMemcpy(&status, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost));
  if (status != 0)
    throw std::runtime_error("GPU 排列表未通过完整 rank/排列重放");
  impl_->metrics.permutation_bytes = impl_->permutations.bytes();
  impl_->metrics.permutation_ms = Milliseconds(start);
  impl_->metrics.total_ms = Milliseconds(impl_->begin);
}

void GpuBootstrap::BuildCompleteGraph(GraphSnapshot* graph) {
  if (graph == nullptr || graph->dimension != impl_->view.dimension || !graph->edges.empty())
    throw std::invalid_argument("GPU 图构造必须接收同维度、尚未创建边的坐标图");
  CheckWorkspaceCuda(cudaSetDevice(impl_->device));
  const auto count = impl_->distance.bytes() / sizeof(std::int64_t);
  CudaWorkspace<Edge> edges(impl_->device);
  edges.Reserve(count);
  CudaWorkspace<int> invalid(impl_->device);
  invalid.Reserve(1);
  CheckWorkspaceCuda(cudaMemset(invalid.get(), 0, sizeof(int)));
  ReplayDistancesKernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads>>>(
      impl_->view, static_cast<std::int64_t>(count), impl_->distance.get(), invalid.get(),
      edges.get());
  CheckWorkspaceCuda(cudaGetLastError());
  int status = 0;
  CheckWorkspaceCuda(cudaMemcpy(&status, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost));
  if (status != 0)
    throw std::runtime_error("GPU 完整图重放失败");
  graph->edges.resize(count);
  CheckWorkspaceCuda(
      cudaMemcpy(graph->edges.data(), edges.get(), edges.bytes(), cudaMemcpyDeviceToHost));
  // 不在 CPU 生成完整 CSR；消除主链的首次几何收缩后直接在设备上生成稀疏 CSR。
  impl_->metrics.total_ms = Milliseconds(impl_->begin);
}

void GpuBootstrap::GenerateIncumbent() {
  CheckWorkspaceCuda(cudaSetDevice(impl_->device));
  const auto start = Clock::now();
  const int n = impl_->view.dimension, starts = std::min(128, n);
  const auto cells = static_cast<std::uint64_t>(n) * starts;
  CudaWorkspace<int> tours(impl_->device), scratch(impl_->device), seen(impl_->device),
      invalid(impl_->device), best(impl_->device);
  CudaWorkspace<std::uint8_t> visited(impl_->device);
  CudaWorkspace<std::int64_t> costs(impl_->device);
  CudaWorkspace<unsigned long long> improvements(impl_->device);
  tours.Reserve(cells);
  scratch.Reserve(cells);
  seen.Reserve(cells);
  visited.Reserve(cells);
  costs.Reserve(starts);
  improvements.Reserve(starts);
  invalid.Reserve(1);
  best.Reserve(1);
  CheckWorkspaceCuda(cudaMemset(invalid.get(), 0, sizeof(int)));
  IncumbentKernel<<<starts, kThreads>>>(impl_->view, tours.get(), scratch.get(), visited.get(),
                                        costs.get(), improvements.get());
  ReplayIncumbentKernel<<<starts, kThreads>>>(impl_->view, tours.get(), costs.get(), seen.get(),
                                              invalid.get());
  CheckWorkspaceCuda(cudaGetLastError());
  int status = 0;
  CheckWorkspaceCuda(cudaMemcpy(&status, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost));
  if (status != 0)
    throw std::runtime_error("GPU incumbent 排列或成本 replay 失败");
  SelectIncumbentKernel<<<1, 1>>>(starts, costs.get(), best.get());
  CheckWorkspaceCuda(cudaGetLastError());
  int selected = 0;
  CheckWorkspaceCuda(cudaMemcpy(&selected, best.get(), sizeof(int), cudaMemcpyDeviceToHost));
  CheckWorkspaceCuda(cudaMemcpy(&impl_->metrics.incumbent_cost, costs.get() + selected,
                                sizeof(std::int64_t), cudaMemcpyDeviceToHost));
  impl_->tour.resize(n);
  CheckWorkspaceCuda(cudaMemcpy(impl_->tour.data(),
                                tours.get() + static_cast<std::int64_t>(selected) * n,
                                n * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<unsigned long long> counts(starts);
  CheckWorkspaceCuda(
      cudaMemcpy(counts.data(), improvements.get(), improvements.bytes(), cudaMemcpyDeviceToHost));
  impl_->metrics.improvements = 0;
  for (const auto count : counts)
    impl_->metrics.improvements += count;
  impl_->metrics.starts = static_cast<std::uint32_t>(starts);
  impl_->metrics.incumbent_ms = Milliseconds(start);
  impl_->metrics.total_ms = Milliseconds(impl_->begin);
}

} // namespace cudaee::detail
