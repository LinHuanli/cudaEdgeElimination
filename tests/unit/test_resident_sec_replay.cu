#include "../../src/cuda/resident_sec_replay.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void Check(const cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}
struct Arena {
  std::vector<void*> memory;
  ~Arena() {
    for (void* p : memory) {
      static_cast<void>(cudaFree(p));
    }
  }
  template <typename T> T* Copy(const std::vector<T>& input) {
    T* p = nullptr;
    Check(cudaMalloc(&p, input.size() * sizeof(T)));
    memory.push_back(p);
    Check(cudaMemcpy(p, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }
};
struct Layout {
  std::int32_t dimension{6}, cut_count{1};
  std::int32_t offset[2]{0, 1}, size[1]{3}, stride[1]{3};
};
} // namespace

int main() {
  int devices = 0;
  Check(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    return 77;
  }
  Check(cudaSetDevice(0));
  for (int fault = 0; fault < 11; ++fault) {
    Arena arena;
    Layout layout;
    std::vector<std::int32_t> membership{0, 0, 0, 3, 3, 3};
    std::vector<std::uint8_t> valid{1, 1, 0, 0, 1, 0, 0}, counts{3, 0};
    std::vector<std::int64_t> dual{10, 2, 0, 0, 3, 0, 0};
    std::vector<std::int32_t> incidence{0, 1, 4, -1, -1, -1};
    // 手工真值：边 0--4 跨三个 SEC，边 0--1 不跨任何 SEC。
    switch (fault) {
    case 1:
      counts[0] = 2;
      break;
    case 2:
      incidence[2] = 1;
      break;
    case 3:
      incidence[2] = 2;
      break;
    case 4:
      counts[0] = 4;
      break;
    case 5:
      dual[4] = -1;
      break;
    case 6:
      membership[0] = -1;
      break;
    case 7:
      membership = {0, 0, 0, 0, 0, 0};
      break;
    case 8:
      valid[2] = 1;
      break;
    case 9:
      layout.size[0] = 6;
      break;
    case 10:
      dual[0] = 1000000000000001LL;
      break;
    default:
      break;
    }
    auto* d_membership = arena.Copy(membership);
    auto* d_valid = arena.Copy(valid);
    auto* d_dual = arena.Copy(dual);
    auto* sizes = arena.Copy(std::vector<std::int32_t>(6, 0));
    auto* invalid = arena.Copy(std::vector<std::int32_t>{0});
    namespace replay = cudaee::detail::resident_sec_replay;
    replay::CountMembersKernel<<<1, 128>>>(6, 1, d_membership, sizes, invalid);
    replay::ValidateCutsKernel<<<1, 128>>>(6, 1, 7, d_membership, sizes, d_valid, d_dual, invalid);
    replay::ValidateIncidenceKernel<<<1, 128>>>(
        2, arena.Copy(std::vector<std::int32_t>{0, 1}), arena.Copy(std::vector<std::int32_t>{0, 0}),
        arena.Copy(std::vector<std::int32_t>{4, 1}),
        arena.Copy(std::vector<std::int32_t>{0, 1, 2, 3, 4, 5}), layout, 1, 1, 3, d_membership,
        d_valid, arena.Copy(counts), arena.Copy(incidence), invalid);
    std::int32_t status = 0;
    Check(cudaMemcpy(&status, invalid, sizeof(status), cudaMemcpyDeviceToHost));
    if ((status == 0) != (fault == 0)) {
      throw std::runtime_error("GPU SEC replay tamper 检查不正确: " + std::to_string(fault));
    }
  }
  std::cout << "SEC independent membership/incidence replay: 11 cases passed\n";
}
