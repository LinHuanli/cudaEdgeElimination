#pragma once

#include <cstdint>
#include <memory>

namespace cudaee::detail {

// 原始稀疏 LP：min c*x，等式或 >= 行，0<=x<=1。列 ID 在删边后不变，
// inactive 列固定为 0。CSR 与 CSC 的整数系数必须表示同一矩阵。
// 本接口只寻找 multiplier，不直接授权任何图修改。
struct SparsePdhgDeviceModel {
  std::int32_t rows{};
  std::int32_t columns{};
  std::int64_t nonzeros{};
  const std::int64_t* row_offsets{};
  const std::int32_t* column_ids{};
  const std::int64_t* row_values{};
  const std::int64_t* column_offsets{};
  const std::int32_t* row_ids{};
  const std::int64_t* column_values{};
  const std::int64_t* rhs{};
  const std::uint8_t* equality{};
  const std::int64_t* objective{};
  const std::uint8_t* active{};
  std::uint64_t version{};
  // 可选的紧凑活动列列表：稳定列数组不搬迁，但每次 primal 更新仅
  // 访问这些列。调用者必须在 active mask 变化时更新 model version。
  const std::int32_t* active_ids{};
  std::int32_t active_count{};
  // 版本内所有设备数组必须不可变；更换地址也会强制重建 CUDA Graph。
  bool operator==(const SparsePdhgDeviceModel&) const = default;
};

struct SparsePdhgDiagnostics {
  std::uint64_t iterations{};
  double primal_violation{};
  double relative_gap{};
  double primal_objective{};
  double dual_objective{};
  double solve_ms{};
  bool converged{};
};

class SparsePdhgCuda {
public:
  explicit SparsePdhgCuda(int device);
  ~SparsePdhgCuda();
  SparsePdhgCuda(const SparsePdhgCuda&) = delete;
  SparsePdhgCuda& operator=(const SparsePdhgCuda&) = delete;

  // 更换模型版本时重建执行图；stable rows/columns 的 warm start 保留。
  // iterations 是一次调度批次长度，不是整个 cutting-plane 的停止预算。
  [[nodiscard]] SparsePdhgDiagnostics Iterate(const SparsePdhgDeviceModel& model,
                                              double objective_scale, std::uint32_t iterations);
  [[nodiscard]] const double* primal() const;
  [[nodiscard]] const double* dual() const;
  [[nodiscard]] std::uint64_t workspace_bytes() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cudaee::detail
