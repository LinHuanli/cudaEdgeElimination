#pragma once

#include "quick_hs_predicate.hpp"
#include "sparse_pdhg.hpp"

#include <cstdint>
#include <memory>

namespace cudaee::detail {

// 过渡适配器：把现有 SEC incidence 转换成通用 CSR/CSC LP。它不增加
// cut 类别，不把静态 SEC 冒充一般 local-cut；新 cut pool 可复用同一求解器。
class ResidentSecModelCuda {
public:
  explicit ResidentSecModelCuda(int device);
  ~ResidentSecModelCuda();
  ResidentSecModelCuda(const ResidentSecModelCuda&) = delete;
  ResidentSecModelCuda& operator=(const ResidentSecModelCuda&) = delete;
  [[nodiscard]] SparsePdhgDeviceModel
  Build(quick_hs::GraphView graph, const std::int64_t* costs, std::int32_t active_count,
        const std::int32_t* active_ids, std::int32_t cut_count, std::int32_t static_cut_count,
        const std::uint8_t* valid_cuts, const std::uint8_t* incidence_count,
        const std::int32_t* incidence_ids, std::int32_t incidence_stride, std::uint64_t version);
  [[nodiscard]] std::uint64_t workspace_bytes() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cudaee::detail
