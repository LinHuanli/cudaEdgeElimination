#include "cuda_edge_elimination/local_search.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool KOptCostCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前二进制未编译 CUDA 后端";
  }
  return false;
}

std::vector<std::int64_t> EvaluateKOptTemplateCostsCuda(const GraphSnapshot&,
                                                        const KOptReconnectTable&,
                                                        const std::vector<KOptCostTask>&, int*,
                                                        KOptCudaCacheUsage*) {
  throw std::runtime_error("当前二进制未编译 CUDA k-opt cost 后端");
}

void ClearKOptCostCudaCache() {}

} // namespace cudaee::detail
