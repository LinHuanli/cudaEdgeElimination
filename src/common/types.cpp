#include "cuda_edge_elimination/types.hpp"

#include <stdexcept>

namespace cudaee {

std::string ToString(const DistanceType type) {
  switch (type) {
  case DistanceType::kEuc2D:
    return "EUC_2D";
  case DistanceType::kCeil2D:
    return "CEIL_2D";
  }
  throw std::logic_error("未知距离类型");
}

std::string ToString(const EliminationMethod method) {
  switch (method) {
  case EliminationMethod::kJv:
    return "JV";
  case EliminationMethod::kHamiltonTutte:
    return "HT";
  case EliminationMethod::kGeometryMain:
    return "GEOM_MAIN";
  case EliminationMethod::kLpBox:
    return "LP_BOX";
  case EliminationMethod::kGpuQuickHs:
    return "GPU_QUICK_HS";
  }
  throw std::logic_error("未知消元方法");
}

} // namespace cudaee
