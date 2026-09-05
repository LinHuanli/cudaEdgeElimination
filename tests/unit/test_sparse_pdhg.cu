#include "../../src/fgpu/sparse_pdhg.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void Check(const cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}
class Arena {
public:
  ~Arena() {
    for (void* p : pointers) {
      static_cast<void>(cudaFree(p));
    }
  }
  template <typename T> T* Copy(const std::vector<T>& values) {
    T* p = nullptr;
    Check(cudaMalloc(&p, values.size() * sizeof(T)));
    pointers.push_back(p);
    Check(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }

private:
  std::vector<void*> pointers;
};

struct Row {
  std::int64_t a, b, rhs;
  bool equality;
};

// 独立的二维 LP 顶点枚举 oracle：不复用 GPU 更新式或迭代过程。
double Oracle(const std::vector<Row>& rows, const std::array<std::int64_t, 2>& c) {
  std::vector<Row> boundaries = rows;
  boundaries.insert(boundaries.end(),
                    {{1, 0, 0, false}, {-1, 0, -1, false}, {0, 1, 0, false}, {0, -1, -1, false}});
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < boundaries.size(); ++i) {
    for (std::size_t j = i + 1; j < boundaries.size(); ++j) {
      const auto first = boundaries[i], second = boundaries[j];
      const double determinant = static_cast<double>(first.a * second.b - first.b * second.a);
      if (determinant == 0.0) {
        continue;
      }
      const double x =
          static_cast<double>(first.rhs * second.b - first.b * second.rhs) / determinant;
      const double y =
          static_cast<double>(first.a * second.rhs - first.rhs * second.a) / determinant;
      bool feasible = true;
      for (const auto& row : boundaries) {
        const double residual =
            static_cast<double>(row.a) * x + static_cast<double>(row.b) * y - row.rhs;
        feasible = feasible && (row.equality ? std::abs(residual) <= 1.0e-9 : residual >= -1.0e-9);
      }
      if (feasible) {
        best = std::min(best, static_cast<double>(c[0]) * x + static_cast<double>(c[1]) * y);
      }
    }
  }
  return best;
}
} // namespace

int main(const int argc, char** argv) {
  const bool sanitizer = argc == 2 && std::string_view(argv[1]) == "--sanitizer";
  if (argc != 1 && !sanitizer) {
    throw std::invalid_argument("仅支持可选 --sanitizer（缩短单元迭代，不影响正式求解）");
  }
  int devices = 0;
  Check(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    return 77;
  }
  Check(cudaSetDevice(0));
  for (std::int32_t sample = 0; sample < (sanitizer ? 2 : 8); ++sample) {
    Arena arena;
    std::vector<Row> rows{{1, 1, 1, sample % 2 == 0}};
    std::array<std::int64_t, 2> costs{1 + sample, 9 - sample};
    if (sample >= 2) {
      rows.push_back({1, -1, 0, sample % 2 != 0});
    }
    if (sample >= 4) {
      rows.push_back({3, 2, 2, false});
    }
    if (sample >= 6) {
      costs[0] = -2;
    }
    std::vector<std::int64_t> row_offsets{0}, column_offsets{0}, values, transpose_values, rhs;
    std::vector<std::int32_t> columns, row_ids;
    std::vector<std::uint8_t> equality;
    for (const Row& row : rows) {
      columns.push_back(0);
      values.push_back(row.a);
      columns.push_back(1);
      values.push_back(row.b);
      row_offsets.push_back(static_cast<std::int64_t>(values.size()));
      rhs.push_back(row.rhs);
      equality.push_back(row.equality ? 1U : 0U);
    }
    for (int column = 0; column < 2; ++column) {
      for (std::size_t row = 0; row < rows.size(); ++row) {
        row_ids.push_back(static_cast<std::int32_t>(row));
        transpose_values.push_back(column == 0 ? rows[row].a : rows[row].b);
      }
      column_offsets.push_back(static_cast<std::int64_t>(transpose_values.size()));
    }
    cudaee::detail::SparsePdhgDeviceModel model{
        static_cast<std::int32_t>(rows.size()),
        2,
        static_cast<std::int64_t>(values.size()),
        arena.Copy(row_offsets),
        arena.Copy(columns),
        arena.Copy(values),
        arena.Copy(column_offsets),
        arena.Copy(row_ids),
        arena.Copy(transpose_values),
        arena.Copy(rhs),
        arena.Copy(equality),
        arena.Copy(std::vector<std::int64_t>{costs[0], costs[1]}),
        nullptr,
        1U};
    cudaee::detail::SparsePdhgCuda solver(0);
    const std::uint32_t first_steps = sanitizer ? 64U : 16001U;
    const std::uint32_t second_steps = sanitizer ? 65U : 15999U;
    const auto first = solver.Iterate(model, 10.0, first_steps);
    const auto second = solver.Iterate(model, 10.0, second_steps);
    const double optimum = Oracle(rows, costs);
    if (!std::isfinite(optimum) || second.iterations != first_steps + second_steps ||
        (!sanitizer &&
         (second.primal_violation > 0.002 || std::abs(second.primal_objective - optimum) > 0.002 ||
          std::abs(second.dual_objective - optimum) > 0.002)) ||
        second.dual_objective > optimum + 1.0e-9 || first.iterations != first_steps) {
      throw std::runtime_error(
          "sparse PDHG 与独立 LP oracle 不一致，case=" + std::to_string(sample) +
          " primal=" + std::to_string(second.primal_objective) +
          " dual=" + std::to_string(second.dual_objective) + " optimum=" + std::to_string(optimum));
    }
    // 同一 solver 更换 snapshot 版本后保留 warm start，但平均迭代计数重置。
    model.version = 2U;
    const std::uint32_t warm_steps = sanitizer ? 65U : 1025U;
    const auto warm = solver.Iterate(model, 10.0, warm_steps);
    if (warm.iterations != warm_steps ||
        (!sanitizer &&
         (warm.primal_violation > 0.002 || std::abs(warm.primal_objective - optimum) > 0.002))) {
      throw std::runtime_error("sparse PDHG warm-start snapshot 重建失败");
    }
    if (sample == 0) {
      // 不可信模型先验证 offsets，再验证转置，不能用坏索引探测设备内存。
      for (int fault = 0; fault < 5; ++fault) {
        auto broken = model;
        if (fault == 0) {
          broken.row_offsets = arena.Copy(std::vector<std::int64_t>{0, 3});
        } else if (fault == 1) {
          broken.column_values = arena.Copy(std::vector<std::int64_t>{1, 2});
        } else if (fault == 2) {
          broken.column_ids = arena.Copy(std::vector<std::int32_t>{0, 2});
        } else if (fault == 3) {
          broken.active_ids = arena.Copy(std::vector<std::int32_t>{0, 0});
          broken.active_count = 2;
        } else {
          broken.active_ids = arena.Copy(std::vector<std::int32_t>{0});
          broken.active_count = 1;
        }
        bool rejected = false;
        try {
          static_cast<void>(solver.Iterate(broken, 10.0, 1U));
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        if (!rejected) {
          throw std::runtime_error("损坏的 sparse PDHG 模型未被拒绝");
        }
        // 坏模型会销毁旧执行图；恢复原版本也必须重建，不能命中空图缓存。
        static_cast<void>(solver.Iterate(model, 10.0, 1U));
      }
      model.active = arena.Copy(std::vector<std::uint8_t>{1U, 0U});
      model.active_ids = arena.Copy(std::vector<std::int32_t>{0});
      model.active_count = 1;
      ++model.version;
      const auto active = solver.Iterate(model, 10.0, sanitizer ? 65U : 16001U);
      double primal[2]{};
      Check(cudaMemcpy(primal, solver.primal(), sizeof(primal), cudaMemcpyDeviceToHost));
      if (primal[1] != 0.0 || active.dual_objective > 1.0 + 1.0e-9 ||
          (!sanitizer && (std::abs(primal[0] - 1.0) > 0.002 || active.primal_violation > 0.002))) {
        throw std::runtime_error("sparse PDHG 紧凑活动列更新不正确");
      }
    }
  }
  std::cout << "sparse PDHG oracle, tail, active-column and invalid-model tests passed\n";
}
