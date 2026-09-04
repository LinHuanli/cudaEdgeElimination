#include "cuda_edge_elimination/fgpu.hpp"

#include "../cpu/elimination_commit.hpp"
#include "cuda_edge_elimination/cuda_device_affinity.hpp"
#include "lp_box_verifier.hpp"
#include "pdlp_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

struct DegreeModel {
  LpEpoch epoch;
  std::vector<std::int32_t> column_edge_id;
};

class DisjointSet {
public:
  explicit DisjointSet(const std::int32_t dimension)
      : parent_(static_cast<std::size_t>(dimension)), rank_(static_cast<std::size_t>(dimension)) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  std::int32_t Find(const std::int32_t node) {
    std::int32_t& parent = parent_[static_cast<std::size_t>(node)];
    if (parent != node) {
      parent = Find(parent);
    }
    return parent;
  }

  void Unite(const std::int32_t first, const std::int32_t second) {
    std::int32_t first_root = Find(first);
    std::int32_t second_root = Find(second);
    if (first_root == second_root) {
      return;
    }
    const std::uint8_t first_rank = rank_[static_cast<std::size_t>(first_root)];
    const std::uint8_t second_rank = rank_[static_cast<std::size_t>(second_root)];
    if (first_rank < second_rank || (first_rank == second_rank && first_root > second_root)) {
      std::swap(first_root, second_root);
    }
    parent_[static_cast<std::size_t>(second_root)] = first_root;
    if (first_rank == second_rank) {
      ++rank_[static_cast<std::size_t>(first_root)];
    }
  }

private:
  std::vector<std::int32_t> parent_;
  std::vector<std::uint8_t> rank_;
};

std::vector<std::int32_t> CanonicalCut(std::vector<std::int32_t> side,
                                       const std::int32_t dimension) {
  std::sort(side.begin(), side.end());
  side.erase(std::unique(side.begin(), side.end()), side.end());
  if (side.size() <= 1U || side.size() + 1U >= static_cast<std::size_t>(dimension)) {
    return {};
  }
  std::vector<std::int32_t> complement;
  complement.reserve(static_cast<std::size_t>(dimension) - side.size());
  std::size_t cursor = 0U;
  for (std::int32_t node = 0; node < dimension; ++node) {
    if (cursor < side.size() && side[cursor] == node) {
      ++cursor;
    } else {
      complement.push_back(node);
    }
  }
  if (complement.size() < side.size() || (complement.size() == side.size() && complement < side)) {
    return complement;
  }
  return side;
}

class SubtourCutPool {
public:
  bool Add(std::vector<std::int32_t> side, const std::int32_t dimension) {
    side = CanonicalCut(std::move(side), dimension);
    if (side.empty() || !identities_.insert(side).second) {
      return false;
    }
    cuts_.push_back(std::move(side));
    return true;
  }

  [[nodiscard]] const std::vector<std::vector<std::int32_t>>& cuts() const { return cuts_; }
  [[nodiscard]] std::size_t size() const { return cuts_.size(); }

private:
  std::set<std::vector<std::int32_t>> identities_;
  std::vector<std::vector<std::int32_t>> cuts_;
};

double ClampedPrimal(double value);

std::string ShellQuote(const std::string& value) {
  std::string result{"'"};
  for (const char character : value) {
    if (character == '\'') {
      result += "'\\''";
    } else {
      result += character;
    }
  }
  result += '\'';
  return result;
}

std::size_t AddConcordeMincutOracleCuts(const GraphSnapshot& graph, const DegreeModel& model,
                                        const std::vector<double>& primal,
                                        const std::filesystem::path& oracle,
                                        const std::uint32_t epoch, SubtourCutPool* const pool) {
  if (oracle.empty()) {
    return 0U;
  }
  const std::filesystem::path repository =
      std::filesystem::weakly_canonical(std::filesystem::path(CUDAEE_SOURCE_DIR));
  const std::filesystem::path executable = std::filesystem::canonical(oracle);
  if (!std::filesystem::is_regular_file(executable)) {
    throw std::invalid_argument("subtour mincut oracle 不是普通文件");
  }
  const std::filesystem::path temporary_directory = repository / ".tmp";
  std::filesystem::create_directories(temporary_directory);
  const std::filesystem::path input =
      temporary_directory /
      ("subtour-mincut-" + std::to_string(epoch) + "-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".edg");
  {
    std::ofstream output(input, std::ios::trunc);
    if (!output) {
      throw std::runtime_error("无法创建 subtour mincut oracle 输入");
    }
    output << graph.dimension << ' ' << model.column_edge_id.size() << '\n';
    output << std::setprecision(17);
    for (std::size_t column = 0U; column < model.column_edge_id.size(); ++column) {
      const Edge& edge = graph.edges[static_cast<std::size_t>(model.column_edge_id[column])];
      output << edge.u << ' ' << edge.v << ' ' << ClampedPrimal(primal[column]) << '\n';
    }
    if (!output) {
      throw std::runtime_error("写入 subtour mincut oracle 输入失败");
    }
  }

  const std::string command = ShellQuote(executable.string()) + " " + ShellQuote(input.string());
  FILE* const process = ::popen(command.c_str(), "r");
  if (process == nullptr) {
    std::filesystem::remove(input);
    throw std::runtime_error("无法启动 Concorde mincut oracle");
  }
  std::size_t added = 0U;
  char* line = nullptr;
  std::size_t line_capacity = 0U;
  try {
    while (::getline(&line, &line_capacity, process) >= 0) {
      std::istringstream stream(line);
      std::string tag;
      std::int32_t count = 0;
      stream >> tag >> count;
      if (tag != "CUT") {
        continue;
      }
      if (count <= 1 || count + 1 >= graph.dimension) {
        throw std::runtime_error("Concorde mincut oracle 返回非法割规模");
      }
      std::vector<std::int32_t> side(static_cast<std::size_t>(count));
      for (std::int32_t index = 0; index < count; ++index) {
        if (!(stream >> side[static_cast<std::size_t>(index)]) ||
            side[static_cast<std::size_t>(index)] < 0 ||
            side[static_cast<std::size_t>(index)] >= graph.dimension) {
          throw std::runtime_error("Concorde mincut oracle 返回截断或越界节点");
        }
      }
      if (pool->Add(std::move(side), graph.dimension)) {
        ++added;
      }
    }
  } catch (...) {
    std::free(line);
    static_cast<void>(::pclose(process));
    std::filesystem::remove(input);
    throw;
  }
  std::free(line);
  const int status = ::pclose(process);
  std::filesystem::remove(input);
  if (status != 0) {
    throw std::runtime_error("Concorde mincut oracle 非零退出: " + std::to_string(status));
  }
  return added;
}

DegreeModel BuildDegreeModel(const GraphSnapshot& graph) {
  DegreeModel model;
  std::vector<std::int32_t> edge_column(graph.edges.size(), -1);
  model.column_edge_id.reserve(graph.ActiveEdgeCount());
  for (std::size_t edge_id = 0U; edge_id < graph.edges.size(); ++edge_id) {
    if (graph.edges[edge_id].active) {
      if (model.column_edge_id.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("PDLP degree relaxation 列数超过 int32");
      }
      edge_column[edge_id] = static_cast<std::int32_t>(model.column_edge_id.size());
      model.column_edge_id.push_back(static_cast<std::int32_t>(edge_id));
    }
  }

  LpEpoch& epoch = model.epoch;
  epoch.rows = graph.dimension;
  epoch.columns = static_cast<std::int32_t>(model.column_edge_id.size());
  epoch.objective_sense = 1;
  epoch.objective_offset = 0.0;
  epoch.objective.reserve(model.column_edge_id.size());
  epoch.lower_bounds.assign(model.column_edge_id.size(), 0.0);
  epoch.upper_bounds.assign(model.column_edge_id.size(), 1.0);
  epoch.variable_types.assign(model.column_edge_id.size(), 'C');
  epoch.edge_u.reserve(model.column_edge_id.size());
  epoch.edge_v.reserve(model.column_edge_id.size());
  for (const std::int32_t edge_id : model.column_edge_id) {
    const Edge& edge = graph.edges[static_cast<std::size_t>(edge_id)];
    epoch.objective.push_back(static_cast<double>(edge.weight));
    epoch.edge_u.push_back(edge.u);
    epoch.edge_v.push_back(edge.v);
  }

  epoch.row_offsets.reserve(static_cast<std::size_t>(graph.dimension) + 1U);
  epoch.row_offsets.push_back(0);
  epoch.senses.assign(static_cast<std::size_t>(graph.dimension), 'E');
  epoch.rhs.assign(static_cast<std::size_t>(graph.dimension), 2.0);
  epoch.column_indices.reserve(2U * model.column_edge_id.size());
  epoch.values.reserve(2U * model.column_edge_id.size());
  for (std::int32_t vertex = 0; vertex < graph.dimension; ++vertex) {
    for (std::int32_t offset = graph.row_offsets[static_cast<std::size_t>(vertex)];
         offset < graph.row_offsets[static_cast<std::size_t>(vertex) + 1U]; ++offset) {
      const std::int32_t edge_id = graph.csr_edge_ids[static_cast<std::size_t>(offset)];
      const std::int32_t column = edge_column[static_cast<std::size_t>(edge_id)];
      if (column < 0) {
        throw std::logic_error("PDLP degree relaxation CSR 引用了非活动边");
      }
      epoch.column_indices.push_back(column);
      epoch.values.push_back(1.0);
    }
    if (epoch.column_indices.size() >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      throw std::overflow_error("PDLP degree relaxation 非零元超过 int32");
    }
    epoch.row_offsets.push_back(static_cast<std::int32_t>(epoch.column_indices.size()));
  }
  epoch.Validate();
  epoch.content_hash = epoch.ComputeHash();
  return model;
}

DegreeModel BuildSubtourModel(const GraphSnapshot& graph, const SubtourCutPool& pool) {
  DegreeModel model = BuildDegreeModel(graph);
  LpEpoch& epoch = model.epoch;
  std::vector<std::uint8_t> membership(static_cast<std::size_t>(graph.dimension), 0U);
  for (const std::vector<std::int32_t>& cut : pool.cuts()) {
    for (const std::int32_t node : cut) {
      membership[static_cast<std::size_t>(node)] = 1U;
    }
    for (std::size_t column = 0U; column < model.column_edge_id.size(); ++column) {
      const Edge& edge = graph.edges[static_cast<std::size_t>(model.column_edge_id[column])];
      if (membership[static_cast<std::size_t>(edge.u)] !=
          membership[static_cast<std::size_t>(edge.v)]) {
        if (column > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
          throw std::overflow_error("subtour LP 列索引超过 int32");
        }
        epoch.column_indices.push_back(static_cast<std::int32_t>(column));
        epoch.values.push_back(1.0);
      }
    }
    for (const std::int32_t node : cut) {
      membership[static_cast<std::size_t>(node)] = 0U;
    }
    if (epoch.column_indices.size() >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      throw std::overflow_error("subtour LP 非零元超过 int32");
    }
    epoch.row_offsets.push_back(static_cast<std::int32_t>(epoch.column_indices.size()));
    epoch.senses.push_back('G');
    epoch.rhs.push_back(2.0);
  }
  if (pool.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() - graph.dimension)) {
    throw std::overflow_error("subtour LP 行数超过 int32");
  }
  epoch.rows = graph.dimension + static_cast<std::int32_t>(pool.size());
  epoch.Validate();
  epoch.content_hash = epoch.ComputeHash();
  return model;
}

double ClampedPrimal(const double value) { return std::max(0.0, std::min(1.0, value)); }

long double CutCapacity(const GraphSnapshot& graph, const DegreeModel& model,
                        const std::vector<double>& primal, const std::vector<std::int32_t>& side,
                        std::vector<std::uint8_t>* const membership) {
  std::fill(membership->begin(), membership->end(), 0U);
  for (const std::int32_t node : side) {
    (*membership)[static_cast<std::size_t>(node)] = 1U;
  }
  long double capacity = 0.0L;
  for (std::size_t column = 0U; column < model.column_edge_id.size(); ++column) {
    const Edge& edge = graph.edges[static_cast<std::size_t>(model.column_edge_id[column])];
    if ((*membership)[static_cast<std::size_t>(edge.u)] !=
        (*membership)[static_cast<std::size_t>(edge.v)]) {
      capacity += ClampedPrimal(primal[column]);
    }
  }
  return capacity;
}

struct SeparationResult {
  std::size_t added{};
  std::size_t support_components{};
};

SeparationResult SeparateSubtourCuts(const GraphSnapshot& graph, const DegreeModel& model,
                                     const std::vector<double>& primal,
                                     const SubtourPdlpOptions& options,
                                     const std::uint32_t separation_epoch,
                                     SubtourCutPool* const pool) {
  if (primal.size() != model.column_edge_id.size() || pool == nullptr) {
    throw std::invalid_argument("subtour separation 的 primal 或 cut pool 非法");
  }
  SeparationResult result;
  std::vector<std::uint8_t> membership(static_cast<std::size_t>(graph.dimension), 0U);
  const std::array<double, 8> thresholds = {
      options.support_epsilon, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90};

  for (std::size_t threshold_index = 0U; threshold_index < thresholds.size(); ++threshold_index) {
    const double threshold = thresholds[threshold_index];
    DisjointSet components(graph.dimension);
    for (std::size_t column = 0U; column < model.column_edge_id.size(); ++column) {
      if (primal[column] <= threshold) {
        continue;
      }
      const Edge& edge = graph.edges[static_cast<std::size_t>(model.column_edge_id[column])];
      components.Unite(edge.u, edge.v);
    }
    std::map<std::int32_t, std::vector<std::int32_t>> sides;
    for (std::int32_t node = 0; node < graph.dimension; ++node) {
      sides[components.Find(node)].push_back(node);
    }
    if (threshold_index == 0U) {
      result.support_components = sides.size();
    }
    for (auto& [unused_root, side] : sides) {
      static_cast<void>(unused_root);
      std::vector<std::int32_t> canonical = CanonicalCut(side, graph.dimension);
      if (canonical.empty()) {
        continue;
      }
      const long double capacity = CutCapacity(graph, model, primal, canonical, &membership);
      if (capacity < 2.0L - static_cast<long double>(options.violation_epsilon) &&
          pool->Add(std::move(canonical), graph.dimension)) {
        ++result.added;
      }
    }
  }

  // 连通 support 上再检查最大生成树的全部 fundamental cuts。树上差分一次计算
  // n-1 个割的真实 LP 容量，可捕获仅靠固定阈值连通分量看不到的窄割。
  if (result.support_components != 1U) {
    result.added += AddConcordeMincutOracleCuts(graph, model, primal, options.mincut_oracle,
                                                separation_epoch, pool);
    return result;
  }
  std::vector<std::int32_t> columns(model.column_edge_id.size());
  std::iota(columns.begin(), columns.end(), 0);
  std::sort(columns.begin(), columns.end(), [&](const std::int32_t lhs, const std::int32_t rhs) {
    const double lhs_value = ClampedPrimal(primal[static_cast<std::size_t>(lhs)]);
    const double rhs_value = ClampedPrimal(primal[static_cast<std::size_t>(rhs)]);
    return lhs_value > rhs_value || (lhs_value == rhs_value && lhs < rhs);
  });

  DisjointSet tree_components(graph.dimension);
  std::vector<std::vector<std::int32_t>> tree(static_cast<std::size_t>(graph.dimension));
  std::size_t tree_edges = 0U;
  for (const std::int32_t column : columns) {
    if (primal[static_cast<std::size_t>(column)] <= options.support_epsilon) {
      break;
    }
    const Edge& edge = graph.edges[static_cast<std::size_t>(
        model.column_edge_id[static_cast<std::size_t>(column)])];
    if (tree_components.Find(edge.u) == tree_components.Find(edge.v)) {
      continue;
    }
    tree_components.Unite(edge.u, edge.v);
    tree[static_cast<std::size_t>(edge.u)].push_back(edge.v);
    tree[static_cast<std::size_t>(edge.v)].push_back(edge.u);
    if (++tree_edges + 1U == static_cast<std::size_t>(graph.dimension)) {
      break;
    }
  }
  if (tree_edges + 1U != static_cast<std::size_t>(graph.dimension)) {
    return result;
  }

  std::vector<std::int32_t> parent(static_cast<std::size_t>(graph.dimension), -1);
  std::vector<std::int32_t> depth(static_cast<std::size_t>(graph.dimension), 0);
  std::vector<std::int32_t> order;
  order.reserve(static_cast<std::size_t>(graph.dimension));
  parent[0] = 0;
  order.push_back(0);
  for (std::size_t cursor = 0U; cursor < order.size(); ++cursor) {
    const std::int32_t node = order[cursor];
    for (const std::int32_t child : tree[static_cast<std::size_t>(node)]) {
      if (parent[static_cast<std::size_t>(child)] >= 0) {
        continue;
      }
      parent[static_cast<std::size_t>(child)] = node;
      depth[static_cast<std::size_t>(child)] = depth[static_cast<std::size_t>(node)] + 1;
      order.push_back(child);
    }
  }
  if (order.size() != static_cast<std::size_t>(graph.dimension)) {
    throw std::logic_error("subtour maximum spanning tree 遍历不完整");
  }
  std::int32_t levels = 1;
  while ((std::int64_t{1} << levels) <= graph.dimension) {
    ++levels;
  }
  std::vector<std::vector<std::int32_t>> ancestor(
      static_cast<std::size_t>(levels),
      std::vector<std::int32_t>(static_cast<std::size_t>(graph.dimension)));
  ancestor[0] = parent;
  for (std::int32_t level = 1; level < levels; ++level) {
    for (std::int32_t node = 0; node < graph.dimension; ++node) {
      ancestor[static_cast<std::size_t>(level)][static_cast<std::size_t>(node)] =
          ancestor[static_cast<std::size_t>(level - 1U)][static_cast<std::size_t>(
              ancestor[static_cast<std::size_t>(level - 1U)][static_cast<std::size_t>(node)])];
    }
  }
  const auto lowest_common_ancestor = [&](std::int32_t first, std::int32_t second) {
    if (depth[static_cast<std::size_t>(first)] < depth[static_cast<std::size_t>(second)]) {
      std::swap(first, second);
    }
    std::int32_t difference =
        depth[static_cast<std::size_t>(first)] - depth[static_cast<std::size_t>(second)];
    for (std::int32_t level = 0; difference != 0; ++level, difference >>= 1) {
      if ((difference & 1) != 0) {
        first = ancestor[static_cast<std::size_t>(level)][static_cast<std::size_t>(first)];
      }
    }
    if (first == second) {
      return first;
    }
    for (std::int32_t level = levels - 1; level >= 0; --level) {
      const std::int32_t first_parent =
          ancestor[static_cast<std::size_t>(level)][static_cast<std::size_t>(first)];
      const std::int32_t second_parent =
          ancestor[static_cast<std::size_t>(level)][static_cast<std::size_t>(second)];
      if (first_parent != second_parent) {
        first = first_parent;
        second = second_parent;
      }
    }
    return parent[static_cast<std::size_t>(first)];
  };

  std::vector<long double> difference(static_cast<std::size_t>(graph.dimension), 0.0L);
  for (std::size_t column = 0U; column < model.column_edge_id.size(); ++column) {
    const long double value = ClampedPrimal(primal[column]);
    if (value == 0.0L) {
      continue;
    }
    const Edge& edge = graph.edges[static_cast<std::size_t>(model.column_edge_id[column])];
    const std::int32_t lca = lowest_common_ancestor(edge.u, edge.v);
    difference[static_cast<std::size_t>(edge.u)] += value;
    difference[static_cast<std::size_t>(edge.v)] += value;
    difference[static_cast<std::size_t>(lca)] -= 2.0L * value;
  }
  for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator) {
    const std::int32_t node = *iterator;
    if (node != 0) {
      difference[static_cast<std::size_t>(parent[static_cast<std::size_t>(node)])] +=
          difference[static_cast<std::size_t>(node)];
    }
  }

  std::vector<std::vector<std::int32_t>> children(static_cast<std::size_t>(graph.dimension));
  for (std::int32_t node = 1; node < graph.dimension; ++node) {
    children[static_cast<std::size_t>(parent[static_cast<std::size_t>(node)])].push_back(node);
  }
  std::vector<std::int32_t> euler;
  euler.reserve(static_cast<std::size_t>(graph.dimension));
  std::vector<std::int32_t> entry(static_cast<std::size_t>(graph.dimension));
  std::vector<std::int32_t> exit(static_cast<std::size_t>(graph.dimension));
  std::vector<std::pair<std::int32_t, std::size_t>> stack{{0, 0U}};
  entry[0] = 0;
  euler.push_back(0);
  while (!stack.empty()) {
    auto& [node, child_index] = stack.back();
    if (child_index < children[static_cast<std::size_t>(node)].size()) {
      const std::int32_t child = children[static_cast<std::size_t>(node)][child_index++];
      entry[static_cast<std::size_t>(child)] = static_cast<std::int32_t>(euler.size());
      euler.push_back(child);
      stack.emplace_back(child, 0U);
    } else {
      exit[static_cast<std::size_t>(node)] = static_cast<std::int32_t>(euler.size());
      stack.pop_back();
    }
  }
  for (std::int32_t node = 1; node < graph.dimension; ++node) {
    if (difference[static_cast<std::size_t>(node)] >=
        2.0L - static_cast<long double>(options.violation_epsilon)) {
      continue;
    }
    std::vector<std::int32_t> side(euler.begin() + entry[static_cast<std::size_t>(node)],
                                   euler.begin() + exit[static_cast<std::size_t>(node)]);
    if (pool->Add(std::move(side), graph.dimension)) {
      ++result.added;
    }
  }
  result.added += AddConcordeMincutOracleCuts(graph, model, primal, options.mincut_oracle,
                                              separation_epoch, pool);
  return result;
}

std::vector<std::int64_t> QuantizeDual(const std::vector<double>& dual,
                                       const std::uint32_t fractional_bits) {
  if (fractional_bits > 40U) {
    throw std::invalid_argument("PDLP dual 定点位数非法");
  }
  const std::int64_t denominator = std::int64_t{1} << fractional_bits;
  std::vector<std::int64_t> quantized(dual.size());
  for (std::size_t index = 0; index < dual.size(); ++index) {
    const long double scaled = static_cast<long double>(dual[index]) * denominator;
    if (!std::isfinite(dual[index]) || std::abs(scaled) > 1.0e15L) {
      throw std::runtime_error("PDLP dual 无法安全量化");
    }
    quantized[index] = static_cast<std::int64_t>(std::round(scaled));
  }
  return quantized;
}

void FillScores(const GraphSnapshot& graph, const DegreeModel& model,
                const std::vector<std::int64_t>& quantized, const std::uint32_t fractional_bits,
                std::vector<double>* const scores) {
  if (quantized.size() != static_cast<std::size_t>(graph.dimension) || fractional_bits > 40U) {
    throw std::invalid_argument("PDLP score 的 dual 维度或定点位数非法");
  }
  const std::int64_t denominator = std::int64_t{1} << fractional_bits;
  scores->assign(graph.edges.size(), -std::numeric_limits<double>::infinity());
  for (std::size_t column = 0; column < model.column_edge_id.size(); ++column) {
    const std::int32_t edge_id = model.column_edge_id[column];
    const Edge& edge = graph.edges[static_cast<std::size_t>(edge_id)];
    const __int128 reduced = static_cast<__int128>(edge.weight) * denominator -
                             quantized[static_cast<std::size_t>(edge.u)] -
                             quantized[static_cast<std::size_t>(edge.v)];
    (*scores)[static_cast<std::size_t>(edge_id)] =
        static_cast<double>(static_cast<long double>(reduced) / denominator);
  }
}

} // namespace

PdlpResult RunFgpuPdlp(const GraphSnapshot& graph, const PdlpOptions& options) {
  PdlpResult result;
  result.backend = ToString(options.backend);
  result.edge_scores.assign(graph.edges.size(), -std::numeric_limits<double>::infinity());
  if (options.backend == PdlpBackend::kOff) {
    return result;
  }
  if (options.iterations == 0U || options.fractional_bits > 40U) {
    throw std::invalid_argument("PDLP iterations 必须大于 0，fractional_bits 不得超过 40");
  }
  const DegreeModel model = BuildDegreeModel(graph);
  const auto begin = std::chrono::steady_clock::now();
  std::vector<double> dual;
  if (options.backend == PdlpBackend::kNative) {
    std::string reason;
    if (!detail::NativePdlpCudaAvailable(&reason)) {
      throw std::runtime_error("native PDLP CUDA 后端不可用: " + reason);
    }
    const detail::NativePdlpDeviceResult device = detail::SolveDegreeRelaxationCuda(graph, options);
    dual = device.vertex_dual;
    result.selected_device = device.selected_device;
    result.iterations = device.iterations;
    result.solve_ms = device.solve_ms;
    result.backend = "native-cuda-degree-subgradient-" + device.implementation;
  } else {
    CuOptSession session(options.cuopt_library);
    const LpSolution solution = session.Solve(model.epoch, false);
    dual = solution.dual;
    result.solve_ms = solution.solve_time_seconds * 1000.0;
    result.iterations = 0U;
    result.backend = "cuopt-baseline-degree-lp";
  }
  if (result.solve_ms == 0.0) {
    result.solve_ms = ElapsedMilliseconds(begin);
  }
  result.exact_bound = BuildExactModelBound(model.epoch, dual, options.fractional_bits);
  result.cpu_certified = result.exact_bound.certified;
  if (!result.cpu_certified) {
    throw std::runtime_error("PDLP multiplier 未通过 __int128 box-Lagrangian 门禁: " +
                             result.exact_bound.reason);
  }
  result.fractional_bits = options.fractional_bits;
  result.vertex_dual_numerator = QuantizeDual(dual, options.fractional_bits);
  FillScores(graph, model, result.vertex_dual_numerator, options.fractional_bits,
             &result.edge_scores);
  return result;
}

SubtourPdlpResult RunFgpuSubtourPdlp(const GraphSnapshot& graph, const std::int64_t incumbent_cost,
                                     const SubtourPdlpOptions& options) {
  if (graph.dimension < 3 || graph.ActiveEdgeCount() == 0U || incumbent_cost < 0 ||
      options.support_epsilon <= 0.0 || options.support_epsilon >= 1.0 ||
      options.violation_epsilon <= 0.0 || options.violation_epsilon >= 1.0 ||
      options.fractional_bits > 40U) {
    throw std::invalid_argument("subtour PDLP 的图、incumbent、epsilon 或定点位数非法");
  }
  const auto total_begin = std::chrono::steady_clock::now();
  std::string affinity_reason;
  if (!detail::SetCudaDevicePreferenceForCurrentThread(options.device, &affinity_reason)) {
    throw std::runtime_error("subtour cuOpt 无法选择 CUDA device: " + affinity_reason);
  }
  SubtourPdlpResult result;
  result.edge_scores.assign(graph.edges.size(), -std::numeric_limits<double>::infinity());
  SubtourCutPool pool;
  CuOptSession session(options.cuopt_library);
  DegreeModel final_model;
  LpSolution final_solution;

  while (true) {
    DegreeModel model = BuildSubtourModel(graph, pool);
    LpSolution solution = session.Solve(model.epoch, true);
    ++result.epochs;
    result.solve_ms += solution.solve_time_seconds * 1000.0;
    if (!solution.numerically_accepted) {
      throw std::runtime_error("subtour cuOpt epoch 未通过数值门禁: status=" + solution.status);
    }
    const SeparationResult separation =
        SeparateSubtourCuts(graph, model, solution.primal, options, result.epochs, &pool);
    result.support_components = separation.support_components;
    if (separation.added == 0U) {
      result.converged = true;
      final_model = std::move(model);
      final_solution = std::move(solution);
      break;
    }
  }

  result.cuts = pool.size();
  result.objective = final_solution.objective;
  result.dual_objective = final_solution.dual_objective;
  const ExactModelEvaluation exact =
      BuildExactModelEvaluation(final_model.epoch, final_solution.dual, options.fractional_bits);
  if (!exact.bound.certified ||
      exact.reduced_cost_numerator.size() != final_model.column_edge_id.size()) {
    throw std::runtime_error("subtour LP 的量化 dual 未通过精确模型门禁: " + exact.bound.reason);
  }
  result.exact_bound = exact.bound;
  const __int128 incumbent_numerator =
      static_cast<__int128>(incumbent_cost) * exact.bound.denominator;
  for (std::size_t column = 0U; column < final_model.column_edge_id.size(); ++column) {
    const __int128 reduced = exact.reduced_cost_numerator[column];
    const std::int32_t edge_id = final_model.column_edge_id[column];
    result.edge_scores[static_cast<std::size_t>(edge_id)] =
        static_cast<double>(static_cast<long double>(reduced) / exact.bound.denominator);
    if (exact.lower_bound_numerator + std::max<__int128>(0, reduced) > incumbent_numerator) {
      ++result.forced_one_candidates;
    }
  }
  result.total_ms = ElapsedMilliseconds(total_begin);
  return result;
}

LpBoxVerificationData BuildLpBoxVerificationData(const GraphSnapshot& graph,
                                                 const LpBoxProof& proof) {
  LpBoxVerificationData result;
  if (proof.snapshot_hash != graph.ContentHash()) {
    result.reason = "LP box proof 的不可变快照哈希不匹配";
    return result;
  }
  if (proof.fractional_bits > 40U || proof.incumbent_cost < 0 ||
      proof.vertex_dual_numerator.size() != static_cast<std::size_t>(graph.dimension)) {
    result.reason = "LP box proof 的分母、incumbent 或 dual 维度非法";
    return result;
  }
  const __int128 denominator = static_cast<__int128>(1) << proof.fractional_bits;
  __int128 lower = 0;
  for (const std::int64_t dual : proof.vertex_dual_numerator) {
    __int128 term = 0;
    if (__builtin_mul_overflow(static_cast<__int128>(2), static_cast<__int128>(dual), &term) ||
        __builtin_add_overflow(lower, term, &lower)) {
      result.reason = "LP box proof 的 degree 常数项溢出";
      return result;
    }
  }
  result.reduced_cost_numerator.assign(graph.edges.size(), 0);
  for (std::size_t edge_id = 0U; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    if (!edge.active) {
      continue;
    }
    __int128 objective = 0;
    if (__builtin_mul_overflow(static_cast<__int128>(edge.weight), denominator, &objective)) {
      result.reason = "LP box proof 的目标系数量化溢出";
      return result;
    }
    __int128 reduced = 0;
    if (__builtin_sub_overflow(
            objective,
            static_cast<__int128>(proof.vertex_dual_numerator[static_cast<std::size_t>(edge.u)]),
            &reduced) ||
        __builtin_sub_overflow(
            reduced,
            static_cast<__int128>(proof.vertex_dual_numerator[static_cast<std::size_t>(edge.v)]),
            &reduced)) {
      result.reason = "LP box proof 的 reduced cost 累加溢出";
      return result;
    }
    result.reduced_cost_numerator[edge_id] = reduced;
    if (reduced < 0 && __builtin_add_overflow(lower, reduced, &lower)) {
      result.reason = "LP box proof 的 box contribution 溢出";
      return result;
    }
  }
  result.lower_bound_numerator = lower;
  result.certified = true;
  result.reason = "degree equality + complete live-variable box bound 已用 __int128 重放";
  return result;
}

bool detail::VerifyLpBoxCandidateForSnapshot(const GraphSnapshot& graph, const LpBoxProof& proof,
                                             const LpBoxVerificationData& verification_data,
                                             const Candidate& candidate,
                                             const std::uint64_t actual_snapshot_hash,
                                             std::string* const reason) {
  const auto fail = [&](const std::string& message) {
    if (reason != nullptr) {
      *reason = message;
    }
    return false;
  };
  if (!verification_data.certified || proof.snapshot_hash != actual_snapshot_hash) {
    return fail("LP box proof 的共享验证数据未认证或快照已变化");
  }
  if (candidate.method != EliminationMethod::kLpBox || candidate.witness != -1 ||
      candidate.second_witness != -1 || candidate.edge_id < 0 ||
      static_cast<std::size_t>(candidate.edge_id) >= graph.edges.size()) {
    return fail("LP box 候选类型、见证或 edge id 非法");
  }
  const Edge& edge = graph.edges[static_cast<std::size_t>(candidate.edge_id)];
  if (!edge.active || graph.Degree(edge.u) <= 2 || graph.Degree(edge.v) <= 2) {
    return fail("LP box 候选边不活动或违反最小度提交前提");
  }
  __int128 forced = verification_data.lower_bound_numerator;
  const __int128 reduced =
      verification_data.reduced_cost_numerator[static_cast<std::size_t>(candidate.edge_id)];
  if (reduced > 0 && __builtin_add_overflow(forced, reduced, &forced)) {
    return fail("LP forced-one 下界溢出");
  }
  __int128 incumbent = 0;
  const __int128 denominator = static_cast<__int128>(1) << proof.fractional_bits;
  if (__builtin_mul_overflow(static_cast<__int128>(proof.incumbent_cost), denominator,
                             &incumbent)) {
    return fail("LP incumbent 定点乘法溢出");
  }
  if (forced <= incumbent) {
    return fail("LP forced-one 下界未严格超过 incumbent");
  }
  return true;
}

bool VerifyLpBoxCandidate(const GraphSnapshot& graph, const LpBoxProof& proof,
                          const LpBoxVerificationData& verification_data,
                          const Candidate& candidate, std::string* const reason) {
  return detail::VerifyLpBoxCandidateForSnapshot(graph, proof, verification_data, candidate,
                                                 graph.ContentHash(), reason);
}

EliminationResult RunLpBoxElimination(GraphSnapshot* const graph, const PdlpResult& pdlp,
                                      const std::int64_t incumbent_cost) {
  if (graph == nullptr || incumbent_cost < 0 || !pdlp.cpu_certified ||
      pdlp.vertex_dual_numerator.size() != static_cast<std::size_t>(graph->dimension)) {
    throw std::invalid_argument("LP box 消元缺少图、incumbent 或已认证量化 dual");
  }
  EliminationResult result;
  result.backend = pdlp.backend + "+cpu-int128-forced-one";
  result.initial_hash = graph->ContentHash();
  LpBoxProof proof{result.initial_hash, pdlp.fractional_bits, incumbent_cost,
                   pdlp.vertex_dual_numerator};
  const LpBoxVerificationData verification = BuildLpBoxVerificationData(*graph, proof);
  if (!verification.certified) {
    throw std::runtime_error("LP box 共享 proof 验证失败: " + verification.reason);
  }
  std::vector<Candidate> candidates;
  candidates.reserve(graph->ActiveEdgeCount() / 4U);
  for (std::size_t edge_id = 0U; edge_id < graph->edges.size(); ++edge_id) {
    Candidate candidate{static_cast<std::int32_t>(edge_id), -1, EliminationMethod::kLpBox, -1};
    if (detail::VerifyLpBoxCandidateForSnapshot(*graph, proof, verification, candidate,
                                                result.initial_hash, nullptr)) {
      candidates.push_back(candidate);
    }
  }
  const std::size_t proposed = candidates.size();
  std::vector<Candidate> committed =
      detail::CommitVerifiedCandidates(graph, std::move(candidates), result.initial_hash);
  if (!committed.empty()) {
    result.lp_box_proofs.push_back(proof);
  }
  for (const Candidate& candidate : committed) {
    const Edge& edge = graph->edges[static_cast<std::size_t>(candidate.edge_id)];
    result.proof.push_back({0U, result.initial_hash, candidate.edge_id, edge.u, edge.v, -1,
                            EliminationMethod::kLpBox, 0U, -1});
  }
  result.epochs.push_back({.epoch = 0U,
                           .edges_before = graph->ActiveEdgeCount() + committed.size(),
                           .proposed = proposed,
                           .verified = proposed,
                           .rejected = 0U,
                           .committed = committed.size()});
  result.final_hash = graph->ContentHash();
  return result;
}

} // namespace cudaee
