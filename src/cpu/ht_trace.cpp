#include "cuda_edge_elimination/ht_trace.hpp"

#include "cuda_edge_elimination/types.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::size_t kMaxTraceCount = 1000000U;
constexpr std::size_t kMaxTraceNodes = 10000000U;

template <typename Integer>
Integer ParseInteger(const std::string& token, const char* const field) {
  Integer value{};
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value, 10);
  if (token.empty() || error != std::errc{} || end != token.data() + token.size()) {
    throw std::runtime_error(std::string("HT trace 的 ") + field + " 非法");
  }
  return value;
}

HtTraceNodeKind ParseKind(const std::string& value) {
  if (value == "leaf") {
    return HtTraceNodeKind::kLeaf;
  }
  if (value == "or") {
    return HtTraceNodeKind::kOr;
  }
  if (value == "and") {
    return HtTraceNodeKind::kAnd;
  }
  throw std::runtime_error("HT trace node kind 非法: " + value);
}

std::uint64_t CheckedAdd(const std::uint64_t first, const std::uint64_t second,
                         const char* const description) {
  if (second > std::numeric_limits<std::uint64_t>::max() - first) {
    throw std::overflow_error(std::string("HT trace replay ") + description + " 溢出");
  }
  return first + second;
}

struct ReplayWork {
  bool value{false};
  std::uint64_t nodes{};
  std::uint64_t work_units{};
  std::uint64_t short_circuits{};
  std::uint64_t peak_ready_width{};
};

ReplayWork ReplayNode(const HtShortCircuitTrace& trace, const std::uint32_t node_index,
                      const std::uint32_t speculation_width) {
  const HtShortCircuitTraceNode& node = trace.nodes.at(node_index);
  ReplayWork work{.value = node.value, .nodes = 1U, .work_units = node.work_units};
  if (node.kind == HtTraceNodeKind::kLeaf) {
    return work;
  }

  const bool decisive_value = node.kind == HtTraceNodeKind::kOr;
  const std::size_t width = speculation_width == 0U
                                ? std::max<std::size_t>(1U, node.children.size())
                                : static_cast<std::size_t>(speculation_width);
  for (std::size_t begin = 0U; begin < node.children.size(); begin += width) {
    const std::size_t end = begin + std::min(width, node.children.size() - begin);
    work.peak_ready_width =
        std::max(work.peak_ready_width, static_cast<std::uint64_t>(end - begin));
    bool window_decisive = false;
    for (std::size_t index = begin; index < end; ++index) {
      const ReplayWork child = ReplayNode(trace, node.children[index], speculation_width);
      work.nodes = CheckedAdd(work.nodes, child.nodes, "节点计数");
      work.work_units = CheckedAdd(work.work_units, child.work_units, "工作量");
      work.short_circuits = CheckedAdd(work.short_circuits, child.short_circuits, "短路计数");
      work.peak_ready_width = std::max(work.peak_ready_width, child.peak_ready_width);
      window_decisive = window_decisive || child.value == decisive_value;
    }
    if (window_decisive) {
      if (end < node.children.size()) {
        work.short_circuits = CheckedAdd(work.short_circuits, 1U, "短路计数");
      }
      break;
    }
  }
  return work;
}

void ValidateTrace(const HtShortCircuitTrace& trace) {
  if (!trace.complete) {
    throw std::invalid_argument("部分 HT trace 不能用于完整 replay");
  }
  if (trace.nodes.empty() || trace.nodes.size() > kMaxTraceNodes ||
      trace.root >= trace.nodes.size()) {
    throw std::invalid_argument("HT trace 根或节点数量非法");
  }
  std::vector<std::uint8_t> incoming(trace.nodes.size(), 0U);
  for (std::size_t index = 0U; index < trace.nodes.size(); ++index) {
    const HtShortCircuitTraceNode& node = trace.nodes[index];
    if (node.work_units == 0U) {
      throw std::invalid_argument("HT trace node work_units 必须大于 0");
    }
    if (node.kind == HtTraceNodeKind::kLeaf && !node.children.empty()) {
      throw std::invalid_argument("HT trace leaf 不得含 child");
    }
    bool aggregate = node.kind == HtTraceNodeKind::kAnd;
    for (std::size_t ordinal = 0U; ordinal < node.children.size(); ++ordinal) {
      const std::uint32_t child_index = node.children[ordinal];
      if (child_index >= trace.nodes.size() || child_index <= index ||
          trace.nodes[child_index].parent != index ||
          trace.nodes[child_index].child_ordinal != ordinal || ++incoming[child_index] != 1U) {
        throw std::invalid_argument("HT trace child 拓扑、parent 或 ordinal 非法");
      }
      if (node.kind == HtTraceNodeKind::kOr) {
        aggregate = aggregate || trace.nodes[child_index].value;
      } else if (node.kind == HtTraceNodeKind::kAnd) {
        aggregate = aggregate && trace.nodes[child_index].value;
      }
    }
    if (node.kind != HtTraceNodeKind::kLeaf && aggregate != node.value) {
      throw std::invalid_argument("HT trace connective 真值与 child 不一致");
    }
  }
  if (trace.nodes[trace.root].parent != kNoHtTraceNode || incoming[trace.root] != 0U) {
    throw std::invalid_argument("HT trace root parent 非法");
  }
  for (std::size_t index = 0U; index < trace.nodes.size(); ++index) {
    if (index != trace.root && incoming[index] != 1U) {
      throw std::invalid_argument("HT trace 含孤立或共享节点");
    }
  }
}

std::string ReadToken(std::istream& input, const char* const description) {
  std::string token;
  if (!(input >> token)) {
    throw std::runtime_error(std::string("HT trace 缺少 ") + description);
  }
  return token;
}

void Expect(std::istream& input, const std::string_view expected) {
  const std::string token = ReadToken(input, "字段");
  if (token != expected) {
    throw std::runtime_error("HT trace 期望字段 " + std::string(expected) + "，实际为 " + token);
  }
}

} // namespace

std::string ToString(const HtTraceNodeKind kind) {
  switch (kind) {
  case HtTraceNodeKind::kLeaf:
    return "leaf";
  case HtTraceNodeKind::kOr:
    return "or";
  case HtTraceNodeKind::kAnd:
    return "and";
  }
  throw std::invalid_argument("未知 HT trace node kind");
}

HtTraceReplayResult ReplayHtShortCircuitTrace(const HtShortCircuitTrace& trace,
                                              const std::uint32_t speculation_width) {
  ValidateTrace(trace);
  const ReplayWork replay = ReplayNode(trace, trace.root, speculation_width);
  const ReplayWork canonical = speculation_width == 1U ? replay : ReplayNode(trace, trace.root, 1U);
  if (replay.value != trace.nodes[trace.root].value || canonical.value != replay.value ||
      replay.nodes < canonical.nodes) {
    throw std::logic_error("HT trace replay 违反规范短路不变量");
  }
  return {.speculation_width = speculation_width,
          .value = replay.value,
          .scheduled_nodes = replay.nodes,
          .scheduled_work_units = replay.work_units,
          .canonical_nodes = canonical.nodes,
          .speculative_nodes = replay.nodes - canonical.nodes,
          .short_circuits = replay.short_circuits,
          .peak_ready_width = replay.peak_ready_width};
}

void WriteHtShortCircuitTraceBundle(const std::filesystem::path& path,
                                    const HtShortCircuitTraceBundle& bundle) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 HT trace: " + path.string());
  }
  output << "CUDAEE_HT_SHORT_CIRCUIT_TRACE_V1\n";
  output << "trace_count " << bundle.traces.size() << '\n';
  for (std::size_t trace_index = 0U; trace_index < bundle.traces.size(); ++trace_index) {
    const HtShortCircuitTrace& trace = bundle.traces[trace_index];
    output << "trace " << trace_index << '\n';
    output << "snapshot_hash " << trace.snapshot_hash << '\n';
    output << "target " << trace.target_edge.u << ' ' << trace.target_edge.v << '\n';
    output << "complete " << (trace.complete ? 1 : 0) << '\n';
    output << "root " << trace.root << '\n';
    output << "node_count " << trace.nodes.size() << '\n';
    for (std::size_t node_index = 0U; node_index < trace.nodes.size(); ++node_index) {
      const HtShortCircuitTraceNode& node = trace.nodes[node_index];
      output << "node " << node_index << ' ' << ToString(node.kind) << ' ' << node.parent << ' '
             << node.child_ordinal << ' ' << node.depth << ' ' << (node.value ? 1 : 0) << ' '
             << node.work_units << ' ' << node.children.size();
      for (const std::uint32_t child : node.children) {
        output << ' ' << child;
      }
      output << '\n';
    }
    output << "end_trace\n";
  }
  output << "END\n";
  if (!output) {
    throw std::runtime_error("写入 HT trace 失败: " + path.string());
  }
}

HtShortCircuitTraceBundle ReadHtShortCircuitTraceBundle(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法读取 HT trace: " + path.string());
  }
  Expect(input, "CUDAEE_HT_SHORT_CIRCUIT_TRACE_V1");
  Expect(input, "trace_count");
  const std::size_t trace_count =
      ParseInteger<std::size_t>(ReadToken(input, "trace_count"), "trace_count");
  if (trace_count > kMaxTraceCount) {
    throw std::runtime_error("HT trace_count 超过安全上限");
  }
  HtShortCircuitTraceBundle bundle;
  bundle.traces.resize(trace_count);
  for (std::size_t trace_index = 0U; trace_index < trace_count; ++trace_index) {
    Expect(input, "trace");
    if (ParseInteger<std::size_t>(ReadToken(input, "trace index"), "trace index") != trace_index) {
      throw std::runtime_error("HT trace index 不连续");
    }
    HtShortCircuitTrace& trace = bundle.traces[trace_index];
    Expect(input, "snapshot_hash");
    trace.snapshot_hash =
        ParseInteger<std::uint64_t>(ReadToken(input, "snapshot_hash"), "snapshot_hash");
    Expect(input, "target");
    trace.target_edge.u = ParseInteger<std::int32_t>(ReadToken(input, "target u"), "target u");
    trace.target_edge.v = ParseInteger<std::int32_t>(ReadToken(input, "target v"), "target v");
    Expect(input, "complete");
    const std::uint32_t complete =
        ParseInteger<std::uint32_t>(ReadToken(input, "complete"), "complete");
    if (complete > 1U) {
      throw std::runtime_error("HT trace complete 必须为 0 或 1");
    }
    trace.complete = complete == 1U;
    Expect(input, "root");
    trace.root = ParseInteger<std::uint32_t>(ReadToken(input, "root"), "root");
    Expect(input, "node_count");
    const std::size_t node_count =
        ParseInteger<std::size_t>(ReadToken(input, "node_count"), "node_count");
    if (node_count > kMaxTraceNodes) {
      throw std::runtime_error("HT trace node_count 超过安全上限");
    }
    trace.nodes.resize(node_count);
    for (std::size_t node_index = 0U; node_index < node_count; ++node_index) {
      Expect(input, "node");
      if (ParseInteger<std::size_t>(ReadToken(input, "node index"), "node index") != node_index) {
        throw std::runtime_error("HT trace node index 不连续");
      }
      HtShortCircuitTraceNode& node = trace.nodes[node_index];
      node.kind = ParseKind(ReadToken(input, "node kind"));
      node.parent = ParseInteger<std::uint32_t>(ReadToken(input, "parent"), "parent");
      node.child_ordinal =
          ParseInteger<std::uint32_t>(ReadToken(input, "child ordinal"), "child ordinal");
      node.depth = ParseInteger<std::uint32_t>(ReadToken(input, "depth"), "depth");
      const std::uint32_t value = ParseInteger<std::uint32_t>(ReadToken(input, "value"), "value");
      if (value > 1U) {
        throw std::runtime_error("HT trace value 必须为 0 或 1");
      }
      node.value = value == 1U;
      node.work_units = ParseInteger<std::uint64_t>(ReadToken(input, "work units"), "work units");
      const std::size_t child_count =
          ParseInteger<std::size_t>(ReadToken(input, "child count"), "child count");
      if (child_count > node_count) {
        throw std::runtime_error("HT trace child_count 非法");
      }
      node.children.resize(child_count);
      for (std::uint32_t& child : node.children) {
        child = ParseInteger<std::uint32_t>(ReadToken(input, "child"), "child");
      }
    }
    Expect(input, "end_trace");
    if (trace.complete) {
      ValidateTrace(trace);
    }
  }
  Expect(input, "END");
  std::string trailing;
  if (input >> trailing) {
    throw std::runtime_error("HT trace END 后含多余数据");
  }
  return bundle;
}

} // namespace cudaee
