#include "cuda_edge_elimination/graph.hpp"

#include "cuda_edge_elimination/distance.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>

#ifdef CUDAEE_HAVE_ZLIB
#include <zlib.h>
#endif

namespace cudaee {
namespace {

std::string Trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string ReadText(const std::filesystem::path& path) {
  if (path.extension() == ".gz") {
#ifdef CUDAEE_HAVE_ZLIB
    gzFile file = gzopen(path.c_str(), "rb");
    if (file == nullptr) {
      throw std::runtime_error("无法打开 gzip 文件: " + path.string());
    }
    std::string result;
    char buffer[64 * 1024];
    int count = 0;
    while ((count = gzread(file, buffer, sizeof(buffer))) > 0) {
      result.append(buffer, static_cast<std::size_t>(count));
    }
    const int status = gzclose(file);
    if (count < 0 || status != Z_OK) {
      throw std::runtime_error("读取 gzip 文件失败: " + path.string());
    }
    return result;
#else
    throw std::runtime_error("当前构建未启用 zlib，无法读取 .gz 文件");
#endif
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法打开输入文件: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    throw std::runtime_error("读取输入文件失败: " + path.string());
  }
  return contents.str();
}

void HashBytes(std::uint64_t* hash, const void* data, const std::size_t size) {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    *hash ^= bytes[i];
    *hash *= kPrime;
  }
}

template <typename T> void HashValue(std::uint64_t* hash, const T& value) {
  HashBytes(hash, &value, sizeof(value));
}

bool IsIntegralCoordinate(const double value, std::int64_t* converted) {
  if (!std::isfinite(value) ||
      value < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      value >= std::ldexp(1.0, 63)) {
    return false;
  }
  double integer_part = 0.0;
  if (std::modf(value, &integer_part) != 0.0) {
    return false;
  }
  *converted = static_cast<std::int64_t>(integer_part);
  return true;
}

} // namespace

namespace {

GraphSnapshot LoadTspCoordinates(const std::filesystem::path& tsp_path) {
  GraphSnapshot graph;
  const std::string tsp_text = ReadText(tsp_path);
  std::istringstream tsp(tsp_text);
  std::string line;
  bool in_coordinates = false;
  std::vector<bool> seen;
  graph.integer_coordinates = true;

  while (std::getline(tsp, line)) {
    line = Trim(line);
    if (line.empty()) {
      continue;
    }
    if (!in_coordinates) {
      if (line == "NODE_COORD_SECTION") {
        if (graph.dimension <= 0) {
          throw std::runtime_error("TSPLIB 在坐标区之前缺少有效 DIMENSION");
        }
        graph.points.resize(static_cast<std::size_t>(graph.dimension));
        seen.assign(static_cast<std::size_t>(graph.dimension), false);
        in_coordinates = true;
        continue;
      }
      const auto separator = line.find(':');
      if (separator == std::string::npos) {
        continue;
      }
      const std::string key = Trim(line.substr(0, separator));
      const std::string value = Trim(line.substr(separator + 1));
      if (key == "DIMENSION") {
        graph.dimension = std::stoi(value);
        if (graph.dimension <= 0 || graph.dimension == std::numeric_limits<std::int32_t>::max()) {
          throw std::runtime_error("DIMENSION 超出首期 CSR 索引范围");
        }
      } else if (key == "EDGE_WEIGHT_TYPE") {
        if (value == "EUC_2D") {
          graph.distance_type = DistanceType::kEuc2D;
        } else if (value == "CEIL_2D") {
          graph.distance_type = DistanceType::kCeil2D;
        } else {
          throw std::runtime_error("首期不支持 EDGE_WEIGHT_TYPE: " + value);
        }
      }
      continue;
    }

    if (line == "EOF") {
      break;
    }
    std::istringstream coordinate_line(line);
    std::int32_t id = 0;
    Point point;
    if (!(coordinate_line >> id >> point.x >> point.y)) {
      throw std::runtime_error("无法解析 TSPLIB 坐标行: " + line);
    }
    if (id < 1 || id > graph.dimension || seen[static_cast<std::size_t>(id - 1)]) {
      throw std::runtime_error("TSPLIB 节点编号越界或重复: " + std::to_string(id));
    }
    const bool x_integral = IsIntegralCoordinate(point.x, &point.integer_x);
    const bool y_integral = IsIntegralCoordinate(point.y, &point.integer_y);
    graph.integer_coordinates = graph.integer_coordinates && x_integral && y_integral;
    graph.points[static_cast<std::size_t>(id - 1)] = point;
    seen[static_cast<std::size_t>(id - 1)] = true;
  }

  if (!in_coordinates || graph.dimension <= 0 ||
      std::find(seen.begin(), seen.end(), false) != seen.end()) {
    throw std::runtime_error("TSPLIB 坐标不完整");
  }

  bool exact_half_coordinates = false;
  if (!graph.integer_coordinates) {
    exact_half_coordinates =
        std::all_of(graph.points.begin(), graph.points.end(), [](const Point& point) {
          std::int64_t numerator;
          return IsIntegralCoordinate(2.0 * point.x, &numerator) &&
                 IsIntegralCoordinate(2.0 * point.y, &numerator);
        });
    if (exact_half_coordinates) {
      graph.integer_coordinate_denominator = 2U;
      for (Point& point : graph.points) {
        point.integer_x = static_cast<std::int64_t>(2.0 * point.x);
        point.integer_y = static_cast<std::int64_t>(2.0 * point.y);
      }
    }
  }
  graph.integer_distance_safe = graph.integer_coordinates || exact_half_coordinates;
  if (graph.integer_distance_safe) {
    auto [min_x, max_x] = std::minmax_element(
        graph.points.begin(), graph.points.end(),
        [](const Point& lhs, const Point& rhs) { return lhs.integer_x < rhs.integer_x; });
    auto [min_y, max_y] = std::minmax_element(
        graph.points.begin(), graph.points.end(),
        [](const Point& lhs, const Point& rhs) { return lhs.integer_y < rhs.integer_y; });
    const __int128 dx = static_cast<__int128>(max_x->integer_x) - min_x->integer_x;
    const __int128 dy = static_cast<__int128>(max_y->integer_y) - min_y->integer_y;
    // 先界定各差值再平方，防止恶意极大坐标使 host 的 signed __int128 也溢出。
    const __int128 maximum_delta = std::numeric_limits<std::uint32_t>::max();
    graph.integer_distance_safe = dx <= maximum_delta && dy <= maximum_delta &&
                                  dx * dx + dy * dy <= std::numeric_limits<std::uint64_t>::max();
  }

  return graph;
}

} // namespace

GraphSnapshot GraphSnapshot::Load(const std::filesystem::path& tsp_path,
                                  const std::filesystem::path& edge_path) {
  GraphSnapshot graph = LoadTspCoordinates(tsp_path);

  const std::string edge_text = ReadText(edge_path);
  std::istringstream edge_input(edge_text);
  std::int32_t edge_dimension = 0;
  std::int64_t edge_count = 0;
  if (!(edge_input >> edge_dimension >> edge_count) || edge_dimension != graph.dimension ||
      edge_count < 0 || edge_count > std::numeric_limits<std::int32_t>::max()) {
    throw std::runtime_error("Concorde 边文件头无效或维度不匹配");
  }
  graph.edges.reserve(static_cast<std::size_t>(edge_count));
  std::unordered_set<std::uint64_t> unique_edges;
  for (std::int64_t index = 0; index < edge_count; ++index) {
    Edge edge;
    if (!(edge_input >> edge.u >> edge.v >> edge.weight)) {
      throw std::runtime_error("边文件在声明数量之前结束");
    }
    if (edge.u < 0 || edge.v < 0 || edge.u >= graph.dimension || edge.v >= graph.dimension ||
        edge.u == edge.v || edge.weight < 0) {
      throw std::runtime_error("边文件包含越界、自环或负权边");
    }
    if (edge.u > edge.v) {
      std::swap(edge.u, edge.v);
    }
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(edge.u)) << 32) |
        static_cast<std::uint32_t>(edge.v);
    if (!unique_edges.insert(key).second) {
      throw std::runtime_error("边文件包含重复无向边");
    }
    const std::int64_t expected = graph.Distance(edge.u, edge.v);
    if (edge.weight != expected) {
      throw std::runtime_error("边权与 TSPLIB 距离不一致: (" + std::to_string(edge.u) + "," +
                               std::to_string(edge.v) + ")");
    }
    graph.edges.push_back(edge);
  }
  std::string extra;
  if (edge_input >> extra) {
    throw std::runtime_error("边文件包含声明数量之外的数据");
  }

  std::sort(graph.edges.begin(), graph.edges.end(), [](const Edge& lhs, const Edge& rhs) {
    return std::tie(lhs.u, lhs.v) < std::tie(rhs.u, rhs.v);
  });
  graph.RebuildCsr();
  return graph;
}

GraphSnapshot GraphSnapshot::LoadCoordinates(const std::filesystem::path& tsp_path) {
  return LoadTspCoordinates(tsp_path);
}

GraphSnapshot GraphSnapshot::LoadComplete(const std::filesystem::path& tsp_path) {
  GraphSnapshot graph = LoadTspCoordinates(tsp_path);
  const std::int64_t dimension = graph.dimension;
  const std::int64_t edge_count = dimension * (dimension - 1) / 2;
  if (edge_count > std::numeric_limits<std::int32_t>::max()) {
    throw std::runtime_error("完全图边数超出首期 32 位边索引范围");
  }

  graph.edges.reserve(static_cast<std::size_t>(edge_count));
  for (std::int32_t u = 0; u < graph.dimension; ++u) {
    for (std::int32_t v = u + 1; v < graph.dimension; ++v) {
      graph.edges.push_back(Edge{u, v, graph.Distance(u, v), true});
    }
  }
  graph.RebuildCsr();
  return graph;
}

void GraphSnapshot::RebuildCsr() {
  const bool canonical_sorted =
      std::all_of(edges.begin(), edges.end(), [](const Edge& edge) { return edge.u < edge.v; }) &&
      std::is_sorted(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
        return std::tie(lhs.u, lhs.v) < std::tie(rhs.u, rhs.v);
      });
  row_offsets.assign(static_cast<std::size_t>(dimension) + 1, 0);
  for (const Edge& edge : edges) {
    if (!edge.active) {
      continue;
    }
    ++row_offsets[static_cast<std::size_t>(edge.u) + 1];
    ++row_offsets[static_cast<std::size_t>(edge.v) + 1];
  }
  for (std::int32_t i = 1; i <= dimension; ++i) {
    row_offsets[static_cast<std::size_t>(i)] += row_offsets[static_cast<std::size_t>(i - 1)];
  }
  neighbors.assign(static_cast<std::size_t>(row_offsets.back()), -1);
  csr_edge_ids.assign(neighbors.size(), -1);
  csr_weights.assign(neighbors.size(), 0);
  std::vector<std::int32_t> cursor = row_offsets;
  for (std::size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
    const Edge& edge = edges[edge_id];
    if (!edge.active) {
      continue;
    }
    for (const auto [from, to] : {std::pair{edge.u, edge.v}, std::pair{edge.v, edge.u}}) {
      const auto slot = static_cast<std::size_t>(cursor[static_cast<std::size_t>(from)]++);
      neighbors[slot] = to;
      csr_edge_ids[slot] = static_cast<std::int32_t>(edge_id);
      csr_weights[slot] = edge.weight;
    }
  }

  // 规范边按 (u,v) 全局排序时，每个 CSR row 已自然按邻点排序；删边只取其子序列。
  // 手工构造的乱序快照仍走逐行排序回退，保持 HasActiveEdge 的二分查找前提。
  if (canonical_sorted) {
    return;
  }
  for (std::int32_t vertex = 0; vertex < dimension; ++vertex) {
    const auto begin = static_cast<std::size_t>(row_offsets[static_cast<std::size_t>(vertex)]);
    const auto end = static_cast<std::size_t>(row_offsets[static_cast<std::size_t>(vertex) + 1]);
    std::vector<std::tuple<std::int32_t, std::int32_t, std::int64_t>> row;
    row.reserve(end - begin);
    for (std::size_t i = begin; i < end; ++i) {
      row.emplace_back(neighbors[i], csr_edge_ids[i], csr_weights[i]);
    }
    std::sort(row.begin(), row.end());
    for (std::size_t i = 0; i < row.size(); ++i) {
      std::tie(neighbors[begin + i], csr_edge_ids[begin + i], csr_weights[begin + i]) = row[i];
    }
  }
}

void GraphSnapshot::WriteActiveEdges(const std::filesystem::path& path) const {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建边输出文件: " + path.string());
  }
  output << dimension << ' ' << ActiveEdgeCount() << '\n';
  for (const Edge& edge : edges) {
    if (edge.active) {
      output << edge.u << ' ' << edge.v << ' ' << edge.weight << '\n';
    }
  }
  if (!output) {
    throw std::runtime_error("写边输出文件失败: " + path.string());
  }
}

std::int64_t GraphSnapshot::Distance(const std::int32_t u, const std::int32_t v) const {
  if (u < 0 || v < 0 || u >= dimension || v >= dimension) {
    throw std::out_of_range("距离查询节点越界");
  }
  if (integer_coordinates) {
    std::int64_t result = 0;
    std::string error;
    if (!ExactIntegerDistance(points[static_cast<std::size_t>(u)],
                              points[static_cast<std::size_t>(v)], distance_type, &result,
                              &error)) {
      throw std::overflow_error(error);
    }
    return result;
  }
  return FloatingDistance(points[static_cast<std::size_t>(u)], points[static_cast<std::size_t>(v)],
                          distance_type);
}

std::size_t GraphSnapshot::ActiveEdgeCount() const {
  return static_cast<std::size_t>(
      std::count_if(edges.begin(), edges.end(), [](const Edge& edge) { return edge.active; }));
}

std::uint64_t GraphSnapshot::ContentHash() const {
  std::uint64_t hash = 14695981039346656037ULL;
  HashValue(&hash, dimension);
  HashValue(&hash, distance_type);
  HashValue(&hash, integer_coordinates);
  for (const Point& point : points) {
    if (integer_coordinates) {
      HashValue(&hash, point.integer_x);
      HashValue(&hash, point.integer_y);
    } else {
      const auto x_bits = std::bit_cast<std::uint64_t>(point.x);
      const auto y_bits = std::bit_cast<std::uint64_t>(point.y);
      HashValue(&hash, x_bits);
      HashValue(&hash, y_bits);
    }
  }
  for (const Edge& edge : edges) {
    HashValue(&hash, edge.u);
    HashValue(&hash, edge.v);
    HashValue(&hash, edge.weight);
    HashValue(&hash, edge.active);
  }
  return hash;
}

std::int32_t GraphSnapshot::Degree(const std::int32_t vertex) const {
  if (vertex < 0 || vertex >= dimension) {
    throw std::out_of_range("度数查询节点越界");
  }
  return row_offsets[static_cast<std::size_t>(vertex) + 1] -
         row_offsets[static_cast<std::size_t>(vertex)];
}

bool GraphSnapshot::HasActiveEdge(const std::int32_t u, const std::int32_t v) const {
  if (u < 0 || v < 0 || u >= dimension || v >= dimension) {
    return false;
  }
  const auto begin = neighbors.begin() + row_offsets[static_cast<std::size_t>(u)];
  const auto end = neighbors.begin() + row_offsets[static_cast<std::size_t>(u) + 1];
  return std::binary_search(begin, end, v);
}

std::string HexHash(const std::uint64_t value) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << value;
  return output.str();
}

} // namespace cudaee
