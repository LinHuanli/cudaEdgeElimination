#include "cuda_edge_elimination/tour.hpp"

#include "cuda_edge_elimination/types.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef CUDAEE_HAVE_ZLIB
#include <zlib.h>
#endif

namespace cudaee {
namespace {

constexpr std::size_t kMaxTourTextBytes = 64U * 1024U * 1024U;

std::string Trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1U);
}

std::string ReadTourText(const std::filesystem::path& path) {
  if (path.extension() == ".gz") {
#ifdef CUDAEE_HAVE_ZLIB
    gzFile file = gzopen(path.c_str(), "rb");
    if (file == nullptr) {
      throw std::runtime_error("无法打开 gzip tour 文件: " + path.string());
    }
    std::string result;
    char buffer[64U * 1024U];
    int count = 0;
    while ((count = gzread(file, buffer, sizeof(buffer))) > 0) {
      result.append(buffer, static_cast<std::size_t>(count));
      if (result.size() > kMaxTourTextBytes) {
        static_cast<void>(gzclose(file));
        throw std::runtime_error("tour 解压后超过 64 MiB 上限");
      }
    }
    const int status = gzclose(file);
    if (count < 0 || status != Z_OK) {
      throw std::runtime_error("读取 gzip tour 文件失败: " + path.string());
    }
    return result;
#else
    throw std::runtime_error("当前构建未启用 zlib，无法读取 gzip tour");
#endif
  }

  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (!error && size > kMaxTourTextBytes) {
    throw std::runtime_error("tour 文件超过 64 MiB 上限");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法打开 tour 文件: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  std::string result = contents.str();
  if ((!input.good() && !input.eof()) || result.size() > kMaxTourTextBytes) {
    throw std::runtime_error("读取 tour 文件失败或超过 64 MiB 上限: " + path.string());
  }
  return result;
}

template <typename Integer>
Integer ParseInteger(const std::string_view text, const std::string_view description) {
  Integer value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
    throw std::runtime_error(std::string(description) + " 不是合法十进制整数");
  }
  return value;
}

void HashBytes(std::uint64_t* const hash, const void* const data, const std::size_t size) {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  const auto* const bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index) {
    *hash ^= bytes[index];
    *hash *= kPrime;
  }
}

template <typename Value> void HashValue(std::uint64_t* const hash, const Value& value) {
  HashBytes(hash, &value, sizeof(value));
}

} // namespace

std::vector<std::int32_t> ReadTsplibTour(const std::filesystem::path& path,
                                         const std::int32_t expected_dimension) {
  if (expected_dimension <= 0) {
    throw std::invalid_argument("tour 的期望维度必须为正数");
  }
  std::istringstream input(ReadTourText(path));
  std::vector<std::int32_t> tour;
  tour.reserve(static_cast<std::size_t>(expected_dimension));
  std::int32_t declared_dimension = -1;
  bool in_tour = false;
  bool terminated = false;
  bool saw_eof = false;
  bool saw_type = false;
  std::string line;
  while (std::getline(input, line)) {
    line = Trim(line);
    if (line.empty()) {
      continue;
    }
    if (!in_tour) {
      if (line == "TOUR_SECTION") {
        in_tour = true;
        continue;
      }
      const auto separator = line.find(':');
      if (separator == std::string::npos) {
        if (line == "EOF") {
          break;
        }
        continue;
      }
      const std::string key = Trim(line.substr(0U, separator));
      const std::string value = Trim(line.substr(separator + 1U));
      if (key == "TYPE") {
        saw_type = true;
        if (value != "TOUR") {
          throw std::runtime_error("TSPLIB tour 的 TYPE 不是 TOUR");
        }
      } else if (key == "DIMENSION") {
        declared_dimension = ParseInteger<std::int32_t>(value, "tour DIMENSION");
      }
      continue;
    }

    if (terminated) {
      if (line != "EOF" || saw_eof) {
        throw std::runtime_error("tour 在 -1 后包含尾随数据");
      }
      saw_eof = true;
      continue;
    }
    std::istringstream line_input(line);
    std::string token;
    while (line_input >> token) {
      const std::int64_t node = ParseInteger<std::int64_t>(token, "tour 节点");
      if (node == -1) {
        if (line_input >> token) {
          throw std::runtime_error("tour 的 -1 终止符后同一行仍有数据");
        }
        terminated = true;
        break;
      }
      if (node < 1 || node > expected_dimension) {
        throw std::runtime_error("tour 节点编号越界: " + std::to_string(node));
      }
      tour.push_back(static_cast<std::int32_t>(node - 1));
      if (tour.size() > static_cast<std::size_t>(expected_dimension)) {
        throw std::runtime_error("tour 节点数量超过 DIMENSION");
      }
    }
  }

  if (!in_tour || !terminated || !saw_type) {
    throw std::runtime_error("tour 缺少 TYPE、TOUR_SECTION 或 -1 终止符");
  }
  if (declared_dimension != expected_dimension ||
      tour.size() != static_cast<std::size_t>(expected_dimension)) {
    throw std::runtime_error("tour DIMENSION 或节点数量与图不一致");
  }
  std::vector<bool> seen(static_cast<std::size_t>(expected_dimension), false);
  for (const std::int32_t node : tour) {
    if (seen[static_cast<std::size_t>(node)]) {
      throw std::runtime_error("tour 包含重复节点: " + std::to_string(node + 1));
    }
    seen[static_cast<std::size_t>(node)] = true;
  }
  return tour;
}

ProtectedTourCheck CheckProtectedTour(const GraphSnapshot& graph,
                                      const std::vector<std::int32_t>& tour) {
  if (graph.dimension <= 0 || tour.size() != static_cast<std::size_t>(graph.dimension)) {
    throw std::invalid_argument("受保护 tour 的节点数量与图不一致");
  }
  std::vector<bool> seen(static_cast<std::size_t>(graph.dimension), false);
  std::vector<std::pair<std::int32_t, std::int32_t>> tour_edges;
  tour_edges.reserve(tour.size());
  ProtectedTourCheck result;
  for (std::size_t index = 0; index < tour.size(); ++index) {
    const std::int32_t from = tour[index];
    const std::int32_t to = tour[(index + 1U) % tour.size()];
    if (from < 0 || from >= graph.dimension || seen[static_cast<std::size_t>(from)]) {
      throw std::invalid_argument("受保护 tour 包含越界或重复节点");
    }
    seen[static_cast<std::size_t>(from)] = true;
    const std::int64_t distance = graph.Distance(from, to);
    if (distance < 0 || result.cost > std::numeric_limits<std::int64_t>::max() - distance) {
      throw std::overflow_error("受保护 tour 成本累加溢出");
    }
    result.cost += distance;
    const auto edge = std::minmax(from, to);
    tour_edges.push_back(edge);
    if (!graph.HasActiveEdge(edge.first, edge.second)) {
      ++result.missing_edges;
    }
  }

  // 对规范无向边集排序后哈希，使起点和巡回方向不影响身份绑定。
  std::sort(tour_edges.begin(), tour_edges.end());
  std::uint64_t hash = 1469598103934665603ULL;
  HashValue(&hash, graph.dimension);
  for (const auto [u, v] : tour_edges) {
    HashValue(&hash, u);
    HashValue(&hash, v);
  }
  result.tour_hash = hash;
  return result;
}

} // namespace cudaee
