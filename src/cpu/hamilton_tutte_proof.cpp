#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace cudaee {
namespace {

constexpr std::size_t kMaxHtProofBytes = 256U * 1024U * 1024U;
constexpr std::size_t kMaxHtNodes = 1000000U;
constexpr std::size_t kMaxHtReplies = 1000000U;
constexpr std::size_t kMaxHtPaths = 100000U;
constexpr std::size_t kMaxHtPathNodes = 10000000U;

void ExpectToken(std::istream* const input, const std::string_view expected) {
  std::string token;
  if (!(*input >> token) || token != expected) {
    throw std::runtime_error("recursive HT proof 缺少字段: " + std::string(expected));
  }
}

template <typename Integer>
Integer ReadInteger(std::istream* const input, const std::string_view field) {
  static_assert(std::is_integral_v<Integer>);
  std::string token;
  if (!(*input >> token) || token.empty()) {
    throw std::runtime_error("recursive HT proof 的 " + std::string(field) + " 非法");
  }
  Integer value{};
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value, 10);
  if (error != std::errc{} || end != token.data() + token.size()) {
    throw std::runtime_error("recursive HT proof 的 " + std::string(field) + " 非法");
  }
  return value;
}

std::uint64_t ReadHexHash(std::istream* const input, const std::string_view field) {
  std::string token;
  if (!(*input >> token) || token.size() != 16U) {
    throw std::runtime_error("recursive HT proof 的 " + std::string(field) + " 非法");
  }
  std::uint64_t value{};
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value, 16);
  if (error != std::errc{} || end != token.data() + token.size()) {
    throw std::runtime_error("recursive HT proof 的 " + std::string(field) + " 非法");
  }
  return value;
}

std::string_view CdModeToken(const HtCdMode mode) {
  switch (mode) {
  case HtCdMode::kActiveIncompatible:
    return "active_incompatible";
  case HtCdMode::kMissingOrIncompatible:
    return "missing_or_incompatible";
  }
  throw std::invalid_argument("recursive HT proof 的 c,d mode 非法");
}

HtCdMode ParseCdMode(std::istream* const input) {
  std::string token;
  if (!(*input >> token)) {
    throw std::runtime_error("recursive HT proof 的 c,d mode 非法");
  }
  if (token == "active_incompatible") {
    return HtCdMode::kActiveIncompatible;
  }
  if (token == "missing_or_incompatible") {
    return HtCdMode::kMissingOrIncompatible;
  }
  throw std::runtime_error("recursive HT proof 的 c,d mode 非法");
}

std::string_view MoveTypeToken(const HtMoveType type) {
  switch (type) {
  case HtMoveType::kLeaf:
    return "leaf";
  case HtMoveType::kCd:
    return "cd";
  case HtMoveType::kPoint:
    return "point";
  case HtMoveType::kEnd:
    return "end";
  }
  throw std::invalid_argument("recursive HT proof 的 move type 非法");
}

HtMoveType ParseMoveType(std::istream* const input) {
  std::string token;
  if (!(*input >> token)) {
    throw std::runtime_error("recursive HT proof 的 move type 非法");
  }
  if (token == "leaf") {
    return HtMoveType::kLeaf;
  }
  if (token == "cd") {
    return HtMoveType::kCd;
  }
  if (token == "point") {
    return HtMoveType::kPoint;
  }
  if (token == "end") {
    return HtMoveType::kEnd;
  }
  throw std::runtime_error("recursive HT proof 的 move type 非法");
}

std::string HexEncode(const std::string_view input) {
  constexpr std::string_view digits = "0123456789abcdef";
  if (input.size() > kMaxHtProofBytes / 2U) {
    throw std::invalid_argument("recursive HT leaf proof 过大");
  }
  std::string output(2U * input.size(), '0');
  for (std::size_t index = 0; index < input.size(); ++index) {
    const auto byte = static_cast<unsigned char>(input[index]);
    output[2U * index] = digits[byte >> 4U];
    output[2U * index + 1U] = digits[byte & 0x0fU];
  }
  return output;
}

std::uint8_t HexNibble(const char character) {
  if (character >= '0' && character <= '9') {
    return static_cast<std::uint8_t>(character - '0');
  }
  if (character >= 'a' && character <= 'f') {
    return static_cast<std::uint8_t>(character - 'a' + 10);
  }
  if (character >= 'A' && character <= 'F') {
    return static_cast<std::uint8_t>(character - 'A' + 10);
  }
  throw std::runtime_error("recursive HT leaf proof 含非法十六进制字符");
}

std::string HexDecode(const std::string_view input) {
  if (input.empty() || input.size() % 2U != 0U || input.size() > kMaxHtProofBytes) {
    throw std::runtime_error("recursive HT leaf proof 的十六进制长度非法");
  }
  std::string output(input.size() / 2U, '\0');
  for (std::size_t index = 0; index < output.size(); ++index) {
    output[index] =
        static_cast<char>((HexNibble(input[2U * index]) << 4U) | HexNibble(input[2U * index + 1U]));
  }
  return output;
}

void WriteHtRecursiveProofStream(std::ostream* const output, const HtRecursiveProof& proof) {
  if (proof.reason.size() > 4096U || proof.nodes.size() > kMaxHtNodes) {
    throw std::invalid_argument("recursive HT proof 的 reason 或节点数超出 V1 上限");
  }
  *output << "CUDAEE_HT_RECURSIVE_PROOF_V1\n";
  *output << "proven " << (proof.proven ? 1 : 0) << '\n';
  *output << "reason " << std::quoted(proof.reason) << '\n';
  *output << "snapshot_hash " << std::hex << std::setfill('0') << std::setw(16)
          << proof.snapshot_hash << std::dec << '\n';
  *output << "target " << proof.target_edge.u << ' ' << proof.target_edge.v << '\n';
  *output << "cd_mode " << CdModeToken(proof.cd_mode) << '\n';
  *output << "cd_candidates_tested " << proof.cd_candidates_tested << '\n';
  *output << "states_expanded " << proof.states_expanded << '\n';
  *output << "replies_expanded " << proof.replies_expanded << '\n';
  *output << "leaf_calls " << proof.leaf_calls << '\n';
  *output << "node_count " << proof.nodes.size() << '\n';

  std::size_t total_replies = 0;
  std::size_t total_path_nodes = 0;
  for (std::size_t node_index = 0; node_index < proof.nodes.size(); ++node_index) {
    const HtTreeNode& node = proof.nodes[node_index];
    if (!node.paths.valid || node.paths.paths.size() > kMaxHtPaths ||
        node.replies.size() > kMaxHtReplies ||
        total_replies > kMaxHtReplies - node.replies.size()) {
      throw std::invalid_argument("recursive HT proof 的路径或 reply 数量超出 V1 上限");
    }
    total_replies += node.replies.size();
    *output << "node " << node_index << '\n';
    *output << "path_count " << node.paths.paths.size() << '\n';
    for (std::size_t path_index = 0; path_index < node.paths.paths.size(); ++path_index) {
      const Path& path = node.paths.paths[path_index];
      if (path.size() < 2U || total_path_nodes > kMaxHtPathNodes - path.size()) {
        throw std::invalid_argument("recursive HT proof 的路径节点数超出 V1 上限");
      }
      total_path_nodes += path.size();
      *output << "path " << path_index << ' ' << path.size();
      for (const std::int32_t value : path) {
        *output << ' ' << value;
      }
      *output << '\n';
    }
    *output << "move_type " << MoveTypeToken(node.move_type) << '\n';
    *output << "move " << node.move_first << ' ' << node.move_second << '\n';
    *output << "reply_count " << node.replies.size() << '\n';
    for (const HtTreeReply& reply : node.replies) {
      *output << "reply " << reply.first_pair.center << ' ' << reply.first_pair.first << ' '
              << reply.first_pair.second << ' ' << reply.second_pair.center << ' '
              << reply.second_pair.first << ' ' << reply.second_pair.second << ' ' << reply.edge.u
              << ' ' << reply.edge.v << ' ' << (reply.path_infeasible ? 1 : 0) << ' '
              << reply.child_index << '\n';
    }
    if (node.move_type == HtMoveType::kLeaf) {
      *output << "leaf_hex " << HexEncode(SerializePathSystemKOptProof(node.leaf_proof)) << '\n';
    } else {
      if (node.leaf_proof.proven) {
        throw std::invalid_argument("recursive HT 非 leaf 节点携带已授权 leaf proof");
      }
      *output << "leaf_hex -\n";
    }
    *output << "endnode\n";
  }
  *output << "END\n";
}

HtRecursiveProof ReadHtRecursiveProofStream(std::istream* const input) {
  ExpectToken(input, "CUDAEE_HT_RECURSIVE_PROOF_V1");
  HtRecursiveProof proof;
  ExpectToken(input, "proven");
  const std::uint32_t proven = ReadInteger<std::uint32_t>(input, "proven");
  if (proven > 1U) {
    throw std::runtime_error("recursive HT proof 的 proven 非法");
  }
  proof.proven = proven == 1U;
  ExpectToken(input, "reason");
  if (!(*input >> std::quoted(proof.reason)) || proof.reason.size() > 4096U) {
    throw std::runtime_error("recursive HT proof 的 reason 非法");
  }
  ExpectToken(input, "snapshot_hash");
  proof.snapshot_hash = ReadHexHash(input, "snapshot_hash");
  ExpectToken(input, "target");
  proof.target_edge.u = ReadInteger<std::int32_t>(input, "target.u");
  proof.target_edge.v = ReadInteger<std::int32_t>(input, "target.v");
  ExpectToken(input, "cd_mode");
  proof.cd_mode = ParseCdMode(input);
  ExpectToken(input, "cd_candidates_tested");
  proof.cd_candidates_tested = ReadInteger<std::uint64_t>(input, "cd_candidates_tested");
  ExpectToken(input, "states_expanded");
  proof.states_expanded = ReadInteger<std::uint64_t>(input, "states_expanded");
  ExpectToken(input, "replies_expanded");
  proof.replies_expanded = ReadInteger<std::uint64_t>(input, "replies_expanded");
  ExpectToken(input, "leaf_calls");
  proof.leaf_calls = ReadInteger<std::uint64_t>(input, "leaf_calls");
  ExpectToken(input, "node_count");
  const std::size_t node_count = ReadInteger<std::size_t>(input, "node_count");
  if (node_count > kMaxHtNodes || (proof.proven && node_count == 0U)) {
    throw std::runtime_error("recursive HT proof 的 node_count 非法");
  }
  proof.nodes.reserve(node_count);

  std::size_t total_replies = 0;
  std::size_t total_path_nodes = 0;
  for (std::size_t node_index = 0; node_index < node_count; ++node_index) {
    ExpectToken(input, "node");
    if (ReadInteger<std::size_t>(input, "node index") != node_index) {
      throw std::runtime_error("recursive HT proof 的 node index 非连续");
    }
    HtTreeNode node;
    node.paths.valid = true;
    ExpectToken(input, "path_count");
    const std::size_t path_count = ReadInteger<std::size_t>(input, "path_count");
    if (path_count == 0U || path_count > kMaxHtPaths) {
      throw std::runtime_error("recursive HT proof 的 path_count 非法");
    }
    node.paths.paths.reserve(path_count);
    for (std::size_t path_index = 0; path_index < path_count; ++path_index) {
      ExpectToken(input, "path");
      if (ReadInteger<std::size_t>(input, "path index") != path_index) {
        throw std::runtime_error("recursive HT proof 的 path index 非连续");
      }
      const std::size_t path_size = ReadInteger<std::size_t>(input, "path size");
      if (path_size < 2U || path_size > kMaxHtPathNodes ||
          total_path_nodes > kMaxHtPathNodes - path_size) {
        throw std::runtime_error("recursive HT proof 的路径节点数非法");
      }
      total_path_nodes += path_size;
      Path path(path_size);
      for (std::int32_t& value : path) {
        value = ReadInteger<std::int32_t>(input, "path node");
      }
      if (node.paths.edge_count > std::numeric_limits<std::size_t>::max() - (path_size - 1U)) {
        throw std::runtime_error("recursive HT proof 的 path edge 数溢出");
      }
      node.paths.edge_count += path_size - 1U;
      node.paths.paths.push_back(std::move(path));
    }
    ExpectToken(input, "move_type");
    node.move_type = ParseMoveType(input);
    ExpectToken(input, "move");
    node.move_first = ReadInteger<std::int32_t>(input, "move.first");
    node.move_second = ReadInteger<std::int32_t>(input, "move.second");
    ExpectToken(input, "reply_count");
    const std::size_t reply_count = ReadInteger<std::size_t>(input, "reply_count");
    if (reply_count > kMaxHtReplies || total_replies > kMaxHtReplies - reply_count) {
      throw std::runtime_error("recursive HT proof 的 reply_count 非法");
    }
    total_replies += reply_count;
    node.replies.reserve(reply_count);
    for (std::size_t reply_index = 0; reply_index < reply_count; ++reply_index) {
      ExpectToken(input, "reply");
      HtTreeReply reply;
      reply.first_pair.center = ReadInteger<std::int32_t>(input, "reply.first.center");
      reply.first_pair.first = ReadInteger<std::int32_t>(input, "reply.first.first");
      reply.first_pair.second = ReadInteger<std::int32_t>(input, "reply.first.second");
      reply.second_pair.center = ReadInteger<std::int32_t>(input, "reply.second.center");
      reply.second_pair.first = ReadInteger<std::int32_t>(input, "reply.second.first");
      reply.second_pair.second = ReadInteger<std::int32_t>(input, "reply.second.second");
      reply.edge.u = ReadInteger<std::int32_t>(input, "reply.edge.u");
      reply.edge.v = ReadInteger<std::int32_t>(input, "reply.edge.v");
      const std::uint32_t infeasible = ReadInteger<std::uint32_t>(input, "path_infeasible");
      if (infeasible > 1U) {
        throw std::runtime_error("recursive HT proof 的 path_infeasible 非法");
      }
      reply.path_infeasible = infeasible == 1U;
      reply.child_index = ReadInteger<std::uint32_t>(input, "child_index");
      node.replies.push_back(reply);
    }
    ExpectToken(input, "leaf_hex");
    std::string leaf_hex;
    if (!(*input >> leaf_hex)) {
      throw std::runtime_error("recursive HT proof 缺少 leaf_hex");
    }
    if (node.move_type == HtMoveType::kLeaf) {
      if (leaf_hex == "-") {
        throw std::runtime_error("recursive HT leaf 缺少嵌套证明");
      }
      node.leaf_proof = ParsePathSystemKOptProof(HexDecode(leaf_hex));
    } else if (leaf_hex != "-") {
      throw std::runtime_error("recursive HT 非 leaf 节点含嵌套证明");
    }
    ExpectToken(input, "endnode");
    proof.nodes.push_back(std::move(node));
  }
  ExpectToken(input, "END");
  std::string trailing;
  if (*input >> trailing) {
    throw std::runtime_error("recursive HT proof 的 END 后存在多余字段");
  }
  return proof;
}

} // namespace

std::string SerializeHtRecursiveProof(const HtRecursiveProof& proof) {
  std::ostringstream output;
  WriteHtRecursiveProofStream(&output, proof);
  if (!output) {
    throw std::runtime_error("序列化 recursive HT proof 失败");
  }
  std::string serialized = output.str();
  if (serialized.size() > kMaxHtProofBytes) {
    throw std::runtime_error("recursive HT proof 超出 V1 文件大小上限");
  }
  return serialized;
}

HtRecursiveProof ParseHtRecursiveProof(const std::string_view serialized) {
  if (serialized.size() > kMaxHtProofBytes) {
    throw std::runtime_error("recursive HT proof 超出 V1 文件大小上限");
  }
  std::istringstream input{std::string(serialized)};
  return ReadHtRecursiveProofStream(&input);
}

void WriteHtRecursiveProof(const std::filesystem::path& path, const HtRecursiveProof& proof) {
  const std::string serialized = SerializeHtRecursiveProof(proof);
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 recursive HT proof: " + path.string());
  }
  output << serialized;
  if (!output) {
    throw std::runtime_error("写入 recursive HT proof 失败: " + path.string());
  }
}

HtRecursiveProof ReadHtRecursiveProof(const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    throw std::runtime_error("无法读取 recursive HT proof 大小: " + path.string());
  }
  if (size > kMaxHtProofBytes) {
    throw std::runtime_error("recursive HT proof 超出 V1 文件大小上限: " + path.string());
  }
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法打开 recursive HT proof: " + path.string());
  }
  return ReadHtRecursiveProofStream(&input);
}

} // namespace cudaee
