#include "cuda_edge_elimination/build_identity.hpp"

#include "cudaee_build_identity.hpp"

#include <openssl/evp.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace cudaee {

std::string Sha256File(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法读取哈希输入: " + path.string());
  }
  const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                        EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("无法初始化 SHA-256");
  }
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(count)) != 1) {
      throw std::runtime_error("SHA-256 流式计算失败");
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("SHA-256 输入读取不完整: " + path.string());
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0U;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1 || length != 32U) {
    throw std::runtime_error("SHA-256 结果无效");
  }
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (unsigned int index = 0U; index < length; ++index) {
    result << std::setw(2) << static_cast<unsigned int>(digest[index]);
  }
  return result.str();
}

std::string JsonString(const std::string_view text) {
  // std::quoted 不是 JSON 转义：换行、控制字符必须编码，否则 manifest
  // 会因合法文件名或编译器字符串而变成不可解析的 JSON。
  constexpr char hex[] = "0123456789abcdef";
  std::string result{"\""};
  for (const char character : text) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (byte == '"' || byte == '\\') {
      result.push_back('\\');
      result.push_back(character);
    } else if (byte < 0x20U) {
      result += "\\u00";
      result.push_back(hex[byte >> 4U]);
      result.push_back(hex[byte & 15U]);
    } else {
      result.push_back(character);
    }
  }
  result.push_back('"');
  return result;
}

void WriteBuildIdentityJson(std::ostream& output) {
  const std::filesystem::path executable = std::filesystem::read_symlink("/proc/self/exe");
  output << "{\"git_commit\":" << JsonString(build_identity::kCommit)
         << ",\"git_dirty\":" << (build_identity::kDirty ? "true" : "false")
         << ",\"git_diff_sha256\":" << JsonString(build_identity::kDiffSha256)
         << ",\"source_tree_sha256\":" << JsonString(build_identity::kSourceSha256)
         << ",\"executable_sha256\":" << JsonString(Sha256File("/proc/self/exe"))
         << ",\"executable\":" << JsonString(executable.string())
         << ",\"compiler\":" << JsonString(build_identity::kCompiler)
         << ",\"cuda_compiler\":" << JsonString(build_identity::kCudaCompiler)
         << ",\"cuda_architectures\":" << JsonString(build_identity::kArchitecture)
         << ",\"configuration\":" << JsonString(build_identity::kConfiguration) << '}';
}

#if !defined(CUDAEE_HAS_CUDA)
void WriteGpuIdentityJson(std::ostream& output, const int) { output << "null"; }
#endif

} // namespace cudaee
