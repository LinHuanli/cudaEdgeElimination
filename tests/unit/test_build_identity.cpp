#include "cuda_edge_elimination/build_identity.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

int main() {
  const std::filesystem::path directory(CUDAEE_TEST_TMP_DIR);
  std::filesystem::create_directories(directory);
  const std::filesystem::path input = directory / "sha256-abc.txt";
  {
    std::ofstream output(input, std::ios::binary);
    output << "abc";
  }
  if (cudaee::Sha256File(input) !=
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
    throw std::runtime_error("SHA-256 标准向量不匹配");
  }
  if (cudaee::JsonString("a\n\t\"\\") != "\"a\\u000a\\u0009\\\"\\\\\"") {
    throw std::runtime_error("JSON 控制字符转义错误");
  }
  bool rejected = false;
  try {
    static_cast<void>(cudaee::Sha256File(directory / "does-not-exist.sha256"));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("未拒绝不存在的哈希输入");
  }
  std::ostringstream identity;
  cudaee::WriteBuildIdentityJson(identity);
  if (identity.str().find("\"source_tree_sha256\":") == std::string::npos ||
      identity.str().find("\"executable_sha256\":") == std::string::npos) {
    throw std::runtime_error("构建身份字段缺失");
  }
  std::cout << "build identity tests passed\n";
}
