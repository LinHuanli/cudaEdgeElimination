#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>

namespace cudaee {

[[nodiscard]] std::string Sha256File(const std::filesystem::path& path);
[[nodiscard]] std::string JsonString(std::string_view text);
// 输出一个 JSON object，不含外层 manifest 的字段名和逗号。
void WriteBuildIdentityJson(std::ostream& output);
void WriteGpuIdentityJson(std::ostream& output, int device);

} // namespace cudaee
