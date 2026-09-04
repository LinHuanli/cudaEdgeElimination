#include "cuda_edge_elimination/lp_epoch.hpp"

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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cudaee {
namespace {

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

void HashDouble(std::uint64_t* hash, const double value) {
  const auto bits = std::bit_cast<std::uint64_t>(value);
  HashValue(hash, bits);
}

template <typename T>
void WriteVector(std::ostream& output, const std::string& name, const std::vector<T>& values) {
  output << name << ' ' << values.size();
  for (const T& value : values) {
    output << ' ' << value;
  }
  output << '\n';
}

template <typename T>
std::vector<T> ReadVector(std::istringstream* fields, const std::string& name) {
  std::size_t count = 0;
  if (!(*fields >> count)) {
    throw std::runtime_error("LP epoch 的 " + name + " 缺少长度");
  }
  std::vector<T> values(count);
  for (T& value : values) {
    if (!(*fields >> value)) {
      throw std::runtime_error("LP epoch 的 " + name + " 数组长度不足");
    }
  }
  *fields >> std::ws;
  if (!fields->eof()) {
    throw std::runtime_error("LP epoch 的 " + name + " 行包含多余字段");
  }
  // 到达行尾会设置 eofbit；清除它，避免调用方把成功读取误判为失败。
  fields->clear();
  return values;
}

std::string Int128ToString(__int128 value) {
  if (value == 0) {
    return "0";
  }
  const bool negative = value < 0;
  unsigned __int128 magnitude = negative ? static_cast<unsigned __int128>(-(value + 1)) + 1
                                         : static_cast<unsigned __int128>(value);
  std::string result;
  while (magnitude != 0) {
    result.push_back(static_cast<char>('0' + magnitude % 10));
    magnitude /= 10;
  }
  if (negative) {
    result.push_back('-');
  }
  std::reverse(result.begin(), result.end());
  return result;
}

bool ToConservativeInteger(const double value, std::int64_t* result) {
  if (!std::isfinite(value) || std::abs(value) > 1.0e12) {
    return false;
  }
  const double rounded = std::round(value);
  if (std::abs(value - rounded) > 1.0e-9) {
    return false;
  }
  *result = static_cast<std::int64_t>(rounded);
  return true;
}

bool CheckedAdd(const __int128 lhs, const __int128 rhs, __int128* result) {
  return !__builtin_add_overflow(lhs, rhs, result);
}

bool CheckedMultiply(const __int128 lhs, const __int128 rhs, __int128* result) {
  return !__builtin_mul_overflow(lhs, rhs, result);
}

std::uint64_t ComputeStableIdentityHashValue(const std::vector<std::uint64_t>& column_ids,
                                             const std::vector<std::uint64_t>& row_ids) {
  std::vector<std::uint64_t> canonical_columns = column_ids;
  std::vector<std::uint64_t> canonical_rows = row_ids;
  std::sort(canonical_columns.begin(), canonical_columns.end());
  std::sort(canonical_rows.begin(), canonical_rows.end());
  std::uint64_t hash = 14695981039346656037ULL;
  constexpr std::uint64_t kColumnDomain = 0x434f4c554d4e4944ULL;
  constexpr std::uint64_t kRowDomain = 0x524f574944454e54ULL;
  HashValue(&hash, kColumnDomain);
  HashValue(&hash, static_cast<std::uint64_t>(canonical_columns.size()));
  for (const std::uint64_t value : canonical_columns) {
    HashValue(&hash, value);
  }
  HashValue(&hash, kRowDomain);
  HashValue(&hash, static_cast<std::uint64_t>(canonical_rows.size()));
  for (const std::uint64_t value : canonical_rows) {
    HashValue(&hash, value);
  }
  return hash;
}

} // namespace

void LpEpoch::Validate() const {
  if (rows < 0 || columns < 0 || objective_sense != 1) {
    throw std::runtime_error("LP epoch 仅支持维度非负的最小化 LP");
  }
  if (objective.size() != static_cast<std::size_t>(columns) ||
      row_offsets.size() != static_cast<std::size_t>(rows) + 1 ||
      senses.size() != static_cast<std::size_t>(rows) ||
      rhs.size() != static_cast<std::size_t>(rows) ||
      lower_bounds.size() != static_cast<std::size_t>(columns) ||
      upper_bounds.size() != static_cast<std::size_t>(columns) ||
      variable_types.size() != static_cast<std::size_t>(columns) ||
      edge_u.size() != static_cast<std::size_t>(columns) ||
      edge_v.size() != static_cast<std::size_t>(columns)) {
    throw std::runtime_error("LP epoch 数组维度不一致");
  }
  if (row_offsets.empty() || row_offsets.front() != 0 || row_offsets.back() < 0 ||
      static_cast<std::size_t>(row_offsets.back()) != column_indices.size() ||
      values.size() != column_indices.size()) {
    throw std::runtime_error("LP epoch CSR 边界无效");
  }
  for (std::size_t i = 1; i < row_offsets.size(); ++i) {
    if (row_offsets[i] < row_offsets[i - 1]) {
      throw std::runtime_error("LP epoch CSR row_offsets 非单调");
    }
  }
  for (const std::int32_t column : column_indices) {
    if (column < 0 || column >= columns) {
      throw std::runtime_error("LP epoch CSR 列索引越界");
    }
  }
  if (!std::isfinite(objective_offset) ||
      std::any_of(objective.begin(), objective.end(),
                  [](double value) { return !std::isfinite(value); }) ||
      std::any_of(values.begin(), values.end(),
                  [](double value) { return !std::isfinite(value); }) ||
      std::any_of(rhs.begin(), rhs.end(), [](double value) { return !std::isfinite(value); })) {
    throw std::runtime_error("LP epoch 的目标、矩阵或 RHS 包含 NaN/Inf");
  }
  for (std::int32_t row = 0; row < rows; ++row) {
    if (senses[static_cast<std::size_t>(row)] != 'L' &&
        senses[static_cast<std::size_t>(row)] != 'G' &&
        senses[static_cast<std::size_t>(row)] != 'E') {
      throw std::runtime_error("LP epoch 包含未知行方向");
    }
  }
  for (std::int32_t column = 0; column < columns; ++column) {
    if (variable_types[static_cast<std::size_t>(column)] != 'C') {
      throw std::runtime_error("cuOpt sidecar 首期只接受连续变量");
    }
    if (std::isnan(lower_bounds[static_cast<std::size_t>(column)]) ||
        std::isnan(upper_bounds[static_cast<std::size_t>(column)]) ||
        lower_bounds[static_cast<std::size_t>(column)] >
            upper_bounds[static_cast<std::size_t>(column)]) {
      throw std::runtime_error("LP epoch 变量边界无效");
    }
  }
}

std::uint64_t LpEpoch::ComputeHash() const {
  std::uint64_t hash = 14695981039346656037ULL;
  HashValue(&hash, rows);
  HashValue(&hash, columns);
  HashValue(&hash, objective_sense);
  HashDouble(&hash, objective_offset);
  for (const double value : objective)
    HashDouble(&hash, value);
  for (const std::int32_t value : row_offsets)
    HashValue(&hash, value);
  for (const std::int32_t value : column_indices)
    HashValue(&hash, value);
  for (const double value : values)
    HashDouble(&hash, value);
  for (const char value : senses)
    HashValue(&hash, value);
  for (const double value : rhs)
    HashDouble(&hash, value);
  for (const double value : lower_bounds)
    HashDouble(&hash, value);
  for (const double value : upper_bounds)
    HashDouble(&hash, value);
  for (const char value : variable_types)
    HashValue(&hash, value);
  for (const std::int32_t value : edge_u)
    HashValue(&hash, value);
  for (const std::int32_t value : edge_v)
    HashValue(&hash, value);
  return hash;
}

LpEpoch ReadLpEpoch(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法打开 LP epoch: " + path.string());
  }
  std::string magic;
  std::getline(input, magic);
  if (magic != "CUDAEE_LP_EPOCH_V1") {
    throw std::runtime_error("LP epoch 版本不受支持");
  }

  LpEpoch epoch;
  bool saw_hash = false;
  bool saw_end = false;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string name;
    fields >> name;
    if (name == "rows") {
      fields >> epoch.rows;
    } else if (name == "columns") {
      fields >> epoch.columns;
    } else if (name == "objective_sense") {
      fields >> epoch.objective_sense;
    } else if (name == "objective_offset") {
      fields >> epoch.objective_offset;
    } else if (name == "objective") {
      epoch.objective = ReadVector<double>(&fields, name);
    } else if (name == "row_offsets") {
      epoch.row_offsets = ReadVector<std::int32_t>(&fields, name);
    } else if (name == "column_indices") {
      epoch.column_indices = ReadVector<std::int32_t>(&fields, name);
    } else if (name == "values") {
      epoch.values = ReadVector<double>(&fields, name);
    } else if (name == "senses") {
      epoch.senses = ReadVector<char>(&fields, name);
    } else if (name == "rhs") {
      epoch.rhs = ReadVector<double>(&fields, name);
    } else if (name == "lower_bounds") {
      epoch.lower_bounds = ReadVector<double>(&fields, name);
    } else if (name == "upper_bounds") {
      epoch.upper_bounds = ReadVector<double>(&fields, name);
    } else if (name == "variable_types") {
      epoch.variable_types = ReadVector<char>(&fields, name);
    } else if (name == "edge_u") {
      epoch.edge_u = ReadVector<std::int32_t>(&fields, name);
    } else if (name == "edge_v") {
      epoch.edge_v = ReadVector<std::int32_t>(&fields, name);
    } else if (name == "hash") {
      std::string value;
      fields >> value;
      epoch.content_hash = std::stoull(value, nullptr, 16);
      saw_hash = true;
    } else if (name == "END") {
      saw_end = true;
      break;
    } else if (!name.empty()) {
      throw std::runtime_error("LP epoch 包含未知字段: " + name);
    }
    if (!fields && name != "END") {
      throw std::runtime_error("LP epoch 字段解析失败: " + name);
    }
  }
  if (!saw_hash || !saw_end) {
    throw std::runtime_error("LP epoch 缺少 hash 或 END");
  }
  epoch.Validate();
  if (epoch.ComputeHash() != epoch.content_hash) {
    throw std::runtime_error("LP epoch 内容哈希不匹配");
  }
  return epoch;
}

void WriteLpEpoch(const std::filesystem::path& path, const LpEpoch& input_epoch) {
  LpEpoch epoch = input_epoch;
  epoch.Validate();
  epoch.content_hash = epoch.ComputeHash();
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 LP epoch: " + path.string());
  }
  output << std::setprecision(17);
  output << "CUDAEE_LP_EPOCH_V1\n";
  output << "rows " << epoch.rows << '\n';
  output << "columns " << epoch.columns << '\n';
  output << "objective_sense " << epoch.objective_sense << '\n';
  output << "objective_offset " << epoch.objective_offset << '\n';
  WriteVector(output, "objective", epoch.objective);
  WriteVector(output, "row_offsets", epoch.row_offsets);
  WriteVector(output, "column_indices", epoch.column_indices);
  WriteVector(output, "values", epoch.values);
  WriteVector(output, "senses", epoch.senses);
  WriteVector(output, "rhs", epoch.rhs);
  WriteVector(output, "lower_bounds", epoch.lower_bounds);
  WriteVector(output, "upper_bounds", epoch.upper_bounds);
  WriteVector(output, "variable_types", epoch.variable_types);
  WriteVector(output, "edge_u", epoch.edge_u);
  WriteVector(output, "edge_v", epoch.edge_v);
  output << "hash " << std::hex << std::setfill('0') << std::setw(16) << epoch.content_hash
         << std::dec << '\n';
  output << "END\n";
}

void WriteLpSolution(const std::filesystem::path& path, const LpEpoch& epoch,
                     const LpSolution& solution) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 LP solution: " + path.string());
  }
  output << std::setprecision(17);
  output << "CUDAEE_LP_SOLUTION_V1\n";
  output << "model_hash " << std::hex << std::setfill('0') << std::setw(16) << epoch.content_hash
         << std::dec << '\n';
  output << "solver " << solution.solver << '\n';
  output << "solver_version " << solution.solver_version << '\n';
  output << "status " << solution.status << '\n';
  output << "termination_status " << solution.termination_status << '\n';
  output << "objective " << solution.objective << '\n';
  output << "dual_objective " << solution.dual_objective << '\n';
  output << "solve_time_seconds " << solution.solve_time_seconds << '\n';
  output << "max_primal_violation " << solution.max_primal_violation << '\n';
  output << "max_reduced_cost_residual " << solution.max_reduced_cost_residual << '\n';
  output << "numerically_accepted " << (solution.numerically_accepted ? 1 : 0) << '\n';
  output << "stable_identity_hash " << std::hex << std::setfill('0') << std::setw(16)
         << solution.stable_identity_hash << std::dec << '\n';
  output << "warm_start_attempted " << (solution.warm_start_attempted ? 1 : 0) << '\n';
  output << "warm_start_applied " << (solution.warm_start_applied ? 1 : 0) << '\n';
  output << "warm_start_column_coverage " << solution.warm_start_column_coverage << '\n';
  output << "warm_start_row_coverage " << solution.warm_start_row_coverage << '\n';
  output << "warm_start_reason " << solution.warm_start_reason << '\n';
  output << "exact_model_bound_certified " << (solution.exact_model_bound.certified ? 1 : 0)
         << '\n';
  output << "exact_model_bound_numerator " << solution.exact_model_bound.numerator << '\n';
  output << "exact_model_bound_denominator " << solution.exact_model_bound.denominator << '\n';
  output << "exact_model_bound_reason " << solution.exact_model_bound.reason << '\n';
  WriteVector(output, "primal", solution.primal);
  WriteVector(output, "dual", solution.dual);
  WriteVector(output, "reduced_costs", solution.reduced_costs);
  output << "END\n";
}

LpStableIdentity ComputeLpStableIdentity(const LpEpoch& epoch) {
  epoch.Validate();
  LpStableIdentity identity;
  identity.column_ids.resize(static_cast<std::size_t>(epoch.columns));
  std::unordered_set<std::uint64_t> unique_columns;
  unique_columns.reserve(static_cast<std::size_t>(epoch.columns));
  for (std::int32_t column = 0; column < epoch.columns; ++column) {
    std::int32_t u = epoch.edge_u[static_cast<std::size_t>(column)];
    std::int32_t v = epoch.edge_v[static_cast<std::size_t>(column)];
    if (u < 0 || v < 0 || u == v) {
      identity.reason = "列缺少合法的 Concorde 边端点，禁止按位置 warm start";
      return identity;
    }
    if (u > v) {
      std::swap(u, v);
    }
    // 端点加一后打包，保留 0 作为非法身份哨兵。
    const std::uint64_t stable_id =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u) + 1U) << 32U) |
        (static_cast<std::uint32_t>(v) + 1U);
    if (!unique_columns.insert(stable_id).second) {
      identity.reason = "列—边映射包含重复稳定身份";
      return identity;
    }
    identity.column_ids[static_cast<std::size_t>(column)] = stable_id;
  }

  identity.row_ids.resize(static_cast<std::size_t>(epoch.rows));
  std::unordered_set<std::uint64_t> unique_rows;
  unique_rows.reserve(static_cast<std::size_t>(epoch.rows));
  for (std::int32_t row = 0; row < epoch.rows; ++row) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> entries;
    const std::int32_t begin = epoch.row_offsets[static_cast<std::size_t>(row)];
    const std::int32_t end = epoch.row_offsets[static_cast<std::size_t>(row) + 1U];
    entries.reserve(static_cast<std::size_t>(end - begin));
    for (std::int32_t offset = begin; offset < end; ++offset) {
      const std::int32_t column = epoch.column_indices[static_cast<std::size_t>(offset)];
      entries.emplace_back(
          identity.column_ids[static_cast<std::size_t>(column)],
          std::bit_cast<std::uint64_t>(epoch.values[static_cast<std::size_t>(offset)]));
    }
    std::sort(entries.begin(), entries.end());
    std::uint64_t row_hash = 14695981039346656037ULL;
    HashValue(&row_hash, epoch.senses[static_cast<std::size_t>(row)]);
    HashDouble(&row_hash, epoch.rhs[static_cast<std::size_t>(row)]);
    const std::uint64_t entry_count = static_cast<std::uint64_t>(entries.size());
    HashValue(&row_hash, entry_count);
    for (const auto [column_id, coefficient_bits] : entries) {
      HashValue(&row_hash, column_id);
      HashValue(&row_hash, coefficient_bits);
    }
    if (!unique_rows.insert(row_hash).second) {
      identity.reason = "模型包含稳定身份相同的重复行，dual 映射不唯一";
      return identity;
    }
    identity.row_ids[static_cast<std::size_t>(row)] = row_hash;
  }

  // 总身份哈希对行列重排不敏感；映射向量本身仍保持当前 epoch 的位置顺序。
  identity.identity_hash = ComputeStableIdentityHashValue(identity.column_ids, identity.row_ids);
  identity.complete = true;
  identity.reason = "stable edge/row identity complete";
  return identity;
}

LpWarmStart BuildLpWarmStart(const LpEpoch& epoch, const LpSolution& solution) {
  epoch.Validate();
  if (!solution.numerically_accepted ||
      solution.primal.size() != static_cast<std::size_t>(epoch.columns) ||
      solution.dual.size() != static_cast<std::size_t>(epoch.rows) ||
      std::any_of(solution.primal.begin(), solution.primal.end(),
                  [](const double value) { return !std::isfinite(value); }) ||
      std::any_of(solution.dual.begin(), solution.dual.end(),
                  [](const double value) { return !std::isfinite(value); })) {
    throw std::invalid_argument("只有数值门禁通过且维度完整的 LP 解可建立 warm start");
  }
  LpWarmStart warm;
  warm.source_model_hash = epoch.ComputeHash();
  warm.identity = ComputeLpStableIdentity(epoch);
  if (!warm.identity.complete) {
    throw std::invalid_argument("LP stable identity 不完整: " + warm.identity.reason);
  }
  warm.primal = solution.primal;
  warm.dual = solution.dual;
  return warm;
}

LpWarmStartProjection ProjectLpWarmStart(const LpWarmStart& source, const LpEpoch& target,
                                         const double minimum_coverage) {
  target.Validate();
  if (!std::isfinite(minimum_coverage) || minimum_coverage < 0.0 || minimum_coverage > 1.0) {
    throw std::invalid_argument("LP warm-start minimum coverage 必须位于 [0,1]");
  }
  LpWarmStartProjection projection;
  if (!source.identity.complete || source.identity.column_ids.size() != source.primal.size() ||
      source.identity.row_ids.size() != source.dual.size()) {
    projection.reason = "源 warm start 的稳定身份或向量维度不完整";
    return projection;
  }
  if (source.identity.identity_hash !=
      ComputeStableIdentityHashValue(source.identity.column_ids, source.identity.row_ids)) {
    projection.reason = "源 warm start 的稳定身份哈希不匹配";
    return projection;
  }
  const LpStableIdentity target_identity = ComputeLpStableIdentity(target);
  if (!target_identity.complete) {
    projection.reason = "目标 LP stable identity 不完整: " + target_identity.reason;
    return projection;
  }

  std::unordered_map<std::uint64_t, double> source_primal;
  std::unordered_map<std::uint64_t, double> source_dual;
  source_primal.reserve(source.primal.size());
  source_dual.reserve(source.dual.size());
  for (std::size_t index = 0U; index < source.primal.size(); ++index) {
    if (!std::isfinite(source.primal[index])) {
      projection.reason = "源 primal 包含 NaN/Inf";
      return projection;
    }
    if (source.identity.column_ids[index] == 0U ||
        !source_primal.emplace(source.identity.column_ids[index], source.primal[index]).second) {
      projection.reason = "源 warm start 包含零值或重复列身份";
      return projection;
    }
  }
  for (std::size_t index = 0U; index < source.dual.size(); ++index) {
    if (!std::isfinite(source.dual[index])) {
      projection.reason = "源 dual 包含 NaN/Inf";
      return projection;
    }
    if (source.identity.row_ids[index] == 0U ||
        !source_dual.emplace(source.identity.row_ids[index], source.dual[index]).second) {
      projection.reason = "源 warm start 包含零值或重复行身份";
      return projection;
    }
  }

  projection.primal.resize(static_cast<std::size_t>(target.columns));
  std::size_t matched_columns = 0U;
  for (std::int32_t column = 0; column < target.columns; ++column) {
    const std::size_t index = static_cast<std::size_t>(column);
    double value = std::clamp(0.0, target.lower_bounds[index], target.upper_bounds[index]);
    const auto source_value = source_primal.find(target_identity.column_ids[index]);
    if (source_value != source_primal.end()) {
      value =
          std::clamp(source_value->second, target.lower_bounds[index], target.upper_bounds[index]);
      ++matched_columns;
    }
    projection.primal[index] = value;
  }
  projection.dual.assign(static_cast<std::size_t>(target.rows), 0.0);
  std::size_t matched_rows = 0U;
  for (std::int32_t row = 0; row < target.rows; ++row) {
    const std::size_t index = static_cast<std::size_t>(row);
    const auto source_value = source_dual.find(target_identity.row_ids[index]);
    if (source_value != source_dual.end()) {
      projection.dual[index] = source_value->second;
      ++matched_rows;
    }
  }
  projection.column_coverage = target.columns == 0 ? 1.0
                                                   : static_cast<double>(matched_columns) /
                                                         static_cast<double>(target.columns);
  projection.row_coverage =
      target.rows == 0 ? 1.0 : static_cast<double>(matched_rows) / static_cast<double>(target.rows);
  projection.accepted =
      projection.column_coverage >= minimum_coverage && projection.row_coverage >= minimum_coverage;
  projection.reason = projection.accepted ? "stable identity coverage accepted"
                                          : "stable identity coverage below threshold";
  return projection;
}

ExactModelEvaluation BuildExactModelEvaluation(const LpEpoch& epoch,
                                               const std::vector<double>& dual,
                                               const std::uint32_t fractional_bits) {
  ExactModelEvaluation evaluation;
  ExactBound& result = evaluation.bound;
  if (dual.size() != static_cast<std::size_t>(epoch.rows) || fractional_bits > 40) {
    result.reason = "对偶维度不匹配或定点位数超限";
    return evaluation;
  }
  const std::uint64_t denominator = std::uint64_t{1} << fractional_bits;
  result.denominator = denominator;

  std::vector<std::int64_t> q(static_cast<std::size_t>(epoch.rows));
  for (std::int32_t row = 0; row < epoch.rows; ++row) {
    double value = dual[static_cast<std::size_t>(row)];
    if (!std::isfinite(value)) {
      result.reason = "对偶包含 NaN/Inf";
      return evaluation;
    }
    if (epoch.senses[static_cast<std::size_t>(row)] == 'L') {
      value = std::min(value, 0.0);
    } else if (epoch.senses[static_cast<std::size_t>(row)] == 'G') {
      value = std::max(value, 0.0);
    }
    const long double scaled = static_cast<long double>(value) * denominator;
    if (std::abs(scaled) > 1.0e15L) {
      result.reason = "量化对偶超过保守范围";
      return evaluation;
    }
    q[static_cast<std::size_t>(row)] = static_cast<std::int64_t>(std::round(scaled));
  }

  std::int64_t offset = 0;
  if (!ToConservativeInteger(epoch.objective_offset, &offset)) {
    result.reason = "目标偏置不是受支持的整数";
    return evaluation;
  }
  __int128 lower_bound = 0;
  if (!CheckedMultiply(offset, denominator, &lower_bound)) {
    result.reason = "目标偏置定点乘法溢出";
    return evaluation;
  }

  std::vector<__int128> reduced_numerator(static_cast<std::size_t>(epoch.columns));
  for (std::int32_t column = 0; column < epoch.columns; ++column) {
    std::int64_t coefficient = 0;
    if (!ToConservativeInteger(epoch.objective[static_cast<std::size_t>(column)], &coefficient) ||
        !CheckedMultiply(coefficient, denominator,
                         &reduced_numerator[static_cast<std::size_t>(column)])) {
      result.reason = "目标系数不是受支持的整数或发生溢出";
      return evaluation;
    }
  }

  for (std::int32_t row = 0; row < epoch.rows; ++row) {
    std::int64_t rhs = 0;
    if (!ToConservativeInteger(epoch.rhs[static_cast<std::size_t>(row)], &rhs)) {
      result.reason = "RHS 不是受支持的整数";
      return evaluation;
    }
    __int128 term = 0;
    if (!CheckedMultiply(rhs, q[static_cast<std::size_t>(row)], &term) ||
        !CheckedAdd(lower_bound, term, &lower_bound)) {
      result.reason = "对偶常数项溢出";
      return evaluation;
    }
    for (std::int32_t offset_index = epoch.row_offsets[static_cast<std::size_t>(row)];
         offset_index < epoch.row_offsets[static_cast<std::size_t>(row) + 1]; ++offset_index) {
      std::int64_t coefficient = 0;
      if (!ToConservativeInteger(epoch.values[static_cast<std::size_t>(offset_index)],
                                 &coefficient) ||
          !CheckedMultiply(coefficient, q[static_cast<std::size_t>(row)], &term)) {
        result.reason = "矩阵系数不是受支持的整数或发生溢出";
        return evaluation;
      }
      const std::int32_t column = epoch.column_indices[static_cast<std::size_t>(offset_index)];
      if (!CheckedAdd(reduced_numerator[static_cast<std::size_t>(column)], -term,
                      &reduced_numerator[static_cast<std::size_t>(column)])) {
        result.reason = "reduced cost 定点累加溢出";
        return evaluation;
      }
    }
  }

  for (std::int32_t column = 0; column < epoch.columns; ++column) {
    const __int128 reduced = reduced_numerator[static_cast<std::size_t>(column)];
    const double selected_bound = reduced >= 0
                                      ? epoch.lower_bounds[static_cast<std::size_t>(column)]
                                      : epoch.upper_bounds[static_cast<std::size_t>(column)];
    std::int64_t bound = 0;
    if (!ToConservativeInteger(selected_bound, &bound)) {
      result.reason = "最小化 reduced cost 需要的变量边界不是有限整数";
      return evaluation;
    }
    __int128 term = 0;
    if (!CheckedMultiply(reduced, bound, &term) || !CheckedAdd(lower_bound, term, &lower_bound)) {
      result.reason = "变量边界贡献溢出";
      return evaluation;
    }
  }

  result.certified = true;
  result.numerator = Int128ToString(lower_bound);
  result.reason = "仅认证 lp-epoch-v1 中显式变量；TSP 删除仍需 Concorde 完整图精确定价";
  evaluation.lower_bound_numerator = lower_bound;
  evaluation.reduced_cost_numerator = std::move(reduced_numerator);
  return evaluation;
}

ExactBound BuildExactModelBound(const LpEpoch& epoch, const std::vector<double>& dual,
                                const std::uint32_t fractional_bits) {
  return BuildExactModelEvaluation(epoch, dual, fractional_bits).bound;
}

} // namespace cudaee
