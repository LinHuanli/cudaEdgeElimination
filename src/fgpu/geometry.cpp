#include "cuda_edge_elimination/fgpu.hpp"

#include "../cpu/elimination_commit.hpp"
#include "geometry_backend.hpp"

#include <mpfr.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr mpfr_prec_t kGeometryPrecision = 256;
constexpr std::uint32_t kMaxPotentialCandidates = 32U;
constexpr std::uint32_t kMaxWitnessesPerEdge = 8U;

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

class MpReal {
public:
  MpReal() { mpfr_init2(value_, kGeometryPrecision); }

  MpReal(const MpReal& other) {
    mpfr_init2(value_, kGeometryPrecision);
    mpfr_set(value_, other.value_, MPFR_RNDN);
  }

  MpReal(MpReal&& other) noexcept {
    mpfr_init2(value_, kGeometryPrecision);
    mpfr_swap(value_, other.value_);
  }

  MpReal& operator=(const MpReal& other) {
    if (this != &other) {
      mpfr_set(value_, other.value_, MPFR_RNDN);
    }
    return *this;
  }

  MpReal& operator=(MpReal&& other) noexcept {
    if (this != &other) {
      mpfr_swap(value_, other.value_);
    }
    return *this;
  }

  ~MpReal() { mpfr_clear(value_); }

  [[nodiscard]] mpfr_ptr get() { return value_; }
  [[nodiscard]] mpfr_srcptr get() const { return value_; }

private:
  mpfr_t value_;
};

struct Interval {
  MpReal lower;
  MpReal upper;
};

Interval ExactInteger(const std::int64_t value) {
  Interval result;
  mpfr_set_sj(result.lower.get(), value, MPFR_RNDN);
  mpfr_set_sj(result.upper.get(), value, MPFR_RNDN);
  return result;
}

Interval ExactHalf() {
  Interval result = ExactInteger(1);
  mpfr_div_2ui(result.lower.get(), result.lower.get(), 1U, MPFR_RNDN);
  mpfr_set(result.upper.get(), result.lower.get(), MPFR_RNDN);
  return result;
}

Interval Add(const Interval& lhs, const Interval& rhs) {
  Interval result;
  mpfr_add(result.lower.get(), lhs.lower.get(), rhs.lower.get(), MPFR_RNDD);
  mpfr_add(result.upper.get(), lhs.upper.get(), rhs.upper.get(), MPFR_RNDU);
  return result;
}

Interval Negate(const Interval& value) {
  Interval result;
  mpfr_neg(result.lower.get(), value.upper.get(), MPFR_RNDD);
  mpfr_neg(result.upper.get(), value.lower.get(), MPFR_RNDU);
  return result;
}

Interval Subtract(const Interval& lhs, const Interval& rhs) { return Add(lhs, Negate(rhs)); }

Interval Multiply(const Interval& lhs, const Interval& rhs) {
  Interval result;
  MpReal candidate;
  bool first = true;
  for (const mpfr_srcptr left : {lhs.lower.get(), lhs.upper.get()}) {
    for (const mpfr_srcptr right : {rhs.lower.get(), rhs.upper.get()}) {
      mpfr_mul(candidate.get(), left, right, MPFR_RNDD);
      if (first || mpfr_less_p(candidate.get(), result.lower.get()) != 0) {
        mpfr_set(result.lower.get(), candidate.get(), MPFR_RNDN);
      }
      first = false;
    }
  }
  first = true;
  for (const mpfr_srcptr left : {lhs.lower.get(), lhs.upper.get()}) {
    for (const mpfr_srcptr right : {rhs.lower.get(), rhs.upper.get()}) {
      mpfr_mul(candidate.get(), left, right, MPFR_RNDU);
      if (first || mpfr_greater_p(candidate.get(), result.upper.get()) != 0) {
        mpfr_set(result.upper.get(), candidate.get(), MPFR_RNDN);
      }
      first = false;
    }
  }
  return result;
}

Interval Square(const Interval& value) {
  Interval result;
  const int lower_sign = mpfr_sgn(value.lower.get());
  const int upper_sign = mpfr_sgn(value.upper.get());
  if (lower_sign <= 0 && upper_sign >= 0) {
    mpfr_set_zero(result.lower.get(), 0);
  } else {
    MpReal first;
    MpReal second;
    mpfr_sqr(first.get(), value.lower.get(), MPFR_RNDD);
    mpfr_sqr(second.get(), value.upper.get(), MPFR_RNDD);
    mpfr_min(result.lower.get(), first.get(), second.get(), MPFR_RNDD);
  }
  MpReal first;
  MpReal second;
  mpfr_sqr(first.get(), value.lower.get(), MPFR_RNDU);
  mpfr_sqr(second.get(), value.upper.get(), MPFR_RNDU);
  mpfr_max(result.upper.get(), first.get(), second.get(), MPFR_RNDU);
  return result;
}

bool IsStrictlyPositive(const Interval& value) { return mpfr_sgn(value.lower.get()) > 0; }

bool ContainsZero(const Interval& value) {
  return mpfr_sgn(value.lower.get()) <= 0 && mpfr_sgn(value.upper.get()) >= 0;
}

Interval Reciprocal(const Interval& value) {
  if (ContainsZero(value)) {
    throw std::domain_error("几何区间除数包含零");
  }
  Interval result;
  mpfr_ui_div(result.lower.get(), 1U, value.upper.get(), MPFR_RNDD);
  mpfr_ui_div(result.upper.get(), 1U, value.lower.get(), MPFR_RNDU);
  if (mpfr_greater_p(result.lower.get(), result.upper.get()) != 0) {
    mpfr_swap(result.lower.get(), result.upper.get());
  }
  return result;
}

Interval Divide(const Interval& numerator, const Interval& denominator) {
  return Multiply(numerator, Reciprocal(denominator));
}

Interval Sqrt(const Interval& value) {
  if (mpfr_sgn(value.lower.get()) < 0) {
    throw std::domain_error("几何区间平方根下界为负");
  }
  Interval result;
  mpfr_sqrt(result.lower.get(), value.lower.get(), MPFR_RNDD);
  mpfr_sqrt(result.upper.get(), value.upper.get(), MPFR_RNDU);
  return result;
}

bool DefinitelyGreater(const Interval& lhs, const Interval& rhs) {
  return mpfr_greater_p(lhs.lower.get(), rhs.upper.get()) != 0;
}

bool DefinitelyGreaterEqual(const Interval& lhs, const Interval& rhs) {
  return mpfr_greaterequal_p(lhs.lower.get(), rhs.upper.get()) != 0;
}

bool IsCosineRange(const Interval& value) {
  MpReal minus_one;
  MpReal one;
  mpfr_set_si(minus_one.get(), -1, MPFR_RNDN);
  mpfr_set_ui(one.get(), 1U, MPFR_RNDN);
  return mpfr_greaterequal_p(value.lower.get(), minus_one.get()) != 0 &&
         mpfr_lessequal_p(value.upper.get(), one.get()) != 0;
}

Interval EuclideanDistance(const Point& lhs, const Point& rhs) {
  const Interval dx = Subtract(ExactInteger(lhs.integer_x), ExactInteger(rhs.integer_x));
  const Interval dy = Subtract(ExactInteger(lhs.integer_y), ExactInteger(rhs.integer_y));
  return Sqrt(Add(Square(dx), Square(dy)));
}

// e,t 均为 [0,pi] 中的角，输入是 cos(e)、cos(t)。条件 cos(e)+cos(t)>=0
// 保证 e+t<=pi，因此下面的代数式没有反三角函数和分支不确定性。
Interval CosineOfAngleSum(const Interval& cosine_first, const Interval& cosine_second) {
  const Interval one = ExactInteger(1);
  const Interval sine_first = Sqrt(Subtract(one, Square(cosine_first)));
  const Interval sine_second = Sqrt(Subtract(one, Square(cosine_second)));
  return Subtract(Multiply(cosine_first, cosine_second), Multiply(sine_first, sine_second));
}

struct PotentialBounds {
  Interval min_p;
  Interval min_q;
};

bool ComputePotentialBounds(const GraphSnapshot& graph,
                            const GeometryVerificationData& verification_data, const Edge& edge,
                            const std::int32_t node, PotentialBounds* const output,
                            std::string* const reason) {
  const auto fail = [&](const std::string& message) {
    if (reason != nullptr) {
      *reason = message;
    }
    return false;
  };
  if (node < 0 || node >= graph.dimension || node == edge.u || node == edge.v) {
    return fail("potential 点越界或等于目标端点");
  }
  const std::int64_t nearest =
      verification_data.nearest_rounded_distance[static_cast<std::size_t>(node)];
  const Interval half = ExactHalf();
  const Interval one = ExactInteger(1);
  const Interval two = ExactInteger(2);
  const Interval delta = Subtract(ExactInteger(nearest), half);
  if (!IsStrictlyPositive(delta)) {
    return fail("potential 点的最近邻半径非正");
  }

  const Point& p = graph.points[static_cast<std::size_t>(edge.u)];
  const Point& q = graph.points[static_cast<std::size_t>(edge.v)];
  const Point& r = graph.points[static_cast<std::size_t>(node)];
  const Interval lpq = ExactInteger(edge.weight);
  const Interval lpr = ExactInteger(graph.Distance(edge.u, node));
  const Interval lqr = ExactInteger(graph.Distance(edge.v, node));
  const Interval dpq = EuclideanDistance(p, q);
  const Interval dpr = EuclideanDistance(p, r);
  const Interval dqr = EuclideanDistance(q, r);
  if (!IsStrictlyPositive(dpq) || !IsStrictlyPositive(dpr) || !IsStrictlyPositive(dqr)) {
    return fail("potential 几何距离退化");
  }

  const Interval length_p = Subtract(Subtract(Add(delta, lpq), lqr), one);
  const Interval length_q = Subtract(Subtract(Add(delta, lpq), lpr), one);
  if (!IsStrictlyPositive(length_p) || !IsStrictlyPositive(length_q) ||
      !DefinitelyGreaterEqual(Add(length_p, length_q), Subtract(lpq, half))) {
    return fail("strongly-potential 条件 (13) 不成立");
  }

  const Interval gamma_numerator = Add(Subtract(Add(length_p, length_q), lpq), half);
  const Interval cos_gamma =
      Subtract(one, Divide(Square(gamma_numerator), Multiply(two, Square(delta))));
  const Interval cos_alpha_p_half =
      Divide(Subtract(Subtract(Square(length_q), Square(delta)), Square(dqr)),
             Multiply(two, Multiply(delta, dqr)));
  const Interval cos_alpha_q_half =
      Divide(Subtract(Subtract(Square(length_p), Square(delta)), Square(dpr)),
             Multiply(two, Multiply(delta, dpr)));
  if (!IsCosineRange(cos_gamma) || !IsCosineRange(cos_alpha_p_half) ||
      !IsCosineRange(cos_alpha_q_half) || !IsStrictlyPositive(cos_alpha_p_half) ||
      !IsStrictlyPositive(cos_alpha_q_half)) {
    return fail("strongly-potential 角度余弦越界");
  }
  const Interval cos_alpha_p = Subtract(Multiply(two, Square(cos_alpha_p_half)), one);
  const Interval cos_alpha_q = Subtract(Multiply(two, Square(cos_alpha_q_half)), one);
  if (!DefinitelyGreater(cos_alpha_p, cos_gamma) || !DefinitelyGreater(cos_alpha_q, cos_gamma)) {
    return fail("strongly-potential 条件 (18) 不成立");
  }

  const Interval cos_e_p = Divide(Subtract(Add(Square(dpq), Square(dpr)), Square(dqr)),
                                  Multiply(two, Multiply(dpq, dpr)));
  const Interval cos_e_q = Divide(Subtract(Add(Square(dpq), Square(dqr)), Square(dpr)),
                                  Multiply(two, Multiply(dpq, dqr)));
  const Interval left_p =
      Divide(Subtract(Add(Square(Add(dqr, delta)), Square(dpq)), Square(length_p)),
             Multiply(two, Multiply(Add(dqr, delta), dpq)));
  const Interval left_q =
      Divide(Subtract(Add(Square(Add(dpr, delta)), Square(dpq)), Square(length_q)),
             Multiply(two, Multiply(Add(dpr, delta), dpq)));
  if (!IsCosineRange(cos_e_p) || !IsCosineRange(cos_e_q) ||
      !DefinitelyGreaterEqual(cos_e_q, left_p) || !DefinitelyGreaterEqual(cos_e_p, left_q)) {
    return fail("Main Edge minima 条件 (21) 不成立");
  }

  const Interval cos_t_p = Divide(Subtract(Add(Square(length_p), Square(dpr)), Square(delta)),
                                  Multiply(two, Multiply(length_p, dpr)));
  const Interval cos_t_q = Divide(Subtract(Add(Square(length_q), Square(dqr)), Square(delta)),
                                  Multiply(two, Multiply(length_q, dqr)));
  if (!IsCosineRange(cos_t_p) || !IsCosineRange(cos_t_q) ||
      !DefinitelyGreaterEqual(Add(cos_e_p, cos_t_p), ExactInteger(0)) ||
      !DefinitelyGreaterEqual(Add(cos_e_q, cos_t_q), ExactInteger(0))) {
    return fail("Main Edge minima 的角度半平面条件不成立");
  }

  try {
    const Interval cos_sum_q = CosineOfAngleSum(cos_e_q, cos_t_q);
    const Interval cos_sum_p = CosineOfAngleSum(cos_e_p, cos_t_p);
    const Interval max_p =
        Sqrt(Subtract(Add(Square(dpq), Square(length_q)),
                      Multiply(two, Multiply(Multiply(dpq, length_q), cos_sum_q))));
    const Interval max_q =
        Sqrt(Subtract(Add(Square(dpq), Square(length_p)),
                      Multiply(two, Multiply(Multiply(dpq, length_p), cos_sum_p))));
    output->min_p = Subtract(Subtract(delta, one), max_p);
    output->min_q = Subtract(Subtract(delta, one), max_q);
  } catch (const std::domain_error&) {
    return fail("Main Edge minima 区间无法安全求平方根");
  }
  return true;
}

using WideUnsigned = unsigned __int128;

WideUnsigned SquaredCoordinateDistance(const Point& lhs, const Point& rhs) {
  const __int128 dx = static_cast<__int128>(lhs.integer_x) - rhs.integer_x;
  const __int128 dy = static_cast<__int128>(lhs.integer_y) - rhs.integer_y;
  return static_cast<WideUnsigned>(dx * dx + dy * dy);
}

class ExactKdTree {
public:
  explicit ExactKdTree(const GraphSnapshot& graph) : graph_(graph) {
    order_.resize(static_cast<std::size_t>(graph.dimension));
    std::iota(order_.begin(), order_.end(), 0);
    nodes_.reserve(order_.size());
    root_ = Build(0U, order_.size(), 0U);
  }

  [[nodiscard]] std::pair<std::int32_t, WideUnsigned> Nearest(const std::int32_t target) const {
    std::int32_t best_node = -1;
    WideUnsigned best_distance = ~WideUnsigned{0};
    Query(root_, target, &best_node, &best_distance);
    if (best_node < 0) {
      throw std::runtime_error("无法为几何证明找到最近邻");
    }
    return {best_node, best_distance};
  }

private:
  struct Node {
    std::int32_t point{-1};
    std::int32_t left{-1};
    std::int32_t right{-1};
    std::uint8_t axis{};
  };

  [[nodiscard]] std::int64_t Coordinate(const std::int32_t point, const std::uint8_t axis) const {
    const Point& value = graph_.points[static_cast<std::size_t>(point)];
    return axis == 0U ? value.integer_x : value.integer_y;
  }

  std::int32_t Build(const std::size_t begin, const std::size_t end, const std::uint8_t axis) {
    if (begin == end) {
      return -1;
    }
    const std::size_t middle = begin + (end - begin) / 2U;
    std::nth_element(order_.begin() + static_cast<std::ptrdiff_t>(begin),
                     order_.begin() + static_cast<std::ptrdiff_t>(middle),
                     order_.begin() + static_cast<std::ptrdiff_t>(end),
                     [&](const std::int32_t lhs, const std::int32_t rhs) {
                       return std::tuple{Coordinate(lhs, axis), lhs} <
                              std::tuple{Coordinate(rhs, axis), rhs};
                     });
    const std::int32_t node_index = static_cast<std::int32_t>(nodes_.size());
    nodes_.push_back(Node{order_[middle], -1, -1, axis});
    const std::uint8_t next_axis = static_cast<std::uint8_t>(1U - axis);
    const std::int32_t left = Build(begin, middle, next_axis);
    const std::int32_t right = Build(middle + 1U, end, next_axis);
    nodes_[static_cast<std::size_t>(node_index)].left = left;
    nodes_[static_cast<std::size_t>(node_index)].right = right;
    return node_index;
  }

  void Query(const std::int32_t tree_node, const std::int32_t target, std::int32_t* const best_node,
             WideUnsigned* const best_distance) const {
    if (tree_node < 0) {
      return;
    }
    const Node& node = nodes_[static_cast<std::size_t>(tree_node)];
    if (node.point != target) {
      const WideUnsigned distance =
          SquaredCoordinateDistance(graph_.points[static_cast<std::size_t>(node.point)],
                                    graph_.points[static_cast<std::size_t>(target)]);
      if (distance < *best_distance || (distance == *best_distance && node.point < *best_node)) {
        *best_distance = distance;
        *best_node = node.point;
      }
    }

    const __int128 signed_delta =
        static_cast<__int128>(Coordinate(target, node.axis)) - Coordinate(node.point, node.axis);
    const std::int32_t near_child = signed_delta <= 0 ? node.left : node.right;
    const std::int32_t far_child = signed_delta <= 0 ? node.right : node.left;
    Query(near_child, target, best_node, best_distance);
    const WideUnsigned axis_distance = static_cast<WideUnsigned>(signed_delta * signed_delta);
    if (axis_distance <= *best_distance) {
      Query(far_child, target, best_node, best_distance);
    }
  }

  const GraphSnapshot& graph_;
  std::vector<std::int32_t> order_;
  std::vector<Node> nodes_;
  std::int32_t root_{-1};
};

std::vector<std::int32_t> MidpointCandidates(const GraphSnapshot& graph, const Edge& edge,
                                             const std::uint32_t count) {
  std::vector<std::pair<long double, std::int32_t>> scored;
  scored.reserve(static_cast<std::size_t>(graph.dimension - 2));
  const Point& p = graph.points[static_cast<std::size_t>(edge.u)];
  const Point& q = graph.points[static_cast<std::size_t>(edge.v)];
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    if (node == edge.u || node == edge.v) {
      continue;
    }
    const Point& r = graph.points[static_cast<std::size_t>(node)];
    const long double dx = static_cast<long double>(2) * r.integer_x - p.integer_x - q.integer_x;
    const long double dy = static_cast<long double>(2) * r.integer_y - p.integer_y - q.integer_y;
    scored.emplace_back(dx * dx + dy * dy, node);
  }
  const std::size_t kept = std::min<std::size_t>(count, scored.size());
  std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(kept),
                    scored.end());
  std::vector<std::int32_t> result;
  result.reserve(kept);
  for (std::size_t index = 0; index < kept; ++index) {
    result.push_back(scored[index].second);
  }
  return result;
}

detail::GeometryProposalBatch
FindGeometryCandidatesCpu(const GraphSnapshot& graph,
                          const GeometryVerificationData& verification_data,
                          const GeometryOptions& options) {
  detail::GeometryProposalBatch batch;
  batch.witnesses_per_edge = options.witnesses_per_edge;
  const std::size_t slots = graph.edges.size() * options.witnesses_per_edge;
  batch.first_witness.assign(slots, -1);
  batch.second_witness.assign(slots, -1);
  batch.backend = "cpu-mpfr";
  const auto begin = std::chrono::steady_clock::now();
  for (std::size_t edge_id = 0; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    if (!edge.active || graph.Degree(edge.u) <= 2 || graph.Degree(edge.v) <= 2) {
      continue;
    }
    const std::vector<std::int32_t> candidates =
        MidpointCandidates(graph, edge, options.potential_candidates);
    std::uint32_t emitted = 0U;
    for (std::size_t first = 0; first < candidates.size() && emitted < options.witnesses_per_edge;
         ++first) {
      for (std::size_t second = first + 1U;
           second < candidates.size() && emitted < options.witnesses_per_edge; ++second) {
        Candidate candidate{static_cast<std::int32_t>(edge_id), candidates[first],
                            EliminationMethod::kGeometryMain, candidates[second]};
        if (VerifyGeometryCandidate(graph, verification_data, candidate, nullptr)) {
          const std::size_t slot = edge_id * options.witnesses_per_edge + emitted++;
          batch.first_witness[slot] = candidate.witness;
          batch.second_witness[slot] = candidate.second_witness;
        }
      }
    }
  }
  batch.kernel_ms = ElapsedMilliseconds(begin);
  return batch;
}

} // namespace

GeometryVerificationData BuildGeometryVerificationData(const GraphSnapshot& graph) {
  if (graph.distance_type != DistanceType::kEuc2D || !graph.integer_coordinates ||
      !graph.integer_distance_safe || graph.dimension < 3) {
    throw std::invalid_argument("几何 Main Edge 证明要求至少 3 个整数坐标的安全 EUC_2D 节点");
  }
  ExactKdTree tree(graph);
  GeometryVerificationData result;
  result.nearest_rounded_distance.resize(static_cast<std::size_t>(graph.dimension));
  result.nearest_node.resize(static_cast<std::size_t>(graph.dimension));
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    const auto [nearest_node, unused_distance] = tree.Nearest(node);
    static_cast<void>(unused_distance);
    result.nearest_node[static_cast<std::size_t>(node)] = nearest_node;
    result.nearest_rounded_distance[static_cast<std::size_t>(node)] =
        graph.Distance(node, nearest_node);
  }
  return result;
}

bool VerifyGeometryCandidate(const GraphSnapshot& graph,
                             const GeometryVerificationData& verification_data,
                             const Candidate& candidate, std::string* const reason) {
  const auto fail = [&](const std::string& message) {
    if (reason != nullptr) {
      *reason = message;
    }
    return false;
  };
  if (candidate.method != EliminationMethod::kGeometryMain) {
    return fail("验证器收到非几何候选");
  }
  if (verification_data.nearest_rounded_distance.size() !=
          static_cast<std::size_t>(graph.dimension) ||
      verification_data.nearest_node.size() != static_cast<std::size_t>(graph.dimension)) {
    return fail("几何最近邻验证数据维度不匹配");
  }
  if (candidate.edge_id < 0 || static_cast<std::size_t>(candidate.edge_id) >= graph.edges.size()) {
    return fail("几何候选边编号越界");
  }
  const Edge& edge = graph.edges[static_cast<std::size_t>(candidate.edge_id)];
  if (!edge.active || graph.Degree(edge.u) <= 2 || graph.Degree(edge.v) <= 2) {
    return fail("几何候选边不活动或端点度数不满足提交前提");
  }
  const std::int32_t r = candidate.witness;
  const std::int32_t s = candidate.second_witness;
  if (r < 0 || s < 0 || r >= graph.dimension || s >= graph.dimension || r == s || r == edge.u ||
      r == edge.v || s == edge.u || s == edge.v) {
    return fail("几何候选的两个 potential 点非法");
  }

  const __int128 original = static_cast<__int128>(edge.weight) + graph.Distance(r, s);
  const __int128 reconnect_first =
      static_cast<__int128>(graph.Distance(edge.u, r)) + graph.Distance(edge.v, s);
  const __int128 reconnect_second =
      static_cast<__int128>(graph.Distance(edge.u, s)) + graph.Distance(edge.v, r);
  if (reconnect_first >= original || reconnect_second >= original) {
    return fail("两个 potential 点对应的边与目标边仍兼容");
  }

  try {
    PotentialBounds r_bounds;
    PotentialBounds s_bounds;
    if (!ComputePotentialBounds(graph, verification_data, edge, r, &r_bounds, reason) ||
        !ComputePotentialBounds(graph, verification_data, edge, s, &s_bounds, reason)) {
      return false;
    }
    const Interval lpq = ExactInteger(edge.weight);
    const Interval lrs = ExactInteger(graph.Distance(r, s));
    const Interval first = Subtract(Add(Add(lpq, s_bounds.min_p), r_bounds.min_q), lrs);
    const Interval second = Subtract(Add(Add(lpq, r_bounds.min_p), s_bounds.min_q), lrs);
    if (!DefinitelyGreater(first, ExactInteger(0)) || !DefinitelyGreater(second, ExactInteger(0))) {
      return fail("Main Edge Elimination 的两个严格下界未同时为正");
    }
  } catch (const std::exception& error) {
    return fail(std::string("几何 MPFR 区间求值失败: ") + error.what());
  }
  return true;
}

GeometryEliminationResult RunGeometryElimination(GraphSnapshot* const graph,
                                                 const GeometryOptions& options) {
  if (graph == nullptr) {
    throw std::invalid_argument("几何消元图不能为空");
  }
  if (options.potential_candidates < 2U || options.potential_candidates > kMaxPotentialCandidates ||
      options.witnesses_per_edge == 0U || options.witnesses_per_edge > kMaxWitnessesPerEdge) {
    throw std::invalid_argument(
        "potential_candidates 必须位于 [2,32]，witnesses_per_edge 位于 [1,8]");
  }
  if (graph->distance_type != DistanceType::kEuc2D || !graph->integer_coordinates ||
      !graph->integer_distance_safe) {
    throw std::invalid_argument("几何消元当前只认证安全整数坐标 EUC_2D");
  }

  GeometryEliminationResult result;
  result.elimination.backend = "fgpu-geometry-cpu-verified";
  result.elimination.initial_hash = graph->ContentHash();
  result.metrics.edges_before = graph->ActiveEdgeCount();
  const std::uint64_t snapshot_hash = result.elimination.initial_hash;

  const auto nearest_begin = std::chrono::steady_clock::now();
  const GeometryVerificationData verification_data = BuildGeometryVerificationData(*graph);
  result.metrics.nearest_ms = ElapsedMilliseconds(nearest_begin);

  bool use_cuda = false;
  std::string unavailable_reason;
  if (options.backend != Backend::kCpu) {
    use_cuda = detail::GeometryCudaBackendAvailable(&unavailable_reason);
    if (options.backend == Backend::kCuda && !use_cuda) {
      throw std::runtime_error("CUDA 几何后端不可用: " + unavailable_reason);
    }
  }
  detail::GeometryProposalBatch proposals =
      use_cuda ? detail::FindGeometryCandidatesCuda(*graph, options)
               : FindGeometryCandidatesCpu(*graph, verification_data, options);
  if (proposals.witnesses_per_edge != options.witnesses_per_edge ||
      proposals.first_witness.size() != graph->edges.size() * options.witnesses_per_edge ||
      proposals.second_witness.size() != proposals.first_witness.size()) {
    throw std::runtime_error("几何候选后端返回了非法维度");
  }
  result.metrics.backend = proposals.backend;
  result.metrics.selected_device = proposals.selected_device;
  result.metrics.upload_ms = proposals.upload_ms;
  result.metrics.kernel_ms = proposals.kernel_ms;
  result.metrics.download_ms = proposals.download_ms;

  const auto verify_begin = std::chrono::steady_clock::now();
  std::vector<Candidate> verified_by_edge(graph->edges.size());
  std::vector<std::uint8_t> accepted(graph->edges.size(), 0U);
  std::vector<std::uint32_t> attempts(graph->edges.size(), 0U);
#ifdef CUDAEE_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
  for (std::int64_t edge_index = 0; edge_index < static_cast<std::int64_t>(graph->edges.size());
       ++edge_index) {
    const std::size_t edge_id = static_cast<std::size_t>(edge_index);
    for (std::uint32_t witness_index = 0; witness_index < options.witnesses_per_edge;
         ++witness_index) {
      const std::size_t slot = edge_id * options.witnesses_per_edge + witness_index;
      if (proposals.first_witness[slot] < 0 || proposals.second_witness[slot] < 0) {
        continue;
      }
      ++attempts[edge_id];
      Candidate candidate{static_cast<std::int32_t>(edge_id), proposals.first_witness[slot],
                          EliminationMethod::kGeometryMain, proposals.second_witness[slot]};
      if (VerifyGeometryCandidate(*graph, verification_data, candidate, nullptr)) {
        verified_by_edge[edge_id] = candidate;
        accepted[edge_id] = 1U;
        break;
      }
    }
  }
  std::vector<Candidate> verified;
  for (std::size_t edge_id = 0; edge_id < graph->edges.size(); ++edge_id) {
    result.metrics.proposed += attempts[edge_id];
    if (accepted[edge_id] != 0U) {
      verified.push_back(verified_by_edge[edge_id]);
    }
  }
  result.metrics.verified = verified.size();
  result.metrics.rejected = result.metrics.proposed - result.metrics.verified;
  result.metrics.verify_ms = ElapsedMilliseconds(verify_begin);

  const auto commit_begin = std::chrono::steady_clock::now();
  std::vector<Candidate> committed =
      detail::CommitVerifiedCandidates(graph, std::move(verified), snapshot_hash);
  result.metrics.committed = committed.size();
  result.metrics.commit_ms = ElapsedMilliseconds(commit_begin);
  for (const Candidate& candidate : committed) {
    const Edge& edge = graph->edges[static_cast<std::size_t>(candidate.edge_id)];
    result.elimination.proof.push_back({0U, snapshot_hash, candidate.edge_id, edge.u, edge.v,
                                        candidate.witness, candidate.method,
                                        kNoEliminationCertificate, candidate.second_witness});
  }
  result.elimination.epochs.push_back(
      {.epoch = 0U,
       .edges_before = result.metrics.edges_before,
       .proposed = result.metrics.proposed,
       .verified = result.metrics.verified,
       .rejected = result.metrics.rejected,
       .committed = result.metrics.committed,
       .snapshot_ms = result.metrics.nearest_ms,
       .propose_ms =
           result.metrics.upload_ms + result.metrics.kernel_ms + result.metrics.download_ms,
       .verify_ms = result.metrics.verify_ms,
       .commit_ms = result.metrics.commit_ms});
  result.elimination.final_hash = graph->ContentHash();
  return result;
}

std::string ToString(const NumericMode mode) {
  switch (mode) {
  case NumericMode::kMixedSafe:
    return "mixed-safe";
  case NumericMode::kFp64:
    return "fp64";
  case NumericMode::kAggressiveFp32:
    return "aggressive-fp32";
  }
  throw std::invalid_argument("未知数值模式");
}

std::string ToString(const VerificationMode mode) {
  switch (mode) {
  case VerificationMode::kEpoch:
    return "epoch";
  case VerificationMode::kDeferred:
    return "deferred";
  }
  throw std::invalid_argument("未知验证模式");
}

std::string ToString(const ProofStatus status) {
  switch (status) {
  case ProofStatus::kProved:
    return "PROVED";
  case ProofStatus::kExhausted:
    return "EXHAUSTED";
  case ProofStatus::kUnresolved:
    return "UNRESOLVED";
  case ProofStatus::kCancelled:
    return "CANCELLED";
  }
  throw std::invalid_argument("未知证明状态");
}

std::string ToString(const PdlpBackend backend) {
  switch (backend) {
  case PdlpBackend::kOff:
    return "off";
  case PdlpBackend::kNative:
    return "native";
  case PdlpBackend::kCuoptBaseline:
    return "cuopt-baseline";
  }
  throw std::invalid_argument("未知 PDLP 后端");
}

} // namespace cudaee
