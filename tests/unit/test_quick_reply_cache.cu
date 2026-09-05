#include "../../src/cuda/quick_reply_cache.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {
using namespace cudaee::detail;
constexpr int kNodes = 20, kEdges = kNodes * (kNodes - 1) / 2;
struct Fixture {
  int degree[kNodes], neighbors[2 * kEdges], edge_ids[2 * kEdges];
  int u[kEdges], v[kEdges], roots[kEdges];
  std::int64_t row_offsets[kNodes + 1], pair_offsets[kNodes + 1];
  std::int64_t distances[kNodes * kNodes];
  std::uint8_t active[kEdges], fixed[kEdges], protected_edges[kEdges];
  std::uint8_t nonpairs[kNodes * (kNodes - 1) * (kNodes - 2) / 2];
};

void Require(const bool condition, const char* reason) {
  if (!condition)
    throw std::runtime_error(reason);
}

template <typename T> std::vector<T> Download(const T* input, const std::size_t count) {
  std::vector<T> result(count);
  if (count)
    CheckWorkspaceCuda(cudaMemcpy(result.data(), input, count * sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}

template <typename T> void Replace(const T* address, const T value) {
  CheckWorkspaceCuda(
      cudaMemcpy(const_cast<T*>(address), &value, sizeof(T), cudaMemcpyHostToDevice));
}

__global__ void Probe(const QuickReplyView cache, const int work, const int root, const int center,
                      const std::uint64_t snapshot, int* result) {
  *result = cache.Find(work, root, center, snapshot).valid;
}
} // namespace

int main() {
  try {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
      return 77;
    Fixture* f{};
    CheckWorkspaceCuda(cudaMallocManaged(&f, sizeof(Fixture)));
    CudaWorkspace<int> probe(0);
    probe.Reserve(1);
    QuickReplyCache cache(0);
    std::mt19937 random(0x20142023U);
    std::uint64_t checked = 0, nonempty = 0, empty = 0, rejected = 0, raw = 0, compact = 0;
    for (int sample = 0; sample < 72; ++sample) {
      *f = {};
      const int n = 3 + sample % 18, edges = n * (n - 1) / 2;
      auto edge_id = [n](int a, int b) {
        if (a > b)
          std::swap(a, b);
        return a * (2 * n - a - 1) / 2 + b - a - 1;
      };
      std::vector<std::pair<int, int>> points(n);
      for (auto& [x, y] : points) {
        x = static_cast<int>(random() % 25);
        y = static_cast<int>(random() % 25);
      }
      for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b) {
          const auto edge = edge_id(a, b);
          f->u[edge] = a;
          f->v[edge] = b;
          const auto dx = points[a].first - points[b].first;
          const auto dy = points[a].second - points[b].second;
          const double length = std::sqrt(static_cast<double>(dx * dx + dy * dy));
          std::int64_t cost =
              sample % 4 == 0 ? 0
              : sample % 4 == 1
                  ? static_cast<std::int64_t>(random() % 500)
                  : static_cast<std::int64_t>(sample % 4 == 2 ? std::floor(length + .5)
                                                              : std::ceil(length));
          if (sample >= 54)
            cost *= 1000000000000LL;
          f->distances[a * n + b] = f->distances[b * n + a] = cost;
          f->active[edge] = sample % 3 == 0 || b == a + 1 || (a == 0 && b == n - 1) || random() % 3;
          f->fixed[edge] = sample % 5 == 0 && random() % 9 == 0;
          f->protected_edges[edge] = sample % 7 == 0 && random() % 7 == 0;
          f->roots[edge] = edge;
          f->degree[a] += f->active[edge] ? 1 : 0;
          f->degree[b] += f->active[edge] ? 1 : 0;
        }
      std::shuffle(f->roots, f->roots + edges, random);
      for (int c = 0; c < n; ++c) {
        std::vector<int> neighbors;
        for (int z = 0; z < n; ++z)
          if (z != c)
            neighbors.push_back(z);
        std::sort(neighbors.begin(), neighbors.end(), [&](int a, int b) {
          return std::pair(f->distances[c * n + a], a) < std::pair(f->distances[c * n + b], b);
        });
        f->row_offsets[c + 1] = f->row_offsets[c] + n - 1;
        f->pair_offsets[c + 1] = f->pair_offsets[c] + (n - 1) * (n - 2) / 2;
        for (int j = 0; j < n - 1; ++j) {
          const auto slot = f->row_offsets[c] + j;
          f->neighbors[slot] = neighbors[j];
          f->edge_ids[slot] = edge_id(c, neighbors[j]);
        }
        for (auto pair = f->pair_offsets[c]; pair < f->pair_offsets[c + 1]; ++pair)
          f->nonpairs[pair] = sample % 3 == 1 && random() % 4 == 0;
      }
      const quick_hs::GraphView graph{.dimension = n,
                                      .degree = f->degree,
                                      .neighbors = f->neighbors,
                                      .distance = f->distances,
                                      .row_offsets = f->row_offsets,
                                      .neighbor_edge_ids = f->edge_ids,
                                      .pair_offsets = f->pair_offsets,
                                      .nonpair_mask = f->nonpairs,
                                      .edge_u = f->u,
                                      .edge_v = f->v,
                                      .edge_active = f->active,
                                      .fixed_edge = f->fixed,
                                      .edge_count = edges,
                                      .complete_graph = true};
      const int limit = sample % 2 ? 16 : 4;
      const auto generation = static_cast<std::uint64_t>(sample + 17);
      const auto metrics =
          cache.Build(graph, edges, f->roots, f->protected_edges, limit, sample % 2, generation);
      const auto view = cache.view();
      const auto rows = Download(view.rows, static_cast<std::size_t>(edges * limit));
      const auto offsets = Download(view.offsets, rows.size() + 1);
      const auto ordinals = Download(view.ordinals, view.total);
      raw += metrics.raw_pairs;
      compact += metrics.compact_pairs;
      Require(metrics.compact_pairs == ordinals.size(), "telemetry does not match actual entries");
      int fault_row = -1;
      for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto row = rows[i];
        Require(row.root == f->roots[i / limit] && row.snapshot == generation, "row identity");
        if (row.center < 0) {
          Require(offsets[i] == offsets[i + 1], "unused candidate row is not empty");
          continue;
        }
        ++checked;
        std::vector<std::uint64_t> expected;
        const int c = row.center, a = f->u[row.root], b = f->v[row.root];
        std::uint64_t ordinal = 0;
        for (auto first = f->row_offsets[c]; first < f->row_offsets[c + 1]; ++first)
          for (auto second = first + 1; second < f->row_offsets[c + 1]; ++second, ++ordinal) {
            if (!f->active[f->edge_ids[first]] || !f->active[f->edge_ids[second]] ||
                f->nonpairs[f->pair_offsets[c] + ordinal])
              continue;
            const int p = f->neighbors[first], q = f->neighbors[second];
            const auto ab = f->distances[a * n + b], pc = f->distances[p * n + c],
                       cq = f->distances[c * n + q];
            if (quick_hs::Opt22(graph, p, c, a, b, pc, ab) &&
                quick_hs::Opt22(graph, c, q, a, b, cq, ab) &&
                quick_hs::Opt23(graph, a, b, p, c, q, ab, pc, cq))
              expected.push_back(ordinal);
          }
        Require(offsets[i + 1] - offsets[i] == expected.size(), "CPU exhaustive reply count");
        Require(std::equal(expected.begin(), expected.end(), ordinals.begin() + offsets[i]),
                "CPU exhaustive reply contents/order");
        expected.empty() ? ++empty : ++nonempty;
        if (expected.size() >= 2)
          fault_row = static_cast<int>(i);
      }
      Require(!cache.Validate(graph, f->roots, generation + 1), "stale snapshot accepted");
      ++rejected;
      Require(cache.Validate(graph, f->roots, generation), "snapshot restore");
      if (fault_row >= 0) {
        const auto row = rows[fault_row];
        const auto first = offsets[fault_row], end = offsets[fault_row + 1];
        auto expect_reject = [&] {
          Require(!cache.Validate(graph, f->roots, generation), "tampered cache accepted");
          ++rejected;
          Probe<<<1, 1>>>(cache.view(), fault_row / limit, row.root, row.center, generation,
                          probe.get());
          Require(Download(probe.get(), 1)[0] == 0, "consumer ignored failed coverage");
        };
        Replace(view.ordinals + first + 1, ordinals[first]);
        expect_reject();
        Replace(view.ordinals + first + 1, ordinals[first + 1]);
        Replace(view.ordinals + first, std::numeric_limits<std::uint64_t>::max());
        expect_reject();
        Replace(view.ordinals + first, ordinals[first]);
        Replace(view.offsets + fault_row + 1, end - 1);
        expect_reject();
        Replace(view.offsets + fault_row + 1, end);
        auto bad_row = row;
        bad_row.root = (row.root + 1) % edges;
        Replace(view.rows + fault_row, bad_row);
        expect_reject();
        bad_row = row;
        bad_row.center = f->u[row.root];
        Replace(view.rows + fault_row, bad_row);
        expect_reject();
        Replace(view.rows + fault_row, row);
        Replace(view.offsets + fault_row + 1, view.total + 1);
        expect_reject();
        Replace(view.offsets + fault_row + 1, end);
        Require(cache.Validate(graph, f->roots, generation), "fault restore");
        Probe<<<1, 1>>>(cache.view(), fault_row / limit, row.root, row.center, generation + 1,
                        probe.get());
        Require(Download(probe.get(), 1)[0] == 0, "consumer accepted wrong generation");
      }
    }
    CheckWorkspaceCuda(cudaFree(f));
    Require(nonempty > 100 && empty > 100 && rejected > 100 && compact < raw, "coverage too weak");
    std::cout << "Quick reply cache: fixtures=72 rows=" << checked << " empty=" << empty
              << " nonempty=" << nonempty << " faults_rejected=" << rejected << " raw_pairs=" << raw
              << " compact_pairs=" << compact << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
