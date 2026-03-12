#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

// ─────────────────────────────────────────────
//  Compiler / optimisation helpers
// ─────────────────────────────────────────────
#if defined(__GNUC__) || defined(__clang__)
static inline void sink(float x) { asm volatile("" : : "r,m"(x) : "memory"); }
static inline void clobber_memory() { asm volatile("" : : : "memory"); }
#else
static inline void sink(float x) { (void)x; }
static inline void clobber_memory() {}
#endif

// ─────────────────────────────────────────────
//  ANSI colour helpers
// ─────────────────────────────────────────────
namespace col {
static constexpr const char *reset = "\033[0m";
static constexpr const char *bold = "\033[1m";
static constexpr const char *dim = "\033[2m";
static constexpr const char *green = "\033[32m";
static constexpr const char *red = "\033[31m";
static constexpr const char *yellow = "\033[33m";
static constexpr const char *cyan = "\033[36m";
static constexpr const char *white = "\033[97m";
} // namespace col

// ─────────────────────────────────────────────
//  CPU warm-up  (~200 ms spin to reach boost clock)
// ─────────────────────────────────────────────
static void cpu_warmup() {
  using clock = std::chrono::steady_clock;
  auto end = clock::now() + std::chrono::milliseconds(200);
  volatile float x = 1.0f;
  while (clock::now() < end)
    x = 1.0f / std::sqrtf(x + 1.0001f);
  (void)x;
}

// ─────────────────────────────────────────────
//  RSQRT Implementations
// ─────────────────────────────────────────────

// [1] Standard library reference
static inline float rsqrt_std(float x) { return 1.0f / std::sqrtf(x); }

// [2] Classic Quake III Fast Inverse Square Root (id Software, 1999)
//     Magic number: 0x5f3759df  —  no Newton refinement
static inline float rsqrt_quake(float x) {
  float xhalf = 0.5f * x;
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  bits = 0x5f3759df - (bits >> 1);
  std::memcpy(&x, &bits, sizeof(x));
  return x * (1.5f - xhalf * x * x); // 1 Newton step (Quake used exactly this)
}

// [3] Quake bit-hack with improved magic number (Jan Kadlec, 2010)
//     0x5f375a86 minimises the maximum relative error before refinement
static inline float rsqrt_magic(float x) {
  float xhalf = 0.5f * x;
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  bits = 0x5f375a86 - (bits >> 1);
  std::memcpy(&x, &bits, sizeof(x));
  x = x * (1.5f - xhalf * x * x); // Newton step 1
  x = x * (1.5f - xhalf * x * x); // Newton step 2
  return x;
}

// [4] ARM NEON — scalar wrapper, 1 Newton step
static inline float rsqrt_neon_1(float x) {
  float32x2_t vx = vdup_n_f32(x);
  float32x2_t y = vrsqrte_f32(vx);
  y = vmul_f32(y, vrsqrts_f32(vx, vmul_f32(y, y)));
  return vget_lane_f32(y, 0);
}

// [5] ARM NEON — scalar wrapper, 2 Newton steps
static inline float rsqrt_neon_2(float x) {
  float32x2_t vx = vdup_n_f32(x);
  float32x2_t y = vrsqrte_f32(vx);
  y = vmul_f32(y, vrsqrts_f32(vx, vmul_f32(y, y)));
  y = vmul_f32(y, vrsqrts_f32(vx, vmul_f32(y, y)));
  return vget_lane_f32(y, 0);
}

// [6] ARM NEON — 4-wide vectorised, 1 Newton step
//     Processes 4 floats simultaneously; we sum the 4 results so the
//     compiler must compute all of them (no dead-code elimination).
static inline float rsqrt_neon_4x(float x) {
  float32x4_t vx = vdupq_n_f32(x);
  float32x4_t y = vrsqrteq_f32(vx);
  y = vmulq_f32(y, vrsqrtsq_f32(vx, vmulq_f32(y, y)));
  // Return lane 0 (all four results are identical since input was broadcast)
  return vgetq_lane_f32(y, 0);
}

// ─────────────────────────────────────────────
//  Method descriptor
// ─────────────────────────────────────────────
struct Method {
  std::string_view name;
  std::string_view short_name; // for bar chart
  float (*fn)(float);
};

// ─────────────────────────────────────────────
//  Benchmark result
// ─────────────────────────────────────────────
struct BenchResult {
  std::string_view name;
  double median_ns = 0;
  double min_ns = 0;
  double max_ns = 0;
  double stddev_ns = 0;
  double max_rel_err = 0; // vs rsqrt_std
  double mean_rel_err = 0;
};

// ─────────────────────────────────────────────
//  Single-run timing  (template keeps F inlined)
// ─────────────────────────────────────────────
template <typename F>
static double bench_once(F f, const float *xs, std::size_t n,
                         std::size_t iters) {
  float acc = 0.0f;
  // Cache warm-up (3 passes, not timed)
  for (int k = 0; k < 3; ++k)
    for (std::size_t i = 0; i < n; ++i)
      acc += f(xs[i]);
  sink(acc);

  clobber_memory();
  auto t0 = std::chrono::steady_clock::now();

  acc = 0.0f;
  for (std::size_t k = 0; k < iters; ++k)
    for (std::size_t i = 0; i < n; ++i)
      acc += f(xs[i]);

  auto t1 = std::chrono::steady_clock::now();
  sink(acc);

  std::chrono::duration<double> dt = t1 - t0;
  return (dt.count() * 1e9) / (double(n) * double(iters));
}

// ─────────────────────────────────────────────
//  Accuracy measurement
// ─────────────────────────────────────────────
template <typename F>
static void measure_accuracy(F f, const float *xs, std::size_t n,
                             double &out_max, double &out_mean) {
  double sum_err = 0.0;
  double max_err = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double ref = 1.0 / std::sqrt(double(xs[i]));
    double got = double(f(xs[i]));
    double err = std::abs(got - ref) / ref;
    if (err > max_err)
      max_err = err;
    sum_err += err;
  }
  out_max = max_err;
  out_mean = sum_err / double(n);
}

// ─────────────────────────────────────────────
//  Pretty printing helpers
// ─────────────────────────────────────────────
static void print_line(char c = '-', int n = 90) {
  std::cout << col::dim;
  for (int i = 0; i < n; ++i)
    std::cout << c;
  std::cout << col::reset << "\n";
}

static void print_header() {
  std::cout << "\n";
  std::cout << col::cyan << col::bold;
  std::cout << "  Fast Inverse Square Root — Algorithm Benchmark\n";
  std::cout << col::reset;
  std::cout << col::dim << "  Compiled: " __DATE__ " " __TIME__;
#if defined(__clang__)
  std::cout << "  |  clang " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
  std::cout << "  |  GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#endif
#if defined(__aarch64__)
  std::cout << "  |  AArch64 / Apple Silicon";
#elif defined(__x86_64__)
  std::cout << "  |  x86-64";
#endif
  std::cout << col::reset << "\n\n";
}

// ASCII bar chart (normalised to fastest)
static void print_bar_chart(const std::vector<BenchResult> &results) {
  double fastest = results[0].median_ns;
  for (auto &r : results)
    fastest = std::min(fastest, r.median_ns);

  constexpr int BAR_WIDTH = 40;
  std::cout << col::bold << "  Throughput (lower = faster)\n" << col::reset;
  print_line();

  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];
    double ratio = r.median_ns / fastest;
    int bar_len = std::min(int(ratio * BAR_WIDTH / 3.0 + 0.5), BAR_WIDTH);

    bool is_fastest = (r.median_ns == fastest);
    bool is_slowest = true;
    for (auto &rr : results)
      if (rr.median_ns < r.median_ns) {
        is_slowest = false;
        break;
      }

    // Check if this is actually the slowest
    double slowest = results[0].median_ns;
    for (auto &rr : results)
      slowest = std::max(slowest, rr.median_ns);
    is_slowest = (r.median_ns == slowest);

    const char *bar_col =
        is_fastest ? col::green : (is_slowest ? col::red : col::yellow);

    std::cout << "  " << std::left << std::setw(28) << r.name;
    std::cout << bar_col;
    for (int b = 0; b < bar_len; ++b)
      std::cout << "█";
    std::cout << col::reset;
    std::cout << col::dim << "  " << std::fixed << std::setprecision(3)
              << r.median_ns << " ns/op";
    if (ratio > 1.001)
      std::cout << "  (×" << std::setprecision(2) << ratio << " slower)";
    std::cout << col::reset << "\n";
  }
  print_line();
  std::cout << "\n";
}

// Full results table
static void print_table(const std::vector<BenchResult> &results) {
  double fastest = results[0].median_ns;
  double slowest = results[0].median_ns;
  for (auto &r : results) {
    fastest = std::min(fastest, r.median_ns);
    slowest = std::max(slowest, r.median_ns);
  }

  std::cout << col::bold << "  Detailed Results\n" << col::reset;
  print_line();

  // Header row
  std::cout << col::bold << col::dim << "  " << std::left << std::setw(28)
            << "Method" << std::right << std::setw(10) << "Median"
            << std::setw(10) << "Min" << std::setw(10) << "Max" << std::setw(10)
            << "Stddev" << std::setw(12) << "Max Err%" << std::setw(12)
            << "Mean Err%" << col::reset << "\n";
  print_line('.');

  for (const auto &r : results) {
    bool is_fastest = (r.median_ns == fastest);
    bool is_slowest = (r.median_ns == slowest);
    const char *row_col =
        is_fastest ? col::green : (is_slowest ? col::red : "");

    std::cout << row_col;
    std::cout << "  " << std::left << std::setw(28) << r.name;
    std::cout << std::right << std::fixed;
    std::cout << std::setw(9) << std::setprecision(3) << r.median_ns << " ";
    std::cout << std::setw(9) << std::setprecision(3) << r.min_ns << " ";
    std::cout << std::setw(9) << std::setprecision(3) << r.max_ns << " ";
    std::cout << std::setw(9) << std::setprecision(3) << r.stddev_ns << " ";
    // Accuracy — std has ~0 error so special-case it
    if (r.max_rel_err < 1e-9) {
      std::cout << std::setw(11) << "reference";
      std::cout << std::setw(11) << "reference";
    } else {
      std::cout << std::setw(10) << std::setprecision(4)
                << (r.max_rel_err * 100.0) << "%";
      std::cout << std::setw(10) << std::setprecision(4)
                << (r.mean_rel_err * 100.0) << "%";
    }
    std::cout << col::reset << "\n";
  }

  print_line();
  std::cout << col::dim << "  All times in ns/op.  "
            << "Errors measured relative to 1/sqrtf reference over "
            << "65536 inputs.\n"
            << col::reset << "\n";
}

// ─────────────────────────────────────────────
//  CSV export
// ─────────────────────────────────────────────
static void write_csv(const std::vector<BenchResult> &results,
                      const std::string &path) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Warning: could not write " << path << "\n";
    return;
  }
  out << "method,median_ns,min_ns,max_ns,stddev_ns,max_rel_err,mean_rel_err\n";
  for (const auto &r : results) {
    out << r.name << "," << r.median_ns << "," << r.min_ns << "," << r.max_ns
        << "," << r.stddev_ns << "," << r.max_rel_err << "," << r.mean_rel_err
        << "\n";
  }
  std::cout << col::dim << "  Results written to " << path << col::reset
            << "\n\n";
}

// ─────────────────────────────────────────────
//  Full testbench  (RUNS timed passes per method)
// ─────────────────────────────────────────────
template <typename F>
static BenchResult run_method(std::string_view name, F f, const float *xs,
                              std::size_t n, std::size_t iters, int runs) {
  std::vector<double> samples;
  samples.reserve(runs);
  for (int r = 0; r < runs; ++r)
    samples.push_back(bench_once(f, xs, n, iters));

  std::sort(samples.begin(), samples.end());

  BenchResult res;
  res.name = name;
  res.median_ns = samples[samples.size() / 2];
  res.min_ns = samples.front();
  res.max_ns = samples.back();

  double mean = 0;
  for (double v : samples)
    mean += v;
  mean /= samples.size();
  double var = 0;
  for (double v : samples)
    var += (v - mean) * (v - mean);
  res.stddev_ns = std::sqrt(var / double(samples.size()));

  measure_accuracy(f, xs, n, res.max_rel_err, res.mean_rel_err);
  return res;
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main(int argc, char **argv) {
  constexpr std::size_t N = 1 << 16; // 65 536 floats (~256 KB, fits in L2)
  constexpr std::size_t ITERS = 200; // inner loop repetitions per timed run
  constexpr int RUNS = 25;           // timed runs per method (median taken)

  // Generate random inputs in (0.001, 1000] — typical game/graphics range
  alignas(64) float xs[N];
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(1e-3f, 1e3f);
  for (std::size_t i = 0; i < N; ++i)
    xs[i] = dist(rng);

  print_header();

  // ── Warm up CPU frequency and caches ──────────────────────────────────
  std::cout << col::dim << "  Warming up CPU and caches…" << col::reset
            << std::flush;
  cpu_warmup();
  // One untimed cache pass per method to warm L2
  {
    volatile float d = 0;
    for (std::size_t i = 0; i < N; ++i)
      d += rsqrt_neon_1(xs[i]);
    for (std::size_t i = 0; i < N; ++i)
      d += rsqrt_std(xs[i]);
    (void)d;
  }
  std::cout << " done.\n\n";

  // ── Run all methods ───────────────────────────────────────────────────
  std::cout << col::dim << "  Benchmarking " << RUNS << " runs × " << ITERS
            << " iterations × " << N << " values per method…\n"
            << col::reset;
  print_line();

  std::vector<BenchResult> results;

#define RUN(label, fn)                                                         \
  std::cout << col::dim << "  " << label << "…\n" << col::reset;               \
  results.push_back(run_method(label, fn, xs, N, ITERS, RUNS));

  RUN("std: 1/sqrtf", rsqrt_std)
  RUN("Quake III (0x5f3759df)", rsqrt_quake)
  RUN("Improved magic (x5f375a86)", rsqrt_magic)
  RUN("NEON vrsqrte (1 Newton)", rsqrt_neon_1)
  RUN("NEON vrsqrte (2 Newton)", rsqrt_neon_2)
  RUN("NEON 4-wide vrsqrteq", rsqrt_neon_4x)

#undef RUN

  std::cout << "\n";

  // ── Sort results by median speed ──────────────────────────────────────
  std::sort(results.begin(), results.end(),
            [](const BenchResult &a, const BenchResult &b) {
              return a.median_ns < b.median_ns;
            });

  // ── Output ────────────────────────────────────────────────────────────
  print_bar_chart(results);
  print_table(results);

  // Resolve CSV path relative to the binary so it works regardless of cwd
  std::string csv_path = "bench.csv"; // fallback
  if (argc > 0) {
    std::string bin(argv[0]);
    auto slash = bin.rfind('/');
    std::string dir =
        (slash != std::string::npos) ? bin.substr(0, slash + 1) : "./";
    csv_path = dir + "data/bench.csv";
  }
  write_csv(results, csv_path);

  return 0;
}
