#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#if !defined(__aarch64__) && !defined(__ARM_NEON)
#error "This benchmark currently requires an ARM target with NEON support."
#endif

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
static inline void clobber_memory() { asm volatile("" : : : "memory"); }
static inline void consume(float value) {
  asm volatile("" : : "r,m"(value) : "memory");
}
#else
#define NOINLINE
static inline void clobber_memory() {}
static inline void consume(float value) { (void)value; }
#endif

namespace style {
bool enabled = false;

const char *reset() { return enabled ? "\033[0m" : ""; }
const char *bold() { return enabled ? "\033[1m" : ""; }
const char *dim() { return enabled ? "\033[2m" : ""; }
const char *cyan() { return enabled ? "\033[36m" : ""; }
const char *green() { return enabled ? "\033[32m" : ""; }
} // namespace style

using Kernel = void (*)(const float *, float *, std::size_t);

// Each benchmarked method is a complete array kernel. Writing results to an
// output buffer avoids the serial accumulator dependency that distorted the
// original benchmark while still making every computed value observable.
static NOINLINE void kernel_std(const float *input, float *output,
                                std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    output[i] = 1.0f / std::sqrt(input[i]);
}

static inline float quake_rsqrt(float value) {
  const float half = 0.5f * value;
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits = 0x5f3759dfu - (bits >> 1);
  std::memcpy(&value, &bits, sizeof(value));
  return value * (1.5f - half * value * value);
}

static NOINLINE void kernel_quake(const float *input, float *output,
                                  std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    output[i] = quake_rsqrt(input[i]);
}

static inline float improved_magic_rsqrt(float value) {
  const float half = 0.5f * value;
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits = 0x5f375a86u - (bits >> 1);
  std::memcpy(&value, &bits, sizeof(value));
  value = value * (1.5f - half * value * value);
  value = value * (1.5f - half * value * value);
  return value;
}

static NOINLINE void kernel_improved_magic(const float *input, float *output,
                                           std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    output[i] = improved_magic_rsqrt(input[i]);
}

static inline float neon_scalar_rsqrt_1(float value) {
  const float32x2_t x = vdup_n_f32(value);
  float32x2_t y = vrsqrte_f32(x);
  y = vmul_f32(y, vrsqrts_f32(x, vmul_f32(y, y)));
  return vget_lane_f32(y, 0);
}

static inline float neon_scalar_rsqrt_2(float value) {
  const float32x2_t x = vdup_n_f32(value);
  float32x2_t y = vrsqrte_f32(x);
  y = vmul_f32(y, vrsqrts_f32(x, vmul_f32(y, y)));
  y = vmul_f32(y, vrsqrts_f32(x, vmul_f32(y, y)));
  return vget_lane_f32(y, 0);
}

static NOINLINE void kernel_neon_scalar_1(const float *input, float *output,
                                          std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    output[i] = neon_scalar_rsqrt_1(input[i]);
}

static NOINLINE void kernel_neon_scalar_2(const float *input, float *output,
                                          std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    output[i] = neon_scalar_rsqrt_2(input[i]);
}

template <int NewtonSteps>
static inline float32x4_t neon_vector_rsqrt(float32x4_t x) {
  float32x4_t y = vrsqrteq_f32(x);
  y = vmulq_f32(y, vrsqrtsq_f32(x, vmulq_f32(y, y)));
  if constexpr (NewtonSteps == 2)
    y = vmulq_f32(y, vrsqrtsq_f32(x, vmulq_f32(y, y)));
  return y;
}

template <int NewtonSteps>
static NOINLINE void kernel_neon_4wide(const float *input, float *output,
                                       std::size_t count) {
  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t x = vld1q_f32(input + i);
    vst1q_f32(output + i, neon_vector_rsqrt<NewtonSteps>(x));
  }
  for (; i < count; ++i) {
    if constexpr (NewtonSteps == 1)
      output[i] = neon_scalar_rsqrt_1(input[i]);
    else
      output[i] = neon_scalar_rsqrt_2(input[i]);
  }
}

struct Config {
  int runs = 15;
  double sample_ms = 15.0;
  bool color = true;
  std::string csv_path;
};

struct Result {
  std::string_view name;
  Kernel kernel;
  std::size_t repetitions = 1;
  std::vector<double> samples;
  double median_ns = 0.0;
  double p10_ns = 0.0;
  double p90_ns = 0.0;
  double max_error_ppm = 0.0;
  double mean_error_ppm = 0.0;

  Result(std::string_view method_name, Kernel method_kernel)
      : name(method_name), kernel(method_kernel) {}
};

static void print_usage(const char *program) {
  std::cout << "Usage: " << program << " [options]\n\n"
            << "  --runs N          Timed samples per method (default: 15)\n"
            << "  --sample-ms MS    Target duration of each sample (default: 15)\n"
            << "  --csv PATH        Also write machine-readable results\n"
            << "  --no-color        Disable ANSI colors\n"
            << "  --help            Show this help\n";
}

static Config parse_args(int argc, char **argv) {
  Config config;
#if defined(__unix__) || defined(__APPLE__)
  config.color = ::isatty(STDOUT_FILENO) != 0;
#else
  config.color = false;
#endif
  if (std::getenv("NO_COLOR") != nullptr)
    config.color = false;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--no-color") {
      config.color = false;
      continue;
    }
    if (arg == "--runs" && i + 1 < argc) {
      config.runs = std::stoi(argv[++i]);
      continue;
    }
    if (arg == "--sample-ms" && i + 1 < argc) {
      config.sample_ms = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--csv" && i + 1 < argc) {
      config.csv_path = argv[++i];
      continue;
    }
    throw std::invalid_argument("unknown or incomplete option: " +
                                std::string(arg));
  }

  if (config.runs < 3 || config.runs > 1000)
    throw std::invalid_argument("--runs must be between 3 and 1000");
  if (!(config.sample_ms >= 1.0 && config.sample_ms <= 10000.0))
    throw std::invalid_argument("--sample-ms must be between 1 and 10000");
  return config;
}

static void cpu_warmup() {
  using Clock = std::chrono::steady_clock;
  const auto end = Clock::now() + std::chrono::milliseconds(200);
  float value = 1.0001f;
  while (Clock::now() < end) {
    value = 1.0f / std::sqrt(value + 1.0001f);
    consume(value);
  }
}

static double time_kernel(Kernel kernel, const float *input, float *output,
                          std::size_t count, std::size_t repetitions) {
  using Clock = std::chrono::steady_clock;
  clobber_memory();
  const auto start = Clock::now();
  for (std::size_t i = 0; i < repetitions; ++i)
    kernel(input, output, count);
  const auto stop = Clock::now();
  clobber_memory();
  consume(output[(repetitions * 997u) % count]);

  return std::chrono::duration<double, std::nano>(stop - start).count();
}

static std::size_t calibrate(Kernel kernel, const float *input, float *output,
                             std::size_t count, double target_ns) {
  kernel(input, output, count);
  consume(output[count / 2]);

  std::size_t repetitions = 1;
  double elapsed_ns = 0.0;
  do {
    elapsed_ns = time_kernel(kernel, input, output, count, repetitions);
    if (elapsed_ns < 1.0e6)
      repetitions *= 2;
  } while (elapsed_ns < 1.0e6 && repetitions < (1u << 20));

  const double scaled =
      std::ceil(double(repetitions) * target_ns / std::max(elapsed_ns, 1.0));
  return std::max<std::size_t>(1, static_cast<std::size_t>(scaled));
}

static double percentile(const std::vector<double> &sorted, double fraction) {
  const double position = fraction * double(sorted.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
  const double weight = position - double(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

static void calculate_statistics(Result &result) {
  std::sort(result.samples.begin(), result.samples.end());
  result.median_ns = percentile(result.samples, 0.5);
  result.p10_ns = percentile(result.samples, 0.1);
  result.p90_ns = percentile(result.samples, 0.9);
}

static void measure_accuracy(Result &result, const float *input, float *output,
                             std::size_t count) {
  result.kernel(input, output, count);
  double total_relative_error = 0.0;
  double maximum_relative_error = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double reference = 1.0 / std::sqrt(static_cast<double>(input[i]));
    const double error = std::abs(static_cast<double>(output[i]) - reference) /
                         reference;
    maximum_relative_error = std::max(maximum_relative_error, error);
    total_relative_error += error;
  }
  result.max_error_ppm = maximum_relative_error * 1.0e6;
  result.mean_error_ppm = total_relative_error * 1.0e6 / double(count);
}

static void print_rule(char character = '-', int width = 108) {
  std::cout << style::dim();
  for (int i = 0; i < width; ++i)
    std::cout << character;
  std::cout << style::reset() << '\n';
}

static void print_header(std::size_t count, const Config &config) {
  std::cout << '\n'
            << style::cyan() << style::bold()
            << "  Fast Inverse Square Root\n"
            << style::reset()
            << "  ARM64 hot-cache batch-throughput benchmark\n\n"
            << style::dim() << "  " << count << " values ("
            << (count * sizeof(float) / 1024) << " KiB input)  |  "
            << config.runs << " interleaved samples  |  ~" << std::fixed
            << std::setprecision(0) << config.sample_ms << " ms/sample";
#if defined(__clang__)
  std::cout << "  |  Clang " << __clang_major__ << '.' << __clang_minor__;
#elif defined(__GNUC__)
  std::cout << "  |  GCC " << __GNUC__ << '.' << __GNUC_MINOR__;
#endif
  std::cout << style::reset() << "\n\n";
}

static void print_speed_table(const std::vector<Result> &results) {
  const double fastest = results.front().median_ns;
  constexpr int bar_width = 20;

  std::cout << style::bold() << "  Speed" << style::reset() << '\n';
  print_rule();
  std::cout << style::dim() << style::bold() << "  " << std::left
            << std::setw(3) << "#" << std::setw(34) << "Method"
            << std::setw(bar_width + 3) << "Relative throughput" << std::right
            << std::setw(11) << "ns/value" << std::setw(12) << "Gvalue/s"
            << std::setw(11) << "vs best" << std::setw(18) << "P10-P90 ns"
            << style::reset() << '\n';
  print_rule('.');

  for (std::size_t index = 0; index < results.size(); ++index) {
    const Result &result = results[index];
    const double relative_throughput = fastest / result.median_ns;
    const int length = std::max(
        1, static_cast<int>(std::lround(relative_throughput * bar_width)));
    if (index == 0)
      std::cout << style::green() << style::bold();
    std::cout << "  " << std::left << std::setw(3) << (index + 1)
              << std::setw(34) << result.name;
    for (int i = 0; i < length; ++i)
      std::cout << "█";
    for (int i = length; i < bar_width + 2; ++i)
      std::cout << ' ';
    std::cout << std::right << std::fixed << std::setprecision(3)
              << std::setw(11) << result.median_ns << std::setw(12)
              << (1.0 / result.median_ns) << std::setw(10)
              << (result.median_ns / fastest) << "x" << std::setw(9)
              << result.p10_ns << "-" << std::left << std::setw(7)
              << result.p90_ns << std::right << style::reset() << '\n';
  }
  print_rule();
  std::cout << style::dim()
            << "  Bars show throughput (longer is faster). P10-P90 shows "
               "run-to-run variation.\n"
            << style::reset() << '\n';
}

static void print_accuracy_table(const std::vector<Result> &results) {
  std::cout << style::bold() << "  Accuracy" << style::reset() << '\n';
  print_rule('-', 74);
  std::cout << style::dim() << style::bold() << "  " << std::left
            << std::setw(38) << "Method" << std::right << std::setw(17)
            << "Max error (ppm)" << std::setw(18) << "Mean error (ppm)"
            << style::reset() << '\n';
  print_rule('.', 74);
  for (const Result &result : results) {
    std::cout << "  " << std::left << std::setw(38) << result.name << std::right
              << std::fixed << std::setprecision(3) << std::setw(17)
              << result.max_error_ppm << std::setw(18)
              << result.mean_error_ppm << '\n';
  }
  print_rule('-', 74);
  std::cout << style::dim()
            << "  Reference: double-precision 1/sqrt(x). 1 ppm = 0.0001%.\n"
            << style::reset() << '\n';
}

static bool write_csv(const std::vector<Result> &results,
                      const std::string &path) {
  std::ofstream output(path);
  if (!output)
    return false;
  output << "rank,method,median_ns_per_value,p10_ns_per_value,"
            "p90_ns_per_value,gvalues_per_second,relative_to_fastest,"
            "max_error_ppm,mean_error_ppm,repetitions_per_sample\n";
  const double fastest = results.front().median_ns;
  output << std::setprecision(10);
  for (std::size_t i = 0; i < results.size(); ++i) {
    const Result &result = results[i];
    output << (i + 1) << ",\"" << result.name << "\"," << result.median_ns
           << ',' << result.p10_ns << ',' << result.p90_ns << ','
           << (1.0 / result.median_ns) << ','
           << (result.median_ns / fastest) << ',' << result.max_error_ppm << ','
           << result.mean_error_ppm << ',' << result.repetitions << '\n';
  }
  return true;
}

int main(int argc, char **argv) {
  try {
    const Config config = parse_args(argc, argv);
    style::enabled = config.color;

    constexpr std::size_t count = 1u << 16;
    std::vector<float> input(count);
    std::vector<float> output(count);
    std::mt19937 random(42);
    std::uniform_real_distribution<float> distribution(1.0e-3f, 1.0e3f);
    for (float &value : input)
      value = distribution(random);

    std::vector<Result> results = {
        {"std::sqrt + division", kernel_std},
        {"Quake III, 1 Newton", kernel_quake},
        {"Improved magic, 2 Newton", kernel_improved_magic},
        {"NEON scalar, 1 Newton", kernel_neon_scalar_1},
        {"NEON scalar, 2 Newton", kernel_neon_scalar_2},
        {"NEON 4-wide, 1 Newton", kernel_neon_4wide<1>},
        {"NEON 4-wide, 2 Newton", kernel_neon_4wide<2>},
    };

    print_header(count, config);
    std::cout << style::dim() << "  Warming up and calibrating..."
              << style::reset() << std::flush;
    cpu_warmup();
    for (Result &result : results) {
      result.repetitions =
          calibrate(result.kernel, input.data(), output.data(), count,
                    config.sample_ms * 1.0e6);
      result.samples.reserve(static_cast<std::size_t>(config.runs));
    }
    std::cout << " done\n";

    std::cout << style::dim() << "  Measuring " << results.size()
              << " methods in randomized order..." << style::reset()
              << std::flush;
    std::vector<std::size_t> order(results.size());
    std::iota(order.begin(), order.end(), 0);
    for (int run = 0; run < config.runs; ++run) {
      std::shuffle(order.begin(), order.end(), random);
      for (std::size_t index : order) {
        Result &result = results[index];
        const double elapsed_ns =
            time_kernel(result.kernel, input.data(), output.data(), count,
                        result.repetitions);
        result.samples.push_back(elapsed_ns /
                                 (double(count) * result.repetitions));
      }
    }
    std::cout << " done\n\n";

    for (Result &result : results) {
      calculate_statistics(result);
      measure_accuracy(result, input.data(), output.data(), count);
    }
    std::sort(results.begin(), results.end(),
              [](const Result &left, const Result &right) {
                return left.median_ns < right.median_ns;
              });

    print_speed_table(results);
    print_accuracy_table(results);

    if (!config.csv_path.empty()) {
      if (!write_csv(results, config.csv_path)) {
        std::cerr << "Error: could not write CSV to " << config.csv_path
                  << '\n';
        return 1;
      }
      std::cout << style::dim() << "  CSV written to " << config.csv_path
                << style::reset() << "\n\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 2;
  }
  return 0;
}
