#include <arm_neon.h>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <random>
#include <string_view>

// Used to avoid Optimizations regarding a variable x
#if defined(__GNUC__) || defined(__clang__)
static inline void do_not_optimize(float x) { asm volatile("" : : "r,m"(x) : "memory"); }
#else
static inline void do_not_optimize(float x) { (void)x; }
#endif

// RSQRT Algorithm 1 using Standard Library
static inline float rsqrt_std(float x) { // 'static inline' to avoid call overhead
    return 1.0f / std::sqrtf(x);
}

// RSQRT Algorithm 2 using ARM's Neon SIMD Architecture (1 Newton Refinement)
static inline float rsqrt_neon_1(float x) {
    float32x2_t vx = vdup_n_f32(x); // 2-Lane duplicate due to SIMD type instruction
    float32x2_t y = vrsqrte_f32(vx); // Initial RSQRT estimate
    
    // Newton Step: y = y * (3 - x*y*y)/2
    // ARM provides vrsqrts(a,b) ~ (3 - a*b)/2
    float32x2_t y2 = vmul_f32(y, y); // Calculate y^2
    float32x2_t step = vrsqrts_f32(vx, y2);
    y = vmul_f32(y, step);

    return vget_lane_f32(y, 0); // Extract Lane 0
}

// RSQRT Algorithm 2 using ARM's Neon SIMD Architecture (2 Newton Refinements)
static inline float rsqrt_neon_2(float x) {
    float32x2_t vx = vdup_n_f32(x); // 2-Lane duplicate due to SIMD type instruction
    float32x2_t y = vrsqrte_f32(vx); // Initial RSQRT estimate
    
    // Newton Step: y = y * (3 - x*y*y)/2
    // ARM provides vrsqrts(a,b) ~ (3 - a*b)/2
    float32x2_t y2 = vmul_f32(y, y); // Calculate y^2
    float32x2_t step = vrsqrts_f32(vx, y2);
    y = vmul_f32(y, step);

    y2 = vmul_f32(y, y); // Calculate y^2
    step = vrsqrts_f32(vx, y2);
    y = vmul_f32(y, step);

    return vget_lane_f32(y, 0); // Extract Lane 0
}

// Benchmark Function
template <typename F> // Template to allow Argument Function being inlined correctly (no Call-Overhead)
static double bench(std::string_view name, F f, const float* xs, std::size_t n, std::size_t iters) {
    volatile float acc = 0.0f; // Avoid unwanted Optimizations

    // Warm-Up (e.g. for Cache)
    for (int k = 0; k < 2; k++) {
        for (std::size_t i = 0; i < n; i++) acc += f(xs[i]);
    }

    // Measure Time
    auto t0 = std::chrono::steady_clock::now();

    for (std::size_t k = 0; k < iters;  k++) {
        for (std::size_t i = 0; i < n; i++) acc += f(xs[i]);
    }

    auto t1 = std::chrono::steady_clock::now();

    do_not_optimize(acc); // Avoid unwanted Optimizations

    // Calculate Result
    std::chrono::duration<double> dt = t1 - t0;
    double ns_per_op = (dt.count() * 1e9) / (double(n) * double(iters)); // Time per Operation (in ns)
    std::cout << name << ": " << ns_per_op << " ns/op\n";
    return ns_per_op;
}

int main() {
    constexpr std::size_t N = 1 << 16;
    constexpr std::size_t ITERS = 300;

    alignas(64) float xs[N]; // Cache-friendly Alignment
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(1e-3f, 1e3f);

    for (std::size_t i = 0; i < N; i++) xs[i] = dist(rng); 

    // Checking results
    float x_test = 10.0f;

    std::cout << "Comparing Outputs of different Algorithms (for x=10.0)" << std::endl;
    std::cout << "rsqrtf_std: " << rsqrt_std(x_test) << std::endl;
    std::cout << "rsqrtf_neon_1: " << rsqrt_neon_1(x_test) << std::endl;
    std::cout << "rsqrtf_neon_2: " << rsqrt_neon_2(x_test) << std::endl;

    // Running benchmarks
    bench("std: 1/sqrtf", rsqrt_std, xs, N, ITERS);
    bench("std: rsqrtf (1 refine)", rsqrt_neon_1, xs, N, ITERS);
    bench("neon: rsqrtf (2 refine)", rsqrt_neon_2, xs, N, ITERS);

    return 0;
}
