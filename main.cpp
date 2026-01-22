#include <arm_neon.h>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <random>

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

int main() {
    constexpr std::size_t N = 1 << 16;
    constexpr std::size_t ITERS = 300;

    alignas(64) float xs[N]; // Cache-friendly Alignment
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(1e-3f, 1e3f);

    for (std::size_t i = 0; i < N; i++) xs[i] = dist(rng); 

    // Running benchmarks
    // ...

    return 0;
}
