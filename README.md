# Fast Inverse Square Root: An ARM64 Benchmarking Case Study

An ARM64 case study in benchmarking inverse square root algorithms, showing how
dependency chains, compiler auto-vectorization, SIMD utilization, and numerical
accuracy affect performance conclusions.

## Motivation

Inspired by the video
[Fast Inverse Square Root — A Quake III Algorithm](https://www.youtube.com/watch?v=p8u_k2LIZyo),
I wanted to experiment with these algorithms on my own hardware. What began as
a small comparison became a useful lesson in how easily a microbenchmark can
measure something other than what its author intended.

This project is not an attempt to declare one universally fastest algorithm.
Instead, it explores what happens when historical approximations, modern ARM
instructions, compiler optimizations, and different accuracy levels meet in an
optimized batch workload.

## What is being benchmarked?

Every method calculates an approximation of:

```text
1 / sqrt(x)
```

The program generates 65,536 positive `float` inputs between `0.001` and
`1000`. Each implementation processes the complete array and writes its results
to an output array. The benchmark reports:

- hot-cache batch throughput in nanoseconds per value and billions of values
  per second;
- median timing and the P10-P90 interval across repeated samples;
- maximum and mean relative error against a double-precision reference.

The following implementations are compared:

| Implementation | Initial approximation | Refinement | Execution |
| --- | --- | ---: | --- |
| Standard C++ | `1 / std::sqrt(x)` | None | Compiler optimized |
| Quake III | Magic constant `0x5f3759df` | 1 Newton step | Compiler optimized |
| Improved magic | Magic constant `0x5f375a86` | 2 Newton steps | Compiler optimized |
| NEON scalar | ARM reciprocal-square-root estimate | 1 or 2 Newton steps | One useful result at a time |
| NEON four-wide | ARM reciprocal-square-root estimate | 1 or 2 Newton steps | Four independent results at a time |

## Scalar source code does not guarantee scalar machine code

The Quake and standard-library implementations are written as ordinary scalar
C++ loops. With `-O3`, Clang can automatically transform those loops into NEON
SIMD instructions because their iterations are independent.

This distinction is central to interpreting the result:

- the explicit NEON kernels use SIMD because the source requests it;
- the standard and Quake kernels may use SIMD because the compiler discovers
  that it is safe;
- the benchmark measures the optimized machine code, not just the appearance
  of the C++ source.

Consequently, this is a **hot-cache batch-throughput benchmark**. It does not
measure the latency of calling a scalar inverse square root function once.

## Build and run

Requirements:

- an ARM64 processor with NEON support, such as an Apple Silicon Mac;
- Clang or another C++17-compatible compiler.

```sh
mkdir -p build
clang++ -std=c++17 -O3 -mcpu=native -Wall -Wextra -Wpedantic \
  src/main.cpp -o build/bench
./build/bench
```

For a longer run with CSV output:

```sh
mkdir -p data
./build/bench --runs 25 --sample-ms 25 --csv data/bench.csv
```

Available options:

```text
--runs N          Timed samples per method (default: 15)
--sample-ms MS    Target duration of each sample (default: 15)
--csv PATH        Also write machine-readable results
--no-color        Disable ANSI colors
--help            Show all options
```

## Reading the output

The speed table contains:

- `ns/value`: median nanoseconds per input; lower is better;
- `Gvalue/s`: billions of values processed per second; higher is better;
- `vs best`: runtime relative to the fastest method in that run;
- `P10-P90`: the middle 80% of measured timings, indicating variability.

The accuracy table reports relative error in parts per million. One ppm is a
relative error of `0.0001%`.

One Newton step is generally faster but less accurate. A second step costs
additional multiplications but approaches single-precision accuracy. The
classic Quake approximation has substantially higher error and is not
guaranteed to outperform modern hardware square root and division instructions.

## Scope and limitations

Benchmark results depend on the processor, compiler, optimization flags,
temperature, power mode, and background activity. Numbers produced on one
machine should not be treated as universal rankings.

This benchmark specifically measures:

- independent single-precision inputs processed in batches;
- a reused 256 KiB input array that remains cache-hot;
- positive finite values between `0.001` and `1000`;
- optimized ARM64 code built with `-O3` and `-mcpu=native`.

It does not currently characterize:

- isolated scalar-call latency;
- cold-memory or memory-bandwidth-limited workloads;
- zero, negative values, infinities, or NaNs;
- double-precision calculations;
- performance on x86 or other non-ARM architectures.

The most important conclusion is therefore not that one algorithm always
wins. It is that benchmark structure, generated machine code, SIMD utilization,
and acceptable numerical error must all be considered before drawing a
performance conclusion.
