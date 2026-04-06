# Introduction

**trueno-zram** is a SIMD and GPU-accelerated memory compression library for Linux systems. It provides userspace Rust implementations of LZ4 and ZSTD compression that leverage modern CPU vector instructions (AVX2, AVX-512, NEON) and optional CUDA GPU acceleration.

## Production Status (2026-01-06)

**MILESTONE DT-005 ACHIEVED:** trueno-zram is running as system swap!

- 8GB device active as primary swap (priority 150)
- Validated under memory pressure with ~185MB swap utilization
- CPU SIMD compress (20-30 GB/s) + CPU parallel decompress (48 GB/s)

**DT-007 COMPLETED:** Swap deadlock issue FIXED via mlock() - 211 MB daemon memory pinned.

**Known Limitations:**
- GPU compression blocked by NVIDIA F082 bug (F081 was falsified, using CPU SIMD instead)
- I/O throughput lower than kernel ZRAM (userspace overhead)

## Why trueno-zram?

trueno-zram provides **better compression efficiency** than kernel ZRAM:

| Advantage | trueno-zram | Kernel ZRAM |
|-----------|-------------|-------------|
| Compression Ratio | **3.87x** | 2.5x |
| Space Efficiency | **55% better** | baseline |
| P99 Latency | **16.5 µs** | varies |

**Trade-off:** Kernel ZRAM has higher raw I/O throughput (operates entirely in kernel space). trueno-zram uses ublk which adds userspace overhead but provides:

- **Runtime SIMD dispatch**: Automatically selects AVX-512, AVX2, or NEON based on CPU
- **Userspace flexibility**: debugging, monitoring, custom algorithms
- **Adaptive algorithm selection**: ML-driven selection based on page entropy
- **Better compression**: 3.87x vs kernel's 2.5x

## Validated Performance (QA-FALSIFY-001)

| Claim | Result | Status |
|-------|--------|--------|
| Compression ratio | **3.87x** | PASS |
| SIMD compression | **20-30 GB/s** | PASS |
| SIMD decompression | **48 GB/s** | PASS |
| P99 latency | **16.5 µs** | PASS |
| mlock (DT-007) | **211 MB** | PASS |

| ~~Falsified Claim~~ | Actual |
|---------------------|--------|
| ~~1.8x vs kernel I/O~~ | Kernel 3-13x faster |
| ~~228K IOPS~~ | 123K IOPS |

> All metrics independently verified via falsification testing (2026-01-06)

## Part of the PAIML Ecosystem

trueno-zram is part of the "Batuta Stack":

- [trueno](https://crates.io/crates/trueno) - High-performance SIMD compute library
- [trueno-gpu](https://crates.io/crates/trueno-gpu) - Pure Rust PTX generation for CUDA
- [aprender](https://crates.io/crates/aprender) - Machine learning in pure Rust
- [certeza](https://crates.io/crates/certeza) - Asymptotic test effectiveness framework

## License

trueno-zram is dual-licensed under MIT and Apache-2.0.
