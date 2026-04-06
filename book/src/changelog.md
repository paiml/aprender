# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-04

### Added

- **Core compression library** (`trueno-zram-core`)
  - LZ4 compression with AVX2/AVX-512/NEON acceleration
  - ZSTD compression with SIMD optimization
  - Runtime CPU feature detection and dispatch
  - Same-fill detection for 2048:1 zero page compression

- **GPU batch compression**
  - CUDA support via cudarc
  - Pure Rust PTX generation via trueno-gpu
  - Warp-cooperative LZ4 kernel (4 warps/block)
  - PCIe 5x rule evaluation
  - Async DMA ring buffer support

- **Kernel compatibility**
  - Sysfs interface compatible with kernel zram
  - Algorithm compatibility layer
  - Statistics matching kernel format

- **Benchmarking utilities**
  - Criterion benchmarks
  - Example programs for testing
  - Performance measurement infrastructure

### Performance

- LZ4 compression: 4.4 GB/s (AVX-512)
- LZ4 decompression: 5.4 GB/s (AVX-512)
- ZSTD compression: 11.2 GB/s (AVX-512)
- ZSTD decompression: 46 GB/s (AVX-512)
- Same-fill detection: 22 GB/s

### Infrastructure

- Published to crates.io as `trueno-zram-core`
- 461 tests passing
- 94% test coverage
- Full documentation with mdBook

## [0.2.0] - 2026-01-06

### Added

- **DT-007: Swap Deadlock Prevention**
  - mlock() integration via duende-mlock crate
  - Daemon memory pinning prevents swap deadlock
  - Works in both foreground and background modes

- **trueno-ublk daemon**
  - ublk-based block device for kernel bypass
  - Hybrid CPU/GPU architecture
  - 12.5 GB/s sequential read (fio verified)
  - 228K IOPS random 4K read

### Performance Improvements

- Sequential I/O: 16.5 GB/s (1.8x vs kernel ZRAM)
- Random 4K IOPS: 249K (4.5x vs kernel ZRAM)
- Compression ratio: 3.87x (+55% vs kernel ZRAM)
- CPU parallel decompression: 50+ GB/s

### Fixed

- Background mode mlock (DT-007e: mlock called after fork)
- Clippy warnings in GPU batch compression module

## [3.17.0] - 2026-01-07

### Added

- **VIZ-001: TruenoCollector** - Renacer visualization integration
  - Implements `renacer::Collector` trait for metrics collection
  - Feeds throughput, IOPS, tier distribution to visualization framework

- **VIZ-002: `--visualize` flag** - Real-time TUI dashboard
  - Tier heatmap, throughput gauge, entropy timeline
  - Requires `--foreground` mode

- **VIZ-003: Benchmark reports** - JSON/HTML export
  - `trueno-ublk benchmark --format json|html|text`
  - `trueno-renacer-v1` schema for ML pipelines
  - Self-contained HTML reports with tier distribution charts

- **VIZ-004: OTLP integration** - Distributed tracing
  - `--otlp-endpoint` and `--otlp-service-name` flags
  - Export traces to Jaeger/Tempo

- **Release Verification Matrix** (`docs/release_qa_checklist.md`)
  - Falsification-first QA protocol
  - Performance thresholds: >7.2 GB/s zero-page, >550K IOPS

### Performance

- **ZSTD Recommendation**: ZSTD-1 is 3x faster than LZ4 on AVX-512
  - Compress: 15.4 GiB/s (vs 5.2 GiB/s LZ4)
  - Decompress: ~10 GiB/s (vs ~1.5 GiB/s LZ4)
  - Usage: `--algorithm zstd`

### Documentation

- New book chapter: [Visualization & Observability](./ublk/visualization.md)
- Updated kernel-zram-parity roadmap (all items COMPLETE)
- New examples:
  - `visualization_demo` - TruenoCollector metrics demo
  - `zstd_vs_lz4` - Algorithm performance comparison
  - `tiered_storage` - Kernel-cooperative architecture demo

## [Unreleased]

### Planned

- trueno-zram-adaptive: ML-driven algorithm selection
- trueno-zram-generator: systemd integration
- trueno-zram-cli: zramctl replacement

---

[0.1.0]: https://github.com/paiml/trueno-zram/releases/tag/v0.1.0
[0.2.0]: https://github.com/paiml/trueno-zram/releases/tag/v0.2.0
[Unreleased]: https://github.com/paiml/trueno-zram/compare/v0.2.0...HEAD
