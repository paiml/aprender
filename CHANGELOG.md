# Changelog

All notable changes to trueno-zram will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-02-25

### Changed
- Version bump for stack release

## [0.3.0] - 2026-02-23

### Added
- Duende integration for trueno-ublk
- Benchmark regression CI workflow
- Quality infrastructure for SQI compliance

### Changed
- Switch duende dependencies to crates.io
- Migrate to PMAT v2.217.0

## [0.2.0] - 2026-02-20

### Added
- ZSTD level 1 compression with AVX-512 acceleration
- LZ4 compression support
- Entropy-based algorithm selection (trueno-zram-adaptive)
- Userspace block device daemon (trueno-ublk)
- TUI dashboard and benchmark reporting
- Tiered storage with entropy routing
