# Changelog

All notable changes to trueno-viz will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-03-10

### Fixed
- Use `c_char` instead of `i8` for AMD GPU FFI buffers — fixes aarch64 cross-compilation (Refs #21)

## [0.2.1] - 2026-02-26

### Fixed
- Updated trueno dependency to 0.16.0 for stack compatibility

## [0.2.0] - 2026-02-26

### Changed
- Major version bump for PAIML Sovereign AI Stack coordinated release
- Code quality improvements and complexity refactoring

## [0.1.27] - 2026-02-25

### Fixed
- Merge duplicate rustdoc-args keys in Cargo.toml

### Changed
- Bump trueno-graph dependency (default-features=false)

## [0.1.26] - 2026-02-24

### Changed
- Update trueno-graph to 0.1.16
- Migrate serde_yaml to serde_yaml_ng
- Update trueno to 0.15

## [0.1.25] - 2026-02-22

### Changed
- Delegate math/truncate_str to batuta-common
- Switch batuta-common to crates.io v0.1.0

## [0.1.14] - 2026-02-10

### Added
- Real SIMD collectors (SSE2/AVX2/NEON platform intrinsics)
- 5.6x measured speedup for byte scanning operations
- Three-tier storage: hot (ring buffer) / warm (LZ4) / cold (disk)
