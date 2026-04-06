# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Domain-specific Verus verification specs for quality gates
- Mutation testing configuration and coverage targets
- Formal verification specifications
- Quality infrastructure for SQI A- compliance

## [0.1.1] - 2026-02-15

### Added
- Scientific benchmarking framework with statistical analysis and reporting
- PMAT integration and CI/CD automation for benchmarks
- mdBook CI workflow for documentation builds
- Stack documentation search via batuta oracle RAG
- docs.rs metadata and release automation configuration
- Security audit and linting configurations

### Changed
- Updated dependencies for compatibility with PAIML Sovereign AI Stack
- Improved Makefile with cargo config optimizations

### Fixed
- Resolved clippy warnings and improved documentation formatting

## [0.1.0] - 2025-11-17

### Added
- Initial release of certeza: asymptotic test effectiveness framework
- `TruenoVec` SIMD-accelerated vector type with complete tiered testing
- Comprehensive trait implementations:
  - Core: `Index`, `IndexMut`, `Clone`, `PartialEq`, `Eq`, `Debug`
  - Collection: `From`, `FromIterator`, `Extend`
  - Ergonomic: `Deref`, `DerefMut`, `AsRef`, `AsMut`
  - Comparison: `PartialOrd`, `Ord`, `Hash`
  - Display: `Display`, `Borrow`
- Advanced TruenoVec methods and iterators
- Property-based testing infrastructure with PMAT compliance
- Integration tests and performance benchmarks
- Mutation testing achieving 97.7% mutation score
- Chaos engineering test suite
- CI/CD pipeline configuration

[Unreleased]: https://github.com/paiml/certeza/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/paiml/certeza/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/paiml/certeza/releases/tag/v0.1.0
