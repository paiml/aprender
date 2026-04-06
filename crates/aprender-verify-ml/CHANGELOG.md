# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2025-11-25

### Added
- `AdvancedDepylerPatternGenerator` with 17 targeted patterns for edge cases:
  - Option to Path unwrapping (4 patterns)
  - Box<dyn Write> trait object polymorphism (5 patterns)
  - serde_json::Value inference for locals (4 patterns)
  - Context manager __enter__/__exit__ translation (4 patterns)
- CLI support for `--category advanced` in depyler subcommand
- Expanded `BashEnumerator` to 1000+ program patterns

### Changed
- Enhanced depyler pattern generation with file I/O detection
- Improved test coverage for advanced pattern categories

## [0.3.1] - 2025-11-24

### Added
- Comprehensive book documentation with chapters on:
  - Mutation testing strategies
  - Verification oracle design
  - ML pipeline integration
- Quality status report with coverage metrics

### Changed
- Refactored `visit_children` to reduce cognitive complexity
- Removed implementation comments for pmat SATD compliance
- Applied cargo fmt to book chapter tests

## [0.3.0] - 2025-11-23

### Added
- Initial release of Verificar synthetic data factory
- Grammar definitions for Python, Bash, C, and Ruchy
- Combinatorial program generation engine with multiple strategies:
  - Exhaustive enumeration
  - Coverage-guided (NAUTILUS-style)
  - Swarm testing
  - Boundary value emphasis
- AST mutation operators (AOR, ROR, LOR, BSR, etc.)
- Verification oracle with sandboxed execution
- Parquet output for large-scale data processing
- ML pipeline integration with aprender
- CLI tool for batch generation
- Depyler-specific pattern generators:
  - File I/O patterns
  - JSON/dict patterns
  - Context manager patterns

### Features
- `parquet` - Columnar data output
- `ml` - Machine learning pipeline
- `tree-sitter` - Grammar parsing
- `pest` - PEG grammar support
- `full` - All features enabled

## [0.2.0] - 2025-11-20 (Internal)

### Added
- Core architecture design
- Initial grammar definitions
- Generator trait system

## [0.1.0] - 2025-11-18 (Internal)

### Added
- Project scaffolding
- Basic module structure
