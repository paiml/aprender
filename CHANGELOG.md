# Changelog

All notable changes to simular will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SVG renderer (`SvgRenderer`) consuming `RenderCommand` to Grid Protocol SVG
- Keyframe animation export (`KeyframeRecorder`) for rmedia integration
- `simular render` CLI subcommand with `--domain`, `--format`, `--fps`, `--duration`, `--seed`
- SVG frames output (one SVG per frame) and SVG keyframes output (template + JSON)
- mdBook chapter: SVG Rendering & Keyframes
- HYPOTHESIS.md with 8 falsifiable claims and thresholds
- Nix flake for reproducible development environment
- Criterion benchmarks with 95% confidence intervals
- CONTRIBUTING.md for contributor guidelines
- ADR documentation for design decisions

### Changed
- Refactored 4 high-complexity functions to meet JPL Power of 10 standards
- Improved PMAT compliance scores across all metrics

### Fixed
- TSP GRASP test YAML indentation (3 tests with missing `name` field parse errors)
- WASM timing compatibility using web_sys::Performance
- RefCell borrow panic in WASM demo

## [0.1.0] - 2024-12-12

### Added
- Core simulation engine with EDD (Equation-Driven Development) methodology
- TSP GRASP optimization demo with WASM support
- Orbital mechanics simulation (N-body Störmer-Verlet integrator)
- Monte Carlo π estimation demo
- Harmonic oscillator physics demo
- Little's Law queueing simulation
- Kingman's formula validation
- Property-based testing with proptest
- TUI applications (orbit-tui, tsp-tui)
- WASM deployment with zero-JavaScript policy
- Chrome DevTools trace export
- Flame graph generation
- CI metrics export

### Technical Details

#### Simulation Domains
- **Physics**: Symplectic integration, energy conservation < 1e-9
- **Optimization**: GRASP with 2-opt, optimality gap < 25%
- **Monte Carlo**: Variance reduction, convergence O(1/√n)
- **Queueing**: Little's Law, Kingman's formula verification

#### Quality Standards
- Test coverage: 95%+
- Mutation coverage: 80%+
- Cyclomatic complexity: ≤15
- All code follows Toyota Production System principles

#### Dependencies
- trueno: SIMD tensor operations
- probar: E2E WASM testing
- ratatui: Terminal UI
- criterion: Benchmarking

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-12-12 | Initial release with EDD methodology |

## Migration Guide

### From Pre-release to 0.1.0

No breaking changes - this is the initial release.

## Deprecations

None currently.

## Security

No security vulnerabilities have been identified.

[Unreleased]: https://github.com/paiml/simular/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/simular/releases/tag/v0.1.0
