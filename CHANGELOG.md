# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-04-05

### Added

- `pv codegen` now generates `contract_inv_<name>!()` macros from equation invariants
- `invariant_count` field on `GeneratedContract` for tracking invariant assertions
- CLI codegen prints invariant count alongside preconditions/postconditions
- Parent directory auto-creation for `pv codegen` default output path

### Changed

- Extracted codegen tests to `codegen_tests.rs` (codegen.rs: 1447 → 435 lines)
- Invariant macros emit `debug_assert!()` for Rust-expression invariants only
  (prose invariants are documented but skipped during codegen)


## [0.2.1] - 2026-03-31

### Added

- `pv kaizen` command: fleet-wide contract enforcement measurement across 25 repos
- Tiered scoring: kernel tier (aprender/entrenar/realizar/trueno) vs tool tier
- A-F letter grades per-repo and per-tier in kaizen output
- Workspace subcrate scanning (`crates/*/src/` + top-level members)
- Feature-gated test discovery (`#[cfg(feature = "gpu")]` → Warning in lint)
- Tool-domain postconditions: configuration, error_handling, display_format, render
- `/kaizen` Claude Code skill with five-whys root cause analysis
- 163 new tests: codegen (74), reverse_coverage (36), gates_extended (44), build_helper (50), infer (28), explain (35)

### Changed

- Fleet enforcement: Grade A (0.92), 636 call sites, 376 E2 (pre+post), 98% penetration
- 24/25 repos at Grade A enforcement
- Test coverage: 95.8% (1254 tests), up from 90.5%
- Codegen: postcondition dereference fix (`*_contract_result` for scalar comparisons)
- Codegen: first-variable substitution for `len()`/`iter()`/`is_finite()` in generic path
- 39 YAML contracts upgraded (`!var.is_empty()` → `input.len() > 0`)
- E1 classifier: added `is_empty()` and `size_of_val` pattern detection
- Spec updated to v2.5.0 with Section 31 (Kaizen Fleet Enforcement)

### Fixed

- Postcondition macro hygiene: `has_unbound_vars` filter applied to postconditions (GH #59)
- Feature-gated tests no longer fail `pv lint` verify gate (downgraded to Warning)
- Workspace repos (decy, depyler, bashrs, presentar, rmedia) now scanned in kaizen

## [0.2.0] - 2026-03-30

### Added

- Initial kaizen implementation with fleet scoring

## [0.1.1] - 2026-03-26

### Added

- 192 YAML contracts (was 107), 16,977 bindings across 13 repos
- 29 CLI commands including `fuzz`, `mirai`, `flux`, `tla`, `infer`, `unlock`
- 7 lint gates: validate, audit, score, verify, enforce, enforcement-level, reverse-coverage
- 26 lint rules with `--explain`, grouped colored output, HTML reports
- `#[must_contract]` proc macro for unannotated pub fn detection
- `EnforcementLevel` (basic/standard/strict/proven) and `locked_level` schema fields
- PVScore 10-dimension geometric mean (`pv score --pvscore`)
- Probe-level score decomposition, per-obligation verification table (`--table`)
- Issue lifecycle tracking (new/pre-existing findings via fingerprint)
- Remediation effort estimation, structured fix suggestions
- Source snippets, counterexample evidence, per-contract timing
- Reverse coverage (`pv coverage --reverse`) and `pv infer`
- Section 21: contract gap analysis (9 ML domains + shape algebra)
- Section 22: diagnostic output (13 gaps falsified against 9 tools)
- Domain contracts: bashrs (parser/classifier/encoder), depyler (type/semantic/memory)
- Gap analysis contracts: speculative-decoding, fp8, dpo-loss, bpe, paged-attention

## [0.1.0] - 2025-01-01

### Added

- Initial release of provable-contracts
- YAML contract parsing and validation
- Scaffold generation for Kani harnesses
- CLI tool (`pv`) for contract management
- Proc macro `#[contract]` attribute for compile-time enforcement
- Support for numerical stability, convergence, and monotonicity contracts
- Property-based testing with proptest
- cargo-deny integration for dependency auditing
- Clippy pedantic linting configuration

[Unreleased]: https://github.com/paiml/provable-contracts/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/paiml/provable-contracts/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/paiml/provable-contracts/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/paiml/provable-contracts/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/paiml/provable-contracts/releases/tag/v0.1.0
