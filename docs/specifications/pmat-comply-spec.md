# PMAT Compliance Specification

**Version:** 1.0.0
**Status:** Active
**Created:** 2026-01-30
**Target:** File Health Grade A (all files <2000 lines)

## Overview

This specification tracks technical debt related to file size and implements a systematic remediation plan using PMAT work methodology with Five-Whys root cause analysis and Toyota Way principles.

## Current State

**TDG Score:** 97.5/100 (A+)
**File Health:** Grade D (24+ files >2000 lines)
**Test Coverage:** 96.16%
**Target Coverage:** ≥95% (must maintain during refactoring)

## Problem Files (>2000 lines)

| Priority | Lines | File | Category | Target |
|---------:|------:|------|----------|--------|
| P0 | 8123 | `src/format/converter.rs` | Format core | <1500 |
| P0 | 6141 | `tests/spec_checklist_tests.rs` | Tests | <2000 |
| P0 | 5906 | `src/format/mod.rs` | Format core | <1500 |
| P1 | 4401 | `src/format/gguf.rs` | Format core | <1500 |
| P1 | 4326 | `crates/apr-cli/src/commands/serve.rs` | CLI | <1500 |
| P1 | 3765 | `crates/apr-cli/src/commands/showcase.rs` | CLI | <1500 |
| P2 | 3515 | `src/tree/tests.rs` | Tests | <2000 |
| P2 | 3512 | `src/format/tests.rs` | Tests | <2000 |
| P2 | 3128 | `src/optim/tests.rs` | Tests | <2000 |
| P2 | 2951 | `src/models/qwen2/mod.rs` | Models | <1500 |
| P2 | 2923 | `src/text/tokenize.rs` | Text | <1500 |
| P2 | 2918 | `src/format/rosetta.rs` | Format | <1500 |
| P3 | 2803 | `src/tree/mod.rs` | ML algo | <1500 |
| P3 | 2760 | `src/cluster/tests.rs` | Tests | <2000 |
| P3 | 2742 | `crates/aprender-shell/src/main.rs` | Shell | <1500 |
| P3 | 2684 | `src/nn/transformer.rs` | Neural | <1500 |
| P3 | 2645 | `src/graph/tests.rs` | Tests | <2000 |
| P3 | 2592 | `src/cluster/mod.rs` | ML algo | <1500 |
| P3 | 2583 | `src/format/v2.rs` | Format | <1500 |
| P3 | 2449 | `src/automl/search.rs` | AutoML | <1500 |
| P3 | 2446 | `crates/apr-cli/src/commands/cbtop.rs` | CLI | <1500 |
| P3 | 2445 | `crates/apr-cli/src/commands/run.rs` | CLI | <1500 |
| P3 | 2335 | `src/text/chat_template.rs` | Text | <1500 |
| P3 | 2311 | `examples/qa_run.rs` | Examples | <1500 |
| P3 | 2242 | `src/citl/neural.rs` | CITL | <1500 |
| P3 | 2231 | `crates/apr-cli/src/lib.rs` | CLI | <1500 |
| P3 | 2230 | `src/graph/mod.rs` | Graph | <1500 |
| P3 | 2216 | `src/bench_viz.rs` | Benchmarks | <1500 |
| P3 | 2108 | `src/format/rosetta_ml.rs` | Format | <1500 |
| P3 | 2037 | `src/format/validation.rs` | Format | <1500 |

**Total:** 30 files >2000 lines

## Five-Whys Root Cause Analysis

1. **WHY are files >2000 lines?** → Organic growth without size limits
2. **WHY no size limits?** → No automated enforcement in CI
3. **WHY no CI enforcement?** → PMAT file-health gate not configured as blocking
4. **WHY not blocking?** → Legacy code predates PMAT adoption
5. **ROOT CAUSE:** Missing file size enforcement + lack of module decomposition strategy

## Toyota Way Principles Applied

- **Jidoka (自働化):** Stop and fix - split files before adding more code
- **Kaizen (改善):** Continuous improvement - reduce file sizes incrementally
- **Mieruka (見える化):** Visualization - track progress in this spec
- **Genchi Genbutsu (現地現物):** Go see - analyze each file's structure before splitting

## Remediation Strategy

### Phase 1: Format Module (P0) - PMAT-197 to PMAT-199

The `src/format/` module is the worst offender with 5 files >2000 lines.

**PMAT-197:** Split `converter.rs` (8123 lines)
- Extract: `converter_gguf.rs` (GGUF conversion logic)
- Extract: `converter_safetensors.rs` (SafeTensors logic)
- Extract: `converter_quantize.rs` (Quantization logic)
- Extract: `converter_merge.rs` (Model merge logic)

**PMAT-198:** Split `mod.rs` (5906 lines)
- Extract: `format_types.rs` (Type definitions)
- Extract: `format_io.rs` (I/O operations)
- Keep: Core re-exports only

**PMAT-199:** Split `gguf.rs` (4401 lines)
- Extract: `gguf_reader.rs` (Reading logic)
- Extract: `gguf_writer.rs` (Writing logic)
- Extract: `gguf_types.rs` (GGUF type definitions)

### Phase 2: CLI Commands (P1) - PMAT-200 to PMAT-201

**PMAT-200:** Split `serve.rs` (4326 lines)
- Extract: `serve_routes.rs` (HTTP routes)
- Extract: `serve_handlers.rs` (Request handlers)
- Extract: `serve_middleware.rs` (Middleware)

**PMAT-201:** Split `showcase.rs` (3765 lines)
- Extract: `showcase_demos.rs` (Demo implementations)
- Extract: `showcase_ui.rs` (UI components)

### Phase 3: Test Files (P2) - PMAT-202 to PMAT-205

Test files can exceed 2000 lines with less penalty, but splitting improves maintainability.

**PMAT-202:** Split `spec_checklist_tests.rs` (6141 lines) ✅
**PMAT-203:** Split `tree/tests.rs` (3515 lines) ✅
**PMAT-204:** Split `format/tests.rs` (3512 lines) ✅
**PMAT-205:** Split `optim/tests.rs` (3430 lines) ✅
**PMAT-206:** Split `cluster/tests.rs` (2760 lines) ✅
**PMAT-207:** Split `graph/tests.rs` (2645 lines) ✅

### Phase 4: Core Modules (P3) - PMAT-208+

Remaining files to be split after P0-P2 complete.

## Splitting Rules

1. **Maintain backward compatibility:** Re-export from original module
2. **Preserve test coverage:** Move tests with code
3. **Atomic commits:** One logical split per commit
4. **Verify after each split:**
   ```bash
   cargo test
   cargo llvm-cov --summary-only  # Must stay ≥95%
   ```

## Progress Tracking

| Work Item | File | Status | Before | After | Coverage |
|-----------|------|--------|--------|-------|----------|
| PMAT-197 | converter.rs | ✅ Partial | 8123 | 7445 | 96%+ |
| PMAT-197 | converter_types.rs | ✅ Created | - | 695 | 96%+ |
| PMAT-198 | mod.rs | ✅ Done | 5909 | 3795 | 96%+ |
| PMAT-198 | types.rs | ✅ Created | - | 703 | 96%+ |
| PMAT-198 | core_io.rs | ✅ Created | - | 788 | 96%+ |
| PMAT-198 | signing.rs | ✅ Created | - | 195 | 96%+ |
| PMAT-198 | encryption.rs | ✅ Created | - | 521 | 96%+ |
| PMAT-199 | gguf/mod.rs | ✅ Done | 4401 | 2259 | 96%+ |
| PMAT-199 | gguf/types.rs | ✅ Created | - | 448 | 96%+ |
| PMAT-199 | gguf/reader.rs | ✅ Created | - | 837 | 96%+ |
| PMAT-199 | gguf/dequant.rs | ✅ Created | - | 691 | 96%+ |
| PMAT-199 | gguf/api.rs | ✅ Created | - | 210 | 96%+ |
| PMAT-202 | spec_checklist_tests.rs | ✅ Done | 6141 | 18 files | 96%+ |
| PMAT-200 | serve/mod.rs | ✅ Done | 4351 | 78 | 96%+ |
| PMAT-200 | serve/types.rs | ✅ Created | - | 769 | 96%+ |
| PMAT-200 | serve/routes.rs | ✅ Created | - | 353 | 96%+ |
| PMAT-200 | serve/handlers.rs | ✅ Created | - | 1397 | 96%+ |
| PMAT-200 | serve/safetensors.rs | ✅ Created | - | 779 | 96%+ |
| PMAT-200 | serve/tests.rs | ✅ Created | - | 1062 | 96%+ |
| PMAT-201 | showcase/mod.rs | ✅ Done | 3765 | 99 | 96%+ |
| PMAT-201 | showcase/types.rs | ✅ Created | - | 275 | 96%+ |
| PMAT-201 | showcase/pipeline.rs | ✅ Created | - | 503 | 96%+ |
| PMAT-201 | showcase/benchmark.rs | ✅ Created | - | 644 | 96%+ |
| PMAT-201 | showcase/demo.rs | ✅ Created | - | 1181 | 96%+ |
| PMAT-201 | showcase/validation.rs | ✅ Created | - | 230 | 96%+ |
| PMAT-201 | showcase/tests.rs | ✅ Created | - | 878 | 96%+ |
| PMAT-203 | tree/tests/mod.rs | ✅ Done | 3515 | 2 | 96%+ |
| PMAT-203 | tree/tests/core.rs | ✅ Created | - | 1595 | 96%+ |
| PMAT-203 | tree/tests/advanced.rs | ✅ Created | - | 1925 | 96%+ |
| PMAT-204 | format/tests/ | ✅ Done | 3512 | 11 files | 96%+ |
| PMAT-205 | optim/tests/mod.rs | ✅ Done | 3430 | 2 | 96%+ |
| PMAT-205 | optim/tests/core.rs | ✅ Created | - | 1635 | 96%+ |
| PMAT-205 | optim/tests/advanced.rs | ✅ Created | - | 1803 | 96%+ |
| PMAT-206 | cluster/tests/mod.rs | ✅ Done | 2760 | 2 | 96%+ |
| PMAT-206 | cluster/tests/core.rs | ✅ Created | - | 1009 | 96%+ |
| PMAT-206 | cluster/tests/advanced.rs | ✅ Created | - | 1756 | 96%+ |
| PMAT-207 | graph/tests/mod.rs | ✅ Done | 2645 | 2 | 96%+ |
| PMAT-207 | graph/tests/core.rs | ✅ Created | - | 1368 | 96%+ |
| PMAT-207 | graph/tests/advanced.rs | ✅ Created | - | 1281 | 96%+ |
| PMAT-208 | converter/tests/mod.rs | ✅ Done | 3603 | 13 | 96%+ |
| PMAT-208 | converter/tests/core.rs | ✅ Created | - | 907 | 96%+ |
| PMAT-208 | converter/tests/errors.rs | ✅ Created | - | 1195 | 96%+ |
| PMAT-208 | converter/tests/coverage.rs | ✅ Created | - | 880 | 96%+ |
| PMAT-208 | converter/tests/pmat.rs | ✅ Created | - | 656 | 96%+ |
| PMAT-209 | gguf/tests.rs | ✅ Created | 2259 | 2212 | 96%+ |
| PMAT-209 | gguf/mod.rs | ✅ Done | 2259 | 51 | 96%+ |
| PMAT-210 | validation.rs | ✅ Done | 2036 | 1102 | 96%+ |
| PMAT-210 | validation_tests.rs | ✅ Created | - | 944 | 96%+ |

### PMAT-197 Progress (2026-01-30)
- Extracted type definitions to `converter_types.rs` (695 lines)
- Reduced `converter.rs` from 8123 to 7445 lines (-678 lines)
- All 7984 tests passing
- Backward compatible via re-exports

### PMAT-198 Progress (2026-01-30)
- Extracted type definitions to `types.rs` (703 lines)
- Extracted core I/O (save/load/inspect) to `core_io.rs` (788 lines)
- Extracted signing functions to `signing.rs` (195 lines)
- Extracted encryption functions to `encryption.rs` (521 lines)
- Reduced `mod.rs` from 5909 to 3795 lines (-2114 lines, source code is 270 lines)
- Remaining 3525 lines are tests (to be split in Phase 3)
- All 8135 tests passing
- Backward compatible via re-exports

### PMAT-199 Progress (2026-01-30)
- Converted `gguf.rs` to directory module `gguf/`
- Extracted type definitions + write ops to `types.rs` (448 lines)
- Extracted binary reader + `GgufReader` to `reader.rs` (837 lines)
- Extracted dequantization kernels to `dequant.rs` (691 lines)
- Extracted high-level API to `api.rs` (210 lines)
- Reduced `gguf/mod.rs` from 4401 to 2259 lines (48 header + 2211 tests)
- All source submodules <1000 lines
- All 8135 tests passing (197 gguf-specific)
- Backward compatible via `pub use` re-exports

### PMAT-202 Progress (2026-01-30)
- Split monolithic `spec_checklist_tests.rs` (6141 lines) into 18 section-based files
- All files under 870 lines (largest: `spec_checklist_j_profiling.rs` at 870 lines)
- 237 total tests: 232 passed, 5 ignored (pre-existing `#[ignore]`), 0 failed
- All 8135 library tests still passing
- Sections: A (model loading), B (tokenization), C (forward pass), D (generation),
  E (visual control), F (WASM), G (code quality), H (lifecycle), I (probador),
  J (profiling), T (realizar), X (anti-stub), U (performance), V (sovereign),
  W (advanced perf), Q (Qwen coder), R (model import), 19 (inference)

### PMAT-200 Progress (2026-01-30)
- Converted monolithic `serve.rs` (4351 lines) to directory module `serve/`
- Extracted type definitions to `types.rs` (769 lines)
- Extracted HTTP routes/middleware to `routes.rs` (353 lines)
- Extracted APR/GGUF handlers to `handlers.rs` (1397 lines)
- Extracted SafeTensors handlers to `safetensors.rs` (779 lines)
- Extracted tests to `tests.rs` (1062 lines)
- Entry point `mod.rs` reduced to 78 lines
- All source submodules <1500 lines, tests <1100 lines
- Deduplicated SafeTensorsTokenizerInfo (canonical in safetensors.rs)
- 60 serve tests passing, 8135 library tests passing
- Backward compatible via `pub use` re-exports

### PMAT-201 Progress (2026-01-30)
- Converted monolithic `showcase.rs` (3765 lines) to directory module `showcase/`
- Extracted type definitions to `types.rs` (275 lines)
- Extracted pipeline steps (import, gguf, convert, apr) to `pipeline.rs` (503 lines)
- Extracted benchmark logic to `benchmark.rs` (644 lines)
- Extracted demo modules (visualize, chat, zram, cuda, brick) to `demo.rs` (1181 lines)
- Extracted validation/summary functions to `validation.rs` (230 lines)
- Extracted tests to `tests.rs` (878 lines)
- Entry point `mod.rs` reduced to 99 lines
- All source submodules <1200 lines, tests <900 lines
- 42 showcase tests passing
- Backward compatible via `pub use types::*` re-exports

### PMAT-203 Progress (2026-01-30)
- Converted `tree/tests.rs` (3515 lines) to directory module `tree/tests/`
- Split into `core.rs` (1595 lines): core DS, Gini, DT training, RF classifier, GB, regression tree
- Split into `advanced.rs` (1925 lines): RF regressor, OOB, feature importances, coverage, helpers
- All 206 tree tests passing
- All submodules <2000 lines

### PMAT-204 Progress (2026-01-30)
- Format tests already split in prior session to `format/tests/` directory (11 files)
- Largest file: `unit.rs` at 1785 lines (under 2000 target)
- Property tests split into: proptests.rs, encryption, signing, metadata, license, integration, error, distillation, x25519

### PMAT-205 Progress (2026-01-30)
- Converted `optim/tests.rs` (3430 lines) to directory module `optim/tests/`
- Split into `core.rs` (1635 lines): SafeCholesky, SGD, Adam, line search, L-BFGS, CG, Damped Newton
- Split into `advanced.rs` (1803 lines): proximal operators, FISTA, ADMM, coordinate descent, projected GD
- All 269 optim tests passing (1 ignored)
- All submodules <2000 lines

### PMAT-206 Progress (2026-01-30)
- Converted `cluster/tests.rs` (2760 lines) to directory module `cluster/tests/`
- Split into `core.rs` (1009 lines): KMeans (all variants, convergence, mutation tests)
- Split into `advanced.rs` (1756 lines): DBSCAN, Agglomerative, GMM, Isolation Forest, LOF, Spectral
- All 140 cluster tests passing
- All submodules <2000 lines

### PMAT-207 Progress (2026-01-30)
- Converted `graph/tests.rs` (2645 lines) to directory module `graph/tests/`
- Split into `core.rs` (1368 lines): construction, centralities, shortest path, Dijkstra
- Split into `advanced.rs` (1281 lines): all-pairs SP, A*, DFS, components, label propagation
- All 248 graph tests passing
- All submodules <1400 lines

### PMAT-208 Progress (2026-01-31)
- Converted `converter/tests.rs` (3603 lines) to directory module `converter/tests/`
- Split into `core.rs` (907 lines): source parsing, name mapping, tensor expectations, conversion
- Split into `errors.rs` (1195 lines): import errors, coverage boost part 1
- Split into `coverage.rs` (880 lines): coverage boost part 2, internal helpers
- Split into `pmat.rs` (656 lines): PMAT-107, GH-165, GH-164, GH-185, GH-190 regression tests
- All 260 converter tests passing
- All submodules <1200 lines

### PMAT-209 Progress (2026-01-31)
- Extracted tests from `gguf/mod.rs` (2259 lines) to `gguf/tests.rs`
- `gguf/mod.rs` reduced to 51 lines (re-exports only)
- `gguf/tests.rs` contains 2212 lines (197 tests including proptests)
- All 197 gguf tests passing

### PMAT-210 Progress (2026-01-31)
- Extracted tests from `validation.rs` (2036 lines) to `validation_tests.rs`
- `validation.rs` reduced to 1102 lines (source only)
- `validation_tests.rs` contains 944 lines (67 tests)
- All 67 validation tests passing
- All 8135 library tests passing

## Falsification Gates

- **F-COMPLY-001:** All P0 files <1500 lines
- **F-COMPLY-002:** All P1 files <1500 lines
- **F-COMPLY-003:** All test files <2000 lines
- **F-COMPLY-004:** Coverage remains ≥95% after each split
- **F-COMPLY-005:** `pmat comply check` passes (file health ≥ grade B)
- **F-COMPLY-006:** All tests pass after refactoring
- **F-COMPLY-007:** No public API changes (backward compatible)

## Acceptance Criteria

1. All 30 problem files reduced to target size
2. Test coverage ≥95% maintained
3. Individual file coverage improved (not just maintained)
4. `pmat comply check` shows file health grade B or better
5. All existing tests pass
6. No breaking API changes

## References

- PMAT Documentation: https://github.com/paiml/pmat
- Toyota Way: Continuous Improvement Principles
- Rust API Guidelines: https://rust-lang.github.io/api-guidelines/
