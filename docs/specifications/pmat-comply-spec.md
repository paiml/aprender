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

**PMAT-202:** Split `spec_checklist_tests.rs` (6141 lines)
**PMAT-203:** Split `tree/tests.rs` (3515 lines)
**PMAT-204:** Split `format/tests.rs` (3512 lines)
**PMAT-205:** Split `optim/tests.rs` (3128 lines)

### Phase 4: Core Modules (P3) - PMAT-206+

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
| PMAT-198 | mod.rs | Pending | 5906 | - | - |
| PMAT-199 | gguf.rs | Pending | 4401 | - | - |
| PMAT-200 | serve.rs | Pending | 4326 | - | - |
| PMAT-201 | showcase.rs | Pending | 3765 | - | - |

### PMAT-197 Progress (2026-01-30)
- Extracted type definitions to `converter_types.rs` (695 lines)
- Reduced `converter.rs` from 8123 to 7445 lines (-678 lines)
- All 7984 tests passing
- Backward compatible via re-exports

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
