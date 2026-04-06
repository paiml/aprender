# QA Review: Trueno-Enhanced Compute Sanitizer Tooling
**Date**: 2026-01-05
**Reviewer**: Gemini Agent
**Target**: `trueno-gpu/src/driver/sanitizer.rs` & `docs/specifications/ublk-batched-gpu-compression.md`

## 1. Executive Summary
**Grade: C+ (Promising but Brittle)**
The implementation provides the core logic for the "Trueno-Enhanced Compute Sanitizer" (Appendix B of the spec) but lacks the robustness required for a production-grade CI/CD tool (Toyota "Jidoka" principle). It is currently a "Raw Metal" prototype.

## 2. Code Quality Analysis

### 2.1 Critical Safety Gaps
- **Mutex Poisoning Risk**: The `AddressRegistry::global().lock().unwrap()` pattern is used. If a test thread panics while holding this lock (e.g., during a crash it's trying to debug), the entire test runner will panic on the next access, obscuring the original error.
    - *Recommendation*: Use `std::sync::Mutex::lock().unwrap_or_else(|e| e.into_inner())` to recover from poisoning, or switch to `parking_lot::Mutex`.

### 2.2 Performance & Efficiency
- **Inefficient PTX Parsing**: `PtxSourceMap::context_around_label` re-splits the `ptx_source` string into lines (`lines().collect()`) on *every* call. For a large kernel with many errors, this is $O(N \times E)$ overhead.
    - *Recommendation*: Parse lines once into `Vec<String>` or `Vec<&str>` (with lifetime handling) inside `PtxSourceMap::new`.

### 2.3 Parsing Robustness
- **Brittle String Slicing**: The parser relies on hardcoded offsets (e.g., `&line[at_pos + 4..]`). If a future `compute-sanitizer` version changes output format slightly (e.g., adds a space), this code will panic at runtime.
    - *Recommendation*: Use `split_once` or safer `find` logic that checks bounds before slicing. Consider adding `regex` dev-dependency for robust capturing groups.

### 2.4 API Usability
- **Leaky Abstraction**: `AddressRegistry::global()` returns `&'static Mutex<AddressRegistry>`. This forces every caller to import `Mutex` and handle locking.
    - *Recommendation*: Encapsulate locking. Provide `AddressRegistry::register_global(...)` and `AddressRegistry::lookup_global(...)` static methods that handle the lock internally.

## 3. Specification Alignment
| Spec Requirement | Implementation Status | Notes |
|------------------|-----------------------|-------|
| **Semantic Mapping** | ✅ Implemented | `AddressRegistry` handles this well. |
| **Source Tracking** | ⚠️ Partial | `PtxSourceMap` parses `.loc` but lacks file ID mapping logic. |
| **Logic Trace** | ❌ Missing | No implementation for "Logic Trace" (Thread -> Logical Token ID). |
| **Runner Integration** | ❌ Missing | The `cargo trueno-test` or `xtask` command is missing. |

## 4. Actionable Recommendations (Prioritized)

1.  **Refactor Locking**: Hide the `Mutex` inside `AddressRegistry` methods to prevent poisoning propagation and improve API ergonomics.
2.  **Optimize Parser**: Cache line splits in `PtxSourceMap` to avoid repeated allocation during error reporting.
3.  **Implement Runner**: Create `xtask/src/sanitize.rs` to wrap `cargo test` + `compute-sanitizer` as defined in Spec Appendix B.2.
4.  **Add Robustness Tests**: Add unit tests with "malformed" or "newer version" sanitizer output to ensure the parser fails gracefully (returns `None` or `Unknown`) rather than panicking.
