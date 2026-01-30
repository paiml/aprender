# Action Items (Round 5 Post-Audit)

## High Priority (P0) - Stop the Line

- [ ] **#171: APR model outputs empty tokens**
  - **Issue**: Inference runs but produces empty strings (e.g., "").
  - **Context**: Separate from #170 (explosion fixed). Likely tokenizer or conversion metadata issue.
  - **Impact**: APR GPU inference unusable.
- [ ] **#168: apr import GGUF 404 Error**
  - **Issue**: Local path resolution logic failing.
  - **Impact**: Import pipeline blocked for local files.

## Pending Infrastructure (P1)

- [x] **F-STRESS-520: Panic 411 (Empty Tensor)** ✅ PMAT-178
  - Requirement: Test loading 0-byte tensor file.
  - Fix: Added `f_stress_520_zero_byte_file_no_panic_pmat178` + truncated file tests
- [ ] **F-STRESS-521: Thread Hammer**
  - Requirement: Concurrent access stress test for `Arc<Mutex>`.

## Completed ✅

- [x] **F-GPU-501: Value Bound Check (Explosion Fixed)**
  - Root cause: Incorrect element ordering in Q4/Q6 dequantization.
  - Fix: Matched sequential layout of fused kernels.
  - Evidence: Hidden states stable (L2 ~82.0).
- [x] **AUDIT-301: Remove Expect from Hot Paths**
- [x] **F-REGR-231: APR v2 Binary Format**
- [x] **F-REGR-236: Cache Ghost**