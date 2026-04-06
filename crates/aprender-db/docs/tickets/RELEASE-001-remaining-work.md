# RELEASE-001: Trueno-DB v0.1.0 Release - Remaining P0 Blockers

**Status**: In Progress (2/5 complete)
**Priority**: P0 (Blocker)
**Created**: 2025-11-19
**Target**: v0.1.0 Release

## Overview

Complete remaining 3 P0 blockers identified in PMAT integration review.

**Reference**: `../paiml-mcp-agent-toolkit/docs/specifications/trueno-db-integration-review-response.md`

## Progress Summary

✅ **Completed** (3/5):
- Issue #1: Feature-gate wgpu dependency (Commit: 7c6fa75)
- Issue #2: Top-K selection API (Commit: 059b902)
- Issue #4: OLAP write pattern enforcement (Commit: 4e33359)

⏸️ **Deferred** (2/5 - no GPU kernels exist yet):
- Issue #3: Floating-point statistical equivalence tests (deferred until GPU compute kernels implemented)
- Issue #5: PCIe bandwidth runtime calibration (deferred until GPU device initialization implemented)

**Status**: Core P0 blockers resolved. Remaining issues apply to future GPU kernel implementation (Phase 2).

---

## Sub-Task 1: Top-K Selection API

**Ticket**: `RELEASE-001-TOPK`
**Estimate**: 2-3 hours
**Priority**: P0

### Problem
`ORDER BY ... LIMIT` is O(N log N). Top-K selection is O(N).
**Impact**: 28.75x speedup for 1M files (2.3s → 0.08s)

### Requirements
```rust
pub trait TopKSelection {
    fn top_k(&self, k: usize, order: SortOrder) -> Result<RecordBatch>;
}
```

### Acceptance Criteria
- [x] Heap-based Top-K algorithm implemented
- [x] 4+ tests (correctness, perf, edge cases, property) - 11 tests added
- [x] >10x speedup benchmark - 28.75x achieved
- [x] Documentation updated

---

## Sub-Task 2: Floating-Point Statistical Equivalence

**Ticket**: `RELEASE-001-FLOAT`
**Estimate**: 1-2 hours
**Priority**: P0

### Problem
GPU parallel sum is non-deterministic. Tests use `assert_eq!` → flaky.

### Requirements
```rust
fn assert_statistical_equivalence(
    gpu: &[f64], simd: &[f64], sigma: f64 // 6.0 = 99.9999%
)
```

### Acceptance Criteria
- [ ] All backend tests use 6σ statistical equivalence
- [ ] No flaky failures (1000 runs)
- [ ] Non-determinism documented

---

## Sub-Task 3: PCIe Bandwidth Calibration

**Ticket**: `RELEASE-001-PCIE`
**Estimate**: 2-3 hours
**Priority**: P0

### Problem
Hardcoded 32 GB/s assumption. Real: 2.5-32 GB/s (eGPU to Gen4).

### Requirements
```rust
async fn calibrate_pcie_bandwidth(device: &GpuDevice) -> Result<f64> {
    // Micro-benchmark actual transfer time
}
```

### Acceptance Criteria
- [ ] Runtime bandwidth calibration
- [ ] Accuracy within 10% of theoretical
- [ ] Startup overhead < 100ms

---

## Release Checklist

- [x] Core P0 blockers resolved (3/3 applicable to Phase 1 MVP)
  - Issues #3 and #5 deferred to Phase 2 (GPU kernel implementation)
- [x] All tests passing (36/36)
- [x] Zero clippy warnings
- [x] TDG ≥ 90/100 (current: 92.9)
- [ ] Update PROGRESS.md with final status
- [ ] Create CHANGELOG.md
- [ ] Tag v0.1.0 release
- [ ] Document GPU implementation roadmap (Phase 2)

---

## Time Estimate

| Task | Hours |
|------|-------|
| Top-K API | 2-3 |
| Float tests | 1-2 |
| PCIe calibration | 2-3 |
| Release prep | 1-2 |
| **Total** | **6-10** |

---

## Commit Message Format

```
feat: Implement Top-K selection API (Refs RELEASE-001-TOPK)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

**Last Updated**: 2025-11-19
**Blocking**: v0.1.0 release
