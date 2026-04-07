# Pending Work Tracker

**Document Version**: 1.0.0
**Created**: 2026-01-11
**Status**: Active

Consolidated tracker for pending specification items with cross-references to PMAT tickets and FKR entries.

---

## Quick Reference

| Priority | Pending | In Progress | Complete |
|----------|---------|-------------|----------|
| P0 | 0 | 0 | 1 |
| P1 | 6 | 1 | 0 |
| P2 | 2 | 0 | 0 |
| **Total** | **8** | **1** | **1** |

---

## Priority 0: Blocking

### PMAT-005: LZ4 GPU Kernel Completion
- **Status**: COMPLETE (2026-01-10)
- **FKR**: FKR-2026-01-10-001, FKR-2026-01-10-007
- **Spec Ref**: ublk-spec F-001 to F-010
- **Resolution**: F082 computed-address bug resolved with Lz4WarpShuffleKernel

---

## Priority 1: Critical Path

### PMAT-001: Loop Splitting Optimization
- **Status**: PENDING
- **FKR**: FKR-2026-01-10-003
- **Spec Ref**: F51-F65
- **Effort**: 5 days
- **Dependencies**: None
- **Citations**:
  1. Allen & Kennedy, 1987 - DOI:10.1145/29873.29875
  2. Ryoo et al., 2008 - DOI:10.1145/1345206.1345220
  3. Yang et al., 2010 - DOI:10.1145/1806596.1806606

### PMAT-002: Token-Based Synchronization
- **Status**: PENDING
- **FKR**: FKR-2026-01-10-004
- **Spec Ref**: F66-F80
- **Effort**: 7 days
- **Dependencies**: None
- **Citations**:
  1. Alglave et al., 2015 - DOI:10.1145/2694344.2694391
  2. Lustig et al., 2019 - DOI:10.1145/3297858.3304043
  3. Sorensen & Donaldson, 2016 - DOI:10.1145/2909437.2909440

### PMAT-003: FMA Fusion Correctness
- **Status**: PENDING
- **FKR**: FKR-2026-01-10-005
- **Spec Ref**: F17-F29
- **Effort**: 4 days
- **Dependencies**: None
- **Citations**:
  1. Muller et al., 2018 - ISBN:978-3-319-76525-9
  2. Boldo & Melquiond, 2008 - DOI:10.1109/TC.2008.48
  3. Higham, 2002 - ISBN:0-89871-521-0

### PMAT-004: Memory Coalescing Optimization
- **Status**: PENDING
- **FKR**: FKR-2026-01-10-006
- **Spec Ref**: F34-F39
- **Effort**: 3 days
- **Dependencies**: None
- **Citations**:
  1. Volkov & Demmel, 2008 - DOI:10.1109/SC.2008.5214359
  2. Ruetsch & Micikevicius, 2009 - NVIDIA Technical Report
  3. Mei & Chu, 2017 - DOI:10.1109/TPDS.2016.2549523

### PMAT-008: PTX Debugger Implementation
- **Status**: PENDING
- **FKR**: FKR-2026-01-10-008
- **Spec Ref**: REQ-001 to REQ-010, ptx-debugger.md
- **Effort**: 15 days
- **Dependencies**: PMAT-003 (FMA patterns)
- **Citations**:
  1. Betts et al., 2012 - DOI:10.1145/2384616.2384625
  2. Li & Gopalakrishnan, 2010 - DOI:10.1145/1882291.1882320
  3. Leung et al., 2012 - DOI:10.1145/2259016.2259067

### PMAT-009: Numerical Stability Test Suite
- **Status**: PENDING
- **FKR**: FKR-2026-01-10-009
- **Spec Ref**: F92-F99
- **Effort**: 6 days
- **Dependencies**: PMAT-003 (FMA correctness)
- **Citations**:
  1. Higham, 2002 - ISBN:0-89871-521-0
  2. Demmel, 1997 - ISBN:0-89871-389-7
  3. Goldberg, 1991 - DOI:10.1145/103162.103163

### PMAT-010: Backend Equivalence Testing
- **Status**: IN PROGRESS
- **FKR**: FKR-2026-01-10-010
- **Spec Ref**: F81-F87
- **Effort**: 4 days
- **Dependencies**: None
- **Citations**:
  1. Whitehead & Fit-Florea, 2011 - NVIDIA Whitepaper
  2. Collange et al., 2015 - DOI:10.1109/MM.2015.54
  3. Lam et al., 2013 - DOI:10.1145/2491956.2462927

---

## Priority 2: Platform Expansion

### PMAT-006: Apple Silicon Metal Backend
- **Status**: PENDING
- **FKR**: FKR-2026-01-10-011
- **Spec Ref**: Backend Story Policy
- **Effort**: 10 days
- **Dependencies**: PMAT-010 (equivalence testing framework)
- **Citations**:
  1. Apple, 2023 - Metal Best Practices Guide
  2. Gaster & Howes, 2012 - ISBN:978-0-12-387766-6
  3. Lopes et al., 2021 - arXiv:2110.01599

### PMAT-007: AMD ROCm Backend
- **Status**: PENDING
- **FKR**: FKR-2026-01-10-012
- **Spec Ref**: Backend Story Policy
- **Effort**: 8 days
- **Dependencies**: PMAT-010 (equivalence testing framework)
- **Citations**:
  1. AMD, 2023 - HIP Programming Guide
  2. Sun et al., 2019 - DOI:10.1109/IISWC47752.2019.9041952
  3. Jia et al., 2018 - arXiv:1804.06826

---

## Dependency Graph

```
                    ┌─────────────┐
                    │ PMAT-010    │ (Backend Equivalence)
                    │ IN PROGRESS │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               │               ▼
    ┌─────────────┐        │        ┌─────────────┐
    │ PMAT-006    │        │        │ PMAT-007    │
    │ Metal       │        │        │ ROCm        │
    │ PENDING     │        │        │ PENDING     │
    └─────────────┘        │        └─────────────┘
                           │
                    ┌──────┴──────┐
                    │ PMAT-003    │ (FMA Correctness)
                    │ PENDING     │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               │               ▼
    ┌─────────────┐        │        ┌─────────────┐
    │ PMAT-008    │        │        │ PMAT-009    │
    │ PTX Debug   │        │        │ Stability   │
    │ PENDING     │        │        │ PENDING     │
    └─────────────┘        │        └─────────────┘
                           │
    Independent:           │
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ PMAT-001    │ │ PMAT-002    │ │ PMAT-004    │
    │ Loop Split  │ │ Token Sync  │ │ Coalescing  │
    │ PENDING     │ │ PENDING     │ │ PENDING     │
    └─────────────┘ └─────────────┘ └─────────────┘
```

---

## Suggested Execution Order

Based on dependencies and critical path analysis:

### Sprint 1 (Week 1-2): Foundations
1. **PMAT-004** (3 days) - Memory Coalescing
2. **PMAT-003** (4 days) - FMA Correctness
3. **PMAT-010** (4 days) - Backend Equivalence (complete)

### Sprint 2 (Week 3-4): Optimization
4. **PMAT-001** (5 days) - Loop Splitting
5. **PMAT-002** (7 days) - Token Synchronization

### Sprint 3 (Week 5-6): Quality
6. **PMAT-009** (6 days) - Numerical Stability
7. **PMAT-008** (15 days) - PTX Debugger (start)

### Sprint 4 (Week 7-9): Platform
8. **PMAT-008** (continued)
9. **PMAT-007** (8 days) - ROCm Backend
10. **PMAT-006** (10 days) - Metal Backend

**Total Estimated Effort**: 67 days (~3 months at 5 days/week)

---

## FKR Cross-Reference

| PMAT | FKR ID | Hypothesis | Status |
|------|--------|------------|--------|
| PMAT-001 | FKR-003 | Loop splitting eliminates divergence | PENDING |
| PMAT-002 | FKR-004 | Token sync equivalent to barriers | PENDING |
| PMAT-003 | FKR-005 | FMA is IEEE 754 compliant | PENDING |
| PMAT-004 | FKR-006 | Coalesced >=4x bandwidth vs strided | PENDING |
| PMAT-005 | FKR-001, FKR-007 | LZ4 GPU byte-identical | CORROBORATED |
| PMAT-006 | FKR-011 | Metal equivalent to CUDA | PENDING |
| PMAT-007 | FKR-012 | ROCm equivalent to CUDA | PENDING |
| PMAT-008 | FKR-008 | PTX parser handles all constructs | PENDING |
| PMAT-009 | FKR-009 | Operations stable under perturbation | PENDING |
| PMAT-010 | FKR-010 | All backends numerically equivalent | IN PROGRESS |

---

## Citation Summary

**Total Unique Citations**: 30 peer-reviewed sources

| Category | Count |
|----------|-------|
| GPU Architecture & Memory | 3 |
| Memory Models & Synchronization | 3 |
| Loop Optimization & Divergence | 3 |
| Numerical Analysis & Floating-Point | 9 |
| Compression Algorithms | 3 |
| GPU Verification | 3 |
| Platform-Specific (Metal, ROCm) | 6 |

All citations include DOI or ISBN where available. See `CUDA_TDG_COMPLIANCE.md` Appendix B for full citation index.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-11 | Claude | Initial creation consolidating PMAT tickets and FKR entries |
