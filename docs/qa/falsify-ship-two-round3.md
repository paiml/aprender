# SHIP-TWO Falsification Round 3

**Date:** 2026-04-12
**Spec:** SPEC-SHIP-TWO-001
**v29 step:** 11,431 / 15,530 (73.6% complete)
**Elapsed:** ~35 hours

---

## Model 2 (albor v29) — Falsification Results

### FALSIFY-SHIP-013: v29 divergence — PASS

val_ppl trajectory (every 500 steps since round 2):
```
step=6000   val_ppl=72.82
step=6500   val_ppl=69.94  (-3.9%)
step=7000   val_ppl=67.63  (-3.3%)
step=7500   val_ppl=61.81  (-8.6%)
step=8000   val_ppl=57.70  (-6.6%)
step=8500   val_ppl=54.94  (-4.8%)
step=9000   val_ppl=52.31  (-4.8%)
step=9500   val_ppl=49.12  (-6.1%)
step=10000  val_ppl=47.19  (-3.9%)
step=10500  val_ppl=46.83  (-0.8%)
step=11000  val_ppl=45.17  (-3.5%)
```

**Monotonically decreasing across 11 eval checkpoints.** Zero divergence.
v28 diverged at step 11K — v29 is past that point with no instability.

### FALSIFY-SHIP-014: val_ppl < 60 (revised gate) — PASS

**Current:** 45.17 at step 11,000.
**Predicted:** 44.7 at step 15,530.
**Gate:** < 60.0.
**Verdict:** **PASS** — already below gate with 26% of training remaining.

### FALSIFY-SHIP-020: Throughput >= 8K tok/s — PASS

**Current:** 9,069 tok/s. Stable throughout.

### FALSIFY-SHIP-022: Timeline <= 5 weeks — PASS

**Elapsed:** ~35 hours. **ETA:** ~16.5 hours. Total: ~51.5 hours (~2.1 days).
Well within the 5-week gate (even within the 2.4-day estimate).

---

## v28 vs v29 Comparison

| Metric | v28 (5B raw) | v29 (2B filtered) |
|--------|-------------|-------------------|
| Best val_ppl | 38.53 (step 3.5K, then diverged) | **45.17** (step 11K, still decreasing) |
| Predicted final | N/A (diverged) | **44.7** |
| Divergence | YES (step 11K) | **NO** (past 11K, stable) |
| Throughput | ~9,800 tok/s | ~9,070 tok/s |
| Data quality | Raw codeparrot | AST-filtered (29% pass rate) |

**Key insight:** v29 hasn't beaten v28's best val_ppl (38.53 vs 45.17) but v28's
number was pre-divergence and unreliable. v29 is stable and will complete its full
cosine schedule — producing a usable checkpoint for SFT distillation.

---

## Updated Gate Status (All 12 Gates)

| Gate | Status | Evidence |
|------|--------|----------|
| GATE-SHIP-001 (HE >=87% M1) | **PASS** | 87.20% |
| GATE-SHIP-002 (MBPP >=80% M1) | PENDING | 76.2% — needs DPO |
| GATE-SHIP-003 (28/28 contracts M1) | PENDING | 67/68 |
| GATE-SHIP-004 (Pipeline M1) | BLOCKED | PMAT-014 |
| GATE-SHIP-005 (Loadable M1) | PENDING | Post-merge |
| GATE-SHIP-006 (Sovereignty M1) | **PASS** | Verified |
| GATE-SHIP-007 (Pre-train M2) | **PASS** | val_ppl 45.17, monotonic decrease, predicted 44.7 |
| GATE-SHIP-008 (Teacher M2) | BLOCKED | MoE Phase 3 pending |
| GATE-SHIP-009 (HE >=30% M2) | PENDING | Requires v29 + SFT |
| GATE-SHIP-010 (54/54 contracts M2) | **PASS** | 54/54 |
| GATE-SHIP-011 (Loadable M2) | PENDING | Post-v29 |
| GATE-SHIP-012 (Sovereignty M2) | **PASS** | Zero cloud, 100% Rust |
