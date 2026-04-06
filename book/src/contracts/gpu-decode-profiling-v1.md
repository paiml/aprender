# gpu-decode-profiling-v1

**Version:** 2.0.0

GPU decode profiling contract — ensures BrickProfiler data reflects real GPU execution time AND report output faithfully represents profiler measurements (no hardcoded scores, no silent truncation, no fake metadata)

## References

- REALIZAR-GPU-PERF-001 v2.10.0 §5 — BrickProfiler Decode Breakdown
- REALIZAR-GPU-PERF-001 v2.9.0 — BrickProfiler Deferred sync mode bug
- trueno BrickProfiler (src/brick/profiler/mod.rs) — SyncMode enum
- realizar executor_api.rs — start_brick_id/stop_brick_id sync gates
- aprender crates/apr-cli/src/commands/gguf.rs — brick_scores_from_profiler (B1-B5)
- aprender crates/apr-cli/src/commands/cbtop_measure_batch.rs — build_and_output_report (B6-B13)
- aprender crates/apr-cli/src/commands/cbtop_get_cpu_memory.rs — simulated path (B14-B18)
- Hoefler & Belli SC'15 — Scientific Benchmarking of Parallel Computing Systems

## Equations

### brick_ordering

```
rank(bricks, by=per_call_avg) must respect kernel complexity
```

**Domain:** $All bricks with count > 0$

**Invariants:**

- $LmHead per-call avg > GateProjection per-call avg (vocab GEMV > layer GEMV)$
- $GateProjection per-call avg > RmsNorm per-call avg (GEMV > elementwise)$
- $AttentionScore per-call avg > Residual per-call avg (flash attn > vector add)$

### graph_disable

```
valid_profiling => NOT has_decode_graph
```

**Domain:** $Boolean$

**Invariants:**

- $CUDA graph replay executes all kernels in one opaque launch$
- $Bricks instrumented during graph CAPTURE only (first token), not REPLAY$
- $Profiling with graphs enabled measures 1/N of actual decode time$

### report_completeness

```
len(JSON.brick_scores) == len(profiler.all_brick_stats())
```

**Domain:** $profiler has collected data$

**Invariants:**

- $Every profiler brick appears in JSON output — no silent truncation$
- $Aggregate brick_score uses all N bricks — zip with fixed-length array forbidden$
- `FalsificationSummary.total_points == len(JSON.brick_scores)`
- `FalsificationSummary.passed + failed == total_points`

### report_denominator

$$
decoded_tokens = LmHead.count (exactly 1 LmHead per decoded token)
$$

**Domain:** $LmHead.count > 0$

**Invariants:**

- `per_decoded_tok_us(b) = (b.count * b.avg_us) / decoded_tokens`
- $profiler.total_tokens counts brick ELEMENTS — must NEVER be used as decoded token count$
- $Dividing total_ns by profiler.total_tokens produces values 100-300x too small$

### report_fidelity

```
for each brick b: JSON.actual_us(b) == profiler.avg_us(b)
```

**Domain:** $All bricks emitted in JSON output, profiler.avg_us > 0$

**Codomain:** $relative_error in [0, 0.01]$

**Invariants:**

- $JSON actual_us must equal profiler per-call avg (not per-element, not per-token)$
- `JSON score must equal compute_brick_score(actual_us, budget_us) — never hardcoded`
- `JSON grade must equal score_to_grade(score) — never hardcoded`
- `JSON gap_factor must equal actual_us / budget_us — never 1.0 unless actual == budget`
- $No BrickScore field may be a compile-time constant (score: 100, grade: 'R', gap: 1.0)$

### report_metadata

$$
metadata fields must be measured or zero — never hardcoded nonzero
$$

**Domain:** $All numeric fields in PmatScores and FalsificationSummary$

**Invariants:**

- `rust_project_score: 0.0 unless computed by pmat in this run`
- $tdg_score: 0.0 unless computed by pmat in this run$
- `cuda_tdg_score: 0.0 unless computed by pmat in this run`
- $FalsificationSummary must derive from actual pass/fail counts, not constants$
- $No hardcoded magic numbers: 137, 173.9, 98.1, 95.2, 976.0$

### sync_verification

```
is_immediate = (measured_brick_us / expected_brick_us) > 0.5
```

**Domain:** `measured_brick_us > 0, expected_brick_us = wall_clock_us / num_bricks_per_token`

**Invariants:**

- $Deferred sync: brick avg < 100us regardless of kernel (CPU launch latency)$
- $Immediate sync: brick avg correlates with kernel complexity (large GEMV > small norm)$
- $LmHead (n=151936) must be >10x RmsNorm in Immediate mode$

### token_accounting

```
decoded_tokens = iterations * tokens_per_iteration
```

**Domain:** `iterations > 0, tokens_per_iteration > 0`

**Invariants:**

- $profiler.total_tokens counts brick elements, NOT decoded tokens$
- `calls_per_decoded_token(LmHead) = 1`
- `calls_per_decoded_token(AttentionScore) = num_layers`
- `calls_per_decoded_token(RmsNorm) = 2 * num_layers + 1`

### wall_coverage

```
coverage = sum(brick_total_ns) / wall_clock_ns
```

**Domain:** `brick_total_ns >= 0, wall_clock_ns > 0`

**Codomain:** $coverage in [0, 1]$

**Invariants:**

- $coverage >= 0.85 when profiling is valid (bricks account for >=85\% of wall time)$
- $coverage < 0.50 indicates graph replay hiding brick instrumentation$
- $coverage > 1.0 is impossible (bricks are subsets of wall time)$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Wall coverage threshold | `sum(brick_total_ns for all bricks) / wall_clock_ns >= 0.85` |
| 2 | invariant | Coverage upper bound | `sum(brick_total_ns) <= wall_clock_ns` |
| 3 | invariant | Graph disable for profiling | `profiling_enabled => !has_decode_graph` |
| 4 | invariant | LmHead call count | `lm_head.count == decoded_tokens` |
| 5 | invariant | Layer brick call count | `attention.count == decoded_tokens * num_layers` |
| 6 | monotonicity | Brick ordering respects complexity | `per_call_avg(LmHead) > per_call_avg(GateProjection) > per_call_avg(RmsNorm)` |
| 7 | invariant | Immediate sync detectable | `sync_mode == Immediate => LmHead.avg_us > 10 * RmsNorm.avg_us` |
| 8 | bound | Deferred sync ceiling | `sync_mode == Deferred => max(brick.avg_us for all bricks) < 200` |
| 9 | invariant | Report fidelity — actual_us matches profiler | $abs(JSON.actual_us(b) - profiler.avg_us(b)) / profiler.avg_us(b) < 0.01$ |
| 10 | invariant | Report fidelity — score computed not hardcoded | `JSON.score(b) == compute_brick_score(JSON.actual_us(b), JSON.budget_us(b))` |
| 11 | invariant | Report completeness — no truncation | `len(JSON.brick_scores) == len(profiler.all_brick_stats())` |
| 12 | invariant | Report completeness — falsification accounting | `JSON.falsification.total_points == len(JSON.brick_scores)` |
| 13 | invariant | Report denominator — decoded tokens from LmHead | `decoded_tokens == LmHead.count AND decoded_tokens != profiler.total_tokens` |
| 14 | invariant | Report metadata — no hardcoded nonzero constants | `rust_project_score == 0 AND tdg_score == 0 AND cuda_tdg_score == 0 (unless pmat computed)` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-GDP-001 | Wall coverage threshold | Brick time accounts for >=85% of wall time | CUDA graph replay hiding brick instrumentation — bricks only measured during capture pass |
| FALSIFY-GDP-002 | Graph disable enforcement | Profiling with graphs enabled produces <50% coverage | Graph replay path has brick instrumentation (unexpected — verify code) |
| FALSIFY-GDP-003 | Immediate sync vs Deferred | Immediate sync LmHead avg > 100us; Deferred LmHead avg < 100us | set_profiler_sync_mode not propagating to executor start/stop_brick calls |
| FALSIFY-GDP-004 | LmHead call count invariant | LmHead.count == decoded_tokens (exactly 1 per token) | LmHead brick instrumentation missing from some code path, or called >1 per token |
| FALSIFY-GDP-005 | Brick ordering by complexity | LmHead per-call > Gate per-call > RmsNorm per-call | Sync mode not Immediate (all bricks show ~same launch latency) or kernel regression |
| FALSIFY-GDP-006 | Token accounting consistency | profiler.total_tokens != decoded_tokens (elements, not tokens) | Profiler counting decoded tokens instead of brick elements — API misuse risk |
| FALSIFY-GDP-007 | Coverage upper bound | sum(brick_total_ns) <= wall_clock_ns (bricks are subset of wall time) | Timer corruption or double-counting in brick profiler |
| FALSIFY-GDP-008 | Reproducibility | CV of per-token time < 0.15 across 3 runs (same config, locked clocks) | Clock frequency unstable (need jetson_clocks) or thermal throttling |
| FALSIFY-GDP-009 | Report fidelity — actual_us matches profiler avg | JSON actual_us for LmHead within 1% of profiler.avg_us (e.g. ~595us, not ~1.9us) | BrickScore construction uses wrong value (per-element instead of per-call avg) — B1 regression |
| FALSIFY-GDP-010 | Report fidelity — no hardcoded scores | At least one brick has score != 100 (real GPU times vary vs budget) | compute_brick_score not called — scores are hardcoded constants (B3 regression) |
| FALSIFY-GDP-011 | Report completeness — brick count | JSON brick_scores has 11 entries (matching profiler brick count for Qwen 1.5B) | Bricks silently dropped by zip truncation (B6 regression) |
| FALSIFY-GDP-012 | Report completeness — falsification accounting | falsification.total_points == len(brick_scores) AND passed + failed == total_points | FalsificationSummary still hardcoded (B10-B12 regression) |
| FALSIFY-GDP-013 | Report denominator — decoded tokens | LmHead per_decoded_tok_us > 100us (not 1.9us from element denominator) | Denominator uses profiler.total_tokens (brick elements ~952K) instead of LmHead.count (~3K) |
| FALSIFY-GDP-014 | Report metadata — no magic constants | rust_project_score == 0.0 AND tdg_score == 0.0 AND cuda_tdg_score == 0.0 | Hardcoded constants 173.9/98.1/95.2 still present (B7-B9 regression) |
| FALSIFY-GDP-015 | Report metadata — no 137 constant | falsification.total_points != 137 (must equal actual brick count) | Hardcoded FalsificationSummary constant still present (B10 regression) |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-GDP-001 | GDP-INV-001 | 4 | bounded_int |
| KANI-GDP-002 | GDP-INV-003 | 2 | exhaustive |
| KANI-GPU_DE-003 | Wall coverage threshold | 8 | exhaustive |
| KANI-GPU_DE-004 | Coverage upper bound | 8 | stub_float |
| KANI-GPU_DE-005 | Graph disable for profiling | 8 | exhaustive |
| KANI-GPU_DE-006 | LmHead call count | 8 | exhaustive |
| KANI-GPU_DE-007 | Layer brick call count | 8 | exhaustive |
| KANI-GPU_DE-008 | Brick ordering respects complexity | 8 | exhaustive |
| KANI-GPU_DE-009 | Immediate sync detectable | 8 | exhaustive |
| KANI-GPU_DE-010 | Deferred sync ceiling | 8 | exhaustive |
| KANI-GPU_DE-011 | Report fidelity — actual_us matches profiler | 8 | exhaustive |
| KANI-GPU_DE-012 | Report fidelity — score computed not hardcoded | 8 | exhaustive |
| KANI-GPU_DE-013 | Report completeness — no truncation | 8 | exhaustive |
| KANI-GPU_DE-014 | Report completeness — falsification accounting | 8 | exhaustive |
| KANI-GPU_DE-015 | Report denominator — decoded tokens from LmHead | 8 | exhaustive |
| KANI-GPU_DE-016 | Report metadata — no hardcoded nonzero constants | 8 | exhaustive |

## QA Gate

**GPU Decode Profiling Contract** (F-GDP-001)

Ensures BrickProfiler data reflects real GPU execution AND report output is faithful

**Checks:** wall_coverage_threshold, graph_disable_enforcement, immediate_sync_verification, lm_head_call_count, brick_ordering, token_accounting, coverage_upper_bound, reproducibility, report_fidelity_actual_us, report_fidelity_no_hardcoded_scores, report_completeness_brick_count, report_completeness_falsification_accounting, report_denominator_decoded_tokens, report_metadata_no_magic_constants, report_metadata_no_137

**Pass criteria:** All 15 falsification tests pass (GDP-001 through GDP-015)

