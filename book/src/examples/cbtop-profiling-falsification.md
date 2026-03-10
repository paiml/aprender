# Case Study: cbtop Profiling Pipeline Falsification (GH-420)

> **Jidoka**: When profiling data is wrong, STOP THE LINE.

## The Incident

During 4090 serial benchmarking (2026-03-06), the `apr cbtop --headless --json` command reported LmHead at **1.9 µs** per call. The BrickProfiler stderr output showed the real value: **595 µs**. A 300x discrepancy.

This triggered a **Jidoka stop** — all optimization work halted. You cannot optimize what you cannot measure.

## Five Whys

1. **Why was LmHead reported as 1.9µs?** — `brick_scores_from_profiler()` constructed `BrickScore` with hardcoded values instead of using `stats.avg_us()`.
2. **Why were values hardcoded?** — The initial implementation used placeholder `score: 100, grade: "R", gap_factor: 1.0` and was never updated to use real profiler data.
3. **Why wasn't this caught earlier?** — No contract existed to verify report fidelity against profiler measurements. Tests only checked that JSON was valid, not that values were correct.
4. **Why did one bug suggest more?** — The hardcoded pattern (compile-time constants instead of computed values) is a systematic error class. If `actual_us` was faked, `score`, `grade`, `gap_factor`, and downstream aggregations were likely faked too.
5. **Why were 18 bugs found, not 1?** — The BrickScore pipeline has 3 construction sites (gguf.rs, cbtop\_measure\_batch.rs, cbtop\_get\_cpu\_memory.rs), each with independent hardcoded values. The bug was replicated across all paths.

## Full Pipeline Falsification

Rather than fix one bug and hope, we falsified the **entire** BrickScore pipeline. Method: trace every field from profiler measurement to JSON output, flag any value that could not be derived from `BrickStats`.

### Bug Taxonomy

**B1-B5: `gguf.rs::brick_scores_from_profiler()`**

| Bug | Field | Was | Should Be |
|-----|-------|-----|-----------|
| B1 | `actual_us` | `per_token_us` (wrong denominator) | `stats.avg_us()` |
| B2 | denominator | `profiler.total_tokens` (brick elements) | `LmHead.count` (decoded tokens) |
| B3 | `score` | `100` (hardcoded) | `compute_brick_score(actual, budget)` |
| B4 | `grade` | `"R"` (hardcoded) | `score_to_grade(score)` |
| B5 | `gap_factor` | `1.0` (hardcoded) | `actual_us / budget_us` |

**B6-B13: `cbtop_measure_batch.rs::build_and_output_report()`**

| Bug | Field | Was | Should Be |
|-----|-------|-----|-----------|
| B6 | brick aggregation | 7 hardcoded weights, zip truncation | Equal-weight avg over all N bricks |
| B7 | `rust_project_score` | `173.9` | `0.0` (not computed) |
| B8 | `tdg_score` | `98.1` | `0.0` (not computed) |
| B9 | `cuda_tdg_score` | `95.2` | `0.0` (not computed) |
| B10 | `total_points` | `137` | `len(brick_scores)` |
| B11 | `passed` | `137` | Count of `gap_factor <= 1.0` |
| B12 | `failed` | `0` | `total - passed` |
| B13 | `status`/`ci_result` | Hardcoded target `976.0` | `all_pass` from brick scores |

**B14-B18: `cbtop_get_cpu_memory.rs` (simulated path)**

Same bug classes as B6-B13, replicated in the simulated code path.

## The Fix

### Before (gguf.rs)

```rust
scores.push(BrickScore {
    name: stats.name.clone(),
    score: 100,                    // B3: hardcoded
    grade: "R".to_string(),        // B4: hardcoded
    budget_us: per_token_us,       // B1: wrong value
    actual_us: per_token_us,       // B1: wrong value
    gap_factor: 1.0,               // B5: hardcoded
});
```

### After (gguf.rs)

```rust
// C-GDP-001: decoded_tokens = LmHead.count (exactly 1 per decoded token)
let decoded_tokens = all.iter()
    .find(|s| s.name == "LmHead")
    .map_or(1u64, |s| s.count.max(1));

for stats in &all {
    let avg_us = stats.avg_us();                              // Real profiler value
    let per_decoded_tok_us = (stats.count as f64 * avg_us)
        / decoded_tokens as f64;                               // Correct denominator
    let budget_us = wall_us_per_token * (pct / 100.0);
    let score = compute_brick_score(per_decoded_tok_us, budget_us); // Computed
    let grade = score_to_grade(score);                              // Computed

    scores.push(BrickScore {
        name: stats.name.clone(),
        score,
        grade: grade.to_string(),
        budget_us: per_decoded_tok_us,
        actual_us: avg_us,
        gap_factor: if budget_us > 0.0 { per_decoded_tok_us / budget_us } else { 1.0 },
    });
}
```

## Contract: gpu-decode-profiling-v1 v2.0.0

The fix is backed by 15 falsification tests in the `gpu-decode-profiling-v1` contract (extended from 8 to 15 to cover report-level invariants):

| Test | What It Catches |
|------|----------------|
| GDP-009 | `actual_us` doesn't match profiler (B1 regression) |
| GDP-010 | All scores hardcoded to 100 (B3 regression) |
| GDP-011 | Bricks silently truncated by zip (B6 regression) |
| GDP-012 | FalsificationSummary hardcoded (B10-B12 regression) |
| GDP-013 | Wrong denominator (element count vs decoded tokens) |
| GDP-014 | Magic PMAT constants (173.9/98.1/95.2) |
| GDP-015 | Magic 137 constant in total\_points |

## Lesson: Falsify the Pipeline, Not the Symptom

Finding one hardcoded value should trigger a **full pipeline audit**. The pattern "placeholder that was never replaced" is a systematic error class — if it happened once, it happened everywhere the same code pattern was used.

The Jidoka principle applies: **do not produce more output with known-bad tooling**. Fix the measurement system first, then resume optimization.

## Related

- [Jidoka (Built-in Quality)](../toyota-way/jidoka.md) — Stop the line principle
- [Popperian Falsification](../advanced-testing/popperian-falsification.md) — Test methodology
- [gpu-decode-profiling-v1](https://github.com/paiml/provable-contracts) — Formal contract
