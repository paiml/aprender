# PMAT-3714 R2 — implementation receipt

Ticket: #3714 (P0, 0.69.1). This row: done_when 2. `Refs #3714`, not a closing PR.
Branch `PMAT-3714-r2-on-v2`, rebuilt on R1-v2 (`01989323d`) after R1 was stacked onto #3750 A.
Code commit `f6c84afd7` (the >=64-position test prompt fix); `341c958af` adds the `--per-op` refusal.

## done_when 2: `apr parity` measures qwen3moe at ≥ 64 positions

`ParityArm::Moe` (`parity_hybrid.rs`) routes a qwen3moe GGUF to
`measure_moe`/`run_moe` (`parity_moe.rs`): the CPU forward
(`forward_single_qwen3_moe_with_cache`, FP32 activations — the same pair the
runtime F2 guard judges) against `Qwen3MoeCudaModel::forward_single`. The
qwen3moe refusal (#3367's `UNMEASURED-TOOL`) is lifted for the two dispatched
spellings only; `qwen3_5moe` keeps it.

**Measured on lambda** (RTX 4090, `apr parity --assert --json`, a 92-token
prompt — the printing-press paragraph, extended to clear 64 positions — under
`gpu-q`):

| file | tokens | passed | failed | min cosine | argmax mismatches | wall |
|---|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M | 92 | 92 | 0 | 1.000000 | 0 | 17 s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 92 | 92 | 0 | 1.000000 | 0 | 17 s |

Both files: every one of 92 positions PASSes, exceeding the ≥64 floor.
Evidence: `evidence/3714/r2-apr-parity-64-lambda.txt`.

**gx10**: build is done (`apr 0.69.0 (f6c84afd7)`), the run is queued but has
not completed — measured queue depth on gx10 at time of writing: two prio-1
`gpu-q` jobs from another session have been waiting on the flock for **4h18m**
(`serve_e2e.sh`, aprender-c7), so the shared lock is severely backed up
independent of anything in this branch. Reported to the cop as an operational
finding. This receipt covers lambda; the gx10 row is a follow-up commit once
the queue drains, or the cop's call to proceed without it (#3714's done_when
2 does not itself name a host, unlike done_when 1, which R1 already
satisfied on both hosts).

## Bug found and fixed en route

The real-file test (`parity_moe_holds_at_64_positions_on_the_real_file`) had
a self-contradicting precondition: `assert!(tokens.len() >= 64, …)` against a
prompt that tokenized to fewer than 64 tokens, so the test failed before
`measure_moe` ever ran — a guard that could not pass on its own fixture.
Fixed by extending the prompt (`f6c84afd7`). Verified: the test now passes
(19.0 s, lambda) and reports the tokenized length, not a hardcoded guess.

## `apr parity --per-op` refuses the hybrid and MoE arms

`--per-op`'s tap is wired into the dense CPU-vs-GPU pair only. Without a
refusal, a qwen3moe or Qwen3.5 file reaches `per_op_run`'s hardcoded
`OwnedQuantizedModel::from_mapped`, which is the dense loader `parity_refusal.rs`
already documents as reading the 0-byte FFN placeholders `from_gguf_for_moe`
leaves — a TOOL gap reported as a MODEL divergence, the exact #3715 failure
class (absence read as conformance), now specifically for `--per-op`.
`per_op_refusal` asks `parity_arm`, the same predicate `run()` dispatches on,
so the two paths cannot drift. Falsifier
`per_op_refuses_the_arms_it_has_no_tap_for`: dense architectures return
`None`; hybrid and MoE return a message naming the arm and what does measure
it (plain `apr parity`, no `--per-op`).

## Checks
- `cargo test -p apr-cli --lib` (no cuda): 7311 passed.
- `cargo test -p apr-cli --features cuda --lib parity`: 212 passed (was 211
  passed / 1 failed before the prompt fix).
- `cargo test -p apr-cli --lib per_op_refus`: 1 passed.
- cuda release build on lambda and gx10.

## Not in this row
- gx10's `apr parity` confirmation (above — queue-blocked, not code-blocked).
- R3 (v2 kernels), R4 (ladder rungs, #3712) are unstarted, per the R1 receipt's split.
