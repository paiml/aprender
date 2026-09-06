---
status: partial
ticket: PMAT-1070
issue: 2971
kind: code
branch: agent/L0-1b
model: claude-fable-5-1
tokens_used: ~400k (across sessions 42763d92; instrument + probe + decomposition)
wall_clock_s: ~14400
turns: ~90
---
# impl receipt — PMAT-1070 / L0-1b: the per-op table names the op

**Row L0-1b (#2971, PP-066).** Steps 0 and 1 are DONE and measured on lambda; step 2 (the fix)
is OPEN — hence `status: partial`. This receipt is the resume point.

## Step 1 — arms (measured, 7 s each, `apr 0.65.2 (cbb511c55)` + cuda, lambda RTX 4090)

Seven arms (baseline; SKIP_CUDA_GRAPH=1; FP8_DECODE=0; FP8_PREFILL=0 FP8_DECODE=0; FLASH_DECODE=0;
FUSED_GATE_UP=0; all off) → the 1.5B reads **0.9508 at position 0 (token 785)** in every arm; the 7B
0.9986. No switchable GPU path is the mechanism. Table: `docs/audits/l0-1b-arms.md` §Step 1;
raw: `.pr/L0-1b/step0/arms-table.txt`, `sweep.sh`.

## Step 0 — `apr parity <gguf> --per-op` (this PR)

- CPU tap on `forward_single_with_cache` (the gate's own comparator) through a thread-local plan
  (`inference_trace/gpu_stage_dump/per_op_tap.rs`); GPU per-phase dumps on the executor
  (`cuda/executor/stage_dump.rs`, arm points in `phase_attention.rs` / `indexed_ffn.rs`); the
  admission gate bypassed INTERNALLY and recorded (`ParityGateRecord::skipped_for_diagnosis`);
  the non-graphed path forced and printed as an override.
- 281 rows × 78 positions in 9 s (1.5B) / 18 s (7B). **1.5B: `post_ffn_residual` layer 26,
  min cosine 0.660150 @pos 0; the `lm_head` row is 0.950827 = the gate's number. 7B: no op
  under 0.98.** Tables in `docs/audits/l0-1b-arms.md` §Step 0.
- The element (layer 26, position 0): residual dim 408 = −3664 (massive activation); CPU
  `ffn_out` dim 408 = +3675.7 cancels it (→ 11.3); GPU = 4084.2 (→ 407). `ffn_norm@26` identical
  on both sides; `ffn_swigl@26` neurons 2908/7035 differ by +14 % / +7.5 %.
- **Which side is wrong (float64 probe, `.pr/L0-1b/step0/probe26.py`, gguf-py dequant):** truth
  −1142.0 / 618.4; GPU −1136.5 / 616.9 (≤ 0.5 %); CPU −996.4 / 573.6 (13 %), reproduced to three
  decimals by Q8-quantising the activation per 256 elements. **The CPU Q8_K reference is the
  inaccurate side on massive-activation tokens; the GPU (per-32 activation scales) is right.**

## Decomposition (hook-clean, verified bitwise)

`forward_single_with_cache` (cyclomatic/cognitive 26/55 → 6/9) and `single_cache_ffn_block`
(21/69 → 6/12) split into `single_cache_qkv`, `single_cache_qk_norm_rope`,
`first_token_attention`, `post_norm_in_place`, `single_cache_ffn_residual`, `ffn_input_normed`,
`ffn_activate`, `single_cache_ffn_fused_gate_up`, `add_position_embedding` and the GH-559 debug
helpers. Oracle: the per-op CPU dump tree of the 1.5B over 78 positions — **24,258 files
bitwise identical** before/after (`diff -rq`), GPU tree identical, same first diverging op.

## Verification (every command re-run by the orchestrator)

| check | result |
|---|---|
| `.pr/L0-1b/accept.sh` (6 legs: table tests, tap tests, `pv validate`, doc, 1.5B names layer 26 with lm_head ≈ 0.9508, 7B clean) | 6/6 |
| `pv validate contracts/apr-parity-per-op-v1.yaml` (pin) | valid |
| `cargo check -p apr-cli` with and without `--features cuda` | 0 / 0 |
| pre-commit quality gates (complexity per staged file) | all passed |
| `check_complexity_ratchet.sh` | PASS (none new, none grown) |

## Dispatch ledger

No Claude sub-agents; no agy lanes yet (P3 quorum for this N-lane row is owed before arming:
three model families on the arms doc + the probe).

## Gaps / next (step 2)

- The fix is on the CPU side: the Q4_K × Q8 dot family (`fused_q4k_q8k_dot`,
  `fused_q4k_q8k_ffn_up_gate_into`, `fused_q4k_q8k_parallel_matvec_into`, and
  `quantize_activations_q8k_into`) takes one activation scale per 32-element sub-block instead of
  one per 256 — the GPU's numerics, strictly more accurate. Revert → the 1.5B per-op table names
  layer 26 again (POP-F-003), 7B GREEN; the GPU m=1 stream is untouched by construction.
- gx10 twin of the table (`make fleet-verify ROW=L0-1b` once G-11b lands).
- P3 quorum (three families) on the finding before the fix PR.
