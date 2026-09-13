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

## P3 quorum (three families) — `.pr/L0-1b/quorum.md`
3/3 endorse fixing the CPU side; lane 2's dissent ("per-32 matches the GPU" is falsified by the probe table) upheld and folded;
the delegate's zero-code arm ran: **`DIRECT_FP32_GEMV=1` → 1.5B min cosine 0.950827 → 0.999896, no diverging op** (arm A7).

## Step 2 — the fix (this PR, measured on lambda)

`quantize::has_crushed_block` (per 256-block `max/second ≥ 8`, basis: the criterion table in the arms doc — never on
77 ordinary positions × 28 layers, ≥ 20 on the first token's crushed blocks) routes ONE matmul to the f32-activation
kernel (`quantize::direct_f32::fused_q4k_parallel_matvec_f32_into`, the `DIRECT_FP32_GEMV` numerics) through
`matvec_into_honest` on the reference forward (QKV, o_proj, gate/up, down) and the fused gate+up driver.

| measurement | before (71a25c2bb) | after |
|---|---|---|
| 1.5B `apr parity` min cosine (78 positions) | 0.950827 @0, 1 < 0.98, 2 argmax≠ | **0.999761** @36, 0 < 0.98, 0 argmax≠ |
| 1.5B `--per-op` first diverging op / lm_head | post_ffn_residual L26 (0.660) / 0.950827 | **none** / 0.999761 (worst row k_post_rope L6 0.996102) |
| 7B `apr parity` min cosine | 0.998607 | **0.999580** (worst per-op row ffn_out L23 0.994065) |
| GPU dump tree (1.5B, 78 positions) | — | 0 differing files vs before |
| GPU m=1 greedy stream, 32 tokens, 5 local models (0.5B, 1.5B, 7B coder; Qwen3-8B; Qwen3.5-0.8B) | — | byte-identical (Qwen3.5-0.8B refuses on both: SSM layers unsupported, rc 8) |
| fallbacks on the 78-position run (`APR_CRUSHED_TRACE=1`) | — | 1.5B 298 / 7B 624 matmul calls (of ~15k / ~15k) |
| CPU decode, 1.5B, 64 tokens, `--backend cpu`, n=3 (tok/s) | 11.6 · 10.8 · 10.8 (an earlier pair under GPU load: 11.6 · 9.8 · 11.4 vs 9.2 · 10.3 · 10.8) | 12.0 · 13.2 · 12.6 — no slowdown resolved at n=3 on a loaded box; basis [U] until an idle n≥5 run |
| refactor oracle: predicate-off mutant vs the pre-fix CPU tree | — | 24,258 / 24,258 files bitwise identical; the mutant names L26 again, lm_head 0.950827; `accept.sh` A5 RED |

Revert falsifier (POP-F-003): make `has_crushed_block` return false → A5 RED (measured above); restore → 6/6.


## The step-0 records are REPRODUCIBLE, not historical (2026-09-08)

`check_hardcoded_paths.sh` refused 11 shipped machine-specific paths on this branch. Seven
were real portability defects in the scripts (`accept.sh`, `probe26.py`, `probe26b.py`,
`sweep.sh`, and `sweep.sh`'s absolute reference to a SIBLING WORKTREE's
`check_model_parity.sh`) and are now `${APR_MODELS_DIR:-$HOME/models}` and a `$ROOT`
derived from the script's own location.

The other four were the recorded `apr parity` outputs, whose `model` field held the
absolute path. Those could not simply be re-run: two of them are the PRE-FIX table, and
the fix has since landed. So they were re-taken from a **pre-fix reproduction build** —
the registered mutant, `has_crushed_block` returning false — with the model named
relatively, and every one came back **bit-for-bit identical**:

| record | comparand | result |
|---|---|---|
| `qwen2.5-coder-1.5b-instruct-q4_k_m.json` | 283 `rows` + every other key | identical; `first_divergence` still `post_ffn_residual` layer 26, min cosine 0.660150 @0 |
| `qwen2.5-coder-7b-instruct-q4_k_m.json` | 283 `rows` + every other key | identical; `first_divergence` none |
| `A7-cpu-fp32-gemv.json` | 78 `metrics` + every other key | identical |
| `perop-A7-cpu-fp32.json` | 283 `rows` + every other key | identical |

Only `model` differs (`/home/noah/models/…` → `./…`). Binary: `apr 0.65.2 (c08e2682d)`
built `--features cuda` from this branch with the predicate flipped, sha256
`0372f1cc846c0992`; the post-fix build of the same tree is `776cbbdb4306b5d8`.

That is worth more than the path cleanup that prompted it. The step-0 evidence is not an
artifact a reader has to take on trust: it is a **function of the committed tree**,
recoverable by flipping one predicate, and the row's central finding — layer 26
`post_ffn_residual` at cosine 0.660150 — reproduces to the bit. `check_hardcoded_paths.sh
--full` → delta +0.

## Gaps / next
- gx10 twin of the table and of the 1.5B gate (`make fleet-verify ROW=L0-1b` once G-11b lands) — the "resolved" line of
  the row needs lambda AND gx10; this receipt stays `partial` until the gx10 row is measured.
- `apr chat --gpu <1.5B>` prints `selected: cuda … parity: PASS` — to be captured on lambda (the `apr run -q` path prints no
  admission line) and posted on #2971.
- The scratch-path forward (`forward_single_with_scratch`, results.rs) and the SHIP-007 `forward_traced` keep their
  Q8_K sites unchanged: they are not on the gate's path and their files carry pre-existing complexity debt the hook
  refuses to let this PR touch; `apr run` generation goes through the reference forward (the fallback trace counted 639
  calls over a 64-token generation).
- The APR-format transformer (`apr_transformer/helpers.rs`, 4 Q8_K sites) is out of this row's manifest scope.
- Speed basis: an idle-box n≥5 CPU decode pair before/after (I4).

## Gaps / next (superseded list kept for the record)

- The fix is on the CPU side (decided spec in docs/audits/l0-1b-arms.md §Fallback criterion): the Q4_K × Q8_K
  drivers run f32 activations for a matmul whose normed residual-stream input has a 256-block with
  max/second-largest ≥ 8 (basis: the measured table; never on 77 ordinary positions, always on the first
  token's crushed blocks); the down projection keeps Q8_K. Expected 0.9999 (arm A7 measured the path);
  fallback plan: per-32 scales as the quorum decided. Revert → the 1.5B per-op table names layer 26 again
  (POP-F-003), 7B GREEN; the GPU m=1 stream is untouched by construction.
- gx10 twin of the table (`make fleet-verify ROW=L0-1b` once G-11b lands).
- P3 quorum (three families) on the finding before the fix PR.
