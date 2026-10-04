# 0.73 L3 side-fix ticket bodies, S1 to S3 (DRAFT, not filed; la-73, 2026-10-03)

These are three small fixes found while writing the 0.73 drafts. None of them blocks 0.73: no 0.73 model reaches the S1 or S2 code, and S3 deletes two files that nothing compiles.
- **Filing:** after the cop writes LIVE 0.70.1, as PROPOSE-TICKET lines next to P1..P5 (`ticket-bodies-P1-P5.md`). L3 never mints (C277).
- **Format:** the P-body shape. Each acceptance names a falsifier that is red at 316dee2cd4, a control that shows the test can tell the two paths apart, and the mutations that must turn it red.
- **Citations:** at origin/main 316dee2cd4. Every defect marked [V] was read there.
- **Contract:** S1 and S2 amend `cpu-q4k-activation-quant-v1`, the contract that `fused_gate_up.rs:173` names. The draft is `contracts-draft/cpu-q4k-activation-quant-v1.yaml` (1.1.0). It adds the equation `activation_path_selector`, the obligations AQ-SEL-005 and AQ-SEL-006, and FALSIFY-AQ-005..007.

## The selector that S1 and S2 restore
A Q4_K matvec on f32 input takes f32 activations when any of these holds, and Q8_K activations otherwise:
1. the calling thread runs inside `with_fp32_activations` (`fp32_activations_scoped()`, parallel_k.rs:210);
2. `DIRECT_FP32_GEMV=1`;
3. the entry checks for crushed blocks and `has_crushed_block(x)` is true (crates/aprender-serve/src/quantize/mod.rs:409). A 256-block is crushed when max|x| / second|x| ≥ 8.0 (:404).

`fused_q4k_parallel_matvec_into` implements 1 and 2 (parallel_k.rs:303-304). `matvec_into_honest` adds 3 and notes each switch (ffn_block.rs:784-795). Three comments promise the selector:
- inside the scope, "every Q4_K matvec it reaches" takes the direct FP32 path (parallel_k.rs:214);
- the scope works "as `DIRECT_FP32_GEMV=1` makes every thread's do" (parallel_k.rs:202-204);
- on a crushed block, "the Q4_K drivers run the f32-activation dot ... for that call" (crates/aprender-serve/src/quantize/mod.rs:394-395).

S1 and S2 fix the two entries that break these promises.

---

## S1 — `fused_gate_up_q4k_into`: follow the selector, and make the #2971 crushed fallback f32
**Refs** #2971 (L0-1b), #3714 (scope), #3999 (found in 0.73 R1 §11a and §11b). Contract: `cpu-q4k-activation-quant-v1` 1.1.0 draft, AQ-SEL-005 and AQ-SEL-006.

**Defect [V]** `fused_gate_up_q4k_into` (fused_gate_up.rs:177) reads neither the scope nor the env var.
- **Crushed input, default route.** The fallback (fused_gate_up.rs:189-213) calls `fused_q4k_parallel_matvec_into` for gate and for up. Outside the scope that function quantizes to Q8_K (parallel_k.rs:303-304, then :326), so the remedy changes nothing. The fallback never calls `note_crushed_fallback` either, so the `[crushed-block]` count that a speed basis cites (crates/aprender-serve/src/quantize/mod.rs:431-432) misses these calls.
- **Uncrushed input, inside the scope or with `DIRECT_FP32_GEMV=1`.** Phase 1 quantizes to Q8_K (fused_gate_up.rs:214-224), then `fused_q4k_q8k_ffn_up_gate_into` runs (:227).

**Reach** The only caller is `fused_gate_up_matmul_into` (fused_matmul_into.rs:163). Its Q4_K arm (:186-187) runs only for seq_len 1 with matching qtypes and dims. That function is called from:
- the non-fused gated FFN branch (ffn_block.rs:58), taken by Gemma-1, LayerNorm models and models with no FFN norm;
- `scratch_q8k_up_gate` when `!use_q8k_path` (crates/aprender-serve/src/gguf/inference/forward/results.rs:103).

No 0.73 model reaches it (R1 §11b). Neither scope caller at 316dee2cd4 reaches it either (apr-cli parity_moe.rs:110, qwen3_moe_dispatch.rs:209).

**Scope**
- Add `pub(crate) fn q4k_takes_f32() -> bool` to parallel_k.rs. It returns `fp32_activations_scoped() || std::env::var("DIRECT_FP32_GEMV").as_deref() == Ok("1")`. parallel_k.rs:303-304 calls it, and its behaviour there does not change.
- Replace fused_gate_up.rs:189-213 with:
  ```rust
  // L0-1b (#2971) and #3714: the Q4_K selector, as fused_q4k_parallel_matvec_into applies it.
  let crushed = super::has_crushed_block(activations);
  if crushed || super::parallel_k::q4k_takes_f32() {
      if crushed {
          // One note per projection, on the calling thread: the rate counts matvecs.
          super::note_crushed_fallback(in_dim, out_dim);
          super::note_crushed_fallback(in_dim, out_dim);
      }
      let (g, u) = rayon::join(
          || super::fused_q4k_parallel_matvec_f32_into(gate_weight_data, activations, in_dim, out_dim, gate_output),
          || super::fused_q4k_parallel_matvec_f32_into(up_weight_data, activations, in_dim, out_dim, up_output),
      );
      g?;
      return u;
  }
  ```
- The doc at fused_gate_up.rs:171-176 says the function always pre-quantizes to Q8_K. Change it to name the selector.
- Add a test-only counter to quantize/mod.rs: a `#[cfg(test)]` thread-local `Cell<usize>`. `note_crushed_fallback` bumps it first thing, before the `APR_CRUSHED_TRACE` check. Because it is thread-local, tests running in parallel never see each other's notes.

**Acceptance** The tests go in `fp32_activation_scope_tests` (parallel_k.rs:502-545), next to the scope's own tests.
- **Inputs.**
  - in_dim 512 and out_dim 8.
  - Weights follow the `create_q4k_weights` pattern (parallel_k_test_helpers.rs:7).
  - Uncrushed x is a smooth sequence that is not made of powers of two, with max/second < 2 in every block.
  - Crushed x has the layer-26 shape in block 0: 146.68, then 7.28, then a 0.5 fill (ratio 20.1; `the_layer_26_block_is_crushed` builds it with `block_with`, crates/aprender-serve/src/quantize/mod.rs:444 and :452-454). Block 1 is uncrushed.
  - The reference is `fused_q4k_parallel_matvec_f32_into` for each projection.
  - Each test first asserts that `DIRECT_FP32_GEMV` is unset, so the environment cannot make a control pass vacuously.
- **FALSIFY-AQ-006.** Inside `with_fp32_activations`, on uncrushed x, gate and up are bit-identical to the reference. It is red at 316dee2cd4, where `fused_q4k_q8k_ffn_up_gate_into` runs. Control: the same call outside the scope differs from the reference.
- **FALSIFY-AQ-007.** Outside the scope, on crushed x, gate and up are bit-identical to the reference, and exactly 2 notes are counted. Uncrushed x gives 0 notes. It is red at 316dee2cd4: Q8_K, and 0 notes. Control: two `fused_q4k_parallel_matvec_into` calls (today's fallback) on crushed x differ from the reference.
- Notes are counted only outside the scope. Inside it the call runs on a pool worker, and the test thread's counter cannot see that thread's notes.
- **Mutations.** Each must turn a test red, and the PR body records the run:
  - drop `q4k_takes_f32()` from the new condition → AQ-006;
  - make `q4k_takes_f32` return `false` → AQ-006; make it return `true` → the AQ-006 control;
  - put `fused_q4k_parallel_matvec_into` back in the fallback → AQ-007;
  - delete one note or both → the AQ-007 count;
  - move the notes into the `rayon::join` closures → the AQ-007 count. Called from a thread outside any pool, `join` runs both closures on pool workers. That is why the sketch notes before the join.
- `cargo test -p aprender-serve --lib quantize` and `cargo test -p aprender-contracts --lib` pass.

**Blocked by** Nothing. S2 can share the PR; whichever of S1 and S2 lands first adds `q4k_takes_f32`.
**Hosts** x86 CI lib tests. No model, no GPU.
**Out of scope**
- Entries that receive activations already quantized to Q8_K (`scratch_q8k_up_gate`, crates/aprender-serve/src/gguf/inference/forward/results.rs:18-33; the traced forward, traced.rs:88). Their callers choose Q8_K themselves.
- Caching the env read. It is read on every call today (parallel_k.rs:304), and the helper keeps that.
- AQ-EQ-001's wording. It says the Q8_K dot is within 0.1% of the f32 dot for every input (`applies_to: all`), which is false on crushed blocks: L0-1b measured 13% low on the layer-26 gate and up outputs (crates/aprender-serve/src/quantize/mod.rs:388-393). The draft records this in the selector's invariants and leaves the obligation's wording to the contract owner.

---

## S2 — `fused_q4k_multirow_matmul_f32_into`: follow the scope
**Refs** #3714 (scope), #4228 (multirow), #3999 (R1 §11a). Contract: `cpu-q4k-activation-quant-v1` 1.1.0 draft, AQ-SEL-005.

**Defect [V]** The multirow entry (q4k_q8k_multirow.rs:161) takes f32 only when `DIRECT_FP32_GEMV=1` (:178-179). With m > 1 and in_dim a multiple of 256, a call inside the scope still quantizes each token row to Q8_K (:194-200) and runs the Q8_K multirow kernel (:201).
- The scope's doc says every Q4_K matvec inside it is f32 (parallel_k.rs:214).
- The multirow's own doc names only the env var (q4k_q8k_multirow.rs:154-157).

**Reach** The only caller is `matmul_rows` (forward_qwen35.rs:1416, call at :1425). Only `prefill_chunk_qwen35` uses that (:1228). Only `forward_prefill_qwen35` calls it (:1200, call at :1223), and only its tests call that (forward_qwen35_prefill_tests.rs). Neither scope caller at 316dee2cd4 runs a qwen35 prefill. So this is a latent trap, not a live bug: an `fp32_act` reference over a qwen35 chunked prefill would run its m > 1 chunks on Q8_K and still report itself as exact.

**Scope**
- q4k_q8k_multirow.rs:178 becomes `let direct_fp32 = q4k_takes_f32();`. The file is `include!`d into parallel_k.rs (:500), so the helper needs no import.
- Update the doc at q4k_q8k_multirow.rs:154-157. The per-row fallback runs when m == 1, when in_dim is not a multiple of 256, or when the selector picks f32 (inside `with_fp32_activations`, or with `DIRECT_FP32_GEMV=1`).
- The bit-identity claim at q4k_q8k_multirow.rs:154-155 still holds: each token row goes through `fused_q4k_parallel_matvec_into`, which takes f32 under the same predicate.

**Acceptance**
- **FALSIFY-AQ-005.** With m = 3, in_dim 512 and out_dim 8, inside `with_fp32_activations`, every token row is bit-identical to `fused_q4k_parallel_matvec_f32_into` on that row. It is red at 316dee2cd4, where the Q8_K multirow runs.
  - Control: outside the scope, the same call differs from that reference.
  - Token 1 carries the layer-26 crushed block, so the Q8_K error is large, not a single ulp.
  - The test asserts that `DIRECT_FP32_GEMV` is unset.
- **Mutations:** restore the env-only read at q4k_q8k_multirow.rs:178 → AQ-005; drop `direct_fp32` from the :179 predicate → AQ-005.
- **Interaction:** NEON-Q4K-008 calls this entry outside the scope on purpose (R3-test-skeletons.md §4c: entry G, and "Why the prefill entries"). S2 does not change what it measures.
- The commands listed for S1 pass.

**Blocked by** Nothing. It shares `q4k_takes_f32` with S1.
**Hosts** x86 CI lib tests.
**Out of scope**
- A crushed check in the multirow. #2971 scoped its remedy to the reference forward (direct_f32.rs:6-7). Whether a default-route chunked prefill needs one is an open question, and nothing here proposes it.
- P1..P5. Their receipts read the path from the trace (BPM-008, f008d), so they neither need S1 or S2 nor block on them.

---

## S3 — delete the orphans `quantize/fused_q4k.rs` and `quantize/fused_q.rs`
**Refs** #3999 (R3 §13 rows 1 and 9; R1 §11a). This body covers both orphan PROPOSE-TICKET lines (07:49Z and 18:21Z). They can be filed as one ticket or two, because the acceptance splits cleanly by file.

**Defect [V]** No `mod` or `include!` names either file, so neither compiles. Both still look live.
- `fused_q4k.rs` (360 lines) is a byte-identical copy of `q4k_dot_avx2.rs`, which `fused_k.rs:370` includes. A NEON arm added to the copy compiles away (R3 §13 row 1).
- `fused_q.rs` (327 lines) is an older copy of `q5k_q6k_matvec.rs` (included at parallel_k.rs:498) that has since diverged: the first 114 lines match, then 155 lines differ.
  - It holds a stale `fused_q4k_q8k_ffn_up_gate_into` (fused_q.rs:253) beside the live one (q5k_q6k_matvec.rs:380). That is the kernel S1 routes around, and a search for it finds two definitions.
- Four falsify tests cite `fused_q.rs` as the CPU path (listed below), and `crates/aprender-serve/.pmat-baseline.json` lists both files.

**Scope**
- Delete `crates/aprender-serve/src/quantize/fused_q4k.rs` and `crates/aprender-serve/src/quantize/fused_q.rs`.
- Keep `quantize/tests/fused_q4k.rs`. The `include!("fused_q4k.rs")` at `quantize/tests/dequantize.rs:509` resolves next to the including file, so it names this test file. The file is live and listed in `scripts/include_fmt_baseline.txt:1479`.
- Re-point the four doc-cites:
  - falsify_q6k_fp_accumulator_order_001.rs:101, falsify_q6k_activation_amplification_002.rs:84 and falsify_q6k_chain_length_003.rs:104 name `q5k_q6k_matvec.rs::fused_q6k_parallel_matvec` (:34);
  - falsify_q4k_bisect_dequant_007.rs:128 ("CPU fused") names `parallel_k.rs::fused_q4k_parallel_matvec` (:190).
- In `crates/aprender-serve/.pmat-baseline.json`, drop the entries keyed `./src/quantize/fused_q.rs` (`.pmat-baseline.json:80471`) and `./src/quantize/fused_q4k.rs` (`.pmat-baseline.json:117217`). Keep `./src/quantize/tests/fused_q4k.rs` (`.pmat-baseline.json:38151`).
- NEON-Q4K-006 used `fused_q4k.rs` as its orphan control. On this branch the control now plants its own unnamed file (R3-test-skeletons.md §4a; the neon-q4k-q6k-v1 draft), so the delete does not weaken that probe.

**Acceptance** These are one-shot checks recorded in the PR body, not a contract falsifier.
- **Compiled code is unchanged.** The output of `cargo test -p aprender-serve --lib -- --list` is identical before and after, compared by sha256.
- **No cite is left.** `git grep -n -E 'quantize/fused_q\.rs|quantize/fused_q4k\.rs' -- crates` prints nothing.
- **Mutations:**
  - delete `q4k_dot_avx2.rs` instead of its byte-identical twin → the build fails at the `fused_k.rs:370` include;
  - also delete `quantize/tests/fused_q4k.rs` → the build fails at the `tests/dequantize.rs:509` include;
  - leave one doc-cite or one baseline entry → the no-cite check prints it.

**Blocked by** Nothing.
**Hosts** x86 CI. The NEON-Q4K-006 probe itself runs later, with NEON-Q4K-000.
**Out of scope** Any other unreferenced file. This ticket deletes only the two that R3 §13 verified.

---

## Status (2026-10-03 22:10Z)
- None of S1..S3 is a 0.73 gate. `falsifier-landing-map.md` lists AQ-005..007 separately from the 42 0.73 falsifiers.
- Filing: held under S-1 (C292) until LIVE 0.70.1, like P1..P5.
