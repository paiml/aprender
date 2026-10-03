# R3 — NEON Q4_K / Q6_K dot kernels for aarch64 (cell C4, gx10 CPU) — DRAFT
la-73, 2026-09-27. Status: draft for epic #3999 (0.73 "Runs Everywhere"). Exit criteria served: E1 (C4 leg), E2 (C4 speed), E5 (kernel key).
Evidence tags: [V] = read in the tree at origin/main aca6f2d7f6; [A] = reported by an agent; [U] = unverified.

## 1. Problem (from the R5 census)
- On aarch64, `detect_simd_backend()` returns `SimdBackend::Neon` (crates/aprender-serve/src/quantize/simd_backend.rs:41-44) [V]. No code dispatches on `Neon`: it is used only by Display and tests [V].
- `fused_q4k_dot_simd` (quantize/fused_k.rs:193), `fused_q6k_dot_simd` (fused_q5k_q6k.rs:118) and `fused_q4k_q8k_dot_simd` (q4k_dot_avx2.rs:338, included at fused_k.rs:370) have only `cfg(target_arch = "x86_64")` arms [V]. On aarch64 each one falls through to the scalar kernel (fused_k.rs:60, fused_q5k_q6k.rs:15, q4k_dot_avx2.rs:232; the identical `fused_q4k.rs:232` is the orphan and is not compiled, corrected 2026-10-03).
- Consequence 1: every gx10 CPU number measures the scalar kernel while being labelled NEON. That breaks verification rule 2.
- Consequence 2: no aarch64 quantized SIMD exists to port. R3 is new kernels plus an honest label.

## 2. Scope
In:
1. aarch64 arms in the three dispatchers above:
   - `fused_q4k_dot_neon`: f32 activations; NEON baseline, so no runtime detection is needed on aarch64.
   - `fused_q6k_dot_neon`: same.
   - `fused_q4k_q8k_dot_neon_dotprod`: int8 `sdot`, gated by `is_aarch64_feature_detected!("dotprod")`, else the scalar kernel.
   - `unsafe` is allowed in aprender-serve (`unsafe_code = "allow"`, crates/aprender-serve/Cargo.toml:32 [V]). Each unsafe block states the invariant, as the AVX2 arm does at fused_k.rs:203.
2. **Honest label:**
   - Add `kernel_path() -> &'static str`, the name of the kernel the dispatcher will actually call, e.g. `q4k-f32/neon`, `q4k-q8k/neon-sdot`, `q4k-f32/scalar`.
   - Emit it in the `apr run --trace` kernel step.
   - `SimdBackend` display stays, but may no longer be cited as evidence of a SIMD path.
3. Contract `contracts/neon-q4k-q6k-v1.yaml`, patterned on `avx512-q4k-v1.yaml`:
   - parity bound ε = 1e-3 abs per super-block dot (avx512-q4k-v1.yaml:28 [V]);
   - super-block layout reuses `q4k-q6k-superblock-v1.yaml` and `q4k-interleaved-scale-min-v1.yaml`.

Out:
- Q5_K / Q8_0 / Q4_0 NEON (follow-on rows; same pattern).
- i8mm / SVE2 (advisory; [U] whether GB10's Grace cores expose them).
- Any GPU work.

## 3. Equations
- **E-R3-1 parity:** ∀ random super-block b (seeded), for k ∈ {q4k-f32, q6k-f32, q4k-q8k}: |neon_k(b, x) − scalar_k(b, x)| < 1e-3 · max(1, |scalar_k(b, x)|).
- **E-R3-2 dispatch honesty:** on aarch64, `kernel_path()` of each dispatcher ∈ {`*/neon*`}. With dotprod absent, `q4k-q8k` reports `*/scalar`. A path never names a kernel the call does not reach.
- **E-R3-3 E1 leg (C4):** logits cosine(C4 apr-CPU-NEON, C0 apr-CPU-x86) ≥ 0.995 on the E1 prompt set, Qwen3-dense Q4_K_M. The composed-leg method is in R1.
- **E-R3-4 E2 speed:** the lower CI bound of decode tok/s ratio apr-C4 / llama.cpp ggml-cpu (pin d1d3c3396, RQ-2) is ≥ 0.5. This is report-only until first-green + 7 green nights (R-6).

## 4. Falsifiers (each must be able to go RED)
| id | prediction | planted RED |
|---|---|---|
| FALSIFY-NEON-Q4K-000 | build floor: `cargo check -p aprender-serve --lib --target aarch64-unknown-linux-gnu` exits 0 (baseline RC=0 @aca6f2d7f6, 138 crates, 2026-09-27) | add a NEON kernel whose `#[cfg(target_arch = "aarch64")]` body calls an x86-only intrinsic; the check must exit non-zero. Report-only until first green + 7 nights (R-6). The x86 CI cannot see this: aarch64 cfg bodies are not compiled on x86 |
| FALSIFY-NEON-Q4K-001 | E-R3-1 holds for 10 000 seeded blocks × 3 kernels | a mutant that swaps the low/high nibble order in `fused_q4k_dot_neon` must FAIL |
| FALSIFY-NEON-Q4K-002 | E-R3-2: the aarch64 test asserts `kernel_path()` starts with `q4k-f32/neon` | revert the dispatcher arm → path is `scalar` → FAIL. On x86 the test is NotRun{reason: "not aarch64"}, never PASS |
| FALSIFY-NEON-Q4K-003 | cross-arch golden: dots on a fixed block set, recorded on x86 scalar, match aarch64 NEON within ε | perturb one scale byte in the golden → FAIL |
| FALSIFY-NEON-Q4K-004 | the Q6K analogue of 001 (6-bit ql/qh recombination) | a mutant that drops `qh` bit 5 must FAIL |
| FALSIFY-NEON-Q4K-005 | E-R3-3 cosine ≥ 0.995; the trace line shows `neon` | a trace without a `neon` kernel line → the cell is refused (R1 F-R5-2 rule), not scored |

## 5. Measurement plan
- **Compile gate now (lambda, no aarch64 host needed):** `cargo check -p aprender-serve --target aarch64-unknown-linux-gnu`. Whether the target is installed is [U]; if not, it goes to gx10.
- **Correctness:** `cargo test -p aprender-serve --lib neon` on gx10 CPU. Use a private CARGO_TARGET_DIR, and check `df` first (gx10 /mnt/nvme-raid0 is its root disk). Waits for train-active to clear.
- **E1/E2 cells:** via the R1 harness on C4, once R1 lands.
- **Kernel key for E5:** `(q4k, gemv, row-major, f32|q8k, aarch64-neon[-sdot])`. This depends on the KREG backend field (K4, with kreg).

## 6. Risks
- **K10:** until E-R3-2 ships, every C4 number is mislabelled. Existing gx10 CPU receipts citing "NEON" must be re-read as scalar.
- **K11:** f32-activation NEON reorders the FMA reduction relative to scalar. ε = 1e-3 relative may be tight for large-magnitude rows. Tune from measured distributions only, never by widening to pass.
- **K12 [U]:** dotprod availability on GB10 is unverified. `is_aarch64_feature_detected!` handles absence, and E-R3-2 reports it honestly.
- **K15 (L):** dispatch shape. The NEON kernels slot into the existing cfg-plus-runtime-detect dispatchers (§9), so nothing is re-plumbed.

## 7. Size
M. Measured 2026-09-27 at aca6f2d7f6 (brace-matched, so approximate). The AVX2 templates are `fused_q4k_dot_avx2` (q4k_dot_avx2.rs:23-210, 188 lines, 16 distinct intrinsics), `fused_q6k_dot_avx2` (fused_q5k_q6k.rs:144-306, 163 lines, 17) and `fused_q4k_q8k_dot_avx2` (requires.rs:10-215, 206 lines, 11). That is **557 lines** of template, so plan **~550–650 lines** of NEON: the 128-bit NEON needs 2× the lanes of AVX2 per step, which adds unrolling. On top: the q8k kernel ships in two variants (sdot and widening), so ~+150; a label hook; a contract; and 6 falsifiers (000–005). The earlier ~150 LOC per kernel was a 20–30% underestimate. Size stays M.

## 8. Baseline (2026-09-27)
- aarch64 `cargo check` of aprender-serve at origin/main aca6f2d7f6: RC=0, 138 crates, 31 s. Log: la-73/r3-check.log.
- 4 warnings, all in aprender-compute and **aarch64-only**: an x86 `cargo check -p aprender-compute --lib` at the same sha gives rc 0 with 0 code warnings. The 4 are all unused variables: `mr_block` and `nr_block` (`blis/compute.rs:94-95`), and `backend` at `brick/quant_ops/mod.rs:219` and `:318`. The two `backend` ones are compiler proof of F-R5-1: on aarch64 the backend argument is read by nothing. The NEON kernels should consume it, and FALSIFY-NEON-Q4K-000 goes to 0 aarch64 warnings. Rule: NEON work must not add new aarch64 warnings; that is part of FALSIFY-NEON-Q4K-000.
- Only a compile check was done. Link and run need C4 (gx10). Watch the gx10 root disk (it filled before, per memory).

## 9. Slot points (read from source at aca6f2d7f6)
The NEON kernels go into three aprender-serve dispatchers. Each has the same shape, `#[cfg(target_arch = "x86_64")] { runtime detect → unsafe avx kernel }` then a scalar fallback:
| dispatcher | file:line | x86 kernels today | NEON slot |
|---|---|---|---|
| `fused_q4k_dot_simd(&[u8], &[f32]) -> Result<f32>` | quantize/fused_k.rs:193 | avx2+fma | `#[cfg(target_arch = "aarch64")]` block before the fallback; NEON is baseline on aarch64, so no detection is needed |
| `fused_q6k_dot_simd(&[u8], &[f32]) -> Result<f32>` | quantize/fused_q5k_q6k.rs:118 | avx2+fma | same |
| `fused_q4k_q8k_dot_simd(&[u8], &[f32], &[i8]) -> Result<f32>` | quantize/q4k_dot_avx2.rs:338 (included at fused_k.rs:370) | avx512vnni v2, then avx2 | aarch64 block with `is_aarch64_feature_detected!("dotprod")` → sdot kernel, else the NEON widening kernel |
- **No signature changes, and no new `backend` parameter.** Dispatch is runtime feature detection, as it is on x86. New risk **K15 (dispatch shape): L**, because nothing has to be re-plumbed.
- The oracle for each NEON kernel is the scalar fallback it sits in front of (`fused_q4k_dot`, `fused_q6k_dot`, `fused_q4k_q8k_dot`). That is the E-R3-1 reference; no new oracle is needed.
- **Out of scope:** trueno `brick/quant_ops` `DotQ5KOp`/`DotQ6KOp` (the unused-`backend` warnings at :219 and :318). Only examples, tests and re-exports call them; the serving path does not. They are a second, cold NEON gap. R3 may silence the warning (`let _ = backend` on non-x86), but that does not count as NEON coverage.
- Q5_K (`fused_q5k_q6k.rs`) has the same shape and is a cheap follow-on. Q4_0 and Q8_0 (`fused_q4_0_q8_0.rs:6`, `fused_q8_0_q8_0.rs:15/208`) are x86-only too. They are listed for the R5 census, not added to R3.

## 10. Contract draft (2026-09-27)
- The contract draft is drafts/neon-q4k-q6k-v1.yaml, `kind: kernel`: 4 equations, FALSIFY-NEON-Q4K-000..005, 5 obligations, and KANI-NEON-Q4K-001..003. Current state (2026-10-03): contracts-draft/neon-q4k-q6k-v1.yaml, 5 equations, FALSIFY-NEON-Q4K-000..008, 8 obligations and the same three Kani harnesses.
- The Kani harnesses cover only integer decode steps (nibble split, 6-bit recombine, scale-unpack bounds), because those are exhaustively provable.
- pv validate: rc 0. The first try failed PROVABILITY-001 because a kernel contract must have kani_harnesses.
- pv score: **0.57 (D)**. By dimension: D1 0.70, D2 1.00, D3 0.60, D4 Lean 0.00, D5 Binding 0.00. There is no binding registry for a draft outside the tree; the binding lands with the code.
- K9 recheck: `pv lint --min-score 0.6` is RED on the real draft **and** on the copy with its falsifiers stripped (0.32, F). At 0.6 the lint therefore does not separate them for kernel contracts; the score does (0.57 vs 0.32). The lint floor is not the gate for this draft.

## 11. Lineage and K10 claim inventory (2026-09-28, read at origin/main c115c5ed02)

**Lineage — R3 is not a duplicate; it is the open successor of two closed rows.**
- #2567 "Q4_K GEMV has no aarch64 SIMD and its 'parallel' variant calls the scalar path" (milestone 0.73.0) is CLOSED as *completed*, but the closing comment is "folded into epic #3999 (triage evidence/triage-1717.md)". Nothing was fixed. R3 carries it.
- #2942 / roadmap PMAT-1027 "W-G GB10: NEON Q4_K GEMV; batched prefill default on sm_121…; pre-compiled kernels" is CLOSED (backlog). The roadmap row is still `status: planned, assigned_to: null`, and it bundles two unrelated GPU items. When R3 is minted, it should name PMAT-1027 as superseded for its NEON part only. Owning the roadmap row is the cop's call; L3 edits no roadmap.
- K10 still holds at c115c5ed02. The three dispatchers (`fused_k.rs`, `fused_q5k_q6k.rs`, `q4k_dot_avx2.rs`) have no `target_arch = "aarch64"` arm. The only aarch64 cfg in `aprender-serve/src/quantize/` is `simd_backend.rs:41`.

**Label sources that say NEON while the Q4K/Q6K dot runs scalar:**

| Source | What it says | Consumed by | Verdict |
|---|---|---|---|
| `aprender-serve/src/quantize/simd_backend.rs:41` `detect_simd_backend()` | `SimdBackend::Neon` on every aarch64 host, from arch alone | serve: tests only (`simd_backend.rs:319`, `tests_coverage_detect_simd.rs`); `aprender-zram/bins/trueno-ublk/src/device/mod.rs:351` | Arch label, not kernel path. Harmless today (no serve prod consumer). A trap if a receipt ever reads it: that is why E-R3-2 asks for `kernel_path()` per dispatcher, never this |
| `docs/audits/impl-PMAT-989-receipt.md:52,60` | gx10 / mini `backend: cpu … class=neon` | receipt prose | Host-class label. It must not be read as "Q4K/Q6K ran NEON". Emitter: `aprender-compute/src/registry/mod.rs:438` `render_entry` prints `class={compute_class}`, set at :606 from `cpu_isa()` (:614), which returns `"neon"` for any aarch64 build (:627) and the widest detected ISA on x86. It is an ISA-availability label for the host, never the kernel a Q4K/Q6K dot took |
| `contracts/trueno/neon-dequant-v1.yaml` (+ generated `contract_*_neon_q4k_dequant!` macros in `aprender-compute/src/generated_contracts.rs`) | NEON Q4K/Q6K/Q8_0 **dequant** equations with Lean theorems | aprender-compute | Dequant, not the fused dot. It is a separate kernel family, but R3's oracle tests can reuse its fixtures (block builders, ε). Check before R3 writes a new block generator |
| `docs/build-ledger/2026-09-13/e45eaab47-reconcile.json:85` | "remaining: NEON path for Q4_K and Q6_K GEMV, and an ARM speed gate" | ledger | Correct (it names the gap) |
| `docs/specifications/0.66-performance-parity-report.md:175` | G5 "GB10 decode 0.66× … Q4_K GEMV scalar on aarch64 (#2567)" | spec | Correct: attributes the GB10 gap to scalar |

**Consequence for R3's exit (adds to E-R3-2):** a NEON number is admissible only with `kernel_path()` in the receipt. The `detect_simd_backend()` value and `class=neon` are never proof, which is the same rule as OBS-18's gpu_proof.

## 12. Status (cop ruling 2026-09-28 10:20Z)
R3 is a **FINDING, not a minted row**. The mint is deferred (only the cop mints, freeze on). Nothing here is code. When it is minted, the scope is §2 plus §11's rule: `kernel_path()` in every receipt; `cpu_isa()`/`class=` and `detect_simd_backend()` are host labels, never proof. Checked at c115c5ed02.

## 13. L25 review of `neon-q4k-q6k-v1` (2026-10-03, origin/main 316dee2cd4, read-only)
| # | Hole | Fix | Falsifier |
|---|---|---|---|
| 1 | **Dead site.** The contract cited `quantize/fused_q4k.rs:338` for the q4k-q8k dispatcher. No `mod` or `include!` names that file: it is a byte-identical 360-line copy of `q4k_dot_avx2.rs`, which `fused_k.rs:370` includes. The file has been an orphan since the APR-MONO subtree merge (3da75b7d3d). A NEON arm added there compiles nothing, and every test would pass against the old path | Cite corrected in §1, §9, §11 and in R5. A planted `compile_error!` must break the aarch64 check at each cited site | NEON-Q4K-006 |
| 2 | Random f16 `d`/`dmin` bytes are Inf/NaN for 1 draw in 32 | Finite, normal f16 by construction | precondition |
| 3 | Uniform scales or `dmin = 0` hide the scale and min terms (the existing tests use uniform scales; R3-test-skeletons.md) | Varied 6-bit scales and mins, `dmin ≠ 0`; second mutation drops the min term | NEON-Q4K-001 |
| 4 | The epsilon floor of 1 makes the check absolute for small dots | At least 90% of cases have \|dot\| ≥ 1, and the test prints the fraction | precondition |
| 5 | One super-block per case never runs the block loop | n ∈ {1, 2, 3, 7} | precondition |
| 6 | The widen fallback is unreachable on gx10, which has dotprod | Test-only direct entry | NEON-Q4K-007 |
| 7 | A golden regenerated in the run is a self-comparison | Committed golden with a pinned sha256 | NEON-Q4K-003 |
| 8 | The C4 leg used prefill-final only and an unnamed reference path, and listed its gate as a postcondition | Decode positions, same `cpu_ref_path`; the gate stays in the formula | R1 BPM-008/-013 |
| 9 | **Second orphan.** `quantize/fused_q.rs` is an older copy of `q5k_q6k_matvec.rs`, the file `parallel_k.rs:498` includes. No `mod` or `include!` names it, yet four falsify tests cite it as the CPU side (falsify_q6k_fp_accumulator_order_001.rs:101 and three others; R1 §11a). It holds matvec wrappers, not dots, so it is never a NEON site | Delete it and fix the four cites (PROPOSE-TICKET, after LIVE). The contract's site list already leaves it out | none (cleanup) |
| 10 | **A dot Err becomes a 0.0 row.** Every CPU matvec writes 0.0 for a row whose dot returns Err (generic_matvec.rs:126, :143, :239; parallel_k.rs:320; direct_f32.rs:63; q5k_q6k_matvec.rs, 12 sites from :249 to :448; bsum_precompute.rs:296, :312; fused_gate_up.rs:134, :137, :160, :163). The fixtures use rows of 1, 2, 3 and 7 blocks, while a 2048-wide row has 8. A NEON arm that errs only on longer rows passes every dot test and zeroes rows in the forward. A few zero rows can pass the C4 cosine gate | The NEON arm has exactly the scalar oracle's Err conditions (shape only). A matvec-level check at production row lengths (8 and 36 blocks) calls each row's dot directly, so a NEON Err fails the test, and checks that the matvec wrote that value to that row | NEON-Q4K-008 |

Cleanup candidates (PROPOSE-TICKET, not done here):
- Delete the orphan `quantize/fused_q4k.rs`. It shows up in `.pmat-baseline.json` and in code search as a live twin of `q4k_dot_avx2.rs`.
- Delete the orphan `quantize/fused_q.rs` (row 9).

Route on gx10 (verified at 316dee2cd4). It shows that the three cited dispatchers are the only arms correctness needs:
- **Default path (no scope).** A Q4_K matvec quantizes to Q8_K and calls `fused_q4k_q8k_parallel_matvec_into`
  (q5k_q6k_matvec.rs:79).
  - The x86 lean and 4-row paths are compiled out off x86_64 (:122, :180). So their `[0.0f32; 4]` placeholder (:230)
    never runs on aarch64.
  - Each row calls `fused_q4k_q8k_dot_with_bsums_simd` (bsum_precompute.rs:220). Off x86 that falls back to
    `fused_k::fused_q4k_q8k_dot_simd`, the `q4k_dot_avx2.rs:338` dispatcher.
  - A NEON bsums variant would be a speed follow-on only.
- **Inside `with_fp32_activations`.** Each Q4_K row calls `fused_q4k_dot_simd` (parallel_k.rs:320 → fused_k.rs:193).
- **Q6_K, either way.** `generic_parallel_matvec_into` calls `fused_q6k_dot_simd` (fused_q5k_q6k.rs:118).
pv 0.70.0: validate 0/0; lint 0 errors, the same 4 lean_theorem warnings. Obligations went from 5 to 7.

NEON-Q4K-008 (2026-10-03, row 10) checks five decode matvec entries and two prefill entries. Each one turns a dot Err into a 0.0 row:
- (A) `fused_q4k_parallel_matvec_into` inside `with_fp32_activations` (parallel_k.rs:320).
- (B) `fused_q4k_parallel_matvec_f32_into`, the crushed-block route of `matvec_honest` (direct_f32.rs:63; ffn_block.rs:790, :799).
- (C) The default path above (q5k_q6k_matvec.rs:331-350).
- (D) `fused_q4k_q8k_ffn_up_gate_into`. The scratch and traced forwards call it through `scratch_q8k_up_gate` (results.rs:33; rows at q5k_q6k_matvec.rs:437-448).
- (E) The Q6_K route (generic_matvec.rs:126, :143).
- (F) `fused_q6k_multirow_matmul_into`, the Q6_K prefill (q4k_q8k_multirow.rs:326; rows at generic_matvec.rs:239).
- (G) `fused_q4k_multirow_matmul_f32_into`, the Q4_K prefill (q4k_q8k_multirow.rs:161). On aarch64 it quantizes each token to Q8_K (:194-199) and runs the row loop of C, `fused_q4k_q8k_parallel_matvec_into`, once per token (:137-147).

The shapes are in_dim 2048 and 9216 (8 and 36 blocks per row) and out_dim 48 and 300. At 9216, C quantizes into heap buffers instead of stack ones (MAX_STACK_DIM 8960, parallel_k.rs:328). Out_dim 48 takes the sequential branches. 300 takes the parallel ones and ends in a 44-row tail tile.

The reference is each row's own dot, called directly, and not the scalar oracle. The oracle bound is relative to the dot, while f32 summation-order error grows with the row's sum of |w·x|. So a long row whose terms cancel could fail with no kernel defect. On aarch64 the bsums dot falls through to `fused_q4k_q8k_dot_simd` (bsum_precompute.rs:237), so C and D reach that one dispatcher whether bsums is set or not.

F and G (2026-10-03) run 3 tokens at once. No C4 run reaches them at 316dee2cd4. Their only production caller, `matmul_rows` (forward_qwen35.rs:1416), is reached only from `forward_prefill_qwen35` (:1200), and only its tests call that. They are covered anyway. FALSIFY-4228-004 and -005 check the batched prefill bit-identical to `forward_single_qwen35`, and a NEON Err would pass them: on aarch64 both sides call the same dot on the same row and swallow the Err into the same 0.0. G ignores the fp32 scope (R1 mechanism list), so the test runs it outside the scope against each token's Q8_K dot. F tiles rows by 64 at 8 blocks and by 32 at 36 (generic_matvec.rs:226), so every shape ends in a part tile. Q5_K multirow is left out, since Q5_K is not an R3 kernel.

pv 0.70.0 after 008: validate 0/0; lint 0 errors and 5 lean_theorem warnings (the new one is `matvec_row_parity`). Obligations went from 7 to 8.

pv 0.70.0 after F and G: validate 0/0; lint 0 errors and the same 5 lean_theorem warnings. Obligations stay at 8; the 008 obligation now ranges over tokens too.

BPM-012 and the C4 leg (2026-10-03). R1 FALSIFY-BPM-012 now compares `kernel_id` per (tensor, op) and ignores `arch`. So a second machine alone does not meet it: one scalar kernel on both hosts is a self-comparison (P1 f012c).
- This leg meets it through the c4_e1_leg precondition that every C4 Q4_K and Q6_K GEMV entry names a `/neon` kernel. dispatch_honesty forbids an x86 run to name one.
- At 316dee2cd4 the f32 Q4_K and Q6_K dots have no aarch64 arm (fused_k.rs:201, fused_q5k_q6k.rs:123). The C4 entries are scalar, so the leg is not_measured until R3 lands.
- The gap: on fp32_act, run token by token like the R1 reference, the C4 run reaches entries A, B and E only. The default Q8_K route C (parallel_k.rs:303-304, where `DIRECT_FP32_GEMV=1` also selects f32), D, F and G are never part of the leg. Their NEON kernel, q4k-q8k, is checked per kernel only (neon_scalar_parity, NEON-Q4K-007, -008).
- A C4 leg on the default route, against the C0 fp32_act reference the way a GPU cell runs, is the next draft (handoff Next, item e).

pv 0.70.0 after this: validate 0/0; lint 0 errors and the same 5 lean_theorem warnings. Counts unchanged (5 equations, 9 falsifiers, 3 KANI, 8 obligations).
