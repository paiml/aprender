# Specification: Sovereign Stack Ship-Two-Models — Shared Methodology & Foundation

**Document ID:** SPEC-SHIP-SHARED
**Version:** 1.0.0
**Parent:** [Ship Two Models Index](./ship-two-models-spec.md)
**Companion specs:**
- [MODEL-1 spec](./ship-model-1-spec.md) — Distilled Qwen2.5-Coder-7B (apr-leaderboard)
- [MODEL-2 spec](./ship-model-2-spec.md) — Sovereign 370M Python student (albor)

This file collects the **foundation sections** (abstract, motivation, design principles, execution plan, risk matrix, failure protocol) and the **cross-cutting methodology + falsifier infrastructure** that applies to both models. Model-specific amendments live in the companion specs.

> **Section numbering**: per-section `§N` markers are preserved verbatim from the original `ship-two-models-spec.md` v3.28.0. Numbering is not contiguous within this file; each section retains its historical number so cross-references and git-log mentions remain valid.

---

# Specification: Ship Two Models — Sovereign AI Stack Proof

**Document ID:** SPEC-SHIP-TWO-001
**Version:** 3.28.0
**Atomic next action (v3.28.0):** **🎯 §82 — P2-A 5000-step training EARLY-STOP at val_loss=4.7111 (epoch 20); P0-trio dispatched against best checkpoint; AC-SHIP2-009 LIVE-DISCHARGED at 325.1 tok/s; AC-SHIP2-010 BLOCKED on NEW P0-G defect (2026-05-15)** (see new §82 below). After §80's EV ranking placed P2-A at the queue head and §81's P0-trio infrastructure fixes (#1699 P0-F + #1701 P0-D/E) landed on main, P2-A dispatched on lambda-vector RTX 4090 against §77's qwen-v2 corpus + Qwen-0.5B init: 27 epochs / 2700 steps / ~40 min wall / OK EARLY_STOP. **§34 ceiling broken further: 9.38 → 5.36 (§78) → 4.71 (§82)** — three orders of MODEL-2 progress in a 16-day arc. P0 trio against `epoch-020.apr`: **P0-A apr qa** infra-pass (only golden_output fails — expected for pretrain-only); **P0-B apr bench PASSED at 325.1 tok/s** with embedded BPE tokenizer + C-03 metadata gate satisfied (confirms #1701 fixes live in production); **P0-C apr export PASSED** (291 tensors, GGUF, lowercase `llama` arch confirms #1699 live). **P0-C step 2 llama-cli load BLOCKED by NEW Class 3 defect P0-G**: GGUF metadata has `llama.vocab_size=151936` but `tokenizer.ggml.tokens=[len=151643]` — Qwen2.5 pads embed_tokens to 151936 for TP-alignment but `apr export` emits the unpadded tokens array. Fix scope: 30-LOC pad in `gguf_export_config.rs::build_tokenizer_gguf_metadata` with `<|pad_N|>` placeholders. **Methodology lesson #29 NEW**: Class 3 packaging defects surface in waves of 4, not 2 — every downstream tool falsifies its own invariant in the checkpoint-emission contract. Sample generation at val_loss=4.71 is repetitive-token gibberish (`def fibonacci(n):` → ` č č č č ...`), confirming P1-B/C eval gates are DEAD until val_loss < 4 — supports §80's deprioritization decision empirically. **MODEL-1 ship %**: 100%. **MODEL-2 ship %**: **77% → 79%** (+1 for AC-SHIP2-009 DISCHARGED, +1 for §34 ceiling break 5.36→4.71). Bounded path to 85%: P0-G landed (79→81) → P2-A2 longer/wider corpus (81→85 if val_loss < 3.5).
**Atomic next action (v3.27.0):** **📚 §79 + §80 + §81 — audit retrospective + prioritized backlog + P0 packaging-gap surfacing (2026-05-15)** (see new §79/§80/§81 below). Triple-amendment captures the §78 → §80 dispatch arc that revealed a Class 3 packaging-defect wave in `apr pretrain` output. §79 synthesizes [`docs/specifications/two-model-spec-audit.md`](../two-model-spec-audit.md) with Five-Whys for Cases A/B/C and methodology lesson #26 (three-class root-cause taxonomy: data starvation / optimization defects / infrastructure masking). §80 ranks all open SHIP-TWO-001 work by ship-% delta ÷ effort: P0 trio + P1 Chinchilla gate + P1 python validity + P1 HumanEval + P2-A long train = MODEL-2 ceiling 92% at ~6-10h compute. §81 surfaces three `apr pretrain` output metadata gaps discovered when dispatching §80's P0 trio: missing embedded tokenizer (blocks `apr qa`), missing arch metadata keys (blocks `apr bench`), HF→GGUF arch case mismatch (blocks llama-cli). Companion code PRs #1699 (P0-F arch case) and #1701 (P0-D embed tokenizer + P0-E arch metadata) close all three. **Methodology lessons #26-28 NEW**: three-class root-cause taxonomy / prioritize by ship-% delta ÷ effort / Class 3 defects come in waves (training works ≠ checkpoint is usable). **MODEL-1 ship %**: 100%. **MODEL-2 ship %**: 75% (unchanged in this PR; will move to 77% on #1701 merge via AC-SHIP2-010 DISCHARGED at 315.5 tok/s).
**Atomic next action (v3.24.0):** **🎯 §78 — 5g.2 CONVERGED — MODEL-2 fine-tune from Qwen-0.5B init produces val_loss=5.36 on 500 steps / 8 min GPU; §34 ceiling broken by 4.02pp; MODEL-2 ship % 57% → 75% (2026-05-15)** (see new §78 below). After §77 retroactively discovered 5g.1 was already complete, 5g.2 was dispatched on RTX 4090 with the canonical Qwen-0.5B init + qwen-v2 corpus. Convergence trajectory: 6.53 → 6.30 → 5.93 → 5.55 → **5.36** val_loss across 5 epochs / 500 steps / 8 min wall. 5 APR checkpoints produced, all integrity-valid (291 tensors / Llama / checksum_valid). **Compared to §49's same-step from-scratch baseline (val_loss=9.73): 44.9% loss reduction — §49 pivot empirically validated.** Discharges AC-SHIP2-003 (vs §34 ceiling), AC-SHIP2-004 (8 min ≪ 21 days), AC-SHIP2-005 (5 valid checkpoints). Newly operator-dispatchable: AC-SHIP2-006/007/008/009/010 against `epoch-004.apr`. **Methodology lesson #25 NEW**: pretrained-init fine-tune dominates from-scratch on small compute (44.9% loss reduction same-budget). First MODEL-2 ship-% movement since §22 (twenty-one days, fifty-six amendments). **MODEL-1 ship %**: 100%. **MODEL-2 ship %**: **57% → 75%**.
**Atomic next action (v3.23.0):** **🔍 §77 — 5g.1 RETROACTIVELY DISCOVERED COMPLETE; MODEL-2 ship-blocker reduced to 5g.2 GPU dispatch (2026-05-15)** (see new §77 below). Live audit on 2026-05-15 finds `/mnt/nvme-raid0/data/codeparrot-python-permissive-shards-qwen-v2/` contains 125 shards / 1,241,692,519 tokens / 4,966,770,076 bytes — byte-exact integrity verified (tokens × 4 = bytes, u32 LE). Manifest confirms NFC + between-doc EOS + Qwen vocab + 405,904 documents from the permissive corpus. **5g.1 has been DONE since ~2026-05-05** but never recorded as complete in twenty subsequent spec amendments. The cascade was always blocked on 5g.3, not on 5g.1. 5g.2 (500-step fine-tune dispatch) is now operator-dispatchable today — all three prerequisites confirmed on disk: Qwen-tokenized corpus, Qwen tokenizer dir, Qwen-0.5B init APR. **Methodology lesson #24 NEW**: mid-run progress logs are not completion records; manifest.json is the contract for "done". **MODEL-1 ship %**: 100%. **MODEL-2 ship %**: unchanged at **57%** (this is a status-discovery, not an evidence-of-training; the flip is gated on 5g.3 verdict).
**Atomic next action (v3.22.0):** **🚢 §76 — v0.33.0 cascade PUBLISHED — MODEL-1 in users' hands; 24 crates live on crates.io; /dogfood verdict GO (2026-05-14)** (see new §76 below). 24-crate topological cascade (contracts-macros → core → gpu → compute → serve → train → apr-cli → aprender root) all published to crates.io. `cargo install aprender --force --locked` from registry produces `apr 0.33.0` that runs SHIP-007 fix end-to-end (`apr run` "What is 2+2?" → "4" on 1.5B teacher). /dogfood 12-gate audit on installed binary: **all gates GO, zero FAILs**. Two production-blockers surfaced + closed in flight: PR #1670 (`cc 1.2.59 → 1.2.62` lockfile bump for rustc 1.93.0 — `cargo publish` re-resolves Cargo.lock during verify, ignoring workspace lock; **methodology lesson #23 NEW**: use `--locked` on every publish or bump-before-cascade); `make publish` `.cargo/config.toml` backup race on parallel invocations (mitigated by serialization; Makefile fix deferred). Companion PR #1672 brings README/book/CLAUDE.md in sync (counts 1105→1134, 80→82; SHIP-007 known-issue warning retired). Closes `feedback_post_publish_qa_required.md` requirement (v0.31.1 yank lesson). **MODEL-1 ship %**: 100% (CODE) → **100% (USERS)** — milestone moves from "shipped in code" to "shipped to users." **MODEL-2 ship %**: unchanged at **57%** (independent track; v0.33.0 carries no MODEL-2 movement).
**Atomic next action (v3.21.0):** **🎉 §75 — MODEL-1 SHIP %  = 100% — SHIP-007 LIVE-DISCHARGED via F32 GEMV PTX layout fix (2026-05-13)** (see new §75 below). PR-E (#1651) ships single-file fix in `crates/aprender-gpu/src/kernels/gemv/mod.rs`: the F32 GEMV kernel assumed `[K rows × N cols]` row-major but actual ML weights are `[output_dim=N, input_dim=K]` row-major (PyTorch/SafeTensors/GGUF convention). Kernel was reading TRANSPOSED weights → systematically anti-correlated logits (cos=-0.005). Fix rewrites inner loop to iterate K within row `block_id`. Empirical discharge: `apr bench` 5-iter 128-tok decode = **124.6 tok/s** on RTX 4090 (4.15× over AC-SHIP1-007 30 tok/s floor); PARITY-GATE PASS; default path, no workarounds. **All 10 AC-SHIP1-* LIVE-DISCHARGED.** **MODEL-1 ship %**: **99% → 100%** 🎉. **MODEL-2 ship %**: unchanged at **57%**. **Methodology lesson #22 NEW**: symptom analysis → bug class localization in O(1); methodology lessons compose.
**Atomic next action (v3.20.0):** **§74 — SHIP-007 bug LOCALIZED to LM head F32 GEMV via PR-B stage bisection (2026-05-13)** (see new §74 below). PR-B (#1649) APR_GPU_STAGE_DUMP scaffold captured GPU embedding + post_ffn_residual L27 + final_norm + lm_head + CPU lm_head on single BOS token. GPU intermediate values look numerically sane (post_ffn_residual rms=26, final_norm rms=2.84). Divergence emerges between final_norm and logits: GPU logits mean=0.013 vs CPU mean=-2.42 (Δ=2.43; CPU has Qwen's typical negative-bias signature). PMAT-333 dequantizes ALL weights to F32 on GPU upload (28.3 GB), so `WeightQuantType::from_size` returns F32 for LM head → dispatches `f32_gemv_into`. The F32 GEMV kernel is the localized bug surface. **Methodology lesson #21 NEW**: stage-by-stage numerical analysis can localize bug class without per-element diffing. **MODEL-1 ship %**: unchanged at **99%** (Layer 2 localized; PR-E for fix). **MODEL-2 ship %**: unchanged at **57%**. Path-to-100% reduced to a single PR-E.
**Atomic next action (v3.19.0):** §72 + §73 combined banner — see both sections below.
**Atomic next action (v3.18.0 §73):** **§73 — SHIP-007 cascade reduced from 3 layers to 1 on re-measurement; only Layer 2 (parity) blocks (2026-05-12)** (see new §73 below). §63's 2026-05-11 3-layer blocker stack — (1) FP8 warmup ILLEGAL_ADDRESS, (2) GPU-vs-CPU parity cos=-0.005190, (3) throughput 5.6 vs 30 tok/s floor — re-measured on 2026-05-12 lambda-vector RTX 4090 reveals 2 of 3 layers already discharged: **Layer 1 fixed** (`[PMAT-082] cuBLASLt FP8 JIT warmed (3584×16×3584)` succeeds), **Layer 3 meets floor** (54.5 tok/s @ 128-tok decode, 5-iter median, 1.82× headroom). Only **Layer 2 still blocks** (byte-identical cos=-0.005190 signature). Path to SHIP-007 LIVE-discharge reduced from "5-10 PR / 1-2 week cascade" to **"3-5 PR / 3-5 day single-layer fix"** — add `forward_gpu_traced` → wire `apr trace --device gpu --save-tensor all` → diff CPU vs GPU stage tensors → fix localized stage → discharge. **Methodology lesson #20 NEW**: re-measure cascade layers before continuing; stale state can be reduced cheaply. **MODEL-1 ship %**: unchanged at **99%** (Layer 2 still blocks). **MODEL-2 ship %**: unchanged at **57%**.
**Atomic next action (v3.18.0 §72):** **§72 — 5-AC LIVE-evidence cascade SHIP-001/003/004/009/010 PARTIAL→LIVE-DISCHARGED (2026-05-12)** (see new §72 below). Single ~30-min session captured LIVE evidence for 5 ACs that were PARTIAL_ALGORITHM_LEVEL (had falsifier code, no LIVE-evidence on canonical teacher): SHIP-001 (`apr run <safetensors>` exit 0 + 62.55s load), SHIP-003 (`apr diff` 20 tensors at `cos_sim=1.000000` vs floor 0.999), SHIP-004 (llama-cli on Q4_K_M GGUF: exit 0, "Hello! How can I help you today", 133.1 gen tok/s), SHIP-009 (`apr inspect`: `license: Apache-2.0`, `data_source: huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct`), SHIP-010 (sha256 `0a854098…` == HF lfs.oid). No code changes — pure evidence cascade. **MODEL-1 ship %**: **95% → 99%** (9/10 AC-SHIP1-* LIVE-discharged; only SHIP-007 remains as multi-PR CUDA cascade per §63). **Methodology lesson #19 NEW**: algorithm-level falsifiers + small evidence runs collapse PARTIAL→LIVE in batches. **MODEL-2 ship %**: unchanged at **57%**.
**Atomic next action (v3.17.0):** **§71 — SHIP-005 LIVE-DISCHARGED at 86.59% pass@1 on gx10 164-run with §70 RC3 fix (2026-05-12)** (see new §71 below). Empirical result on canonical 7B Qwen2.5-Coder-Instruct Q4_K APR teacher: **142/164 problems passed → pass@1 = 86.59%**. AC-SHIP1-005 floor is 84.80% (86.0% nominal with 1.2% tolerance). **Headroom above floor: +1.79pp**. Pre-fix (§67) was 80.49% (132/164); RC3 fix flipped 10 more problems = +6.10pp gain. pass@10 ≈ 100%, pass@100 = 100% — model is fully capable. **MODEL-1 ship %**: **94% → 95%** (4/5 §17.5 PARTIALs LIVE-discharged: SHIP-002/005/006/008; remaining SHIP-007 is multi-PR CUDA cascade per §63). **MODEL-2 ship %**: unchanged at **57%**.
**Atomic next action (v3.16.0):** **§70 — §69 RC3 CONFIRMED on gx10 + FIX DISCHARGED via 3/3 §68-trio flips (2026-05-12)** (see new §70 below). Empirical disambiguation on gx10 via `APR_EVAL_DEBUG=1` (PR #1634 diagnostic surface): HumanEval/1 `exit_code=1, stderr="NameError: name 'List' is not defined"`. **RC3 (format!() drops imports) CONFIRMED**; RC1/RC2/RC4 FALSIFIED. PR #1635 1-PR fix: new `extract_prompt_preamble(prompt, entry_point)` helper + ChatML-branch prepend. Discharge proof — rerun §68's known-failed trio (HumanEval/1, /3, /6): **3/3 flip to PASS** (all `exit_code=0`). §68's "Class B sampling/quantization" interpretation FALSIFIED — those were Class C harness-RC3 false-failures all along. **Methodology lesson #17 NEW**: pre-fix RED smoke can mask the bug class; diagnostic instrumentation (not flip rate) identifies the class. **MODEL-1 ship %**: stays at **94%** pending 164-run completion; path to 95% is now a single 164-run + verdict check, no further code changes needed. **MODEL-2 ship %**: unchanged at **57%**.
**Atomic next action (v3.15.0):** **§69 — Q4K hypothesis FALSIFIED; bug is in the `apr eval` harness, not the model (2026-05-12)** (see new §69 below). 4-step smoking-gun on HumanEval/1: (1) `apr run` emits 50-line response with valid `\`\`\`python\`\`\`` code block; (2) extracted code passes manual `python3` test with exit 0; (3) `apr eval` on same problem reports FAIL; (4) Rust `extract_python_code_block_targeted` returns identical code as Python regex. Conclusion: bug is between Rust extraction and Python test verdict — **HARNESS, not model**. Q4K hypothesis (§67/§68) FALSIFIED. R3 (FP16) and R4 (sampling) DEPRIORITISED. Four candidate root causes surface: **RC1** (apr eval produces different completions than apr run — model state leak), **RC2** (`execute_python_test` false-negative), RC3 (`format!()` bug), RC4 (max_tokens truncation). **Methodology lesson #16 NEW**: compose falsifiers via manual end-to-end replication — saves 10h of wrong-hypothesis investigation. **MODEL-1 ship %**: stays at **94%**; path to 95% requires diagnosing the harness bug (RC1-RC4), NOT model changes. **MODEL-2 ship %**: unchanged at **57%**.
**Atomic next action (v3.14.0):** **§68 — R1+R2 robustness baseline shipped (PR #1630); 3-problem smoke reveals failures are Class B (sampling/quantization), not Class A (extraction) (2026-05-12)** (see new §68 below). R1 (multi-block extraction) + R2 (function-targeted, `def {entry_point}(` preferred) shipped as the cheapest 1-PR refinement candidate from §67's R1-R4 menu. Empirical 3-problem LIVE smoke on gx10 against known-failed HumanEval/1/3/6: **0/3 flip** — model emits SINGLE fenced blocks with subtly-wrong solutions, not multi-block explanatory snippets. R1+R2 didn't help these three. Refined scope: SHIP-005's 4.31pp gap now requires **R3 (Q4K→FP16, needs separate artifact)** or **R4 (temperature=0.2 + 3 samples, ~17h gx10 compute)** to close — R1+R2 is the necessary robustness baseline but insufficient on its own. **Methodology lesson #15 NEW**: smoke-test-driven scope reduction — a 3-problem smoke saves 5h compute by upper-bounding refinement gain BEFORE the full rerun. **MODEL-1 ship %**: stays at **94%** (bounded path to 95% now requires R3 or R4 — multi-day work). **MODEL-2 ship %**: unchanged at **57%**.
**Atomic next action (v3.13.0):** **§67 — H4 fix LIVE result: pass@1 = 80.49% on gx10 164-run (+46pp gain, 4.31pp below floor) (2026-05-12)** (see new §67 below). PR #1628 H4 fix (ChatML wrap + `extract_python_code_block`) shipped; gx10 164-run on canonical 7B APR teacher took 5.8h CPU wall → 132/164 = **80.49% pass@1**. Up from 34.15% (§65) = **+46pp gain**. pass@10 ≈ 100%, pass@100 = 100% — model fully capable; SHIP-005 stays PARTIAL but gap is now **refinement-scale (4.31pp)**, not fundamental. Four refinement candidates surface: R1 (extraction robustness, est 2-3pp), R2 (function-targeted extraction, 1-2pp), R3 (Q4K→FP16 quantization, 2-3pp), R4 (sampling refinement, 1-2pp). R1+R2 are cheapest (eval-harness code + 5h gx10 rerun). **Methodology lesson #14 NEW**: near-miss results bound refinement scope (50pp gap = methodology; 4pp gap = refinement). **MODEL-1 ship %**: stays at **94%**. **MODEL-2 ship %**: unchanged at **57%**.
**Atomic next action (v3.09.0):** **§63 — SHIP-007 empirical floor — CUDA structurally broken on Qwen 7B; multi-PR cascade scope (2026-05-11)** (see new §63 below). LIVE `apr bench` on canonical 7B APR teacher surfaces a 3-layer blocker stack for SHIP-007 (decode tps ≥ 30 tok/s): (1) `CUDA_ERROR_ILLEGAL_ADDRESS` in cuBLASLt FP8 JIT warmup (workaround: `APR_SKIP_FP8_WARMUP=1`); (2) PARITY-GATE rejects with cosine = -0.005 because GPU forward computes a DIFFERENT function than CPU on Qwen2.5-Coder-Instruct dimensions (hidden=3584, heads=28, kv_heads=4); (3) even with both gates skipped, throughput is 5.6 tok/s (well below 30 floor). SHIP-007 is multi-PR cascade scope, not a 1-PR LIVE-discharge. **Methodology lesson #11 NEW**: an unblocking closure (§60) may transitively unblock SOME §17.5 PARTIALs (SHIP-002/006/008, and likely SHIP-005 from in-progress 164-run) but leave OTHERS requiring their own multi-PR cascades. **MODEL-1 ship %**: stays at **94%** (pending 164-run → SHIP-005 → potentially 95%). SHIP-007 estimated to flip 95% → 96% on multi-PR cascade close. **MODEL-2 ship %**: unchanged at **57%**. Coverage tally: snapshot + empirical-floor record + 3-layer blocker bound (no new falsifier flips this cycle).
**Atomic next action (v3.06.0):** **§61 — Post-§60 LIVE-discharge cascade — direct-prompt SHIP-002 GREEN; ChatML-prompt SHIP-006/008 surface a generation-quality gap (2026-05-10)** (see new §61 below). §60 closure unblocked the §17.5 chain. This session shipped the SHIP-002 LIVE discharge (PR #1609) — `apr run --prompt "def fib(n):" --max-tokens 128` on canonical 7B APR teacher emits coherent fib() Python with 0 syntax errors / 68 AST nodes / 1 FunctionDef. But the parallel `apr qa` LIVE attempt surfaced a NEW empirical finding: the SAME canonical teacher fails the `golden_output` gate ("gibberish, fragment '\\ns\\ns' repeats 3+ times") under the ChatML-wrapped prompt `<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n`. Forward-parity (§60) ≠ generation parity. SHIP-006/008 blocked on this ChatML degenerate-output bug; SHIP-007 separately blocked on perf (8.8 tok/s vs 30 floor on CPU fallback path). §61 records the two falsifiable predictions for the next bisection: PRED-61-A (GGUF + ChatML → CLEAN? localizes bug to APR side); PRED-61-B (APR + direct continuation "What is 2+2? The answer is " → CLEAN? localizes bug to special-token handling vs cumulative drift). Cascade-this-session: 6 PRs (#1604/#1606/#1607/#1608/#1609 + this §61). **MODEL-1 ship %**: **91% → 92%** (1 of 5 §17.5 PARTIALs LIVE-discharged via #1609; SHIP-005/006/007/008 stay PARTIAL). **MODEL-2 ship %**: unchanged at **57%** until step 5g.3 produces val_loss < 9.38. Coverage tally: 1 new LIVE discharge (SHIP-002 in `qwen2-e2e-verification-v1.yaml` v1.10.0 → v1.12.0); plus 1 status flip (`apr-vs-gguf-forward-parity-v1` v1.1.0 → v1.2.0 PROPOSED → ACTIVE_FUNCTIONAL via PR #1608); plus 3 cascade fixes in `aprender-train` CUDA forward path (Q/K/V bias dispatch / RMSNorm eps cache key / RoPE theta cache key — PRs #1604/#1606/#1607).
**Atomic next action (v3.05.0):** **§60 — SHIP-007 §22 FULLY CLOSED — H1 CONFIRMED apples-to-apples on canonical 7B teacher; layer-3 ratio 18.23× → 1.245× (2026-05-07)** (see companion-spec entries M91-M103 + parity #89 for full per-PR narrative; aprender contract `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.0.0 → v1.13.0 across 13 amendments). M-FFN-GGUF-5 fix shipped (aprender PR #1550 squash pending) + M-FFN-GGUF-7 multi-layer real-teacher chain shipped (aprender PR #1548 MERGED). **MAJOR PLOT TWIST in M103 fix PR**: §27's 18.23× std-ratio was a TEST METHODOLOGY ARTIFACT, NOT a numerical bug. GGUF's `forward_traced` does Phase 1 prefill silently and only captures stats on the LAST token; APR's `forward_traced` captured stats across ALL 7 tokens. The §27 measurement compared multi-token APR std (7-token × 28672 elements) vs single-token GGUF std (1-token × 4096 elements) — fundamentally incomparable distributions. **Two coherent fixes in M-FFN-GGUF-5 PR #1550**: (1) `forward_traced` now uses Q4K+Q8K dispatch via new helper `matmul_q4k_or_f32_traced` (multi-token aware, F32 fallback when Q4K unavailable, 7 call sites updated); (2) M89 harness compares APR's `last_token.ffn_swiglu_inner_stats` against GGUF's `ffn_swiglu_inner_stats` (apples-to-apples last-token-only on both sides). **EMPIRICAL END-TO-END VERIFICATION** (2026-05-07, lambda-vector RTX 4090, 178s wall): all 28 layers within H1 band [0.5, 2.0]; **layer-3 ratio = 1.245×** (was 18.23× pre-methodology-fix). **Verdict flipped: H2 (apparent APR-side bug) → H1 CONFIRMED (apples-to-apples agreement)**. The cascade's per-tensor mechanism (M94 0.077% Path A vs Path B per matmul) and compounding (M95 5.70× synthetic / M-FFN-GGUF-7 1.81× real-saturating) ARE real numerical findings — but the §27 1723% magnitude that made the bug look severe was test-methodology-inflated. **M-FFN-GGUF-7 finding** (M102 PR #1548): real-layer chain SATURATES at 1.81× over 5 layers (vs synthetic M95's 5.70×); Layer 2 drops to 0.029% from weight-pattern cancellation; naive growth-factor exponentiation gives 1.81^22.4 = 5.78e5× at 28-layer depth — physically impossible; real systems saturate. **Methodology lesson #7 NEW** (`feedback_test_methodology_can_fake_bugs.md`): when comparing two implementations via summary statistics (std/mean/cosine), VERIFY both sides measure the SAME distribution shape (count, dim, element selection) BEFORE trusting the comparison. Mismatched distribution shapes can amplify a small real divergence into an apparent magnitude that looks like a bug. SHIP-007 §22 burned ~3 weeks pre-cascade + 2 days cascade + 2 hours fix on a methodology issue that produced a fake apparent magnitude on top of the real per-matvec mechanism. **15,233 lib tests pass, 0 failures**; production hot paths byte-unchanged (only `forward_traced` touched in PR #1550). **Discharge potential**: per §17.5, M-FFN-GGUF-5 closure transitively enables individual discharge of 5 MODEL-1 PARTIALs (SHIP-002, SHIP-005, SHIP-006, SHIP-007, SHIP-008); each may need its own contract-level promotion follow-up. **MODEL-1 ship %**: 91% → **96% pending individual partial discharges**. **MODEL-2 ship %**: unchanged at **57%** until step 5g.3 produces val_loss < 9.38. Coverage tally: 12 falsifiers + 1 fix DISCHARGED across `trace-ffn-sub-block-gguf-v1` v1.0.0 → v1.13.0 cascade. **Total session: 28 PRs across 2 days** including 1 actual fix landing.
**Atomic next action (v3.04.0):** **§59 — SHIP-007 §22 falsifier cascade CLOSED — 11 PRs (M91-M101) decompose §27 1723% within rounding; fix scope EMPIRICALLY VALIDATED as Option-A (2026-05-06+07)** (see companion-spec entries M91-M101 in `claude-code-parity-apr/docs/specifications/claude-code-parity-apr-poc.md` for the full per-PR cascade narrative; aprender contract `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.0.0 → v1.12.0 across 12 amendments). Two-day autonomous /loop session shipped 11 lib-test + 1 integration-test falsifiers (aprender PRs #1535/#1536/#1537/#1538/#1540/#1541/#1542/#1543/#1544/#1545) decomposing the §27 layer-3 ffn_swigl 18.23× APR-vs-GGUF std-ratio (=1723% deviation from 1.0). **Final empirical decomposition (2026-05-07)**: 0.077% per-tensor mechanism (M94, FALSIFY-FFN-GGUF-008 — first CONFIRMED bit-divergence between APR's standalone-dequant + F32-matmul "Path A" semantics vs GGUF's Q8K-activation-quant + fused-inline-dequant "Path B" semantics on synthetic 144-byte Q4K super-block) × 5.70× super-linear compounding (M95, 5 chained matvecs grow 0.077% → 0.4391%) × 50× std-ratio measurement sensitivity (M99, batch-dimension std measurement vs per-tensor rel_diff) × 5.56× LIVE real-teacher amplification (M100, FALSIFY-FFN-GGUF-014 LIVE on canonical 7B Qwen2.5-Coder-Instruct-Q4_K_M layer-3 ffn_down_weight Q4K bytes from `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr`: Path A=-1.658492 [`0xbfd44977`] vs Path B=-1.665596 [`0xbfd5323e`], rel_diff 0.428%) × 14× residual = ~1715% — **within rounding of §27's 1723%**. **Six synthetic amplifier candidates resolved**: A1 (RoPE phase, M98) FALSIFIED 1.00× UNITARY; A2 (softmax saturation, M97) FALSIFIED 0.01× COMPRESSES; A3 (block-scale variance, M96) FALSIFIED 1.00× SCALE-INVARIANT; A4 (multi-token batch, M99) FALSIFIED 0.26× per-token PLUS 50× std-ratio measurement sensitivity finding; A5 (real-weight non-uniformity, M100) **PARTIALLY CONFIRMED 5.56× LIVE on canonical 7B**; A6 (RMSNorm rsqrt, M101) FALSIFIED 1.00× HOMOGENEOUS. **14× residual gap is now attributed entirely to cumulative-layer interaction** (synthetic single-layer + homogeneous-RMSNorm tests cannot capture it; M-FFN-GGUF-7 multi-layer real-teacher chain is the only remaining test path but does NOT block fix PR). **SHIP-007 §22 fix scope EMPIRICALLY VALIDATED as Option-A (PROMOTE GGUF-PATH semantics into APR forward)**: switching APR's `f32_matmul` to Q8K activation quant + fused matvec semantics will recover the 5.56× per-matvec amplification on every matmul, eliminating cumulative APR-vs-GGUF drift. Estimated fix scope ~250-400 LOC; transitively discharges 5 MODEL-1 PARTIALs (SHIP-002, SHIP-005, SHIP-006, SHIP-007, SHIP-008) per §17.5. Cascade methodology lessons consolidated to `~/.claude/projects/-home-noah-src-aprender/memory/feedback_falsifier_cascade_decomposes_magnitude.md` and `feedback_falsifier_chain_assert_difference.md`. **MODEL-1 ship %**: unchanged at **91%** until M-FFN-GGUF-5 (the actual fix PR) lands. **MODEL-2 ship %**: unchanged at **57%** until step 5g.3 produces val_loss < 9.38. Coverage tally: 11 new falsifiers DISCHARGED across `trace-ffn-sub-block-gguf-v1` v1.0.0 → v1.12.0 cascade.
**Atomic next action (v3.03.0):** **§58 — v0.32.0 cascade publish + release-engineering hygiene snapshot (Issue #1514 CLOSED, 6 PRs, 4 hidden defects surfaced + closed) (2026-05-05)** (see new §58 below). Issue #1514 (v0.32.0 cascade publish) CLOSED at 16:14:56Z. Four user-facing crates now live on crates.io at v0.32.0: `aprender`, `aprender-rag`, `aprender-core`, `apr-cli` (verified via `cargo search`). Cascade surfaced 4 release-engineering defects, all closed in their own PRs: #1512 (aprender-rag `[lib] name = "trueno_rag"` → `"aprender_rag"` BREAKING — `use aprender_rag::*` was uncompilable in v0.31.x), #1513 (aprender-orchestrate `cmd_code` 7→8 arg drift on upstream `emit_trace` addition), #1515 + #1517 (aprender-core dev-dep publish-time cycle: path-only and then permissive `version = ">=0.27"` + path, after clean-room sed-strip left invalid `{ package = "..." }` entries), #1518 (apr-cli `include_str!("../../../../configs/aliases.yaml")` failed cargo publish — files outside crate dir excluded; fix copies aliases.yaml into `crates/apr-cli/configs/`). PR #1511 ships `pv lint --strict-test-binding`, closing §57.4's foreshadowed prevention rule. 5g.1 corpus retokenize (PID 2767124) at 62 shards / 16h19m wall (past initial 57-shard estimate; rate ≈ 15-16 min/shard; manifest pending end-of-run). **MODEL-1 ship %**: unchanged at **91%**. **MODEL-2 ship %**: unchanged at **57%** until step 5g.3 produces val_loss < 9.38. Coverage tally: snapshot (release-engineering hygiene, not falsifier flip).
**Atomic next action (v3.02.0):** **§57 — drift sweep cleans §50.4 cascade contracts (3 PRs); 5g.1 full corpus run on track (2026-05-05)** (see new §57 below). Three same-class drift fixes shipped this session — PR #1502 (apr-pretrain-arch-polymorphic-v1 v1.4 binds CUDA-001), PR #1505 (apr-pretrain-arch-polymorphic-v1 v1.5 fixes FALSIFY-005/006 names), PR #1506 (apr-cli-tokenize-import-hf-v1 v1.1 binds FALSIFY-001 with integration test). PR #1504 (apr-pretrain-from-init-v1 v1.2 drift correction) closed the largest instance via operator/agent collaboration. After this sweep, `pv lint contracts/` reports **0 PV-VER-001 errors across all 870+ contracts** — every cited test exists. 5g.1 full corpus retokenization (PID 2767124) progresses steadily at 16.3 min/shard; 13/57 shards complete in 3h22min wall; ETA ~22:00Z (5g.1.3 verdict ~12hr from now). **MODEL-1 ship %**: unchanged at **91%**. **MODEL-2 ship %**: unchanged at **57%** until step 5g.3 produces val_loss < 9.38. Coverage tally: snapshot (drift sweep is hygiene, not falsifier flip).
**Atomic next action (v3.01.0):** **§56 — 5g.1 LIVE smoke: corpus retokenization with Qwen vocab is correctness-validated; full run is ~17hr operator-dispatch (2026-05-05)** (see new §56 below). Smoke ran `apr tokenize encode-corpus` on first 5000 docs of `python-permissive.jsonl` through the §54-extracted Qwen tokenizer dir; produced 13 valid u32 shards (~13M tokens) at ~110 sec / M-token before being killed. 5g.1 is correctness-validated and operator-dispatchable. Wall projection for full 565M-token corpus: ~17 hours single-thread (1.7× legacy 50257-vocab wall — Qwen's 3× larger merge table is the dominant cost). **Below the 48hr `feedback_compute_pre_authorized.md` ceiling.** Full run dispatched 2026-05-05T07:00Z. Spec v3.00.0 → **v3.01.0**. **MODEL-1 ship %**: unchanged at **91%**. **MODEL-2 ship %**: unchanged at **57%** until step 5g.3 produces val_loss < 9.38. Coverage tally: 5g.1 reaches LIVE-SMOKE level; promotion to FULL-VALIDATED waits for full-corpus run + manifest.json.
**Atomic next action (v3.00.0):** **§55 — Polymorphic preflight relaxation: tokenizer_vocab ≤ model_vocab when init=Some; LIVE smoke confirms Qwen extracted tokenizer passes preflight (2026-05-05)** (see new §55 below). §54's LIVE smoke surfaced that public Qwen2.5-Coder-0.5B-Instruct/tokenizer.json materializes 151643 BPE entries + 22 added = 151665 effective strings, but config.json declares vocab_size=151936 (271 reserved/special slots not in tokenizer.json). Strict equality preflight was correct for §24/§25 from-scratch but too strict for HF-distributed pretrained checkpoints with reserved slots. §55 introduces the relaxed bound `tokenizer_vocab ≤ model_vocab` for the polymorphic path (init=Some); strict equality is preserved for the from-scratch path (init=None, regression-free). New helper `assert_tokenizer_vocab_within_model_bound` + extended preflight signature + 4 new tests (2 helper + 2 integration). Contract `apr-pretrain-arch-polymorphic-v1` v1.2.0 → **v1.3.0 FUNCTIONAL** with FALSIFY-009 (relaxed accept) + FALSIFY-010 (oversize reject — OOB safety). LIVE smoke 2026-05-05T05:48Z: rebuilt apr binary + §54-extracted Qwen tokenizer + Qwen 0.5B init APR → preflight PASSED (no GATE-ARCH errors); process proceeded past preflight to weight load (timeout-killed at 30s mid-load). Spec v2.99.0 → **v3.00.0**. **MODEL-1 ship %**: unchanged at **91%**. **MODEL-2 ship %**: unchanged at **57%** until step 5g.3 produces val_loss < 9.38. Coverage tally: 8 → 10 falsifiers in apr-pretrain-arch-polymorphic-v1 (+2 new, all PASS). 5g.1 (corpus retokenize) now technically dispatchable.
**Atomic next action (v2.99.0):** **§54 — Step 5g has multi-step prerequisites; live preflight smoke proves polymorphic gate fires on Qwen --init + legacy 50257-vocab tokenizer (2026-05-05)** (see new §54 below). Live empirical smoke on canonical 0.5B init APR + canonical 565M-token codeparrot corpus + canonical 50257-vocab tokenizer + freshly-built apr binary (commit 92c7e237b post-#1494) FIRED CORRECTLY: `GATE-ARCH-370M-011 (INV-ARCH-370M-006) violated: tokenizer vocab_size (50257) != model vocab_size (151936)`. This is the FIRST end-to-end runtime evidence that the §50.4 cascade's polymorphic preflight (PR #1476) works in the user-facing CLI (FALSIFY-APR-PRETRAIN-ARCH-005/006 reach LIVE-INTEGRATION level beyond unit-test PARTIAL). But the smoke also surfaces 5g's true scope: a Qwen-vocab tokenizer dir + Qwen-tokenized corpus must exist BEFORE the preflight passes — neither exists on this host today. Step 5g is re-scoped from "1 dispatch, 0 LOC" to **5g.0 (Qwen tokenizer extraction, ~50 LOC) → 5g.1 (Qwen-tokenized corpus, multi-hour wall) → 5g.2 (LIVE 500-step fine-tune, 0 LOC operator-dispatch) → 5g.3 (val_loss < 9.38 verdict)**. Spec v2.98.0 → **v2.99.0**. **MODEL-1 ship %**: unchanged at **91%**. **MODEL-2 ship %**: unchanged at **57%** until step 5g.3 produces val_loss < 9.38 evidence. Coverage tally: snapshot + roadmap re-scoping (no contract status flip — the polymorphic preflight evidence reinforces v1.2.0 FUNCTIONAL but doesn't yet promote to DISCHARGED).
**Atomic next action (v2.98.0):** **§53 — §50.4 cascade INTEGRATION-COMPLETE on main; `apr pretrain --init` end-to-end runnable; only 5g LIVE remains (2026-05-05)** (see new §53 below). PR #1494 (§50.4 step 5f.4 CLI wireup) MERGED at 01:48:14Z. The `apr pretrain --init <PATH>` flow is now end-to-end functional on CPU: magic-byte validation → arch extraction via `model_config::read_apr_architecture` → polymorphic preflight with EXTRACTED vocab → `build_shared_trainer_with_init` composing 5f.1 (encoder rejection) + 5f.2 (load) + 5f.3 (populate). The legacy "not yet wired" Err from §49 step 4 is RETIRED. CUDA path fail-fasts with FALSIFY-APR-PRETRAIN-INIT-CUDA-001 (5f.5 follow-up). Contract `apr-pretrain-arch-polymorphic-v1` ready for v1.1.0 PARTIAL_ALGORITHM_LEVEL → v1.2.0 FUNCTIONAL bump (8/8 falsifiers PASS on main, integration verified). **§50.4 cascade ships 11 PRs over 2 days** (#1471/#1472/#1473/#1474/#1475/#1476/#1478/#1479/#1481/#1482/#1483/#1486/#1494). Spec v2.97.0 → **v2.98.0**. **MODEL-1 ship %**: unchanged at **91%**. **MODEL-2 ship %**: unchanged at **57%** until step 5g produces val_loss < 9.38 — but step 5g is now operator-dispatchable (the only blocker resolved). Coverage tally: snapshot, contract status flip pending v1.2.0 bump.
**Atomic next action (v2.97.0):** **§52 — §50.4 cascade ALGORITHM-COMPLETE on main; new step 5f.4 CLI wireup gap identified before 5g LIVE (2026-05-04)** (see new §52 below). Same-day continuation of §51 cascade landed PR #1479 (FALSIFY-APR-PRETRAIN-ARCH-007 encoder/decoder family validator) and PR #1481 (`load_init_tensors_from_apr` in `aprender-train`). #1483 (`populate_trainer_from_init_tensors`, §50.4 step 5f.3) and #1482 (contract `apr-pretrain-arch-polymorphic-v1` v1.0.0 → v1.1.0 PARTIAL_ALGORITHM_LEVEL bump) are MERGEABLE in queue. **All 8 falsifiers in `apr-pretrain-arch-polymorphic-v1` are now bound on main or about to land**: 6 already MERGED (#1474/#1475/#1476/#1478/#1479/#1473), #1483 + #1482 cover the remaining 2 (5f.3 populate + the v1.1.0 contract status bump). **NEW finding from live source inspection of `apr-cli/src/commands/pretrain.rs:259-297`**: even with all helper functions merged (`load_init_tensors_from_apr` + `validate_pretrain_init_arch_compatible` + `populate_trainer_from_init_tensors` + `build_transformer_config` + polymorphic preflight), the CLI dispatch `validate_init_apr_path` HARDCODES an `Err(...not yet wired...)` return — so an operator running `apr pretrain --init <Qwen>.apr` STILL gets a "not yet wired" runtime error. **Step 5f.4 (CLI wireup, ~150 LOC) is the missing connecting step that makes 5g LIVE actually runnable.** Roadmap re-scoped: 5a-5f.3 (algorithm machinery, COMPLETE) → **5f.4 (CLI wireup, NOT YET STARTED)** → 5g (LIVE 500-step fine-tune, gates ship-%) → 5h (stamp + publish). Spec v2.96.0 → **v2.97.0**. **MODEL-1 ship %**: unchanged at **91%**. **MODEL-2 ship %**: unchanged at **57%** until step 5g produces val_loss < 9.38 evidence — but step 5g now requires step 5f.4 to land first. Coverage tally unchanged this cycle (snapshot + roadmap re-scoping, not falsifier flips).
**Atomic next action (v2.96.0):** **§51 — §50.4 cascade snapshot: 7/8 falsifiers PARTIAL_ALGORITHM_LEVEL bound; MODEL-2 ship-% gate narrowed to step 5g LIVE (2026-05-04)** (see new §51 below). Same-day continuation cycle landed 8 PRs across the architecture-polymorphic infrastructure track (§50.4 steps 5a-5f.1). Falsifier scoreboard for `apr-pretrain-arch-polymorphic-v1`: FALSIFY-001 (#1474) qwen2_0_5b matches HF + tie_word_embeddings DEFECT FIX; FALSIFY-002 + 003 (#1475) build_transformer_config polymorphic dispatch; FALSIFY-004 (#1478 MERGED) GQA-7:1 forward smoke; FALSIFY-005 + 006 (#1476 MERGED) polymorphic preflight Qwen vocab; FALSIFY-007 (#1479) encoder/decoder family validator; FALSIFY-008 contract-level pv-validate. Three PRs MERGED (#1472 §50, #1476, #1478); four still in auto-merge queue (#1473 contract, #1474 fix, #1475 dispatch, #1479 validator). **Step 5f.2 (APR weight load + tensor materialization, ~80 LOC) deliberately deferred** to let cascade settle; doing 5f.2 now would mean rebasing onto 4 in-flight PRs as they land. **Step 5g LIVE 500-step fine-tune is the only remaining load-bearing test** that moves MODEL-2 ship-%; everything else is infrastructure. Per §47-§48 lesson: "infrastructure shipped ≠ ship-% movement." Spec v2.95.0 → **v2.96.0**. **MODEL-1 ship %**: unchanged at **91%**. **MODEL-2 ship %**: unchanged at **57%** until step 5g produces val_loss < 9.38 evidence. Coverage tally unchanged (snapshot, not falsifier flip).
**Atomic next action (v2.95.0):** **§50 — MODEL-2 architecture-coupling finding: §49.6 step 5 is multi-PR scope, not single-PR (re-scoped 5a-5h)** (see new §50 below). After §49.6 steps 3 + 4 landed (PR #1470 contract + PR #1471 wire-up), live source inspection of `pretrain_real.rs:38-46` revealed the trainer hardcodes every architectural constant from `Llama370MConfig` (hidden=1024, heads=16/4, ffn=2816, vocab=50_257). Qwen2.5-Coder-0.5B has different shape (hidden=896, heads=14/2, ffn=4864, vocab=151_936, GQA-7:1). Every tensor mismatches; §49.6 step 5's "0 LOC, just run apr pretrain --init" assumption fails. Three options surfaced (A: find/build a Llama-shaped 0.5B checkpoint; B: make trainer arch-polymorphic; C: replace Llama370MConfig with Qwen-shaped). **Recommend Option B** — preserves §24/§25 falsification evidence, exercises `TransformerConfig`'s designed polymorphism, binds each new component (Qwen tokenizer, GQA-7:1, extracted-arch loader) to its own falsifier. Re-scoped roadmap: 5a (new contract `apr-pretrain-arch-polymorphic-v1`) → 5b (TransformerConfig::qwen2_0_5b constructor) → 5c (extract arch from init APR) → 5d (Qwen tokenizer surface) → 5e (GQA-7:1 verification) → 5f (weight load) → 5g (LIVE 500-step fine-tune) → 5h (publish). **Total: ~410 LOC + 1 LIVE run, not 0 LOC.** Spec v2.94.0 → **v2.95.0**. **MODEL-1 ship % unchanged at 91%. MODEL-2 ship % unchanged at 57%** until 5g produces val_loss < 9.38. Coverage tally unchanged (architecture finding, not a falsifier flip).
**Atomic next action (v2.94.0):** **§49 — MODEL-2 strategy pivot: from-scratch was a methodology defect; pretrained-init + fine-tune is the correct path** (see new §49 below). After 11 SHIP-007 cascade PRs without ship-% movement, operator asked "why aren't we training models?" Honest re-diagnosis of the MODEL-2 architecture revealed that §34's "capacity-limited at val_loss=9.38" framing is **wrong** — it's **data-limited**. Live evidence (2026-05-04 session): a fresh 500-step `apr pretrain --mode from-scratch --device cuda` run on the existing 565M-token codeparrot corpus converged to **val_loss=9.7255**, identical to §24's 9.7507 ceiling — confirming the corpus is the binding constraint. Industry comparison: SmolLM-360M (similar param count) hits val_loss ~2.9 but was trained on 1T tokens. MODEL-2 saw 565M. The "from-scratch on 565M tokens" math just doesn't reach val_loss=3.0 regardless of step budget. The right strategy is **initialize from a public pretrained 370M-class checkpoint and fine-tune on the existing corpus** — Qwen2.5-Coder-0.5B-Instruct (already in HF cache at `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/`, 950 MB) is at val_loss ~2-3 already. Fine-tuning on Python+permissive code shifts the distribution without losing the 1T-token pretraining. This pivot is **NOT a punt** — it matches industry best practice (StableCode ← StableLM, Qwen2.5-Coder ← Qwen2.5; nobody trains 0.5B from scratch for production code-LMs because the data efficiency math fails). Spec v2.93.0 → **v2.94.0**. **MODEL-2 ship % stays at 57%** until the fine-tune produces measurable val_loss < 9.38 evidence. **MODEL-1 ship % unchanged at 91%**. Coverage tally unchanged this cycle (strategic amendment, no falsifier flips yet).
**Atomic next action (v2.93.0):** **§48 — SHIP-007 layer-0 attention bisection cascade ALGORITHM-LEVEL COMPLETE (PRs #1455 + #1456 + #1457)** (see new §48 below). Three more PRs after §47 closed the §47.1 cascade roadmap to step 6 of 8 at the algorithm level: (i) PR #1455 — `forward_traced_with_plan` wires 4 attention sub-stages (`QPostRope`, `KPostRope`, `AttnScores`, `AttnSoftmax`); FALSIFY-ATTN-SUB-002 PARTIAL_ALGORITHM_LEVEL; closes the §47.4 parent-contract drift as a side effect (+ memory cost: 112 bytes/forward at BOS). (ii) PR #1456 — drift-prevention test for FALSIFY-ATTN-SUB-003 in `crates/apr-cli/src/commands/diff_05_aprt_stage.rs`; 2 new tests (`falsify_attn_sub_003_new_stages_per_stage_agnostic` + `falsify_attn_sub_003_cosine_detects_softmax_divergence`); pins that `apr diff --values` is per-stage-agnostic for the 2 new stage suffixes. (iii) PR #1457 — extends `scripts/generate_qwen25_coder_fp16_stages.py` with `--with-attn-substages` (default ON) installing per-instance `Qwen2Attention.forward` monkeypatch under `attn_implementation="eager"`; captures the 4 missing stages (`q_post_rope`, `k_post_rope`, `attn_scores`, `attn_softmax`); pre-condition for FALSIFY-ATTN-SUB-004 LIVE bisection now algorithm-bound (BLOCKER_FIXTURE_ABSENT → PARTIAL_ALGORITHM_LEVEL on this PR's merge). Toyota Way correction during research: the pre-impl note estimated 7 missing stages + ~140 LOC; live source inspection of the existing script found 3 already-captured (`qkv_matmul`, `qkv_bias`, `attention`), reducing scope to **4 stages, ~80 LOC**. **Steps 7-8 (LIVE RTX 4090 bisection + root-cause fix) require operator action**: (a) canonical `apr` release binary needs rebuild post-#1451 (the `/mnt/nvme-raid0/targets/aprender/release/apr` rejects `attn_scores` stage today); (b) PyTorch/CUDA driver mismatch on the host blocks `--device cuda` (workaround: `--device cpu` is multi-min but functional). **MODEL-1 ship %**: 91% (cascade is scaffold; ship % moves at SUB-004 LIVE DISCHARGE in step 7). **MODEL-2 ship %**: 57%. Spec v2.92.0 → **v2.93.0**. Coverage tally: 20+32 → **20+36** (+4 PARTIAL_ALGORITHM_LEVEL from `trace-attn-sub-stages-v1` v1.1.0 falsifiers landing on main via #1450; the 5th — SUB-004 — remains BLOCKER until #1457 ships and an operator runs the live RTX 4090 bisection).
**Atomic next action (v2.92.0):** **§47 — SHIP-007 layer-0 attention bisection cascade STARTED (PRs #1450 + #1451 + #1452)** (see new §47 below). Three more PRs ship the §46.7(a) follow-up scaffold: (i) PR #1450 — new contract `trace-attn-sub-stages-v1.yaml` v1.0.0 PROPOSED → v1.1.0 PROPOSED (Toyota Way correction within the same branch). v1.0.0 originally claimed 5 new `SaveTensorStage` variants; live inspection of `inference_trace::save_tensor_stage` showed 3 already exist (`QPostRope`, `KPostRope`, `Attention`) — only **2 are truly new** (`AttnScores`, `AttnSoftmax`). v1.1.0 corrected scope to those 2 + documented the 9-stage `bisection_chain_layer_0` equation across parent + new stages. (ii) PR #1451 — `SaveTensorStage` enum gains the 2 new variants in canonical computation order (`KPostRope → AttnScores → AttnSoftmax → Attention`); 5 new tests for FALSIFY-ATTN-SUB-001 (round-trip, ordering, parser-list); 167/167 inference_trace tests PASS; `cargo check --workspace --lib` clean. (iii) PR #1452 — research evidence note documenting a **pre-existing capture gap discovered while authoring the wire-plan**: `QPostRope` + `KPostRope` are in the parent enum but have NO `emit()` calls in `forward_traced_with_plan`. The parent contract `apr-cli-trace-save-tensor-v1.yaml` v1.4.0 (FUNCTIONAL) silently overstates coverage for those 2 stages. The next-cycle FALSIFY-ATTN-SUB-002 PR will wire **4 stages, not 2**, closing this drift as a side effect. **MODEL-1 ship % unchanged at 91%** (cascade is scaffold; ship % moves when a falsifier flips DISCHARGED, expected at FALSIFY-ATTN-SUB-004 LIVE bisection in a future cycle). **MODEL-2 ship % unchanged at 57%**. Spec v2.91.0 → **v2.92.0**. Coverage tally unchanged this cycle (5 falsifier slots PARTIAL_ALGORITHM_LEVEL added to a NEW contract — these increment coverage when the contract YAML lands on main, which gates on PR #1450 merge).
**Atomic next action (v2.91.0):** **§46 — v0.32.0 release-cut decision: HOLD, gated on SHIP-007 layer-0 attention bisection (PR #1448 in flight)** (see new §46 below). After landing the v2.90.0 §45 milestone (PR #1447 merged), the next decision was whether the 238-commit body of work since v0.31.2 warrants a `cargo publish` cut. **Verdict: HOLD.** The release-readiness audit found exactly one load-bearing blocker — SHIP-007 layer-0 attention divergence is empirically pinpointed (cos=0.99999995 attn_norm → 0.9966 attn_out per memory `2026-05-03 SHIP-007 finding`) but **not yet fixed**, so cutting v0.32.0 today would crates.io-ship a binary where `apr run` on a 7B GPU teacher still emits gibberish unless the user passes `--no-gpu`. The §41-§45 jidoka armor makes the failure visible + fail-closed (which is shippable behaviour), but a user-facing `## [0.32.0]` headline that reads "5/5 DISCHARGE on apr-cpu-vs-gpu-output-parity-v1" implies the GPU correctness hole is closed when in truth it is only contained. Two pre-flight artifacts shipped along with this decision: (i) PR #1448 fills the empty `[Unreleased]` CHANGELOG section with the full session body of work; (ii) PR #1448 also repairs the `bash scripts/check_readme_claims.sh` drift gate which was FAILING on `main` (1096→1105 contracts, 79→80 CLI commands). Per `feedback_post_publish_qa_required.md`, the next cut also requires `cargo install aprender --force` + `/dogfood` GO verdict — v0.31.1 was yanked for skipping that gate. Spec v2.90.0 → **v2.91.0**. Coverage tally unchanged (no falsifiers flipped this cycle; this amendment is a release-decision audit record).
**Atomic next action (v2.90.0):** **§45 — `apr-cpu-vs-gpu-output-parity-v1` 5/5 LIVE DISCHARGE milestone (PRs #1445 + #1446)** (see new §45 below). Two live smokes on canonical Qwen2.5-Coder-7B teacher (RTX 4090, binary built from main @ 817ec0553) closed every falsifier in the parity contract: (i) PR #1445 — default-mode `apr run` smoke fired all three jidoka tags in stderr (CUDA cos=-0.005 + argmax 8127≠334; `Backend: wgpu (Vulkan)`; wgpu cos=0.766 < 0.99) and produced correct CPU output "2 + 2 equals 4." → FALSIFY-CPU-GPU-005 PARTIAL_ALGORITHM_LEVEL → DISCHARGED. (ii) PR #1446 — `--no-gpu` smoke produced 9.02s CPU-only run with zero GPU log lines + correct output → joined #1445 evidence to flip FALSIFY-CPU-GPU-001/002/003 PARTIAL → DISCHARGED + FALSIFY-CPU-GPU-004 FUNCTIONAL → DISCHARGED. **All 5/5 falsifiers in apr-cpu-vs-gpu-output-parity-v1 are now DISCHARGED**; the contract is COMPLETE. Coverage tally: 15+37 → **20+32** (+5 in this 2-PR cycle, the largest single-cycle coverage flip of the SHIP-TWO program). MODEL-1 ship % nudges 89% → **91%** (the silent-gibberish loophole that v5/§40 originated is now both implemented closed AND end-to-end live-verified on the canonical broken-GPU model). The §41 → §43 → §44 → §45 jidoka chain is contract-complete; only the underlying SHIP-007 GPU kernel root-cause fix per §40 remains for full GPU-shipability of MODEL-1. Spec v2.89.0 → **v2.90.0**. Contract apr-cpu-vs-gpu-output-parity-v1 v1.3.0 → v1.5.0 ACTIVE.
**Atomic next action (v2.89.0):** **§44 — FALSIFY-CPU-GPU-005 part b implementation + distill-train 9/9 sweep close (PRs #1442 + #1443)** (see new §44 below). Today's continuation cycle landed two more PRs across both ship tracks: (i) PR #1442 — FALSIFY-CPU-GPU-005 part b live implementation: ~70 LOC inline at `try_apr_wgpu_inference` running a CPU-vs-wgpu cosine probe on the BOS token before the autoregressive loop, with a separate tiny probe_kv_caches (max_seq=2) so the real cache stays uncontaminated; emits `WGPU_FALLBACK_LOG_PREFIX` and returns None on cosine < 0.99 OR any probe error (fail-closed). Contract `apr-cpu-vs-gpu-output-parity-v1` v1.2.0 → v1.3.0 ACTIVE. (ii) PR #1443 — closes the last three falsifiers in `apr-cli-distill-train-v1`: TRAIN-007 + TRAIN-008 PARTIAL_ALGORITHM_LEVEL via existing tests (`pv validate` + `cli_commands::test_no_unregistered_commands`), and TRAIN-009 explicitly marked BLOCKER_FIXTURE_ABSENT pending the §35 real-training implementation. **All 9 TRAIN-* falsifiers now have explicit `algorithm_evidence` blocks** (8× PARTIAL + 1× BLOCKER); the distill contract has reached terminal-binding state — no further drift gaps remain. **MODEL-1 ship % nudges 88% → 89%** (wgpu silent-gibberish loophole now closed at the init boundary, symmetric to the CUDA `parity_gate` from §41); **MODEL-2 ship % nudges 56% → 57%** (last falsifier-binding gap closed for the distill contract; only remaining lever is §35 real-training implementation, multi-PR scope). Spec v2.88.0 → **v2.89.0**. Coverage tally 15+35 → **15+37** (+2 PARTIAL_ALGORITHM_LEVEL closed; TRAIN-009 explicitly blocked, not counted).
**Atomic next action (v2.88.0):** **§43 — distill-train algorithm-binding + wgpu cosine helper for FALSIFY-CPU-GPU-005 part b (PRs #1438-#1440)** (see new §43 below). Today's session shipped 3 additional PRs across both ship tracks: (i) PR #1438 — FALSIFY-APR-DISTILL-TRAIN-005 PARTIAL_ALGORITHM_LEVEL via 2 unit tests (precompute byte-determinism, local + remote-stub branches); (ii) PR #1439 — FALSIFY-APR-DISTILL-TRAIN-006 PARTIAL_ALGORITHM_LEVEL via 2 unit tests (cache-resume idempotency, negative + positive halves); (iii) PR #1440 — `cpu_vs_gpu_cosine_similarity` helper at `infer/gguf_gpu_generate.rs` module scope + 3 fail-closed unit tests, lifting the cosine math out of `cuda::mod_parity_gate` so the future wgpu cosine gate can call it without a `--features cuda` build dependency. Two contract drifts closed (TRAIN-005 + TRAIN-006: tasks #195/#196 claimed PARTIAL_ALGORITHM_LEVEL on 2026-04-30 but YAML had no `algorithm_evidence` until today). **MODEL-2 ship % nudges 54% → 56%** (TRAIN-005/006 algorithm-bindings lock in the math invariants of the precompute/train idempotency contract); **MODEL-1 ship % nudges 87% → 88%** (cosine helper + 3 tests is infrastructure-ready for the part b wgpu single-step decode in a future PR). Spec v2.87.0 → **v2.88.0**. Coverage tally 15+33 → **15+35** (+2 PARTIAL_ALGORITHM_LEVEL closed). The underlying SHIP-007 GPU kernel fix and `apr distill --stage train` real-training implementation remain open per §40 + §35.
**Atomic next action (v2.87.0):** **§42 — hub-feature build chain repair + hf_pipeline distill-train falsifier-parity (PRs #1432-#1436)** (see new §42 below). Today's session shipped 5 additional PRs that close two pre-existing defect classes: (i) the `--features hub` build was unbuildable on main due to a syntactic bug in `quantize_to_gguf_bytes` that masked 11 pre-existing test failures (PR #1432 fixed the build → 2 surfaced empty-data contract drifts closed by #1433 → 9 GGUF roundtrip alignment-padding test-helper bugs closed by #1434); (ii) the hf_pipeline distill path lacked falsifier-parity coverage with the canonical `distill::loss::DistillationLoss` (PRs #1435 wgpu drift-prevention + #1436 hf_pipeline FALSIFY-APR-DISTILL-TRAIN-003/004 parity tests). Net `--features hub` health: build-error → 7986/7986 pass / 16 ignored. **MODEL-2 ship % nudges 50% → 54%** because the falsifier-coverage parity between the canonical and parallel distillation impls is the prerequisite for any future MODEL-2 distill-train PRs not regressing the math silently. Spec v2.86.0 → **v2.87.0**. Coverage tally unchanged (the underlying SHIP-007 GPU kernel fix and `apr distill --stage train` real-training implementation remain open per §40 + §35).
**Atomic next action (v2.86.0):** **§41 — `apr-cpu-vs-gpu-output-parity-v1` chain landed (PRs #1427-#1430): three-layer jidoka armor at the GPU-CPU dispatch boundary** (see new §41 below). Today's session shipped 4 PRs that close §40's silent-gibberish loophole *as a regression class*, without (yet) fixing the underlying GPU kernel bug. (i) PR #1427 — contract `apr-cpu-vs-gpu-output-parity-v1` v1.0.0 PROPOSED authoring (4 falsifiers, 3 equations); (ii) PR #1428 — converts CUDA fallback log from verbose-only to unconditional, contract v1.0.0 PROPOSED → v1.1.0 ACTIVE with corrected algorithm_evidence (the parity_gate IS already wired on the .apr → OwnedQuantizedModelCuda path via with_max_seq_len:268-279, contradicting v1.0.0's claim — empirically verified via `apr -v run`); (iii) PR #1429 — drift-prevention: promotes the eprintln tag to `pub(crate) const CUDA_FALLBACK_LOG_PREFIX` + unit test that asserts the contract-tagged prefix shape (locks against rename without contract bump and re-wrapping in `if verbose`); (iv) PR #1430 — adds FALSIFY-CPU-GPU-005 wgpu visibility + parity-gate at PARTIAL_ALGORITHM_LEVEL, lands the wgpu visibility fix immediately (symmetric to #1428's CUDA fix), bumps contract v1.1.0 → v1.2.0 ACTIVE. Net behavioural change for `apr run` on a SHIP-007-broken GPU build: stderr now emits `[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected, attempting fallback: ... | Backend: wgpu (Vulkan) | ...` so users always know which backend is actually serving their tokens — the `--no-gpu` workaround is now self-evidently the correct path. **MODEL-1 ship % nudges 80% → 87%** because shipping `apr run` users with the `--no-gpu` documented workaround is now jidoka-safe (no silent garbage). Spec v2.85.0 → **v2.86.0**. Coverage tally unchanged (the underlying GPU-kernel SHIP-007 root-cause fix remains an open track per §40).
**Atomic next action (v2.85.0):** **§40 — SHIP-007 root cause LOCALIZED to FP8/cuBLASLt GPU path; CPU path is CORRECT** (see new §40 below). Live evidence on canonical 7B teacher (RTX 4090): `apr run --no-gpu` (CPU path via `OwnedQuantizedModel` + Q4K-fused SIMD kernels) produces "**2 + 2 equals**" (correct) at temp=0; `apr run` (default, GPU path via cuBLASLt FP8 + JIT-warmed kernels) produces "**ampiezza = 1**" (gibberish). Same model, same prompt, same greedy sampling. The bug is in the GPU dispatch chain — specifically in the `cuBLASLt FP8 JIT warmed` kernels (per `[PMAT-082]` log) and/or `FP8 weight cache` (per `[PMAT-053]` log). Notably, task #147 already established `APR_SKIP_FP8_WARMUP` env var as a "reproducer stabilization" — confirming FP8 has been a known issue and a workaround exists. **MODEL-1 is shippable today via CPU path**; the GPU FP8 path needs a fix or a fallback gate. This narrows SHIP-007 from an unbounded layer-by-layer hunt to a SPECIFIC dispatch-chain defect. SHIP-002/005/006/007/008 may all auto-discharge if "MODEL-1 ships via CPU path" is acceptable scope. Spec v2.81.0 → **v2.85.0**. Coverage scoreboard 15+33 (pending CPU-path-shippable verdict).

**Atomic next action (v2.81.0):** **§36 — plain-language status of the two-model goal** (see new §36 below). Each of the two models is blocked by a single concrete problem. **MODEL-1**: numerical bug at layer 3 of FFN (18× std anomaly; three theories refuted; sub-FFN telemetry just landed via PR #1082 + #1083 in flight). **MODEL-2**: converged at val_loss=9.38 (capacity-limited; spec target 3.0 unreachable from-scratch; needs distillation, but `apr distill` is a stub — contract authored as #1097 awaiting impl). Spec v2.80.0 → **v2.81.0**. No coverage flip; this is a landmark for plain-language readers.

**Atomic next action (v2.80.0):** **§35 — `apr distill` Standard strategy is a STUB; §34.5 distillation track requires authoring contract + extending apr** (see new §35 below). Live execution of `apr distill` on the canonical 7B teacher + §33 student finished in 45s (suspicious for a real epoch over 565.6M tokens). Source at `crates/apr-cli/src/commands/distill.rs:1464` reveals the Standard strategy is "Copy all tensors (student is same architecture, will be trained)" — no gradient training implemented. Per §26.8 stack-tool-extension methodology, the missing feature requires a `contracts/apr-cli-distill-train-v1.yaml` contract + implementation. The §34.5 distillation recommendation is correct in DIRECTION but blocked on implementation. Spec v2.79.0 → **v2.80.0**. Coverage scoreboard unchanged (15+33).

**Atomic next action (v2.79.0):** **§34 — MODEL-2 200K-step retrain confirms convergence ceiling at val_loss=9.38 — capacity-limited, not corpus-limited** (see new §34 below). 200K-step run terminated EARLY_STOP at the SAME 51 epoch / 5100 step / val_loss=9.3831 outcome as the §33 50K-step run (delta=0.0006 = numeric noise). Identical seed=42 + identical data + same LR/batch/seq → deterministic convergence. **The 370M-from-scratch capacity is the binding constraint at this configuration**, NOT corpus diversity or step budget. To reach the spec target val_loss=3.0, scaling either model size (>1B params), training methodology (distillation from teacher), or both is required. Spec v2.78.0 → **v2.79.0**. Coverage scoreboard unchanged (15+33).

**Atomic next action (v2.78.0):** **§33 — MODEL-2 retrain on 565.6M-token codeparrot corpus pushes val_loss to 9.3837 (4.7% below the 9.75 plateau)** (see new §33 below). P1 corpus pipeline (P1.0–P1.5) completed end-to-end through the spec-canonical `apr pull dataset` extension. P2 training EARLY_STOP at 51 epochs / 5100 steps / 83.5M tokens seen / 47 min wall on RTX 4090. Best val_loss=9.3837 at epoch 44 (vs 4× CSN-Python's 9.7507 plateau established by §24/§25). Confirms §25's hypothesis that **corpus diversity is the binding constraint for MODEL-2** — a 7.6× corpus expansion (74.3M → 565.6M tokens) yielded 0.367-nat improvement. Spec v2.77.0 → **v2.78.0**. Coverage scoreboard +1 MODEL-2 PARTIAL→DISCHARGED expected (SHIP-021 corpus-diversity gate).

**Atomic next action (v2.77.0):** **§32 — §31's "qkv_bias is the bug" hypothesis REFUTED by byte-compare (APR ≡ GGUF)** (see new §32 below). Live `diag_compare_qkv_bias.rs` shows APR layer-0 q/k/v_bias values are byte-for-byte identical to GGUF. The 9× std gap was a TRACE-CAPTURE-POINT MISMATCH (GGUF traces pre-bias matmul output, APR traces post-bias). Both forward passes are correct. The actual SHIP-007 bug surface is narrowed to LAYER-3-specific FFN divergence (ffn_gate first diverges 1.36× at layer 3 per existing trace). Next-step diagnostic: layer-3 sub-FFN bisection, NOT layer-0 QKV. Spec v2.76.0 → **v2.77.0**. Coverage scoreboard unchanged (15+33).

**Atomic next action (v2.76.0):** ~~§31 — SHIP-007 root cause PINNED to APR `qkv_bias` (std=10.24, ~10× too large)~~ — **REFUTED by §32**. The bias values themselves are correct (byte-for-byte equal to GGUF). The std=10.24 was a property of the trained Qwen2.5-7B biases, NOT an APR defect. Live three-stage bisection on canonical 7B teacher proves: post-matmul pre-bias APR std=0.92 matches GGUF std=1.14 (Q4K tolerance OK); but APR's `qkv_bias` ITSELF has mean=0.272, std=10.243 — adding it produces the post-bias std=10.33 that matches the existing trace and generates the 9× layer-0 gap. K-part bias is most extreme (post-bias std=29.49). The bug is either in `load_qkv_bias` byte interpretation OR in the GGUF→APR converter's bias-handling. PR E v2 is scoped to one specific dump-and-compare investigation per §31.4. Spec v2.75.0 → **v2.76.0**. Coverage scoreboard unchanged (15+33) — still pre-DISCHARGE.

**Atomic next action (v2.75.0):** **§30 — PR E investigation refutes §28 narrow hypothesis; PR E paused, qkv-bias / RoPE / per-head-norm bisection load-bearing** (see new §30 below). Live diagnostics on canonical 7B teacher: `q4k_layers` IS fully populated for all 28 layers; APR's F32-fused-qkv weight is numerically equivalent to per-Q/K/V Q4K dispatch (max |diff|=0.005, RMS=0.0007). The §28 mechanical "switch matmul kernel" fix would change <0.5% of std — the 9× layer-0 qkv std gap (APR=10.33 vs GGUF=1.14) lives elsewhere. PR E is paused; next session must bisect post-matmul/post-bias/post-RoPE to localize the actual divergence point. Spec v2.74.0 → **v2.75.0**. Coverage scoreboard unchanged (15+33).

**Atomic next action (v2.72.0):** **§26.4 P3 binding criterion DECIDED — APR vs GGUF layer-3 ffn_swigl ratio = 18.23×, SHIP-007 bug confirmed APR-side at `apr_transformer/inference.rs:160-164`** (see new §27 below). Live evidence on noah-Lambda-Vector RTX 4090 2026-04-27: built `apr` from PR #1083 branch (commits 77c016bc2 + c6579685b + f24946412 from PR A+B+C cascade), ran `apr trace --payload` on canonical 7B teacher in BOTH APR and GGUF formats with identical prompt + tokenizer. APR layer-3 ffn_swigl std = **1.2216** (matches §23 reading); GGUF layer-3 ffn_swigl std = **0.0670** (1.0× layer 1-2 baseline = no anomaly on GGUF side). Ratio 1.2216/0.0670 = **18.23×** — far exceeds the §26.4 ≥10× threshold by 8× absolute. Layers 0-2 agree (~1.1× ratio); layer 3 anomaly is APR-only; layers 6+ recover to ~1× ratio. The bug is localized to APR's SwiGLU element-wise multiply at `silu_g * u`; GGUF's path produces normal output. **Discharges 5 MODEL-1 PARTIALs once fix lands per §17.5** (SHIP-002/005/006/007/008). Fix scope: investigate `inference.rs:160-164` for off-by-one slice indexing, buffer corruption, or F32-vs-Q4K dequant anomaly at layer-3 specifically. Spec v2.71.0 → **v2.72.0**. Coverage flip pending fix (§26.5 expected: 33+12 → 28+17).

**Atomic next action (v2.71.0):** **Stack-tool extension rule codified + `apr` is the canonical stack CLI post-monorepo — when `apr` lacks a feature, we extend `apr` via contract→code, NEVER route around to non-stack shims like `huggingface-cli` or to deprecated namespaces like `batuta hf pull`** (see new §26.8 + revised §26.2). Triggering incident 2026-04-27: P1 sub-agent recommended `huggingface-cli download --include 'data/train-000[0-7][0-9]-of-00880.parquet'` because `apr pull` is model-only today (no dataset asset-type, no `--include` for shard-pattern selection, no `--license-allowlist`); this is muda per `feedback_fix_root_cause_never_route_around.md` + `feedback_pv_not_bash_for_contracts.md` + `feedback_monorepo_single_source_of_truth.md` (post-APR-MONO consolidation, `apr` subsumes batuta's HF-pull surface — batuta namespace is no longer relevant for dataset/model pulls). Correct path: author `apr-cli-pull-dataset-v1.yaml` provable contract → extend `apr pull` with dataset asset-type + `--include <glob>` + `--license-allowlist` → use the extended stack tool for P1. P1 is now gated on the `apr pull` extension landing. Spec v2.70.0 → **v2.71.0**. Coverage tally unchanged.

**Atomic next action (v2.70.0):** **Three-priority execution plan adopted — operator authorization issued for Stack v2 download (P1) + GGUF forward_traced (P3) parallel start; convergence run (P2) gated on P1 completion** (see new §26 below). Per user directive "proceed with these priorities" 2026-04-27, the spec records the concrete execution path that takes the chains §24+§25 (corpus diversity is binding for MODEL-2) and §15-§17+§23 (layer-3 ffn_swigl is the SHIP-007 bug surface) to their respective discharges. P1 and P3 are independent and run in parallel; each accomplishment is binding-criterion measurable. Spec v2.69.0 → **v2.70.0**. Coverage tally unchanged at amendment time; expected to flip 9 MODEL-2 PARTIALs (P2 outcome) + 5 MODEL-1 PARTIALs (P3 outcome) = up to **14 PARTIAL→DISCHARGED** within next session.

**Atomic next action (v2.69.0):** **§24.8 LR-budget hypothesis FALSIFIED — 80K-step run on 4× corpus early-stopped at val_loss=9.7507 epoch 4, identical to 20K best** (see new §25 below). Per spec §24.8 prediction "If val_loss plateaus near 9.5–9.7 with no breakthrough → only Stack v2 will move the needle" — exactly what happened. 4× cosine-decay LR budget allocated 80K steps total; early-stop fired at epoch 10 (22K steps actual) because val_loss never improved past epoch-4 best of 9.7507 (within 6e-4 of the 20K run's 9.7513). The 370M model on 74.3M-token CSN-Python corpus has a hard val_loss floor of ~9.75 driven by **corpus diversity exhaustion**, not LR scheduling. Falsification is clean: 1h32min wall, 11 ckpts produced, lambda-labs lane pre-authorized. Conclusion: contract `target_val_loss=3.0` is unreachable on CSN-Python at any LR/step budget; The Stack v2 Python (multi-billion tokens) is the only on-spec corpus path. Spec v2.68.0 → **v2.69.0**. Coverage tally unchanged.

**Atomic next action (v2.68.0):** **MODEL-2 4×-corpus experiment — 74.3M-token CSN-Python re-training quantifies the memorization signature in the prior 18M-token "best" run** (see new §24 below). User mandate "train this model: now!" delivered second from-scratch run on a corpus 4.10× the original (74.3M vs 18.1M tokens). Same hyperparameters as the v2.65.0 best run (20K steps × 264ms = 88 min wall, 10 epochs). Result: final val_loss=9.806, best val_loss=9.751 at epoch 4. **Critical comparison**: 1× run's epoch-9 "best" of val_loss=8.911 had train_loss=9.467 (val < train by 0.556 — *memorization* signature from 9.1× corpus wraps); 4× run's epoch-9 has val_loss=9.806 / train_loss=9.816 (val ≈ train, healthy generalization). The 4× model is materially **healthier** per the train-val gap; the 1× run's lower absolute val_loss was driven by memorizing the small wrapped corpus, not by better learning. Best 4× checkpoint validates as APR v2 / 219 tensors / 1.39 GiB / checksum VALID. Empirical proof that the SHIP-TWO-001 corpus path requires Stack v2 (multi-billion tokens) to push val_loss below 8.91 via real generalization rather than wrap-induced memorization. Spec v2.67.0 → **v2.68.0**. Coverage tally unchanged.

**Atomic next action (v2.67.0):** **SHIP-007 sub-FFN bisection executed on canonical 7B teacher — layer-3 ffn_swigl is the first 17×-anomaly site** (see new §23 below). PR #1066's sub-FFN telemetry impl + #1064's §17 layer-3 finding combined: live `apr trace --payload` shows ffn_silu at layer 3 = 0.168 (3.2× layers 1-2 baseline = 0.04-0.05; precursor) → ffn_swigl at layer 3 = 1.222 (17.2× layer 2's 0.071 — first anomaly) → ffn_out at layer 3 = 11.459 (53× — cascaded post-down-proj). Gate/up individually normal at layer 3. Fix surface refined to `inference.rs:160-164` `ffn_hidden.push(silu_g * u)` element-wise multiply (possibly off-by-one slice indexing). Pin requires GGUF-side `forward_traced` extension (next session) per `project_ship_007_gguf_forward_traced_plan.md`. Spec v2.66.0 → **v2.67.0**. Coverage tally unchanged.

**Atomic next action (v2.66.0):** **First real MODEL-2 training run on RTX 4090 — three stack bugs found + fixed at root + first format-validated checkpoint produced** (see new §22 below). User mandate "train a model unless the path is broken, then fix" delivered: (1) `ShardBatchIter` corpus exhaustion silently emitted `(1.0, 1.0)` placeholders for 1000s of steps — fixed via `with_wrap_around(true)` opt-in (PR #1073 first commit); (2) `HELD_OUT_BATCHES=2` + `patience_epochs=2` triggered spurious early-stop on val-noise — fixed by widening to 16 batches + 5 patience (PR #1073 second commit); (3) corpus is 18M tokens vs Chinchilla-optimal 7.4B for 370M params — overfit at epoch 3+ (data engineering deferred to next session). **Best MODEL-2 checkpoint** at `/mnt/nvme-raid0/runs/model-2-from-scratch-006-50k-tuned/ckpt/epoch-002.apr`: val_loss=9.78, 49.2M tokens seen, APR v2 / 219 tensors / 1.39 GiB / checksum VALID / arch=LlamaForCausalLM. AC-SHIP2-005 structurally discharged at format level; awaits contract-level promotion. Spec v2.65.0 → **v2.66.0**. No coverage tally change.

**Atomic next action (v2.65.0):** **Live CUDA training dispatch on RTX 4090 — GATE-GPUTRAIN-004 dischargeable** (see new §20 below). Rebuilt the canonical apr binary at `/mnt/nvme-raid0/targets/aprender/release/apr` with `--features cuda` (40s incremental build). Dispatched `apr pretrain --device cuda --num-steps 50 --seq-length 512` against `/mnt/nvme-raid0/data/csn-python-shards` + `/mnt/nvme-raid0/models/model-2-tokenizer-v1` (vocab=50,257). 100 per-step JSONL records emitted with the new `wall_ms` field (from PR #1069). **Median wall_ms = 264.74 ms** (well under GATE-GPUTRAIN-004's 500ms budget — 47% headroom). PID 1658504 / 6636 MiB GPU memory captured mid-run via `nvidia-smi --query-compute-apps`, confirming GPU-residency (no silent CPU fallback). train_loss step 0→99: 11.02 → 10.50 (Δ=−0.52, real learning). Run aborted at epoch boundary via GATE-TRAIN-005 (val_loss=10.31 > 10.0 ship-blocker — correct behavior for fresh-init 370M). Evidence persisted to `evidence/task-132-residual-b/`. Step (a) of §19.5's long path — "rebuild canonical apr binary with `--features cuda`" — is **DONE**. Spec v2.64.0 → **v2.65.0**. Contract `gpu-training-backend-v1.yaml` GATE-GPUTRAIN-004 promotion is a follow-up PR (the live data is captured; the durable verdict is pending the contract bump).

**Atomic next action (v2.64.0):** **§18.5 corrected — Task #132 (`apr pretrain --device cuda` wiring) has substantially shipped** (see new §19 below). Sub-agent investigation on 2026-04-26 confirmed §18.5's premise was outdated by ~5 days: task #132 closed at commit f7ad11408 (2026-04-21). The CLI dispatch path `apr pretrain --device {cpu|cuda|auto}` → `resolve_device()` → `drive_real_cuda(...)` → `CudaTransformerTrainer::new(cfg)` is wired and live, with all GPU kernels (forward/backward/optimizer/loss/AdamW state) invoked from `crates/aprender-train/src/autograd/`. Live smoke test confirmed: a non-CUDA-built apr binary produces a graceful contract-cited error citing GATE-GPUTRAIN-002, proving the wiring exists. Three real residuals remain: (A) INV-TRAIN-003 GPU AdamW-state sha256 [small PR]; (B) GATE-GPUTRAIN-004/005 → ACTIVE via 50-step cuda:0 dispatch + JSONL `wall_ms` emission [small PR + operator dispatch]; (C) operator authorization for the 10K-step convergence run [decision, not engineering]. Spec v2.63.0 → **v2.64.0**. No coverage tally change. The corrected long path to MODEL-2 publish is much shorter than §18.8 stated. Methodological lesson: status sections that cite a stale memory entry as evidence MUST re-verify against current code (binding rule per `feedback_no_guessing.md`).

**Atomic next action (v2.63.0):** **Training status snapshot recorded as chain-of-thought (§18 below).** This section walks the deduction chain that connects the spec's two-model goal to the current state, so future sessions can re-enter the work without re-reading every prior section. Coverage tally unchanged: **33 PARTIAL + 12 DISCHARGED** across 45 contract-bound levers. **MODEL-1 status:** 5/10 ACs DISCHARGED via live RTX 4090 evidence (SHIP-001/003/004/009/010); 5/10 PARTIAL (SHIP-002/005/006/007/008) all transitively gated on the SHIP-007 root-cause fix tracked in §15–§17 + PRs #1063/#1064/#1065/#1066. **MODEL-2 status:** 3/12 ACs DISCHARGED (SHIP-011/021/022); 9/12 PARTIAL gated on a converged 370M run, blocked at task #132 (`apr pretrain --device cuda` not yet wired through `TransformerTrainer::new` — kernels exist, just not entry-pointed). **GPUTRAIN suite:** 7/7 DISCHARGED (full closure, including bit-exact FP32 reproducibility via cuBLAS PEDANTIC_MATH + atom-free PTX reduction). Two parallel paths to next observable state-change: (a) short — sub-FFN bisection on the canonical 7B teacher names the SHIP-007 bug site → 5 MODEL-1 PARTIALs auto-discharge; (b) long — task #132 + The Stack v2 tokenization + convergence → 9 MODEL-2 PARTIALs auto-discharge. Spec v2.62.0 → **v2.63.0**. No coverage tally change.

**Atomic next action (v2.62.0):** **SHIP-007 layer-3 ffn_out anomaly identified — first-divergent layer named** (see new §17 below). The §16.4 falsifier was executed against the APR teacher's `apr trace --payload` output, which already emits per-layer mean/std for all 28 transformer blocks. The full 28-layer `ffn_out` std progression shows a **31× discontinuity at layer 3** (std=11.46) vs layer 2 (std=0.22) and the layer-4-26 median of 0.5-2.0. The residual stream's `output` std jumps from 0.72 (layer 2) to 11.78 (layer 3), then stays elevated. Three signals point at layer 3 ffn_out specifically: (a) magnitude discontinuity 31× isn't architecture-driven (SHIP-003 PR #1059's 339-tensor cosine sweep proved the underlying weights are byte-equivalent to SafeTensors); (b) damps in one layer (layer 4 ffn_out std=3.84), which is a one-off perturbation pattern, not a stable feature; (c) mean shift -0.082 is 100× the median magnitude, suggesting a sign-bias defect. Surviving suspect surface narrowed from §16.3's four candidates to **layer-composition glue in `forward_single_with_scratch` at layer 3 in the FFN sub-block** plus three new §17.3 candidates (Q4K dequant under load on 18944-dim FFN; SiLU numerical stability; fused gate+up dispatch defect). Sub-layer bisection (`gate_proj_out` / `silu(up_proj_out)` / `down_proj_out`) is now the load-bearing follow-up — requires the §15.5 TraceStep enum extension. Spec v2.61.0 → **v2.62.0**. No coverage tally change.

**Atomic next action (v2.61.0):** **SHIP-007 root cause materially isolated to the CPU APR forward path on 7B Qwen2.5-Coder** (see new §16 below). Live evidence on noah-Lambda-Vector RTX 4090 (2026-04-26): `apr trace --payload` on the canonical paiml/qwen2.5-coder-7b-apache-q4k-v1 teacher in BOTH formats, same prompt "What is 2+2?", same encoded tokens `[3838, 374, 220, 17, 10, 17, 30]`, same embedded BPE tokenizer:
- **GGUF teacher** → Top-1 token=17 ("2"), full output `" 2+2 is 4."` ← **CORRECT** language model output.
- **APR teacher** → Top-1 token=220 (" "), logit=16.7368 ← **WRONG** (whitespace prediction).

Both ran on CPU. Same model, same weights (verified by SHIP-003 PR #1059's sweep: SafeTensors↔APR cos≥0.9999999 across all 339 tensors). The bug is **inside the APR-side `forward_single_with_scratch` codepath**, not the GPU stack and not the loader-side data layout. Combined with §15.4 (PR #1061 — GPU GQA-7:1 attention kernel ruled out via 3 passing CPU/GPU parity tests), the surviving suspect surface is now: **APR-format inference codepath**, exclusive of the kernel arithmetic that's twice-ruled-out. Spec v2.60.0 → **v2.61.0**. No coverage tally change. The remaining 5 MODEL-1 PARTIALs (SHIP-002/005/006/007/008) all transitively block on this fix; the new isolation makes the next root-cause-fix PR much more focused — start with `crates/aprender-serve/src/gguf/inference/forward/single_cache.rs` and the APR-specific `forward_single_with_scratch` path.

**Atomic next action (v2.59.0):** **SHIP-007 GQA-7:1 root-cause analysis recorded** (Five Whys + tensor-shape evidence, see §15 below). The 7B Qwen2.5-Coder teacher's GPU forward path produces logits whose argmax differs from CPU forward (CPU=334, GPU=8127, cosine=−0.005, max abs logit Δ=19.5). Cross-format `apr qa --json` on the GGUF teacher reveals the same divergence class on a different surface: `format_parity` reports `GGUF argmax=17 != SafeTensors argmax=59260`. Five Whys traces the surface symptom to a **transpose-handling defect on GQA-7:1 K/V projections** that exposes only when `num_heads ≠ num_kv_heads` (28 Q heads / 4 KV heads / head_dim 128 / hidden 3584). Evidence: GGUF stores 2D weights as `[in, out]` (col-major-flavor) while APR + SafeTensors store as `[out, in]` (row-major) — `apr diff --values --transpose-aware --json` confirms `down_proj` GGUF=[18944, 3584] vs APR=[3584, 18944], cos=0.00006. LAYOUT-001/002 transpose-at-import IS supposed to apply at GGUF→APR boundary; APR shapes confirm the data-side transpose worked. The bug is therefore in the inference *consumer* path (CPU and/or GPU forward), not the loader's storage layout. SHIP-007 stays at PARTIAL_ALGORITHM_LEVEL on `verdict_from_decode_tps`; full discharge blocks on a single-tensor reproducer (Q × K^T element-by-element CPU vs GPU on `model.layers.0.self_attn.k_proj.weight` from APR row-major) plus the kernel fix once the divergent stage is localized. Spec v2.58.0 → **v2.59.0**. No coverage tally change (no new discharge); this amendment is investigation-recording, not rule promotion. The remaining 5 MODEL-1 PARTIALs (SHIP-002/005/006/007/008) all transitively block on this fix. See §15 for the full Five Whys + the targeted next investigation step.

**Atomic next action (v2.58.0):** pretokenize The Stack v2 filtered-Python for MODEL-2 convergence (unchanged from v2.51.0). The GPUTRAIN suite is now **7/7 DISCHARGED**. FALSIFY-GPUTRAIN-006 (same-device seed reproducibility) flipped PARTIAL_ALGORITHM_LEVEL → **DISCHARGED** on 2026-04-25 (task #144) via four-root-cause world-class fix bundle + empirical reproducibility study. Root causes: (1) `atom.global.add.f32` on `grad_gamma[i]` → per-row partial buffer + new deterministic-iteration-order `RmsNormGammaReduceKernel`; (2) cuBLAS `DEFAULT_MATH` → `CUBLAS_PEDANTIC_MATH` (full FP32, no Tensor Cores); (3) APR-MONO single-source-of-truth dep migration — `aprender-train` and `aprender-serve` switched from crates.io `trueno-gpu = "0.4"` to in-tree `aprender-gpu`; (4) confirmed via `/usr/include/cublasLt.h` 12.6 inspection that no `DETERMINISTIC` API flag exists in cuBLAS-LT — bit-exact FP32 GEMM is physically unachievable through configuration alone. The 10-run × 100-step empirical study on noah-Lambda-Vector RTX 4090 produced max per-step `|Δ_train_loss| = 9.2e-4` (~772× ULP at loss~10), random-walk ε=2.74e-4, worst pair-wise cos-sim=0.999_999_999_7, final_val_loss range=1.34e-3. Contract `gpu-training-backend-v1.yaml` bumped **v1.3.0 → v1.4.0** with 4 new `AC_GPUTRAIN_006_*` constants + `verdict_from_reproducibility_study(study: &ReproducibilityStudyResult) -> Gputrain006Verdict` 4-bound aggregate verdict + 8-section mutation survey. Evidence: `evidence/task-132/gputrain-006-empirical-v1.json`. Spec v2.57.0 → **v2.58.0**. Coverage tally is now **33 PARTIAL + 12 DISCHARGED** (was 34 + 11; GPUTRAIN-006 promoted; final GPUTRAIN closure).

**Atomic next action (v2.51.0):** pretokenize The Stack v2 filtered-Python (≥~15B tokens) into MODEL-2-tokenizer (vocab=50257) `.bin` shards and dispatch the multi-hour MODEL-2 convergence run. The infrastructure corridor is now fully evidence-discharged: FALSIFY-GPUTRAIN-004 flipped PARTIAL_ALGORITHM_LEVEL → **DISCHARGED** on 2026-04-24 (task #140) via three seed=0 `apr pretrain --device cpu --synthetic --num-steps 4` dispatches on `noah-Lambda-Vector` (binary apr 0.31.2 built `--features cuda`). All three runs produced **byte-identical scripted-loss traces** (sha256 `aeea198418ec50ba9897f95fa641bcdf43f4e05f99f4e5f7b9e308af29e7ff78` after stripping `tokens_per_sec` and `gpu_util_pct`) AND `nvidia-smi --query-compute-apps` returned empty during the CPU dispatch (training pid 1029007 absent from the CUDA-compute-apps list). This proves the CPU path remained functional after the Task #132 device-dispatch refactor — no silent GPU promotion, no silent CPU fallback, peer-contract gates GATE-TRAIN-005 (decreasing scripted loss) / GATE-TRAIN-007 (no NaN) / GATE-TRAIN-008 (grad_norm finite) all PASS under aggregate `status: "OK"`. Evidence: `evidence/task-132/cpu-fallback-peer-gates.json`. Contract `gpu-training-backend-v1.yaml` bumped v1.2.0 → **v1.3.0** (same ACTIVE status) to record the second Phase 3 evidence cycle. Coverage tally is now **39 PARTIAL + 6 DISCHARGED** (was 40 + 5; GPUTRAIN-004 promoted). The only remaining GPUTRAIN PARTIAL is **FALSIFY-GPUTRAIN-006** (same-device seed reproducibility, runtime_falsification_observed at |Δ|=1.19e-3 vs 1e-5 tolerance — holds at PARTIAL pending a `--deterministic` Rust code path that pins cuBLAS math mode + workspace + disables atomic reductions). Per Toyota Way: not widening the 1e-5 tolerance; authoring the deterministic code path is tracked separately from MODEL-2 convergence.

**Atomic next action (v2.46.0):** run `apr pretrain --num-steps 10000+ --device cuda:0` on `noah-Lambda-Vector` (RTX 4090, 24 GB VRAM, local) for a real MODEL-2 convergence pass. Phase 3 dispatch is PROVEN — FALSIFY-GPUTRAIN-002 and FALSIFY-GPUTRAIN-003 just flipped PARTIAL_ALGORITHM_LEVEL → **DISCHARGED** on 2026-04-24 via two `apr pretrain --mode from-scratch --device cuda:0` runs (5-step smoke + 50-step full) with nvidia-smi residency proof (pids 3960483 @ 404 MiB / 3964206 @ 10,300 MiB, both far above the 1 MiB floor). Evidence persisted to `evidence/task-132/rtx4090-370m-residency.json`. The hostname confusion — "lambda-labs" in §14.5 naming a remote cloud host vs. noah-Lambda-Vector as a local Lambda-class RTX 4090 workstation — falsified itself the moment the code-check ran: this box IS a lambda-labs-equivalent and Phase 3 dispatch runs locally. With Phase 2 runtime wiring already shipped (verified 2026-04-24 in prior §14.5 correction) and Phase 3 residency discharged, MODEL-2 training is unblocked at the *infrastructure* layer. What's pending is a multi-hour convergence run that brings val CE down toward the SHIP-013 ≤ 2.2 floor — the 5/50-step smokes ran fast (5s / 63s wall) with val CE = 10.08 (untrained), so the machinery is ready; only GPU clock-time remains. Contract `gpu-training-backend-v1.yaml` also flipped v1.1.0 PROPOSED → **v1.2.0 ACTIVE** per §14 Phase 4. Coverage tally is now **40 PARTIAL + 5 DISCHARGED** (was 42 + 3; two GPUTRAIN-promoted).

**Atomic next action (v2.45.0):** Task #132 **Phase 3** — lambda-labs RTX 4090 live dispatch + residency-evidence collection. **Phase 2 runtime wiring is SHIPPED** (my v2.44/v2.45 narrative above initially said "pending, 2 days" — that was stale, code-checked 2026-04-24 and falsified): `crates/apr-cli/src/commands/pretrain.rs::drive_real` at lines 252–301 already takes `device: Device`, dispatches `if device.is_cuda() { drive_real_cuda(...) } else { drive_real_cpu(...) }`, and `drive_real_cuda` (line 336, `#[cfg(feature = "cuda")]`) builds a real `CudaTransformerTrainer` via `entrenar::train::pretrain_real_cuda::build_shared_cuda_trainer` + wires `CudaRealStepFn`/`CudaRealValFn`/`CudaAprCheckpointFn`. The `#[cfg(not(feature = "cuda"))]` companion at line 373 returns a clear GATE-GPUTRAIN-002 error instead of silent CPU fallback. nvidia-smi querying lives in `crates/aprender-train/src/config/train/loader/mod.rs:445` + `crates/aprender-train/src/gpu/ledger.rs:404`. What's actually pending is pure hardware work: one `cargo build --release --features cuda` on lambda-labs, then `apr pretrain --mode from-scratch --device cuda:0 --num-steps 50 --json` emitting `evidence/task-132/rtx4090-370m-step-budget.json` with median step-wall < 500 ms + nvidia-smi residency proof — no Rust left to write. After Phase 3 evidence lands, FALSIFY-GPUTRAIN-003..007 flip PARTIAL_ALGORITHM_LEVEL → DISCHARGED, `gpu-training-backend-v1.yaml` goes PROPOSED → ACTIVE (Phase 4), and the same lambda-labs session can run MODEL-1 PARTIAL → DISCHARGED harnesses (`apr convert --quantize q4_k_m`, `apr eval --benchmark humaneval`, `apr qa`, `apr bench`) in parallel against the published 7B teacher artifact. MODEL-2 training itself (SHIP-013 val CE ≤ 2.2 + SHIP-014 ≤ 21 days) is a separate multi-day compute run that starts immediately after Phase 3 residency proof — the dispatch glue is ready, only the GPU clock hasn't started.

**Spec-drift five-whys (recorded 2026-04-24 during v2.45.0 authoring):**
 1 Why did v2.45.0 initially name "Phase 2 runtime wiring" as the atomic next action? I trusted §14.5's "Phase 2 (live-wire, pending) — 2 days" row without code-checking.
 2 Why was §14.5 stale? It was authored 2026-04-21 when the bug was fresh; the wiring landed in a later PR that didn't amend the spec.
 3 Why didn't the wiring PR amend the spec? The spec is narrative; the code passes tests; nothing enforces that narrative-claims and code-reality agree. Same class as README drift and const drift pre-v2.44.
 4 Why didn't v2.44 drift-prevention catch this? v2.44 guards (a) README numbers (`readme-claims-v1.yaml`) and (b) `AC_*` threshold consts (`ship_two_001_const_pinning.rs`). It does NOT guard *prose claims about which Rust functions exist*. Runtime-wiring claims were unguarded.
 5 Why is there no mechanism? Same five-whys that produced SHIP-TWO-001 in the first place: spec narrative outruns code verification unless someone bolts on a falsification test. This is the **next** drift-prevention class to add — a test that greps for `fn drive_real_cuda` existence under `#[cfg(feature = "cuda")]` and fails if the claimed runtime path is absent. Filed as follow-up in the same PR as this correction.

**v2.46.0 amendment (2026-04-24, first live Phase 3 discharge):** My v2.45.0 narrative above claimed Phase 3 dispatch required a remote lambda-labs cloud box. Noah falsified that within minutes: `hostname` on this workstation returns `noah-Lambda-Vector` (a Lambda Labs Vector workstation class), and `nvidia-smi` reports an **NVIDIA GeForce RTX 4090 with 24 GB VRAM**, driver 570.207, CUDA 12.8. The local workstation IS the lambda-labs-class host the spec wanted. Built `apr` from HEAD with `cargo build --release --features "cuda training" -p apr-cli --bin apr`, ran two `apr pretrain --mode from-scratch --device cuda:0` dispatches (5-step smoke: pid 3960483 / 404 MiB / val CE 10.082615; 50-step full: pid 3964206 / 10,300 MiB / val CE 10.082788; both `exit 0`, both `"status": "OK"` in the JSON output). `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` confirmed GPU residency in both runs, `verdict_from_residency` returned Pass, `AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB = 1` floor cleared by 404× and 10,300× respectively. Task #132 outcome: FALSIFY-GPUTRAIN-002 (no silent CPU fallback — the exact defect Task #132 was filed against on 2026-04-21) **DISCHARGED**; FALSIFY-GPUTRAIN-003 (GPU residency proof) **DISCHARGED**. `contracts/entrenar/gpu-training-backend-v1.yaml` v1.1.0 PROPOSED → **v1.2.0 ACTIVE** per §14 Phase 4. Evidence JSON: `evidence/task-132/rtx4090-370m-residency.json` (schema-validated, captures hostname / GPU / driver / CUDA / dispatch command / both runs' nvidia-smi output / both runs' `apr` exit JSONs). Coverage shifts 42 PARTIAL + 3 DISCHARGED → **40 PARTIAL + 5 DISCHARGED** (two GPUTRAIN promotions). Remaining blockers for full 7/7 FALSIFY-GPUTRAIN discharge: 005 (step-time < 500 ms — needs per-step `--json` telemetry emission in `apr pretrain`), 006 (same-seed reproducibility — two back-to-back runs + loss-trajectory diff), 007 (apr --version --json schema — quick wire-up). Remaining blocker for MODEL-2 weights: a multi-hour `--num-steps 10000+` convergence run using the now-proven dispatch.

**v2.47.0 amendment (2026-04-24, GPUTRAIN-005 DISCHARGED + GPUTRAIN-006 runtime-FALSIFIED):** Extended `apr pretrain`'s `PretrainReport` JSON schema to emit per-step `StepMetrics` (the 6-field `GATE-TRAIN-001` tuple: step, train_loss, grad_norm, lr, tokens_per_sec, gpu_util_pct) so downstream Phase-3 harnesses don't have to parse run-dir checkpoint metadata. With per-step visibility in hand, ran two more Phase 3 checks against the live RTX 4090. **FALSIFY-GPUTRAIN-005** (step time < 500 ms at batch=1 seq=2048 per spec canonical) — PASSED cleanly: derived step_ms from tokens_per_sec (batch * seq * 1000 / tps) across 25 warmed-up steps, observed **median 101.30 ms** (min 97.58, max 110.39), well inside the 500 ms budget at 20.3% utilization. PARTIAL_ALGORITHM_LEVEL → **DISCHARGED**. **FALSIFY-GPUTRAIN-006** (same-seed reproducibility, |Δloss[k]| ≤ 1e-5) — ran two back-to-back `--device cuda:0 --seed 0 --num-steps 50` dispatches and diffed per-step train_loss trajectories. Observed max **|Δ| = 8.27e-4 at step 27**, with even step 0 (pure forward pass, no training update applied) showing **Δ = 2.3e-5**. **83× over the 1e-5 tolerance**. The algorithm-level verdict fn is provably correct; CUDA is simply not bit-deterministic without explicit mode configuration (cublasSetMathMode / CUBLAS_WORKSPACE_CONFIG / cuDNN determinism / atomics-off reductions). Per Toyota Way: **NOT widening** `AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA` to match observed drift — that would hide the defect. Instead: GPUTRAIN-006 **held at PARTIAL_ALGORITHM_LEVEL** with a `runtime_falsification_observed` block in the contract, a root-cause hypothesis, and a follow-up ticket for a `--deterministic` opt-in code path (estimate: 1-2 days Rust, acceptance: re-run the harness under `--deterministic` and assert Pass). Evidence: `evidence/task-132/rtx4090-370m-step-budget-and-repro.json`. Coverage: 40 PARTIAL + 5 DISCHARGED → **39 PARTIAL + 6 DISCHARGED** (005 promoted; 006 held with documented runtime falsification).

**v2.48.0 amendment (2026-04-24, FALSIFY-GPUTRAIN-007 DISCHARGED + MODEL-2 convergence training launched):** Added `emit_version_json()` in `crates/apr-cli/src/lib.rs` + `--version --json` intercept in `crates/apr-cli/src/main.rs` before clap's default `--version` exit. Output on a `--features cuda` build on noah-Lambda-Vector: `{ "cuda_feature": true, "cuda_runtime_available": true, "git_sha": "...", "name": "apr", "version": "0.31.2", "visible_devices": ["0"] }`. Both `verdict_from_version_json_keys` (schema completeness) and `verdict_from_version_json_fields` (len ≤ 16, no FM-GPUTRAIN-STALE-BUILD inconsistency) return Pass. FALSIFY-GPUTRAIN-007 PARTIAL_ALGORITHM_LEVEL → **DISCHARGED**. Second deliverable: **MODEL-2 convergence training launched** in background (pid 805184, 22.2 GB GPU residency on the RTX 4090, 2000-step `--batch-size 1 --seq-length 2048 --seed 0 --warmup-steps 200 --mode from-scratch` run, run-dir `/mnt/nvme-raid0/runs/model-2-convergence-20260424-181022`). At ~100 ms/step the full 2000-step sweep should complete in ~3-4 minutes of GPU time; the val-loss trajectory will be the first live descent signal toward the SHIP-013 ≤ 2.2 val CE floor (starting point 10.08 from the 50-step smoke). Evidence for 007: `evidence/task-132/version-json-schema.json`. Coverage: 39 PARTIAL + 6 DISCHARGED → **38 PARTIAL + 7 DISCHARGED** (full GPUTRAIN surface: 002/003/005/007 DISCHARGED + 001/004 ACTIVE in device.rs + 006 held with runtime-falsification note).

**v2.49.0 amendment (2026-04-24, MODEL-2 first live training — 1300-step EARLY_STOP smoke):** The 2000-step background convergence run launched by v2.48.0 completed with `status: EARLY_STOP` at step 1300 (val-loss plateau detected by the pretrain loop's guard). First real MODEL-2 training data on the stack. **Infrastructure: fully proven**; **SHIP-013 convergence: not yet achieved**. Observed val-loss descent: `[10.60, 10.50, 10.36, 10.22, 10.19, 10.21, 10.21, 10.14, 10.15, 10.10, 10.14, 10.17, 10.16]` over 1300 steps. `final_val_loss = 10.103599` vs `AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS = 2.2` target — **gap of 7.90 nats**. train_loss descended 11.04 → min 9.27 → settled ~10.47. Step-time median 100.83 ms (consistent with v2.47's 101.30 ms measurement under the same config). GPU peak residency 22.2 GB / 24 GB on the RTX 4090 (batch=1, seq=2048). Per Toyota Way: **NOT** promoting FALSIFY-SHIP-013 beyond PARTIAL_ALGORITHM_LEVEL — the decision rule is correct, the infrastructure works, but observed val CE (10.10) does not meet the 2.2 floor. The gap is a training/tuning problem, not an algorithm or infrastructure problem: longer runs, LR-schedule sweep, larger corpus, or larger effective batch are the next levers. Coverage unchanged at **38 PARTIAL + 7 DISCHARGED** — this amendment documents live training evidence without promoting the rule. Evidence file: `evidence/task-132/rtx4090-370m-convergence-smoke.json` (dispatch command + run-dir + val-loss history + train-loss trajectory + step-time median + gap analysis + next-action list). Next convergence iterations should re-run at `--num-steps 50000+` with tuned `--lr`/`--warmup-steps`/`--batch-size` to exit the plateau region around val CE ≈ 10 and push toward 2.2.

**v2.50.0 amendment (2026-04-24, second independent run confirms corpus bottleneck):** Launched a 10000-step re-run with explicit `--target-val-loss 2.2 --steps-per-epoch 500 --num-steps 10000 --seed 0` (4× longer budget than the v2.49 smoke). Result: early-stop again, this time at step **2500** with `final_val_loss = 10.103826` — essentially identical to the v2.49 run's 10.103599. Two independent runs converge to the **same val_loss ceiling (~10.10)** despite 4× step budget. Train-loss 10-bucketed means show slow descent on training data (10.53 → 10.14 over 2500 steps) but val-loss is frozen on plateau. Bucketed bucket-to-bucket deltas: `-0.17, -0.12, +0.03, +0.01, -0.10, +0.11, +0.02, -0.10, -0.09` — drift noise, not descent. Signature is textbook corpus-undersizing: a 370M model on ~10M tokens (CSN-Python-shards from task #118 pretokenization) has memorized what it can and cannot extract further generalization signal. Typical 370M-class convergence requires 10-100 **billion** tokens — this corpus is **1000-10000× undersized**. The bottleneck is NOT step count, NOT learning rate, NOT infrastructure — it is the input dataset. Per §5.1 MODEL-2 current state, the *intended* training corpus is 60 GB deduplicated Python from The Stack v2 filtered subset (≥~15B tokens); CSN-Python-shards is a task #118 pretokenize smoke, not the ship corpus. Evidence file: `evidence/task-132/rtx4090-370m-convergence-10k-plateau.json` (dispatch + plateau metrics + bucketed train-loss + corpus-bottleneck analysis + next-action list). Per Toyota Way: SHIP-013 **remains NOT DISCHARGED**; instead of widening the 2.2 floor, this amendment names the root cause (corpus size) with live two-run evidence and points at the corrective action (pretokenize The Stack v2 Python full). Coverage unchanged at **38 PARTIAL + 7 DISCHARGED** — the amendment is narrative + evidence, not rule promotion.

**Status:** SHIP-TWO-001-MODEL-1-TEACHER **RELEASED**; MODEL-2 pretraining **loop driver landed** (task #105 CLOSED — commit `9a5af3ac2`); 370M Llama scaffold + pretrain loop + `apr pretrain` CLI all dogfood-ready; Zero-Tolerance design principle codified (§3 row #8); `pv validate` dogfooded across all 760 contracts (task #101); 8 legacy contracts backfilled with kani_harnesses + falsification parity (task #102 CLOSED); MODEL-2 `--min-frequency` threaded end-to-end through aprender-train BPE (task #103 CLOSED); gx10 third-party framework capacity gate PASS at 38.0 tok/s decode with 26.7% margin (task #104 CLOSED); loader hardened to ignore co-located ModelFamilyVariant contracts (task #108 CLOSED — 32→0 workspace-test failures); **task #132 BLOCKER surfaced 2026-04-21** on lambda-labs RTX 4090 — `apr pretrain --mode from-scratch` ran 14 min at 114% CPU + 0 MiB GPU memory because `TransformerTrainer::new` has no Device argument; `contracts/entrenar/gpu-training-backend-v1.yaml` PROPOSED (task #132 Phase 0) with INV-GPUTRAIN-001..007 + GATE-GPUTRAIN-001..006, ship-blocks task #126 real-compute dispatch
**Author:** PAIML Engineering
**Reviewer:** Noah Gift
**Date:** 2026-04-17 (v1.0.0) / 2026-04-17 (v2.0.0 audit + pivot) / 2026-04-18 (v2.5.0 pre-flight Poka-Yoke) / 2026-04-18 (v2.6.0 PM-008 GGUF tensor-type Poka-Yoke) / 2026-04-18 (v2.7.0 PM-009 APR magic-bytes Poka-Yoke) / 2026-04-18 (v2.8.0 HF Hub Xet large-file upload contract) / 2026-04-18 (v2.8.1 Xet impl landed) / 2026-04-18 (v2.9.0 EX-04 DISCHARGED via NDJSON lfsFile schema) / 2026-04-18 (v2.10.0 MODEL-1 v2 QLoRA divergence root cause — teacher-only ship) / 2026-04-18 (v2.11.0 EX-05/06/07 DISCHARGED — teacher tagged SHIP-TWO-001-MODEL-1-TEACHER) / 2026-04-18 (v2.12.0 post-ship artifacts — MODEL-2 contracts + MODEL-1 retry plan + SHARD-003 probe) / 2026-04-18 (v2.13.0 FALSIFY-SHARD-003 DISCHARGED live yoga vs gx10) / 2026-04-18 (v2.14.0 MODEL-2 dataset contract drafted + BPE NFC gap identified) / 2026-04-18 (v2.15.0 MODEL-2 scaffold LANDED — BPE NFC + tokenizer CLI + corpus ingest binary) / 2026-04-18 (v2.16.0 Zero-Tolerance design principle codified — no bugs, no perf regressions, no carve-outs) / 2026-04-18 (v2.17.0 contracts schema harmonization shipped — pv validate works across all 760 contracts, unblocks dogfooded gate) / 2026-04-18 (v2.18.0 parallel dispatch lanes #102/#103/#104 all closed — 8 contracts backfilled + MODEL-2 --min-frequency plumbed + gx10 38.0 tok/s PASS) / 2026-04-18 (v2.19.0 MODEL-2 pretrain loop driver landed via task #105 sub-agent — GATE-TRAIN-005 + INV-TRAIN-007 wired; `apr pretrain` CLI gated by `training` feature; loader hardened for ModelFamilyVariant contracts via task #108) / 2026-04-19 (v2.20.0 FALSIFY-SHIP-021 + FALSIFY-SHIP-022 DISCHARGED — MODEL-2 seed-reproducibility harness + apr inspect provenance block wired; tasks #112 #113 closed on chore/post-v2.19-evidence) / 2026-04-19 (v2.21.0 FALSIFY-SHIP-011 DISCHARGED + FALSIFY-SHIP-012/015 PARTIAL_ALGORITHM_LEVEL — C-LLAMA-370M-SOVEREIGN v1.0.0 PROPOSED → v1.2.0 ACTIVE with Rust-YAML byte-equality binding + param-count algorithm proof; C-TOK-BPE v1.1.0 wires 3 tokenizer tests; tasks #114 #115 #116 closed; 3/12 ACTIVE + 2/12 PARTIAL) / 2026-04-19 (v2.22.0 FALSIFY-SHIP-019 PARTIAL_ALGORITHM_LEVEL — C-LLAMA-370M-SOVEREIGN v1.2.0 → v1.3.0 stays ACTIVE; GATE-ARCH-370M-004 wired to 3 algorithm proofs reusing `layout_contract.rs` per Spec §9 Risk #2; task #117 closed on commit `846cc1dbb`; 3/12 ACTIVE + 3/12 PARTIAL = 6/12 touched) / 2026-04-21 (v2.23.0 **task #132 CUDA training backend gap** surfaced on lambda-labs RTX 4090 real-compute dispatch at commit `f7ad11408` — `apr pretrain --mode from-scratch` 14min on CPU (0 MiB GPU memory) because `TransformerTrainer` has no Device awareness; `contracts/entrenar/gpu-training-backend-v1.yaml` PROPOSED (Phase 0) with INV-GPUTRAIN-001..007 + GATE-GPUTRAIN-001..006 + FALSIFY-GPUTRAIN-001..007; production `CudaTransformerTrainer` already exists at `crates/aprender-train/src/train/transformer_trainer/cuda_trainer.rs` — gap is wiring, not kernels; task #126 real-compute dispatch BLOCKED until Phase 3 residency-proof evidence lands) / 2026-04-22 (v2.24.0 **FALSIFY-SHIP-008 PARTIAL_ALGORITHM_LEVEL** — task #155 wires MODEL-1 chat-template render gate: `contracts/chat-template-v1.yaml` v1.0.0 → v1.1.0 adds `GATE-CHAT-SHIP-008` binding `ChatMLTemplate::format_conversation` to the canonical Qwen2.5-Coder-7B golden via a pure `verdict_from_chat_template_render` const fn + 5-section mutation survey (empty / missing-gen-prompt / wrong-delim / swapped-roles / single-byte flip) + provenance pin; `cargo test -p aprender-core --lib falsify_ship_008_chat_template_render_bind` green; full discharge blocks on live `apr run paiml/qwen2.5-coder-7b-apache-q4k-v1` completion diff; **MODEL-1 coverage 1/10 → 2/10** touched — first MODEL-1 non-provenance PARTIAL, mirrors MODEL-2 pattern set by SHIP-016/017/018/020; 8 PARTIAL + 3 DISCHARGED across both models) / 2026-04-22 (v2.25.0 **FALSIFY-SHIP-006 PARTIAL_ALGORITHM_LEVEL** — task #156 wires MODEL-1 `apr qa` 8-gate aggregate gate: `contracts/apr-model-qa-v1.yaml` v1.1.0 → v1.2.0 adds `FALSIFY-QA-SHIP-006` binding the aggregate-AND verdict fn `verdict_from_qa_gates(&[bool]) -> Ship006Verdict` to the 8-gate ship criterion (golden / throughput / ollama parity / gpu speedup / tensor contracts / format parity / ptx parity / metadata per `docs/specifications/components/qa.md` §3) via pure const fn + 7-section mutation survey (all-Pass / all-Fail / single-gate-flip × 8 / exhaustive 2^8=256 bitmask proof / monotonicity / length drift 0-7-9-16 / provenance pin); `cargo test -p aprender-core --lib falsify_ship_006_apr_qa_eight_gates_aggregate` green; full discharge blocks on live `apr qa paiml/qwen2.5-coder-7b-apache-q4k-v1 --json` on RTX 4090 host; **MODEL-1 coverage 2/10 → 3/10** touched — mirrors MODEL-2 SHIP-016 aggregate-AND shape but authored self-contained because SHIP-016 branch not yet on main; 9 PARTIAL + 3 DISCHARGED across both models) / 2026-04-22 (v2.26.0 **FALSIFY-SHIP-002 PARTIAL_ALGORITHM_LEVEL** — task #159 wires MODEL-1 canonical `def fib(n):` Python-syntax gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.0.0 → v1.1.0 adds `FALSIFY-QW2E-SHIP-002` binding zero-tolerance `const fn verdict_from_syntax_error_count(usize) -> Ship002Verdict` in `crates/aprender-core/src/qa/ship_002.rs` + 6-section mutation survey (zero-errors → Pass / exactly-one-error → Fail / many-errors Fail band {2, 10, 100} / monotonicity sweep 0..=256 / `usize::MAX` sanity Fail / provenance pin tolerance = 0); `cargo test -p aprender-core --lib falsify_ship_002_python_syntax_error_threshold_logic` green; full discharge blocks on live `apr run paiml/qwen2.5-coder-7b-apache-q4k-v1 --prompt "def fib(n):"` on RTX 4090 + `rustpython`/`ruff` AST parse; **MODEL-1 coverage 3/10 → 4/10** touched — tightest MODEL-1 rule (0 tolerance on single canonical prompt vs MODEL-2 SHIP-017 which tolerates ≤1 across 100 held-out prompts) because spec §4.2 "emits valid Python" carries no noise allowance; 10 PARTIAL + 3 DISCHARGED across both models) / 2026-04-22 (v2.27.0 **FALSIFY-SHIP-005 PARTIAL_ALGORITHM_LEVEL** — task #158 wires MODEL-1 `apr eval --benchmark humaneval` pass@1 ship floor: `contracts/qwen2-e2e-verification-v1.yaml` v1.1.0 → v1.2.0 adds `FALSIFY-QW2E-SHIP-005` binding `AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT = 86.00` + `AC_SHIP1_005_NOISE_ALLOWANCE_PP = 1.20` + `AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT ≈ 84.80` to pure two-number threshold verdict fn `verdict_from_pass_at_1(correct, total, threshold_pct) -> Ship005Verdict` in `crates/aprender-core/src/metrics/ship_005.rs` + 8-section mutation survey (safe-margin Pass above effective / above nominal Pass / noise-window Fail at nominal / below-effective Fail including HumanEval-canonical 139/164 = 84.756% / monotonicity sweep 0..=164 / div-safety + sanity guards / non-finite threshold → Fail conservatively / tolerance-bounded provenance pin on all three constants because f32 `86.0 − 1.2 ≈ 84.79999924` ≠ exact 84.80); `cargo test -p aprender-core --lib falsify_ship_005_humaneval_pass_at_1_threshold_logic` green; full discharge blocks on live `apr eval --benchmark humaneval paiml/qwen2.5-coder-7b-apache-q4k-v1 --json` median of 3 seed=0 runs ≥ 86.00 (or ≥ 84.80 under the 1.2 pp noise allowance) on RTX 4090 with `--features cuda`; **MODEL-1 coverage 4/10 → 5/10** touched — mirrors MODEL-2 SHIP-018 pass@1 threshold shape but uniquely carries a 1.2 pp noise allowance (MODEL-2 has no noise window) and is self-contained because SHIP-018 PR #1004 and SHIP-007 PR #1019 are not yet on main; 11 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.28.0 **FALSIFY-SHIP-010 PARTIAL_ALGORITHM_LEVEL** — task #161 wires MODEL-1 published-artifact URL + SHA-256 ship gate: `contracts/publish-manifest-v1.yaml` v1.3.0 → v1.4.0 adds `FALSIFY-SHIP-010` binding two constants — `AC_SHIP1_010_SHA256_HEX_LEN = 64` (sha256sum canonical output) + `AC_SHIP1_010_REQUIRED_URL_SCHEME = "https://"` (TLS floor per §4.2) — to two pure verdict fns `verdict_from_sha256_match(expected_hex, actual_hex) -> Ship010Verdict` + `verdict_from_manifest_url(url) -> Ship010Verdict` in `crates/aprender-core/src/format/ship_010.rs` + twin 7-section mutation surveys: SHA-256 side covers identical-Pass / single-hex-flip / wrong-length / uppercase-rejected / non-hex-rejected / all-zero guard / provenance pin; URL side covers HF canonical / S3 canonical / plaintext-http rejected / scheme-less rejected / empty-host rejected / whitespace-control rejected / provenance pin; `cargo test -p aprender-core --lib format::ship_010` green (3/3 tests); full discharge blocks on live `curl -sSI <artifact_url>` 200-OK + `sha256sum <local_file> == <manifest_hash>` against `paiml/qwen2.5-coder-7b-apache-q4k-v1` on a host with network egress; **MODEL-1 coverage 5/10 → 6/10** touched — first MODEL-1 network-dependent PARTIAL, uniquely carries a TLS-floor byte-literal constant (no MODEL-2 counterpart because AC-SHIP2-012 is provenance-metadata not artifact-resolution); 12 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.29.0 **FALSIFY-SHIP-007 PARTIAL_ALGORITHM_LEVEL** — task #160 wires MODEL-1 `apr bench` decode throughput gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.2.0 → v1.3.0 adds `FALSIFY-QW2E-SHIP-007` binding `verdict_from_decode_tps(f32) -> Ship007Verdict` in `crates/aprender-core/src/bench/ship_007.rs` at `AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B = 30.0` tok/s + 7-section mutation survey (boundary at 30.0 → Pass / one-ULP-below → Fail / clear Pass band {45, 100} / clear Fail band {0, 10, 29.999999} / monotonicity above+below floor / non-finite → Fail conservatively {NaN, ±∞} / provenance pin 30.0); `cargo test -p aprender-core --lib falsify_ship_007_decode_tps_threshold_logic` green; full discharge blocks on live `apr bench --iterations 5 --max-tokens 128 paiml/qwen2.5-coder-7b-apache-q4k-v1` on RTX 4090 with `--features cuda` + median ≥ 30.0; **MODEL-1 coverage 6/10 → 7/10** touched — MODEL-1 twin of MODEL-2 SHIP-020 (identical f32-threshold shape, floor 30 vs 100 tok/s — 7B Q4_K is bandwidth-bound at ~3.5× the size of the 370M target); 13 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.30.0 **FALSIFY-SHIP-003 PARTIAL_ALGORITHM_LEVEL** — task #162 wires MODEL-1 `apr convert --quantize q4_k_m` per-layer cosine round-trip gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.3.0 → v1.4.0 adds `FALSIFY-QW2E-SHIP-003` binding `AC_SHIP1_003_MIN_COSINE_SIMILARITY = 0.999` to two pure verdict fns in `crates/aprender-core/src/format/ship_003.rs`: `verdict_from_cosine_similarity(sim, threshold) -> Ship003Verdict` (single-layer threshold + cosine-range guard `[-1.0, 1.0]` + non-finite guard) and `verdict_from_per_layer_cosines(sims, threshold) -> Ship003Verdict` (aggregate-AND combinator, conservative Fail on empty input) + twin 8-section and 7-section mutation surveys (exact 0.999 boundary Pass / ULP-below `0x3F7FBE76` Fail / safe-above {0.9999, 1.0} Pass / safe-below {0.998, 0.5, 0.0, −1.0} Fail / monotonicity sweep [0.990..=1.000] step 1e-4 / non-finite sim+threshold Fail / out-of-range sim+threshold Fail / provenance pin 0.999 on the single-layer side, plus all-Pass / 1-of-196 single-flip Fail / all-Fail / empty-Fail / single-element both directions / first-layer NaN+OOR short-circuit Fail / last-layer Fail not short-circuited on the aggregate side); `cargo test -p aprender-core --lib format::ship_003` green (2/2); full discharge blocks on live `apr convert --quantize q4_k_m paiml/qwen2.5-coder-7b-apache-q4k-v1.safetensors` on RTX 4090 + `apr diff` per-layer cosine harness walking 28 × 7 = 196 projection matrices; **MODEL-1 coverage 7/10 → 8/10** touched — eighth compute-free MODEL-1 PARTIAL lever, first to combine a single-number threshold (mirrors SHIP-007/SHIP-020 decode-tps shape) with an aggregate-AND combinator (mirrors SHIP-016 `verdict_from_qa_gates`) in one discharge; 14 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.31.0 **FALSIFY-SHIP-004 PARTIAL_ALGORITHM_LEVEL** — task #164 wires MODEL-1 `apr export --format gguf` → llama.cpp round-trip gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.4.0 → v1.5.0 adds `FALSIFY-QW2E-SHIP-004` binding three independent format-boundary verdict fns in `crates/aprender-core/src/format/ship_004.rs`: `verdict_from_llama_cli_exit(i32) -> Ship004Verdict` (POSIX zero-tolerance exit-code boundary bound to `AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE = 0`), `verdict_from_gguf_magic_bytes(&[u8]) -> Ship004Verdict` (canonical 4-byte `AC_SHIP1_004_GGUF_MAGIC_BYTES = b"GGUF"` with short-slice Fail and single-byte-flip rejection), and `verdict_from_gguf_version(u32) -> Ship004Verdict` (set-membership over `AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS = &[2, 3]` with Fail-closed above-band rejection) + triple mutation survey: exit-code 6 sections (POSIX success / adjacent-value Fail / classic failure bands {2, 127, 137, 139, 255} / i32 extrema / monotonicity sweep [-256, 256] / provenance pin); magic 6 sections (canonical Pass / magic+version header Pass / 6 single-byte flips / 4 short-slice lengths / wrong-format magics {b"APR\\0", b"APRN", zeros} / 4-byte provenance pin); version 5 sections (supported {2, 3} / predecessor {0, 1} / above-band {4, 5, 10, 100, 1M, u32::MAX} / exhaustive 0..=64 / set-length pin); `cargo test -p aprender-core --lib format::ship_004` green (3/3); full discharge blocks on live `apr export --format gguf paiml/qwen2.5-coder-7b-apache-q4k-v1.safetensors` on RTX 4090 + shell out to upstream `llama-cli -m qwen2.5-coder-7b.gguf --prompt "hello" --n-predict 4` and assert magic, version, AND exit code each Pass; **MODEL-1 coverage 8/10 → 9/10** touched — ninth compute-free MODEL-1 PARTIAL lever, first MODEL-1 discharge binding three independent verdict fns in one AC (each on a different format boundary — executable tool, magic bytes, version set); 15 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.32.0 **FALSIFY-SHIP-001 PARTIAL_ALGORITHM_LEVEL** — task #115 wires MODEL-1 `realizar::Model::load_safetensors` safetensors-load ship gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.5.0 → v1.6.0 adds `FALSIFY-QW2E-SHIP-001` binding three independent pure verdict fns in `crates/aprender-core/src/format/ship_001.rs`: `verdict_from_load_result(bool) -> Ship001Verdict` (Result-boundary collapse to Pass-on-Ok with no further decoding required), `verdict_from_safetensors_header_size(u64, u64) -> Ship001Verdict` (safetensors header-size invariant `0 < N <= file_len − AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN` bound at `AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN = 8` per the canonical little-endian u64 prefix), and `verdict_from_safetensors_json_open_byte(u8) -> Ship001Verdict` (byte-literal check that the JSON header starts with `AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE = b'{' = 0x7B`) + triple mutation survey: Result-boundary 2 sections (Ok → Pass / Err → Fail / provenance pin on the two-state boundary); header-size multiple sections (canonical in-band Pass / zero-size Fail / overflow-past-file-len Fail / off-by-one at `file_len - 8` boundary / 8-byte prefix constant pin); JSON open-byte sections (canonical `0x7B` Pass / adjacent-byte Fail / exhaustive 0..=255 sweep with only `0x7B` allowed / 0x7B constant pin); `cargo test -p aprender-core --lib format::ship_001` green (3/3); full discharge blocks on live `realizar::Model::load_safetensors(paiml/qwen2.5-coder-7b-apache-q4k-v1.safetensors)` on RTX 4090 with `--features cuda` (or a realizar binary equivalent that loads the tensor index and asserts the Result is Ok); **MODEL-1 coverage 9/10 → 10/10** touched — tenth compute-free MODEL-1 PARTIAL lever, completes MODEL-1 to 10/10 touched; only AC-SHIP1-009 (license / provenance metadata) remains untouched in the table if you count SHIP-009 as pending; second triple-verdict decomposition after SHIP-004, reinforcing the pattern where a tool-accepted-the-artifact rule is split into independent format-boundary gates (Result-boundary × header-size × open-brace byte); 16 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.33.0 **FALSIFY-SHIP-009 PARTIAL_ALGORITHM_LEVEL** — task #116 wires the FINAL MODEL-1 AC row (AC-SHIP1-009 "MODEL-1 teacher license + data provenance recorded in `model.apr` metadata") via a SECOND binding on `contracts/apr-provenance-v1.yaml` v1.0.0 → v1.1.0 (stays ACTIVE); the same contract already discharges MODEL-2's AC-SHIP2-012, so one contract now cleanly carries BOTH model bindings — the AprV2Metadata + serde-JSON decision rule is model-agnostic; GATE-APR-PROV-004 is added as the 2nd gate on that SAME contract (first multi-model multi-bind across the entire SHIP-TWO-001 surface); two harness tests in `crates/aprender-core/src/format/tests/provenance_tests.rs`: `falsify_ship_009_apr_metadata_applies_to_model_1_teacher` drives an AprV2Metadata teacher-representative round-trip (license="apache-2.0" + data_source="qwen2.5-coder-7b-instruct" + data_license="apache-2.0") through the serde-JSON path and asserts field-level recovery, and `falsify_ship_009_gate_apr_prov_004_has_partial_discharge_marker` uses `include_str!` on the YAML contract and `serde_yaml::Value` to assert that the new GATE-APR-PROV-004 block carries the correct `binds_to` / `falsification_id` / `discharge_status: PARTIAL_ALGORITHM_LEVEL` / `ship_blocking: true` flags — a byte-equal YAML-to-Rust binding test in the same style as the SHIP-011 Rust-scaffold binding; `cargo test -p aprender-core --lib provenance` green (5/5 including the 3 pre-existing SHIP-022 tests + the 2 new SHIP-009 tests); full ACTIVE promotion blocks on teacher `model.apr` republish under PMAT-686 populating license / data_source / data_license as named fields (fixture-swap only, no code change); **MODEL-1 coverage 9/10 → 10/10 touched** — SHIP-009 is the last MODEL-1 AC row needing a PARTIAL annotation, so MODEL-1 is now fully covered at the PARTIAL_ALGORITHM_LEVEL ship-gate surface; first multi-model multi-bind on a single contract (proves `apr-provenance-v1`'s decision rule is not model-specific) and sixth falsification of the repeatedly stated "exhausted" verdict (SHIP-019 → SHIP-017 → SHIP-020 → SHIP-018 → SHIP-016 → SHIP-009), this one strictly more surprising than the prior five because it is cross-model rather than just another MODEL-2 lever; 17 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.34.0 **FALSIFY-SHIP-017 PARTIAL_ALGORITHM_LEVEL (restacked)** — task #149 wires MODEL-2 (albor 370M Sovereign) `apr run` held-out Python-syntax gate on top of the v2.33.0 MODEL-1 stack: `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.5.0 → v1.6.0 (stays ACTIVE) adds `GATE-ARCH-370M-005` binding AC-SHIP2-007 ↔ FALSIFY-SHIP-017 with `discharge_status: PARTIAL_ALGORITHM_LEVEL`; decision rule is a pure integer threshold — "≤ 1 SyntaxError tolerated out of 100 held-out prompts, ≥ 2 is a ship-blocker" — bound in `crates/aprender-train/src/models/llama_370m.rs` via `AC_SHIP2_007_HELDOUT_PROMPT_COUNT = 100` + `AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS = 1` consts feeding `verdict_from_syntax_error_count(errors: usize) -> Ship017Verdict` + 2 falsification tests (`falsify_ship_017_syntax_error_count_threshold_logic` covers Pass boundary 0/1 / Fail boundary 2/50/100 / monotonicity ∈ [0, 100] / provenance pin; `falsify_ship_017_gate_arch_370m_005_has_partial_discharge_marker` byte-binds the sovereign contract YAML shape via `include_str!`); `cargo test -p aprender-train --lib llama_370m` → 12/12 pass; full discharge blocks on real trained 370M .apr + 100-prompt `apr run` harness with EX-06-style Python AST parse (AC-SHIP2-003/004 pretraining compute-dispatch); **MODEL-2 coverage 4/12 → 5/12** touched — this is the first MODEL-2 PARTIAL to land on top of a completed 10/10 MODEL-1 stack, carrying the same integer-threshold shape as MODEL-1 SHIP-002 but with a 1-error tolerance (MODEL-1 has 0 on a single canonical prompt) to accommodate the 100-prompt held-out harness noise floor; 18 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.35.0 **FALSIFY-SHIP-020 PARTIAL_ALGORITHM_LEVEL (restacked)** — task #150 wires MODEL-2 `apr bench` decode-throughput gate on top of the v2.34.0 SHIP-017 restack: `contracts/model-families/llama-370m-sovereign-v1.yaml` stays at v1.6.0 ACTIVE adding `GATE-ARCH-370M-006` binding AC-SHIP2-010 ↔ FALSIFY-SHIP-020 with `discharge_status: PARTIAL_ALGORITHM_LEVEL`; decision rule is a pure f32 threshold — "median decode throughput ≥ 100 tok/s on RTX 4090 (370M target)" — bound in `crates/aprender-train/src/models/llama_370m.rs` via `AC_SHIP2_010_MIN_DECODE_TPS_RTX4090 = 100.0` + `verdict_from_decode_tps(measured_tps: f32) -> Ship020Verdict` + 2 falsification tests (`falsify_ship_020_decode_tps_threshold_logic` covers exact 100.0 Pass / one-ULP-below Fail / generous-green {120, 500} / hard-red {0, 50} / monotonicity in both directions / non-finite conservative Fail for {NaN, ±∞} / provenance pin; `falsify_ship_020_gate_arch_370m_006_has_partial_discharge_marker` byte-binds the sovereign YAML shape via `include_str!`); full discharge blocks on real trained 370M .apr + 3 seed=0 `apr bench --tokens 128 --json` medians on RTX 4090; **MODEL-2 coverage 5/12 → 6/12** touched — MODEL-2 twin of MODEL-1 SHIP-007 (identical f32-threshold shape, floor 100 vs 30 tok/s since the 370M target is ~3.5× smaller than the 7B Q4_K teacher); 19 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.36.0 **FALSIFY-SHIP-018 PARTIAL_ALGORITHM_LEVEL (restacked)** — task #151 wires MODEL-2 `apr eval --benchmark humaneval` pass@1 ship floor on top of the v2.35.0 SHIP-020 restack: `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.7.0 → v1.8.0 (stays ACTIVE) adds `GATE-ARCH-370M-007` binding AC-SHIP2-008 ↔ FALSIFY-SHIP-018 with `discharge_status: PARTIAL_ALGORITHM_LEVEL`; decision rule is a pure (correct, total, threshold_pct) → Pass/Fail comparison — "HumanEval pass@1 ≥ 30.0% on 164 tasks" — bound in `crates/aprender-train/src/models/llama_370m.rs` via `AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT = 30.0` + `verdict_from_pass_at_1(correct, total, threshold_pct) -> Ship018Verdict` + 2 falsification tests (`falsify_ship_018_humaneval_pass_at_1_threshold_logic` covers inclusive floor (30/100, 60/200, 50/164 = 30.49% all Pass) + just-below Fail (49/164 ≈ 29.88%, 29/100 = 29.0%) + inclusive-floor proof at f32-exact 50/100 with ±ULP asymmetry + generous-green {82/164, 164/164} + hard-red {0/164, 1/164} + monotonicity sweep correct∈[0,164] + div-safety total=0 Fail + correct>total sanity Fail + non-finite threshold {NaN, ±∞} conservative Fail + provenance pin 30.0; `falsify_ship_018_gate_arch_370m_007_has_partial_discharge_marker` byte-binds the sovereign YAML shape via `include_str!`); full discharge blocks on real trained 370M .apr + 3 independent `apr eval --benchmark humaneval --json` seed=0 medians each feeding `verdict_from_pass_at_1`; **MODEL-2 coverage 6/12 → 7/12** touched — MODEL-2 twin of MODEL-1 SHIP-005 (pass@1 threshold; MODEL-2 has 30.0% floor with no noise allowance vs MODEL-1's 86.0% ± 1.2pp because the 370M target is a trained-from-scratch student, not a distilled teacher); 20 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.37.0 **FALSIFY-SHIP-016 PARTIAL_ALGORITHM_LEVEL (restacked)** — task #152 wires MODEL-2 `apr qa <model>.apr` 8-gate aggregate gate on top of the v2.36.0 SHIP-018 restack: `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.8.0 → v1.9.0 (stays ACTIVE) adds `GATE-ARCH-370M-008` binding AC-SHIP2-006 ↔ FALSIFY-SHIP-016 with `discharge_status: PARTIAL_ALGORITHM_LEVEL`; decision rule is a pure aggregate-AND over 8 Boolean gate-results (golden_output / throughput / ollama_parity / gpu_vs_cpu_speedup / tensor_contract / cross_format_parity / ptx_parity / probar) bound in `crates/aprender-train/src/models/llama_370m.rs` via `AC_SHIP2_006_REQUIRED_QA_GATE_COUNT = 8` + `verdict_from_qa_gates(&[bool]) -> Ship016Verdict` + 2 falsification tests (`falsify_ship_016_apr_qa_aggregate_and_logic` covers exhaustive 2^8 = 256-combination proof (exactly 1 input yields Pass, 255 yield Fail) + 8-way single-gate-flip falsifiability + monotonicity + 4 contract-drift guards (length {0, 7, 9, 16} → Fail even when all-true) + provenance pin; `falsify_ship_016_gate_arch_370m_008_has_partial_discharge_marker` byte-binds the sovereign YAML shape via `include_str!`); full discharge blocks on real trained 370M .apr + exit-0 `apr qa <model>.apr` with all 8 gates PASS; **MODEL-2 coverage 7/12 → 8/12** touched — first MODEL-2 aggregate-AND PARTIAL (mirrors MODEL-1 SHIP-006's shape exactly, since both models gate on the same 8-of-8 `apr qa` aggregate); this completes the compute-free MODEL-2 PARTIAL harvest — remaining 4 MODEL-2 gates are all genuinely compute-bound (003 val loss / 004 wall-clock / 005 already covered / 009 handled via AC-SHIP2-009 from SHIP-019); 21 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.38.0 **FALSIFY-SHIP-013 + FALSIFY-SHIP-014 BUNDLED PARTIAL_ALGORITHM_LEVEL** — task #118 discharges the last two untouched MODEL-2 AC rows (AC-SHIP2-003 val CE ≤ 2.2 + AC-SHIP2-004 training ≤ 21 days on RTX 4090) as a SINGLE bundled PR on top of the v2.37.0 SHIP-016 restack: `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.9.0 → v1.10.0 (stays ACTIVE) adds TWO new gates in one bump — GATE-ARCH-370M-013 binding AC-SHIP2-003 ↔ FALSIFY-SHIP-013 via pure f32 threshold fn `verdict_from_val_ce_loss(f32) -> Ship013Verdict` + const `AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS = 2.2` + 7-section mutation survey (exact 2.2 boundary Pass inclusive-floor / ULP-above Fail + ULP-below Pass asymmetry / clear Pass band {0.0, 0.5, 1.0, 2.0, 2.199} / clear Fail band {2.201, 3.0, 10.0, f32::MAX} / non-finite {NaN, ±∞} conservative Fail / negative-CE domain-violation Fail because H(p,q) ≥ 0 by definition / 2.2 provenance pin); and GATE-ARCH-370M-014 binding AC-SHIP2-004 ↔ FALSIFY-SHIP-014 via pure u32 threshold fn `verdict_from_training_duration_days(u32) -> Ship014Verdict` + const `AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS = 21` + 6-section mutation survey (exact 21 boundary Pass inclusive-ceiling / adjacent 20→Pass + 22→Fail / clear Pass band {0, 1, 7, 14, 20, 21} / clear Fail band {22, 30, 100, u32::MAX} / monotonicity sweep 0..=42 flipping exactly once at 21→22 / 21 provenance pin); both verdict fns colocated in `crates/aprender-train/src/models/llama_370m.rs`; `cargo test -p aprender-train --lib ship_013` + `cargo test -p aprender-train --lib ship_014` each green (1/1); full discharge of SHIP-013 blocks on live `apr pretrain --mode from-scratch --validate` loop on RTX 4090 with `--features cuda` producing a real MODEL-2 val cross-entropy at the final validation step; full discharge of SHIP-014 blocks on real wall-clock measurement of a MODEL-2 pretraining run from first `apr pretrain` dispatch to final checkpoint write, converted to integer days; **MODEL-2 coverage 8/12 → 12/12 PARTIAL_ALGORITHM_LEVEL touched — completes MODEL-2** — this is the first bundled double-discharge on the SHIP-TWO-001 surface, lifting the last two genuinely compute-bound gates (SHIP-013 and SHIP-014 both block on actual pretraining compute) to PARTIAL at the decision-rule level ahead of the AC-SHIP2-003/004 compute-dispatch; across both models: 23 PARTIAL + 3 DISCHARGED) / 2026-04-23 (v2.39.0 spec hygiene — back-annotates §5.2 MODEL-2 AC table (6 rows: 3 DISCHARGED + 3 PARTIAL_ALGORITHM_LEVEL from v2.20/21/22 amendments) + §7.1/§7.2 Falsification Tests (12 new PARTIAL cross-references) so the tables are a true single source of truth for SHIP-TWO-001 algorithm-level coverage; task #119; no Rust/contract/test changes) / 2026-04-23 (v2.40.0 **FALSIFY-SHIP-023 + FALSIFY-SHIP-024 BUNDLED PARTIAL_ALGORITHM_LEVEL** — task #120 discharges the last two MODEL-1 §7.1 stability rows (FALSIFY-SHIP-023 cross-run score drift + FALSIFY-SHIP-024 adversarial-suite 0-tolerance) as a SINGLE bundled PR on top of the v2.39.0 back-annotation: `contracts/qwen2-e2e-verification-v1.yaml` v1.6.0 → v1.7.0 adds TWO new falsification_tests in one bump — FALSIFY-QW2E-SHIP-023 binding `verdict_from_score_drift(day1_pct, day2_pct, tolerance_pp) -> Ship023Verdict` in `crates/aprender-core/src/format/ship_023.rs` at `AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP = 1.2` + 7-section mutation survey (exact 1.2 boundary Pass / drift = 1.2 + 1e-4 Fail / clear Pass band {0, 0.5, 1.0, 1.199} / clear Fail band {1.3, 2.0, 10.0, 86.0} / symmetric `.abs()` order invariance / non-finite {NaN, ±∞} conservative Fail / out-of-range day values {−0.1, 100.1} + negative-tolerance + zero-tolerance boundary + degenerate 0.0/100.0 cases + provenance pin); and FALSIFY-QW2E-SHIP-024 binding `const fn verdict_from_adversarial_suite(inputs_run: usize, panic_count: u32, nan_count: u32) -> Ship024Verdict` in `crates/aprender-core/src/format/ship_024.rs` at `AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE = 50` + `AC_SHIP1_024_MAX_TOLERATED_PANIC_COUNT = 0` + `AC_SHIP1_024_MAX_TOLERATED_NAN_COUNT = 0` + 7-section mutation survey (zero-tolerance boundaries (50,0,0) Pass + (50,1,0) (50,0,1) Fail / insufficient suite {0, 49, 1, 10, 25, 40} Fail / over-size Pass band {51, 100, 200, 500, 1000, 10_000, usize::MAX} / single-failure-class counts {panic∈[1,1000], nan∈[1,1000]} / compound failures {(50,1,1), (50,5,5), (10,10,10)} / u32::MAX overflow guard / all-three-constants provenance pin); `cargo test -p aprender-core --lib format::ship_023` + `cargo test -p aprender-core --lib format::ship_024` each green (1/1); full discharge of SHIP-023 blocks on live 2-day `apr eval --benchmark humaneval paiml/qwen2.5-coder-7b-apache-q4k-v1 --json` re-run on RTX 4090 with `--features cuda` + `(day1 − day2).abs() ≤ 1.2`; full discharge of SHIP-024 blocks on real 50-prompt adversarial torture suite feeding `verdict_from_adversarial_suite` with panic and NaN-logit counters; **completes MODEL-1 §7.1 at 12/12 falsification tests algorithmically bound** — SHIP-023 + SHIP-024 are the FIRST non-ship-blocking PARTIAL levers on the SHIP-TWO-001 surface because they live in §7.1 stability tests, not §4.2 AC table; SHIP-023's 1.2 pp budget is intentionally shared with SHIP-005's `AC_SHIP1_005_NOISE_ALLOWANCE_PP` (same budget, different semantics: noise vs nominal vs drift between two runs); across both models: 25 PARTIAL + 3 DISCHARGED) / 2026-04-23 (v2.41.0 **FALSIFY-GPUTRAIN-003..007 BUNDLED PARTIAL_ALGORITHM_LEVEL** — task #121 wires §14 Phase 2 algorithm-level discharges for the last five unbound FALSIFY-GPUTRAIN tests (001 and 002 already bound in `crates/aprender-train/src/train/device.rs` at 17 tests): `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED → v1.1.0 PROPOSED (stays PROPOSED until Phase 3 live evidence) adds FIVE new `evidence_discharged_by` + `full_discharge_blocks_on` + `counter_example_classes` blocks on one bump — FALSIFY-GPUTRAIN-003 binding `nvidia-smi --query-compute-apps` parser + residency verdict via pure `parse_nvidia_smi_compute_apps(&str) -> Result<Vec<NvidiaSmiComputeApp>, ()>` + `verdict_from_residency(u32, &[NvidiaSmiComputeApp]) -> Gputrain003Verdict` bound to `AC_GPUTRAIN_003_NVIDIA_SMI_POLL_WINDOW_SECONDS = 5` + `AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB = 1` (7-section survey: happy path / zero-mem Fail / other-pid Fail / empty Fail / multi-process both orderings Pass / malformed lines parse-Err / u32::MAX-u64::MAX boundary / dual provenance pin); FALSIFY-GPUTRAIN-004 binding device-class dispatch invariant via pure `verdict_from_dispatch_label(&str, &str) -> Gputrain004Verdict` bound to disjoint `AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS = &["Cpu"]` + `AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS = &["Cuda"]` (7-section survey: cpu→cpu Pass / cuda→cuda Pass / cpu→cuda silent-promotion Fail / cuda→cpu task-#126 silent-fallback Fail / unknown labels Fail / empty string Fail / case-sensitivity `CPU` vs `Cpu` Fail / disjointness proof); FALSIFY-GPUTRAIN-005 binding 500-ms step-time ceiling via pure `const fn verdict_from_step_time_ms(f32, f32) -> Gputrain005Verdict` bound to `AC_GPUTRAIN_005_MAX_STEP_TIME_MS_RTX4090_370M = 500.0` (7-section survey: exact 500.0 inclusive-ceiling Pass / one-ULP above Fail / clear Pass band / clear Fail band incl. f32::MAX / non-finite Fail / negative measured + non-positive threshold Fail / provenance pin); FALSIFY-GPUTRAIN-006 binding same-device seed reproducibility via pure `const fn verdict_from_loss_delta(f32, f32)` + aggregate `fn verdict_from_loss_trajectories(&[f32], &[f32], f32)` bound to `AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA = 1e-5` (7-section survey: exact boundary Pass / ULP-above Fail / trajectory single-step-fail at k=42 / length mismatch 50-vs-100 Fail / empty Fail / NaN+∞ Fail / negative delta + negative tolerance + infinite tolerance Fail / provenance pin); FALSIFY-GPUTRAIN-007 binding `apr --version --json` schema + field-shape invariants via pure `fn verdict_from_version_json_keys(&[&str])` + `fn verdict_from_version_json_fields(&VersionJsonCudaFields)` bound to `AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS = &["cuda_feature", "cuda_runtime_available", "visible_devices"]` (7-section survey: all-keys-present Pass + extras tolerated / missing-each-key Fail × 3 / 3 valid (feature, runtime) combos Pass / claims-feature-without-runtime Fail (FM-GPUTRAIN-STALE-BUILD) / boundary exactly 16 Pass, 17 Fail, 100 Fail / combined happy path / provenance pin); `cargo test -p aprender-train --lib gputrain_003` + `gputrain_004` + `gputrain_005` + `gputrain_006` + `gputrain_007` each green (1/1), `cargo test -p aprender-train --lib device` still green (17/17) regression-clean; full discharge of every GPUTRAIN gate still blocks on the live lambda-labs RTX 4090 harness per §14 Phase 3 (nvidia-smi residency proof + 50-step median timing + 100-step cuda:0 seed=0 replay + `apr --version --json` dogfood); **§14 Phase 0 algorithm-level complete — 7/7 FALSIFY-GPUTRAIN tests now bound at or above PARTIAL_ALGORITHM_LEVEL** (GPUTRAIN-001/002 already bound in device.rs); promotes `gpu-training-backend-v1` v1.0.0 PROPOSED → v1.1.0 PROPOSED (stays PROPOSED until §14 Phase 3 lambda-labs RTX 4090 residency evidence lands and flips the C-GPUTRAIN-BACKEND status to ACTIVE per §14 Phase 4); across both models: 30 PARTIAL + 3 DISCHARGED) / 2026-04-23 (v2.42.0 **§6 Compound Ship Gates bundle — 6 PARTIAL_ALGORITHM_LEVEL bindings** — task #122 algorithmically binds the 6 bindable §6 compound ship gates (GATE-SHIP-001..006; 007-012 are CI/lint meta-policy and remain out of scope) via a NEW `contracts/compound-ship-gates-v1.yaml` v1.0.0 PROPOSED (kind: pattern, cross-cutting CompoundShipGatesContract) and 6 new sibling modules in `crates/aprender-core/src/format/gate_ship_00X.rs`: GATE-SHIP-001 binds `verdict_from_model1_ac_aggregate(&[bool]) -> GateShip001Verdict` to `AC_GATE_SHIP_001_MODEL_1_AC_COUNT = 10` via aggregate-AND + 6-section survey incl. exhaustive 2^10 = 1024 bitmask proof; GATE-SHIP-002 binds `verdict_from_model2_ac_aggregate(&[bool]) -> GateShip002Verdict` to `AC_GATE_SHIP_002_MODEL_2_AC_COUNT = 12` via aggregate-AND + 6-section survey incl. exhaustive 2^12 = 4096 bitmask proof; GATE-SHIP-003 binds `verdict_from_golden_output_diff(&[u8], &[u8]) -> GateShip003Verdict` via byte-identity + non-empty conservative Fail + 6-section survey (identical / length mismatch / single-byte flip / both-empty Fail / one-empty Fail / 10_000-byte stress); GATE-SHIP-004 binds `verdict_from_identical_humaneval_scores(f32, f32) -> GateShip004Verdict` via `to_bits()` bitwise-identity (STRICTLY STRICTER than FALSIFY-SHIP-023 1.2 pp drift) + 7-section survey (identical Pass / single-ULP Fail / close-but-not-equal Fail / non-finite Fail / out-of-range Fail / boundary 0.0 + 100.0 Pass / SHIP-023 contrast pin); GATE-SHIP-005 binds `verdict_from_license_metadata(&str, &str) -> GateShip005Verdict` to `AC_GATE_SHIP_005_REQUIRED_LICENSE_FIELD = "license"` via case-sensitive byte-equal + ASCII-printable guard + 6-section survey (happy path / case drift / empty / non-ASCII incl. NUL + BOM + tab + newline / trailing whitespace / provenance pin); GATE-SHIP-006 binds `const fn verdict_from_first_token_probability_delta(f32, f32, f32) -> GateShip006Verdict` to `AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA = 1e-3` via symmetric `.abs()` threshold + 7-section survey (delta=0 Pass / exact-tolerance Pass / one-ULP-over Fail / Pass band small probs + symmetric / out-of-range Fail / negative tolerance Fail / non-finite Fail / provenance pin); `cargo test -p aprender-core --lib format::gate_ship` green (6/6) + `cargo test -p aprender-core --doc format::gate_ship` green (6/6 doctests); `pv validate contracts/compound-ship-gates-v1.yaml` → 0 errors, 0 warnings; full discharge of each gate blocks on live compound-gate harness (all 10 per-AC MODEL-1 checks for GATE-SHIP-001 / all 12 per-AC MODEL-2 checks for GATE-SHIP-002 / `apr qa --golden-output` on pre+post quantize checkpoints for GATE-SHIP-003 / two consecutive `apr eval --seed 0` runs in the same session for GATE-SHIP-004 / `apr inspect .metadata.license` vs upstream HF card YAML for GATE-SHIP-005 / `apr run --emit-logprobs` + `apr export --format gguf` + `llama-cli --logits-all` first-token softmax diff for GATE-SHIP-006); promotes NEW compound-gates contract to PROPOSED; across both models: 36 PARTIAL + 3 DISCHARGED) / 2026-04-23 (v2.43.0 **§6 Compound Ship Gates meta-policy bundle — 6 PARTIAL_ALGORITHM_LEVEL bindings (12/12 §6 total)** — task #123 completes §6 by algorithmically binding the 6 merge-gate meta-policy rows (GATE-SHIP-007..012 — unwrap / contract-density / CI-green / deny / tdg / coverage) via `contracts/compound-ship-gates-v1.yaml` v1.0.0 → v1.1.0 (stays PROPOSED) and 6 new sibling modules in `crates/aprender-core/src/format/gate_ship_0XX.rs`: GATE-SHIP-007 binds `const fn verdict_from_unwrap_count(u32) -> GateShip007Verdict` to `AC_GATE_SHIP_007_MAX_TOLERATED_UNWRAP_COUNT = 0` via zero-tolerance threshold + 5-section survey (count=0 Pass / count=1 Fail / Fail band {2, 10, 100, u32::MAX} / monotonicity sweep 0..=256 / provenance pin); GATE-SHIP-008 binds `verdict_from_contract_density(u32, u32, f32) -> GateShip008Verdict` to `AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE = 1.0` via divide-by-zero-guarded ratio threshold + 7-section survey (10/10 @ 1.0 Pass / 9/10 @ 1.0 Fail / 10/10 @ 0.9 Pass / zero-total Fail / contracted > total sanity Fail / non-finite or OOR density Fail / provenance pin); GATE-SHIP-009 binds `const fn verdict_from_ci_aggregate(bool, bool, bool) -> GateShip009Verdict` to `AC_GATE_SHIP_009_REQUIRED_CHECK_COUNT = 3` via aggregate-AND + 8-section survey incl. exhaustive 2^3 = 8 bitmask proof + AND-symmetry across argument permutations + provenance pin on 3-count; GATE-SHIP-010 binds `const fn verdict_from_advisory_count(u32) -> GateShip010Verdict` to `AC_GATE_SHIP_010_MAX_TOLERATED_ADVISORY_COUNT = 0` via zero-tolerance threshold + 5-section survey mirroring GATE-SHIP-007 at the security-advisory semantic layer; GATE-SHIP-011 binds `const fn verdict_from_tdg_score(f32, f32) -> GateShip011Verdict` to `AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE = 90.0` via inclusive-floor threshold + 7-section survey (exact 90.0 Pass / ULP-below Fail / clear Pass band {95, 100, 92.5} / clear Fail band {89.9, 80, 0, 50} / non-finite Fail / negative measured + threshold range guards / provenance pin); GATE-SHIP-012 binds `const fn verdict_from_line_coverage_pct(f32, f32) -> GateShip012Verdict` to `AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT = 95.0` via inclusive-floor threshold + 7-section survey mirroring GATE-SHIP-011 at the line-coverage 95% floor; `cargo test -p aprender-core --lib format::gate_ship_0XX` green (6/6) + `cargo test -p aprender-core --doc format::gate_ship_0XX` green (6/6 doctests); `pv validate contracts/compound-ship-gates-v1.yaml` → 0 errors, 0 warnings; full discharge of each gate blocks on live CI tooling invocation (`cargo clippy --all-targets --all-features -- -D warnings` for GATE-SHIP-007; `pmat density --new-code --json` for GATE-SHIP-008; branch-protection `ci / gate` + `workspace-test` for GATE-SHIP-009; `cargo deny check advisories` for GATE-SHIP-010; `pmat tdg . --format json` for GATE-SHIP-011; `cargo llvm-cov report --json` for GATE-SHIP-012); **§6 Compound Ship Gates now 12/12 algorithmically bound** (6 ship-blocking 001..006 already covered by v2.42.0; 6 merge-gate 007..012 added this release); across both models: **42 PARTIAL + 3 DISCHARGED**). / 2026-04-24 (v2.44.0 **drift-prevention enforcement layer codified** — tasks #128 + #129 close the specification loop by pinning every quantitative README claim AND every `AC_*` threshold constant to a falsifiable re-derivation, so silent edits to either the narrative or the verdict-fn thresholds fail CI instead of shipping. No new PARTIAL levers — this is the meta-layer that keeps the 42-lever surface honest over time. TWO deliverables landed in PRs #1044 + #1046 off main: (1) `contracts/readme-claims-v1.yaml` v1.0.0 `kind: pattern` `status: enforced` + FALSIFY-README-001..004 + `scripts/check_readme_claims.sh` — binds the README's workspace crate count (`ls crates/`), provable contract count (`find contracts/ -name '*.yaml'`), CLI command count (`cargo run -p apr-cli --bin apr -- --help` subcommand lines), and apr-cookbook link presence (`grep -F 'apr-cookbook'`) to live repository state; `--regen` mode prints the current values for quick README edit, `--claim <name>` targets a single check; bashrs-clean with documented SC1020/SC1140/SC1009/SC2102 false-positive suppressions for regex-bracket parsing; `bash scripts/check_readme_claims.sh` → `PASS FALSIFY-README-001..004` on HEAD (80 / 1096 / 79 / present); (2) `crates/aprender-train/tests/ship_two_001_const_pinning.rs` — a 345-line integration test that imports every `AC_*` const across the 27 SHIP-TWO-001 verdict modules (4 crates) and asserts each value matches the spec section-by-section: 20 MODEL-1 consts (§4.2 AC-SHIP1-001..010 + §7.1 SHIP-023/024), 7 MODEL-2 consts (§5.2 AC-SHIP2-003..010), 10 compound-gate consts (§6 GATE-SHIP-001..012), 8 GPUTRAIN consts (§14 FALSIFY-GPUTRAIN-003..007); 44 value assertions + 1 tripwire meta-test (`pinned_const_count_tripwire` checks count stays synced to `^pub const AC_` across `crates/`); catches the class where a contributor edits `AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS` from 2.2 → 2.5 without touching the spec — every verdict fn still compiles, every unit test passes, but the ship floor silently drifts; `cargo test -p aprender-train --test ship_two_001_const_pinning` → 44 passed. Pairs with the README contract to close the loop: README contract guards narrative drift, const-pinning test guards threshold drift, `pv validate` guards contract-schema drift — all three fail CI on violation, all three re-derive from live repo state instead of frozen snapshots. Across both models: **42 PARTIAL + 3 DISCHARGED** (numbers unchanged from v2.43.0 — this is enforcement, not coverage; the 42 levers are now *durable* against silent drift). / 2026-04-24 (v2.45.0 **atomic-next-action clarity: Task #132 Phase 2 runtime wiring** — task #131. No code changes; 2-line spec edit (Version + new `**Atomic next action**` callout at top of header, ahead of Status line). Surfaces the distinction §14.5 already encodes but that was only discoverable by reading the full §14: Task #132 Phase 2 splits into TWO surfaces — **algorithm-level** (FALSIFY-GPUTRAIN-003..007 bound at PARTIAL_ALGORITHM_LEVEL, shipped v2.41.0, status: DONE) and **runtime wiring** (`SharedTrainer::CudaVariant` + `drive_real` dispatch glue to the existing `CudaTransformerTrainer`, status: PENDING, ~2 days Rust). Without runtime wiring, `apr pretrain --device cuda:0` silently runs on CPU — the exact Task #132 bug. With it, one lambda-labs dispatch promotes SHIP-013/014/016/017/018/020 from PARTIAL → DISCHARGED on live MODEL-2 weights, and MODEL-1 live-compute harnesses (apr convert q4_k_m / humaneval / apr qa) can run in parallel on the same host. After v2.44.0 made the 42-lever surface *durable*, this amendment makes the *next action* discoverable. Across both models: **42 PARTIAL + 3 DISCHARGED** (unchanged — still enforcement-layer, not coverage-layer).

**v2.21.0 amendment (2026-04-19):** Three MODEL-2 architecture + tokenizer
gates landed in the same post-v2.19 evidence window, on branch
`chore/post-v2.19-evidence`:

1. **FALSIFY-SHIP-011 (AC-SHIP2-001) — DISCHARGED** at commit `338c6eb3c`
   (task #114). `contracts/model-families/llama-370m-sovereign-v1.yaml`
   promoted v1.0.0 PROPOSED → v1.1.0 ACTIVE. Rust scaffold
   `Llama370MConfig` (crates/aprender-train/src/models/llama_370m.rs) now
   binds **byte-equally** to the YAML contract via the harness test
   `falsify_ship_011_rust_scaffold_matches_yaml_contract`, which uses
   `include_str!` to embed the contract at compile time and
   `serde_yaml::Value` to parse-and-compare every architecture.* and
   constraints.* field against the corresponding `Llama370MConfig::*`
   const. Any edit to either side that diverges fails
   `cargo test -p aprender-train --lib llama_370m` before a single step
   of compute runs. INV-ARCH-370M-002..008 remain enforced at compile
   time via `const _: () = Llama370MConfig::validate();`, so the
   compile-time tier is intact even without the new YAML-binding test.
   The deliberate *sibling* approach over amending `llama.yaml` with a
   `370m` entry is recorded in the discharge memo: albor's
   `tied_embeddings=true` and `rope_theta=10000.0` conflict with
   Meta Llama-3's family-wide `tied_embeddings=false` /
   `rope_theta=500000.0`, and GATE-ARCH-370M-001's
   "llama.yaml (or this sibling contract)" language explicitly permits
   it.

2. **FALSIFY-SHIP-012 (AC-SHIP2-002) — PARTIAL_ALGORITHM_LEVEL** at
   commit `2e8b8b8e2` (task #115). `contracts/tokenizer-bpe-v1.yaml`
   bumped v1.0.0 → v1.1.0, **status intentionally stays PROPOSED**.
   GATE-BPE-003 gains `evidence_discharged_by` pointing at 3 harness
   tests in
   `crates/apr-cli/tests/falsify_ship_012_tokenizer_roundtrip.rs`:
   byte-exact round-trip on a 20-doc Python-like holdout (ASCII
   keywords + Unicode identifiers + docstrings + emoji + combining
   marks) under `aprender::text::tokenize::BpeTokenizer`, standalone
   NFC idempotence (INV-BPE-005), and train/holdout disjointness. The
   gate's `evidence_required` explicitly asks for **10K** docs; the
   current harness runs 20 on a synthetic fixture, so the gate lands
   with `discharge_status: PARTIAL_ALGORITHM_LEVEL` and
   `full_discharge_blocks_on: "task #91 (10K The Stack v2 Python
   holdout)"`. The harness module doc-comment locks in the zero-rewrite
   swap path: when task #91's 10K corpus materializes, replacing
   `HOLDOUT_CORPUS` + `TRAIN_CORPUS` with shard readers is a data-only
   change, then the contract can bump to 2.0.0 and promote to ACTIVE.
   This is the first spec-level use of a PARTIAL gate inside a
   PROPOSED contract — the pattern is: if the algorithm is provable
   today but the production-scale evidence is deferred, wire the
   algorithm proof and surface the data gap as first-class contract
   state rather than leaving the `evidence_discharged_by` slot blank.

3. **FALSIFY-SHIP-015 (AC-SHIP2-005) — PARTIAL_ALGORITHM_LEVEL** at
   commit `bfb883199` (task #116). Sovereign contract v1.1.0 → v1.2.0,
   stays ACTIVE. GATE-ARCH-370M-003 gains `evidence_discharged_by`
   pointing at the pre-existing `estimated_param_count_within_contract_band`
   unit test plus the `estimated_param_count` /
   `estimated_stored_param_count` const fns in
   `crates/aprender-train/src/models/llama_370m.rs`. The gate's
   `evidence_required` asks for `apr inspect --json model.apr |
   jq '.param_count'` to yield an integer in [366_000_000,
   374_000_000]; no on-disk `.apr` exists pre-compute, so the gate
   lands with `discharge_status: PARTIAL_ALGORITHM_LEVEL` and
   `full_discharge_blocks_on: "real 370M .apr checkpoint from
   pretraining compute-dispatch (AC-SHIP2-003/004)"`. The unit test
   hard-asserts p ∈ [366M, 374M], |p − 370M|/370M < 5%, and that
   embedding tying reduces stored params by exactly
   VOCAB_SIZE × HIDDEN_DIM; any edit to `Llama370MConfig` that moves
   the count out of the INV-ARCH-370M-001 band fails
   `cargo test -p aprender-train --lib llama_370m` before any compute
   runs. Contract remains ACTIVE because SHIP-011 (not SHIP-015) is
   what gates the sovereign contract's ACTIVE promotion — a gate-level
   PARTIAL nested inside an ACTIVE contract is a valid shape.

**Pattern codified by v2.21.0 (PARTIAL_ALGORITHM_LEVEL):** when a gate's
`evidence_required` text describes a production-scale check (10K docs,
on-disk artifact, benchmark run) that is not yet runnable, but the
underlying invariant is provable at algorithm / compile / unit-test
level today, emit the gate with `evidence_discharged_by` listing the
algorithm proofs + `discharge_status: PARTIAL_ALGORITHM_LEVEL` +
`partial_discharge_note:` + `full_discharge_blocks_on:` +
`ship_blocking: true`. The last field is load-bearing: PARTIAL gates
MUST still block `apr publish` until full discharge lands. Downstream
auditors must treat `evidence_discharged_by` alone (without checking
`discharge_status`) as **not** sufficient green — the two fields
together are the authoritative read.

**v2.24.0 amendment (2026-04-22):** First **MODEL-1** PARTIAL discharge
lands on branch `feat/falsify-ship-009-partial-discharge` — and it does
so by reusing the exact contract that already discharges MODEL-2's
AC-SHIP2-012, proving the decision-rule / compute-harness separation
pattern extends across model families:

1. **FALSIFY-SHIP-009 (AC-SHIP1-009) — PARTIAL_ALGORITHM_LEVEL** (task
   #153). `contracts/apr-provenance-v1.yaml` v1.0.0 → v1.1.0, status
   **stays ACTIVE** (no downgrade — v1.0.0's MODEL-2 discharge is
   unaffected). A new `GATE-APR-PROV-004` block is appended alongside
   GATE-APR-PROV-001/002/003, binding `AC-SHIP1-009` / `FALSIFY-SHIP-009`
   with `discharge_status: PARTIAL_ALGORITHM_LEVEL`. The gate's rule
   declares that the SAME AprV2Metadata + serde-JSON decision rule
   proved for MODEL-2 also applies to MODEL-1, and the two new harness
   tests in `crates/aprender-core/src/format/tests/provenance_tests.rs`
   confirm it:

   - `falsify_ship_009_apr_metadata_applies_to_model_1_teacher` —
     constructs a teacher-representative `AprV2Metadata { license:
     Some("apache-2.0"), data_source:
     Some("qwen2.5-coder-7b-instruct"), data_license:
     Some("apache-2.0"), .. }`, round-trips it through `to_json` /
     `from_json`, and asserts byte-identical recovery of all three
     provenance fields with NO leak into the `custom` HashMap.
   - `falsify_ship_009_gate_apr_prov_004_has_partial_discharge_marker`
     — `include_str!()`s the updated contract YAML, parses via
     `serde_yaml`, and asserts the new gate has
     `binds_to: AC-SHIP1-009`,
     `falsification_id: FALSIFY-SHIP-009`,
     `discharge_status: PARTIAL_ALGORITHM_LEVEL`,
     `ship_blocking: true`, non-empty `evidence_discharged_by`, and
     both `partial_discharge_note` and `full_discharge_blocks_on` set.

   The algorithm is model-agnostic — only the input string values
   differ between the MODEL-1 teacher test and the MODEL-2 SHIP-022
   test. `full_discharge_blocks_on: "teacher .apr republish (planned
   regen lane, PMAT-686) with AprV2Metadata.license / data_source /
   data_license populated as named fields; apr inspect --json must
   emit each as non-null. Fixture-swap only — no code change, no
   contract change."` The teacher is already shipped on HF
   (paiml/qwen2.5-coder-7b-apache-q4k-v1) under Apache-2.0, distilled
   from Qwen2.5-Coder-7B-Instruct — the provenance fact is known; only
   embedding it in a .apr binary remains.

**Pattern extensions from v2.24.0:**

- **First MODEL-1 PARTIAL.** All prior algorithm-level PARTIAL work
  (SHIP-015, -017, -018, -019, -020, -016) targeted MODEL-2 ship
  gates. SHIP-009 is the first MODEL-1 gate to be PARTIAL-discharged
  via the decision-rule/compute-harness split pattern, demonstrating
  the pattern is not MODEL-2-specific.
- **First multi-model multi-bind on ONE contract.** Prior PARTIAL
  discharges each had their own contract. SHIP-009 attaches a second
  gate to `apr-provenance-v1` — a second binding to AC-SHIP1-009 sits
  alongside the existing v1.0.0 bindings to AC-SHIP2-012. Because the
  decision rule is literally the same code, one YAML file cleanly
  carries both discharges with no schema extension.
- **"Exhausted" verdict now falsified SIX times.** Counter-example
  hunting after v2.22.0's "exhausted" verdict has now found genuine
  PARTIAL levers six times: SHIP-019 → SHIP-017 → SHIP-020 → SHIP-018
  → SHIP-016 → SHIP-009. The sixth falsification is cross-model (a
  MODEL-1 gate discharged by a MODEL-2 contract) — strictly more
  surprising than the prior five.

Combined ship-gate status after v2.24.0:

- **MODEL-2:** 3/12 fully ACTIVE (001, 011, 012) + 7/12 PARTIAL (002,
  005, 006, 007, 008, 009, 010) = **10/12 touched (83.3%)**. Remaining
  2 truly compute-blocked: AC-SHIP2-003 (val loss ≤ 2.2), AC-SHIP2-004
  (≤21-day wall-clock).
- **MODEL-1:** 1/10 PARTIAL (009), with the remaining 9 gates already
  DISCHARGED via the tagged teacher release
  `SHIP-TWO-001-MODEL-1-TEACHER`. MODEL-1 provenance will flip to
  fully ACTIVE the moment teacher.apr republish (PMAT-686) populates
  the three named fields.

**v2.22.0 amendment (2026-04-19):** One additional MODEL-2 ship gate
attained PARTIAL_ALGORITHM_LEVEL in the same post-v2.19 evidence window,
on branch `chore/post-v2.19-evidence`:

4. **FALSIFY-SHIP-019 (AC-SHIP2-009) — PARTIAL_ALGORITHM_LEVEL** at
   commit `846cc1dbb` (task #117). Sovereign contract v1.2.0 → v1.3.0,
   stays ACTIVE. GATE-ARCH-370M-004 gains `evidence_discharged_by`
   pointing at two new harness tests + an enumerator helper in
   `crates/aprender-train/src/models/llama_370m.rs` plus three
   cross-referenced assets (`LayoutContract`, `validate_apr_shape`,
   `contracts/tensor-layout-v1.yaml`). The gate's `evidence_required`
   asks for GGUF-exported 370M first-token cosine similarity ≤ 1e-3 vs
   APR on 100 canary prompts — that runner is blocked on
   AC-SHIP2-003/004 pretraining compute plus GATE-SHIP-006 harness
   invocation, so the gate lands with `discharge_status:
   PARTIAL_ALGORITHM_LEVEL` + `full_discharge_blocks_on: "real 370M .apr
   checkpoint from pretraining compute-dispatch (AC-SHIP2-003/004) +
   harness invocation of GATE-SHIP-006 cosine-parity runner"`. The
   algorithm-level proofs collectively establish the conditional: *if*
   GGUF export invokes `LayoutContract::validate_apr_shape` on every
   tensor, *then* row-major layout and GH-202 regression rejection are
   mathematically enforced. The enumerator counts
   **3 + 9 × NUM_LAYERS = 219** tensors and cross-checks each with
   `LayoutContract::get_apr_contract`; adding a tensor to
   `Llama370MConfig` without a matching entry in
   `layout_contract_specs.rs` now fails
   `cargo test -p aprender-train --lib llama_370m` before any compute
   runs. Spec §9 Risk #2's explicit instruction to "reuse
   `layout_contract.rs` validator" was the load-bearing hint that
   pointed at a non-compute, algorithm-level asset.

**Pattern lesson codified by v2.22.0 (counter-example hunting):** the
v2.21.0 cycle declared all non-compute PARTIAL levers for MODEL-2
"exhausted". Re-running the 7-gate FALSIFY-SHIP survey (013/014/016/017/
018/019/020) with explicit counter-example hunting found exactly one
genuine lever (SHIP-019); SHIP-017/018/020 truly need compute,
SHIP-013/014/016 collapse into SHIP-011's wiring. Prior verdict was ~86%
correct. **Rule: before declaring a search space exhausted, re-run the
survey with explicit counter-example hunting — the spec's own Risk
mitigations are the highest-leverage hint source.**

Combined MODEL-2 ship-gate status after v2.22.0: **3/12 AC-SHIP2 gates
fully ACTIVE** (001, 011, 012) + **3/12 PARTIAL_ALGORITHM_LEVEL** (002
via SHIP-012, 005 via SHIP-015, 009 via SHIP-019) = **6/12 touched**
(50%). The remaining 6 (003/004/006/007/008/010) all require either
real 370M training compute, a trained on-disk `.apr` with evaluation
harness, or a wall-clock benchmark on RTX 4090, and will remain
untouched until compute-dispatch lands — the pretrain loop driver + CLI
from v2.19.0 are ready for them. Genuine algorithm-level PARTIAL
harvesting is now exhausted for MODEL-2.

**v2.34.0 amendment (2026-04-23):** After the full MODEL-1 stack landed
(v2.24.0 → v2.33.0 covering SHIP-008 → SHIP-006 → SHIP-002 → SHIP-005 →
SHIP-010 → SHIP-007 → SHIP-003 → SHIP-004 → SHIP-001 → SHIP-009, 10/10
MODEL-1 AC rows touched), MODEL-2 PARTIAL harvesting resumes on top of
the completed MODEL-1 surface — **FALSIFY-SHIP-017 (AC-SHIP2-007) —
PARTIAL_ALGORITHM_LEVEL** restacked at task #149. The decision rule of
SHIP-017 ("`apr run` produces syntactically valid Python on 100
held-out prompts; ≥ 2 SyntaxError → FAIL, tolerate ≤ 1") is a pure
integer threshold function that can be proven correct today, even
though the full 100-prompt harness requires a trained 370M .apr.
`contracts/model-families/llama-370m-sovereign-v1.yaml` bumped v1.5.0
→ v1.6.0 (stays ACTIVE) with new GATE-ARCH-370M-005 binding
AC-SHIP2-007 ↔ FALSIFY-SHIP-017 and `discharge_status:
PARTIAL_ALGORITHM_LEVEL`. Algorithm proof = two unit tests in
`crates/aprender-train/src/models/llama_370m.rs`:
(1) `falsify_ship_017_syntax_error_count_threshold_logic` — covers
Pass boundary (0, 1 errors), Fail boundary (2 errors), pathological
cases (50, 100 errors all Fail), monotonicity over all errors ∈
[0, 100], and provenance pinning
(`AC_SHIP2_007_HELDOUT_PROMPT_COUNT == 100`,
`AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS == 1`). (2)
`falsify_ship_017_gate_arch_370m_005_has_partial_discharge_marker` —
binds the sovereign contract YAML shape (falsification_id, binds_to,
discharge_status, evidence_discharged_by, full_discharge_blocks_on,
ship_blocking) to the Rust tests via `include_str!`. Full discharge
blocks on real trained 370M .apr + 100-prompt `apr run` harness with
EX-06-style Python AST parse pipeline — fixture swap only, no
harness rewrite. Unlike the original v2.25.0 discharge, SHIP-017 now
lands on a surface where MODEL-1 is already at 10/10 PARTIAL-touched,
so the integer-threshold shape is shared across models (MODEL-1
SHIP-002 = 0 tolerance on 1 canonical prompt; MODEL-2 SHIP-017 = 1
tolerance on 100 held-out prompts). New combined status: **3/12
ACTIVE + 5/12 MODEL-2 PARTIAL = 8/12 touched** on MODEL-2 side, with
MODEL-1 fully saturated at **10/10 PARTIAL**; 18 PARTIAL + 3
DISCHARGED across both models.

**v2.35.0 amendment (2026-04-23):** Second MODEL-2 PARTIAL in the
post-v2.33.0 MODEL-1-stack restacking window — **FALSIFY-SHIP-020
(AC-SHIP2-010) — PARTIAL_ALGORITHM_LEVEL** restacked at task #150 on
top of v2.34.0 SHIP-017. The decision rule of SHIP-020 ("`apr bench`
median decode throughput ≥ 100 tok/s on RTX 4090 for the 370M
target") is a pure f32 threshold function, separable from the
compute-heavy `apr bench --median` harness which requires a trained
370M .apr. `contracts/model-families/llama-370m-sovereign-v1.yaml`
stays at v1.6.0 ACTIVE with new GATE-ARCH-370M-006 binding
AC-SHIP2-010 ↔ FALSIFY-SHIP-020 and `discharge_status:
PARTIAL_ALGORITHM_LEVEL`. Algorithm proof = two unit tests in
`crates/aprender-train/src/models/llama_370m.rs`: (1)
`falsify_ship_020_decode_tps_threshold_logic` — covers exact 100.0
tok/s Pass boundary / one-f32-ULP below → Fail / generous-green
{120.0, 500.0} / hard-red {0.0, 50.0} / monotonicity in both
directions / non-finite inputs {NaN, ±∞} → conservatively Fail (a
real `apr bench` median is always a finite positive; +∞ as Pass
would let an instrumentation bug silently green the ship-gate) /
provenance pinning (`AC_SHIP2_010_MIN_DECODE_TPS_RTX4090 == 100.0`).
(2) `falsify_ship_020_gate_arch_370m_006_has_partial_discharge_marker`
— byte-binds the sovereign contract YAML shape (falsification_id,
binds_to, discharge_status, evidence_discharged_by,
full_discharge_blocks_on, ship_blocking) to the Rust tests via
`include_str!`. Full discharge blocks on a real trained 370M .apr +
three independent `apr bench --tokens 128 --json` medians on the
RTX 4090 host — fixture-swap only, no decision-rule rewrite.
SHIP-020 is the MODEL-2 twin of MODEL-1 SHIP-007 (same f32-threshold
shape; floor 100 tok/s for 370M vs 30 tok/s for 7B Q4_K because the
student is ~3.5× smaller than the teacher). New combined status:
MODEL-2 at **6/12 touched**; MODEL-1 still fully saturated at
**10/10 PARTIAL**; **19 PARTIAL + 3 DISCHARGED** across both models.

**v2.36.0 amendment (2026-04-23):** Third MODEL-2 PARTIAL in the
post-v2.33.0 MODEL-1-stack restacking window — **FALSIFY-SHIP-018
(AC-SHIP2-008) — PARTIAL_ALGORITHM_LEVEL** restacked at task #151 on
top of v2.35.0 SHIP-020. The decision rule of SHIP-018 ("`apr eval
--benchmark humaneval` pass@1 ≥ 30.0% on 164 tasks") is a pure
(correct, total, threshold_pct) → Pass/Fail comparison, separable
from the compute-heavy 164-task sampling harness which requires a
trained 370M .apr. `contracts/model-families/llama-370m-sovereign-v1.yaml`
v1.7.0 → v1.8.0 ACTIVE with new GATE-ARCH-370M-007 binding
AC-SHIP2-008 ↔ FALSIFY-SHIP-018 and `discharge_status:
PARTIAL_ALGORITHM_LEVEL`. Algorithm proof = two unit tests in
`crates/aprender-train/src/models/llama_370m.rs`: (1)
`falsify_ship_018_humaneval_pass_at_1_threshold_logic` covers
inclusive floor (30/100, 60/200, 50/164=30.49% all Pass) +
just-below Fail (49/164≈29.88%, 29/100=29.0%) + f32-exact 50/100
±ULP asymmetry proof of `>=` (inclusive, not `>`) + generous-green
{82/164, 164/164} + hard-red {0/164, 1/164} + monotonicity sweep
correct∈[0,164] with Fail→Pass allowed once, Pass→Fail forbidden +
div-safety (total=0 → Fail) + sanity (correct>total → Fail) +
non-finite threshold {NaN, +∞, −∞} → conservatively Fail + provenance
pin (`AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT == 30.0`). (2)
`falsify_ship_018_gate_arch_370m_007_has_partial_discharge_marker`
byte-binds the sovereign YAML shape via `include_str!`. Full
discharge blocks on a real trained 370M .apr + three independent
seed=0 `apr eval --benchmark humaneval --json` medians each feeding
`verdict_from_pass_at_1` → all three Pass — fixture-swap only, no
decision-rule rewrite. SHIP-018 is the MODEL-2 twin of MODEL-1
SHIP-005 (both pass@1 threshold gates; MODEL-2 30.0% floor with no
noise allowance vs MODEL-1 86.0% ± 1.2 pp noise window because the
370M student is trained from scratch whereas the 7B teacher is a
Qwen2.5-Coder-Instruct derivative). New combined status: MODEL-2 at
**7/12 touched**; MODEL-1 still fully saturated at **10/10 PARTIAL**;
**20 PARTIAL + 3 DISCHARGED** across both models.

**v2.37.0 amendment (2026-04-23):** Fourth (and final compute-free)
MODEL-2 PARTIAL in the post-v2.33.0 MODEL-1-stack restacking window
— **FALSIFY-SHIP-016 (AC-SHIP2-006) — PARTIAL_ALGORITHM_LEVEL**
restacked at task #152 on top of v2.36.0 SHIP-018. The decision rule
of SHIP-016 ("`apr qa <model>.apr` — all 8 gates PASS") is a pure
aggregate-AND over a Boolean slice, separable from the compute-heavy
gate runner which requires a trained 370M .apr and a real RTX 4090
host. `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.8.0
→ v1.9.0 ACTIVE with new GATE-ARCH-370M-008 binding AC-SHIP2-006 ↔
FALSIFY-SHIP-016 and `discharge_status: PARTIAL_ALGORITHM_LEVEL`.
Algorithm proof = two unit tests in
`crates/aprender-train/src/models/llama_370m.rs`: (1)
`falsify_ship_016_apr_qa_aggregate_and_logic` covers the exhaustive
2^8 = 256-combination sweep (exactly one input — all-true — yields
Pass; 255 yield Fail) + 8-way single-gate-flip falsifiability +
monotonicity (flipping false→true never regresses Pass→Fail) + 4
contract-drift guards (gate slice of length {0, 7, 9, 16} → Fail
conservatively even when supplied entries are all true) + provenance
pin (`AC_SHIP2_006_REQUIRED_QA_GATE_COUNT == 8`). (2)
`falsify_ship_016_gate_arch_370m_008_has_partial_discharge_marker`
byte-binds the sovereign YAML shape via `include_str!`. Full
discharge blocks on real trained 370M .apr + `apr qa <model>.apr`
exit 0 with all 8 gates green on RTX 4090 — fixture-swap only, no
harness rewrite. SHIP-016 is the MODEL-2 twin of MODEL-1 SHIP-006
(both 8-gate aggregate-AND; same 8 canonical gates). This completes
the compute-free MODEL-2 PARTIAL harvest within the restacking
window: remaining 4 MODEL-2 gates (003 val loss ≤ 2.2 / 004 ≤21-day
wall-clock) + already-touched {005, 009} / {007-covered, 008-covered,
010-covered via 017/018/020} are all either compute-bound or already
discharged. New combined status: MODEL-2 at **8/12 touched**;
MODEL-1 still fully saturated at **10/10 PARTIAL**; **21 PARTIAL + 3
DISCHARGED** across both models.

**v2.20.0 amendment (2026-04-19):** Two MODEL-2 ship gates **DISCHARGED**
in the post-v2.19 evidence window on branch `chore/post-v2.19-evidence`:

1. **FALSIFY-SHIP-021 (AC-SHIP2-011) — DISCHARGED** at commit `0b8ca8c84`
   (task #112). `falsify_ship_021_seed_0_100_step_reproducibility` proves
   two seed=0 × 100-step training runs produce |Δloss| ≤ 1e-6 at every
   step and bit-identical AdamW-state sha256; a counter-test
   `falsify_ship_021_different_seeds_do_diverge` proves seed=0 vs seed=1
   diverge > 1e-4 within 10 steps. Root cause of the original green-run
   flake (step-0 6.854 vs 6.928 under parallel cargo test) was a sibling
   test racing on the global `INIT_SEED` atomic; fix landed as
   `transformer::init::lock_init_seed(seed) -> MutexGuard` which any
   future caller doing concurrent weight init under a set-before-read
   global MUST hold across the full init work. Contract
   `training-loop-pretrain-v1.yaml` bumped 1.0.0 → 1.1.0,
   status PROPOSED → ACTIVE, INV-TRAIN-006 + GATE-TRAIN-006 got
   harness/evidence_discharged_by blocks.

2. **FALSIFY-SHIP-022 (AC-SHIP2-012) — DISCHARGED** at commit `8f0607d42`
   (task #113). `apr inspect` now surfaces the three provenance keys —
   `license`, `data_source`, `data_license` — from every .apr binary,
   rendering absent values as the literal `(missing)` in text mode and
   `null` in JSON mode. Key design: `AprV2Metadata` gained
   `data_source` + `data_license` as NAMED Option<String> fields (not
   buried in `custom: HashMap`); no `skip_serializing_if` is allowed on
   any provenance field on either `AprV2Metadata` or `MetadataInfo`,
   because silent-skip via serde is the exact failure mode
   (`FM-APR-PROV-SILENT-SKIP`) the contract guards against. Text
   rendering goes through a pure helper `format_provenance_block()` so
   tests assert on a returned `String` rather than capturing stdout
   (`gag::BufferRedirect` is NOT parallel-test-safe — recorded as a
   reusable pattern). New schema contract `apr-provenance-v1.yaml`
   (C-APR-PROVENANCE v1.0.0 ACTIVE, `kind: schema`) declares 3
   invariants (round-trip, always-emit, publish-gate-rejects), 3 gates,
   and 3 failure modes, all bound to AC-SHIP2-012. `pv validate` PASS
   (0 errors). Live smoke test on
   `qwen2.5-coder-1.5b-instruct-q4k.apr` (no provenance stored)
   correctly prints the Provenance block with `(missing)` on all three
   rows. Together with PM-003/PM-008/PM-009/PM-007 pre-flight gates,
   any operator can now answer "what data trained this, under what
   license?" from a `.apr` alone — the sidecar-manifest dependency is
   severed.

Combined MODEL-2 ship-gate status after v2.20.0: 2/12 AC-SHIP2 gates
DISCHARGED (011, 012). The remaining 10 (001–010) all block on the
actual 370M checkpoint, which is the compute-dispatch long-pole; the
pretrain loop driver from v2.19.0 is ready to exercise them once
compute-dispatch for real weights lands.

**v2.17.0 amendment (2026-04-18):** Task #101 contracts schema
harmonization **SHIPPED** on `feat/pm-007-preflight-poka-yoke` at commit
`4fc453d57`. Closes the last parser barrier preventing `pv validate`
from serving as the canonical dogfooded gate across all SHIP-TWO-001
contract work. `crates/aprender-contracts/src/schema/types.rs` now
(a) accepts legacy ProofObligation field spellings (`statement`/
`verification`) via `#[serde(alias)]`, (b) accepts both map
`{id: Equation}` and list `[{id, ...}]` equation forms via a custom
polymorphic `deserialize_equations`, (c) adds `Safety` + `Liveness` to
`ObligationType` (28 variants now, up from 26), (d) uplifts 6 legacy
contracts (decode-gpu-resident-sampling, decode-hot-path-*,
eval-harness-humaneval, eval-sharding, profile-graph-vs-per-op-
methodology, publish-manifest) to the metadata-block form. Target
tests `load_contracts_real` + `parse_missing_metadata_returns_error`
both green; 1368/1371 aprender-contracts lib tests pass. Remaining
3 failures (lint_passes_on_real_contracts, validate_gate_passes,
lint_findings_on_failure) are downstream content checks — empty
`formula:` bodies, missing `kani_harnesses`, falsifications <
proof_obligations on the same 6 legacy contracts — dispatched as
task #102 follow-up content-authoring lane. This amendment ties the
"pv not bash for contracts" MEMORY.md policy (2026-04-18) to
concrete unblocked state: no more adhoc bash/grep workarounds when
the dogfood tool covers the workflow.

**v2.18.0 amendment (2026-04-18):** Parallel dispatch lanes #102/#103/#104
**ALL CLOSED** in a single concurrent compute window against
non-overlapping surfaces, demonstrating the monorepo's sub-agent
workflow scales. Results:
(a) **#102 contract backfill CLOSED** — 8 legacy contracts
(`decode-gpu-resident-sampling`, `decode-hot-path-{first-tokens,
prefix-cache,zero-syscalls}`, `eval-harness-humaneval`, `eval-sharding`,
`profile-graph-vs-per-op-methodology`, `publish-manifest`) received
metadata references, formula bodies, kani_harnesses, and falsification
parity. 22 ERROR findings → 0, `lint_passes_on_real_contracts` green.
Verified live via `pv validate` dogfood: 8/8 contracts parse clean
(1 advisory SCHEMA-013 qa_gate-missing on eval-sharding kept as
forward work). No bash/grep workaround needed.
(b) **#103 MODEL-2 `--min-frequency` plumbing CLOSED** — `apr-cli`
tokenize call-site swapped from `aprender::text::tokenize::BpeTokenizer`
→ `entrenar::tokenizer::BPETokenizer::train` via `train_bpe_via_entrenar`
helper; `TokenizerConfig::bpe().with_min_frequency(..).with_normalization(..)`
now threads user-provided `--min-frequency` + `--normalization` into
merge pruning. Public read-only `vocab()`/`merges()` accessors added to
`aprender-train::tokenizer::BPETokenizer`. 17 `apr-cli` tokenize tests
pass including new `run_train_honors_min_frequency_pruning` which
asserts singleton byte-pairs ("xyz" single occurrence) are pruned
from `merges.txt`/`vocab.json` at threshold 2. Closes v2.15.0 §1
"Known gap". Redundant `build_normalizer()` call-site removed since
`aprender-train`'s BPE applies NFC internally (no double-normalization).
(c) **#104 gx10 capacity gate PASS** — llama.cpp (b1-b0f0dd3 CUDA)
on teacher GGUF (sha256 `e6cac5d6…7981`) measured **38.0 tok/s decode**
(prompt eval 509.0 tok/s, 7.7 GiB VRAM, 5 s wall) vs 30 tok/s gate
threshold = PASS 26.7% margin. 2.45× the forbidden 15.5 tok/s fused
NF4 steady-state fallback — Zero-Tolerance §3 row #8 "no perf
regression" clause preserved. Two follow-ups flagged: (1) decode drift
from memory's 46 tok/s → 38.0 tok/s on current build; (2) gx10 disk
95% full (44 GB free) needs cleanup before MODEL-2 7B training lands.
Evidence: `evidence/ship-two-001/gx10-capacity-baseline-20260418-213928.json`.

With #102+#103 closed, **task #105** (370M MODEL-2 pretraining loop
wiring per `training-loop-pretrain-v1` GATE-TRAIN-005) is now the sole
long-pole item. Expected surface: `aprender-train/src/train/pretrain.rs`
loop driver calling the `llama_370m` forward pass with AdamW optimizer
and gradient accumulation, gated by the dataset ingest binary shipped
in v2.15.0.

**v2.19.0 amendment (2026-04-18):** Task #105 **CLOSED** via background
sub-agent `ac479445bcd722bf7` — commit `9a5af3ac2` on
`feat/pm-007-preflight-poka-yoke`. Surface landed (6 files, +1379 LOC):
(a) `crates/aprender-train/src/train/pretrain.rs` (963 LOC) — PretrainConfig
with `model_2_defaults()` baking LR=5e-5 + rank=32 + seed=42 remedies
from MODEL-1 v2 QLoRA divergence post-mortem;
(b) `crates/apr-cli/src/commands/pretrain.rs` (332 LOC) — CLI entrypoint
gated behind `training` cargo feature;
(c) extended_commands.rs + dispatch_analysis.rs — wired `apr pretrain`
into the apr-cli dispatch table.
Contract compliance verified: `contracts/training-loop-pretrain-v1.yaml`
passes `pv validate` with 0 errors; GATE-TRAIN-005 (val_loss[N] ≤ 2.0×
val_loss[N-1]) wired in `check_non_divergence`; INV-TRAIN-007 NaN/Inf
guard wired in `check_numerical_stability` before metric logging;
GATE-TRAIN-008 throughput bounds wired via `PretrainAbort::ThroughputOutOfRange`.
`per_step_metrics.required` and `per_epoch_artifacts.required_fields`
enforced as struct invariants. Checkpoint path template
`{run_dir}/ckpt/epoch-{N:03d}.apr` frozen in `EpochArtifact::new`.
Synthetic drive via injected `StepFn`/`ValFn` traits allows exercising
the full gate surface today while the real 370M forward pass wiring
(llama_370m.rs) completes. Test verification: 15/15 pretrain unit tests
pass, 3/3 CLI tests pass, 947/947 aprender-train lib tests no
regressions. Abort errors map 1:1 to contract gate IDs so operators
see the tripped gate via shell `$?`.

Concurrent with #105, **task #108** closed the 32-way workspace-test
regression discovered by CI run 24614757928. Root cause: five directory
iterators in `aprender-core/src/format/` were treating
`contracts/model-families/llama-370m-sovereign-v1.yaml` (a
ModelFamilyVariant CONTRACT starting with `contract_id:`) as a
ModelFamily REGISTRY entry. Fix (commit `21d43bd7a`): all iterators
now skip files whose first top-level key is `contract_id:` (family
registry YAMLs all begin with `metadata:` — a clean discriminator,
verified by corpus scan). `cargo test -p aprender-core --lib format::`
re-green at 13031 passed / 0 failed.

The ci/lint workspace package-ambiguity blocker (`aprender@0.27.8` vs
`aprender@0.31.0` — caused by transitive deps on published
`realizar ^0.7/^0.8`, `renacer ^0.9/^0.10`, `trueno ^0.15/^0.16/^0.17`,
`entrenar ^0.7`, `bashrs ^6.35/^6.65`, `pacha ^0.2` that all re-export
old `aprender@0.27`) was split into task #109 for a separate
path-dependency migration pass. This is orthogonal to SHIP-TWO-001 and
pre-dated the branch; the monorepo's `[patch.crates-io]` block was
removed during RC4 cc-cleanup and was never restored for the
aprender/realizar/renacer/trueno/trueno-gpu chain documented in
CLAUDE.md.

**Parallel dispatch state (2026-04-18 post-v2.17.0 — preserved for audit
trail):** three lanes ran concurrently against non-overlapping surfaces —
(a) task #102 contract backfill (content-authoring, contracts/*.yaml),
(b) task #103 MODEL-2 CLI `--min-frequency` plumbing (swap apr-cli
tokenize call-site from aprender-core BPE to aprender-train BPE;
closes v2.15.0 §1 "Known gap" — 0.5 day), (c) task #104 gx10
third-party framework capacity gate (llama.cpp on teacher GGUF,
enforces Zero-Tolerance §3 row #8). Tasks #105 (370M pretraining
loop wiring per training-loop-pretrain-v1 GATE-TRAIN-005) remains
the long-pole item awaiting #102+#103 closure. Compute pool utilization
is deliberately heterogeneous: lambda-labs (x86_64 RTX 4090) does
contract+code surgery, gx10 (aarch64 GB10 Blackwell) does remote
bench, yoga (x86_64 RTX 4060 Laptop) stays idle pending apr 0.31.0
upgrade per Zero-Tolerance §3 row #8. Jetson remains blocked per
`project_ship_two_001_jetson_blocked.md`.

**v2.16.0 amendment (2026-04-18):** Codified **Zero-Tolerance** as §3 row
#8. The operationalization, verbatim: "We never accept bugs or poor
performance. Defects and perf regressions are both blockers, not trade-
offs. All work improves or holds the line; never degrades it. No 'pre-
existing' carve-outs. No `#[ignore]` as a release valve." Why now: the
SHIP-TWO-001 compute-pool reality (lambda-labs x86_64 RTX 4090 +
yoga x86_64 RTX 4090 Laptop + gx10 aarch64 GB10 Blackwell + jetson
aarch64) surfaces cases where it is tempting to accept a regression
("gx10 is Blackwell — 15.5 tok/s fused NF4 is fine") or a bug ("yoga's
apr 0.4.11 is stale but it works for small models"). The Zero-Tolerance
principle writes the refusal explicitly: when a host drops to a slower
path OR runs stale software, that is a blocker on the host, not a
baseline to ship against. Concrete application to in-flight work:
(a) yoga stays blocked from SHIP-TWO-001 eval dispatch until apr
binary is upgraded to 0.31.0 AND cuBLAS smoke passes (no "it works
with the old binary" ship), (b) gx10 must run a non-fused third-party
framework (llama.cpp / PyTorch nightly cu128 / vllm) at ≥ reference
tok/s before counting as GPU capacity for MODEL-2 parity training —
the 15.5 tok/s fused fallback is FORBIDDEN as a steady-state (see
`project_pmat_587_*` memos for prior perf discipline). Ties to the
existing Toyota Way feedback memory: "all defects are your defects;
never 'pre-existing'" now extends to performance.

**v2.15.0 amendment (2026-04-18):** MODEL-2 pretraining scaffold **LANDED**
on `feat/pm-007-preflight-poka-yoke`. Three commits close the three
P0 blockers identified in the v2.14.0 readiness audit:

1. **Task #89 — BPE NFC patch SHIPPED (commit `b0e0a280b`):**
   `crates/aprender-train/src/tokenizer/{config,bpe}.rs` now enforce
   C-TOK-BPE-001 INV-TOK-003 (NFC before optional lowercase).
   `TokenizerConfig::normalization` defaults to `None` (`#[serde(default)]`
   for backward compat); set `Normalization::NFC` via
   `.with_normalization()` builder. Two falsification tests locked:
   (a) `test_bpe_nfc_composed_decomposed_parity` — composed `café`
   U+00E9 and decomposed `cafe\u{0301}` encode to identical token IDs
   under NFC; (b) `test_bpe_without_nfc_composed_decomposed_diverge` —
   live falsification witness: without NFC the two forms MUST diverge.
   If the witness test starts passing under `Normalization::None`, the
   invariant is no longer load-bearing and the contract should be
   revisited. `preprocess()` doc-comment records **why NFC before
   lowercase**: `char::to_lowercase()` is not closed over non-NFC input
   for every grapheme — normalizing first keeps the pipeline
   deterministic for composed/decomposed variants.

2. **Task #90 — `apr tokenize train` subcommand SHIPPED (commit
   `512ea51a6`):** new `TokenizeCommands::Train { corpus, vocab_size,
   min_frequency, output, normalization }` variant. Walks `.jsonl`
   files (file or directory), extracts `content` field per line,
   applies NFC via `unicode-normalization::UnicodeNormalization::nfc`
   when `--normalization nfc` (default), calls the BPE trainer, emits
   `vocab.json` + `merges.txt`. `--json` mode round-trips all
   parameters. 3 unit tests pass (happy-path JSONL, directory walk,
   unknown-normalization rejection). **Known gap** (follow-up, NOT a
   ship blocker): `--min-frequency` is accepted for contract parity
   but NOT threaded through — the CLI currently calls
   `aprender::text::tokenize::BpeTokenizer::train(corpus, vocab_size)`
   (aprender-core) which has no public `min_frequency` parameter.
   Strategic fix: switch the CLI to
   `aprender-train::tokenizer::BPETokenizer` (which both honors
   `with_min_frequency()` AND has the NFC plumbing task #89 added).
   Documented in memory `project_ship_two_001_nfc_bpe_patch.md`.

3. **Task #91 — `apr-corpus-ingest` binary SHIPPED (commit
   `512ea51a6`):** new `crates/apr-cli/src/bin/apr-corpus-ingest.rs`
   (+517 LOC) with `plan` and `validate-contract` subcommands over
   `C-DATA-THESTACK-PYTHON` v1.0.0. `plan` reads the contract,
   asserts the 6 required top-level keys (source, license_whitelist,
   pii_scrub, deduplication, split, budget), validates 7
   `INV-DATA-*` + 5 `FALSIFY-DATA-*` + 5 `GATE-DATA-*` prefixes, and
   emits `./output/dry-run-manifest.yaml` with TODO placeholders + UTC
   timestamp. `validate-contract` is exit-code-only. **Hard constraints
   honored:** NO network, NO writes outside `./output/`, deps limited
   to workspace `serde`/`serde_yaml`/`anyhow`/`clap`. Does NOT touch
   `aprender-train/` or `aprender-core/`. 2 unit tests pass.

**MODEL-2 training readiness estimate (post-v2.15.0):** 4 contracts +
3 scaffolding commits shipped. Remaining work to first pretraining
loss curve:
- Thread `--min-frequency` through CLI (switch call to aprender-train
  BPE) — 0.5 day, follow-up ticket.
- Actual corpus download + validated ingest into train/val split
  honoring the 6 C-DATA-THESTACK-PYTHON gates (MinHash-LSH dedup,
  PII scrub, license whitelist, deterministic hash-by-sha256 split,
  corpus_sha256 merkle gate yoga vs gx10) — 2-3 days.
- 370M Llama architecture implementation + pretraining loop wiring
  honoring `training-loop-pretrain-v1.yaml` GATE-TRAIN-005 (val_loss
  divergence abort) — 5-7 days.
- First pretraining smoke run on gx10 — 1-2 days.

**Total: ~10-14 days to first loss curve** (revised up from
v2.14.0's 5-7d estimate now that the scaffold is concrete and the
370M arch implementation is clearly the gating path). Post-v2.15.0,
MODEL-2 moves from contract+scaffold into execution.

**v2.14.0 amendment (2026-04-18):** MODEL-2 pretraining readiness audit
closed two gaps in contract + impl surface:

1. **Dataset contract drafted:** `contracts/dataset-thestack-python-v1.yaml`
   (C-DATA-THESTACK-PYTHON v1.0.0 PROPOSED). 7 invariants + 5
   falsification tests + 5 compound gates covering (a) upstream
   revision pin + raw_tar_sha256 reproducibility, (b) permissive-
   license whitelist (Apache/MIT/BSD/ISC/Unlicense/CC0/0BSD) with
   unknown→reject policy, (c) PII scrub (AWS/PEM/GH PAT/Slack/Google),
   (d) MinHash-LSH near-duplicate removal (seed=42, Jaccard ≥0.85 →
   drop), (e) deterministic hash-by-file-sha256 split (train=0.98,
   val=0.02, assertion: same seed → byte-identical split across
   hosts), (f) corpus_sha256 merkle-style parity gate (FALSIFY-DATA-003
   yoga vs gx10), (g) UTF-8 + NFC round-trip encoding hygiene
   (INV-DATA-007). Closes the P0 blocker identified by the 2026-04-18
   MODEL-2 training-readiness audit: `training-loop-pretrain-v1.yaml`
   line 22 referenced this peer contract, but the file did not exist.

2. **BPE NFC gap identified (IMPLEMENTATION BLOCKER):** The BPE
   tokenizer at `crates/aprender-train/src/tokenizer/bpe.rs` does NOT
   implement NFC normalization, despite `contracts/tokenizer-bpe-v1.yaml`
   INV-TOK-003 / `dataset-thestack-python-v1.yaml` INV-DATA-007
   requiring it. No HF `tokenizers` dep to defer to.
   `TokenizerConfig`/`BpeConfig` have no normalizer field. Fix surface:
   (a) add `normalization: Option<Normalization>` to BpeConfig, (b)
   apply `unicode_normalization::nfc()` at `encode()` entry, (c) add a
   round-trip property test on `café` (composed vs decomposed) + emoji.
   Without this, MODEL-2 tokenizer will drift between train-time and
   inference-time on non-ASCII code and GATE-DATA-005 will ship-block.

**MODEL-2 training readiness estimate (post-v2.14.0):** contract surface
is complete (4 contracts: llama arch, BPE tokenizer, pretrain loop,
dataset). Remaining code work: BPE NFC patch (~1 day), tokenizer
trainer CLI wiring (~3 days), corpus ingest harness honoring the
dataset contract (~2 days). **5-7 days to first pretraining run**
modulo Blackwell JIT warm-up and corpus download time.

**v2.13.0 amendment (2026-04-18):** FALSIFY-SHARD-003 DISCHARGED. Live
probe run yoga (RTX 4090, x86_64) vs gx10 (GB10 aarch64) on the released
teacher GGUF (`paiml/qwen2.5-coder-7b-apache-q4k-v1`, sha
`e6cac5d6…7981`) returned **16/16 byte-identical completions** on
HumanEval/0..15 at temperature=0.0, top-k=1, max_tokens=512. Evidence:
`evidence/ship-two-001/shard-003-determinism/probe_20260418_143041.json`.
Contract `contracts/eval-sharding-v1.yaml` bumped 1.0.0 → 1.1.0 and
flipped **DRAFT → ACTIVE**; `discharged:` block recorded on
FALSIFY-SHARD-003 mirroring the SHARD-004 pattern. Combined with the
prior SHARD-004 discharge (Δ=0.0039 pp merged-score identity), both
correctness gates for AC-EX-007 are green. The parallel eval-shard lane
(yoga+gx10) is now a legitimate accelerator for any future SHIP-TWO-001
re-audit that respects the contract prerequisites (temp=0.0, top-k=1).
Task #79 closed.


**v2.12.0 amendment (2026-04-18):** Post-ship artifacts landed (commit
`cc52e7bfc`) while the teacher is live on HF. All of these are
**out-of-scope for the current ship** but advance the next-wave deliverables:

1. **MODEL-2 Phase 1-B contracts** (task #81) — three new YAMLs:
   - `contracts/model-families/llama-370m-sovereign-v1.yaml` (9 invariants,
     4 gates, sovereign 370M arch with frozen intermediate_dim=2816)
   - `contracts/tokenizer-bpe-v1.yaml` (7 inv, 7 gates; vocab bounds,
     special tokens, byte-exact round-trip, NFC normalization)
   - `contracts/training-loop-pretrain-v1.yaml` (8 inv, 8 gates;
     GATE-TRAIN-005 ship-blocking: `val_loss[N] > 2.0 × val_loss[N-1]`
     → ABORT — encodes the MODEL-1 v2 divergence lesson)
2. **MODEL-1 QLoRA retry plan** (task #86) —
   `docs/specifications/aprender-train/model-1-qlora-retry-plan.md`,
   6 falsification gates, hyperparameter deltas from v2 (LR 2e-4 →
   5e-5, rank 16 → 32, temperature 4.0 → 2.0).
3. **FALSIFY-SHARD-003 determinism probe** (task #88) —
   `scripts/ship-two-001/eval-shard-determinism-probe.sh` (239 lines).
   Closes the one blocking gap for AC-EX-007 found by the eval-shard
   audit (contract `eval-sharding-v1.yaml` line 151 referenced a script
   that did not exist). DRY_RUN=1 validates the JSONL builder without
   dispatch. Full `--hosts yoga,gx10 --model <gguf> --probe-tasks 0-15`
   run requires teacher GGUF pre-cached on both hosts.

**Compute-pool reality check (2026-04-18):** yoga RTX 4090 + gx10 GB10
aarch64 are today's effective parallel pool. Jetson remains blocked by
the 5 blockers documented in memory `project_ship_two_001_jetson_blocked.md`.
Lambda-labs is referenced in spec docs but **not provisioned** — no SSH
alias, no memory file, no credentials surfaced; treat as aspirational
until provisioning is in place.

**v2.11.0 amendment (2026-04-18):** SHIP-TWO-001-MODEL-1-TEACHER **RELEASED**.
EX-05, EX-06, EX-07 all DISCHARGED on the teacher artifact (`paiml/qwen2.5-coder-7b-apache-q4k-v1`):

1. **EX-05 verify-manifest (live, 3 formats)**:
   `apr validate-manifest <m> --live --json` PASS for `.apr` (8.0 GiB, sha
   `0a854098…c73666`), `.safetensors` (15.2 GiB, sha `c1058ce7…d8954`),
   `.gguf` (7.5 GiB, sha `e6cac5d6…7981`). All five gates fire green:
   PM-001 (schema), PM-003 (HEAD content-length), PM-002-live
   (streaming sha256), PM-004 (SPDX), PM-005 (recipe_sha256), PM-006
   (parent chain). Evidence:
   `evidence/ship-two-001/ex-05-manifest-verify-*.json` (3 per-format +
   1 summary).

2. **EX-06 apr pull + re-inference**: `apr pull
   paiml/qwen2.5-coder-7b-apache-q4k-v1` → cached GGUF at
   `~/.cache/pacha/models/7bcabb852fedb36b.gguf`; sha256 of pulled file
   exactly matches the declared GGUF manifest sha (harness v3 auto-
   detects pulled format from file extension, fixes v2 bug that hard-
   coded the APR manifest and produced a spurious format-mismatch FAIL);
   `apr run <pulled> --prompt 'def fib(n):' --max-tokens 64 --temp 0 --top-k 1`
   produces output whose longest parseable prefix contains ≥1 non-trivial
   Python statement (spec §12.3 AC-EX-006 literal: "emits syntactically
   valid Python"). Both **AC-EX-005 (sha256 roundtrip)** and **AC-EX-006
   (Python validity)** PASS. Evidence:
   `evidence/ship-two-001/ex-06-pull-rerun.json` → `overall: PASS`.

3. **EX-07 tag release**: Git tag `SHIP-TWO-001-MODEL-1-TEACHER` created
   at HEAD of the ship branch; announcement blurb embedded in this
   amendment. The teacher artifact is live on HF Hub at
   https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1 (3 formats),
   and downloadable via `apr pull paiml/qwen2.5-coder-7b-apache-q4k-v1`.

**Announcement (v2.11.0):** Aprender ships its first sovereign model:
Qwen2.5-Coder-7B-Instruct Q4_K (Apache-2.0), 85.98% HumanEval pass@1
(141/164, confirmed via `apr eval --benchmark humaneval` on 2026-03-28),
8.0 GiB APR / 7.5 GiB GGUF / 15.2 GiB SafeTensors. Runs end-to-end on
`apr run` / `apr serve`. MODEL-1 v2 (distilled student) is falsified
at the adapter (non-converged QLoRA, task #86 holds the retry plan);
MODEL-2 (albor sovereign) follows in a separate ship per spec §12.4.

**v2.10.0 amendment (2026-04-18):** MODEL-1 v2 root cause is **DEFINITIVE**:
non-converged QLoRA adapter. Deep-probe sub-agent (memory:
`project_ship_two_001_model1_qlora_divergence.md`) found the smoking gun in
`instruct-qlora-7b/best/metadata.json` — `train_loss=15.41`,
`val_loss=31.99`, `train_perplexity=1e6`, `val_perplexity=1e6`,
`epoch=0` (of planned 3). The `best/` and `epoch-0/` adapter safetensors
are byte-identical; training halted at epoch 0 with both losses
diverging and perplexity saturated at the 1M cap. Merging this
non-converged adapter into Qwen2.5-Coder-7B produced the mode-collapsed
`ylkoylkoylko…` output observed by AC-SHIP1-005. **Hypotheses all
FALSIFIED**: tokenizer (embedded BPE loads cleanly, `embed_tokens`
byte-identical to teacher), tensor layout (`apr qa` Tensor Contract PASS,
339 tensors pass PMAT-235), quantization (Q4K lm_head stats match
teacher f32 within quant noise). Probable failure mode: LR=2e-4 too
hot for rank-16 actual (recipe specified rank=32) × soft-label
temperature=4.0. **Ship decision**: TEACHER-ONLY
(`qwen2.5-coder-7b-instruct-q4k.apr`, 85.98% pass@1 confirmed via
`/home/noah/src/apr-leaderboard/results/humaneval_20260328_121327.json`
— 141/164 pass). AC-SHIP1-005 (distilled student ≥30% HumanEval)
blocked by MODEL-1 retry (task #86, out of scope for current ship).
EX-05/06/07 proceeds with teacher artifacts only. Reduced-gate ship per
§ Failure Protocol (Hansei).

**v2.9.0 amendment (2026-04-18):** EX-04 **DISCHARGED**. Two falsifications
of the v2.8.1 code motivated two fixes:
1. v1.1.2 — `upload_via_xet` was early-returning on the Xet branch,
   skipping the LFS-pointer commit entirely (bytes in CAS but invisible
   on repo tree). Evidence: `evidence/ship-two-001/ex-04-xet-clobber-falsification.json`.
2. v1.1.3 — after the v1.1.2 fix, `commit_lfs_pointer` was using
   `application/json` with an `{operations:[{op:addOrUpdate,...}]}`
   schema that HF Hub accepts with HTTP 200 + `success:true` but
   silently no-ops (produces empty commits identical to parent tree).
   Evidence: `evidence/ship-two-001/ex-04-xet-postfix-still-falsified.json`.
   Fix: NDJSON body (newline-delimited JSON) with `Content-Type: application/x-ndjson`
   and `{"key":"header",...}` + `{"key":"lfsFile","value":{...}}` line
   schema. New gate **FALSIFY-PUB-LFS-011** (NDJSON schema) with source-
   invariant test `commit_lfs_pointer_uses_ndjson_lfsFile_schema`.
   Live discharge: `evidence/ship-two-001/ex-04-xet-postfix-v1.1.3-discharged.json`
   — all three formats (8.0 GiB .apr, 8.0 GiB .gguf, 15.2 GiB .safetensors)
   now present on `/tree/main` with sha256 oids matching staging; GGUF
   idempotent re-upload completed in 16.9s (CAS cache-hit). Contract
   `contracts/apr-publish-hf-large-file-v1.yaml` bumped to v1.1.3,
   `status` → `DISCHARGED`. **FALSIFY-PUB-LFS-009/010/011** all DISCHARGED.
   Next: EX-05 (verify-manifest live), EX-06 (apr pull + re-inference),
   EX-07 (tag release SHIP-TWO-001-MODEL-1-TEACHER).

**v2.8.1 amendment (2026-04-18):** Phase 2 of F-PUB-LFS-001 shipped in
commit `18fd9536e` (PR #882). The `xet` sub-feature wires `hf-xet`
1.5.1 (HF's Apache-2.0 reference impl) into `apr publish`. The
`reject_oversized_file` hard-abort is deleted; files > 5 GiB now
dispatch through `crates/aprender-core/src/hf_hub/xet.rs::XetUploader`,
which uses the `hf-xet` blocking API (`XetSessionBuilder` → token-
refresh URL → `upload_from_path_blocking` → `commit_blocking`). The
client-side surface is 178 lines because phases 3–7 of the Xet
protocol (chunking, dedup, xorb/shard CAS upload, hash encoding) are
delegated wholesale to the reference impl. **FALSIFY-PUB-LFS-001**
(file-size dispatch) and **-002** (token-refresh URL shape) are
deterministically discharged by 4 unit tests; **-003..-009** are
inherited from `hf-xet`; **-010** (three-format dogfood) still
pending HF_TOKEN in the ship environment. Contract
`contracts/apr-publish-hf-large-file-v1.yaml` bumped to v1.1.0 with
status `IMPLEMENTED`.

**v2.8.0 amendment (2026-04-18):** EX-04 discovered that `apr publish`
aborts on every SHIP-TWO-001 teacher artifact because all three formats
exceed the 5 GiB HTTP preupload threshold (.apr 8.0 GiB / .gguf 8.0 GiB /
.safetensors 15.2 GiB). The fix is NOT sharding (workaround) and NOT a
self-hosted S3 mirror (not sovereign — AWS-dependent). The fix is to
implement HF Hub's actual current large-file protocol: **Xet**
(huggingface.co/docs/xet/index v1.0.0, reference Rust impl
github.com/huggingface/xet-core Apache-2.0 v1.4.3). New contract
`contracts/apr-publish-hf-large-file-v1.yaml` v1.0.0 codifies the
10-gate falsification set **FALSIFY-PUB-LFS-001..010** (file-size
dispatch, token acquisition, chunk/xorb invariants, shard ordering,
idempotency, retry policy, hash-string encoding, LFS pointer commit,
three-format dogfood). See §12.8 for the full protocol amendment.

**v2.7.0 amendment (2026-04-18):** the pre-flight gate set grows to nine.
**FALSIFY-PM-009** (APR magic-bytes Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.3.0) closes the three-format ship symmetry
— every shipped format (`.safetensors`, `.gguf`, `.apr`) now has a
pre-flight gate that aborts BEFORE any network I/O when the staged file
disagrees with the manifest. v1.0 scope for PM-009 is magic-bytes only:
first 4 bytes must be one of `APR\0`, `APRN`, `APR1`, `APR2`. The exact
class it catches is "wrong file staged under format=apr manifest" (e.g.
a GGUF renamed `.apr`, or a stray `.safetensors`). Tensor-index quant
validation deferred to v1.1. 45 unit tests on every push; real-artifact
dogfood evidence in §12.7.

**v2.6.0 amendment (2026-04-18):** the pre-flight gate set grew from seven
to eight. **FALSIFY-PM-008** (GGUF tensor-type Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.2.1) closes the same ship-blocker class as
PM-007 but for the `.gguf` format. Evidence surfaced during the discharge
run that `general.file_type` is advisory: our own 8 GiB teacher GGUF ships
with stale `file_type = 0` (ALL_F32) despite fully Q4_K tensors, so PM-008
treats the **predominant GGML tensor type** as authoritative and the
metadata field as a fallback. Real-artifact verification at
`evidence/ship-two-001/ex-04-preflight-gate-smoketest.json`.

**v2.5.0 amendment (2026-04-18):** all seven ship manifest gates (PM-001..007)
now run inside `scripts/ship-two-001/ex-04-upload-hf.sh` as a pre-flight
Poka-Yoke. Any manifest-vs-artifact divergence aborts with non-zero exit
BEFORE any network I/O (contract `apr-cli-publish-extra-v1.yaml` v1.2.0,
`publish-manifest-v1.yaml` v1.1.0). Local validation shows all three ship
artifacts (`.apr`, `.safetensors`-fp16, `.gguf`) PASS every gate — the ship
is unblocked on `HF_TOKEN` alone.

---

## 1. Abstract

This specification defines the contract-first, falsification-driven plan to ship **production models**
through the aprender monorepo, proving end-to-end sovereignty (training → format → inference → eval) of the
Sovereign AI Stack.

**v2.0.0 scope change:** the original distilled-student artifact failed the 2026-04-17 contract-first
audit (see §1.5). The spec now pivots to an **expedited teacher-first ship** (see §12) while defering
distillation and sovereign training to follow-on releases. Either artifact reaching SHIP status
falsifies the null hypothesis "the stack cannot produce shippable weights"; the teacher-first ship
alone satisfies that falsification.

Original (v1.0.0) scope, retained for reference:
1. **MODEL-1 (apr-leaderboard):** A distilled Qwen2.5-Coder-7B student targeting **87.20% HumanEval pass@1**,
   shippable in **~36 engineering hours** from the current trained checkpoint.
2. **MODEL-2 (albor):** A sovereign, from-scratch **370M Python code-completion model** targeting **≥30%
   HumanEval pass@1**, shippable in **3–4 weeks** of compute + engineering.

All shipped artifacts must load via `apr run` (realizar backend), pass `apr qa` Golden Output gates, and
carry a contract-conforming manifest (`contracts/publish-manifest-v1.yaml`).

---

## 1.5. Audit Findings (2026-04-17)

**Verdict:** v1.0.0 MODEL-1 SHIP PATH IS BLOCKED. AC-SHIP1-005 falsified; teacher-first pivot in progress.

### 1.5.1 What was audited

Under contract `F-EVAL-HUMANEVAL-AUDIT-001` (`contracts/eval-harness-humaneval-v1.yaml` v1.1.0):
- Primary student checkpoint: `qwen2.5-coder-7b-distilled-v2-q4k.apr` (5.8 GB, Apr-3)
- Audit tool: `apr qa` + partial `apr eval --benchmark humaneval` via apr-leaderboard harness
- Binary: `/mnt/nvme-raid0/targets/aprender/release/apr` 0.31.0 (commit 9217e9c8a), RTX 4090

### 1.5.2 What we found

| Gate                 | Result            | Measured                                                           |
|----------------------|-------------------|--------------------------------------------------------------------|
| Capability Match     | ✓ PASS            | —                                                                  |
| Tensor Contract      | ✓ PASS            | 339 tensors pass PMAT-235                                          |
| Metadata Plausibility| ✓ PASS            | arch=qwen2, rope_theta=1000000                                     |
| **Golden Output**    | **✗ FAIL**        | For "2+2=" expected "4"; got `xxx9,x,x,,,,,,,,,,,,,999`            |
| Throughput           | ✓ PASS            | 9.7 tok/s (threshold=1)                                            |
| HumanEval pass@1     | **~0 (inferred)** | Batch output was incoherent BPE ("uardsylkoylkoiaÅĤ...") on 2/164  |
| Teacher pass@1       | 85.98 (prior run) | `results/humaneval_20260328_121327.json` — pipeline is sound       |

**The distilled student cannot generate coherent text.** Its tensors are structurally valid (all pass
shape, dtype, and non-finite checks) but its weights do not represent a working model.

### 1.5.3 Five-Whys (recorded in contract `validation_result_v1_1`)

1. **Why did the audit fail?** Student emits garbage BPE sequences on every prompt.
2. **Why is output garbage if weights load?** Tensor Contract validates structure, not semantics —
   weights with legal dtype+shape+finite values can still be a broken model.
3. **Why might weights be broken?** Three candidates, in decreasing likelihood:
   (a) distillation diverged and the run was saved without a sanity gate;
   (b) `apr convert --quantize q4_k_m` introduced a LAYOUT-001-class transpose bug;
   (c) BPE tokenizer / chat-template drift so generation samples from wrong token space.
4. **Why can't we tell which?** Diagnostics (`apr diff`, merged-checkpoint run, tokenizer round-trip)
   were not required gates — they remain diagnostic follow-ups in the contract.
5. **Why did no earlier gate catch this?** `apr qa` Tensor Contract exits PASS before Golden Output
   runs; Golden Output failure does NOT block publish in the current gate matrix. This is the
   root contract gap, now promoted to the expedited plan's first action (§12.1).

### 1.5.4 Notable gap surfaced

The 87.20% figure traces back to recipe-h-32b-distill.yaml's comment labelling the *base 7B-Instruct
few-shot* HumanEval — not a distilled-student zero-shot run. No `apr eval` result file for a
distilled student exists in `apr-leaderboard/results/`; all 17 archived HumanEval runs measure the
teacher. The headline number in v1.0.0 §4.1 was therefore never a reproducible claim.

---

## 2. Motivation

### 2.1 Why These Two Models

| Criterion                    | MODEL-1 (apr-leaderboard) | MODEL-2 (albor)        |
|------------------------------|---------------------------|------------------------|
| Current state                | trained, needs packaging  | architecture designed, pretraining required |
| Engineering distance to SHIP | 36 h                      | 3–4 weeks              |
| Proves distillation path     | yes                       | no                     |
| Proves sovereign path        | partial (uses HF teacher) | **yes (end-to-end)**   |
| Proves eval harness          | yes (HumanEval)           | yes (HumanEval)        |
| Risk profile                 | LOW                       | MEDIUM-HIGH (training) |

Shipping both gives orthogonal proof: one demonstrates the stack can finish what PyTorch started;
the other demonstrates the stack can start AND finish without PyTorch in the loop.

### 2.2 Explicit Non-Goals (v1)

- Not shipping: `entrenar-rl` (POC), `entrenar-rlhf` (POC), `verificar-agent` (research).
- Not targeting: chat tuning, multimodal, tool use, >10B params.
- Not blocking on: full leaderboard automation (post-SHIP), wandb integration, distributed training.

---

## 3. Design Principles

| #  | Principle                   | Operationalization                                                    |
|----|-----------------------------|-----------------------------------------------------------------------|
| 1  | Contract-first              | Every weight file, config, and eval path has a YAML contract BEFORE code |
| 2  | Falsification-driven        | Every acceptance criterion has a named, executable FALSIFY-* test     |
| 3  | Sovereign                   | No PyTorch in the production path; GGUF/APR/SafeTensors only          |
| 4  | Lean on existing artifacts  | Reuse `contracts/model-families/qwen2.yaml` and `llama.yaml` — do not fork |
| 5  | Dogfood tooling             | `apr qa`, `apr bench`, `apr trace`, `apr eval` — never bespoke scripts |
| 6  | Binary gates                | Every GATE-SHIP-* is pass/fail; no partial credit                     |
| 7  | Five-Whys on failure        | Any FALSIFY-* failure triggers documented Hansei (§10) before retry   |
| 8  | Zero tolerance              | We never accept bugs or poor performance. Defects and perf regressions are both blockers, not trade-offs. All work improves or holds the line; never degrades it. No "pre-existing" carve-outs. No `#[ignore]` as a release valve. |

---

## 6. Compound Ship Gates

All gates are binary; any failure blocks publish.

| Gate             | Description                                                    | Blocks          |
|------------------|----------------------------------------------------------------|-----------------|
| GATE-SHIP-001    | MODEL-1: all 10 AC-SHIP1-* PASS **(PARTIAL_ALGORITHM_LEVEL v2.42.0)**          | MODEL-1 publish |
| GATE-SHIP-002    | MODEL-2: all 12 AC-SHIP2-* PASS **(PARTIAL_ALGORITHM_LEVEL v2.42.0)**          | MODEL-2 publish |
| GATE-SHIP-003    | Both models: `apr qa` Golden Output never regresses post-quantize **(PARTIAL_ALGORITHM_LEVEL v2.42.0)** | publish     |
| GATE-SHIP-004    | HumanEval harness produces identical score on two consecutive runs (seed=0) **(PARTIAL_ALGORITHM_LEVEL v2.42.0)** | AC-005, AC-008 |
| GATE-SHIP-005    | License metadata is present AND matches upstream declaration **(PARTIAL_ALGORITHM_LEVEL v2.42.0)**   | publish         |
| GATE-SHIP-006    | GGUF round-trip: APR → GGUF → load in llama.cpp → first-token match (tol ≤ 1e-3) **(PARTIAL_ALGORITHM_LEVEL v2.42.0)** | AC-004, AC-009 |
| GATE-SHIP-007    | No unwrap() in new code (enforced by `.clippy.toml`) **(PARTIAL_ALGORITHM_LEVEL v2.43.0)**          | merge           |
| GATE-SHIP-008    | Contract density: every new public fn has `#[contract]` **(PARTIAL_ALGORITHM_LEVEL v2.43.0)**        | merge           |
| GATE-SHIP-009    | CI green: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --workspace` **(PARTIAL_ALGORITHM_LEVEL v2.43.0)** | merge |
| GATE-SHIP-010    | `cargo deny check advisories` — zero vulnerabilities in weight/tokenizer dependencies **(PARTIAL_ALGORITHM_LEVEL v2.43.0)** | merge |
| GATE-SHIP-011    | PMAT quality score ≥ A- (project), TDG ≥ 90 **(PARTIAL_ALGORITHM_LEVEL v2.43.0)**                    | merge           |
| GATE-SHIP-012    | Coverage ≥ 95% line on new modules (per `.pmat-gates.toml`) **(PARTIAL_ALGORITHM_LEVEL v2.43.0)**    | merge           |

---

## 7. Falsification Tests

Each test is named, executable, and has a defined failure signal.

### 7.1 MODEL-1 Falsification (12 tests)

| ID                 | Test                                                              | Failure Signal                         |
|--------------------|-------------------------------------------------------------------|----------------------------------------|
| FALSIFY-SHIP-001   | `realizar::Model::load_safetensors(path)` returns Ok (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-001` in `contracts/qwen2-e2e-verification-v1.yaml` v1.6.0; `cargo test -p aprender-core --lib format::ship_001`) | Err(_) returned |
| FALSIFY-SHIP-002   | Run `apr run ... --prompt "def fib(n):"`; parse output as Python AST (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-002` in `contracts/qwen2-e2e-verification-v1.yaml` v1.1.0; `cargo test -p aprender-core --lib falsify_ship_002_python_syntax_error_threshold_logic`) | SyntaxError (> 0 errors) |
| FALSIFY-SHIP-003   | Convert then compare per-layer cosine similarity (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-003` in `contracts/qwen2-e2e-verification-v1.yaml` v1.4.0; `cargo test -p aprender-core --lib format::ship_003`) | any layer cos < 0.999 |
| FALSIFY-SHIP-004   | Shell out to `llama-cli` on exported GGUF; prompt → logits (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-004` in `contracts/qwen2-e2e-verification-v1.yaml` v1.5.0; `cargo test -p aprender-core --lib format::ship_004`) | llama.cpp exit ≠ 0 |
| FALSIFY-SHIP-005   | Run HumanEval 164× via `apr eval`; pass@1 computed (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-005` in `contracts/qwen2-e2e-verification-v1.yaml` v1.2.0; `cargo test -p aprender-core --lib falsify_ship_005_humaneval_pass_at_1_threshold_logic`) | pass@1 < 86.00% (or < 84.80% under 1.2 pp noise allowance) |
| FALSIFY-SHIP-006   | `apr qa <model>` exit code = 0 (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QA-SHIP-006` in `contracts/apr-model-qa-v1.yaml` v1.2.0; `cargo test -p aprender-core --lib falsify_ship_006_apr_qa_eight_gates_aggregate`) | any gate reports FAIL |
| FALSIFY-SHIP-007   | `apr bench --iterations 5 --max-tokens 128`; median tok/s (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-007` in `contracts/qwen2-e2e-verification-v1.yaml` v1.3.0; `cargo test -p aprender-core --lib falsify_ship_007_decode_tps_threshold_logic`) | median < 30 |
| FALSIFY-SHIP-008   | Render chat template on canonical system+user; diff vs golden (**PARTIAL_ALGORITHM_LEVEL** — `GATE-CHAT-SHIP-008` in `contracts/chat-template-v1.yaml` v1.1.0; `cargo test -p aprender-core --lib falsify_ship_008_chat_template_render_bind`) | diff ≠ 0 |
| FALSIFY-SHIP-009   | `apr inspect <model>.apr`; grep for `license: apache-2.0` (**PARTIAL_ALGORITHM_LEVEL** — `GATE-APR-PROV-004` in `contracts/apr-provenance-v1.yaml` v1.1.0; `cargo test -p aprender-core --lib provenance`) | missing or mismatched |
| FALSIFY-SHIP-010   | curl + sha256sum against manifest (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-SHIP-010` in `contracts/publish-manifest-v1.yaml` v1.4.0; `cargo test -p aprender-core --lib format::ship_010`) | hash mismatch or 404 |
| FALSIFY-SHIP-023   | Re-run AC-005 on second day; score drift (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-023` in `contracts/qwen2-e2e-verification-v1.yaml` v1.7.0; `cargo test -p aprender-core --lib format::ship_023`) | drift > 1.2 pp |
| FALSIFY-SHIP-024   | Prompt-injection torture suite (50 adversarial inputs) (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-024` in `contracts/qwen2-e2e-verification-v1.yaml` v1.7.0; `cargo test -p aprender-core --lib format::ship_024`) | any panic or NaN in logits |

### 7.2 MODEL-2 Falsification (10 tests)

| ID                 | Test                                                              | Failure Signal                         |
|--------------------|-------------------------------------------------------------------|----------------------------------------|
| FALSIFY-SHIP-011   | `llama.yaml` `370m` entry validates against `_schema.yaml` (**DISCHARGED v2.21.0** — `GATE-ARCH-370M-001` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.1.0 ACTIVE; `cargo test -p aprender-train --lib llama_370m`) | schema error |
| FALSIFY-SHIP-012   | Tokenize 10K docs, detokenize, byte-compare (**PARTIAL_ALGORITHM_LEVEL** — `GATE-BPE-003` in `contracts/tokenizer-bpe-v1.yaml` v1.1.0; `cargo test -p apr-cli --test falsify_ship_012_tokenizer_roundtrip`) | any byte mismatch |
| FALSIFY-SHIP-013   | Training val CE at final step (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-013` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.10.0; `cargo test -p aprender-train --lib ship_013`) | CE > 2.2 |
| FALSIFY-SHIP-014   | Wall-clock from train start to final checkpoint (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-014` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.10.0; `cargo test -p aprender-train --lib ship_014`) | > 21 days |
| FALSIFY-SHIP-015   | Load checkpoint via `apr inspect`; count params (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-003` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.2.0; `cargo test -p aprender-train --lib llama_370m`) | params ≠ 370M ± 1% |
| FALSIFY-SHIP-016   | `apr qa <model>.apr` exit code (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-008` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.9.0; `cargo test -p aprender-train --lib llama_370m`) | any gate FAIL |
| FALSIFY-SHIP-017   | 100 prompts → Python AST parse (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-005` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.6.0; `cargo test -p aprender-train --lib llama_370m`) | ≥2 SyntaxError (tolerate ≤1) |
| FALSIFY-SHIP-018   | `apr eval --benchmark humaneval` pass@1 (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-007` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.8.0; `cargo test -p aprender-train --lib llama_370m`) | < 30.0% |
| FALSIFY-SHIP-019   | GGUF export; first-token probability vs APR (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-004` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.3.0; `cargo test -p aprender-train --lib llama_370m`) | |Δp| > 1e-3 on top-1 |
| FALSIFY-SHIP-020   | `apr bench` median tok/s on RTX 4090 (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-006` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.6.0; `cargo test -p aprender-train --lib llama_370m`) | < 100 |
| FALSIFY-SHIP-021   | Run training 100 steps × 2 with seed=0; diff loss trajectories (**DISCHARGED v2.20.0**) | any step diff > 1e-6 |
| FALSIFY-SHIP-022   | `apr inspect`; check `license`, `data_source`, `data_license` (**DISCHARGED v2.20.0** — `GATE-APR-PROV-001..003` in `contracts/apr-provenance-v1.yaml` v1.0.0 ACTIVE) | any field missing |

---

## 8. Execution Plan

### 8.1 Phase DAG

```
                    ┌─────────────────────────────────────┐
                    │          Phase 0: Scaffold          │
                    │  (contracts, schema, test harness)  │
                    └──────────────────┬──────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │   Phase 1-A (MODEL-1)   │       │   Phase 1-B (MODEL-2)   │
         │  36h — packaging path   │       │  Week 1: tokenizer + dry │
         │                         │       │  run (AC-001,002,011)    │
         │  AC-001..004 load/conv  │       └───────────┬─────────────┘
         └───────────┬─────────────┘                   │
                     ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │  Phase 2-A: eval + qa   │       │  Phase 2-B: pretraining │
         │  AC-005..008            │       │  Weeks 2-3: AC-003,004  │
         └───────────┬─────────────┘       └───────────┬─────────────┘
                     ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │  Phase 3-A: publish     │       │  Phase 3-B: eval + qa   │
         │  AC-009,010  — SHIP-1   │       │  Week 4: AC-005..012    │
         └─────────────────────────┘       └───────────┬─────────────┘
                                                       ▼
                                           ┌─────────────────────────┐
                                           │  Phase 4-B: publish     │
                                           │  SHIP-2                 │
                                           └─────────────────────────┘
```

Phase 1-A and Phase 1-B are independent and run in parallel.

### 8.2 Effort Budget

| Phase | Model    | Effort       | Calendar  | Owner        |
|-------|----------|--------------|-----------|--------------|
| 0     | shared   | 6 h          | day 0     | eng          |
| 1-A   | MODEL-1  | 10 h         | day 1     | eng          |
| 2-A   | MODEL-1  | 12 h         | day 2     | eng          |
| 3-A   | MODEL-1  | 4 h          | day 3     | eng          |
| 1-B   | MODEL-2  | 40 h         | week 1    | eng          |
| 2-B   | MODEL-2  | compute-bound | weeks 2-3 | GPU node    |
| 3-B   | MODEL-2  | 16 h         | week 4    | eng          |
| 4-B   | MODEL-2  | 4 h          | week 4    | eng          |
| **Σ** |          | 92 h + 2 wk  | ~4 weeks  |              |

### 8.3 Integration with `apr run`

After SHIP, both models must satisfy:

```bash
apr run ./model-1.apr --prompt "def quicksort(arr):"       # MODEL-1
apr run ./model-2.apr --prompt "def binary_search(xs, t):" # MODEL-2
```

Both paths resolve through `realizar::Model` (see `crates/aprender-serve/CLAUDE.md` Realizar-First Architecture).
No code path in `aprender-core` may invoke generation.

---

## 9. Risk Matrix

| # | Risk                                           | Probability | Impact | Mitigation                                                           |
|---|------------------------------------------------|-------------|--------|----------------------------------------------------------------------|
| 1 | HumanEval eval non-deterministic               | MED         | HIGH   | seed=0, greedy; GATE-SHIP-004 enforces two-run identity              |
| 2 | GGUF export has tensor-layout bug (LAYOUT-001) | HIGH        | HIGH   | FALSIFY-SHIP-019 parity check; reuse `layout_contract.rs` validator  |
| 3 | MODEL-2 training diverges                      | MED         | HIGH   | AC-SHIP2-003 loss gate; fallback = reduce LR, resume from ckpt       |
| 4 | RTX 4090 insufficient for 370M in 21 days      | LOW         | HIGH   | AC-SHIP2-004 budget; overflow → rent 2× H100 week 3                  |
| 5 | Teacher (Qwen2.5-Coder-7B) license ambiguity   | LOW         | MED    | AC-SHIP1-009; confirm Apache-2.0 in `config.json`                    |
| 6 | `apr convert` quantize drops accuracy > 1pp    | MED         | MED    | FALSIFY-SHIP-003 cos-sim gate; fallback = Q5_K_M                     |
| 7 | Tokenizer round-trip bytes mismatch            | LOW         | HIGH   | FALSIFY-SHIP-012 on 10K corpus                                       |
| 8 | `cargo install aprender` breaks during release | MED         | HIGH   | CI `cargo install --path .` smoke test (GATE-SHIP-009)               |
| 9 | HF artifact hosting outage                     | LOW         | LOW    | dual-publish to HF + self-hosted bucket                              |
| 10| CUDA JIT regression (trueno#200/203)           | MED         | MED    | Pin trueno version per memory note 2026-03-22                        |

---

## 10. Failure Protocol (Hansei)

Any FALSIFY-SHIP-* failure triggers the following sequence. No retry is permitted before completion.

### 10.1 Five Whys

1. What check failed? (name the FALSIFY-*)
2. What invariant did it violate?
3. What code path was responsible?
4. Why was the code path wrong? (bug class: layout, numeric, eval harness, toolchain)
5. Why did no earlier gate catch it? (contract gap → file follow-up)

### 10.2 Decision Gate

| Condition                                              | Action                         |
|--------------------------------------------------------|--------------------------------|
| Single test failed, root cause known, fix ≤ 2 h        | Fix + re-run full gate         |
| Multiple tests failed OR root cause unknown            | Escalate to design review      |
| AC breaks but SHIP is blocking deadline                | Reduced-gate ship (below)      |
| Hardware / compute budget breached                     | Full failure escalation        |

### 10.3 Reduced-Gate Ship (emergency only)

Acceptable only with written Noah Gift approval. A reduced ship may drop:
- AC-SHIP1-007 / AC-SHIP2-010 (bench speed) — ship with "beta performance" label
- AC-SHIP1-010 / AC-SHIP2-012 (artifact publication) — ship to internal bucket only

All 8 `apr qa` gates (Golden Output, layout, tensor stats, license, etc.) MUST still pass.

### 10.4 Full Failure Escalation

If neither model ships within the budget, this specification is void. Retrospective must
answer: (a) which assumption was wrong, (b) what contract should have caught it earlier,
(c) what gets deleted from scope before restart.

---

## 13. Why Did This Take So Long? (Retrospective)

### 13.1 Timeline

- **2026-01-01 → 2026-04-17:** 3.5 months of work on the Sovereign AI Stack.
- 2141 commits to aprender (of which 181 = 8.5% are perf-path-to-1.5× commits).
- apr-leaderboard ran **12 distillation recipes** (a → l); multiple had broken checkpoint output.
- Commit `0fc5436 fix: LoRA merge was element-wise, not matrix multiply — root cause of distilled
  model garbage` on apr-leaderboard — **this exact failure class has been fixed before**.
- Commit `a20f234 docs: document Q4K roundtrip corruption blocker` — **also previously known**.
- Current distilled-v2 checkpoints are dated 2026-04-03; they sat **14 days** before any
  `apr qa`-driven audit.
- Spec SPEC-SHIP-TWO-001 v1.0.0 was written 2026-04-17, *after* the broken checkpoint had been
  sitting for 2 weeks and cited as "trained, needs packaging."

### 13.2 Root causes

1. **Contract came after POC, not before.** The 87.20% number lived in a recipe comment since
   Q1 and was promoted to a spec headline without a falsification test run. Design Principle 1
   (Contract-first) was violated by its own spec.
2. **`apr qa` gate matrix lets Golden Output fail silently.** A model that cannot generate "4"
   for "2+2=" passed `apr qa` overall because Golden Output was reported but non-blocking. Tensor
   Contract PASS became a false confidence signal. This is the structural defect that allowed
   broken weights to persist for 14 days.
3. **Perf work crowded out ship work.** 181 perf commits toward 1.5× Ollama parity between
   January and April; **zero** commits on publishing a *model* artifact (as opposed to a *crate*).
   Perf gains are visible in benchmarks; ship state was not gated anywhere.
4. **Monorepo reorg (APR-MONO) consumed weeks.** Phases 1–11 moved 70 crates, introduced shim
   layers, debugged publishing cycles — necessary ceremony but directly competed with model-ship
   bandwidth. Commits like "Phase 10d+10e done" / "Phase 11 (CI fix + publish babysit)" show
   sustained multi-week focus.
5. **Distillation recipe churn without shipping discipline.** Recipes a/b/c/d/e/f/g/h/i/j/k/l —
   twelve experiments, each generating a checkpoint — but no contract defining what makes a
   recipe's output "ship-quality." Each recipe was treated as a new chance; no recipe was ever
   contractually retired. `contracts/distillation-pipeline-v1.yaml` (per v1.0.0 §4.4) is listed as
   "NEW" but has never existed — the proof of this is that broken checkpoints sat unaudited for weeks.
6. **Two-model scope from the start.** v1.0.0 bundled distilled (quick) with sovereign
   (multi-week). This guaranteed the multi-week item would block the quick item from any
   expedited ship path. The fix is to ship MODEL-1 alone as a v1 and move MODEL-2 to v2.
7. **Tooling investment vs tooling usage.** `apr qa`, `apr trace`, `apr profile`, `apr diff` all
   exist and are well-built. They were not being *dogfooded on the shipping artifact* until
   2026-04-17. The audit that exposed this was a 10-minute `apr qa` invocation.
8. **"87.20%" was never in a results JSON.** 17 HumanEval result files exist in
   `apr-leaderboard/results/`; all 17 are teacher runs. The student has no recorded result. A
   spec claim not traceable to a results file is an unfalsified claim.

### 13.3 Lessons codified as contracts

| Contract (new)                                  | Prevents future occurrence of                                |
|-------------------------------------------------|--------------------------------------------------------------|
| `contracts/eval-harness-humaneval-v1.yaml` v1.1 | Headline pass@1 numbers without a results-JSON trail         |
| `contracts/publish-manifest-v1.yaml` v1.0       | Artifacts shipping without sha256 / license / provenance     |
| `contracts/publish-manifest-v1.yaml` v1.1 (PM-007) | Uploading `.safetensors` whose header dtype contradicts the manifest |
| `contracts/publish-manifest-v1.yaml` v1.2 (PM-008) | Trusting stale `general.file_type` over per-tensor `ggml_type` histogram for GGUF |
| `contracts/publish-manifest-v1.yaml` v1.3 (PM-009) | Uploading a renamed `.gguf`/`.safetensors` under a `.apr` manifest |
| **APR-QA GATE AMENDMENT (§12.1)**               | Tensor Contract masking a Golden Output failure              |
| `contracts/distillation-pipeline-v1.yaml` (TBD) | Recipes run without per-epoch Golden Output gating           |

### 13.4 Publish-fastest rules going forward

1. **No claim without a JSON.** Every pass@1 number in any spec must be a `jq`-extractable field
   in a file under version control. If it isn't, it doesn't exist.
2. **Golden Output is a ship-blocking gate.** `apr qa` must exit non-zero if Golden Output fails,
   even when all structural gates pass.
3. **Ship-first, proof-second.** Ship the teacher in 10 hours; use the published artifact as the
   reference against which distillation is measured. Do not wait to ship because distillation
   isn't done.
4. **One artifact per release.** v1 = teacher. v1.1 = distilled (if/when it works). v2 =
   sovereign. Coupling them couples their risks.
5. **Dogfood before declaring "trained."** A checkpoint is not "trained" until `apr qa` and
   `apr eval` agree it is. Until then it is "saved."

---

## 11. References

### 11.1 Existing Contracts

- `contracts/model-families/qwen2.yaml` — Qwen2 architecture descriptor
- `contracts/model-families/llama.yaml` — LLaMA architecture descriptor
- `contracts/model-families/_schema.yaml` — family schema validator
- `contracts/tensor-layout-v1.yaml` — row-major APR invariant (LAYOUT-001/002)
- `contracts/chat-templates-v1.yaml` — chat-template engine spec
- `contracts/apr-cli-commands-v1.yaml` — 57-command CLI contract
- `contracts/publish-manifest-v1.yaml` v1.3.0 — artifact-shipping schema (sha256, provenance, license) + **FALSIFY-PM-007** safetensors header dtype Poka-Yoke + **FALSIFY-PM-008** GGUF tensor-type Poka-Yoke (tensor-authoritative; `general.file_type` is advisory fallback) + **FALSIFY-PM-009** APR magic-bytes Poka-Yoke (three-format ship symmetry)
- `contracts/apr-cli-publish-extra-v1.yaml` v1.2.0 — **F-PUBLISH-EXTRA-001** (§12.7): manifest consumption, `--extra-file` passthrough, three-format ship, safetensors fp16 dtype, **preflight_validate_manifest** (FALSIFY-PUB-EXTRA-009/-010)
- `contracts/eval-harness-humaneval-v1.yaml` — pass@1 harness / AC-EX-003 floor
- `contracts/apr-model-qa-v1.yaml` — `apr qa` gate matrix / AC-EX-001/-002 (Golden Output hard-block)
- `contracts/training-loop-pretrain-v1.yaml` v1.4.0 — MODEL-2 training loop (GATE-TRAIN-001..010), peer of the new GPU backend contract below
- `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED — **§14 (v2.23.0)** task #132 GPU training backend dispatch (INV-GPUTRAIN-001..007, GATE-GPUTRAIN-001..006, FALSIFY-GPUTRAIN-001..007)

### 11.2 Related Specifications

- `docs/specifications/aprender-train/hugging-face-distill-learn-pipeline-spec.md`
- `docs/specifications/aprender-train/comprehensive-qa-falsification.md`
- `docs/specifications/aprender-train/model-eval-framework-spec.md`
- `docs/specifications/aprender-monorepo-consolidation.md`

### 11.3 External

- HumanEval: Chen et al. 2021, *Evaluating Large Language Models Trained on Code*
- Qwen2.5-Coder: Hui et al. 2024, *Qwen2.5-Coder Technical Report*
- The Stack v2: BigCode, CC-BY-4.0

---

## 18. Training Status Snapshot — Chain of Thought (2026-04-26)

This section walks the reasoning that connects the spec's two-model
goal to the current state. Read it as the deduction chain that future
sessions can re-enter without re-reading every prior section.

### 18.1 Why are we training models at all?

The spec's purpose is **Sovereign AI Stack Proof** — demonstrating
that a Rust-only stack (aprender + apr-cli + realizar in-tree)
can both *package an existing teacher model* AND *pretrain a new
model from scratch* using only stack tooling. Two models because
one alone proves only half the loop:

- **MODEL-1 (Qwen2.5-Coder-7B teacher fork)** proves the
  **packaging + serving** half: load → inspect → quantize → export
  → cross-format round-trip. Existence-proof that we can ship
  somebody else's weights through our format and tooling.
- **MODEL-2 (Llama-370M sovereign)** proves the **pretrain**
  half: tokenize → train → checkpoint → eval → publish. Existence-
  proof that we can produce weights from scratch with no PyTorch
  in the loop.

The 22 acceptance criteria across §3.2 and §5.2 decompose those
two halves into binary, falsifiable gates.

### 18.2 What does "DISCHARGED" mean here, and where are we?

A discharge is one of three levels:

| Level | Meaning |
|-------|---------|
| **PARTIAL_ALGORITHM_LEVEL** | The verdict function exists in Rust + has unit tests, but no live evidence has been captured against the real teacher / dataset. |
| **DISCHARGED** | Live evidence on noah-Lambda-Vector RTX 4090 against the canonical fixtures pins all the verdict's constants. |
| **(unbound)** | No verdict function authored yet. |

Coverage tally as of v2.62.0: **33 PARTIAL + 12 DISCHARGED** across
45 contract-bound levers (5 of those 12 promoted in a single cycle
on 2026-04-25).

### 18.3 MODEL-1 — five fully discharged, five blocked on one bug

| AC | Goal | Status | Evidence |
|----|------|--------|----------|
| 001 | safetensors loads | **DISCHARGED** | 15.23 GB load via `realizar::Model::load_safetensors`, tensor_count=339, total_params=7,615,616,512 |
| 002 | `apr run` produces Python `def fib(n):` syntax | PARTIAL | needs working `apr run` — blocked on §16/§17 bug |
| 003 | per-layer cos ≥ 0.999 | **DISCHARGED** | 339-tensor sweep; min cos = 0.99999994 (6 OOM headroom) |
| 004 | GGUF round-trip via llama-cli | **DISCHARGED** | 8.04 GB Q4K passthrough, llama-cli emits "Hello! How can" at 127.5 tok/s |
| 005 | HumanEval ≥ 86% | PARTIAL | needs `apr eval` on a working APR teacher — blocked on §16 |
| 006 | `apr qa` 8 gates strict pass | PARTIAL | half the gates (ollama_parity / format_parity / ptx_parity) fail because of the SHIP-007 surface manifestation |
| 007 | decode tps ≥ 30 | PARTIAL | this AC literally IS the SHIP-007 root-cause investigation |
| 008 | chat template render | PARTIAL | needs `apr run` |
| 009 | `apr inspect` provenance fields | **DISCHARGED** | live `apr stamp` + `apr inspect` round-trip; 339 tensors preserved post-stamp |
| 010 | `apr validate-manifest --live` | **DISCHARGED** | 31 GB streamed from HF Hub CDN, 3 sha256s byte-identical, 18 gate verdicts PASS |

**Conclusion of 18.3:** The teacher is shippable today via the
GGUF path. Five of the ten ACs are closed by live evidence; the
other five all transitively depend on a single APR-format
inference bug (§17), so resolving that one bug discharges all
five at once.

### 18.4 MODEL-2 — three discharged, nine blocked on convergence

| AC | Goal | Status | Why |
|----|------|--------|-----|
| 011 | architecture in `llama.yaml` 370m | **DISCHARGED** v2.21.0 | Pure-Rust schema validation |
| 012 | tokenizer round-trip 10K docs | PARTIAL | algorithm proven on synthetic 20-doc holdout; needs The Stack v2 Python 10K corpus |
| 013 | val CE ≤ 2.2 | PARTIAL | needs converged 370M model |
| 014 | training within 21 days | PARTIAL | needs converged 370M model |
| 015 | `.apr` checkpoint with 370M params | PARTIAL | algorithm proven; needs converged 370M model |
| 016 | `apr qa` 8 gates pass | PARTIAL | needs converged 370M model |
| 017 | 100 prompts → Python AST | PARTIAL | needs converged 370M model |
| 018 | HumanEval ≥ 30% | PARTIAL | needs converged 370M model |
| 019 | GGUF export round-trip | PARTIAL | needs converged 370M model |
| 020 | `apr bench` ≥ 100 tok/s on RTX 4090 | PARTIAL | needs converged 370M model |
| 021 | seed-fixed reproducibility | **DISCHARGED** v2.20.0 | Two seed=0 runs match within 1e-6 |
| 022 | provenance fields published | **DISCHARGED** v2.20.0 | `apr inspect` shows license / data_source / data_license |

**Conclusion of 18.4:** Nine of twelve ACs are gated on a single
upstream event — a successful 370M convergence run. Three of those
nine (012/015 algorithm-only) could in principle discharge on
synthetic data, but the chain is currently bottlenecked at the
convergence step.

### 18.5 What's blocking the convergence run?

Walking the chain backward from "we have a trained 370M `.apr`":

1. **Convergence requires a real corpus** — synthetic was used for
   the 1300-step EARLY_STOP smoke (task #137 ✅) and the 10K-step
   re-run that confirmed the corpus bottleneck (task #138 ✅).
2. **Real corpus requires The Stack v2 Python pretokenized into
   `.bin` shards** with vocab=50,257. Today
   `apr tokenize encode-corpus` exists (task #123 ✅) and the
   `ShardBatchIter` reads `.bin` (u32 LE) format from a directory.
   The Stack v2 download + filter + tokenize is the single missing
   data-engineering step.
3. **Tokenizer is unblocked** — BPE quadratic-time was fixed
   (task #118 ✅): real MODEL-2 corpus now tokenizes in 51 min,
   was 25h+ non-completing. vocab=50,257 settled via Option A
   bump (task #131 ✅, replaces albor's 50,000).
4. **Training compute is the real risk** — `apr pretrain --device
   cuda` is **NOT functional today** (task #132). `apr pretrain`'s
   `TransformerTrainer::new` lacks a `Device` parameter, so real-
   compute training is CPU-only. 370M × CPU is impractical for full
   convergence. **This is the single critical-path code change
   between us and a converged MODEL-2.**

The GPUTRAIN suite itself (7/7 DISCHARGED, including same-device
seed reproducibility via cuBLAS PEDANTIC_MATH + atom-free PTX
reduction) proves the kernels work — they just aren't wired into
the `apr pretrain` entry point yet.

### 18.6 The SHIP-007 narrowing — chain of deductions

The bug that blocks all 5 remaining MODEL-1 PARTIALs has been
narrowed step by step over 4 spec amendments. Each step is a
falsifiable result, not a hypothesis:

```
Premise    : APR teacher's GPU forward path emits cos=−0.005 vs CPU
            (argmax 334 vs 8127) on the canonical 7B Qwen2.5-Coder
            (§15 — recorded 5 Whys + tensor-shape evidence)
              │
              ▼
Hypothesis : Bug is in GQA-7:1 attention kernel transpose/stride
            (§15 candidate)
              │
              ▼ ran 3 CPU/GPU GQA parity tests on canonical 28:4:128:3584
              │
              ▼
Result §15.4 : Attention kernel ELIMINATED — 3/3 tests PASS on
              the canonical shape (PR #1061)
              │
              ▼
Hypothesis : Bug is somewhere in the GPU forward path outside
            the attention kernel — Q/K/V projection, RMSNorm,
            FFN, LM head, multi-layer KV cache
              │
              ▼ ran apr trace --payload twice on the SAME teacher
              │   in BOTH formats, on CPU
              │
              ▼
Result §16 : APR-format CPU forward path → " " (token 220)
            GGUF-format CPU forward path → " 2+2 is 4."
            Both ran on CPU. GPU stack is ELIMINATED entirely.
            (PR #1063, spec v2.61.0)
              │
              ▼
Hypothesis : Bug is in the APR-format-specific layer-composition
            glue, NOT in the kernel arithmetic that's ruled out
            twice (once on GPU by §15.4, once on CPU by SHIP-003
            PR #1059's 339-tensor cosine sweep cos≥0.9999999)
              │
              ▼ examined the 28-layer per-layer ffn_out std on the
              │   APR side
              │
              ▼
Result §17 : Layer 3 ffn_out std=11.46 vs layer 2 std=0.22 — a
            53× spike that damps in 1 layer (one-off perturbation,
            not stable architectural feature). Mean shift -0.082
            is 100× median (sign-bias signature). Bug surface
            narrowed from 28×4 candidates to (layer=3, sub-block=FFN).
            (PR #1064, spec v2.62.0)
              │
              ▼
Next step  : Sub-FFN bisection — emit gate_proj_out, up_proj_out,
            silu(gate), silu(gate)*up, down_proj_out for layer 3.
            Whichever first shows the discontinuity is the bug site.
            Requires the §15.5 TraceStep extension — load-bearing.
            (PR #1065 contract envelope; PR #1066 implementation
            in flight 2026-04-26)
```

### 18.7 What "knowing" looks like at each step

Each premise above is *falsifiable*: someone reading the trace
output, the parity test, or the cosine sweep can re-derive the
conclusion. No deduction depends on private state. This is the
**Genchi Genbutsu** ("go and see") discipline the spec mandates
in §3 row #10 and `feedback_apr_trace_not_eprintln.md`. The
5-step narrowing took ~3 sessions of compute time but produced a
durable record at every step (sections §15–§17 + memory entries
+ raw evidence files in `evidence/ship-007-layer-3-anomaly/`).

### 18.8 What's the next observable state-change?

Two parallel paths, ranked by lead time:

**Short path (1–2 sessions):** PR #1066 lands → `apr trace
--payload` on the canonical 7B teacher emits 4 new sub-FFN lines
per layer → whichever of {ffn_gate, ffn_up, ffn_silu_gate,
ffn_swiglu_inner, ffn_out} carries the layer-3 53× spike names
the bug site → fix lands → 5 MODEL-1 PARTIALs auto-discharge.

**Long path (multi-session):** Address task #132 (`Device`
parameter on `TransformerTrainer::new` + `apr pretrain --device
cuda` wiring) → tokenize The Stack v2 Python with vocab=50,257
→ run convergence to CE ≤ 2.2 on val → checkpoint as `.apr` →
9 MODEL-2 PARTIALs auto-discharge.

The two paths are independent. Closing the short path is what
unlocks live evidence for SHIP-007 itself; closing the long path
is what makes MODEL-2 a ship-able artifact.

### 18.9 Methodological invariant

Every section of §15–§17, every contract authored under
`contracts/trace-ffn-sub-block-v1.yaml`, every PR landed since
2026-04-23 follows the same loop:

1. **Premise from live evidence**, not speculation
   (per `feedback_no_guessing.md`).
2. **Contract before code** when extending instrumentation
   (per `feedback_apr_trace_not_eprintln.md`).
3. **Drift-prevention test** that pins the new state into Rust
   (per `feedback_coverage_contracts_coevolution.md`).
4. **Spec amendment** that records the falsifier result, not the
   plan (per Toyota Way "fact at the gemba").
5. **PR with auto-merge** so the cascade flows without manual
   intervention (per `feedback_auto_merge_green_prs.md`).

Spec progression in this session: **v2.58.0 → v2.59.0 → v2.60.0 →
v2.61.0 → v2.62.0 → v2.63.0** (this section). No coverage tally
change from §18 — chain-of-thought recording, not a discharge.

---

## §45. `apr-cpu-vs-gpu-output-parity-v1` 5/5 LIVE DISCHARGE milestone (2026-05-04)

The `apr-cpu-vs-gpu-output-parity-v1` contract reaches its terminal state today. All five falsifiers (FALSIFY-CPU-GPU-001..005) are now DISCHARGED with live empirical evidence on the canonical Qwen2.5-Coder-7B teacher. The §41 → §43 → §44 → §45 jidoka chain is contract-complete: silent-GPU-gibberish on canonical broken-GPU is no longer possible — both the implementation closure AND end-to-end live verification have been delivered.

### 45.1 What landed

| PR | Smoke | What it discharged |
|----|------|-------|
| [#1445](https://github.com/paiml/aprender/pull/1445) | Default-mode `apr run` (RTX 4090, post-#1442 binary) — full jidoka chain emits 3 tagged stderr lines + delivers correct CPU output | FALSIFY-CPU-GPU-005 PARTIAL_ALGORITHM_LEVEL → DISCHARGED. Contract v1.3.0 → v1.4.0. |
| [#1446](https://github.com/paiml/aprender/pull/1446) | `apr run --no-gpu` smoke (9.02s, 0 GPU log lines, correct output) + reuse of #1445 wgpu-smoke.log evidence | FALSIFY-CPU-GPU-001/002/003 PARTIAL_ALGORITHM_LEVEL → DISCHARGED. FALSIFY-CPU-GPU-004 FUNCTIONAL → DISCHARGED. Contract v1.4.0 → v1.5.0. |

### 45.2 The complete observed jidoka chain (verbatim from #1445 wgpu-smoke.log)

```
[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected, attempting fallback:
    Inference error: PARITY-GATE FAILED: GPU computes a DIFFERENT function than CPU.
    Cosine similarity: -0.005190 (required: ≥0.98)
    CPU argmax: 334 | GPU argmax: 8127
    Max absolute logit difference: 19.5053

Backend: wgpu (Vulkan)
[PMAT-333] Dequantizing 28 layers ...
[apr-cpu-vs-gpu-output-parity-v1] wgpu path rejected, attempting fallback:
    cosine vs CPU = 0.766079 (< 0.99)

Output:
2 + 2 equals 4.
```

Three jidoka tags fire in deterministic order. The user observes which backends were rejected and why, with no `--verbose` flag required. Fallback proceeds CUDA → wgpu → CPU and delivers the correct CPU output. This is the first end-to-end live verification of the entire jidoka chain on the canonical broken-GPU model.

### 45.3 Coverage flip

This is the **largest single-cycle coverage flip** of the SHIP-TWO program: **+5 falsifiers** moved into DISCHARGED in one 2-PR cycle.

| Falsifier | Before today | After PR #1446 |
|-----------|--------------|---------------|
| FALSIFY-CPU-GPU-001 | PARTIAL_ALGORITHM_LEVEL | **DISCHARGED** |
| FALSIFY-CPU-GPU-002 | PARTIAL_ALGORITHM_LEVEL | **DISCHARGED** |
| FALSIFY-CPU-GPU-003 | PARTIAL_ALGORITHM_LEVEL | **DISCHARGED** |
| FALSIFY-CPU-GPU-004 | FUNCTIONAL | **DISCHARGED** |
| FALSIFY-CPU-GPU-005 | PARTIAL_ALGORITHM_LEVEL | **DISCHARGED** (#1445) |

Coverage tally: **15+37 → 20+32**. Contract `apr-cpu-vs-gpu-output-parity-v1` reaches v1.5.0 ACTIVE with all 5/5 falsifiers DISCHARGED — the contract is COMPLETE.

### 45.4 Why this milestone matters

**For SHIP-TWO program audit**: every contract-claimed gate against silent GPU gibberish is now live-verified, not just impl-closed. A future PR that regresses the parity gate (e.g., re-wraps the eprintln in `if verbose`, swaps the cosine helper, removes the wgpu probe) trips one of three drift-prevention surfaces: unit-test (cosine helper math + log prefix const), integration test (`test_no_unregistered_commands`), and reproducible smoke evidence file.

**For MODEL-1**: today's discharge does NOT fix the underlying SHIP-007 GPU kernel bug (the GPU path still produces wrong output on canonical 7B). What it DOES is close the user-visible failure mode — the user can still ship MODEL-1 via `apr run --no-gpu` (CPU path, correct output) and see clear stderr signaling when GPU is rejected. MODEL-1 ship % moves to **91%** because the entire MODEL-1-blocking jidoka contract is closed; only the §40 GPU kernel root-cause fix remains for full GPU-shipability.

**For the spec amendment cadence**: today's session shipped §41 (jidoka chain), §42 (hub build chain), §43 (distill-train falsifier-parity), §44 (part b impl + distill 9/9), §45 (5/5 contract DISCHARGE). Five spec amendments + 16 PRs in flight in a single session, each preserving the audit story for a future maintainer.

### 45.5 Five Whys

1. **Why is this milestone significant?** It's the first contract in the SHIP-TWO program to reach 5/5 DISCHARGED — a complete-evidence terminal state. Other contracts have multiple PARTIAL or FUNCTIONAL gates; this is the first all-DISCHARGED contract.
2. **Why was the multi-discharge bundleable into one PR?** Because the same #1445 wgpu-smoke.log evidence already covered FALSIFY-CPU-GPU-001/002/003 in addition to 005. Adding one `--no-gpu` smoke (9.02s) covered FALSIFY-CPU-GPU-004. Bundling preserved the audit story (one PR = one discharge cycle = one evidence dir).
3. **Why does cosine=0.766 (wgpu) matter for the contract verdict?** It's empirical justification for the 0.99 floor (rather than 0.95 or 0.98). Argmax-only catches CUDA (cos=-0.005, fully orthogonal) but might pass wgpu (cos=0.766, similar direction but wrong scale). The 0.99 floor catches both. Future contract revisions that propose loosening the floor have a concrete data point to argue against.
4. **Why doesn't this discharge close MODEL-1?** Because the parity contract is the *defensive* layer. The *underlying* GPU bug per §40 still produces wrong output. The defensive layer ensures users don't see silent gibberish; the offensive fix would let MODEL-1 ship via GPU rather than CPU. SHIP-007 root-cause is the remaining blocker for full GPU shipability.
5. **Why bound this cycle to a §45 spec amendment now?** The §41-§44 cadence has been "amend after each ≥3-PR cycle". Today's #1445 + #1446 (only 2 PRs) is below that threshold but the *milestone* (first 5/5 contract) warrants its own §45 record so it's auditable from the spec alone, not buried in a multi-cycle §46.

### 45.6 Net effects

- **Contract `apr-cpu-vs-gpu-output-parity-v1`**: v1.3.0 → **v1.5.0 ACTIVE**, 5/5 falsifiers DISCHARGED. Terminal complete state.
- **MODEL-1 ship %**: 89% → **91%**.
- **MODEL-2 ship %**: 57% (unchanged this cycle).
- **Coverage tally**: 15+37 → **20+32** (+5 DISCHARGED).
- **Today's session**: 16 PRs in flight (#1437-#1446 + #1444 + this one). 6 spec amendments (§41 records pre-session, §42-§45 in-session; §44 just merged via #1444).

### 45.7 Next-session pickup

The remaining MODEL-1 / MODEL-2 levers are both multi-PR research tracks:

(a) **MODEL-1 SHIP-007 GPU kernel root-cause fix** — per memory's `2026-05-03 SHIP-007 finding`, divergence is pinpointed to layer-0 attn_out (cos drops from 0.99999995 attn_norm → 0.9966 attn_out). Next-step bisection inside the attention block requires extending `apr trace --save-tensor` with sub-stage granularity (qkv, RoPE, softmax, V, O) and re-running the layer-0 oracle bisection. Single highest-leverage MODEL-1 work; could push ship % to 95%+.

(b) **MODEL-2 §35 real-training implementation** — extend `run_config_train` from a manifest-only stub to actual gradient-descent over precomputed teacher logits. Math is in place (canonical `distill::loss::DistillationLoss` + parallel `hf_pipeline::distillation::DistillationLoss`). Optimizer + checkpoint loop is missing. Multi-PR (3-5) scope; would simultaneously discharge TRAIN-001/002/009 (currently 1× PARTIAL + 1× PARTIAL + 1× BLOCKER_FIXTURE_ABSENT).

(c) **Cross-contract sweep** — other contracts likely have similar PARTIAL → DISCHARGED candidates if today's smoke evidence is reusable. E.g., `apr-vs-gguf-forward-parity-v1` may benefit from the same canonical-7B smoke for its own gates.

Operator preference decides which lands first. Each is roughly single-day-of-work scope.

---

## §44. FALSIFY-CPU-GPU-005 part b implementation + distill-train 9/9 sweep close (2026-05-04)

Two PRs that close the two outstanding scope items §43.6 named as next-session pickup: (a) the wgpu cosine parity gate implementation, and (b) the remaining unbound falsifiers in the distill contract. Both pass `pv validate`, both build clean, and both close concrete gaps that the v2.88.0 spec called out as deferred.

### 44.1 What landed

| PR | What | Effect |
|----|------|--------|
| [#1442](https://github.com/paiml/aprender/pull/1442) | FALSIFY-CPU-GPU-005 part b live implementation | ~70 LOC inline at `try_apr_wgpu_inference` (`crates/aprender-serve/src/infer/gguf_gpu_generate.rs` ~line 441-510), between `kv_caches` init and the autoregressive loop. Algorithm: probe-token CPU forward via `OwnedQuantizedModel::forward_single_with_cache` with a tiny `OwnedQuantizedKVCache::from_config(cfg, 2)` → wgpu single-step replay using the same `fwd.forward_layer` code path as the autoregressive loop, with a separate `probe_kv_caches` vec (max_seq=2) → output norm + LM head argmax math mirrors loop body → `cpu_vs_gpu_cosine_similarity` (helper from §43 PR #1440, module-scope, no `--features cuda` dep) → if `!(cos.is_finite() && cos >= 0.99)` emit `WGPU_FALLBACK_LOG_PREFIX` tagged stderr line and return None. Probe error paths (CPU forward failure, wgpu probe layer failure) also fail-closed with the contract-tagged log. Cost: one extra forward pass at init (~2-5ms on 7B), paid once per `apr run`, not per token. Real autoregressive `kv_caches` are NOT touched by the probe. Contract `apr-cpu-vs-gpu-output-parity-v1` v1.2.0 → v1.3.0 ACTIVE. |
| [#1443](https://github.com/paiml/aprender/pull/1443) | distill-train 9/9 falsifier sweep close | Adds `algorithm_evidence` blocks to the last three unbound falsifiers in `apr-cli-distill-train-v1`: TRAIN-007 PARTIAL_ALGORITHM_LEVEL via `pv validate` (live: 0 errors / 0 warnings); TRAIN-008 PARTIAL_ALGORITHM_LEVEL via `cargo test -p apr-cli --test cli_commands registered_commands` (live: 1 pass — `test_no_unregistered_commands` enforces the 3-surface invariant); TRAIN-009 BLOCKER_FIXTURE_ABSENT (pending §35 real-training implementation — there is no val_loss to compare without gradient descent). All 9 TRAIN-* falsifiers now have explicit status blocks. |

### 44.2 Coverage flips

| Falsifier | Status before | Status after | Notes |
|-----------|---------------|--------------|-------|
| FALSIFY-CPU-GPU-005 (wgpu parity gate) | PARTIAL_ALGORITHM_LEVEL (visibility-log + cosine helper only) | PARTIAL_ALGORITHM_LEVEL (gate impl in place; live broken-GPU smoke deferred) | Code-evidence half complete. Live discharge needs operator smoke on canonical 7B teacher. |
| FALSIFY-APR-DISTILL-TRAIN-007 | unbound (drift between task #218/#247 list and YAML) | PARTIAL_ALGORITHM_LEVEL | `pv validate` is meta-discharge; runs as a precondition for every contract amendment. |
| FALSIFY-APR-DISTILL-TRAIN-008 | unbound (drift between task list and YAML) | PARTIAL_ALGORITHM_LEVEL | Existing `test_no_unregistered_commands` integration test enforces 3-surface invariant per `feedback_cli_subcommand_three_surface_drift`. |
| FALSIFY-APR-DISTILL-TRAIN-009 | unbound | BLOCKER_FIXTURE_ABSENT | Honest classification: blocker is §35 real-training implementation (multi-PR scope). Path to DISCHARGED documented in YAML notes. |

Coverage tally: **15 + 35 → 15 + 37** (+2 PARTIAL_ALGORITHM_LEVEL closed; TRAIN-009 explicitly blocked, not counted).

### 44.3 Why this chain matters

**MODEL-1 (#1442 part b)**: Closes the last loophole in the §41 jidoka chain. Pre-#1442, even after §41/§43 made wgpu lifecycle visible, a wgpu-broken-but-cosine-acceptable model could still ship subtly wrong output. The cosine probe at init is the same algorithmic shape as the CUDA `parity_gate` (`mod_parity_gate.rs:cosine_similarity` + `parity_gate`) — wgpu now gets the same defensive check, paid once per `apr run` rather than per token. The implementation reuses the autoregressive loop's exact `fwd.forward_layer` code path (so a probe-vs-loop divergence would itself be a separate bug, not a parity gap), and uses the §43 module-scope `cpu_vs_gpu_cosine_similarity` helper with no `--features cuda` dependency.

**MODEL-2 (#1443 sweep close)**: The distill-train contract has reached terminal-binding state. Every falsifier (9/9) has an explicit `algorithm_evidence` block — 8 PARTIAL_ALGORITHM_LEVEL with concrete test_locations, 1 explicitly BLOCKER_FIXTURE_ABSENT pending §35. There is no longer any "claimed PARTIAL but YAML doesn't show it" drift between the task list and the contract. Future MODEL-2 work on this contract has a clean slate: the only remaining lever is §35 real-training implementation, which is multi-PR scope and will discharge TRAIN-001/002/009 simultaneously when it lands.

### 44.4 Five Whys

1. **Why ship part b implementation now?** v2.88.0 §43.6 (a) named it as the bounded next-session pickup, and the cosine helper from #1440 (also v2.88.0) had unblocked the path. Toyota Way: each scoped commitment lands as its own focused PR.
2. **Why inline implementation, not extracted helper?** Loop body is ~30 LOC; extracting a separate function would force passing 8+ borrowed locals (max_seq, eps, vocab_size, hidden_dim, num_layers, output_norm, lm_head_f32, fwd) or wrapping them in a single-use struct. Inline block scope localizes `probe_kv_caches`, `hidden`, `normed`, `wgpu_logits` cleanly.
3. **Why fail-closed on probe errors (return None) instead of propagating?** Per `feedback_fix_root_cause_never_route_around` and §40/§41 jidoka pattern: the gate's job is to NEVER ship silent gibberish. CPU probe failure or wgpu kernel failure both indicate the wgpu path is unsafe — the correct user experience is fall-to-CPU with a tagged stderr line, not crash or hide.
4. **Why mark TRAIN-009 BLOCKER_FIXTURE_ABSENT instead of leaving it untyped?** Honest classification beats false PARTIAL claims. TRAIN-009 has a clear test design (`tests/distill_smoke.rs`) and a clear blocker (§35 real-training). Marking it as BLOCKER makes the dependency explicit so a future PR cannot accidentally promote it without the prerequisite. The path to DISCHARGED is documented in the YAML notes.
5. **Why bound this cycle to 2 PRs?** Each falsifier (or family) gets its own focused PR per Toyota Way. #1442 was 70 LOC + 9 LOC YAML; #1443 was 45 LOC YAML. Bundled they would be ~125 LOC across two distinct contracts and would obscure the audit story.

### 44.5 Net effects

- **MODEL-1 ship %**: 88% → **89%** (wgpu jidoka armor complete at the init boundary, symmetric to §41's CUDA parity_gate).
- **MODEL-2 ship %**: 56% → **57%** (last falsifier-binding gap closed for distill contract; only remaining MODEL-2 lever is §35 real-training implementation, multi-PR scope).
- **Coverage scoreboard**: 15+35 → **15+37** (+2 PARTIAL_ALGORITHM_LEVEL closed; TRAIN-009 explicitly blocked).
- **Contracts touched**: `apr-cpu-vs-gpu-output-parity-v1` v1.2.0 → v1.3.0 ACTIVE (gate impl recorded); `apr-cli-distill-train-v1` 9/9 falsifiers now algorithm-bound (no version bump — adding evidence is patch-level, contract still v1.0.0 PROPOSED).
- **`pv validate`** exits 0 on both touched contracts.

### 44.6 Next-session pickup

Three natural levers, ordered by ROI:

(a) **FALSIFY-CPU-GPU-005 live discharge** — operator-driven smoke on canonical 7B teacher: rebuild `apr` with `--features cuda gpu`, run `apr run /mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr --prompt 'What is 2+2?' --max-tokens 8 --temperature 0.0 2>&1 | tee /tmp/wgpu-smoke.log`, expect stderr to contain `[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected` (existing) AND `[apr-cpu-vs-gpu-output-parity-v1] wgpu path rejected, attempting fallback: cosine vs CPU = ...` (NEW from #1442) AND CPU produces "2 + 2 equals 4." Promotes FALSIFY-CPU-GPU-005 from PARTIAL_ALGORITHM_LEVEL → DISCHARGED. Cost: ~10min binary rebuild + ~30s model load + ~5s inference.

(b) **MODEL-2 §35 real-training implementation** — the only remaining MODEL-2 lever; would simultaneously discharge TRAIN-001/002/009 from PARTIAL/BLOCKER → FUNCTIONAL when it lands. Multi-PR scope: extend `run_config_train` from a manifest-only stub to actual gradient-descent over precomputed teacher logits, building on the existing `distill::loss::DistillationLoss` and `hf_pipeline::distillation::DistillationLoss` impls (math is in place; only the optimizer + checkpoint loop is missing). Estimated 3-5 PRs depending on slice size.

(c) **MODEL-1 SHIP-007 GPU kernel root-cause fix** — the underlying GPU forward bug per §40. The §41/§44 jidoka chain prevents silent gibberish (user sees a fallback log + correct CPU output), but the GPU path itself still produces wrong output. Fixing it would let MODEL-1 ship via GPU rather than `--no-gpu`. This is the highest-leverage MODEL-1 work but also the largest scope (memory's `2026-05-03 SHIP-007 finding` pinpoints layer-0 attn_out as the divergence point; further bisection inside attn block is needed).

(a) is single-PR + ~10min compute + closes one PARTIAL → DISCHARGED. (b) and (c) are multi-PR research tracks. Operator preference decides which lands first.

---

## §41. `apr-cpu-vs-gpu-output-parity-v1` chain — three-layer jidoka armor at the dispatch boundary (2026-05-03)

### 41.1 What landed

Four PRs in one session, all merged on `main`, all under the `apr-cpu-vs-gpu-output-parity-v1` umbrella contract:

| PR | What | Effect |
|----|------|--------|
| [#1427](https://github.com/paiml/aprender/pull/1427) | Author contract v1.0.0 PROPOSED | 3 equations + 4 falsifiers (FALSIFY-CPU-GPU-001..004) codify the regression class triggered by §40's "GPU silent gibberish" finding |
| [#1428](https://github.com/paiml/aprender/pull/1428) | Make CUDA fallback log visible without `--verbose` + bump v1.0.0 PROPOSED → v1.1.0 ACTIVE | Empirically verified (live `apr -v run` on canonical 7B teacher) that the `parity_gate` IS already wired on the `.apr` → `OwnedQuantizedModelCuda` path via `with_max_seq_len:268-279`. v1.0.0's claim "no gate runs for the trueno-graph .apr load path" was incorrect; the v6 evidence file documents the correction. The user-visible gap was that the gate's failure (e.g. `CUDA_ERROR_ILLEGAL_ADDRESS` during the gate's own GPU forward) was logged behind `if verbose` at `gguf_gpu_generate.rs:487-489`. PR converts to unconditional `eprintln` with contract tag |
| [#1429](https://github.com/paiml/aprender/pull/1429) | Drift-prevention: `pub(crate) const CUDA_FALLBACK_LOG_PREFIX` + unit test | Locks the contract-tagged prefix shape at the type/test level. Future regressions (rename without bump, re-wrapping in `if verbose`) fail at `cargo test` — no GPU required for the test |
| [#1430](https://github.com/paiml/aprender/pull/1430) | FALSIFY-CPU-GPU-005 (wgpu visibility + parity-gate symmetry) + bump v1.1.0 → v1.2.0 ACTIVE | (a) wgpu lifecycle log made unconditional (symmetric to #1428's CUDA fix); (b) wgpu cosine-similarity gate bound at PARTIAL_ALGORITHM_LEVEL pending follow-up implementation (~100-150 LOC). Contract now 5 falsifiers / 5 obligations |

### 41.2 Net behavioural change for `apr run`

Before today: default `apr run <model.apr>` on a SHIP-007-broken GPU build emitted gibberish ("ampiezza = 10\\nampie") with **zero** stderr signal that GPU had been rejected. User had no diagnostic path to `--no-gpu` workaround.

After today: same invocation emits the full backend-fallback chain on stderr without `--verbose`:

```
[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected, attempting fallback:
  Inference error: PARITY-GATE: GPU forward failed: ...CUDA_ERROR_ILLEGAL_ADDRESS...
Backend: wgpu (Vulkan)
[wgpu] Skipping weight 'lm_head' (2180.0 MB > 2147.5 MB limit) — CPU fallback
...
```

The user now has the diagnostic to know `--no-gpu` is the correct path on this build.

### 41.3 What this is NOT

This chain does **not** fix the SHIP-007 GPU kernel bug. The `cuBLASLt FP8` / trueno_gpu manual-graph (646 kernels) defect identified in §40 still produces wrong output. What today's chain does is convert the failure mode from "silent gibberish" to "loud, contracted, user-visible fallback decision". This is jidoka (stop-the-line / defect visibility) — separate from the actual fix.

### 41.4 Five Whys — why ship visibility before the kernel fix?

1. **Why armor the dispatch boundary first?** Because MODEL-1 is shippable today via `apr run --no-gpu` (per §40 — CPU path produces correct "2 + 2 equals" in 9.81s on RTX 4090, faster than the broken 72.95s GPU path). The blocker for shipping with confidence was that users running default `apr run` would get garbage with no signal to switch flags. Closing that signal gap is necessary for ship and was achievable in 4 small PRs across one session.
2. **Why three layers (visibility + drift-prevention test + wgpu symmetry)?** Because each layer addresses a distinct regression class:
   - Visibility (#1428) closes "silent fallback today"
   - Drift-prevention test (#1429) closes "future refactor reverts visibility"
   - wgpu symmetry (#1430) closes "fallback hits wgpu which has its own gibberish, leaving user back in silent-garbage land"
3. **Why bump the contract three times in one day?** Because each PR materially changed the contract's claims (corrected wrong algorithm_evidence in #1428, added a new falsifier in #1430). Memory `feedback_pv_not_bash_for_contracts.md` and the spec's contract-first methodology require the YAML to track reality.
4. **Why not implement the wgpu cosine gate today?** Because it requires extracting the per-token wgpu decode loop body into a callable single-step function (~100-150 LOC + test), which is bigger than one /loop iteration. Bound at PARTIAL_ALGORITHM_LEVEL with a clear implementation sketch in v1.2.0 algorithm_evidence; deferred to a follow-up PR.
5. **Why is this MODEL-1 ship % progress and not just paperwork?** Because under the spec's "MODEL-1 ships GPU only" memory rule (`feedback_model_1_ships_gpu_only.md`) the ship gate was previously ambiguous when GPU produces garbage. With the visibility chain, MODEL-1 has a documented `--no-gpu` recovery path that's automatically discoverable from any default-mode `apr run` invocation. Per §40's own conclusion, "MODEL-1 is shippable today via CPU path" — today's chain makes that actually true for users, not just for spec authors.

### 41.5 Coverage update

No PARTIAL→DISCHARGED flips today. The contract `apr-cpu-vs-gpu-output-parity-v1` itself was authored fresh as v1.0.0 PROPOSED → v1.2.0 ACTIVE within one session, so it's not part of the coverage scoreboard's PARTIAL/DISCHARGED count yet (it would need explicit binding via tasks/spec rules to count). Tally: **15 + 33** (unchanged).

### 41.6 Next-session pickup

Two natural levers:

(a) **FALSIFY-CPU-GPU-005 part b** (wgpu cosine gate) — extract wgpu single-step decode body, run one CPU-vs-wgpu BOS forward at init, cosine-compare logits, return None on < 0.99. ~100-150 LOC + test. Promotes FALSIFY-CPU-GPU-005 from PARTIAL_ALGORITHM_LEVEL → FUNCTIONAL.

(b) **MODEL-2 distill-train scaffolding** (§35 / `apr-cli-distill-train-v1`) — start the Rust impl. Memory says ~600-1200 LOC + tests, multi-day. One iteration = one bounded sub-task (e.g. KL divergence loss helper, or temperature-scaled softmax kernel).

Both are bounded. (a) closes one more loophole on the MODEL-1 fallback layer; (b) starts moving MODEL-2 ship % off its 50% plateau. Operator preference decides which lands first.

---

## §36. Plain-language status — what's left to ship the two models (2026-04-28)

A landmark section in plain language for readers who don't want to chase the §15→§35 chain. Two paragraphs.

### 36.1 MODEL-1 — the bug

MODEL-1 (Qwen2.5-Coder-7B Apache-Q4K) is already published to HuggingFace. But when you run inference through APR, the math goes wrong inside the FFN block at layer 3 — outputs are 18× too spread compared to the GGUF reference on the same prompt. We've tested three theories today and refuted all three:

- **§28** said the matmul kernel was wrong; **§30** proved it isn't (q4k_layers populated, kernels equivalent within Q4K tolerance).
- **§31** said qkv_bias values were wrong; **§32** proved they aren't (APR ≡ GGUF byte-for-byte) — the apparent gap was a trace-capture-point mismatch.
- **PR #1082 + #1083 sub-FFN comparison** said layer-3 weights might differ — they don't (also byte-identical APR vs GGUF).

The actual bug is somewhere in cumulative F32 precision drift through residual connections. The remaining work: with PR #1082 (sub-FFN telemetry) just merged and PR #1083 (CLI wiring) auto-merging, run `apr trace --payload <gguf-teacher>` and `apr trace --payload <apr-teacher>` on the same prompt and compare layer-by-layer until you find where the gap appears. Then fix at root.

### 36.2 MODEL-2 — the model

MODEL-2 (paiml/albor-llama-370m-python-v1) was trained today end-to-end on 565M tokens of Python+permissive code. Best val_loss is **9.38**. Spec target is **3.0**. The 370M-from-scratch architecture has converged — running 4× more steps (50K → 200K) yielded the same outcome (§34). The model is capacity-limited; the corpus and step budget are not the bindings.

The only realistic path to val_loss=3.0 is **distillation**: use the shipped MODEL-1 7B as a teacher to teach the smaller 370M student, transferring knowledge from the larger model's logits. The `apr distill` command exists but its training loop is a stub (`distill.rs:1464` just clones tensors — §35 finding). The contract for the missing implementation was authored today as PR #1097 (`apr-cli-distill-train-v1.yaml`); the implementation itself is a multi-day Rust task. Once shipped, distillation runs in 2-4h on RTX 4090 and is expected to push val_loss substantially below the 9.38 ceiling toward the spec target.

### 36.3 Where the work is

| Model | What's blocked | What's needed |
|-------|---------------|---------------|
| MODEL-1 | Layer-3 FFN inference bug | Run `apr trace --payload` once #1083 lands; bisect; fix |
| MODEL-2 | Capacity ceiling at val_loss=9.38 | Implement `apr distill --stage train` per #1097; run distillation |

Both blockers are **fixable with code**, not training time or compute. The compute is sufficient (565.6M-token corpus + 7B teacher live on the host); the gap is implementation work.

### 36.4 What today's session shipped

11 PRs landed in 24 hours:
- 6 spec amendments (§30, §31/§32, §33, §34, §35, §36 — this one)
- 2 contracts (apr-cli-pull-dataset-v1 ACTIVE, apr-cli-distill-train-v1 PROPOSED)
- 1 implementation (P1.1 apr pull dataset extension)
- 1 contract (PR D apr-vs-gguf-forward-parity drift-prevention)
- 1 contract (parallel-bpe PROPOSED)
- 2 SHIP-007 sub-FFN telemetry PRs (PR B + PR C)

Plus the operational achievement: the full P1.0 → P2 corpus pipeline executed end-to-end with zero muda — every step went through the spec-canonical `apr` extension, never `huggingface-cli` or deprecated `batuta hf pull`.

---

