# aprender BEAT Scoreboard

The five-pillar mission: **replace _and beat_** scikit-learn, PyTorch, Unsloth,
Ollama/llama.cpp, **and Claude Code** (`apr code` agentic coding) in one pure-Rust
binary. A "beat" is never parity — it is a **falsifiable, CI-gated benchmark**
showing apr ≥ the incumbent on the incumbent's own canonical task (PMAT-741).

> **Pillar 5 (Claude Code) is TRACKED, not yet a clean WON.** Unlike pillars 1–4,
> the Claude-Code pillar splits into two scales with two different verdicts:
> function-scale outcome parity is **WON (1.0000)**, project-scale agentic
> multi-turn Arena is an **OPEN GAP (0.20)**. The honesty rule below forbids
> reporting the function-scale win as a project-scale win — see the dedicated
> Pillar 5 section.

> **Honesty rule:** a beat that apr fails never ships. We do **not** make a blanket
> "speed conceded" claim — apr wins speed in many measured, CI-gated cases (see the
> Speed scoreboard below). Where apr structurally loses, we name the **specific,
> narrow case** (e.g. *llama.cpp single-request c=1 decode*, *LAPACK/BLAS-bound
> sklearn*) rather than conceding speed wholesale.
>
> **Falsifier rule:** every beat is adversarially mutation-verified — injecting a
> regression into its property must make the gate FAIL (a beat that can't fail is
> theater). Verified 2026-06-14: NF4 / LoRA-merge / fail-closed each fail under a
> targeted mutation; autograd fails by construction (pinned-reference tolerance). See
> `evidence/beats-adversarial-verification-2026-06-14/findings.md`.
>
> **Withdrawal rule:** a beat whose re-measurement no longer supports the claim is
> **withdrawn from the scoreboard and recorded**, never quietly downgraded or
> deleted. Under-claiming is as much a reporting failure as over-claiming, so a
> withdrawal cites the numbers that replaced the claim. See
> [Withdrawn beats](#withdrawn-beats).

## Speed scoreboard (evidence-backed, sourced)

apr **wins** speed in these measured, CI-gated cases — every row cites a `contracts/beat-*.yaml`
contract (status: `enforced`) or in-repo evidence:

| Speed win | Result | Source (contract / evidence) |
|-----------|--------|------------------------------|
| **Cold-start vs sklearn** (static binary, no Python import) | apr **~528× faster** end-to-end (ratio 0.0019) | `beat-sklearn-coldstart-speed-v1.yaml` · `crates/aprender-core/tests/beat_sklearn_coldstart_speed.rs` |
| **Cold-start vs PyTorch** | apr **~1500× faster** (ratio 0.0007) | `beat-pytorch-coldstart-speed-v1.yaml` · `crates/aprender-core/tests/beat_pytorch_coldstart_speed.rs` |
| **Cold-start vs HF transformers** (inference import) | apr **~1388× faster** (ratio 0.0007) | `beat-hf-inference-coldstart-speed-v1.yaml` · `crates/aprender-core/tests/beat_hf_inference_coldstart_speed.rs` |
| **Cold-start vs Unsloth/PEFT** (LoRA-init one-shot) | apr **~5000× faster** (ratio 0.0002) | `beat-unsloth-coldstart-speed-v1.yaml` · `crates/aprender-train/tests/beat_unsloth_coldstart_speed.rs` |
| **GaussianNB fit+predict vs sklearn** | apr **~4.91× faster** (ratio 0.203, ln-hoist) | `beat-sklearn-gaussiannb-speed-v1.yaml` |
| **GMM fit vs sklearn** | apr **~4× faster** (ratio 0.25) | `beat-sklearn-gmm-speed-v1.yaml` |
| **BernoulliNB vs sklearn** | apr **~1.90× faster** (ratio 0.526) | `beat-sklearn-bernoullinb-speed-v1.yaml` |
| **LinearRegression vs sklearn** | apr **~1.78× faster** (ratio 0.56) | `beat-sklearn-linreg-speed-v1.yaml` |
| **ComplementNB vs sklearn** | apr **~1.68× faster** (ratio 0.596) | `beat-sklearn-complementnb-speed-v1.yaml` |
| **MultinomialNB vs sklearn** | apr **~1.60× faster** (ratio 0.625) | `beat-sklearn-multinomialnb-speed-v1.yaml` |
| **cuda-oxide attention vs hand-PTX** (GB10 Blackwell) | apr pure-Rust kernel **1.7–2.9× faster** than production hand-PTX (NW=8) | `experiments/cuda-oxide/PMAT-882-STATUS.md` (A/B on sm_121) |
| **cuda-oxide RMSNorm vs hand-PTX** (GB10 Blackwell) | apr pure-Rust kernel **1.4–1.5× faster** | `experiments/cuda-oxide/rmsnorm/RESULTS.md` (PMAT-893) |
| **trueno SIMD dot product** | AVX-512 **6–17× faster** / AVX2 **10–12× faster** than scalar | `crates/aprender-compute/AVX512_COMPUTE_BOUND_VALIDATION.md` |
| **Deploy footprint vs PyTorch** | apr **53.9 MiB** static binary, **15.8–17.1× smaller** than torch ~853–921 MiB | `beat-pytorch-deploy-footprint-v1.yaml` |

apr is at measured **parity** — not a win, not a loss — in this case:

| Parity (no win claimed) | Result | Source (contract / evidence) |
|-------------------------|--------|------------------------------|
| **GPU decode vs Ollama** | apr **1.015–1.109×** ollama on RTX 4090 sm_89 (three post-#2323 medians: 1.109 / 1.042 / 1.015). Inside measurement noise — apr does **not** currently win GPU decode vs ollama. The earlier **1.371× headline is WITHDRAWN** (see [Withdrawn beats](#withdrawn-beats)) | `beat-ollama-decode-throughput-speed-v1.yaml` (`beat_threshold: 0.9000` — a **no-collapse floor**, not a beat) · `crates/aprender-serve/tests/beat_ollama_decode_throughput_speed.rs` (`ENFORCED_THRESHOLD: f64 = 0.90`) |

apr **loses** speed in these **specific, narrow** cases (not a blanket concession):

| Speed loss (narrow case) | Result | Note |
|--------------------------|--------|------|
| **llama.cpp** single-request c=1 decode | llama.cpp ~1.55× faster (431 vs 277 tok/s, RTX 4090) | This is *llama.cpp*, not Ollama; against Ollama on the same host apr is at **parity** (1.015–1.109×), not ahead — the 1.371× win previously cited here is withdrawn |
| **7B-Q4K on GB10 Blackwell** | ~12 tok/s (bandwidth-bound; DP4A path degraded) | Memory-wall + degraded DP4A on Blackwell, not a kernel-design loss |
| **Short-prompt one-shot wall-clock vs Ollama** | apr CLI ~2.7–3.9 s fixed startup vs Ollama's resident daemon | Decode-rate beat is steady-state; one-shot startup is a separate, scoped comparison (see Pillar 4) |
| **PCA fit_transform vs sklearn** | apr ~18.6× *slower* | sklearn delegates to LAPACK-SVD; apr's decomposition is unoptimized |
| **KMeans / Ridge / Lasso vs sklearn** | apr ~2× / slower / ~19× slower | LAPACK/BLAS-bound (sklearn Cholesky/coordinate-descent); apr wins the LAPACK-free O(nd) tasks above |
| **2-layer MLP training time vs PyTorch** | apr ~11× *slower* | Overhead-bound; PyTorch MKL + fused autograd (apr is provably *correct* — see autograd-equivalence beat) |

## Status legend

| Status | Meaning |
|--------|---------|
| ✅ **WON** | CI-gated; apr ≥ incumbent; a regression hard-fails the gate |
| 📊 **TRACKING** | measured, pinned baseline; gate not yet wired (or stretch target) |
| 🚧 **PLANNED** | identified; not yet built |
| ⚖️ **NARROW LOSS** | measured; apr loses in this *specific* case (named root cause) — an optimization target, not a blanket concession |
| 🟰 **PARITY (no-collapse floor)** | measured within noise of the incumbent. The contract still gates, but it gates against *collapse* (`beat_threshold < 1.0`), so **no win may be claimed** from a green run |
| ⛔ **WITHDRAWN** | was published as WON; re-measurement did not reproduce it. Kept on the scoreboard with the retraction and the replacing numbers — never deleted |

## Pillar 1 — scikit-learn

| Beat | Metric | Result | Gate |
|------|--------|--------|------|
| Iris classification (RandomForest) | test accuracy | ✅ **WON** — apr 0.94 ≥ sklearn floor 0.94 (threshold 0.92) | per-PR `ci.yml` · `beat_sklearn_iris` |
| **Cold-start** (static binary, no Python import) | wall-clock ratio | ✅ **WON** — apr **~528× faster** end-to-end (ratio 0.0019, gate ≤ 0.10) | `beat_sklearn_coldstart_speed` · `beat-sklearn-coldstart-speed-v1` |
| LinearRegression fit+predict | wall-clock ratio | ✅ **WON** — apr **~1.78× faster** (ratio 0.56, gate ≤ 0.90) | nightly · `beat_sklearn_linreg_speed` · `beat-sklearn-linreg-speed-v1` |
| GaussianNB fit+predict | wall-clock ratio | ✅ **WON** — apr **~4.91× faster** (ratio 0.203, gate ≤ 0.50; ln-hoist) | nightly · `beat_sklearn_gaussiannb_speed` · `beat-sklearn-gaussiannb-speed-v1` |
| GMM fit | wall-clock ratio | ✅ **WON** — apr **~4× faster** (ratio 0.25, gate ≤ 0.70) | nightly · `beat_sklearn_gmm_speed` · `beat-sklearn-gmm-speed-v1` |
| BernoulliNB fit+predict | wall-clock ratio | ✅ **WON** — apr **~1.90× faster** (ratio 0.526, gate ≤ 0.90) | nightly · `beat_sklearn_bernoullinb_speed` · `beat-sklearn-bernoullinb-speed-v1` |
| ComplementNB fit+predict | wall-clock ratio | ✅ **WON** — apr **~1.68× faster** (ratio 0.596, gate ≤ 0.90) | nightly · `beat_sklearn_complementnb_speed` · `beat-sklearn-complementnb-speed-v1` |
| MultinomialNB fit+predict | wall-clock ratio | ✅ **WON** — apr **~1.60× faster** (ratio 0.625, gate ≤ 0.90) | nightly · `beat_sklearn_multinomialnb_speed` · `beat-sklearn-multinomialnb-speed-v1` |
| PCA fit_transform | wall-clock ratio | ⚖️ **NARROW LOSS** — apr ~18.6× *slower* (sklearn is LAPACK-SVD-bound) | — |
| Ridge fit+predict | wall-clock ratio | ⚖️ **NARROW LOSS** — apr ~1.5× *slower* (sklearn Ridge defaults to fast Cholesky; the LinReg-SVD win doesn't transfer) | — |
| Lasso fit+predict | wall-clock ratio | ⚖️ **NARROW LOSS** — apr ~19× *slower* (apr coordinate-descent unoptimized) | — |
| KMeans fit+predict | wall-clock ratio | ⚖️ **NARROW LOSS** — apr ~2× *slower* (both Lloyd; sklearn BLAS-bound) | — |

**Why apr wins / loses sklearn:** apr's wedge is the cache-friendly ikj SIMD matmul
plus a static, Python-free runtime — so it **wins** the LAPACK-free O(nd) tasks
(cold-start ~528×, normal-equations LinReg 1.78×, the Naive-Bayes family 1.6–1.9×,
GaussianNB 4.9×, GMM ~4×) and loses only the **LAPACK/BLAS-bound** ones (PCA-SVD,
KMeans, Ridge/Lasso) until apr's decomposition is optimized. This is a **specific,
named** set of losses — not a blanket speed concession. See
`memory project_sklearn_speed_beat_selection`.

## Pillar 2 — PyTorch

| Beat | Metric | Result | Gate |
|------|--------|--------|------|
| **Cold-start** (static binary, no torch import) | wall-clock ratio | ✅ **WON** — apr **~1500× faster** end-to-end (ratio 0.0007, gate ≤ 0.10) | CI `beat_pytorch_coldstart_speed` · `beat-pytorch-coldstart-speed-v1` |
| **Inference deploy footprint** | on-disk deploy bytes (model excluded) | ✅ **WON** — apr's self-contained pure-Rust static binary is **~53.9 MiB** (release; 44.9 MiB stripped) vs the PyTorch/transformers CPU inference deploy **~853 MiB** site-packages (torch CPU 698 + transformers 51 + numpy/sympy/tokenizers/… deps) — **921 MiB** with the CPython interpreter. Ratio **~15.8×** (site-packages) / **~17.1×** (full deploy); **~50×+** vs a 2.5–3.5 GB CUDA torch wheel. Host-independent; apr ships NO Python/framework runtime. | CI `beat_pytorch_deploy_footprint` · `beat-pytorch-deploy-footprint-v1` |
| **Autograd gradient equivalence** | max \|apr_grad − pytorch_grad\| | ✅ **WON** (PMAT-746) — apr's reverse-mode autograd ≡ PyTorch on a fixed 2-layer MLP (max \|Δ\|=**5.0e-7**, forward loss parity); a provable-correctness win to pair with apr's cold-start + footprint speed wins | CI `beat_pytorch_autograd_grad` · `apr-pytorch-autograd-equivalence-beat-v1` |
| 2-layer MLP training time | wall-clock ratio + MSE ≤ 0.05 | ⚖️ **NARROW LOSS** (PMAT-725) — apr ~11× *slower* (overhead-bound; PyTorch MKL + fused autograd; apr training is correct after #2000 but autograd Tensor ops don't yet use the SIMD Matrix path) | — |

## Pillar 3 — Unsloth

| Beat | Metric | Result | Gate |
|------|--------|--------|------|
| **Cold-start** (LoRA-init one-shot, static binary) | wall-clock ratio | ✅ **WON** — apr **~5000× faster** end-to-end (ratio 0.0002, gate ≤ 0.10) vs unsloth/peft LoRA-init | CI `beat_unsloth_coldstart_speed` · `beat-unsloth-coldstart-speed-v1` |
| **NF4 numerical-equivalence** | max \|apr_recon − bitsandbytes_recon\| | ✅ **WON** — apr's pure-Rust NF4 ≡ bitsandbytes (max \|Δ\|=**4.92e-7**, MSE 0.007378 == bnb 0.007378; same codebook + blockwise convention, contract + Lean-gated) | CI `beat_nf4_bitsandbytes_equivalence` · `apr-nf4-bitsandbytes-equivalence-beat-v1` |
| **LoRA merge forward-equivalence** | max \|merged_fwd − factored_fwd\| | ✅ **WON** (PMAT-747) — apr's `MergeEngine` folds the adapter faithfully (max \|Δ\|=**1.49e-8**); merged-weight forward ≡ independent LoRA-factor forward. PEFT/Unsloth merge_and_unload has no such contract | CI `beat_lora_merge_forward_equivalence` · `apr-lora-merge-equivalence-beat-v1` |
| QLoRA fine-tune | loss-monotone + 4-bit footprint ≤ 0.30× f16 | 🚧 **PLANNED** (PMAT-711) | — |
| LoRA→GGUF merge | forward max-abs-diff < 1e-2 | 🚧 **PLANNED** (PMAT-712) | — |

## Pillar 4 — Ollama / llama.cpp

| Beat | Metric | Result | Gate |
|------|--------|--------|------|
| **GPU decode throughput vs Ollama** | tok/s ratio (RTX-4090 sm_89) | 🟰 **PARITY (no-collapse floor)** — apr **1.015–1.109×** ollama, same qwen2.5-coder-1.5b Q4_K_M GGUF, same host. Gate = apr median-of-7 ≥ ollama median **× 0.90** (`beat_threshold: 0.9000`). That is a floor against collapse, **not** a win — a green run proves apr did not fall off a cliff, nothing more. ⛔ The prior **1.371× WON claim is WITHDRAWN** (see below) | manual/GPU gate (no NVIDIA CI runner) · `beat_ollama_decode_throughput_speed` · `beat-ollama-decode-throughput-speed-v1` |
| **Fail-closed correctness** (headline correctness) | broken-artifact classes rejected | ✅ **WON** — apr rejects **10/10** semantically-broken tensor classes (zero/NaN/Inf/L2~0/constant/shape) fail-closed; **llama.cpp accepts** the same (measured: zeroed-ffn GGUF → `apr validate` ✗ FAIL, `llama-cli` 0 errors + ran it) | CI `beat_fail_closed_garbage` · `apr-fail-closed-garbage-beat-v1` |
| **llama.cpp** single-request c=1 decode | tok/s ratio (RTX-4090) | ⚖️ **NARROW LOSS** — llama.cpp ~1.55× *faster* (431 vs 277 tok/s) at concurrency=1; this is *llama.cpp*, not Ollama, against which apr measures at parity (1.015–1.109×) | — |
| 7B-Q4K decode on GB10 Blackwell | tok/s | ⚖️ **NARROW LOSS** — ~12 tok/s (bandwidth-bound; DP4A path degraded on Blackwell) | — |
| Short-prompt one-shot wall-clock vs Ollama | end-to-end seconds | ⚖️ **NARROW LOSS** — apr CLI ~2.7–3.9 s fixed startup vs Ollama's resident daemon (separate from the steady-state decode beat above) | — |

**GPU decode vs Ollama — PARITY, and the gate is a floor (PMAT-755, withdrawn
2026-07-31):** for the *same* qwen2.5-coder-1.5b Q4_K_M GGUF on the *same* RTX 4090
(sm_89), the three reproducible post-#2323 medians are **1.109×** (2026-07-29, apr
332.7 vs ollama 299.9), **1.042×** (2026-07-31, 342.4 vs 328.6) and **1.015×**
(2026-07-31 idle box, 318.2 vs 313.5). The contract pins `baseline_value: 1.0150` —
the **worst** of the three, not the best — and sets `beat_threshold: 0.9000`, matched
by `ENFORCED_THRESHOLD: f64 = 0.90` in the harness. **0.90 is a no-collapse floor: it
cannot express a win.** It exists to catch the class that actually hurts (silent CPU
fallback lands near ratio 0.065), and it sits 12% under the worst observed median so
it does not flake. Pillar 4 therefore claims **no GPU decode win over Ollama on
sm_89**; restoring a ≥1.10× win is tracked separately. Still a **manual/GPU gate**
(`#[ignore]`, NVIDIA host only) — there is no NVIDIA CI runner, same caveat as the
cuda-oxide throughput gate. The narrow losses above (llama.cpp at c=1,
7B-on-Blackwell, one-shot startup) remain **specific, named** cases, not a blanket
"speed conceded" stance — see the Speed scoreboard for the cases apr does win.

**Headline correctness beat — "we provably never ship garbage; they provably do":** a
model that *parses* but is *semantically* dead (all-zero / NaN / Inf weights) loads and
runs in llama.cpp/Ollama with exit 0 and no warning; apr's Poka-Yoke validation
(F-DATA-QUALITY-001..004) rejects it. Measured head-to-head 2026-06-13 in
`evidence/pillar4-fail-closed-2026-06-13/`. Correctness is a wedge none of the four
incumbents have. It is Pillar 4's **only** WON beat: the GPU decode-rate row above is
parity, not a second win.

**PMAT-742 (unblocked Pillar 4):** the default `apr run --gpu` was silently 8 tok/s — its
CUDA first-token parity gate false-rejected the correct fast path (a single BOS-only probe
diverges CPU-vs-GPU even when real generation is correct). Fixed by validating on the real
prompt context + near-tie tolerance → default `apr run` reaches the GPU fast path that the
re-measured 412.3 tok/s decode beat is built on.

## Pillar 5 — Claude Code parity (`apr code`)

The fifth incumbent: **Claude Code** (Anthropic's agentic coding agent). The beat
is `apr code` — aprender's pure-Rust, sovereign agentic coding agent — measured
against Claude Code at the **action-stream level** by the CCPA harness (record →
replay → distill). This pillar is **TRACKED, not yet a clean WON**: it splits into
two scales with two honest verdicts.

| Beat | Scale / Metric | Result | Gate |
|------|----------------|--------|------|
| **Function-scale outcome parity** | aggregate parity score on 30 canonical fixtures + HumanEval | ✅ **WON** — apr code ≡ Claude Code, **1.0000** (corpus **30/30**, HumanEval **5/5**, cross-swap **test-survival 1.0000**); the two are outcome-interchangeable at function scale | CCPA `ccpa corpus fixtures/canonical/` (FALSIFY-CCPA-008/013/016) · `claude-code-parity-apr-v1.yaml` v1.32.0 · `fixtures/canonical/measured-parity.json` |
| **Project-scale Arena** (live multi-turn) | oracle-pass rate, 5 real GitHub-issue fixtures | 📊 **TRACKED GAP** — claude teacher **0.20 (1/5)**, apr code student **0.00 (0/5)**; the static-fixture predictor is **Popperian-falsified** at project scale (StaticFalsified, M224 §5). This is the **open work** to advance in parallel | CCPA `ccpa arena fixtures/project-scale/` (FALSIFY-CCPA-017/018, PROPOSED) · `evidence/phase-5/arena-scores.json` |
| **Sovereignty on replay** | zero `api.anthropic.com` egress | ✅ **WON** — `apr code` replay opens **0** outbound sockets to Anthropic; pure-Rust local model, no Claude API | CCPA `FALSIFY-CCPA-006` · `claude-code-parity-apr-v1.yaml` |

**Honest split — function-scale WON, project-scale GAP (do NOT conflate):** at
**function scale** the two systems are functionally interchangeable — the CCPA
canonical corpus aggregates to **1.0000** (30/30 fixtures, 0 drift), the
real-binary HumanEval bilateral bench (claude 2.1.139 + apr 0.32.0 +
Qwen2.5-Coder-1.5B-Instruct-Q4_K_M) scores **1.0000** on MultiPL-E-Rust 5/5, and
cross-swap **test-survival is 1.0000** (10/10). That leg is a real, measured WON.
At **project scale**, the live multi-turn Arena (claude teacher vs `apr code`
student over real GitHub-issue fixtures) is the **OPEN GAP**: the claude teacher
itself only clears **0.20 (1/5)** and `apr code` clears **0.00 (0/5)**
(`evidence/phase-5/arena-scores.json`, M234). CCPA self-describes its
static-fixture approach as **Popperian-falsified** as a *project-scale* predictor
(M224 `design-audit.md` §5, verdict `StaticFalsified`) — the 1.0000 corpus number
validates **the meter** (the differ recognizes equivalent traces), **not**
system-level project-scale parity. Per the honesty rule, the specific gap is
**project-scale agentic multi-turn**, NOT function-scale code quality.

**Leading hypothesis for the gap (V1_004 chain, M286–M294):** the load-bearing
variable for 0% tool-call emission is the **model family**, not the inference
stack, active-param count, or MoE-vs-dense architecture. The **Qwen-Coder
finetune family emits 0 tool_calls**, while the non-Coder
**Qwen3-30B-A3B-Instruct** emits them (`{"name":"file_read",...}` in 20 tokens).
So the project-scale gap is an **agentic-emission / model-family** gap, not a
function-scale code-quality gap — the leading lever to advance Pillar 5 is to swap
the Qwen-Coder finetune for a tool-call-emitting non-Coder model.

**Source-of-truth split (monorepo policy):** the authoritative parity contract is
[`contracts/claude-code-parity-apr-v1.yaml`](../contracts/claude-code-parity-apr-v1.yaml)
(**v1.32.0**, 20 gates = 16 ACTIVE_RUNTIME + 4 PROPOSED) — aprender stays canonical
for contract TEXT. The thin aprender-side **tracking pointer** is
[`contracts/beat-claude-code-parity-v1.yaml`](../contracts/beat-claude-code-parity-v1.yaml),
which owns the obligation *"project-scale Arena parity must reach the CCPA-018
floor"* and points at the CCPA arena evidence as the measurement. Runtime
**enforcement** (CI, coverage, the live Arena bench) lives in the companion repo
**[paiml/claude-code-parity-apr](https://github.com/paiml/claude-code-parity-apr)**.
(`paiml/aprender#1078` authored the M0 spec + DRAFT contract; its body — 12 gates,
pre-v1.0.0 — is now **stale** vs the current v1.32.0 / 20-gate contract in-tree,
which is the authoritative, up-to-date copy. aprender-side tracking only; the CCPA
repo is not rewritten here.)

## Compute kernels — trueno / cuda-oxide

apr's compute foundation (trueno SIMD + cuda-oxide pure-Rust GPU kernels) wins on raw
kernel throughput, measured A/B against the incumbent hand-written path:

| Beat | Metric | Result | Source |
|------|--------|--------|--------|
| **cuda-oxide incremental attention** | µs ratio vs hand-PTX (GB10 sm_121) | ✅ **WON** (PMAT-882) — pure-Rust `#[kernel]` is **1.7–2.9× faster** than the production hand-PTX `multi_warp_attention` (NW=8) at every decode shape; bit-parity (cos=1.0000), no GH-480 JIT workaround | `experiments/cuda-oxide/PMAT-882-STATUS.md` |
| **cuda-oxide RMSNorm** | µs ratio vs hand-PTX (GB10 sm_121) | ✅ **WON** (PMAT-893) — matched pure-Rust kernel is **1.4–1.5× faster** per row than hand-PTX at every hidden size; parity cos=1.0000000 | `experiments/cuda-oxide/rmsnorm/RESULTS.md` |
| **trueno SIMD dot product** | speedup vs scalar | ✅ **WON** — AVX-512 **6–17× faster** (avg 10.8×), AVX2 **10–12×** for compute-bound dot/max/min | `crates/aprender-compute/AVX512_COMPUTE_BOUND_VALIDATION.md` |

*Class boundary (honest):* the cuda-oxide pure-Rust GO class is FMA/softmax/transcendental
kernels (attention, RMSNorm, RoPE, SwiGLU). The DP4A-bound Q4K GEMV/FFN path (PMAT-881)
stays hand-PTX — a **named NO-GO**, not a blanket loss.

## Withdrawn beats

A beat is withdrawn when re-measurement stops supporting the published claim. The
row is **not deleted** — deleting it would hide that the scoreboard was wrong, and a
reader who saw the old claim deserves the retraction next to it.

### ⛔ GPU decode throughput vs Ollama — "apr 1.371× faster" (WITHDRAWN 2026-07-31)

| | |
|---|---|
| **Claimed** | ✅ WON, apr **1.371×** ollama median (apr median-of-7 **412.3** vs ollama **300.7** tok/s; worst run 1.230×, best 1.523×), gate ≥ **1.10×** |
| **Claimed on** | 2026-06-15 (measurement), published 2026-06-25 via #2067 (PMAT-755), promoted TRACKING → ENFORCED |
| **Replaced by** | 🟰 PARITY, **1.015–1.109×**, gate `beat_threshold: 0.9000` (no-collapse floor) |
| **Contract** | `contracts/beat-ollama-decode-throughput-speed-v1.yaml` v2.0.0 — `baseline_value: 1.0150`, `baseline_floor: 0.9000`, `beat_threshold: 0.9000` |
| **Harness** | `crates/aprender-serve/tests/beat_ollama_decode_throughput_speed.rs` — `ENFORCED_THRESHOLD: f64 = 0.90` |

Four measurements on one host (lambda RTX 4090, sm_89), same GGUF on both sides:

| Date | apr median | ollama median | ratio | Source |
|------|-----------:|--------------:|------:|--------|
| 2026-06-15 | 412.3 | 300.7 | **1.371×** | promotion claim (#2067) — **not reproducible** |
| 2026-07-29 | 332.7 | 299.9 | 1.109× | cuda-nightly, PASSED |
| 2026-07-31 | 342.4 | 328.6 | 1.042× | cuda-nightly, FAILED |
| 2026-07-31 | 318.2 | 313.5 | 1.015× | idle box, this harness |

**Why it is apr's number that moved, not the rig:** the *ollama* column reproduces
across six weeks — 300.7 / 299.9 / 328.6 / 313.5. A measurement fault would drift
both columns. apr's moved 412 → ~318–342.

**Why the gate did not catch it:** the 2026-07-29 run **PASSED at 1.109×** against a
1.10× gate — 0.8% of headroom. It went unexamined because it was green. *A gate
passing with <1% headroom is a finding, not a pass.*

**Attribution, stated honestly:** #2323 (2026-07-27) made `auto_q4k` return `Mwv` on
every device; sm_89 previously defaulted to `HwDp4a`. The 412.3 figure predates that
change. This is **not** "#2323 cost 23%" — re-running today with `HW_DP4A_Q4K=1`
measures **20.3 tok/s**, because `HwDp4a` fails the F2 first-token cosine floor
(0.9186 < 0.95) and the run finishes on CPU SIMD. The claim is withdrawn as
**unreproducible**, not reattributed to a cause we have not proven.

**Consequence for the scoreboard:** Pillar 4 has **one** WON beat (fail-closed
correctness), not two. Restoring a ≥1.10× GPU decode win is tracked separately; until
a re-measurement supports it, no GPU decode win over Ollama on sm_89 may be published
here.

## Beat infrastructure (PMAT-741)

The machinery every beat plugs into:

- **`ContractKind::BeatBenchmark`** + validator (`BEAT-001..007`) — each beat is a
  contract under `contracts/beat-*.yaml` with a pinned incumbent baseline.
- **`Beat::evaluate(measured) -> Won | Regressed`** — the single-source verdict
  (`aprender-contracts`); a malformed contract is an error, never a silent pass.
- **`apr beat-run <contract> [--measured V]`** — CLI runner; reports the pinned
  baseline and exits non-zero on regression.
- **`.github/workflows/beat-speed-nightly.yml`** — speed gates run nightly and time
  apr vs the incumbent **same-host / same-run**, gating the *relative ratio* so
  CI-host speed variance can't cause flakes.

## Discipline

1. Every beat ships **as** a falsifiable CI gate — a throwaway regression must turn it red.
2. Speed gates use **same-host relative ratios** (flaky-resistant), never absolute wall-clock.
3. **Measure before claiming.** apr **wins** speed in many CI-gated cases (Speed
   scoreboard above); where it loses, name the **specific case** and root cause — never
   a blanket "speed conceded" stance.
4. **The contract is the gate of record.** This file must state the same threshold the
   contract carries; a `beat_threshold < 1.0` is a floor and may never be reported as a
   win. `crates/aprender-core/tests/readme_contract.rs` fails the build on drift
   (FALSIFY-DOCS-BEATS-001/002/003).
5. **A gate passing with <1% headroom is a finding.** Green is not the same as proven —
   the 2026-07-29 Ollama run passed at 1.109× against a 1.10× gate and was the
   regression.

_Last updated: 2026-08-13 — **withdrew the Ollama GPU-decode "1.371× WON" headline**
(#2349 carve-out). Measured reality on RTX 4090 sm_89 is **1.015–1.109×** — parity —
and the contract enforces `beat_threshold: 0.9000`, a no-collapse floor, not a beat.
The row moved out of the Speed-wins table into a new PARITY table; Pillar 4's table,
the llama.cpp comparison note and the headline paragraph were corrected to match; the
full claim history is preserved under [Withdrawn beats](#withdrawn-beats); ⛔/🟰
statuses and a withdrawal rule were added to the legend, and Discipline gained rules 4
and 5. `beats_doc_contract.rs` now gates this file against the contract._

_2026-06-25 — promoted the mission from FOUR-pillar to FIVE-pillar: added Pillar 5
(Claude Code / `apr code`) with the honest split — function-scale outcome parity WON
(1.0000, corpus 30/30 + HumanEval + test-survival), project-scale Arena TRACKED GAP
(0.20, 1/5), per the CCPA harness (claude-code-parity-apr-v1.yaml v1.32.0, 20 gates) +
the new aprender-side tracking pointer beat-claude-code-parity-v1.yaml; V1_004
model-family finding noted as the leading hypothesis for the gap. Earlier 2026-06-25
edit promoted the Ollama GPU-decode beat from TRACKING to ENFORCED (1.371× median,
412.3 vs 300.7 tok/s — **since withdrawn, see above**); asserted the evidence-backed
cold-start (~528–5000×), LAPACK-free ML (1.6–4.9×), cuda-oxide/SIMD (1.4–17×), and
footprint (15.8–17.1×) speed wins; replaced blanket-concession wording with specific,
sourced narrow-loss cases._
