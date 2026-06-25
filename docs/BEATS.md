# aprender BEAT Scoreboard

The four-pillar mission: **replace _and beat_** scikit-learn, PyTorch, Unsloth, and
Ollama/llama.cpp in one pure-Rust binary. A "beat" is never parity — it is a
**falsifiable, CI-gated benchmark** showing apr ≥ the incumbent on the incumbent's
own canonical task (PMAT-741).

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

## Speed scoreboard (evidence-backed, sourced)

apr **wins** speed in these measured, CI-gated cases — every row cites a `contracts/beat-*.yaml`
contract (status: `enforced`) or in-repo evidence:

| Speed win | Result | Source (contract / evidence) |
|-----------|--------|------------------------------|
| **GPU decode vs Ollama** (headline) | apr **1.371× faster** median (412.3 vs 300.7 tok/s; worst single run 1.230×), qwen2.5-coder-1.5b Q4_K_M, RTX 4090, same GGUF/host | `beat-ollama-decode-throughput-speed-v1.yaml` (ENFORCED, gate ≥1.10×) · `crates/aprender-serve/tests/beat_ollama_decode_throughput_speed.rs` |
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

apr **loses** speed in these **specific, narrow** cases (not a blanket concession):

| Speed loss (narrow case) | Result | Note |
|--------------------------|--------|------|
| **llama.cpp** single-request c=1 decode | llama.cpp ~1.55× faster (431 vs 277 tok/s, RTX 4090) | This is *llama.cpp*, not Ollama; apr **wins** the same-host steady-state decode vs Ollama 1.371× |
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
| **GPU decode throughput vs Ollama** (headline speed) | tok/s ratio (RTX-4090) | ✅ **WON** (PMAT-755) — apr **1.371× faster** median (apr median-of-7 **412.3** vs ollama **300.7** tok/s; worst single run 1.230×, best 1.523×), same qwen2.5-coder-1.5b Q4_K_M GGUF, same host. Gate = apr median-of-7 ≥ ollama × 1.10 (wide margin under 1.371×; ~0% bootstrapped flake rate) | manual/GPU gate (no NVIDIA CI runner) · `beat_ollama_decode_throughput_speed` · `beat-ollama-decode-throughput-speed-v1` |
| **Fail-closed correctness** (headline correctness) | broken-artifact classes rejected | ✅ **WON** — apr rejects **10/10** semantically-broken tensor classes (zero/NaN/Inf/L2~0/constant/shape) fail-closed; **llama.cpp accepts** the same (measured: zeroed-ffn GGUF → `apr validate` ✗ FAIL, `llama-cli` 0 errors + ran it) | CI `beat_fail_closed_garbage` · `apr-fail-closed-garbage-beat-v1` |
| **llama.cpp** single-request c=1 decode | tok/s ratio (RTX-4090) | ⚖️ **NARROW LOSS** — llama.cpp ~1.55× *faster* (431 vs 277 tok/s) at concurrency=1; this is *llama.cpp*, not Ollama, which apr beats 1.371× same-host | — |
| 7B-Q4K decode on GB10 Blackwell | tok/s | ⚖️ **NARROW LOSS** — ~12 tok/s (bandwidth-bound; DP4A path degraded on Blackwell) | — |
| Short-prompt one-shot wall-clock vs Ollama | end-to-end seconds | ⚖️ **NARROW LOSS** — apr CLI ~2.7–3.9 s fixed startup vs Ollama's resident daemon (separate from the steady-state decode beat above) | — |

**Headline speed beat — apr beats Ollama 1.371× on GPU decode (PMAT-755):** for the
*same* qwen2.5-coder-1.5b Q4_K_M GGUF on the *same* RTX 4090, apr's steady-state GPU
decode is **412.3 tok/s** (median of 7) vs Ollama's **300.7 tok/s** (median, tight
294–306 band) = **1.371× median**; every single apr run clears the **1.10×** enforced
gate (worst 369.9 = 1.230×, best 1.523×). This was promoted from TRACKING to an
**ENFORCED** beat once the ~1-in-6 decode-stall variance was fixed (#2049) and a
median-of-7 estimator brought the false-FAIL rate to ~0% — the contract is now the
source of truth (`contracts/beat-ollama-decode-throughput-speed-v1.yaml`, status
`enforced`). It stays a **manual/GPU gate** (`#[ignore]`, NVIDIA host only) because
there is no NVIDIA CI runner — same caveat as the cuda-oxide throughput gate. The
narrow losses above (llama.cpp at c=1, 7B-on-Blackwell, one-shot startup) are
**specific, named** cases, not a blanket "speed conceded" stance.

**Headline correctness beat — "we provably never ship garbage; they provably do":** a
model that *parses* but is *semantically* dead (all-zero / NaN / Inf weights) loads and
runs in llama.cpp/Ollama with exit 0 and no warning; apr's Poka-Yoke validation
(F-DATA-QUALITY-001..004) rejects it. Measured head-to-head 2026-06-13 in
`evidence/pillar4-fail-closed-2026-06-13/`. Correctness is a wedge none of the four
incumbents have — *and* apr now also wins the GPU decode-rate beat above.

**PMAT-742 (unblocked Pillar 4):** the default `apr run --gpu` was silently 8 tok/s — its
CUDA first-token parity gate false-rejected the correct fast path (a single BOS-only probe
diverges CPU-vs-GPU even when real generation is correct). Fixed by validating on the real
prompt context + near-tie tolerance → default `apr run` reaches the GPU fast path that the
re-measured 412.3 tok/s decode beat is built on.

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

_Last updated: 2026-06-25 — promoted the Ollama GPU-decode beat from TRACKING to
ENFORCED (1.371× median, 412.3 vs 300.7 tok/s); asserted the evidence-backed cold-start
(~528–5000×), LAPACK-free ML (1.6–4.9×), cuda-oxide/SIMD (1.4–17×), and footprint
(15.8–17.1×) speed wins; replaced blanket-concession wording with specific, sourced
narrow-loss cases._
