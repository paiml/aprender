# aprender BEAT Scoreboard

The four-pillar mission: **replace _and beat_** scikit-learn, PyTorch, Unsloth, and
Ollama/llama.cpp in one pure-Rust binary. A "beat" is never parity — it is a
**falsifiable, CI-gated benchmark** showing apr ≥ the incumbent on the incumbent's
own canonical task (PMAT-741).

> **Honesty rule:** a beat that apr fails never ships. Where apr structurally can't
> win, we **CONCEDE** in the open rather than publish a false claim.

## Status legend

| Status | Meaning |
|--------|---------|
| ✅ **WON** | CI-gated; apr ≥ incumbent; a regression hard-fails the gate |
| 📊 **TRACKING** | measured, pinned baseline; gate not yet wired (or stretch target) |
| 🚧 **PLANNED** | identified; not yet built |
| ⚖️ **CONCEDED** | measured; apr currently loses — an optimization target, not a beat |

## Pillar 1 — scikit-learn

| Beat | Metric | Result | Gate |
|------|--------|--------|------|
| Iris classification (RandomForest) | test accuracy | ✅ **WON** — apr 0.94 ≥ sklearn floor 0.94 (threshold 0.92) | per-PR `ci.yml` · `beat_sklearn_iris` |
| LinearRegression fit+predict | wall-clock ratio | ✅ **WON** — apr **2.0× faster** (ratio 0.50, gate ≤ 0.90) | nightly · `beat_sklearn_linreg_speed` |
| PCA fit_transform | wall-clock ratio | ⚖️ **CONCEDED** — apr 1.79× *slower* (sklearn is LAPACK-SVD-bound) | — |
| Ridge fit+predict | wall-clock ratio | ⚖️ **CONCEDED** — apr 1.52× *slower* (sklearn Ridge defaults to fast Cholesky; the LinReg-SVD win doesn't transfer) | — |
| Lasso fit+predict | wall-clock ratio | ⚖️ **CONCEDED** — apr 19× *slower* (apr coordinate-descent unoptimized) | — |
| KMeans fit+predict | wall-clock ratio | ⚖️ **CONCEDED** — apr 1.02× *slower* (tied; both Lloyd) | — |

**Why apr wins / loses sklearn:** apr's wedge is the cache-friendly ikj SIMD matmul,
so it wins **matmul-bound** tasks (normal-equations regression) and concedes
**LAPACK/SVD-bound** ones (PCA) until apr's decomposition is optimized. See
`memory project_sklearn_speed_beat_selection`.

## Pillar 2 — PyTorch

| Beat | Metric | Result | Gate |
|------|--------|--------|------|
| 2-layer MLP training time | wall-clock ratio + MSE ≤ 0.05 | ⚖️ **CONCEDED** (PMAT-725) — apr ~11× *slower* (PyTorch MKL + fused autograd; apr training is correct after #2000 but autograd Tensor ops don't use the SIMD Matrix path) | — |
| Autograd gradient correctness | analytic vs finite-diff | 📊 **TRACKING** (needs relative-tolerance gate) | — |

## Pillar 3 — Unsloth

| Beat | Metric | Result | Gate |
|------|--------|--------|------|
| QLoRA fine-tune | loss-monotone + 4-bit footprint ≤ 0.30× f16 | 🚧 **PLANNED** (PMAT-711) | — |
| LoRA→GGUF merge | forward max-abs-diff < 1e-2 | 🚧 **PLANNED** (PMAT-712) | — |

## Pillar 4 — Ollama / llama.cpp

| Beat | Metric | Result | Gate |
|------|--------|--------|------|
| **Fail-closed correctness** (headline) | broken-artifact classes rejected | ✅ **WON** — apr rejects **10/10** semantically-broken tensor classes (zero/NaN/Inf/L2~0/constant/shape) fail-closed; **llama.cpp accepts** the same (measured: zeroed-ffn GGUF → `apr validate` ✗ FAIL, `llama-cli` 0 errors + ran it) | CI `beat_fail_closed_garbage` · `apr-fail-closed-garbage-beat-v1` |
| Decode throughput | tok/s ratio (RTX-4090) | 📊 **TRACKING** — apr **1.23× faster** (apr ~405 vs ollama 330 tok/s, same qwen2.5-coder-1.5b Q4_K_M GGUF, steady-state decode; `apr qa` Ollama-Parity 1.37) | — |

**Headline beat — "we provably never ship garbage; they provably do":** a model that
*parses* but is *semantically* dead (all-zero / NaN / Inf weights) loads and runs in
llama.cpp/Ollama with exit 0 and no warning; apr's Poka-Yoke validation
(F-DATA-QUALITY-001..004) rejects it. Measured head-to-head 2026-06-13 in
`evidence/pillar4-fail-closed-2026-06-13/`. This is apr's structural win where it
concedes raw decode speed — correctness is the wedge none of the four incumbents have.

**PMAT-742 (unblocks Pillar 4):** the default `apr run --gpu` was silently 8 tok/s — its
CUDA first-token parity gate false-rejected the correct fast path (a single BOS-only probe
diverges CPU-vs-GPU even when real generation is correct). Fixed by validating on the real
prompt context + near-tie tolerance → default `apr run` now reaches the ~405 tok/s GPU path.
*Caveat:* this is decode throughput; apr's one-shot CLI has ~2.7s startup that ollama's daemon
avoids, so short-prompt wall-clock still favors ollama until `apr serve` (warm) is benched.

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
3. **Measure before claiming.** Concede honestly where apr can't (yet) win.

_Last updated: 2026-06-13 (workspace v0.49.1)._
