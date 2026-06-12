# BEAT Campaign — 10-Day Autonomous Schedule (2026-06-12)

Grounded day-by-day from the `beat-campaign-10day-schedule` workflow (wf_23a89a7f-b19,
8 agents). Companion to `campaign-ev-reprioritization-2026-06-12.md`. **Lane = BEATS-as-CI-
artifacts** across the 4 pillars + cuda-oxide, parallelized over 4 autonomous hosts.

**Friday crates.io cascades:** Day 1 = Fri 2026-06-12 (today; catch crates.io up 0.35.0→0.49.x)
and Day 8 = Fri 2026-06-19. GitHub release per increment.

---

## ⟳ RE-BASELINE — 2026-06-12 (evening): verified state + corrected plan

> Authored after a 6-agent state audit (`campaign-state-audit`, run `wf_1a21097f-335`)
> verified the true state of `origin/main` against the morning plan below. **Net: the
> campaign is ON TRACK — the highest-priority Day-1 prereq cleared itself, and an unplanned
> foundation-hardening pass bought the CI reliability the beat-gates structurally require —
> but the beat-infra _above the enum_ (validator rules, runner, gate, real contracts) is
> still essentially Day-1 work, and one foundation gap (the lint sweep has no PR) is now the
> top risk.** The day-by-day below this section is the original morning plan, kept for
> provenance; the corrected sequence here supersedes it.

### Chain of thought

1. **The morning plan's #1 PREREQ is already satisfied.** It assumed PMAT-741 lived only on
   `feat/beat-benchmark-contract-kind` (`main=7cd3f3626`) and that a Day-1 fast-track merge
   gated the whole campaign. Verified: `ContractKind::BeatBenchmark` (`kind.rs:63`) + the
   pilot `contracts/beat-sklearn-iris-v1.yaml` are **on origin/main** at **v0.49.0** (squash
   commit `85b64ee05`; git tag + GitHub release both exist and are ancestors of main). The
   morning Day-1 critical-path item is **DONE** and its rebase-on-feat-branch contingency is
   **moot** — reclaim that slack.

2. **An unplanned "foundation" pass consumed most of real Day 1 — and that was correct, not a
   detour.** The entire BEAT thesis is *falsifiable CI gates that hard-fail on regression*. A
   gate is only trustworthy if **RED means a real regression** — not a flaky proptest seed, a
   stranded un-linted crate failing `clippy -D warnings`, a duplicate crates.io `trueno`
   re-entering via a registry pin, or an Intel runner at 100% disk. So what landed today is a
   **precondition** for the campaign, not a distraction from it:
   - `#1975` **self-contained-DAG** monorepo (MERGED `e54c1a3e1`) — all 92 sibling deps are
     now `workspace=true` path aliases; a 16-crate dependency cycle broken → no
     published-sibling drift can poison a beat run.
   - **Flaky-test root-fixes** `#1988` (numerically-stable variance — kills the stddev
     translation-invariance flake) + `#1989` (proptest 1.11.0 float-sampler false
     `debug_assert` suppressed per-package + `INIT_SEED` determinism lock) → a green run now
     *means* green. **These ARE the campaign's single biggest CI-reliability lever.**
   - **§S consolidation** `#1982/1983/1984/1985` MERGED; `#1986/1987` in-flight (assumed to
     land per the user's "assuming these will all work").
   - **Intel disk-recycle** live (keeps runners online).

3. **But the beat-infra above the `ContractKind` enum is ~all still Day-1 work.** Verified
   gaps on main: **no** `validate_beat_benchmark()` rules (the variant falls through to the
   generic non-kernel branch in `validator.rs`); **no** validator_tests (only a serde
   round-trip + one happy-path "pilot validates"); **no** `apr beat-run` CLI; **no**
   `beat-gate.yml` workflow; only **one** beat contract, whose threshold is **hardcoded** in
   `beat_sklearn_iris.rs:25-27` (it never deserializes the YAML) — and that test **isn't even
   executed by CI** (it's an integration test absent from `ci.yml`'s integration list). We
   have the *vocabulary* for beats (the kind) but **zero CI-gated beats**.

4. **Per-pillar, the honest status is "precursors exist, gates don't":**
   - **P1 sklearn:** iris *accuracy*-parity test exists (not run by CI); matmul **1.78×** +
     matvec **1.44×** speed wins are real (`matrix.rs:163` ikj) but live only in commit
     messages / CHANGELOG — **no harness, no gate**; RF/KMeans/PCA speed-beats not started.
   - **P2 PyTorch:** autograd `finite_difference` util exists but proptests assert tol **0.2**,
     not the spec's **1e-3**, with no FALSIFY-GRADIENT contract; the PyTorch-CPU training beat
     has **no harness, no baseline CSV, no contract**.
   - **P3 Unsloth:** `QLoRALayer` (NF4) exists (`qlora.rs:22`) but is **not wired** into
     `InstructPipeline` (still `Vec<LoRALayer>` f32, `mod.rs:176`); `merge_export` has **no
     GGUF path**; `apr finetune` training is an explicit **stub** (`finetune.rs:735`).
   - **P4 Ollama:** **no contract, no workflow** — numbers are memory-only, and the **honest
     5-run median is ~1.32×, not 1.43×** (the higher figure is Ollama-baseline ±8% variance).
     `FUSION-004` DP4A (the 1.32→1.5× lever) is a PLANNED contract entry only. cuda-oxide on
     gx10: **GB10 GPU confirmed idle (0%)**, but LLVM-21 absent + `cargo-oxide` + nightly not
     installed (exactly the predicted Day-1/2 provisioning).

5. **crates.io is split, not uniformly behind.** `aprender-core` + `aprender-contracts` are
   already **0.49.0**; the facade + dependents (`apr-cli`, `aprender`, `aprender-mcp`,
   `-compute`, `-serve`, `-train`) are stranded at **0.41.0**, so `cargo install aprender`
   still yields 0.41.0. The Friday cascade is therefore **finish the partial 0.49.0 publish**
   (6 lagging crates, dependency order, `apr-cli` last, from a clean `origin/main`) — **not** a
   new version bump. (The morning intro's "0.35.0→0.49.x" is itself stale; it's a 0.41.0 split.)

6. **One NEW risk the morning plan didn't carry:** the **lint-harmonization sweep** (31
   crates, +595/−1933, real bugs fixed incl. `await_holding_lock` + two boolean-logic bugs) is
   **committed on `feat/apr-mono-self-containment` but has NO open PR** and is **not on main**.
   Now that those ~30 ex-repo crates are in the workspace graph, `clippy --all-targets -D
   warnings` (the `ci/lint` gate) will fail for them until this lands. **This is the top
   foundation risk and the first action below — ahead of any beat work.** (Land on a fresh
   uniquely-named branch; the original was already squash-merged as #1975, and reuse strands
   commits — see `feedback_checkout_b_branch_reuse_noop`.)

### Corrected day-by-day (window 2026-06-12 → 2026-06-21)

Legend: **[F]** = foundation closeout, **[I]** = beat-infra, **[P1..P4]** = pillar beat,
**[X]** = quant eval, **[R]** = release/cadence. Host tags as in the original.

**Day 1 — Fri 2026-06-12 (TODAY, largely spent — status)**
- ✅ [I] Day-1 prereq auto-cleared: PMAT-741 BeatBenchmark + pilot on main, v0.49.0 tagged + released.
- ✅ [F] Foundation hardened: `#1975` self-contained DAG MERGED; §S `#1982/83/84/85` MERGED; flaky root-fixes `#1988/#1989` armed; disk-recycle live.
- ⏳ Carried to Day 2: BeatBenchmark validator rules + tests; cuda-oxide gx10 provisioning.
- 🔴 Surfaced: lint-sweep has **no PR** (top risk).

**Day 2 — Sat 2026-06-13 — foundation closeout + first beat actually runs in CI**
- [F][cpu-ci] **Open + land the lint-harmonization PR** (fresh branch) → green `clippy -D warnings` for all ~30 ex-repo crates. *Gates everything; do first.*
- [F][cpu-ci] Land §S `#1986/#1987` (update-branch + auto-merge).
- [I][cpu-ci] `validate_beat_benchmark()` in `validator.rs` (incumbent ∈ four-pillar enum, metric present, numeric `beat_threshold`, direction) + 6 negative-case validator_tests.
- [I][cpu-ci] Make `beat_sklearn_iris` **contract-driven** (read `beat-sklearn-iris-v1.yaml`) **and add `--test beat_*` to `ci.yml`** so the gate actually executes → first beat live in CI.
- [gx10] cuda-oxide Spike1: provision LLVM-21 + nightly-2026-04-03 + `cargo-oxide` (GPU is idle, pre-authorized).

**Day 3 — Sun 2026-06-14 — beat-runner + beat-gate LIVE (first WON)**
- [I][rtx4090] `apr beat-run` CLI (clap variant + harness parsing the `beat` block + JSON + non-zero exit on regression) + 3-surface update (`apr-cli-commands-v1.yaml`, `cli_commands.rs`).
- [I][rtx4090] `beat-gate.yml` (consume the prebuilt workspace-test binary, ~2 min/beat) wired into `ci.yml`; verify with the deliberate-regression falsifier (force iris acc<0.92 → blocks merge). **iris accuracy = first WON.** [R] release v0.49.1.
- [gx10] cuda-oxide Spike2: saxpy `#[kernel]` → PTX gen + launch.

**Day 4 — Mon 2026-06-15 — Pillar-1 speed WON, autograd WON, cuda-oxide verdict**
- [P1][cpu-ci] PMAT-722 apr-vs-sklearn wall-clock harness (iris/digits/california, CSV+JSON) → capture matmul **1.78×** + matvec **1.44×** as hard gates → **Pillar-1 speed WON.** [R] v0.49.2.
- [P2][cpu-ci] PMAT-724: tighten finite-diff proptests to **<1e-3** + `FALSIFY-GRADIENT-CORRECTNESS` contract → **autograd WON** (gates the Day-5 training beat).
- [gx10] cuda-oxide Spike3: port `dequant_q4k`, parity <1e-4 vs hand-PTX → **GO/NO-GO** (routes the 1.5× closure to cuda-oxide vs hand-PTX DP4A).

**Day 5 — Tue 2026-06-16 — the marquee differentiator + QLoRA + honest Ollama pin**
- [P2][cpu-ci] PMAT-725/728 PyTorch-CPU training beat (2-layer MLP 1024→512→1, N=1024, pinned `pytorch_beat_baseline.csv`, wall ≤ PyTorch+20% **AND** MSE≤0.05) → **Pillar-2 marquee WON.** [R] v0.49.3.
- [P3][cpu-ci] PMAT-711: wire `QLoRALayer` NF4 into `InstructPipeline` + loss-monotone falsifier (16-sample/20-step, 4-bit ≤0.30× f16) → **Pillar-3 QLoRA WON.**
- [P4][rtx4090] **Re-measure the honest Ollama baseline first** (5-run p50 on Qwen2.5-Coder-1.5B Q4_K_M) → pin `beat-ollama-rtx4090-v1.yaml` at the **TRUE** number (~1.32×, not 1.43×) → **Pillar-4 CI-GATED-AT-CURRENT-NUMBER** (avoids an immediately-red gate).

**Day 6 — Wed 2026-06-17 — QLoRA pipeline + Pillar-1 3/4 + GPU baselines off idle Blackwell**
- [P3][cpu-ci] PMAT-712 LoRA→GGUF merge (add GGUF writer to `merge_export.rs`) + golden test (forward max-abs-diff <1e-2, LAYOUT-002 row-major).
- [P1][cpu-ci] RandomForest (digits, 5-run median, acc floor ≥0.95) + KMeans (seed-42, inertia parity) speed-beats.
- [P2/P3][gx10] PMAT-728-GPU Qwen3-370M Blackwell baseline + PMAT-715 Unsloth throughput, both **REPORT-ONLY** (pre-warm kernels for trueno#200).

**Day 7 — Thu 2026-06-18 — Unsloth single-command UX, Pillar-1 cascade COMPLETE, CRUX-E live**
- [P3][cpu-ci] PMAT-713 single-command `apr finetune --qlora --export gguf` (replace `execute_training_stub` + export TODOs) → **Unsloth-parity UX CPU WON.** [R] v0.49.4.
- [P1][cpu-ci] PCA speed-beat (20k→3 comp, ≥99.5% explained-variance) → completes the Pillar-1 4-beat cascade.
- [X][cpu-ci] CRUX-E-19 perplexity-per-bit quant-sweep (Q2_K..Q8_0 vs FP16, pinned WikiText-2 slice).

**Day 8 — Fri 2026-06-19 (FRIDAY) — crates.io 0.49.0 cascade FINISHED, CRUX-E complete, DP4A spike**
- [R][FRIDAY] **Finish the partial 0.49.0 cascade** (NOT a new bump): publish the 6 lagging crates in dep order `aprender-compute → -serve → -train → -mcp → aprender → apr-cli` (last) from a clean `origin/main`; run the pre-release gates (CB-510 include checks, `cargo deny`) + post-publish `cargo install aprender` → `apr --version` GO/NO-GO. → `cargo install aprender` finally yields 0.49.x.
- [X][cpu-ci] CRUX-E-20 KL-divergence (`apr diff --metric kl`, Gibbs + self-identity) + promote E-19/E-20 contracts to enforced.
- [P4][rtx4090] FUSION-004 DP4A INT8 spike (CoalescedGemvInt8Kernel + Q8_1 quant skeleton) toward 1.5×.

**Day 9 — Sat 2026-06-20 — single-source registry, v0.50.0, nightly Ollama soak**
- [I][cpu-ci] `beat-baselines.yaml` consolidated registry (single source of truth for all pinned baselines).
- [R][cpu-ci] v0.50.0 coordination + CHANGELOG (4 sklearn speed-beats + iris + PyTorch-CPU + QLoRA + 2 quant evals).
- [P4][rtx4090] 5-day nightly-beat-ollama variance soak (±5%); [gx10] finalize GPU report cards vs RTX-4090/Unsloth.

**Day 10 — Sun 2026-06-21 — campaign closed: ≥9 falsifiable beats CI-gated**
- [R][cpu-ci] Closeout README **BEAT scoreboard** (WON vs tracking, live CI ratios) + spec amendment; release v0.50.1.
- [P4][rtx4090] Final marquee verify (nightly green at the pinned Ollama number); file DP4A + cuda-oxide 1.5× closure as a **post-campaign operator sprint**.
- [R] Next forward crates.io bump (0.50.x) deferred to the following Friday (2026-06-26, outside window) unless release-worthy sooner.

### What changed vs the morning plan (summary)

| Axis | Morning plan | Corrected |
|---|---|---|
| Day-1 PMAT-741 merge | the gating prereq, do first | **DONE** (auto via §S squash leapfrog) |
| Real Day-1 spend | validator + beat-runner + cuda-oxide | **foundation hardening** (self-containment, flaky-kill, §S) |
| Top risk | PMAT-741 merge slips | **lint sweep has no PR** → `ci/lint` red for ~30 crates |
| First WON | iris (Day 2) | iris (Day 3, after the test is made to actually RUN in CI) |
| Ollama pin | 1.43× / 440 tok/s | **honest ~1.32× 5-run p50** (re-measure before pinning) |
| Friday cascade | "catch up 0.35→0.49" | **finish the 0.49.0 split-cascade** (6 lagging crates, not a bump) |
| Definition-of-done | unchanged | unchanged — ≥9 falsifiable beats CI-gated, 1.5× marquee out-of-window |

---

## Thesis

Convert aprender's proven wins into permanent, falsifiable, CI-gated BEATS plugged into PMAT-741 ContractKind::BeatBenchmark (shipped on feat/beat-benchmark-contract-kind, NOT yet on main, a Day-0 prereq: main=7cd3f3626, the enum and pilot YAML live only on the feat branch). Front-load credibility: by end of Day 2 the beat-runner CI infra is live and the two cheapest proven wins are pinned as regression gates (sklearn matmul 1.78x LinearRegression from commit 548a24032; the current Ollama 1.43x/440-tok-s decode measured on this RTX-4090 host). Parallelize across four free hosts for multiple beats per day. Honest by design: Pillar-1 sklearn beats, the Pillar-2 PyTorch-CPU beat, and the Ollama-1.43x gate are CI-GATED-AND-WON by Day 10; the Ollama-1.5x marquee and GPU throughput beats are CI-GATED-AT-CURRENT-NUMBER (1.43x pinned, tracking toward 1.5x) because closing them depends on the cuda-oxide GO/NO-GO and DP4A kernel authoring that exceed the window. GitHub release per increment; crates.io batches Fridays (Day 5, Day 10).

## Host allocation (4 concurrent autonomous hosts)

Four hosts run concurrently for 10 days. RTX-4090 (session host and the canonical Ollama-parity baseline where the 1.43x/440-tok-s numbers were measured; CUDA apr binary confirmed at /mnt/nvme-raid0/targets/aprender/release/apr): owns the Pillar-4 Ollama marquee track (Day 2-3 beat-runner binary plus beat-gate workflow, Day 3-4 rebuild-and-capture 1.43x plus the nightly-beat-ollama self-hosted GPU runner, Day 8 DP4A spike, Day 9-10 variance soak) and the variance-sensitive RandomForest beat (Day 5, 5-run-median). Only host where the Ollama gate is meaningful. gx10 (GB10 Blackwell, confirmed idle 0 percent util, LLVM none today): runs three GPU tracks off otherwise-wasted Blackwell capacity. Day 1-4 the cuda-oxide spike (LLVM-21 to saxpy to dequant_q4k port to GO/NO-GO) - the right host because cuda-oxide targets Blackwell, so the sm_89 blocker that killed it on RTX-4090 does not apply, only LLVM-21 remains. Day 6-9 after the spike frees it: Pillar-2 GPU PyTorch-CUDA baseline (PMAT-728-GPU Qwen3-370M) and Pillar-3 Unsloth throughput (PMAT-715), both REPORT-ONLY with kernel pre-warm for trueno#200. intel-clean-room CPU-CI: the workhorse, runs the entire Pillar-1 sklearn cascade (iris D2, LinearReg D3, KMeans D6, PCA D7), the Pillar-2 PyTorch-CPU differentiator (PMAT-724 D4, 725/728 D5), the Pillar-3 CPU QLoRA pipeline (711/712/713 D5-7), the CRUX-E quant evals (D7-8), and all contract/validator/registry plumbing. yoga (RTX-4060): reserve, holds the pinned Unsloth 6000-tok-s reference plus a 370M fallback if gx10 provisioning slips. Dependency spine: D1 PMAT-741-to-main gates everything; D2 beat-run binary gates the D3 beat-gate; D4 PMAT-724 gates the D5 PyTorch-CPU beat; D4 cuda-oxide verdict routes the D8 DP4A spike. The three host lanes are otherwise independent, enabling 2-4 beats per day.

## Day-by-day

### Day 1 — PMAT-741 on main, validator hardened, Blackwell toolchain provisioning started
- [cpu-ci] Merge feat/beat-benchmark-contract-kind to main (PMAT-741 enum and pilot beat-sklearn-iris-v1.yaml), release v0.49.0
- [cpu-ci] BeatBenchmark validator hardening plus 6 validator_tests (bad incumbent or missing beat_threshold fails)
- [gx10] Spike1: provision LLVM-21 and nightly-2026-04-03 (gx10 has none today), falsifier llvm-config >=21

### Day 2 — Beat-runner binary live, iris pilot fully contract-driven
- [rtx4090] apr beat-run CLI and core harness (parse BEAT-PILLAR-TASK, emit JSON, exit non-zero on regression, prebuilt binary dodges cold compile)
- [cpu-ci] Capture win 1: iris test reads thresholds from the deserialized contract (was hardcoded)
- [gx10] Spike2: cargo-oxide saxpy kernel to PTX gen and launch

### Day 3 — Beat-gate LIVE, iris and matmul regression-gated, Ollama ground-truth captured
- [rtx4090] beat-gate.yml per-PR fail-on-regression, wired into ci.yml (force iris acc<0.92 fails)
- [cpu-ci] Capture win 2: Pillar-1 LinearRegression 1.78x speed-beat (548a24032), release v0.49.1, WON
- [rtx4090] bg: rebuild apr cuda, confirm 1.43x over 50 runs
- [gx10] Spike3: port dequant_q4k to cuda-oxide, parity <1e-4 vs hand-PTX

### Day 4 — Ollama 1.43x CI-pinned, autograd gate green, cuda-oxide verdict in hand
- [rtx4090] beat-ollama-rtx4090 (460.7 baseline, ratio >=1.43) plus nightly p50 gate, release v0.49.2, CI-GATED-AT-CURRENT-NUMBER
- [cpu-ci] Pillar-2 PMAT-724 finite-diff autograd correctness, 1000+ cases per op, WON
- [gx10] Spike4: GO/NO-GO report, routes 1.43x to 1.5x closure to cuda-oxide vs DP4A, frees gx10

### Day 5 — FRIDAY. PyTorch-CPU training beat plus QLoRA plus first cascade, 5 beats gated
- [cpu-ci] Pillar-2 PMAT-725/728 PyTorch-CPU training beat (wall <= PyTorch+20 percent AND MSE<=0.05), release v0.49.3, the differentiator WON
- [cpu-ci] Pillar-3 PMAT-711 QLoRA loss-monotone gate, footprint <=0.30x f16, WON
- [rtx4090] Pillar-1 RandomForest speed-beat digits, 5-run-median, acc floor >=0.95
- [FRIDAY] crates.io batch 1 (v0.49.0-v0.49.3)

### Day 6 — QLoRA full CPU pipeline verifiable, GPU baselines off idle Blackwell, Pillar-1 3/4
- [cpu-ci] Pillar-3 PMAT-712 LoRA-to-GGUF merge golden-test (max-abs-diff <1e-2, LAYOUT-002 verified)
- [cpu-ci] Pillar-1 KMeans speed-beat (seed-42, wall_ms <= sklearn, inertia parity)
- [gx10] Pillar-2 PMAT-728-GPU Qwen3-370M Blackwell baseline, REPORT-ONLY
- [gx10] Pillar-3 PMAT-715 throughput vs Unsloth, pre-warm kernels for trueno#200, REPORT-ONLY

### Day 7 — Unsloth single-command UX, Pillar-1 sklearn cascade COMPLETE, CRUX-E harness live
- [cpu-ci] Pillar-3 PMAT-713 single-command apr finetune --qlora --export gguf, release v0.49.4, Unsloth-parity UX CPU WON
- [cpu-ci] CRUX-E-19 perplexity-per-bit quant-sweep (Q2_K..Q8_0 vs FP16, WikiText-2 pinned slice)
- [cpu-ci] Pillar-1 PCA speed-beat (20k to 3 comp, >=99.5 percent explained-variance), completes the 4-beat cascade

### Day 8 — CRUX-E complete, heterogeneous dispatch, Ollama-gap closure path scoped
- [cpu-ci] CRUX-E-20 KL-divergence apr diff --metric kl (Gibbs, self-identity) plus promote E-19/E-20 contracts active
- [cpu-ci] multi-beat CPU/GPU dispatch (GPU beats SKIP and exit0 when CUDA absent so CPU PRs stay green)
- [rtx4090] DP4A INT8 kernel spike (FUSION-004), CoalescedGemvInt8Kernel stub plus PTX compile-check toward 1.5x

### Day 9 — Single-source registry, v0.50.0, nightly Ollama gate stable a full work-week
- [cpu-ci] beat-baselines.yaml consolidated registry (single source of truth)
- [cpu-ci] Pillar-1 v0.50.0 release-coordination plus CHANGELOG (four speed-beats plus iris)
- [rtx4090] 5-day soak verify nightly-beat-ollama variance within +/-5 percent
- [gx10] finalize Pillar-2/3 GPU report cards vs RTX-4090 and Unsloth

### Day 10 — Campaign closed: 9+ falsifiable beats CI-gated, two Friday cascades, release per increment
- [cpu-ci] closeout README BEAT scoreboard (WON vs tracking with live ratios) plus spec amendment, release v0.50.1
- [FRIDAY] crates.io batch 2 (v0.49.4 plus v0.50.0 plus v0.50.1) plus post-publish dogfood GO
- [rtx4090] final marquee verify nightly green at 1.43x, DP4A/cuda-oxide closure filed as post-campaign sprint

## Definition of done

10 days successful = beat-runner CI infra live on main plus at least 9 falsifiable beats wired into ContractKind::BeatBenchmark gating every PR, with two Friday crates.io cascades and a GitHub release per increment. CI-GATED-AND-WON by Day 10 (apr provably greater-or-equal incumbent, hard-fails CI on regression): Pillar-1 beat-sklearn-iris (RandomForest accuracy >=0.92 vs sklearn 0.94-0.96 floor, already green, contract-driven D2); Pillar-1 linreg-speed (proven 1.78x matmul-ikj, D3); Pillar-1 kmeans/randomforest-plus-acc-floor/pca speed-beats (D5-7, v0.50.0); Pillar-2 FALSIFY-GRADIENT-CORRECTNESS autograd gate (D4); Pillar-2 beat-pytorch-cpu-training (apr wall-clock <= pinned PyTorch-CPU +20 percent AND MSE <=0.05, the marquee differentiator, D5); Pillar-3 FALSIFY-QLORA-001 plus LORA-MERGE-001 plus E2E-FINETUNE-001 CPU path (D5-7); CRUX-E-19 PPL-per-bit plus CRUX-E-20 KL with Gibbs and self-identity (D7-8). CI-GATED-AT-CURRENT-NUMBER (pinned baseline, fails CI on regression below the current number, tracking the target, NOT yet beating the stretch goal): Pillar-4 beat-ollama-rtx4090 pinned at 1.43x (>=440 tok/s, p50-over-5-runs) on the canonical host - the 1.5x marquee is NOT closed in-window (it depends on the cuda-oxide GO/NO-GO plus 48-60h of DP4A kernel authoring, filed as a post-campaign operator sprint); Pillar-2 GPU (PMAT-728-GPU) and Pillar-3 GPU (PMAT-715) are REPORT-ONLY gates (no hard CI fail) because Blackwell GPU training is experimental (trueno#200 JIT risk), emitting a tracking ratio every run. The campaign is CI-gated end-to-end when a throwaway commit that regresses ANY won beat (force iris acc<0.92, or matmul slower than 1.78x, or PyTorch-CPU MSE>0.05) FAILS beat-gate and blocks merge, verified by the deliberate-regression falsifier on Day 3. The Day-10 README BEAT scoreboard publicly labels each beat WON vs tracking with its live CI ratio.

## Risks & contingency

PREREQ (highest): PMAT-741 is on feat/beat-benchmark-contract-kind, NOT main (verified main=7cd3f3626; the enum plus pilot YAML live only on the feat branch). If the Day-1 merge slips the whole campaign stalls. Contingency: Day-1 fast-track merge is the first action; if CI blocks, rebase the beat-runner work on the feat branch and merge as one. Branch-reuse footgun from memory: use unique branch names plus post-merge git show origin/main:FILE for each load-bearing change (the matmul beat was dropped once by a squash branch-reuse, re-landed 8235f90ec). COLD-COMPILE TIMEOUT: beat-gate invoking cargo test triggers a ~40-min cold workspace compile (879+ crates). Contingency (baked into Day 2): consume the PRE-BUILT test binary from the workspace-test artifact and invoke it directly (~2min/beat). CUDA-OXIDE / LLVM-21 (blocks the 1.5x closure, NOT the campaign): gx10 has no LLVM-21 today (confirmed). If provisioning fails or cuda-oxide emits invalid sm_121a PTX, the spike returns NO-GO. Designed non-blocking: the Ollama beat ships CI-GATED-AT-1.43x regardless; NO-GO simply routes closure to the hand-PTX DP4A plus cuBLAS path (PMAT-715, blackwell-backend-fix-spec.md) as the documented post-campaign sprint. BLACKWELL JIT HANG (trueno#200): GPU training beats can crash on on-demand backward-kernel JIT. Contingency: pre-warm all backward kernel variants before the training loop (PMAT-698 recipe) and fall back to cuBLAS; ship these as REPORT-ONLY so a JIT failure degrades to no-number-this-run rather than campaign-red. PERF VARIANCE / CI FLAKE: RandomForest bootstrap-plus-rayon timing and RTX-4090 nightly throughput both jitter +/-5 percent. Contingency: 5-run-median plus warm-up plus p50 statistical gates (Days 5, 9), with tolerance bands anchored to a commit SHA not floating sklearn/PyTorch versions. INFRA: concurrent autonomous sessions share the git index (commit from /tmp worktree only); self-hosted runners recur disk-full or post-hiatus deregistration with standard runbooks. Auto-merge every green PR; on behind mergeable_state run gh pr update-branch. SCHEDULE SLACK: summed critical-path effort per lane is ~3-4 days, all overlapping, so the 10-day window carries 3-6 days of slack per lane even with the GPU lane fully deferred.

## cuda-oxide — COMMITTED (operator-approved 2026-06-12)

The gx10 GB10 Blackwell lane runs the cuda-oxide spike→adopt (Days 1-4): provision LLVM-21
+ nightly-2026-04-03 → `cargo install cargo-oxide` → saxpy `#[kernel]` PTX gen+launch →
port `dequant_q4k` to pure-Rust→PTX (parity <1e-4 vs hand-PTX). Blackwell IS cuda-oxide's
target HW, so the Ada/sm_89 blocker is gone; only LLVM-21 provisioning remains. Strategic:
pure-Rust GPU kernels (north-star) + escape the hand-PTX `trueno-cuda-edge` Blackwell JIT pain.
The Day-4 GO routes the Ollama 1.43x→1.5x closure to cuda-oxide vs hand-PTX DP4A. Runs in
PARALLEL with the 4090 Ollama lane — not a dependency of the inference beat.