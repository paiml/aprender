# Campaign EV Re-Prioritization — 2026-06-12

**Trigger:** the 10-day autonomous release campaign had shipped ~34 sklearn-**parity**
features (v0.44.0→v0.48.6: metrics, encoders, NB variants, Estimator impls). The
operator flagged that the goal is **highest-EV features across the ENTIRE roadmap**,
not sklearn-parity breadth. A grounded 7-agent cross-capability EV survey
(`cross-capability-ev-survey`, run wf_4bdcad93-664) re-scored the whole roadmap.

## Chain of thought

1. **The mission** is *replace **AND BEAT** sklearn / PyTorch / Unsloth / Ollama in one
   pure-Rust binary*, where **beat = a committed, CI-wired, falsifiable benchmark**
   (PMAT-717/718/719 + PMAT-741).
2. **What we'd been doing**: sklearn *parity* breadth — the *replace* half. Marginal EV
   has collapsed; the 14th metric moves the mission needle by epsilon.
3. **EV under campaign constraints** (unattended, CPU-only, no operator):
   `EV ≈ (impact × falsifiability × autonomy) / effort`. A GPU-gated item scores low
   for the *autonomous* track regardless of impact — it can't be verified unattended.
4. **Untapped EV concentrates in the BEAT half + the beat-benchmark infrastructure**,
   which **does not exist today**: `ContractKind` has 11 variants and no `BeatBenchmark`;
   there is exactly one beat test (`beat_sklearn_iris.rs`, 74 LOC, hardcoded
   `SKLEARN_IRIS_FLOOR=0.94`); zero machine-pinned baselines.
5. **The shift**: *stop manufacturing parity, start manufacturing BEATS-as-CI-artifacts.*
   Three-step force multiplier: (1) build the `BeatBenchmark` contract kind + CI runner;
   (2) plug in the CPU-autonomous beats (PyTorch-CPU training, sklearn speed); (3) quant-QA
   evals that fortify the Ollama moat. Defer GPU beats to an operator track.

## EV ranking (grounded)

| Rank | Capability | Action | EV | Autonomy |
|---|---|---|---|---|
| 1 | **PMAT-741** beat-benchmark infra | `BeatBenchmark` variant in `ContractKind` (`crates/aprender-contracts/src/schema/kind.rs`) + validator (incumbent_oracle, canonical_task, baseline+date, beat_threshold, ci_gate_name, approved_compute) + CI runner emitting `BEAT-{PILLAR}-{TASK}` ratios, non-zero exit on regression. Pilot: migrate `beat_sklearn_iris.rs` → `contracts/beat-baselines.yaml`. | 92 | 5 (CPU) |
| 2 | **PMAT-725** PyTorch-CPU training beat | Fixed 2-layer MLP (1024→512→1), N=1024 deterministic synthetic, SGD lr=0.01/1000 steps, wall-clock + peak RSS, MSE≤0.05 gate, `benchmarks/pytorch_beat_baseline.csv` + PMAT-724 finite-diff gradient gate. | 88 | 5 (CPU) |
| 3 | **PMAT-722** sklearn speed-beats | apr-vs-sklearn wall-clock fit+predict harness (iris/digits/california), CSV+JSON, CI-fails on regression; capture the real matmul 1.78× + matvec 1.44× wins + RF/KMeans/PCA gates. | 85 | 5 (CPU) |
| 4 | **CRUX-E-19/E-20** quant evals | perplexity-per-bit-budget + KL-divergence-vs-FP16 (`apr qa --bench perplexity --quant-sweep`, `apr diff --metric kl`), cached WikiText-2, CI-deterministic. | 78 | 5 (CPU) |
| 5 | **PMAT-711** Unsloth QLoRA (CPU half) | Integrate existing `QLoRALayer`(NF4) into `InstructPipeline.lora_layers` + loss-monotone falsifier (16-sample, 20 steps, 4-bit ≤0.30× f16). Unblocks 712/713. | 72 | 4 (CPU) |
| 6 | **FUSION-004** Ollama 1.5× | INT8 DP4A CUDA kernel, 396→460.7 tok/s RTX-4090. | 34 | **2 (GPU)** |

## Tracks

**Autonomous (ship unattended, CPU/CI):** PMAT-741 → 741-pilot → PMAT-725(+724) →
PMAT-722 → CRUX-E-19 → CRUX-E-20 → PMAT-711.

**GPU-gated (surface for operator scheduling):** FUSION-004 (1.5× decode);
PMAT-715 (Unsloth throughput beat, after 711/712/713); PMAT-728-GPU (PyTorch-GPU
baseline, after the CPU half lands); Pillar-1 GPU reference timing where needed.

**Rule going forward:** never ship a parity feature when a falsifiable *beat* is on the
table. See [[project_10day_autonomous_release_campaign]] and the four-pillar mission.
