# APR-VERIFY-001: Pipeline Verification & Visualization System

**Status**: Phase 1 IMPLEMENTED (Core Infrastructure + Whisper.apr Case Study)
**Author**: Aprender Team
**Date**: 2025-12-16
**PMAT Ticket**: APR-VERIFY-001

---

## Executive Summary

A deterministic, visual pipeline verification system for ML model debugging that combines:
- **Pixar/Weta-style stage gates** with ground truth comparison
- **Probar-style TUI testing** with pixel-perfect reproducibility
- **Toyota Production System** quality methodology
- **Popperian falsification** for rigorous validation

This differentiates from existing MLOps tools (MLflow, Weights & Biases, Neptune) by focusing on **deterministic debugging** and **differential testing** rather than experiment tracking.

---

## ⚠️ CRITICAL CONSTRAINTS

### NO PYTHON - RUST ONLY

**This project is 100% Rust. Python is NEVER used.**

- ❌ **NEVER** create Python scripts for ground truth extraction
- ❌ **NEVER** use `uv run`, `pip`, `torch`, `transformers`, or any Python tooling
- ❌ **NEVER** suggest "just use Python for this one thing"
- ✅ **ALWAYS** use Rust for all tooling, extraction, and verification
- ✅ **ALWAYS** use existing JSON/binary ground truth files already extracted
- ✅ **ALWAYS** load reference data via `aprender::verify::GroundTruth::from_json_file()`

**Ground Truth Workflow (Rust-Only):**
1. Reference values are pre-extracted and stored in `test_data/*.json` or `golden_traces/*.bin`
2. Rust code loads these via `GroundTruth::from_json_file()` or `GroundTruth::from_bin_file()`
3. If new ground truth is needed, create a Rust example that calls the reference implementation
4. The aprender crate has `safetensors-compare` feature for loading reference weights

**Why No Python:**
- WASM-first architecture requires pure Rust
- Deterministic builds require single-language toolchain
- probar/jugar test ecosystem is Rust-native
- Avoids "works on my machine" Python environment issues

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Literature Review](#2-literature-review)
3. [System Architecture](#3-system-architecture)
4. [Integration Points](#4-integration-points)
5. [API Design](#5-api-design)
6. [TUI Specification](#6-tui-specification)
7. [Playbook System](#7-playbook-system)
8. [Toyota Way Implementation](#8-toyota-way-implementation)
9. [Popperian Falsification QA](#9-popperian-falsification-qa)
10. [Implementation Phases](#10-implementation-phases)

---

## 1. Problem Statement

### 1.1 Current State

ML model debugging is typically ad-hoc:
- `print()` statements scattered through code
- Manual comparison of tensor shapes
- No systematic ground truth comparison
- "Works on my machine" syndrome

### 1.2 The Whisper.apr Case Study

During WAPR-TRANS-001 debugging, we discovered:
```
Step C (Mel): Our mean=+0.1841, Ground truth=-0.2150
             → 89.4% divergence, SIGN IS FLIPPED
```

This took 20+ hours of ad-hoc debugging. With pipeline verification:
- Automated detection in <1 minute
- Visual indication of exact failure point
- Ground truth comparison at each stage

### 1.3 Gap Analysis

| Existing Tools | What They Do | What's Missing |
|----------------|--------------|----------------|
| MLflow | Experiment tracking | No stage-by-stage verification |
| W&B | Metric visualization | No ground truth comparison |
| TensorBoard | Training curves | No inference pipeline debugging |
| pytest | Unit tests | No visual pipeline state |

### 1.4 Our Solution

**aprender-verify**: Deterministic pipeline verification with:
- Stage-by-stage ground truth comparison
- Visual TUI (probar-style)
- Pixel-perfect reproducibility
- Playbook-driven test scenarios
- Statistical distribution verification (KL Divergence, Wasserstein)

---

## 2. Literature Review

### 2.1 Foundational Citations (Original)

1. **Amershi, S., et al. (2019).** "Software Engineering for Machine Learning: A Case Study." *ICSE*.
2. **Breck, E., et al. (2017).** "The ML Test Score: A Rubric for ML Production Readiness." *IEEE Big Data*.
3. **Sculley, D., et al. (2015).** "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*.
4. **Polyzotis, N., et al. (2018).** "Data Validation for Machine Learning." *MLSys*.
5. **Baylor, D., et al. (2017).** "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform." *KDD*.
6. **Ratner, A., et al. (2019).** "MLSys: The New Frontier of Machine Learning Systems." *arXiv*.
7. **Paleyes, A., et al. (2022).** "Challenges in Deploying Machine Learning: A Survey." *ACM Computing Surveys*.
8. **Shankar, S., et al. (2022).** "Operationalizing Machine Learning: An Interview Study." *arXiv*.
9. **Lwakatare, L.E., et al. (2019).** "A Taxonomy of Software Engineering Challenges for ML Systems." *ESEM*.
10. **Xin, D., et al. (2021).** "Production Machine Learning Pipelines." *SIGMOD*.

### 2.2 Enhanced Citations (New Support)

11. **Zhang, J. M., et al. (2020).** "Machine Learning Testing: Survey, Landscapes and Horizons." *IEEE Transactions on Software Engineering*.
    - *Relevance*: Provides the taxonomy for "differential testing" which underpins our Ground Truth comparison strategy.

12. **Pei, K., et al. (2017).** "DeepXplore: Automated Whitebox Testing of Deep Learning Systems." *SOSP*.
    - *Relevance*: Introduces "neuron coverage" concepts, supporting our coverage integration (Section 4.3).

13. **Ma, L., et al. (2018).** "DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems." *ASE*.
    - *Relevance*: Justifies the need for multi-level verification (stage-level vs neuron-level).

14. **Odena, A., et al. (2019).** "TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing." *ICML*.
    - *Relevance*: Supports our "Fuzzing Integration" in Playbooks (Section 7.4).

15. **Tian, Y., et al. (2018).** "DeepTest: Automated Testing of Deep-Neural-Network-driven Autonomous Cars." *ICSE*.
    - *Relevance*: Demonstrates the value of metamorphic testing relations for pipeline validation.

16. **Hohman, F., et al. (2019).** "Gamut: A Design Probe for Model Performance Interpretation." *CHI*.
    - *Relevance*: Supports the design of our Visual TUI for interpretability (Section 6).

17. **Wexler, J., et al. (2019).** "The What-If Tool: Interactive Probing of Machine Learning Models." *IEEE TVCG*.
    - *Relevance*: Influences our "Playbook" design for counterfactual/scenario testing.

18. **Sato, D., et al. (2019).** "Continuous Delivery for Machine Learning: Patterns and Best Practices." *Sato et al*.
    - *Relevance*: Validates our CI/CD integration approach (Section 4.4).

19. **Schelter, S., et al. (2018).** "Automating Large-Scale Data Quality Verification." *VLDB*.
    - *Relevance*: Provides the mathematical basis for our statistical checks (KL Divergence/Wasserstein).

20. **Renggli, C., et al. (2019).** "Continuous Integration of Machine Learning Models with Ease.ml/ci." *SysML*.
    - *Relevance*: Supports the "Quality Gates" concept in PMAT integration.

---

## 3. System Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        APRENDER-VERIFY SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Pipeline  │    │   Ground    │    │    Delta    │    │     TUI     │  │
│  │  Definition │───▶│    Truth    │───▶│   Compute   │───▶│   Render    │  │
│  │   (YAML)    │    │   Loader    │    │   Engine    │    │  (ratatui)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Stage     │    │   Golden    │    │  Statistics │    │   Pixel     │  │
│  │  Executor   │    │   Traces    │    │   Report    │    │   Tests     │  │
│  │             │    │  (.bin/.json)│    │             │    │  (probar)   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┴──────────────────┴──────────────────┘          │
│                                    │                                        │
│                                    ▼                                        │
│                          ┌─────────────────┐                                │
│                          │    Playbook     │                                │
│                          │    Engine       │                                │
│                          │   (scenarios)   │                                │
│                          └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
Input Audio                    Pipeline Stages                      Output
    │                                                                  │
    ▼                                                                  ▼
┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
│ Audio │──▶│  Mel  │──▶│ Conv  │──▶│ Enc   │──▶│ Dec   │──▶│ Text  │
│ 16kHz │   │ 80bin │   │ Front │   │ Blocks│   │ Blocks│   │Output │
└───────┘   └───────┘   └───────┘   └───────┘   └───────┘   └───────┘
    │           │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼           ▼
┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
│ GT:A  │   │ GT:C  │   │ GT:D  │   │ GT:G  │   │ GT:I  │   │ GT:M  │
│.bin   │   │.bin   │   │.bin   │   │.bin   │   │.bin   │   │.txt   │
└───────┘   └───────┘   └───────┘   └───────┘   └───────┘   └───────┘
    │           │           │           │           │           │
    └───────────┴───────────┴───────────┴───────────┴───────────┘
```

### 3.4 Statistical Divergence Engine

Beyond simple mean/std comparison, the engine supports advanced distribution metrics (Schelter et al. 2018):

- **KL Divergence**: For comparing probability distributions (e.g., softmax outputs).
- **Wasserstein Distance**: For comparing geometric distributions (e.g., embeddings).
- **Cosine Similarity**: For high-dimensional vector alignment.

```rust
pub enum Metric {
    MeanSquaredError,
    CosineSimilarity,
    KLDivergence,
    WassersteinDistance,
    L2Norm,
}
```

---

## 4. Integration Points

### 4.1 Probar Integration

```yaml
# probar.yaml
name: aprender-verify-tui
tests:
  - name: pipeline_verification_display
    command: cargo run --example verify_whisper
    snapshots:
      - name: initial_state
        wait_for: "PIPELINE VERIFICATION"
        capture: full_screen
      - name: mel_failure
        wait_for: "Step C"
        capture: region(0, 5, 80, 1)
        assert_contains: "✗"
```

### 4.2 APR-CLI Integration

```bash
apr verify pipeline.yaml              # Run verification
apr verify --tui pipeline.yaml        # Interactive TUI mode
apr verify --fuzz --duration 60s      # Run fuzzing mode
```

---

## 5. API Design

### 5.1 Pipeline Builder API

```rust
use aprender::verify::{Pipeline, Stage, GroundTruth, Tolerance};

// Declarative pipeline definition
let pipeline = Pipeline::builder("whisper-tiny")
    // Stage C: Mel Spectrogram
    .stage("mel")
        .input_type::<AudioSamples>()
        .output_type::<MelSpectrogram>()
        .transform(|x| mel_spectrogram(x, n_mels=80))
        .ground_truth(GroundTruth::from_file("golden/step_c.bin"))
        .tolerance(Tolerance::Stats {
            mean_delta: 0.05,
            std_delta: 0.05,
            kl_div: 0.01, // Advanced metric
        })
    .build()?;
```

---

## 6. TUI Specification

### 6.1 Layout

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    APRENDER PIPELINE VERIFICATION                            ║
║                    Model: whisper-tiny | Input: test.wav                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ PIPELINE STAGES                                                         │ ║
║  ├─────┬───────────────┬────────┬──────────────┬──────────────┬───────────┤ ║
║  │ # │ Stage         │ Status │ Our Value    │ Ground Truth │ Delta     │ ║
║  ├─────┼───────────────┼────────┼──────────────┼──────────────┼───────────┤ ║
║  │ A │ audio         │   ✓    │ μ=+0.001     │ μ=+0.001     │   0.1%    │ ║
║  │ B │ stft          │   ✓    │ μ=+0.234     │ μ=+0.235     │   0.4%    │ ║
║  │ C │ mel           │   ✗    │ μ=+0.184     │ μ=-0.215     │  89.4%    │ ║
║  └─────┴───────────────┴────────┴──────────────┴──────────────┴───────────┘ ║
```

### 6.5 Latency Timeline View (New)

Inspired by trace visualizers (Sato et al. 2019):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TIMELINE                                                                    │
│                                                                             │
│ audio   [==] 2ms                                                            │
│ mel     [=====] 5ms                                                         │
│ encoder [==================================================] 50ms           │
│ decoder [=========================] 25ms                                    │
│                                                                             │
│ Total: 82ms (Budget: 100ms)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Playbook System

### 7.1 Playbook Format

```yaml
# playbooks/whisper_verification.yaml
name: Whisper Pipeline Verification
scenarios:
  - name: short_audio
    input: test_audio/speech_1.5s.wav
    expected_output: "The birds can fly"
```

### 7.4 Fuzzing Integration (New)

Leveraging `proptest` strategies (Odena et al. 2019):

```yaml
# playbooks/fuzz_verification.yaml
name: Fuzz Testing
strategies:
  - name: audio_noise_injection
    base_input: test_audio/clean.wav
    mutation: gaussian_noise(std=0.1)
    iterations: 100
    assertions:
      - no_panic
      - output_length_within(0.5, 2.0)
```

---

## 8. Toyota Way Implementation

### 8.1 Principles Applied

| Toyota Principle | Implementation |
|------------------|----------------|
| **Jidoka** (Stop the Line) | Pipeline halts at first failure, downstream stages blocked |
| **Genchi Genbutsu** (Go See) | Visual TUI shows actual data vs expected |
| **Kaizen** (Continuous Improvement) | Delta tracking over time, regression detection |
| **Andon** (Visual Signal) | ✓/✗/○ status indicators with color coding |

### 8.3 5 Whys Diagnosis

```rust
impl StageResult {
    pub fn diagnose(&self) -> Vec<String> {
        let mut whys = Vec::new();
        // Why 1: What failed?
        whys.push(format!("Stage '{}' failed with delta {:.1}%", self.name, self.delta.percent()));
        // Why 2: Sign flip?
        if self.our_mean.signum() != self.gt_mean.signum() {
            whys.push("Sign is FLIPPED (positive vs negative)".into());
            whys.push("Likely cause: Normalization formula error".into());
        }
        whys
    }
}
```

---

## 9. Popperian Falsification QA

### 9.2 100-Point QA Checklist

#### Section A: Pipeline Definition (20 points)
(Standard checks A01-A10)

#### Section B: Ground Truth (20 points)
(Standard checks B01-B10)

#### Section C: Delta Computation (20 points)

| # | Test | Falsification Target | Points |
|---|------|---------------------|--------|
| C01 | Mean delta correct | Manual calculation matches | 2 |
| C02 | KL Divergence correct | Compare two distributions | 2 |
| C03 | Wasserstein correct | Compare geometric shifts | 2 |
| ... | ... | ... | ... |

#### Section F: Fuzzing & Latency (New - 20 Points)

| # | Test | Falsification Target | Points |
|---|------|---------------------|--------|
| F01 | Fuzzing generates inputs | Inputs differ each iteration | 2 |
| F02 | Panics caught | System survives panic in stage | 2 |
| F03 | Latency captured | Timing > 0ms | 2 |
| F04 | Latency budget enforced | Fail if time > budget | 2 |
| F05 | Timeline renders | Visual width matches time | 2 |
| F06 | Noise injection works | SNR decreases | 2 |
| F07 | Fuzz report generated | Stats on failures | 2 |
| F08 | Strategy randomization | Seeds produce different results | 2 |
| F09 | Deterministic replay | Same seed = same result | 2 |
| F10 | Resource limits | Fuzzing respects RAM limit | 2 |

---

## 10. Implementation Phases

### Phase 1: Core Infrastructure (Sprint 1)
- Pipeline, Stage, GroundTruth, Basic Delta

### Phase 2: TUI Visualization (Sprint 2)
- Ratatui TUI, Timeline View

### Phase 3: Playbook & Fuzzing (Sprint 3)
- Playbook Engine, Proptest Integration

### Phase 4: Integration (Sprint 4)
- APR-CLI, PMAT, Documentation

---

## 11. Case Study: Whisper.apr Mel Spectrogram Fix

### 11.1 Problem Discovery

During WAPR-TRANS-001 debugging, APR-VERIFY identified a critical issue:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  │  A  │ audio           │   ✓    │ μ=+0.0005 σ=0.0711 │ μ=+0.0002 σ=0.0696 │   2.5% │
║  │  B  │ mel             │   ✗    │ μ=+0.1841 σ=0.4466 │ μ=-0.2148 σ=0.4479 │  89.4% │
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Key Finding**: Sign is FLIPPED (our mean positive, GT negative), but std is nearly identical.

### 11.2 Root Cause Analysis (5-Whys)

1. **Why is mean flipped?** Constant offset of +0.3989
2. **Why is there an offset?** Mel energies are ~55x larger than expected
3. **Why are energies larger?** Filterbank weights are 38x too large
4. **Why are weights wrong?** Model doesn't embed filterbank, uses fallback computation
5. **Why is fallback wrong?** Missing **Slaney normalization** in `compute_filterbank()`

### 11.3 Fix Applied

Added Slaney normalization to `src/audio/mel.rs`:

```rust
// Slaney normalization: divide by bandwidth to get equal-area filters
let bandwidth_hz = hz_points[m + 2] - hz_points[m];
let slaney_norm = if bandwidth_hz > 0.0 { 2.0 / bandwidth_hz } else { 1.0 };
```

### 11.4 Verification After Fix

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  │  A  │ audio           │   ✓    │ μ=+0.0005 σ=0.0711 │ μ=+0.0002 σ=0.0696 │   2.5% │
║  │  B  │ mel             │   ✓    │ μ=-0.2424 σ=0.4436 │ μ=-0.2148 σ=0.4479 │   7.1% │
╚══════════════════════════════════════════════════════════════════════════════╝
✓ All 2 stages passed
```

### 11.5 Methodology Validation

This case demonstrates APR-VERIFY's value:
- **Systematic**: Ground truth comparison at each stage
- **Visual**: TUI shows exactly where divergence occurs
- **Diagnostic**: 5-Whys tracing from symptoms to root cause
- **Rust-Only**: No Python scripts, all JSON ground truth pre-extracted

---

*Specification enhanced with 10 additional peer-reviewed citations and advanced verification features.*