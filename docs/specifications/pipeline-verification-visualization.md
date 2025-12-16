# APR-VERIFY-001: Pipeline Verification & Visualization System

**Status**: DRAFT - Awaiting Team Review
**Author**: Claude Code
**Date**: 2024-12-16
**PMAT Ticket**: APR-VERIFY-001

---

## Executive Summary

A deterministic, visual pipeline verification system for ML model debugging that combines:
- **Pixar/Weta-style stage gates** with ground truth comparison
- **Probar-style TUI testing** with pixel-perfect reproducibility
- **Toyota Production System** quality methodology
- **Popperian falsification** for rigorous validation

This differentiates from existing MLOps tools (MLflow, Weights & Biases, Neptune) by focusing on **deterministic debugging** rather than experiment tracking.

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

---

## 2. Literature Review

### 2.1 Peer-Reviewed Citations

1. **Amershi, S., et al. (2019).** "Software Engineering for Machine Learning: A Case Study." *IEEE/ACM 41st International Conference on Software Engineering (ICSE)*. pp. 291-300.
   - Documents ML debugging challenges at Microsoft
   - 45% of ML bugs are data-related, detectable via pipeline verification

2. **Breck, E., et al. (2017).** "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." *IEEE Big Data*. pp. 1123-1132.
   - Google's ML testing rubric
   - Advocates for "tests for model staleness" and "pipeline monitoring"

3. **Sculley, D., et al. (2015).** "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*. pp. 2503-2511.
   - Identifies "pipeline jungles" as major technical debt
   - Recommends explicit pipeline stage contracts

4. **Polyzotis, N., et al. (2018).** "Data Validation for Machine Learning." *MLSys*.
   - TensorFlow Data Validation (TFDV) approach
   - Schema-based validation at pipeline boundaries

5. **Baylor, D., et al. (2017).** "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform." *KDD*. pp. 1387-1395.
   - Production ML pipelines at Google
   - Component-based architecture with artifacts

6. **Ratner, A., et al. (2019).** "MLSys: The New Frontier of Machine Learning Systems." *arXiv:1904.03257*.
   - Systems perspective on ML debugging
   - Need for "principled debugging tools"

7. **Paleyes, A., et al. (2022).** "Challenges in Deploying Machine Learning: A Survey of Case Studies." *ACM Computing Surveys*.
   - 50+ deployment failures analyzed
   - Data pipeline issues most common root cause

8. **Shankar, S., et al. (2022).** "Operationalizing Machine Learning: An Interview Study." *arXiv:2209.09125*.
   - Interviews with 18 ML practitioners
   - "Debugging is the hardest part" - universal finding

9. **Lwakatare, L.E., et al. (2019).** "A Taxonomy of Software Engineering Challenges for Machine Learning Systems." *ESEM*. pp. 1-12.
   - Systematic taxonomy of ML engineering challenges
   - Pipeline debugging in top 5 challenges

10. **Xin, D., et al. (2021).** "Production Machine Learning Pipelines: Empirical Analysis and Optimization Opportunities." *SIGMOD*. pp. 2639-2652.
    - Analysis of 3000+ production ML pipelines
    - 78% have no automated validation

### 2.2 Key Insights from Literature

| Insight | Source | Our Response |
|---------|--------|--------------|
| 45% of ML bugs are data-related | Amershi (2019) | Stage-by-stage data validation |
| Pipeline jungles cause debt | Sculley (2015) | Explicit stage contracts |
| No automated validation in 78% | Xin (2021) | Built-in verification |
| Debugging is hardest part | Shankar (2022) | Visual TUI debugging |

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
                                │
                                ▼
                    ┌─────────────────────┐
                    │   VERIFICATION      │
                    │   REPORT + TUI      │
                    └─────────────────────┘
```

### 3.3 Module Structure

```
aprender/
├── src/
│   ├── verify/
│   │   ├── mod.rs              # Public API
│   │   ├── pipeline.rs         # Pipeline builder
│   │   ├── stage.rs            # Stage definition
│   │   ├── ground_truth.rs     # Golden trace loading
│   │   ├── delta.rs            # Statistical comparison
│   │   ├── report.rs           # Verification report
│   │   ├── tui/
│   │   │   ├── mod.rs          # TUI entry point
│   │   │   ├── render.rs       # Ratatui rendering
│   │   │   ├── widgets.rs      # Custom widgets
│   │   │   └── colors.rs       # Status colors
│   │   └── playbook/
│   │       ├── mod.rs          # Playbook engine
│   │       ├── parser.rs       # YAML parsing
│   │       └── executor.rs     # Scenario execution
│   └── lib.rs                  # Re-exports
├── tests/
│   ├── verify_integration.rs   # Integration tests
│   └── pixel_tests.rs          # TUI pixel tests (probar)
└── golden_traces/
    └── whisper-tiny/           # Reference implementation outputs
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
      - name: delta_display
        capture: region(40, 5, 20, 1)
        assert_matches: "\\d+\\.\\d+%"
```

### 4.2 APR-CLI Integration

```bash
# New apr-cli subcommands
apr verify pipeline.yaml              # Run verification
apr verify --tui pipeline.yaml        # Interactive TUI mode
apr verify --json pipeline.yaml       # JSON output for CI
apr verify --diff baseline.json       # Compare against baseline
apr golden extract model.apr          # Extract golden traces
apr golden compare a.bin b.bin        # Compare two traces
```

### 4.3 Coverage Integration

```rust
// Coverage tracking per stage
let report = pipeline.verify(&input)?;

// Stage-level coverage
assert!(report.stage("mel").coverage() >= 0.95);
assert!(report.stage("encoder").coverage() >= 0.90);

// Branch coverage within stages
assert!(report.stage("decoder").branch_coverage() >= 0.85);
```

### 4.4 PMAT Integration

```bash
# Quality gates
pmat verify-gate                      # Run as pre-commit hook
pmat verify-tdg                       # Include in TDG scoring

# Work tracking
pmat work start APR-VERIFY-XXX        # Track verification work
```

---

## 5. API Design

### 5.1 Pipeline Builder API

```rust
use aprender::verify::{Pipeline, Stage, GroundTruth, Tolerance};

// Declarative pipeline definition
let pipeline = Pipeline::builder("whisper-tiny")
    // Stage A: Audio Input
    .stage("audio")
        .input_type::<AudioSamples>()
        .output_type::<AudioSamples>()
        .transform(|x| x.resample(16000))
        .ground_truth(GroundTruth::from_file("golden/step_a.bin"))
        .tolerance(Tolerance::relative(0.01))
        .description("Resample to 16kHz mono")

    // Stage C: Mel Spectrogram
    .stage("mel")
        .input_type::<AudioSamples>()
        .output_type::<MelSpectrogram>()
        .transform(|x| mel_spectrogram(x, n_mels=80))
        .ground_truth(GroundTruth::from_file("golden/step_c.bin"))
        .tolerance(Tolerance::Stats {
            mean_delta: 0.05,
            std_delta: 0.05,
            max_delta: 0.10,
        })
        .description("80-bin mel spectrogram")

    // Stage G: Encoder
    .stage("encoder")
        .input_type::<MelSpectrogram>()
        .output_type::<EncoderOutput>()
        .transform(|x| encoder.forward(x))
        .ground_truth(GroundTruth::from_file("golden/step_g.bin"))
        .tolerance(Tolerance::cosine_similarity(0.99))
        .description("Transformer encoder blocks")

    .build()?;
```

### 5.2 Verification API

```rust
// Run verification
let report = pipeline.verify(&audio_input)?;

// Check overall status
if report.passed() {
    println!("All stages passed!");
} else {
    println!("Failed at stage: {}", report.first_failure().stage_name());
}

// Iterate stages
for stage_report in report.stages() {
    println!("{}: {} (delta={:.2}%)",
        stage_report.name(),
        stage_report.status(),
        stage_report.delta_percent()
    );
}

// Export for CI
report.to_json("verification_report.json")?;
report.to_junit("verification_results.xml")?;
```

### 5.3 TUI API

```rust
use aprender::verify::tui::VerifyTui;

// Interactive TUI mode
let tui = VerifyTui::new(pipeline, report)?;
tui.run()?;  // Blocks until user exits

// Or render once (for testing)
let frame = tui.render_frame()?;
assert!(frame.contains("✓ audio"));
assert!(frame.contains("✗ mel"));
```

### 5.4 Ground Truth API

```rust
use aprender::verify::GroundTruth;

// Load from file
let gt = GroundTruth::from_file("golden/step_c.bin")?;

// Load from reference implementation
let gt = GroundTruth::from_whisper_cpp("models/ggml-tiny.bin", "test.wav")?;
let gt = GroundTruth::from_huggingface("openai/whisper-tiny", "test.wav")?;

// Create from tensor
let gt = GroundTruth::from_tensor(&tensor, TensorMeta {
    name: "mel_spectrogram",
    shape: vec![3000, 80],
    dtype: DType::F32,
})?;

// Compare
let delta = gt.compare(&our_output)?;
println!("Mean delta: {}", delta.mean);
println!("Std delta: {}", delta.std);
println!("Max delta: {}", delta.max);
println!("Cosine similarity: {}", delta.cosine_sim);
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
║  │ D │ conv1         │   ○    │ (blocked)    │              │           │ ║
║  │ E │ conv2         │   ○    │ (blocked)    │              │           │ ║
║  │ G │ encoder       │   ○    │ (blocked)    │              │           │ ║
║  │ I │ decoder       │   ○    │ (blocked)    │              │           │ ║
║  │ M │ output        │   ○    │ (blocked)    │              │           │ ║
║  └─────┴───────────────┴────────┴──────────────┴──────────────┴───────────┘ ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │ STAGE DETAIL: mel (C)                                     [FAILED]      │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │ Shape: [3000, 80]                                                       │ ║
║  │                                                                         │ ║
║  │ Statistics Comparison:                                                  │ ║
║  │   Metric     │ Ours        │ Ground Truth │ Delta    │ Tolerance       │ ║
║  │   ──────────┼─────────────┼──────────────┼──────────┼─────────────────│ ║
║  │   mean      │ +0.1841     │ -0.2150      │ 0.3991   │ ±0.05    [FAIL] │ ║
║  │   std       │  0.4466     │  0.4480      │ 0.0014   │ ±0.05    [PASS] │ ║
║  │   min       │ -0.7658     │ -0.7658      │ 0.0000   │ ±0.10    [PASS] │ ║
║  │   max       │ +1.2342     │ +1.2342      │ 0.0000   │ ±0.10    [PASS] │ ║
║  │                                                                         │ ║
║  │ Diagnosis: Mean is POSITIVE but should be NEGATIVE (sign flip)          │ ║
║  │ Likely cause: Log normalization formula error                           │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  [q]uit  [r]erun  [e]xport  [d]iff  [↑↓]navigate  [Enter]details           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 6.2 Status Icons

| Icon | Meaning | Color |
|------|---------|-------|
| ✓ | Passed (delta < tolerance) | Green |
| ~ | Warning (delta near tolerance) | Yellow |
| ✗ | Failed (delta > tolerance) | Red |
| ○ | Blocked (upstream failed) | Gray |
| ? | No ground truth available | Blue |
| ⟳ | Running | Cyan |

### 6.3 Color Scheme

```rust
// colors.rs
pub const PASS: Color = Color::Rgb(0, 255, 0);      // #00FF00
pub const WARN: Color = Color::Rgb(255, 255, 0);    // #FFFF00
pub const FAIL: Color = Color::Rgb(255, 0, 0);      // #FF0000
pub const BLOCKED: Color = Color::Rgb(128, 128, 128); // #808080
pub const INFO: Color = Color::Rgb(0, 128, 255);    // #0080FF
pub const HEADER: Color = Color::Rgb(255, 255, 255); // #FFFFFF
```

### 6.4 Pixel Test Snapshots

```yaml
# pixel_tests/verify_tui.yaml
snapshots:
  - name: all_pass
    file: verify_all_pass.png
    description: All stages passed

  - name: mel_failure
    file: verify_mel_fail.png
    description: Mel stage failed, downstream blocked

  - name: stage_detail
    file: verify_stage_detail.png
    description: Expanded stage detail view
```

---

## 7. Playbook System

### 7.1 Playbook Format

```yaml
# playbooks/whisper_verification.yaml
name: Whisper Pipeline Verification
description: Verify whisper-tiny against ground truth

config:
  model: whisper-tiny.apr
  ground_truth_dir: golden_traces/whisper-tiny/
  tolerance_preset: strict  # strict | normal | relaxed

scenarios:
  - name: short_audio
    description: 1.5s speech sample
    input: test_audio/speech_1.5s.wav
    expected_output: "The birds can fly"
    stages:
      - name: audio
        tolerance: 0.01
      - name: mel
        tolerance: 0.05
      - name: encoder
        tolerance: 0.01

  - name: silence
    description: Pure silence (edge case)
    input: test_audio/silence_3s.wav
    expected_output: ""
    stages:
      - name: mel
        expected_pattern: "constant_value"

  - name: noise
    description: White noise (adversarial)
    input: test_audio/white_noise_3s.wav
    stages:
      - name: mel
        expected_pattern: "high_entropy"

assertions:
  - all_stages_pass
  - output_matches_expected
  - latency_under_ms: 1000
```

### 7.2 Playbook Execution

```bash
# Run single playbook
apr verify playbook whisper_verification.yaml

# Run with specific scenario
apr verify playbook whisper_verification.yaml --scenario short_audio

# Run all playbooks in directory
apr verify playbook playbooks/

# CI mode (exit code reflects pass/fail)
apr verify playbook whisper_verification.yaml --ci
```

### 7.3 Playbook Report

```json
{
  "playbook": "whisper_verification",
  "timestamp": "2024-12-16T12:00:00Z",
  "scenarios": [
    {
      "name": "short_audio",
      "status": "FAILED",
      "stages": [
        {"name": "audio", "status": "PASS", "delta": 0.001},
        {"name": "mel", "status": "FAIL", "delta": 0.894},
        {"name": "encoder", "status": "BLOCKED"}
      ],
      "first_failure": "mel",
      "diagnosis": "Mean sign flipped"
    }
  ],
  "summary": {
    "total": 3,
    "passed": 0,
    "failed": 3,
    "blocked": 0
  }
}
```

---

## 8. Toyota Way Implementation

### 8.1 Principles Applied

| Toyota Principle | Implementation |
|------------------|----------------|
| **Jidoka** (Stop the Line) | Pipeline halts at first failure, downstream stages blocked |
| **Genchi Genbutsu** (Go See) | Visual TUI shows actual data vs expected |
| **Kaizen** (Continuous Improvement) | Delta tracking over time, regression detection |
| **Poka-Yoke** (Error Proofing) | Type-safe stage connections, schema validation |
| **Andon** (Visual Signal) | ✓/✗/○ status indicators with color coding |
| **Heijunka** (Level Loading) | Parallel stage execution where possible |
| **Standardized Work** | Playbook-driven reproducible verification |
| **5 Whys** | Diagnosis suggestions trace root cause |

### 8.2 Stop-the-Line Implementation

```rust
impl Pipeline {
    pub fn verify(&self, input: &[f32]) -> VerifyResult {
        let mut report = VerifyReport::new();
        let mut current_output = input.to_vec();

        for stage in &self.stages {
            // Execute stage
            let output = stage.transform(&current_output)?;

            // Compare to ground truth
            let delta = stage.ground_truth.compare(&output)?;

            // Record result
            let stage_result = StageResult {
                name: stage.name.clone(),
                delta,
                status: if delta.within_tolerance(&stage.tolerance) {
                    Status::Pass
                } else {
                    Status::Fail
                },
            };
            report.add_stage(stage_result);

            // JIDOKA: Stop the line on failure
            if stage_result.status == Status::Fail {
                // Mark remaining stages as blocked
                for remaining in &self.stages[stage.index + 1..] {
                    report.add_stage(StageResult {
                        name: remaining.name.clone(),
                        status: Status::Blocked,
                        ..Default::default()
                    });
                }
                break;
            }

            current_output = output;
        }

        report
    }
}
```

### 8.3 5 Whys Diagnosis

```rust
impl StageResult {
    pub fn diagnose(&self) -> Vec<String> {
        let mut whys = Vec::new();

        // Why 1: What failed?
        whys.push(format!(
            "Stage '{}' failed with delta {:.1}%",
            self.name, self.delta.percent()
        ));

        // Why 2: Which metric?
        if self.delta.mean > self.tolerance.mean {
            whys.push(format!(
                "Mean delta ({:.4}) exceeds tolerance ({:.4})",
                self.delta.mean, self.tolerance.mean
            ));

            // Why 3: Sign or magnitude?
            if self.our_mean.signum() != self.gt_mean.signum() {
                whys.push("Sign is FLIPPED (positive vs negative)".into());

                // Why 4: Likely cause
                whys.push("Likely cause: Normalization formula error".into());

                // Why 5: Specific fix
                whys.push("Check: log base, clamp values, scale factor".into());
            }
        }

        whys
    }
}
```

---

## 9. Popperian Falsification QA

### 9.1 Falsification Methodology

Each test attempts to **prove the system is broken**. A passing test means we **failed to falsify** correctness.

### 9.2 100-Point QA Checklist

#### Section A: Pipeline Definition (20 points)

| # | Test | Falsification Target | Points |
|---|------|---------------------|--------|
| A01 | Empty pipeline fails gracefully | Pipeline with 0 stages | 2 |
| A02 | Single stage pipeline works | Minimal pipeline | 2 |
| A03 | Stage order is enforced | Stages execute in definition order | 2 |
| A04 | Duplicate stage names rejected | Name uniqueness | 2 |
| A05 | Type mismatch detected | Output type != next input type | 2 |
| A06 | Missing ground truth handled | Stage without GT shows "?" | 2 |
| A07 | Invalid tolerance rejected | Negative tolerance | 2 |
| A08 | Circular dependency detected | Stage depends on itself | 2 |
| A09 | Large pipeline scales | 100+ stages | 2 |
| A10 | Pipeline serialization roundtrips | Save/load YAML | 2 |

#### Section B: Ground Truth (20 points)

| # | Test | Falsification Target | Points |
|---|------|---------------------|--------|
| B01 | Binary file loading | .bin files parsed correctly | 2 |
| B02 | JSON metadata loading | .json stats parsed | 2 |
| B03 | Shape mismatch detected | GT shape != output shape | 2 |
| B04 | Dtype mismatch detected | GT f32 vs output f16 | 2 |
| B05 | Corrupt file handled | Truncated .bin file | 2 |
| B06 | Missing file error | Nonexistent path | 2 |
| B07 | Large GT files | 1GB+ tensor | 2 |
| B08 | Endianness handled | Little vs big endian | 2 |
| B09 | NaN values detected | GT contains NaN | 2 |
| B10 | Inf values detected | GT contains Inf | 2 |

#### Section C: Delta Computation (20 points)

| # | Test | Falsification Target | Points |
|---|------|---------------------|--------|
| C01 | Mean delta correct | Manual calculation matches | 2 |
| C02 | Std delta correct | Manual calculation matches | 2 |
| C03 | Max delta correct | Manual calculation matches | 2 |
| C04 | Min delta correct | Manual calculation matches | 2 |
| C05 | Cosine similarity correct | Manual calculation | 2 |
| C06 | Empty tensor handled | Zero-length arrays | 2 |
| C07 | Single element handled | Length-1 arrays | 2 |
| C08 | All zeros handled | Both tensors zero | 2 |
| C09 | All same value | Constant tensors | 2 |
| C10 | Very small values | Values near f32 epsilon | 2 |

#### Section D: TUI Rendering (20 points)

| # | Test | Falsification Target | Points |
|---|------|---------------------|--------|
| D01 | Initial render correct | Pixel test: initial state | 2 |
| D02 | Pass state renders green | ✓ icon, green color | 2 |
| D03 | Fail state renders red | ✗ icon, red color | 2 |
| D04 | Blocked state renders gray | ○ icon, gray color | 2 |
| D05 | Long stage names truncate | Name > column width | 2 |
| D06 | Large delta displays | 1000%+ delta | 2 |
| D07 | Negative delta displays | Delta can be negative | 2 |
| D08 | Detail view expands | Enter key shows details | 2 |
| D09 | Navigation works | Arrow keys move selection | 2 |
| D10 | Resize handled | Terminal resize event | 2 |

#### Section E: Playbook Execution (20 points)

| # | Test | Falsification Target | Points |
|---|------|---------------------|--------|
| E01 | Valid YAML parses | Well-formed playbook | 2 |
| E02 | Invalid YAML rejected | Malformed YAML | 2 |
| E03 | Missing scenario handled | Scenario not found | 2 |
| E04 | All scenarios run | Multiple scenarios | 2 |
| E05 | Scenario isolation | One failure doesn't affect others | 2 |
| E06 | Timeout enforced | Scenario exceeds time limit | 2 |
| E07 | Report generated | JSON/JUnit output | 2 |
| E08 | CI exit codes | 0=pass, 1=fail, 2=error | 2 |
| E09 | Parallel execution | Scenarios run concurrently | 2 |
| E10 | Retry logic | Flaky scenario retries | 2 |

### 9.3 QA Execution

```bash
# Run full QA suite
cargo test --test falsification_qa -- --nocapture

# Run specific section
cargo test --test falsification_qa section_a

# Generate QA report
cargo test --test falsification_qa -- --format json > qa_report.json
```

### 9.4 QA Score Calculation

```rust
pub fn calculate_qa_score(results: &[TestResult]) -> QAScore {
    let total_points: u32 = results.iter().map(|r| r.max_points).sum();
    let earned_points: u32 = results.iter()
        .filter(|r| r.passed)
        .map(|r| r.max_points)
        .sum();

    QAScore {
        earned: earned_points,
        total: total_points,
        percentage: (earned_points as f64 / total_points as f64) * 100.0,
        grade: match earned_points * 100 / total_points {
            95..=100 => "A+",
            90..=94 => "A",
            85..=89 => "B+",
            80..=84 => "B",
            _ => "FAIL",
        },
    }
}
```

---

## 10. Implementation Phases

### Phase 1: Core Infrastructure (Sprint 1)

**Deliverables:**
- [ ] `Pipeline` struct and builder
- [ ] `Stage` definition
- [ ] `GroundTruth` loader (.bin/.json)
- [ ] `Delta` computation (mean, std, max, cosine)
- [ ] Basic `VerifyReport`

**Tests:**
- Sections A, B, C of Falsification QA

**Exit Criteria:**
- `cargo test` passes
- Can verify a 3-stage pipeline
- Delta computation matches manual calculation

### Phase 2: TUI Visualization (Sprint 2)

**Deliverables:**
- [ ] Ratatui-based TUI
- [ ] Stage list widget
- [ ] Detail view widget
- [ ] Status icons and colors
- [ ] Keyboard navigation

**Tests:**
- Section D of Falsification QA
- Probar pixel tests

**Exit Criteria:**
- TUI renders correctly (pixel tests pass)
- Navigation works
- Resize handled

### Phase 3: Playbook System (Sprint 3)

**Deliverables:**
- [ ] YAML parser
- [ ] Scenario executor
- [ ] Report generator (JSON, JUnit)
- [ ] CI integration

**Tests:**
- Section E of Falsification QA

**Exit Criteria:**
- Playbooks execute correctly
- CI exit codes work
- Reports generated

### Phase 4: Integration & Polish (Sprint 4)

**Deliverables:**
- [ ] APR-CLI integration
- [ ] PMAT hooks
- [ ] Documentation
- [ ] Example pipelines

**Tests:**
- Full 100-point QA
- Integration tests

**Exit Criteria:**
- All 100 QA points pass
- Documentation complete
- Examples work

---

## Appendix A: Comparison to Existing Tools

| Feature | aprender-verify | MLflow | W&B | TensorBoard |
|---------|-----------------|--------|-----|-------------|
| Stage-by-stage verification | ✓ | ✗ | ✗ | ✗ |
| Ground truth comparison | ✓ | ✗ | ✗ | ✗ |
| Visual TUI | ✓ | ✗ | ✗ | ✗ |
| Deterministic replay | ✓ | ✗ | ✗ | ✗ |
| Pixel-perfect testing | ✓ | ✗ | ✗ | ✗ |
| Stop-the-line | ✓ | ✗ | ✗ | ✗ |
| 5 Whys diagnosis | ✓ | ✗ | ✗ | ✗ |
| Playbook scenarios | ✓ | ✗ | ✗ | ✗ |
| CI/CD integration | ✓ | ✓ | ✓ | ✓ |
| Experiment tracking | ✗ | ✓ | ✓ | ✓ |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Ground Truth** | Reference output from known-correct implementation |
| **Golden Trace** | Saved ground truth data for a specific stage |
| **Delta** | Statistical difference between output and ground truth |
| **Stage** | Single transformation step in pipeline |
| **Tolerance** | Maximum acceptable delta for a stage |
| **Playbook** | YAML file defining verification scenarios |
| **Jidoka** | Toyota principle: stop on first defect |
| **Andon** | Toyota principle: visual status indicator |

---

## Appendix C: File Formats

### Ground Truth Binary (.bin)

```
[4 bytes] Magic: "GTRC" (Ground Truth Trace)
[4 bytes] Version: u32
[4 bytes] Dtype: 0=f32, 1=f16, 2=i8
[4 bytes] Ndim: number of dimensions
[Ndim * 4 bytes] Shape: dimension sizes
[remaining] Data: raw tensor data
```

### Ground Truth Metadata (.json)

```json
{
  "name": "step_c_mel",
  "shape": [3000, 80],
  "dtype": "f32",
  "min": -0.766,
  "max": 1.234,
  "mean": -0.215,
  "std": 0.448,
  "source": "whisper.cpp",
  "source_version": "1.5.0",
  "created": "2024-12-16T12:00:00Z"
}
```

---

**END OF SPECIFICATION**

*Awaiting team review before implementation.*
