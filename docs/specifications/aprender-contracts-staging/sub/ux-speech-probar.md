# Sub-spec: UX Contracts, Speech/Whisper.apr, and Probar Integration

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [pv-spec.md](../pv-spec.md) Section 20

---

## 1. UX Contract Taxonomy

UX elements have four categories of provable properties:

### Category A: Geometric Invariants

Pure functions on geometric primitives. Same proof techniques as kernels.

| Contract | Property | Proof Method |
|---|---|---|
| `rect-geometry-v1` | `intersection(A,A) = A` (idempotent) | Kani exhaustive |
| `rect-geometry-v1` | `intersection(A,B) = intersection(B,A)` (commutative) | Kani exhaustive |
| `constraints-layout-v1` | `min ≤ output.width ≤ max` (bounded) | Kani bounded_int |
| `constraints-layout-v1` | `Σ child.width ≤ parent.width` (no overflow) | Kani bounded_int |
| `constraints-layout-v1` | `output.width ≥ 0 ∧ output.height ≥ 0` (non-negative) | Kani exhaustive |

**presentar contracts needed:**

```yaml
# flex-layout-v1.yaml
equations:
  flex_distribute:
    formula: "Σ child_flex_size = available_space"
    invariants:
      - "No child receives negative space"
      - "Flex factors sum correctly: Σ flex_basis + Σ flex_grow*remaining = total"
      - "Minimum sizes respected: child.size ≥ child.min_size"

# grid-layout-v1.yaml
equations:
  grid_place:
    formula: "cell(row, col) = origin + (row * row_height, col * col_width)"
    invariants:
      - "No cell overlap: cell(i) ∩ cell(j) = ∅ for i ≠ j"
      - "All cells within grid bounds"
      - "Row/col indices in range: 0 ≤ row < rows, 0 ≤ col < cols"
```

### Category B: Perceptual Correctness

Standards-based formulas (W3C, ITU-R). Mathematically falsifiable.

| Contract | Standard | Formula |
|---|---|---|
| `color-wcag-v1` | WCAG 2.1 | `(L1+0.05)/(L2+0.05) ≥ 4.5` |
| `color-lerp-v1` | sRGB | `lerp(a, b, 0) = a, lerp(a, b, 1) = b` |
| `loudnorm-v1` | ITU-R BS.1770-4 | K-weighting + gated LUFS |

**New contracts needed:**

```yaml
# focus-order-v1.yaml (accessibility)
equations:
  tab_order:
    formula: "focus_sequence = depth_first_traversal(widget_tree)"
    invariants:
      - "Every focusable widget is reachable"
      - "Tab order matches visual order (left-to-right, top-to-bottom)"
      - "No focus trap: Escape always returns to parent"

# animation-timing-v1.yaml (frame budget)
equations:
  frame_budget:
    formula: "render_time(frame_n) < 1/fps for all n"
    invariants:
      - "No frame exceeds 16.67ms at 60fps"
      - "Animation progress is monotonic: progress(t+dt) ≥ progress(t)"
      - "Animation completes: progress(duration) = 1.0"
```

### Category C: Pipeline Correctness (batuta)

Orchestration invariants — routing, templating, privacy.

| Contract | Exists? | Property |
|---|---|---|
| `privacy-enforcement-v1` | Yes (banco) | Sovereign requests never reach external APIs |
| `routing-determinism-v1` | Yes (banco) | Same input → same backend |
| `template-correctness-v1` | Yes (banco) | No unescaped user input (XSS prevention) |
| `budget-conservation-v1` | Yes (banco) | Token budget consumed ≤ allocated |

**New contracts needed:**

```yaml
# session-isolation-v1.yaml
equations:
  isolation:
    formula: "state(session_a) ∩ state(session_b) = ∅"
    invariants:
      - "No cross-session data leakage"
      - "Session timeout enforced: active(s) → last_activity(s) < timeout"

# rate-limit-v1.yaml
equations:
  token_bucket:
    formula: "tokens = min(capacity, tokens + rate * elapsed)"
    invariants:
      - "requests_allowed ≤ capacity per window"
      - "Burst ≤ capacity"
```

### Category D: Visual Regression

Pixel-level correctness. Hardest to express in YAML but possible:

```yaml
# visual-regression-v1.yaml
equations:
  pixel_diff:
    formula: "diff(render(state), golden) < tolerance"
    domain: "state ∈ TestStates, golden ∈ Screenshots"
    invariants:
      - "SSIM(render, golden) > 0.99 for all test states"
      - "No pixel outside valid RGBA range"
```

**Enforcement:** Cannot use Kani (pixel rendering is not bounded).
Use probar property tests comparing against golden screenshots.

---

## 2. Whisper.apr Contracts

### APR Format Contracts

The APR format (aprender's model serialization) needs correctness contracts:

```yaml
# apr-serialization-v1.yaml
metadata:
  description: "APR model format — lossless tensor serialization with metadata"
  references:
    - "aprender/src/serialization/apr/mod.rs"

equations:
  roundtrip:
    formula: "read(write(tensors, metadata)) = (tensors, metadata)"
    invariants:
      - "Lossless: read_tensor_f32(write(data)) = data bit-for-bit"
      - "Metadata preserved: get_metadata(key) = original value"
      - "Filtered read: open_filtered(f) loads only tensors matching f"

  tensor_integrity:
    formula: "read_tensor_f32_checked(name).len() = descriptor.shape.product()"
    invariants:
      - "Shape matches data length"
      - "Data type matches descriptor dtype"
      - "No NaN/Inf in loaded tensors (for verified models)"

proof_obligations:
  - type: roundtrip
    property: "APR write/read roundtrip is lossless"
    formal: "∀ tensors, metadata: read(write(tensors, metadata)) = (tensors, metadata)"
  - type: invariant
    property: "Tensor shape consistency"
    formal: "∀ name: tensor(name).len() = shape(name).product()"
  - type: bound
    property: "No corrupt data"
    formal: "∀ name: tensor(name).iter().all(|x| x.is_finite())"
```

### Whisper ASR Pipeline Contracts

```yaml
# whisper-asr-v1.yaml
metadata:
  description: "Whisper automatic speech recognition — audio to text transcription"
  references:
    - "Radford et al. (2023) Robust Speech Recognition via Large-Scale Weak Supervision"
    - "aprender/src/speech/asr/mod.rs"

equations:
  transcribe:
    formula: "text = decode(encode(mel_spectrogram(audio)))"
    domain: "audio ∈ ℝ^n (16kHz PCM), language ∈ Languages"
    codomain: "Transcription with segments and word timings"
    invariants:
      - "Segment timestamps are monotonically increasing"
      - "Segments cover entire audio: segments[-1].end ≈ audio.duration"
      - "Word timings within segment bounds"

  mel_spectrogram:
    formula: "mel = log(max(ε, mel_filterbank @ stft(audio)))"
    domain: "audio ∈ ℝ^n, n_fft=400, hop=160, n_mels=80"
    invariants:
      - "Output shape: (n_mels, n_frames) where n_frames = ceil(len/hop)"
      - "No -inf values (ε floor prevents log(0))"

  language_detect:
    formula: "lang = argmax(softmax(encoder(mel[:30s])))"
    invariants:
      - "Returns valid ISO 639-1 code"
      - "Confidence ∈ [0, 1]"
      - "English audio → P(en) > 0.5 for standard speech"

proof_obligations:
  - type: monotonicity
    property: "Segment timestamps monotonic"
    formal: "∀ i: segments[i].start ≤ segments[i].end ≤ segments[i+1].start"
  - type: bound
    property: "Mel spectrogram finite"
    formal: "∀ i,j: mel[i][j].is_finite()"
  - type: postcondition
    property: "Transcription covers full audio"
    formal: "|segments.last().end - audio.duration()| < 0.5s"

falsification_tests:
  - id: FALSIFY-WHI-001
    rule: "Silence produces empty transcription"
    prediction: "transcribe(silence_30s).segments.is_empty()"
    if_fails: "Model hallucinates on silence"
  - id: FALSIFY-WHI-002
    rule: "Timestamp monotonicity"
    prediction: "All segments have start < end, ordered"
    if_fails: "Decoder produces out-of-order timestamps"
  - id: FALSIFY-WHI-003
    rule: "Language detection accuracy"
    prediction: "English audio classified as 'en' with >0.8 confidence"
    if_fails: "Encoder representations not language-discriminative"

kani_harnesses:
  - id: KANI-WHI-001
    obligation: "Mel spectrogram finite"
    bound: 256
    strategy: stub_float
    solver: cadical
```

### VAD (Voice Activity Detection) Contracts

```yaml
# vad-v1.yaml
equations:
  detect:
    formula: "segments = {(start, end) : energy(window) > threshold}"
    domain: "samples ∈ ℝ^n, sample_rate ∈ {8000, 16000, 44100, 48000}"
    invariants:
      - "Segments are non-overlapping"
      - "Segments are within audio bounds: 0 ≤ start < end ≤ duration"
      - "Energy threshold respected: ∀ seg: energy(seg) > threshold"
      - "Silence gaps respected: ∀ gap: energy(gap) < threshold"

proof_obligations:
  - type: invariant
    property: "Non-overlapping segments"
    formal: "∀ i,j (i≠j): segments[i] ∩ segments[j] = ∅"
  - type: bound
    property: "Within audio bounds"
    formal: "∀ seg: 0 ≤ seg.start < seg.end ≤ audio.duration()"
```

---

## 3. Probar Integration

probar (214K LOC) is the property testing framework. It generates and runs
the falsification tests that `pv probar` outputs.

### Current Integration

```
pv probar contracts/softmax-kernel-v1.yaml
  → generates #[probar::property] test stubs
  → tests call kernel functions with random inputs
  → asserts contract postconditions hold
```

### Enhanced Integration: probar as PVScore D2 Data Source

Currently D2 (falsification coverage) counts whether tests EXIST.
Enhanced: probar reports whether tests PASS and their coverage.

```yaml
# .pmat-metrics/probar.result
tests_run: 4827
tests_passed: 4827
properties_checked: 638
properties_falsified: 0
coverage_lines: 87.3%
coverage_branches: 72.1%
```

PVScore D2 becomes: `D2 = (properties_passed / properties_total) * 100`

### probar + whisper.apr Integration

```bash
# Generate property tests for whisper ASR from contract
pv probar contracts/whisper-asr-v1.yaml --binding contracts/aprender/binding.yaml

# Output: probar test that loads a whisper.apr model and runs inference
#[probar::property]
fn prop_transcription_timestamps_monotonic(audio: AudioFixture) {
    let model = AprReader::open("whisper-tiny.apr").unwrap();
    let result = transcribe(&model, &audio.samples, audio.sample_rate);
    for window in result.segments.windows(2) {
        probar::assert!(window[0].end <= window[1].start,
            "FALSIFY-WHI-002: timestamps not monotonic");
    }
}
```

### probar + UX Integration

```bash
# Generate property tests for layout contracts
pv probar contracts/presentar/flex-layout-v1.yaml --binding contracts/presentar/binding.yaml

# Output:
#[probar::property]
fn prop_flex_no_negative_child(
    parent_width: u16,    // 0..1000
    child_count: u8,      // 1..10
    flex_factors: Vec<f32>,
) {
    let layout = flex_distribute(parent_width, &children);
    for child in &layout.children {
        probar::assert!(child.width >= 0, "Negative child width");
    }
    probar::assert!(
        layout.children.iter().map(|c| c.width).sum::<i32>() <= parent_width as i32,
        "Children overflow parent"
    );
}
```

---

## 4. Implementation Roadmap

| Priority | Contract | Repo | Method |
|---|---|---|---|
| **P0** | `apr-serialization-v1` | aprender | Kani roundtrip proof |
| **P0** | `whisper-asr-v1` | aprender | probar property tests |
| **P1** | `vad-v1` | aprender | Kani bounded + probar |
| **P1** | `flex-layout-v1` | presentar | Kani bounded_int |
| **P1** | `grid-layout-v1` | presentar | Kani bounded_int |
| **P2** | `focus-order-v1` | presentar | probar + golden tests |
| **P2** | `animation-timing-v1` | rmedia | probar frame budget |
| **P2** | `session-isolation-v1` | batuta | Kani + TLA+ |
| **P3** | `visual-regression-v1` | presentar | probar + screenshot diff |

---

## 5. apr-model-qa-playbook Integration

The Model Quality Score (MQS) from `apr-model-qa-playbook` (86K LOC) provides
a production model certification pipeline. It already has contracts:

| Contract | Bindings | What it certifies |
|---|---|---|
| `apr-format-invariants-v1` | 5 | APR roundtrip, tensor bijection, no silent fallbacks |
| `mqs-scoring-v1` | registry | MQS composite: QUAL+PERF+STAB+COMP+EDGE+REGR (0-1000) |
| `gateway-contract-v1` | 3 | Gate pass/fail determinism |
| `garbage-oracle-v1` | 2 | Corrupt model detection |

### MQS → PVScore Integration

MQS scores individual MODELS. PVScore scores CODEBASES. They compose:

```
PVScore D7 (mutation testing) can be enhanced:
  D7 = 0.5 * mutation_kill_rate + 0.5 * mqs_pass_rate
```

Where `mqs_pass_rate` = fraction of models passing all MQS gates.
A codebase that passes code-level mutation testing but fails model-level
quality gates still has a quality gap.

### Whisper.apr QA Pipeline

The apr-model-qa-playbook certifies whisper.apr models:

```bash
# Run MQS certification on a whisper model
apr-qa run whisper-tiny.apr --playbook playbooks/whisper-qa.yaml

# Output: MQS score + gate evidence
MQS: 847/1000 (Grade A-)
  QUAL: 185/200 (mel spectrogram quality)
  PERF: 140/150 (RTF < 0.1 for tiny model)
  STAB: 180/200 (deterministic across runs)
  COMP: 142/150 (language detection accuracy)
  EDGE: 120/150 (silence handling, noise robustness)
  REGR: 80/150 (no regression from previous version)
```

### probar Property Tests for MQS

```bash
# Generate probar tests from MQS scoring contract
pv probar contracts/mqs-scoring-v1.yaml

# Output: property test that MQS formula is deterministic
#[probar::property]
fn prop_mqs_deterministic(evidence: EvidenceFixture) {
    let score1 = MqsCalculator::calculate(&evidence);
    let score2 = MqsCalculator::calculate(&evidence);
    probar::assert_eq!(score1, score2, "MQS must be deterministic");
}

#[probar::property]
fn prop_mqs_bounded(evidence: EvidenceFixture) {
    let score = MqsCalculator::calculate(&evidence);
    probar::assert!(score.raw >= 0 && score.raw <= 1050);
    probar::assert!(score.normalized >= 0.0 && score.normalized <= 100.0);
}
```

---

## References

1. Radford, A. et al. (2023). "Robust Speech Recognition via Large-Scale
   Weak Supervision." *ICML 2023*. (Whisper paper)

2. Meyer, B. (1997). *Object-Oriented Software Construction*, 2nd ed.
   Ch. 11: Design by Contract. (Contract taxonomy for UI components)

3. W3C (2018). "Web Content Accessibility Guidelines (WCAG) 2.1."
   www.w3.org/TR/WCAG21/. (Perceptual correctness standards)

4. ITU-R BS.1770-4 (2015). "Algorithms to measure audio programme loudness."
   (Loudness measurement standard)

5. Claessen, K. & Hughes, J. (2000). "QuickCheck: A Lightweight Tool for
   Random Testing of Haskell Programs." *ICFP 2000*.
   (Foundation for property-based testing / probar)
