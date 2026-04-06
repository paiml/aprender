---
title: "Pixel Coverage PNG Heatmap with Popperian Falsification and Pixel-Perfect Verification"
issue: PIXEL-001
status: Under Review (v2.1)
created: 2025-12-13T10:02:21.219401743+00:00
updated: 2025-12-13T19:55:00.000000000+00:00
version: 2.1.0
methodology: Popperian Falsification + Film-Study Pixel Verification
---

# PIXEL-001: Pixel Coverage Heatmap with Scientific Rigor

**Ticket ID**: PIXEL-001
**Status**: 🔬 Under Review (v2.1 Enhancement)
**Methodology**: Popperian Falsification Framework

> *"A theory is scientific if and only if it is falsifiable."*
> — Karl Popper, *The Logic of Scientific Discovery* (1934)

---

## Abstract

This specification enhances probar's pixel coverage system with **Popperian scientific methodology** and **film-study pixel-perfect verification** ("pixel fucking"). The system provides falsifiable hypotheses about UI coverage, implements perceptually-grounded image comparison metrics (SSIM, CIEDE2000), and delivers **first-class terminal output** as the primary interface for coverage visualization.

---

## 1. Popperian Falsification Framework

### 1.1 Core Principle: The Falsifiability Gateway

Following the PAIML MCP Agent Toolkit's Popper Score methodology [1], pixel coverage claims must pass a **falsifiability gateway** before being considered scientifically valid:

```rust
/// Popperian Falsifiability Gate (Jidoka - stop-the-line)
/// If hypothesis cannot be falsified, the entire analysis is invalid.
pub struct FalsifiabilityGate {
    /// Minimum threshold for falsifiability score (0-25)
    pub gateway_threshold: f32,  // Default: 15.0
}

impl FalsifiabilityGate {
    pub fn evaluate(&self, hypothesis: &CoverageHypothesis) -> GateResult {
        if hypothesis.falsifiability_score < self.gateway_threshold {
            GateResult::Failed {
                score: 0.0,
                reason: "INSUFFICIENT FALSIFIABILITY - NOT EVALUABLE AS SCIENCE",
            }
        } else {
            GateResult::Passed
        }
    }
}
```

### 1.2 Falsifiable Coverage Hypotheses

Each coverage claim must be expressed as a **falsifiable hypothesis** with explicit failure conditions:

```rust
/// A falsifiable hypothesis about pixel coverage
#[derive(Debug, Clone)]
pub struct CoverageHypothesis {
    /// H₀: The null hypothesis to falsify
    pub null_hypothesis: String,
    /// Measurable threshold (falsification criterion)
    pub threshold: f32,
    /// Confidence interval for statistical rigor
    pub confidence_interval: ConfidenceInterval,
    /// What would falsify this claim
    pub falsification_conditions: Vec<FalsificationCondition>,
}

/// Example Hypotheses:
/// H₀-COV-01: "Coverage exceeds 85% of screen pixels" (falsifiable via pixel count)
/// H₀-COV-02: "No gap region exceeds 5% of total area" (falsifiable via gap detection)
/// H₀-COV-03: "Rendered heatmap matches reference within SSIM ≥ 0.99" (falsifiable via SSIM)
```

### 1.3 Three Layers of Falsification Testing

| Layer | Type | Description | Example |
|-------|------|-------------|---------|
| **L1** | Unit Tests | Direct falsification via assertions | `assert!(coverage >= 0.85)` |
| **L2** | Property Tests | Statistical falsification via proptest | `proptest!(coverage roundtrips)` |
| **L3** | Mutation Tests | Meta-falsification via mutation score | `mutants killed ≥ 80%` |

---

## 2. Pixel-Perfect Verification ("Pixel Fucking")

### 2.1 Film-Study Methodology

Borrowing from VFX compositing best practices [2], pixel coverage verification requires **perceptually-grounded metrics** that align with human visual perception:

```rust
/// Pixel-perfect verification metrics following film-study standards
pub struct PixelVerificationSuite {
    /// Structural Similarity Index (Wang et al., 2004)
    pub ssim: SsimMetric,
    /// Peak Signal-to-Noise Ratio
    pub psnr: PsnrMetric,
    /// CIE Delta E 2000 color difference (Sharma et al., 2005)
    pub delta_e: CieDe2000Metric,
    /// Perceptual hash for content fingerprinting
    pub phash: PerceptualHash,
}
```

### 2.2 CIEDE2000 Color Difference Thresholds

Following dental and industrial color science research [3][4]:

| Threshold Type | ΔE₀₀ Value | Interpretation |
|----------------|------------|----------------|
| **Just Noticeable Difference (JND)** | 0.8 - 1.0 | Barely perceptible |
| **Perceptibility Threshold** | 1.0 - 1.2 | 50% of observers notice |
| **Acceptability Threshold** | 1.8 - 2.8 | 50% find acceptable |
| **Clinically Unacceptable** | ≥ 3.7 | Clearly wrong |

```rust
/// CIEDE2000 color difference metric (ISO/CIE 11664-6:2014)
pub struct CieDe2000Metric {
    /// Perceptibility threshold (JND)
    pub jnd_threshold: f32,      // Default: 1.0
    /// Acceptability threshold
    pub accept_threshold: f32,   // Default: 2.0
}

impl CieDe2000Metric {
    /// Compare two colors and return ΔE₀₀
    pub fn delta_e(&self, lab1: &Lab, lab2: &Lab) -> f32;

    /// Is the difference imperceptible?
    pub fn is_imperceptible(&self, delta_e: f32) -> bool {
        delta_e < self.jnd_threshold
    }
}
```

### 2.3 SSIM (Structural Similarity Index)

Following Wang et al.'s foundational work [5]:

```rust
/// Structural Similarity Index Measure
/// Range: -1 to 1 (1 = identical, 0 = no similarity)
pub struct SsimMetric {
    /// Window size for local comparison
    pub window_size: u32,        // Default: 11
    /// Threshold for "pixel perfect"
    pub perfect_threshold: f32,  // Default: 0.99
    /// Threshold for "acceptable"
    pub accept_threshold: f32,   // Default: 0.95
}

impl SsimMetric {
    /// Compare two images
    pub fn compare(&self, reference: &RgbImage, generated: &RgbImage) -> SsimResult;

    /// Generate SSIM map showing local differences
    pub fn ssim_map(&self, reference: &RgbImage, generated: &RgbImage) -> GrayscaleImage;
}
```

### 2.4 Perceptual Hashing

For robust content fingerprinting [6][7]:

```rust
/// Perceptual hash for image fingerprinting
/// Robust to minor compression artifacts and scaling
pub struct PerceptualHash {
    pub algorithm: PhashAlgorithm,  // DCT-based, SVD, Wavelet
    pub hash_size: u32,             // Default: 64 bits
}

pub enum PhashAlgorithm {
    /// Average hash (fastest, least robust)
    AHash,
    /// Difference hash (good balance)
    DHash,
    /// Perceptual hash using DCT (most robust)
    PHash,
    /// Wavelet hash
    WHash,
}

impl PerceptualHash {
    /// Compute hash for image
    pub fn compute(&self, image: &RgbImage) -> u64;

    /// Hamming distance between hashes (0 = identical)
    pub fn distance(&self, hash1: u64, hash2: u64) -> u32;

    /// Are images perceptually similar? (distance ≤ threshold)
    pub fn is_similar(&self, hash1: u64, hash2: u64, threshold: u32) -> bool;
}
```

---

## 3. First-Class Terminal Output

### 3.1 Design Philosophy

Terminal output is **not secondary** to PNG export—it is the **primary interface** for developers. The terminal heatmap must provide:

1. **Immediate visual feedback** without opening external files
2. **Gap detection** with clear highlighting
3. **Numeric score** with statistical confidence
4. **Actionable recommendations**

### 3.2 Rich Terminal Heatmap Display

```
╔══════════════════════════════════════════════════════════════════════╗
║              PIXEL COVERAGE HEATMAP - TSP Demo v1.2.0                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ████████████████████████████████████████████████████████████████    ║
║  ████████████████████████████████████████████████████████████████    ║
║  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████████████    ║
║  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████████████    ║
║  ████████████░░░░░░⚠ GAP DETECTED ⚠░░░░░████████████████████████    ║
║  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████████████    ║
║  ████████████████████████████████████████████████████████████████    ║
║  ████████████████████████████████████████████████████████████████    ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  LEGEND: █ 76-100%  ▓ 51-75%  ▒ 26-50%  ░ 1-25%  · 0% (GAP)         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  COVERAGE SCORE                                              │    ║
║  │  ═══════════════════════════════════════════════════════════ │    ║
║  │                                                               │    ║
║  │    Pixel Coverage:  87.3%  ████████████████████░░░░░         │    ║
║  │    Line Coverage:   92.1%  ██████████████████████░░░         │    ║
║  │    Combined Score:  89.7%  █████████████████████░░░░         │    ║
║  │                                                               │    ║
║  │    Threshold: 85.0%    Status: ✅ PASS                       │    ║
║  │    Confidence: 95% CI [88.2%, 91.2%]                         │    ║
║  │                                                               │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠ GAPS DETECTED (1 region, 12.7% of screen)                        ║
║  ├─ Gap #1: rows 3-6, cols 12-28 (center panel - UNCOVERED)         ║
║  └─ Recommendation: Add test for "settings_panel" component         ║
╠══════════════════════════════════════════════════════════════════════╣
║  FALSIFICATION STATUS                                                ║
║  ├─ H₀-COV-01: Coverage ≥ 85%        → ✅ NOT FALSIFIED (87.3%)     ║
║  ├─ H₀-COV-02: No gap > 15% area     → ✅ NOT FALSIFIED (12.7%)     ║
║  └─ H₀-COV-03: SSIM ≥ 0.99 vs ref    → ✅ NOT FALSIFIED (0.9987)    ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 3.3 Terminal Heatmap API

```rust
/// First-class terminal heatmap with rich output
pub struct RichTerminalHeatmap {
    /// Grid cells with coverage data
    cells: Vec<Vec<CoverageCell>>,
    /// Color palette for ANSI output
    palette: ColorPalette,
    /// Show numeric scores
    show_scores: bool,
    /// Show gap analysis
    show_gaps: bool,
    /// Show falsification status
    show_hypotheses: bool,
    /// Statistical confidence level
    confidence_level: f32,
}

impl RichTerminalHeatmap {
    /// Render complete terminal display
    pub fn render(&self) -> String;

    /// Render just the heatmap grid
    pub fn render_grid(&self) -> String;

    /// Render score panel
    pub fn render_scores(&self) -> String;

    /// Render gap analysis
    pub fn render_gap_analysis(&self) -> String;

    /// Render falsification status
    pub fn render_hypotheses(&self, hypotheses: &[CoverageHypothesis]) -> String;

    /// Interactive mode (for TUI)
    pub fn interactive(&self) -> TerminalApp;
}
```

### 3.4 Score Visualization

```rust
/// Visual score bar for terminal
pub struct ScoreBar {
    /// Score value (0.0 - 1.0)
    score: f32,
    /// Width in characters
    width: usize,
    /// Threshold for pass/fail
    threshold: f32,
}

impl ScoreBar {
    /// Render: "87.3%  ████████████████████░░░░░"
    pub fn render(&self) -> String {
        let filled = (self.score * self.width as f32) as usize;
        let empty = self.width - filled;
        let bar = format!(
            "{:5.1}%  {}{}",
            self.score * 100.0,
            "█".repeat(filled),
            "░".repeat(empty)
        );
        if self.score >= self.threshold {
            format!("\x1b[32m{}\x1b[0m", bar)  // Green
        } else {
            format!("\x1b[31m{}\x1b[0m", bar)  // Red
        }
    }
}
```

### 3.5 Accessibility & CI Integration

The terminal output must be accessible and CI-friendly.

*   **No Color Mode**: When `NO_COLOR` env var is set or `--no-color` flag is used, use ASCII shading (`@`, `%`, `#`, `*`, `+`, `=`, `-`, `:`, `.`, ` `) instead of ANSI colors.
*   **JSON Output**: For CI tools, support `--json` to output raw data.

```rust
pub enum OutputMode {
    RichAnsi,
    NoColorAscii,
    Json,
}
```

---

## 4. Enhanced Architecture

### 4.1 Design Overview

```text
PixelCoverageTracker
        │
        ├── RichTerminalHeatmap (FIRST-CLASS)
        │       ├── GridRenderer (Unicode blocks)
        │       ├── ScorePanel (visual bars)
        │       ├── GapAnalyzer (recommendations)
        │       └── HypothesisReporter (falsification status)
        │
        ├── PngHeatmap (export)
        │       ├── ColorPalette (Viridis, Magma, Heat)
        │       ├── LegendRenderer
        │       ├── GapHighlighter
        │       └── MetadataOverlay
        │
        └── PixelVerificationSuite (pixel-perfect)
                ├── SsimMetric
                ├── PsnrMetric
                ├── CieDe2000Metric
                └── PerceptualHash
```

### 4.2 Verification Pipeline

```rust
/// Complete pixel-perfect verification pipeline
pub struct VerificationPipeline {
    /// Reference image for comparison
    reference: Option<RgbImage>,
    /// Verification suite
    suite: PixelVerificationSuite,
    /// Falsifiable hypotheses
    hypotheses: Vec<CoverageHypothesis>,
}

impl VerificationPipeline {
    /// Run complete verification
    pub fn verify(&self, generated: &RgbImage) -> VerificationResult {
        let ssim = self.suite.ssim.compare(&self.reference, generated);
        let psnr = self.suite.psnr.compute(&self.reference, generated);
        let delta_e = self.suite.delta_e.mean_delta(&self.reference, generated);
        let phash_dist = self.suite.phash.distance(
            self.suite.phash.compute(&self.reference),
            self.suite.phash.compute(generated),
        );

        VerificationResult {
            ssim,
            psnr,
            mean_delta_e: delta_e,
            phash_distance: phash_dist,
            hypotheses_status: self.evaluate_hypotheses(),
        }
    }
}
```

### 4.3 Performance Strategy

Pixel-perfect verification is computationally intensive. The implementation must use:

*   **Rayon Parallelism**: Use `par_iter()` for pixel-wise operations (ΔE calculation, SSIM windows).
*   **SIMD Optimization**: Use `wide` or `packed_simd` crates for vectorized color space conversions (RGB -> Lab).
*   **Downscaling**: For rapid "L1" checks, perform comparisons on 50% downscaled images before full resolution.
*   **Caching**: Cache perceptual hashes of reference images to avoid re-computation.

---

## 5. Implementation Plan (v2.1)

### Phase 5: Popperian Framework 🆕 ✅

- [x] Add `CoverageHypothesis` struct with falsification conditions (`FalsifiableHypothesis`)
- [x] Implement `FalsifiabilityGate` with gateway threshold (15/25 default)
- [x] Add confidence interval calculation for coverage scores
- [x] Integrate with PAIML Popper Score methodology

### Phase 6: Pixel-Perfect Metrics 🆕 ✅

- [x] Implement SSIM metric with window-based comparison (8x8 default)
- [x] Implement PSNR metric with quality classification
- [x] Implement CIEDE2000 ΔE calculation (Lab color space)
- [x] Implement perceptual hashing (AHash, DHash, PHash algorithms)
- [x] Add `PixelVerificationSuite` orchestrator

### Phase 7: First-Class Terminal Output 🆕 ✅

- [x] Create `RichTerminalHeatmap` with box-drawing characters
- [x] Implement `ScoreBar` visual progress bars
- [x] Add gap detection with actionable recommendations
- [x] Add hypothesis falsification status display
- [x] Support ANSI true-color (24-bit) output

### Phase 8: Statistical Rigor 🆕 ✅

- [x] Add confidence interval calculation (Wilson score)
- [ ] Implement bootstrap resampling for robustness (deferred)
- [ ] Add statistical significance tests for regression detection (deferred)
- [x] Document measurement uncertainty (via ConfidenceInterval)

### Phase 9: Performance & Config 🆕 ✅

- [x] Implement parallel processing abstraction (Rayon-ready, sequential fallback)
- [x] Add `[pixel_coverage]` configuration parsing (`PixelCoverageConfig`)
- [x] Implement `--no-color` and `--json` output modes (`OutputMode` enum)

---

## 6. Testing Strategy (Enhanced)

### 6.1 Falsification Test Matrix

| Hypothesis | Test | Falsification Criterion | Status |
|------------|------|-------------------------|--------|
| H₀-SSIM-01 | Reference comparison | SSIM < 0.99 | ✅ Done |
| H₀-PSNR-01 | Signal quality | PSNR < 30 dB | ✅ Done |
| H₀-DE-01 | Color accuracy | Mean ΔE₀₀ > 2.0 | ✅ Done |
| H₀-PHASH-01 | Content identity | Hamming distance > 5 | ✅ Done |
| H₀-COV-01 | Coverage threshold | Coverage < 85% | ✅ Done |

### 6.2 Property-Based Tests

```rust
proptest! {
    /// Property: SSIM is symmetric
    #[test]
    fn ssim_symmetric(img1: TestImage, img2: TestImage) {
        let ssim = SsimMetric::default();
        let forward = ssim.compare(&img1, &img2);
        let reverse = ssim.compare(&img2, &img1);
        prop_assert!((forward.score - reverse.score).abs() < 0.001);
    }

    /// Property: Identical images have SSIM = 1.0
    #[test]
    fn ssim_identity(img: TestImage) {
        let ssim = SsimMetric::default();
        let result = ssim.compare(&img, &img);
        prop_assert!((result.score - 1.0).abs() < 0.0001);
    }

    /// Property: ΔE₀₀ satisfies triangle inequality
    #[test]
    fn delta_e_triangle(c1: Lab, c2: Lab, c3: Lab) {
        let metric = CieDe2000Metric::default();
        let d12 = metric.delta_e(&c1, &c2);
        let d23 = metric.delta_e(&c2, &c3);
        let d13 = metric.delta_e(&c1, &c3);
        prop_assert!(d13 <= d12 + d23 + 0.001); // Epsilon for float
    }
}
```

### 6.3 Mutation Testing Scope

Following PAIML methodology [1], mutation testing targets core domain logic:

```toml
# mutants.toml
[scope]
include = [
    "src/pixel_coverage/metrics/*.rs",   # SSIM, PSNR, ΔE
    "src/pixel_coverage/tracker.rs",      # Coverage tracking
    "src/pixel_coverage/hypothesis.rs",   # Falsification logic
]

exclude = [
    "src/pixel_coverage/heatmap.rs",      # Rendering (visual verification)
    "tests/**/*.rs",
]

[thresholds]
mutation_score = 80  # ≥80% mutants killed
```

---

## 7. Peer-Reviewed References

1. **Popper, K.R.** (1959). *The Logic of Scientific Discovery*. Hutchinson & Co. [Foundational work on falsificationism]

2. **Selan, J.** (2012). "Cinematic Color: From Photons to Film." *SIGGRAPH Course Notes*. ACM. [Film-study color pipeline]

3. **Sharma, G., Wu, W., & Dalal, E.N.** (2005). "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations." *Color Research & Application*, 30(1), 21-30. doi:10.1002/col.20070 [CIEDE2000 reference implementation]

4. **Luo, M.R., Cui, G., & Rigg, B.** (2001). "The Development of the CIE 2000 Colour-Difference Formula: CIEDE2000." *Color Research & Application*, 26(5), 340-350. doi:10.1002/col.1049 [Original CIEDE2000 formula]

5. **Wang, Z., Bovik, A.C., Sheikh, H.R., & Simoncelli, E.P.** (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity." *IEEE Transactions on Image Processing*, 13(4), 600-612. doi:10.1109/TIP.2003.819861 [SSIM foundational paper]

6. **Zauner, C.** (2010). "Implementation and Benchmarking of Perceptual Image Hash Functions." *Master's Thesis*, Upper Austria University of Applied Sciences. [Perceptual hashing comparison]

7. **Hamadouche, M., et al.** (2023). "Robust Perceptual Fingerprint Image Hashing: A Comparative Study." *International Journal of Biometrics*, 15(1). doi:10.1504/IJBM.2023.127724 [Recent perceptual hashing research]

8. **Jia, Y., & Harman, M.** (2011). "An Analysis and Survey of the Development of Mutation Testing." *IEEE Transactions on Software Engineering*, 37(5), 649-678. doi:10.1109/TSE.2010.62 [Mutation testing survey]

9. **Georges, A., Buytaert, D., & Eeckhout, L.** (2007). "Statistically Rigorous Java Performance Evaluation." *ACM SIGPLAN Notices*, 42(10), 57-76. doi:10.1145/1297027.1297033 [Statistical rigor in benchmarking]

10. **Alégroth, E., Feldt, R., & Kolström, P.** (2017). "Visual GUI Testing in Practice: Challenges, Problems and Limitations." *Empirical Software Engineering*, 22(3), 1109-1158. doi:10.1007/s10664-016-9454-2 [Visual regression testing research]

---

## 8. Success Criteria (v2.1)

### Existing (v1.0) ✅
- ✅ PNG heatmap renders correctly with Viridis palette
- ✅ Legend shows coverage scale
- ✅ Gaps are visually highlighted
- ✅ Combined report includes both line and pixel coverage
- ✅ Test coverage ≥85%
- ✅ Zero clippy warnings

### New (v2.1) 🆕
- ✅ Falsifiability gateway implemented and enforced (falsification.rs: 32 tests)
- ✅ SSIM metric with window size 11, threshold 0.99 (metrics.rs)
- ✅ CIEDE2000 ΔE with JND threshold 1.0 (metrics.rs)
- ✅ Perceptual hash with Hamming distance ≤ 5 (metrics.rs)
- ✅ Rich terminal output with score bars and gap analysis (terminal.rs: 18 tests)
- ✅ Confidence intervals via Wilson score (terminal.rs)
- ✅ Property-based tests with proptest (30 tests, L2 falsification layer)
- ✅ Parallel processing abstractions (parallel.rs: 16 tests)
- ✅ Configuration via probar.toml schema (config.rs: 14 tests)
- [ ] Mutation score ≥80% (requires cargo-mutants run)
- [ ] Rayon parallelism (requires rayon crate dependency)

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **Falsifiability** | Property of a claim being testable and potentially refutable |
| **JND** | Just Noticeable Difference - minimum perceptible change |
| **SSIM** | Structural Similarity Index Measure (0-1 scale) |
| **PSNR** | Peak Signal-to-Noise Ratio (dB scale, higher is better) |
| **ΔE₀₀** | CIEDE2000 color difference metric |
| **Perceptual Hash** | Content fingerprint robust to minor changes |
| **Mutation Score** | Percentage of injected bugs caught by tests |
| **Gateway** | Mandatory threshold that blocks further evaluation if failed |

---

## 10. Configuration Schema

The pixel coverage system is configured via `probar.toml`.

```toml
[pixel_coverage]
# Enable pixel-perfect verification
enabled = true

# Primary methodology
methodology = "falsification" # or "simple"

[pixel_coverage.thresholds]
# Minimum coverage percentage
min_coverage = 85.0
# Maximum allowed gap size (percent of screen)
max_gap_size = 5.0
# Falsifiability gateway threshold (0-25)
falsifiability_threshold = 15.0

[pixel_coverage.verification]
# Structural Similarity Index (0.0 - 1.0)
ssim_threshold = 0.99
# CIEDE2000 Color Difference (JND)
delta_e_threshold = 1.0
# Perceptual Hash distance
phash_distance = 5

[pixel_coverage.output]
# Generate PNG heatmap
heatmap = true
# Enable rich terminal output
terminal_gui = true
# Palette: "viridis", "magma", "inferno", "plasma"
palette = "viridis"
```

---

## Appendix A: Terminal Output Color Codes

```rust
/// ANSI true-color (24-bit) codes for terminal output
pub mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";

    /// RGB color escape sequence
    pub fn rgb_fg(r: u8, g: u8, b: u8) -> String {
        format!("\x1b[38;2;{};{};{}m", r, g, b)
    }

    pub fn rgb_bg(r: u8, g: u8, b: u8) -> String {
        format!("\x1b[48;2;{};{};{}m", r, g, b)
    }

    // Semantic colors
    pub const PASS: &str = "\x1b[32m";   // Green
    pub const FAIL: &str = "\x1b[31m";   // Red
    pub const WARN: &str = "\x1b[33m";   // Yellow
    pub const INFO: &str = "\x1b[36m";   // Cyan
}
```

---

## Appendix B: Lab Color Space Conversion

```rust
/// Convert RGB to CIE Lab color space for ΔE calculation
pub fn rgb_to_lab(rgb: Rgb) -> Lab {
    // Step 1: RGB to XYZ (sRGB with D65 illuminant)
    let r = linearize(rgb.r as f32 / 255.0);
    let g = linearize(rgb.g as f32 / 255.0);
    let b = linearize(rgb.b as f32 / 255.0);

    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // Step 2: XYZ to Lab (D65 reference white)
    const D65: (f32, f32, f32) = (0.95047, 1.0, 1.08883);

    let fx = lab_f(x / D65.0);
    let fy = lab_f(y / D65.1);
    let fz = lab_f(z / D65.2);

    Lab {
        l: 116.0 * fy - 16.0,
        a: 500.0 * (fx - fy),
        b: 200.0 * (fy - fz),
    }
}

fn linearize(v: f32) -> f32 {
    if v > 0.04045 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        v / 12.92
    }
}

fn lab_f(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    if t > DELTA.powi(3) {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}
```