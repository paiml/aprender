# Pixel Coverage Heatmaps (PIXEL-001 v2.1)

> **Pixel-Perfect Visual Coverage Analysis**: See exactly which screen regions are tested with statistical rigor and Popperian falsification

Probar's pixel coverage system provides comprehensive visual verification:
- **Heatmap visualization** with Viridis/Magma/Heat color palettes
- **Statistical rigor** via Wilson score confidence intervals
- **Popperian falsification** with FalsifiabilityGate (15/25 threshold)
- **Pixel-perfect metrics**: SSIM, PSNR, CIEDE2000 (ΔE₀₀), Perceptual Hash
- **Rich terminal output** with score bars and ANSI colors

## Quick Start

```rust
use jugar_probar::pixel_coverage::{
    PixelCoverageTracker, PixelRegion, PngHeatmap, ColorPalette
};

// Create tracker for 800x600 screen with 20x15 grid
let mut tracker = PixelCoverageTracker::new(800, 600, 20, 15);

// Record covered regions during tests
tracker.record_region(PixelRegion::new(0, 0, 800, 100));   // Header
tracker.record_region(PixelRegion::new(0, 100, 400, 400)); // Left panel
tracker.record_region(PixelRegion::new(0, 500, 800, 100)); // Footer

// Generate PNG heatmap
PngHeatmap::new(800, 600)
    .with_palette(ColorPalette::viridis())
    .with_title("UI Coverage")
    .with_legend()
    .with_gap_highlighting()
    .export_to_file(tracker.cells(), "coverage.png")
    .unwrap();
```

## CLI Usage

Generate heatmaps from the command line:

```bash
# Basic heatmap
probar coverage --png output.png

# With all options
probar coverage --png heatmap.png \
  --palette viridis \
  --legend \
  --gaps \
  --title "My Coverage Report" \
  --width 1920 \
  --height 1080

# Export JSON report
probar coverage --json report.json

# Available palettes: viridis, magma, heat
probar coverage --png output.png --palette magma
```

## Color Palettes

### Viridis (Default)

Perceptually uniform, colorblind-safe palette. Dark purple (0%) to yellow (100%).

```rust
PngHeatmap::new(800, 600)
    .with_palette(ColorPalette::viridis())
```

![Viridis Heatmap](../assets/coverage_viridis.png)

### Magma

Dark to bright palette. Black (0%) through purple/magenta to light yellow (100%).

```rust
PngHeatmap::new(800, 600)
    .with_palette(ColorPalette::magma())
```

![Magma Heatmap](../assets/coverage_magma.png)

### Heat

Classic heat map. Black (0%) through red/orange/yellow to white (100%).

```rust
PngHeatmap::new(800, 600)
    .with_palette(ColorPalette::heat())
```

![Heat Heatmap](../assets/coverage_heat.png)

## Title and Subtitle

Add text labels to your heatmaps:

```rust
PngHeatmap::new(800, 600)
    .with_title("Coverage Analysis")
    .with_subtitle("Sprint 42 - Login Flow")
    .with_legend()
    .export_to_file(tracker.cells(), "output.png")
    .unwrap();
```

## Gap Highlighting

Highlight untested regions with a red border:

```rust
PngHeatmap::new(800, 600)
    .with_gap_highlighting()  // Red 3px border on 0% cells
    .export_to_file(tracker.cells(), "output.png")
    .unwrap();
```

## Combined Coverage Report

Combine line coverage (from GUI testing) with pixel coverage:

```rust
use jugar_probar::pixel_coverage::{
    LineCoverageReport, CombinedCoverageReport, PngHeatmap
};

// Line coverage from GUI tests
let line_report = LineCoverageReport::new(
    0.90,  // 90% element coverage
    1.0,   // 100% screen coverage
    0.85,  // 85% journey coverage
    22,    // total elements
    20,    // covered elements
);

// Pixel coverage from tracker
let pixel_report = tracker.generate_report();

// Combined report (50/50 weighted average)
let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

// Print summary
println!("{}", combined.summary());
// Output:
// Combined Coverage Report
// ========================
// Line Coverage:  90.0% (20/22 elements)
// Pixel Coverage: 75.0% (225/300 cells)
// Overall Score:  82.5%
// Threshold Met:  ✓

// Generate PNG with stats panel
PngHeatmap::new(800, 700)
    .with_title("Combined Coverage")
    .with_combined_stats(&combined)
    .with_legend()
    .export_to_file(tracker.cells(), "combined.png")
    .unwrap();
```

![Combined Coverage Report](../assets/coverage_combined.png)

## Terminal Heatmap (STDOUT)

Display coverage in the terminal with Unicode blocks:

```rust
let terminal = tracker.terminal_heatmap();
println!("{}", terminal.render_with_border());
println!("{}", terminal.legend());
```

Output:
```
┌────────────────────┐
│████████████████████│
│████████████        │
│████████████████████│
│████████████████████│
│        ████████████│
│████████████████████│
└────────────────────┘
Legend: █ = 76-100%  ▓ = 51-75%  ▒ = 26-50%  ░ = 1-25%    = 0%
```

## Coverage Report

Get detailed coverage metrics:

```rust
let report = tracker.generate_report();

println!("Overall Coverage: {:.1}%", report.overall_coverage * 100.0);
println!("Covered Cells: {}/{}", report.covered_cells, report.total_cells);
println!("Meets Threshold: {}", report.meets_threshold);
println!("Uncovered Regions: {}", report.uncovered_regions.len());
```

## Defining UI Regions

Track specific UI components:

```rust
// Define your UI layout
fn my_app_layout() -> Vec<(&'static str, PixelRegion)> {
    vec![
        ("header", PixelRegion::new(0, 0, 800, 60)),
        ("sidebar", PixelRegion::new(0, 60, 200, 500)),
        ("main_content", PixelRegion::new(200, 60, 600, 400)),
        ("footer", PixelRegion::new(0, 560, 800, 40)),
    ]
}

// Track during tests
let layout = my_app_layout();
for (name, region) in &layout {
    if test_covers_region(name) {
        tracker.record_region(*region);
    }
}
```

## Trueno-viz Style Output

PngHeatmap uses trueno-viz style rendering with:

- **Margins**: Configurable padding around the plot area
- **Background**: White background (configurable)
- **Borders**: Optional cell borders
- **Legend**: Color scale bar with labels
- **Title area**: Top section for title/subtitle text

```rust
PngHeatmap::new(800, 600)
    .with_margin(40)                    // 40px margin
    .with_background(Rgb::new(255, 255, 255))  // White
    .with_borders(true)                 // Show cell borders
    .with_legend()
    .with_title("My Heatmap")
    .export_to_file(cells, "output.png")
    .unwrap();
```

## Running the Example

```bash
cargo run --example pixel_coverage_heatmap -p jugar-probar
```

Output:
```
Pixel Coverage Heatmap Example
===============================

Step 1: Creating coverage tracker (10x8 grid on 800x600 screen)...

Step 2: Simulating coverage with gaps...
  ✓ Header area covered (rows 0-1)
  ✓ Left sidebar covered
  ✓ Right content covered
  ⚠ Middle content area is a GAP (uncovered)
  ✓ Footer area covered

Step 3: Coverage Report
  Overall Coverage: 75.0%
  Covered Cells: 60/80
  Uncovered Regions: 1
  Meets Threshold: ✗

Step 4: Generating PNG heatmaps...
  ✓ Viridis heatmap: /tmp/coverage_viridis.png
  ✓ Magma heatmap: /tmp/coverage_magma.png
  ✓ Heat heatmap: /tmp/coverage_heat.png

...

✅ Pixel coverage heatmap example completed!
```

## API Reference

### PixelCoverageTracker

| Method | Description |
|--------|-------------|
| `new(width, height, cols, rows)` | Create tracker |
| `record_point(x, y)` | Record single pixel |
| `record_region(region)` | Record rectangular region |
| `generate_report()` | Get PixelCoverageReport |
| `cells()` | Get coverage grid |
| `terminal_heatmap()` | Get terminal renderer |

### PngHeatmap

| Method | Description |
|--------|-------------|
| `new(width, height)` | Create PNG exporter |
| `with_palette(palette)` | Set color palette |
| `with_title(text)` | Add title text |
| `with_subtitle(text)` | Add subtitle text |
| `with_legend()` | Show color legend |
| `with_gap_highlighting()` | Red border on gaps |
| `with_margin(px)` | Set margin size |
| `with_combined_stats(report)` | Add stats panel |
| `export(cells)` | Export to bytes |
| `export_to_file(cells, path)` | Export to file |

### ColorPalette

| Method | Description |
|--------|-------------|
| `viridis()` | Colorblind-safe (default) |
| `magma()` | Dark to bright |
| `heat()` | Classic heat map |
| `interpolate(coverage)` | Get color for 0.0-1.0 |

### LineCoverageReport

| Field | Description |
|-------|-------------|
| `element_coverage` | Percentage 0.0-1.0 |
| `screen_coverage` | Percentage 0.0-1.0 |
| `journey_coverage` | Percentage 0.0-1.0 |
| `total_elements` | Total trackable elements |
| `covered_elements` | Elements exercised |

### CombinedCoverageReport

| Method | Description |
|--------|-------------|
| `from_parts(line, pixel)` | Create from reports |
| `with_weights(line_w, pixel_w)` | Custom weighting |
| `summary()` | Text summary |
| `overall_score` | Weighted average |
| `meets_threshold` | Above 80% default |

## Visual Regression Testing

Probar includes a `visual_regression` module for verifying PNG output consistency:

```rust
use jugar_probar::pixel_coverage::heatmap::visual_regression::*;

// Generate deterministic test data
let cells = reference_gradient_cells(10, 15);

// Generate and checksum PNG
let png = PngHeatmap::new(800, 600).export(&cells)?;
let checksum = compute_checksum(&png);

// Compare images with tolerance
let result = compare_png_with_tolerance(&reference, &generated, 5)?;
assert!(result.matches);
println!("Diff: {:.2}%, Max diff: {}", result.diff_percentage, result.max_diff);
```

### Reference Cell Generators

| Function | Description |
|----------|-------------|
| `reference_gradient_cells(rows, cols)` | Diagonal gradient pattern |
| `reference_gap_cells(rows, cols)` | Gradient with deterministic gaps |
| `reference_uniform_cells(rows, cols, coverage)` | Uniform coverage value |

Example gradient pattern with gaps (generated by `reference_gap_cells`):

![Pattern Heatmap with Gaps](../assets/coverage_pattern.png)

### ComparisonResult

| Field | Description |
|-------|-------------|
| `matches` | Whether images match within tolerance |
| `diff_percentage` | Percentage of differing pixels |
| `max_diff` | Maximum per-channel color difference |
| `diff_count` | Number of differing pixels |
| `total_pixels` | Total pixels compared |

---

## PIXEL-001 v2.1 Features

### Popperian Falsification

The falsification framework implements Karl Popper's scientific methodology for coverage testing:

```rust
use jugar_probar::pixel_coverage::{
    FalsifiabilityGate, FalsifiableHypothesis
};

// Create gate with 15/25 threshold (default)
let gate = FalsifiabilityGate::new(15.0);

// Build falsifiable hypothesis
let hypothesis = FalsifiableHypothesis::coverage_threshold("H0-COV-01", 0.95);

// Evaluate with actual coverage
let result = hypothesis.evaluate(0.98); // 98% coverage

// Check if falsified (coverage < threshold)
println!("Falsified: {}", result.falsified); // false (98% >= 95%)

// Gate evaluation
let gate_result = gate.evaluate(&result);
println!("Gate passed: {}", gate_result.is_passed());
println!("Score: {}", gate_result.score()); // 20.0
```

#### Hypothesis Types

| Constructor | Description | Falsification Criterion |
|-------------|-------------|------------------------|
| `coverage_threshold(id, threshold)` | Coverage must exceed threshold | Coverage < threshold |
| `max_gap_size(id, max_gap)` | No gap larger than max | Gap > max_gap |
| `ssim_threshold(id, min_ssim)` | SSIM must exceed minimum | SSIM < min_ssim |

### Wilson Score Confidence Intervals

Statistical rigor for coverage proportions:

```rust
use jugar_probar::pixel_coverage::ConfidenceInterval;

// Calculate 95% Wilson score interval
let ci = ConfidenceInterval::wilson_score(
    85,   // successes (covered cells)
    100,  // total (all cells)
    0.95, // confidence level
);

println!("Coverage: 85% [{:.1}%, {:.1}%]",
    ci.lower * 100.0, ci.upper * 100.0);
// Output: Coverage: 85% [76.7%, 90.9%]
```

### Score Bars

Visual progress indicators with threshold highlighting:

```rust
use jugar_probar::pixel_coverage::{ScoreBar, OutputMode};

let bar = ScoreBar::new("Coverage", 0.85, 0.80); // 85% vs 80% threshold
println!("{}", bar.render(OutputMode::RichAnsi));
// Output: [32m        Coverage:  85.0%  █████████████████████    [0m
```

### Rich Terminal Output

Full-featured terminal heatmap with ANSI colors:

```rust
use jugar_probar::pixel_coverage::{RichTerminalHeatmap, OutputMode};

let heatmap = RichTerminalHeatmap::new(cells)
    .with_title("Coverage Analysis")
    .with_mode(OutputMode::RichAnsi);

println!("{}", heatmap.render());
```

Output modes:
- `RichAnsi`: 24-bit true color (default)
- `NoColorAscii`: Plain ASCII for `NO_COLOR` environments
- `Json`: Machine-readable for CI tools

### Pixel-Perfect Metrics

#### SSIM (Structural Similarity Index)

```rust
use jugar_probar::pixel_coverage::{SsimMetric, Rgb};

let ssim = SsimMetric::default(); // 8x8 window
let result = ssim.compare(&reference, &generated, 800, 600);

println!("SSIM: {:.4}", result.score); // 0.0 to 1.0
println!("Per-channel: {:?}", result.channel_scores);
```

| Score | Quality |
|-------|---------|
| > 0.99 | Identical |
| 0.95-0.99 | Excellent |
| 0.90-0.95 | Good |
| < 0.90 | Degraded |

#### PSNR (Peak Signal-to-Noise Ratio)

```rust
use jugar_probar::pixel_coverage::PsnrMetric;

let psnr = PsnrMetric::default();
let result = psnr.compare(&reference, &generated);

println!("PSNR: {:.1} dB", result.psnr);
println!("Quality: {:?}", result.quality);
```

| dB | Quality |
|----|---------|
| > 40 | Excellent |
| 30-40 | Good |
| 20-30 | Acceptable |
| < 20 | Poor |

#### CIEDE2000 (ΔE₀₀ Color Difference)

```rust
use jugar_probar::pixel_coverage::{CieDe2000Metric, Lab};

let metric = CieDe2000Metric::default();
let lab1 = Lab::from_rgb(&Rgb::new(255, 0, 0));
let lab2 = Lab::from_rgb(&Rgb::new(250, 5, 5));

let delta_e = metric.delta_e(&lab1, &lab2);
println!("ΔE₀₀: {:.2}", delta_e);
```

| ΔE₀₀ | Perception |
|------|------------|
| < 1.0 | Imperceptible |
| 1.0-2.0 | Perceptible on close inspection |
| 2.0-10.0 | Perceptible at a glance |
| > 10.0 | Colors appear different |

#### Perceptual Hashing

```rust
use jugar_probar::pixel_coverage::{PerceptualHash, PhashAlgorithm};

let hasher = PerceptualHash::new(PhashAlgorithm::PHash);
let hash1 = hasher.compute(&image1, 100, 100);
let hash2 = hasher.compute(&image2, 100, 100);

let distance = hasher.hamming_distance(hash1, hash2);
println!("Hamming distance: {}", distance); // 0 = identical
```

### Configuration Schema

Configure pixel coverage via `probar.toml`:

```toml
[pixel_coverage]
enabled = true
methodology = "popperian"

[pixel_coverage.thresholds]
minimum = 0.60
target = 0.85
complete = 1.0
falsifiability_gateway = 15.0

[pixel_coverage.verification]
ssim_threshold = 0.95
psnr_threshold = 30.0
delta_e_threshold = 2.0
phash_max_distance = 5

[pixel_coverage.output]
format = "rich_ansi"
show_heatmap = true
show_confidence_intervals = true
show_score_bars = true

[pixel_coverage.performance]
parallel = true
threads = 0  # auto-detect
batch_size = 1024
```

### Calculator Demo (Dogfooding Example)

Run the calculator demo with full PIXEL-001 v2.1 integration:

```bash
cargo run -p showcase-calculator --example gui_coverage_report
```

Output:
```
===============================================================
    SHOWCASE CALCULATOR - PIXEL-PERFECT COVERAGE (v2.1)
===============================================================

--- GUI ELEMENT COVERAGE ---
  [32m        Elements: 100.0%  █████████████████████████[0m
  Covered: 21/21 elements, 4/4 screens

--- PIXEL-LEVEL COVERAGE ---
  [32m          Pixels: 100.0%  █████████████████████████[0m
  Cells: 24/24 covered

--- STATISTICAL RIGOR (Wilson Score 95% CI) ---
  Pixel Coverage: 100.0% [86.2%, 100.0%]
  GUI Coverage:   100.0% [84.5%, 100.0%]

--- POPPERIAN FALSIFICATION ---
  H0-PIX-CALC-01: [NOT FALSIFIED]
    Actual: 100.0% vs Threshold: 100.0%
  Gate Status: [PASSED]

===============================================================
  [OK] STATUS: PIXEL-PERFECT COVERAGE ACHIEVED!
===============================================================
```
