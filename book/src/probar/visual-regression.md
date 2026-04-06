# Visual Regression Testing

Visual regression testing catches unintended UI changes by comparing screenshots against baselines. Probar provides pure Rust image comparison with perceptual diffing.

## Quick Start

```rust
use jugar_probar::{VisualRegressionTester, VisualRegressionConfig};

// Create tester with configuration
let tester = VisualRegressionTester::new(
    VisualRegressionConfig::default()
        .with_threshold(0.01)       // 1% of pixels can differ
        .with_color_threshold(10)   // Allow minor color variations
        .with_baseline_dir("__baselines__")
);

// Compare screenshot against baseline
let screenshot = capture_screenshot(); // Your screenshot bytes (PNG)
let result = tester.compare_against_baseline("login-page", &screenshot)?;

assert!(result.matches, "Visual regression detected!");
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `threshold` | 0.01 | Percentage of pixels that can differ (0.0-1.0) |
| `color_threshold` | 10 | Per-pixel color difference allowed (0-255) |
| `baseline_dir` | `__baselines__` | Directory for baseline images |
| `diff_dir` | `__diffs__` | Directory for diff images on failure |
| `update_baselines` | false | Automatically update baselines |

```rust
let config = VisualRegressionConfig::default()
    .with_threshold(0.05)         // 5% tolerance
    .with_color_threshold(20)     // More lenient color matching
    .with_baseline_dir("snapshots")
    .with_update_baselines(true); // Update on mismatch
```

## Direct Image Comparison

Compare two images directly without baseline files:

```rust
use jugar_probar::VisualRegressionTester;

let tester = VisualRegressionTester::default();

// Compare two PNG images
let result = tester.compare_images(&actual_png, &expected_png)?;

println!("Matches: {}", result.matches);
println!("Diff pixels: {}", result.diff_pixel_count);
println!("Diff percentage: {:.2}%", result.diff_percentage);
println!("Max color diff: {}", result.max_color_diff);
println!("Avg color diff: {:.1}", result.avg_color_diff);
```

## ImageDiffResult

The comparison result provides detailed metrics:

```rust
pub struct ImageDiffResult {
    pub matches: bool,           // Within threshold?
    pub diff_pixel_count: usize, // Number of differing pixels
    pub total_pixels: usize,     // Total pixels compared
    pub diff_percentage: f64,    // Percentage different (0-100)
    pub max_color_diff: u32,     // Maximum color difference found
    pub avg_color_diff: f64,     // Average color difference
    pub diff_image: Option<Vec<u8>>, // PNG diff visualization (red = difference)
}

// Utility methods
assert!(result.is_identical());           // No differences at all
assert!(result.within_threshold(0.02));   // Custom threshold check
```

## Masking Dynamic Areas

Exclude dynamic areas (timestamps, ads, animations) from comparison:

```rust
use jugar_probar::{ScreenshotComparison, MaskRegion};

let comparison = ScreenshotComparison::new()
    .with_threshold(0.01)
    .with_max_diff_pixels(100)
    .with_mask(MaskRegion::new(10, 10, 200, 50))  // Header area
    .with_mask(MaskRegion::new(0, 500, 300, 100)); // Footer area

// Use with your comparison logic
for mask in &comparison.mask_regions {
    if mask.contains(x, y) {
        // Skip this pixel in comparison
    }
}
```

## Perceptual Diff

For human-vision-weighted comparison, use the perceptual diff function:

```rust
use jugar_probar::perceptual_diff;
use image::Rgba;

let pixel_a = Rgba([255, 0, 0, 255]);   // Red
let pixel_b = Rgba([200, 50, 50, 255]); // Darker red

let diff = perceptual_diff(pixel_a, pixel_b);

// Uses weighted RGB based on human perception:
// - Red: 0.299
// - Green: 0.587  (most sensitive)
// - Blue: 0.114
```

## Baseline Management

### Creating Baselines

```rust
let config = VisualRegressionConfig::default()
    .with_update_baselines(true);

let tester = VisualRegressionTester::new(config);

// First run creates the baseline
let result = tester.compare_against_baseline("home-page", &screenshot)?;
// Baseline saved to __baselines__/home-page.png
```

### Updating Baselines

```rust
// Set update_baselines when you want to accept new changes
let config = VisualRegressionConfig::default()
    .with_update_baselines(true);
```

### Diff Images

When comparison fails, a diff image is saved showing differences in red:

```
__diffs__/
  home-page_diff.png   # Red overlay on differing pixels
```

## Integration with TUI

Visual regression works great with TUI screenshots:

```rust
use jugar_probar::{TuiTestBackend, VisualRegressionTester};
use ratatui::Terminal;

let backend = TuiTestBackend::new(80, 24);
let mut terminal = Terminal::new(backend)?;

// Render your UI
terminal.draw(|f| {
    render_app(f, &app_state);
})?;

// Capture frame as image
let frame = terminal.backend().current_frame();
let screenshot = frame.to_png()?;

// Compare against baseline
let tester = VisualRegressionTester::default();
let result = tester.compare_against_baseline("app-home", &screenshot)?;
```

## Best Practices

1. **Use meaningful names** - Name baselines after the page/component being tested
2. **Set appropriate thresholds** - Too strict causes flakiness, too loose misses bugs
3. **Mask dynamic content** - Exclude timestamps, ads, random content
4. **Review diff images** - When tests fail, examine the diff to understand changes
5. **Version control baselines** - Commit baselines so the whole team uses the same
6. **Update intentionally** - Only enable `update_baselines` when accepting changes

## Examples

```bash
# Run visual regression demo
cargo run --example visual_regression_demo -p jugar-probar
```

## See Also

- [Pixel Coverage](./pixel-coverage.md) - Heatmap visualization
- [Media Recording](./media-recording.md) - Screenshot and video capture
- [PNG Screenshots](./media-png.md) - PNG export utilities
