//! TUI Comparison Engine
//!
//! Compares two TUI outputs (e.g., ttop vs ptop) using multiple metrics:
//! - CLD (Character-Level Difference): Percentage of differing characters
//! - ΔE00 (CIEDE2000): Perceptual color difference
//! - SSIM (Structural Similarity): Layout similarity

use std::collections::HashMap;
use std::fmt::Write;

use presentar_core::Color;

use super::color_diff::{ciede2000, rgb_to_lab, Rgb};
use crate::direct::CellBuffer;

/// Configuration for TUI comparison
#[derive(Debug, Clone)]
pub struct TuiComparisonConfig {
    /// Character-level difference threshold (0.0-1.0)
    pub cld_threshold: f64,
    /// CIEDE2000 color difference threshold
    pub delta_e_threshold: f64,
    /// Structural similarity threshold (0.0-1.0)
    pub ssim_threshold: f64,
    /// Per-panel thresholds (optional stricter limits)
    pub panel_thresholds: HashMap<String, PanelThreshold>,
}

impl Default for TuiComparisonConfig {
    fn default() -> Self {
        Self {
            cld_threshold: 0.01,    // <1% character diff
            delta_e_threshold: 2.0, // Barely perceptible color
            ssim_threshold: 0.95,   // 95% structural match
            panel_thresholds: HashMap::new(),
        }
    }
}

/// Per-panel threshold overrides
#[derive(Debug, Clone)]
pub struct PanelThreshold {
    pub cld: Option<f64>,
    pub delta_e: Option<f64>,
    pub ssim: Option<f64>,
}

/// Result of comparing two cells
#[derive(Debug, Clone)]
pub struct DiffCell {
    pub x: u16,
    pub y: u16,
    pub reference_char: char,
    pub target_char: char,
    pub reference_fg: Rgb,
    pub target_fg: Rgb,
    pub delta_e: f64,
}

/// Result of panel comparison
#[derive(Debug, Clone)]
pub struct PanelResult {
    pub name: String,
    pub bounds: (u16, u16, u16, u16), // x, y, width, height
    pub cld: f64,
    pub delta_e: f64,
    pub ssim: f64,
    pub passed: bool,
}

/// Full comparison result
#[derive(Debug)]
pub struct TuiComparisonResult {
    /// Overall pass/fail
    pub passed: bool,
    /// Character-level difference (0.0-1.0)
    pub cld: f64,
    /// Average CIEDE2000 color difference
    pub delta_e: f64,
    /// Structural similarity index
    pub ssim: f64,
    /// Per-panel results
    pub panel_results: Vec<PanelResult>,
    /// Cells that differ
    pub diff_cells: Vec<DiffCell>,
    /// Total cells compared
    pub total_cells: usize,
    /// Cells with character differences
    pub char_diff_count: usize,
    /// Cells with color differences (ΔE > 2.0)
    pub color_diff_count: usize,
}

/// Compute color delta-E between two color values.
#[inline]
fn compute_color_delta(ref_fg: Rgb, tgt_fg: Rgb) -> f64 {
    let ref_lab = rgb_to_lab(ref_fg);
    let tgt_lab = rgb_to_lab(tgt_fg);
    ciede2000(ref_lab, tgt_lab)
}

/// Cell comparison result (used internally).
struct CellComparison {
    char_differs: bool,
    color_differs: bool,
    delta_e: f64,
    ref_fg: Rgb,
    tgt_fg: Rgb,
}

/// Compare two cells and return comparison result.
#[inline]
fn compare_cells(ref_cell: &crate::direct::Cell, tgt_cell: &crate::direct::Cell) -> CellComparison {
    let ref_fg = color_to_rgb(&ref_cell.fg);
    let tgt_fg = color_to_rgb(&tgt_cell.fg);
    let delta_e = compute_color_delta(ref_fg, tgt_fg);
    CellComparison {
        char_differs: ref_cell.symbol != tgt_cell.symbol,
        color_differs: delta_e > 2.0,
        delta_e,
        ref_fg,
        tgt_fg,
    }
}

/// Compute CLD and average delta-E metrics.
#[inline]
fn compute_comparison_metrics(
    char_diff_count: usize,
    total_delta_e: f64,
    total_cells: usize,
) -> (f64, f64) {
    if total_cells == 0 {
        return (0.0, 0.0);
    }
    let cld = char_diff_count as f64 / total_cells as f64;
    let avg_delta_e = total_delta_e / total_cells as f64;
    (cld, avg_delta_e)
}

/// Check if result passes the configured thresholds.
#[inline]
fn check_thresholds(cld: f64, avg_delta_e: f64, ssim: f64, config: &TuiComparisonConfig) -> bool {
    cld < config.cld_threshold
        && avg_delta_e < config.delta_e_threshold
        && ssim > config.ssim_threshold
}

/// Compare two TUI cell buffers
pub fn compare_tui(
    reference: &CellBuffer,
    target: &CellBuffer,
    config: &TuiComparisonConfig,
) -> TuiComparisonResult {
    let width = reference.width().min(target.width());
    let height = reference.height().min(target.height());
    let total_cells = (width as usize) * (height as usize);

    let mut diff_cells = Vec::new();
    let mut char_diff_count = 0;
    let mut color_diff_count = 0;
    let mut total_delta_e = 0.0;

    // Compare each cell
    for y in 0..height {
        for x in 0..width {
            let Some(ref_cell) = reference.get(x, y) else {
                continue;
            };
            let Some(tgt_cell) = target.get(x, y) else {
                continue;
            };

            // Compare cells using helper
            let cmp = compare_cells(ref_cell, tgt_cell);
            total_delta_e += cmp.delta_e;
            char_diff_count += cmp.char_differs as usize;
            color_diff_count += cmp.color_differs as usize;

            // Record differing cells
            if cmp.char_differs || cmp.color_differs {
                diff_cells.push(DiffCell {
                    x,
                    y,
                    reference_char: ref_cell.symbol.chars().next().unwrap_or(' '),
                    target_char: tgt_cell.symbol.chars().next().unwrap_or(' '),
                    reference_fg: cmp.ref_fg,
                    target_fg: cmp.tgt_fg,
                    delta_e: cmp.delta_e,
                });
            }
        }
    }

    // Calculate metrics using helpers
    let (cld, avg_delta_e) =
        compute_comparison_metrics(char_diff_count, total_delta_e, total_cells);
    let ssim = calculate_ssim(reference, target);
    let passed = check_thresholds(cld, avg_delta_e, ssim, config);

    TuiComparisonResult {
        passed,
        cld,
        delta_e: avg_delta_e,
        ssim,
        panel_results: Vec::new(), // Panel detection not implemented yet
        diff_cells,
        total_cells,
        char_diff_count,
        color_diff_count,
    }
}

/// Calculate SSIM (Structural Similarity Index)
///
/// Uses 8x8 windows to compare local structure.
fn calculate_ssim(reference: &CellBuffer, target: &CellBuffer) -> f64 {
    let width = reference.width().min(target.width());
    let height = reference.height().min(target.height());

    if width < 8 || height < 8 {
        // Too small for windowed SSIM, fall back to simple comparison
        return simple_similarity(reference, target);
    }

    let window_size = 8;
    let mut ssim_sum = 0.0;
    let mut window_count = 0;

    // Slide window across the buffers
    for wy in (0..height - window_size).step_by(window_size as usize / 2) {
        for wx in (0..width - window_size).step_by(window_size as usize / 2) {
            let window_ssim = calculate_window_ssim(reference, target, wx, wy, window_size);
            ssim_sum += window_ssim;
            window_count += 1;
        }
    }

    if window_count > 0 {
        ssim_sum / window_count as f64
    } else {
        1.0
    }
}

/// Calculate SSIM for a single window
fn calculate_window_ssim(
    reference: &CellBuffer,
    target: &CellBuffer,
    wx: u16,
    wy: u16,
    size: u16,
) -> f64 {
    let mut ref_lum = Vec::new();
    let mut tgt_lum = Vec::new();

    for y in wy..wy + size {
        for x in wx..wx + size {
            let Some(ref_cell) = reference.get(x, y) else {
                continue;
            };
            let Some(tgt_cell) = target.get(x, y) else {
                continue;
            };

            // Use luminance of foreground color
            let ref_rgb = color_to_rgb(&ref_cell.fg);
            let tgt_rgb = color_to_rgb(&tgt_cell.fg);

            ref_lum.push(luminance(ref_rgb));
            tgt_lum.push(luminance(tgt_rgb));
        }
    }

    // SSIM formula: (2*μx*μy + C1)(2*σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
    let c1 = 0.01_f64.powi(2);
    let c2 = 0.03_f64.powi(2);

    let mean_ref = mean(&ref_lum);
    let mean_tgt = mean(&tgt_lum);

    let var_ref = variance(&ref_lum, mean_ref);
    let var_tgt = variance(&tgt_lum, mean_tgt);
    let covar = covariance(&ref_lum, &tgt_lum, mean_ref, mean_tgt);

    let numerator = (2.0 * mean_ref * mean_tgt + c1) * (2.0 * covar + c2);
    let denominator = (mean_ref.powi(2) + mean_tgt.powi(2) + c1) * (var_ref + var_tgt + c2);

    if denominator > 0.0 {
        numerator / denominator
    } else {
        1.0
    }
}

/// Simple similarity for small buffers
fn simple_similarity(reference: &CellBuffer, target: &CellBuffer) -> f64 {
    let width = reference.width().min(target.width());
    let height = reference.height().min(target.height());
    let total = (width as usize) * (height as usize);

    if total == 0 {
        return 1.0;
    }

    let mut matches = 0;
    for y in 0..height {
        for x in 0..width {
            let ref_sym = reference.get(x, y).map(|c| &c.symbol);
            let tgt_sym = target.get(x, y).map(|c| &c.symbol);
            if ref_sym == tgt_sym {
                matches += 1;
            }
        }
    }

    matches as f64 / total as f64
}

/// Convert Cell color to Rgb
fn color_to_rgb(color: &Color) -> Rgb {
    Rgb {
        r: (color.r * 255.0) as u8,
        g: (color.g * 255.0) as u8,
        b: (color.b * 255.0) as u8,
    }
}

/// Calculate luminance (Y) from RGB
fn luminance(rgb: Rgb) -> f64 {
    0.2126 * (rgb.r as f64 / 255.0)
        + 0.7152 * (rgb.g as f64 / 255.0)
        + 0.0722 * (rgb.b as f64 / 255.0)
}

/// Calculate mean
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate variance
fn variance(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Calculate covariance
fn covariance(x: &[f64], y: &[f64], mean_x: f64, mean_y: f64) -> f64 {
    if x.is_empty() || x.len() != y.len() {
        return 0.0;
    }
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / x.len() as f64
}

/// Generate a text report of comparison results
pub fn generate_report(result: &TuiComparisonResult, config: &TuiComparisonConfig) -> String {
    let mut report = String::new();

    report.push_str(
        "╔══════════════════════════════════════════════════════════════════════════════╗\n",
    );
    report.push_str(
        "║                    TUI PIXEL COMPARISON REPORT                                ║\n",
    );
    report.push_str(
        "╠══════════════════════════════════════════════════════════════════════════════╣\n",
    );
    report.push_str(
        "║                                                                               ║\n",
    );
    report.push_str(
        "║  METRIC                    VALUE           THRESHOLD       STATUS             ║\n",
    );
    report.push_str(
        "║  ─────────────────────────────────────────────────────────────────────────── ║\n",
    );

    // CLD
    let cld_status = if result.cld < config.cld_threshold {
        "✓ PASS"
    } else {
        "✗ FAIL"
    };
    let _ = writeln!(
        report,
        "║  Character Diff (CLD)      {:<15.4} < {:<15.2} {}             ║",
        result.cld, config.cld_threshold, cld_status
    );

    // ΔE00
    let de_status = if result.delta_e < config.delta_e_threshold {
        "✓ PASS"
    } else {
        "✗ FAIL"
    };
    let _ = writeln!(
        report,
        "║  Color Diff (ΔE00)         {:<15.2} < {:<15.2} {}             ║",
        result.delta_e, config.delta_e_threshold, de_status
    );

    // SSIM
    let ssim_status = if result.ssim > config.ssim_threshold {
        "✓ PASS"
    } else {
        "✗ FAIL"
    };
    let _ = writeln!(
        report,
        "║  Structural (SSIM)         {:<15.3} > {:<15.2} {}             ║",
        result.ssim, config.ssim_threshold, ssim_status
    );

    report.push_str(
        "║                                                                               ║\n",
    );
    report.push_str(
        "║  ─────────────────────────────────────────────────────────────────────────── ║\n",
    );
    let _ = writeln!(
        report,
        "║  Total cells: {}    Char diffs: {}    Color diffs: {}                ║",
        result.total_cells, result.char_diff_count, result.color_diff_count
    );
    report.push_str(
        "║                                                                               ║\n",
    );
    report.push_str(
        "╠══════════════════════════════════════════════════════════════════════════════╣\n",
    );

    let verdict = if result.passed {
        "║  VERDICT: PASSING - All metrics within threshold                              ║"
    } else {
        "║  VERDICT: FAILING - One or more metrics above threshold                       ║"
    };
    report.push_str(verdict);
    report.push('\n');
    report.push_str(
        "╚══════════════════════════════════════════════════════════════════════════════╝\n",
    );

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_buffers() {
        let mut buf = CellBuffer::new(10, 5);
        buf.write_str(0, 0, "Hello");

        let config = TuiComparisonConfig::default();
        let result = compare_tui(&buf, &buf, &config);

        assert!(result.passed);
        assert_eq!(result.cld, 0.0);
        assert_eq!(result.delta_e, 0.0);
        assert_eq!(result.ssim, 1.0);
    }

    #[test]
    fn test_different_characters() {
        let mut buf1 = CellBuffer::new(10, 5);
        buf1.write_str(0, 0, "Hello");

        let mut buf2 = CellBuffer::new(10, 5);
        buf2.write_str(0, 0, "World");

        let config = TuiComparisonConfig::default();
        let result = compare_tui(&buf1, &buf2, &config);

        // 4 characters differ out of 50 total = 8% (position 3 'l'='l' matches)
        assert!(!result.passed); // CLD > 1%
        assert!(result.cld > 0.07);
    }

    #[test]
    fn test_ssim_calculation() {
        let buf = CellBuffer::new(20, 20);
        let ssim = calculate_ssim(&buf, &buf);
        assert!((ssim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_luminance() {
        // Black
        let black = luminance(Rgb { r: 0, g: 0, b: 0 });
        assert!(black < 0.001);

        // White
        let white = luminance(Rgb {
            r: 255,
            g: 255,
            b: 255,
        });
        assert!((white - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_report_generation() {
        let result = TuiComparisonResult {
            passed: true,
            cld: 0.005,
            delta_e: 1.5,
            ssim: 0.98,
            panel_results: vec![],
            diff_cells: vec![],
            total_cells: 1000,
            char_diff_count: 5,
            color_diff_count: 10,
        };

        let config = TuiComparisonConfig::default();
        let report = generate_report(&result, &config);

        assert!(report.contains("PASSING"));
        assert!(report.contains("✓ PASS"));
    }
}
