#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Pixel Coverage Heatmap Example (PIXEL-001 v2.1)
//!
//! Demonstrates generating PNG heatmaps showing coverage and gaps,
//! using trueno-viz style output with margins and color palettes,
//! plus PIXEL-001 v2.1 verification features.
//!
//! Features demonstrated:
//! - Terminal heatmap (STDOUT) with Unicode blocks
//! - PNG heatmap with Viridis/Magma/Heat color palettes
//! - Title and subtitle text rendering
//! - Combined coverage report (line + pixel)
//! - Gap highlighting (red outline for untested regions)
//! - **PIXEL-001 v2.1**: FalsifiabilityGate, Wilson CI, ScoreBar
//! - CLI usage: `probar coverage --png output.png`
//!
//! Run with: `cargo run --example pixel_coverage_heatmap -p jugar-probar`

use jugar_probar::pixel_coverage::{
    ColorPalette, CombinedCoverageReport, ConfidenceInterval, CoverageCell, FalsifiabilityGate,
    FalsifiableHypothesis, LineCoverageReport, OutputMode, PixelCoverageTracker, PixelRegion,
    PngHeatmap, ScoreBar,
};

fn main() {
    println!("Pixel Coverage Heatmap Example (PIXEL-001 v2.1)");
    println!("=================================================\n");

    // Step 1: Create a coverage tracker and simulate some interactions
    println!("Step 1: Creating coverage tracker (10x8 grid on 800x600 screen)...");
    let mut tracker = PixelCoverageTracker::new(800, 600, 10, 8);

    // Simulate coverage: cover most of the screen but leave gaps
    println!("\nStep 2: Simulating coverage with gaps...");

    // Cover the header area (top 2 rows)
    tracker.record_region(PixelRegion::new(0, 0, 800, 150));
    println!("  ✓ Header area covered (rows 0-1)");

    // Cover the main content area (but leave gaps)
    tracker.record_region(PixelRegion::new(0, 150, 320, 300));
    println!("  ✓ Left sidebar covered");

    tracker.record_region(PixelRegion::new(480, 150, 320, 300));
    println!("  ✓ Right content covered");

    // Leave middle area as a gap!
    println!("  ⚠ Middle content area is a GAP (uncovered)");

    // Cover the footer
    tracker.record_region(PixelRegion::new(0, 525, 800, 75));
    println!("  ✓ Footer area covered");

    // Generate report
    let report = tracker.generate_report();
    println!("\nStep 3: Coverage Report");
    println!(
        "  Overall Coverage: {:.1}%",
        report.overall_coverage * 100.0
    );
    println!(
        "  Covered Cells: {}/{}",
        report.covered_cells, report.total_cells
    );
    println!("  Uncovered Regions: {}", report.uncovered_regions.len());
    println!(
        "  Meets Threshold: {}",
        if report.meets_threshold { "✓" } else { "✗" }
    );

    // Step 3b: PIXEL-001 v2.1 - Popperian Falsification
    println!("\nStep 3b: PIXEL-001 v2.1 Verification...");

    // FalsifiabilityGate with 15.0 threshold
    let gate = FalsifiabilityGate::new(15.0);

    // Test hypotheses
    let h1 = FalsifiableHypothesis::coverage_threshold("H0-HEATMAP-COV", 0.75)
        .evaluate(report.overall_coverage);
    let h2 = FalsifiableHypothesis::max_gap_size("H0-HEATMAP-GAP", 8.0)
        .evaluate(report.uncovered_regions.len() as f32);

    println!(
        "  H0-HEATMAP-COV (≥75%): {}",
        if h1.falsified {
            "FALSIFIED"
        } else {
            "NOT FALSIFIED"
        }
    );
    println!(
        "  H0-HEATMAP-GAP (≤8 gaps): {}",
        if h2.falsified {
            "FALSIFIED"
        } else {
            "NOT FALSIFIED"
        }
    );

    let gate_result = gate.evaluate(&h1);
    println!(
        "  FalsifiabilityGate: {}",
        if gate_result.is_passed() {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    // Wilson Score CI
    println!("\n  Wilson Score CI (95%):");
    let ci = ConfidenceInterval::wilson_score(report.covered_cells, report.total_cells, 0.95);
    println!(
        "    Coverage: [{:.1}%, {:.1}%]",
        ci.lower * 100.0,
        ci.upper * 100.0
    );

    // Score Bars
    println!("\n  Score Bars:");
    let mode = OutputMode::from_env();
    let coverage_bar = ScoreBar::new("Pixel", report.overall_coverage, 0.80);
    println!("    {}", coverage_bar.render(mode));

    // Step 4: Generate PNG heatmaps with different styles
    println!("\nStep 4: Generating PNG heatmaps...");

    // Viridis palette (default) with title and gaps highlighted
    let output_path = std::env::temp_dir().join("coverage_viridis.png");
    PngHeatmap::new(800, 600)
        .with_palette(ColorPalette::viridis())
        .with_title("Pixel Coverage Heatmap")
        .with_subtitle("Viridis Palette - Colorblind Safe")
        .with_legend()
        .with_gap_highlighting()
        .with_margin(40)
        .export_to_file(tracker.cells(), &output_path)
        .expect("Failed to export PNG");
    println!("  ✓ Viridis heatmap: {}", output_path.display());

    // Magma palette with title
    let output_path = std::env::temp_dir().join("coverage_magma.png");
    PngHeatmap::new(800, 600)
        .with_palette(ColorPalette::magma())
        .with_title("Coverage Analysis")
        .with_subtitle("Magma Palette - Dark to Bright")
        .with_legend()
        .with_gap_highlighting()
        .with_margin(40)
        .export_to_file(tracker.cells(), &output_path)
        .expect("Failed to export PNG");
    println!("  ✓ Magma heatmap: {}", output_path.display());

    // Heat palette (black-red-yellow-white) with title
    let output_path = std::env::temp_dir().join("coverage_heat.png");
    PngHeatmap::new(800, 600)
        .with_palette(ColorPalette::heat())
        .with_title("Heat Map View")
        .with_subtitle("Classic Heat Palette")
        .with_legend()
        .with_gap_highlighting()
        .with_margin(40)
        .export_to_file(tracker.cells(), &output_path)
        .expect("Failed to export PNG");
    println!("  ✓ Heat heatmap: {}", output_path.display());

    // Step 5: Show the coverage grid in terminal (STDOUT heatmap)
    println!("\nStep 5: Terminal Heatmap (STDOUT)");
    println!("{}", "─".repeat(50));
    let terminal = tracker.terminal_heatmap();
    println!("{}", terminal.render_with_border());
    println!("{}", terminal.legend());

    // Step 6: Combined Coverage Report (line + pixel)
    println!("\nStep 6: Combined Coverage Report (STDOUT)");
    println!("{}", "─".repeat(50));

    // Simulate line coverage data (would come from GuiCoverage in real use)
    let line_report = LineCoverageReport::new(
        0.90, // element coverage
        1.0,  // screen coverage
        0.85, // journey coverage
        22,   // total elements
        20,   // covered elements
    );

    let combined = CombinedCoverageReport::from_parts(line_report, report);
    println!("{}", combined.summary());

    // Step 6b: Generate combined PNG with stats panel
    println!("\nStep 6b: Generating Combined PNG with Stats Panel...");
    let output_path = std::env::temp_dir().join("coverage_combined.png");
    PngHeatmap::new(800, 700)
        .with_palette(ColorPalette::viridis())
        .with_title("Combined Coverage Report")
        .with_subtitle("Line + Pixel Coverage")
        .with_legend()
        .with_gap_highlighting()
        .with_margin(40)
        .with_combined_stats(&combined)
        .export_to_file(tracker.cells(), &output_path)
        .expect("Failed to export PNG");
    println!("  ✓ Combined stats heatmap: {}", output_path.display());

    // Step 7: Create a synthetic example with clear patterns
    println!("\nStep 7: Generating pattern heatmap...");
    let cells = create_pattern_cells();

    // Show pattern as ASCII art first
    println!("\n  Pattern Preview (ASCII):");
    print_ascii_heatmap(&cells);

    let output_path = std::env::temp_dir().join("coverage_pattern.png");
    PngHeatmap::new(800, 600)
        .with_palette(ColorPalette::viridis())
        .with_legend()
        .with_gap_highlighting()
        .with_margin(40)
        .export_to_file(&cells, &output_path)
        .expect("Failed to export PNG");
    println!("\n  ✓ Pattern heatmap: {}", output_path.display());

    // Step 8: Print detailed gap analysis
    println!("\nStep 8: Gap Analysis (STDOUT)");
    println!("{}", "─".repeat(50));
    print_gap_analysis(&cells);

    println!("\n✅ Pixel coverage heatmap example completed (PIXEL-001 v2.1)!");
    println!("\nGenerated PNG files:");
    for name in ["viridis", "magma", "heat", "combined", "pattern"] {
        println!(
            "  • {}/coverage_{}.png",
            std::env::temp_dir().display(),
            name
        );
    }

    // PIXEL-001 v2.1 Summary
    println!("\n--- PIXEL-001 v2.1 Features Demonstrated ---");
    println!("  • FalsifiabilityGate: Popperian methodology with 15/25 threshold");
    println!("  • FalsifiableHypothesis: coverage_threshold, max_gap_size");
    println!("  • ConfidenceInterval: Wilson score for statistical rigor");
    println!("  • ScoreBar: Visual progress indicators");
    println!("  • CombinedCoverageReport: Line + Pixel coverage");

    // CLI usage reminder
    println!("\n--- CLI Usage ---");
    println!("Generate heatmaps via CLI:");
    println!("  probar coverage --png output.png --palette viridis --legend --gaps");
    println!("  probar coverage --png output.png --title \"My Coverage\" --palette magma");
    println!("  probar coverage --json report.json");
}

/// Print ASCII representation of the heatmap to stdout.
fn print_ascii_heatmap(cells: &[Vec<CoverageCell>]) {
    println!("  ┌{}┐", "─".repeat(cells[0].len()));
    for row in cells {
        print!("  │");
        for cell in row {
            let ch = match cell.coverage {
                c if c <= 0.0 => '·', // Gap
                c if c <= 0.25 => '░',
                c if c <= 0.50 => '▒',
                c if c <= 0.75 => '▓',
                _ => '█',
            };
            print!("{}", ch);
        }
        println!("│");
    }
    println!("  └{}┘", "─".repeat(cells[0].len()));
    println!("  Legend: · = Gap (0%)  ░ = 1-25%  ▒ = 26-50%  ▓ = 51-75%  █ = 76-100%");
}

/// Print gap analysis to stdout.
fn print_gap_analysis(cells: &[Vec<CoverageCell>]) {
    let mut gaps = Vec::new();
    for (row_idx, row) in cells.iter().enumerate() {
        for (col_idx, cell) in row.iter().enumerate() {
            if cell.coverage <= 0.0 {
                gaps.push((row_idx, col_idx));
            }
        }
    }

    let total_cells = cells.len() * cells[0].len();
    let covered = total_cells - gaps.len();
    let coverage_pct = (covered as f32 / total_cells as f32) * 100.0;

    println!("  Total Cells:    {}", total_cells);
    println!("  Covered Cells:  {}", covered);
    println!(
        "  Gap Cells:      {} (highlighted in red on PNG)",
        gaps.len()
    );
    println!("  Coverage:       {:.1}%", coverage_pct);

    if !gaps.is_empty() {
        println!("\n  Gap Locations:");
        for (row, col) in &gaps {
            println!("    ⚠ Cell ({}, {}) - UNTESTED", row, col);
        }
    } else {
        println!("\n  ✓ No gaps detected - 100% coverage!");
    }
}

/// Check if a position is in a gap region
fn is_gap_position(row: usize, col: usize) -> bool {
    let middle_gap = row == 5 && (5..=7).contains(&col);
    let end_gap = row == 2 && col > 10;
    middle_gap || end_gap
}

/// Calculate gradient coverage value
fn gradient_coverage(row: usize, col: usize, rows: usize, cols: usize) -> f32 {
    if is_gap_position(row, col) {
        return 0.0;
    }
    let x_factor = col as f32 / (cols - 1) as f32;
    let y_factor = row as f32 / (rows - 1) as f32;
    (x_factor + y_factor) / 2.0
}

/// Create a pattern of cells showing gradient and gaps.
fn create_pattern_cells() -> Vec<Vec<CoverageCell>> {
    const ROWS: usize = 10;
    const COLS: usize = 15;

    (0..ROWS)
        .map(|row| {
            (0..COLS)
                .map(|col| {
                    let coverage = gradient_coverage(row, col, ROWS, COLS);
                    CoverageCell {
                        coverage,
                        hit_count: (coverage * 10.0) as u64,
                    }
                })
                .collect()
        })
        .collect()
}
