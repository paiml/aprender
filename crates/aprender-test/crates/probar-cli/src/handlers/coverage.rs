//! Coverage command handler

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::config::CliConfig;
use crate::error::{CliError, CliResult};
use crate::{CoverageArgs, PaletteArg};
use jugar_probar::pixel_coverage::{ColorPalette, CoverageCell, PixelCoverageReport, PngHeatmap};
use std::path::Path;

/// Execute the coverage command
pub fn execute_coverage(_config: &CliConfig, args: &CoverageArgs) -> CliResult<()> {
    println!("Generating coverage heatmap...");

    let cells: Vec<Vec<CoverageCell>> = if let Some(ref input) = args.input {
        println!("Loading coverage data from {}...", input.display());
        load_coverage_from_json(input)?
    } else {
        println!("No input file specified, using sample data");
        create_sample_coverage_data()
    };

    let palette = match args.palette {
        PaletteArg::Viridis => ColorPalette::viridis(),
        PaletteArg::Magma => ColorPalette::magma(),
        PaletteArg::Heat => ColorPalette::heat(),
    };

    let mut heatmap = PngHeatmap::new(args.width, args.height).with_palette(palette);

    if args.legend {
        heatmap = heatmap.with_legend();
    }

    if args.gaps {
        heatmap = heatmap.with_gap_highlighting();
    }

    if let Some(ref title) = args.title {
        heatmap = heatmap.with_title(title);
    }

    if let Some(ref png_path) = args.png {
        heatmap
            .export_to_file(&cells, png_path)
            .map_err(|e| CliError::report_generation(e.to_string()))?;
        println!("PNG heatmap exported to: {}", png_path.display());
    }

    if let Some(ref json_path) = args.json {
        let report = generate_coverage_report(&cells);
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| CliError::report_generation(e.to_string()))?;
        std::fs::write(json_path, json).map_err(|e| CliError::report_generation(e.to_string()))?;
        println!("Coverage report exported to: {}", json_path.display());
    }

    if args.png.is_none() && args.json.is_none() {
        let report = generate_coverage_report(&cells);
        println!("\nCoverage Summary:");
        println!(
            "  Overall Coverage: {:.1}%",
            report.overall_coverage * 100.0
        );
        println!(
            "  Covered Cells: {}/{}",
            report.covered_cells, report.total_cells
        );
        println!(
            "  Meets Threshold: {}",
            if report.meets_threshold { "Y" } else { "N" }
        );
        println!("\nUse --png <path> to export a heatmap image.");
    }

    Ok(())
}

/// Load coverage data from a JSON file
pub fn load_coverage_from_json(path: &Path) -> CliResult<Vec<Vec<CoverageCell>>> {
    #[derive(serde::Deserialize)]
    struct CoverageData {
        cells: Option<Vec<Vec<CoverageCell>>>,
        #[serde(flatten)]
        _extra: std::collections::HashMap<String, serde_json::Value>,
    }

    let content = std::fs::read_to_string(path).map_err(|e| {
        CliError::report_generation(format!("Failed to read {}: {}", path.display(), e))
    })?;

    if let Ok(data) = serde_json::from_str::<CoverageData>(&content) {
        if let Some(cells) = data.cells {
            return Ok(cells);
        }
    }

    serde_json::from_str::<Vec<Vec<CoverageCell>>>(&content)
        .map_err(|e| CliError::report_generation(format!("Invalid JSON format: {e}")))
}

/// Check if a cell is in a gap region (no coverage)
#[must_use]
pub fn is_gap_cell(row: usize, col: usize) -> bool {
    let middle_gap = row == 5 && (5..=7).contains(&col);
    let end_gap = row == 2 && col > 10;
    middle_gap || end_gap
}

/// Calculate coverage value for a cell based on position
#[must_use]
pub fn calculate_coverage(row: usize, col: usize, rows: usize, cols: usize) -> f32 {
    if is_gap_cell(row, col) {
        return 0.0;
    }
    let x_factor = col as f32 / (cols - 1) as f32;
    let y_factor = row as f32 / (rows - 1) as f32;
    (x_factor + y_factor) / 2.0
}

/// Create sample coverage data for demonstration
#[must_use]
pub fn create_sample_coverage_data() -> Vec<Vec<CoverageCell>> {
    const ROWS: usize = 10;
    const COLS: usize = 15;

    (0..ROWS)
        .map(|row| {
            (0..COLS)
                .map(|col| {
                    let coverage = calculate_coverage(row, col, ROWS, COLS);
                    CoverageCell {
                        coverage,
                        hit_count: (coverage * 10.0) as u64,
                    }
                })
                .collect()
        })
        .collect()
}

/// Generate coverage report from cells
#[must_use]
pub fn generate_coverage_report(cells: &[Vec<CoverageCell>]) -> PixelCoverageReport {
    let total_cells = cells.iter().map(std::vec::Vec::len).sum::<usize>() as u32;
    let covered_cells = cells
        .iter()
        .flat_map(|r| r.iter())
        .filter(|c| c.coverage > 0.0)
        .count() as u32;

    let overall_coverage = if total_cells > 0 {
        covered_cells as f32 / total_cells as f32
    } else {
        0.0
    };

    PixelCoverageReport {
        grid_width: cells.first().map_or(0, |r| r.len() as u32),
        grid_height: cells.len() as u32,
        overall_coverage,
        covered_cells,
        total_cells,
        min_coverage: 0.0,
        max_coverage: 1.0,
        total_interactions: 0,
        meets_threshold: overall_coverage >= 0.8,
        uncovered_regions: Vec::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_is_gap_cell_middle() {
        assert!(is_gap_cell(5, 5));
        assert!(is_gap_cell(5, 6));
        assert!(is_gap_cell(5, 7));
    }

    #[test]
    fn test_is_gap_cell_end() {
        assert!(is_gap_cell(2, 11));
        assert!(is_gap_cell(2, 12));
        assert!(is_gap_cell(2, 100));
    }

    #[test]
    fn test_is_gap_cell_not_gap() {
        assert!(!is_gap_cell(0, 0));
        assert!(!is_gap_cell(5, 4));
        assert!(!is_gap_cell(5, 8));
        assert!(!is_gap_cell(2, 10));
        assert!(!is_gap_cell(3, 11));
    }

    #[test]
    fn test_calculate_coverage_gap_cell() {
        assert_eq!(calculate_coverage(5, 6, 10, 15), 0.0);
        assert_eq!(calculate_coverage(2, 11, 10, 15), 0.0);
    }

    #[test]
    fn test_calculate_coverage_corners() {
        assert_eq!(calculate_coverage(0, 0, 10, 15), 0.0);
        assert_eq!(calculate_coverage(9, 14, 10, 15), 1.0);
    }

    #[test]
    fn test_calculate_coverage_middle() {
        let coverage = calculate_coverage(4, 7, 10, 15);
        assert!(coverage > 0.0);
        assert!(coverage < 1.0);
    }

    #[test]
    fn test_create_sample_coverage_data() {
        let data = create_sample_coverage_data();
        assert_eq!(data.len(), 10);
        assert_eq!(data[0].len(), 15);

        // Check corners
        assert_eq!(data[0][0].coverage, 0.0);
        assert_eq!(data[9][14].coverage, 1.0);

        // Check gap cells have 0 coverage
        assert_eq!(data[5][5].coverage, 0.0);
        assert_eq!(data[5][6].coverage, 0.0);
        assert_eq!(data[5][7].coverage, 0.0);
    }

    #[test]
    fn test_generate_coverage_report() {
        let data = create_sample_coverage_data();
        let report = generate_coverage_report(&data);

        assert_eq!(report.grid_width, 15);
        assert_eq!(report.grid_height, 10);
        assert_eq!(report.total_cells, 150);
        assert!(report.covered_cells < 150); // Some cells are gaps
        assert!(report.overall_coverage > 0.0);
        assert!(report.overall_coverage < 1.0);
    }

    #[test]
    fn test_generate_coverage_report_empty() {
        let data: Vec<Vec<CoverageCell>> = vec![];
        let report = generate_coverage_report(&data);

        assert_eq!(report.total_cells, 0);
        assert_eq!(report.covered_cells, 0);
        assert_eq!(report.overall_coverage, 0.0);
        assert!(!report.meets_threshold);
    }

    #[test]
    fn test_generate_coverage_report_full_coverage() {
        let data = vec![vec![
            CoverageCell {
                coverage: 1.0,
                hit_count: 10,
            };
            10
        ]];
        let report = generate_coverage_report(&data);

        assert_eq!(report.total_cells, 10);
        assert_eq!(report.covered_cells, 10);
        assert_eq!(report.overall_coverage, 1.0);
        assert!(report.meets_threshold);
    }

    #[test]
    fn test_load_coverage_from_json_array_format() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("coverage.json");

        let json = r#"[[{"coverage": 0.5, "hit_count": 5}]]"#;
        std::fs::write(&path, json).unwrap();

        let cells = load_coverage_from_json(&path).unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].len(), 1);
        assert_eq!(cells[0][0].coverage, 0.5);
    }

    #[test]
    fn test_load_coverage_from_json_wrapped_format() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("coverage.json");

        let json = r#"{"cells": [[{"coverage": 0.8, "hit_count": 8}]]}"#;
        std::fs::write(&path, json).unwrap();

        let cells = load_coverage_from_json(&path).unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0][0].coverage, 0.8);
    }

    #[test]
    fn test_load_coverage_from_json_not_found() {
        let result = load_coverage_from_json(Path::new("/nonexistent/path.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_coverage_from_json_invalid() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("invalid.json");

        std::fs::write(&path, "not valid json").unwrap();

        let result = load_coverage_from_json(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_coverage_sample_data() {
        let config = CliConfig::default();
        let args = CoverageArgs {
            png: None,
            json: None,
            palette: PaletteArg::Viridis,
            legend: false,
            gaps: false,
            title: None,
            width: 800,
            height: 600,
            input: None,
        };

        // Should not panic with sample data
        let result = execute_coverage(&config, &args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_coverage_with_json_output() {
        let temp = TempDir::new().unwrap();
        let json_path = temp.path().join("output.json");

        let config = CliConfig::default();
        let args = CoverageArgs {
            png: None,
            json: Some(json_path.clone()),
            palette: PaletteArg::Magma,
            legend: true,
            gaps: true,
            title: Some("Test Coverage".to_string()),
            width: 400,
            height: 300,
            input: None,
        };

        let result = execute_coverage(&config, &args);
        assert!(result.is_ok());
        assert!(json_path.exists());

        let content = std::fs::read_to_string(&json_path).unwrap();
        let _: PixelCoverageReport = serde_json::from_str(&content).unwrap();
    }
}
