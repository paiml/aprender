//! Output formatters for trueno-explain
//!
//! Supports text (terminal), JSON, and future TUI output.

use crate::analyzer::{AnalysisReport, MudaType};
use colored::{ColoredString, Colorize};
use std::io::{self, Write};

/// Output format options
#[derive(Debug, Clone, Copy, Default)]
pub enum OutputFormat {
    /// Colored text output for terminal
    #[default]
    Text,
    /// JSON output for tooling and CI
    Json,
}

/// Format an analysis report as colored text for terminal
#[must_use]
pub fn format_text(report: &AnalysisReport) -> String {
    let mut output = String::new();

    // Header
    output.push_str(&format!(
        "{} Analysis: {}\n",
        report.target.cyan().bold(),
        report.name.white().bold()
    ));
    output.push_str(&"═".repeat(60));
    output.push('\n');

    // Register Pressure
    let reg_status = if report.registers.total() < 64 {
        "[OK]".green()
    } else if report.registers.total() < 128 {
        "[WARN]".yellow()
    } else {
        "[HIGH]".red()
    };

    output.push_str(&format!(
        "\n{}                                    {}\n",
        "Register Pressure:".white().bold(),
        reg_status
    ));
    output.push_str(&format!(
        "  ├── .reg .f32: {} / 255 ({:.1}%)\n",
        report.registers.f32_regs,
        report.registers.f32_regs as f32 / 255.0 * 100.0
    ));
    output.push_str(&format!(
        "  ├── .reg .b32: {} / 255 ({:.1}%)\n",
        report.registers.b32_regs,
        report.registers.b32_regs as f32 / 255.0 * 100.0
    ));
    output.push_str(&format!(
        "  ├── .reg .b64: {} / 255 ({:.1}%)\n",
        report.registers.b64_regs,
        report.registers.b64_regs as f32 / 255.0 * 100.0
    ));
    // PTX has 8 predicate registers (p0-p7)
    output.push_str(&format!(
        "  ├── .reg .pred: {} / 8 ({:.1}%)\n",
        report.registers.pred_regs,
        report.registers.pred_regs as f32 / 8.0 * 100.0
    ));
    output.push_str(&format!(
        "  └── {}: {} registers → {:.0}% occupancy possible\n",
        "Total".bold(),
        report.registers.total(),
        report.estimated_occupancy * 100.0
    ));

    // Memory Access Pattern
    let mem_status = if report.memory.coalesced_ratio >= 0.9 {
        "[OK]".green()
    } else if report.memory.coalesced_ratio >= 0.7 {
        "[WARN]".yellow()
    } else {
        "[BAD]".red()
    };

    output.push_str(&format!(
        "\n{}                                {}\n",
        "Memory Access Pattern:".white().bold(),
        mem_status
    ));
    output.push_str(&format!(
        "  ├── Global loads: {} (coalesced: {:.1}%)\n",
        report.memory.global_loads,
        report.memory.coalesced_ratio * 100.0
    ));
    output.push_str(&format!(
        "  ├── Global stores: {}\n",
        report.memory.global_stores
    ));
    output.push_str(&format!(
        "  ├── Shared loads: {}\n",
        report.memory.shared_loads
    ));
    output.push_str(&format!(
        "  └── Shared stores: {}\n",
        report.memory.shared_stores
    ));

    // Roofline
    output.push_str(&format!("\n{}\n", "Performance Estimate:".white().bold()));
    output.push_str(&format!(
        "  ├── Arithmetic Intensity: {:.2} FLOPs/Byte\n",
        report.roofline.arithmetic_intensity
    ));
    output.push_str(&format!(
        "  └── Bottleneck: {}\n",
        if report.roofline.memory_bound {
            "Memory bandwidth".yellow()
        } else {
            "Compute".green()
        }
    ));

    // Muda Warnings
    if report.warnings.is_empty() {
        output.push_str(&format!("\n{} No Muda detected\n", "✓".green()));
    } else {
        output.push_str(&format!("\n{}\n", "Muda (Waste) Detection:".white().bold()));
        for warning in &report.warnings {
            let icon = match warning.muda_type {
                MudaType::Transport => "⚠".yellow(),
                MudaType::Waiting => "⏳".yellow(),
                MudaType::Overprocessing => "🔄".yellow(),
            };
            output.push_str(&format!(
                "  {} {}: {}\n",
                icon,
                muda_name(&warning.muda_type),
                warning.description
            ));
            if let Some(ref suggestion) = warning.suggestion {
                output.push_str(&format!(
                    "     └── {}: {}\n",
                    "Suggestion".cyan(),
                    suggestion
                ));
            }
        }
    }

    output
}

fn muda_name(muda: &MudaType) -> ColoredString {
    match muda {
        MudaType::Transport => "Muda of Transport (Spills)".yellow(),
        MudaType::Waiting => "Muda of Waiting (Stalls)".yellow(),
        MudaType::Overprocessing => "Muda of Overprocessing".yellow(),
    }
}

/// Format an analysis report as JSON
///
/// # Errors
///
/// Returns `serde_json::Error` if serialization fails.
pub fn format_json(report: &AnalysisReport) -> serde_json::Result<String> {
    serde_json::to_string_pretty(report)
}

/// Write report to stdout in the specified format
///
/// # Errors
///
/// Returns `io::Error` if writing to stdout fails or JSON serialization fails.
pub fn write_report(report: &AnalysisReport, format: OutputFormat) -> io::Result<()> {
    let mut stdout = io::stdout().lock();

    match format {
        OutputFormat::Text => {
            write!(stdout, "{}", format_text(report))?;
        }
        OutputFormat::Json => {
            let json = format_json(report).map_err(io::Error::other)?;
            writeln!(stdout, "{}", json)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{MemoryPattern, MudaWarning, RegisterUsage, RooflineMetric};

    fn sample_report() -> AnalysisReport {
        AnalysisReport {
            name: "test_kernel".to_string(),
            target: "PTX".to_string(),
            registers: RegisterUsage {
                f32_regs: 24,
                b32_regs: 18,
                b64_regs: 12,
                pred_regs: 4,
                ..Default::default()
            },
            memory: MemoryPattern {
                global_loads: 100,
                global_stores: 50,
                coalesced_ratio: 0.95,
                ..Default::default()
            },
            roofline: RooflineMetric {
                arithmetic_intensity: 2.5,
                theoretical_peak_gflops: 15000.0,
                memory_bound: true,
            },
            warnings: vec![],
            instruction_count: 150,
            estimated_occupancy: 0.875,
        }
    }

    #[test]
    fn test_format_text_contains_kernel_name() {
        let report = sample_report();
        let text = format_text(&report);
        assert!(text.contains("test_kernel"));
    }

    #[test]
    fn test_format_text_contains_registers() {
        let report = sample_report();
        let text = format_text(&report);
        assert!(text.contains("24"));
        assert!(text.contains("f32"));
    }

    #[test]
    fn test_format_text_contains_memory() {
        let report = sample_report();
        let text = format_text(&report);
        assert!(text.contains("Global loads"));
        assert!(text.contains("100"));
    }

    #[test]
    fn test_format_json_valid() {
        let report = sample_report();
        let json = format_json(&report).unwrap();

        // Verify it's valid JSON by parsing it
        let parsed: AnalysisReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test_kernel");
    }

    #[test]
    fn test_format_text_with_warnings() {
        let mut report = sample_report();
        report.warnings.push(MudaWarning {
            muda_type: MudaType::Transport,
            description: "5 spills detected".to_string(),
            impact: "High latency".to_string(),
            line: None,
            suggestion: Some("Reduce variables".to_string()),
        });

        let text = format_text(&report);
        assert!(text.contains("Muda"));
        assert!(text.contains("5 spills"));
        assert!(text.contains("Suggestion"));
    }

    #[test]
    fn test_format_text_no_warnings() {
        let report = sample_report();
        let text = format_text(&report);
        assert!(text.contains("No Muda detected"));
    }
}
