//! PTX analysis and explanation command
//!
//! Bridges trueno-explain's PTX analysis into the apr CLI.
//! Runs PtxAnalyzer (register pressure, memory patterns, occupancy, roofline, muda)
//! and PtxBugAnalyzer (15+ bug detectors) on PTX source files or generated kernels.

use crate::error::Result;
use std::path::Path;

/// Run PTX analysis on a file or kernel name.
pub(crate) fn run(
    file: Option<&Path>,
    kernel: Option<&str>,
    strict: bool,
    bugs_only: bool,
    json: bool,
    verbose: bool,
) -> Result<()> {
    let ptx_source = if let Some(path) = file {
        std::fs::read_to_string(path).map_err(|e| {
            crate::error::CliError::Aprender(format!(
                "Failed to read PTX file '{}': {}",
                path.display(),
                e
            ))
        })?
    } else if let Some(name) = kernel {
        generate_kernel_ptx(name)?
    } else {
        return Err(crate::error::CliError::Aprender(
            "Provide a PTX file path or --kernel <name>".to_string(),
        ));
    };

    if json {
        run_json(&ptx_source, strict, bugs_only)
    } else {
        run_human(&ptx_source, strict, bugs_only, verbose)
    }
}

/// Human-readable PTX analysis output.
fn run_human(ptx: &str, strict: bool, bugs_only: bool, verbose: bool) -> Result<()> {
    use trueno_explain::analyzer::Analyzer;

    if !bugs_only {
        // Run PtxAnalyzer for register/memory/roofline analysis
        let analyzer = trueno_explain::PtxAnalyzer::new();
        match analyzer.analyze(ptx) {
            Ok(report) => {
                println!("\x1b[1;36m=== PTX Analysis: {} ===\x1b[0m", report.name);
                println!();

                // Register usage
                println!("\x1b[1mRegisters:\x1b[0m");
                println!(
                    "  f32: {}  f64: {}  b32: {}  b64: {}  pred: {}",
                    report.registers.f32_regs,
                    report.registers.f64_regs,
                    report.registers.b32_regs,
                    report.registers.b64_regs,
                    report.registers.pred_regs,
                );
                println!(
                    "  Total: {}  Estimated occupancy: {:.0}%",
                    report.registers.total(),
                    report.estimated_occupancy * 100.0,
                );
                println!();

                // Memory patterns
                println!("\x1b[1mMemory:\x1b[0m");
                println!(
                    "  Global loads: {}  Global stores: {}",
                    report.memory.global_loads, report.memory.global_stores
                );
                println!(
                    "  Shared loads: {}  Shared stores: {}",
                    report.memory.shared_loads, report.memory.shared_stores
                );
                println!(
                    "  Coalescing ratio: {:.1}%",
                    report.memory.coalesced_ratio * 100.0
                );
                println!();

                // Roofline
                println!("\x1b[1mRoofline:\x1b[0m");
                println!("  Instructions: {}", report.instruction_count);
                println!(
                    "  Arithmetic intensity: {:.2} FLOP/byte",
                    report.roofline.arithmetic_intensity
                );
                println!(
                    "  Bottleneck: {}",
                    if report.roofline.memory_bound {
                        "MEMORY-BOUND"
                    } else {
                        "COMPUTE-BOUND"
                    }
                );
                println!();

                // Muda warnings
                if !report.warnings.is_empty() {
                    println!("\x1b[1;33mMuda (Waste) Warnings:\x1b[0m");
                    for w in &report.warnings {
                        println!("  [{:?}] {}", w.muda_type, w.description);
                        println!("    Impact: {}", w.impact);
                        if let Some(suggestion) = &w.suggestion {
                            println!("    Fix: {suggestion}");
                        }
                    }
                    println!();
                }

                if verbose {
                    // Show PTX source with line numbers
                    println!("\x1b[1mPTX Source:\x1b[0m");
                    for (i, line) in ptx.lines().enumerate() {
                        println!("  {:4} | {line}", i + 1);
                    }
                    println!();
                }
            }
            Err(e) => {
                eprintln!("PTX analysis error: {e}");
            }
        }
    }

    // Run PtxBugAnalyzer
    let bug_analyzer = if strict {
        trueno_explain::PtxBugAnalyzer::strict()
    } else {
        trueno_explain::PtxBugAnalyzer::with_performance_whitelist()
    };

    let bug_report = bug_analyzer.analyze(ptx);

    if bugs_only || !bug_report.bugs.is_empty() {
        println!(
            "\x1b[1;{}m=== PTX Bug Analysis{} ===\x1b[0m",
            if bug_report.bugs.is_empty() {
                "32"
            } else {
                "31"
            },
            if let Some(name) = &bug_report.kernel_name {
                format!(": {name}")
            } else {
                String::new()
            },
        );
        println!("  Lines analyzed: {}", bug_report.lines_analyzed);
        println!("  Strict mode: {}", bug_report.strict_mode);
        println!("  Bugs found: {}", bug_report.bugs.len());
        println!();

        if bug_report.bugs.is_empty() {
            println!("  \x1b[32mNo bugs detected.\x1b[0m");
        } else {
            for bug in &bug_report.bugs {
                let severity_color = match bug.class.severity() {
                    trueno_explain::BugSeverity::Critical => "31",
                    trueno_explain::BugSeverity::High => "33",
                    trueno_explain::BugSeverity::Medium => "35",
                    trueno_explain::BugSeverity::FalsePositive => "36",
                };
                println!(
                    "  \x1b[{severity_color}m[{:?}]\x1b[0m Line {}: {}",
                    bug.class, bug.line, bug.message
                );
                if !bug.instruction.is_empty() {
                    println!("    Instruction: {}", bug.instruction);
                }
                if let Some(fix) = &bug.fix {
                    println!("    Fix: {fix}");
                }
                println!();
            }
        }
    }

    Ok(())
}

/// JSON output for PTX analysis.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn run_json(ptx: &str, strict: bool, bugs_only: bool) -> Result<()> {
    use trueno_explain::analyzer::Analyzer;

    let mut output = serde_json::Map::new();

    if !bugs_only {
        let analyzer = trueno_explain::PtxAnalyzer::new();
        if let Ok(report) = analyzer.analyze(ptx) {
            if let Ok(report_json) = serde_json::to_value(&report) {
                output.insert("analysis".to_string(), report_json);
            }
        }
    }

    let bug_analyzer = if strict {
        trueno_explain::PtxBugAnalyzer::strict()
    } else {
        trueno_explain::PtxBugAnalyzer::with_performance_whitelist()
    };

    let bug_report = bug_analyzer.analyze(ptx);
    let bugs_json = serde_json::json!({
        "kernel_name": bug_report.kernel_name,
        "lines_analyzed": bug_report.lines_analyzed,
        "strict_mode": bug_report.strict_mode,
        "bug_count": bug_report.bugs.len(),
        "bugs": bug_report.bugs.iter().map(|b| {
            serde_json::json!({
                "class": format!("{:?}", b.class),
                "severity": format!("{:?}", b.class.severity()),
                "line": b.line,
                "message": b.message,
                "instruction": b.instruction,
                "fix": b.fix,
            })
        }).collect::<Vec<_>>(),
    });
    output.insert("bugs".to_string(), bugs_json);

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::Value::Object(output))
            .unwrap_or_else(|_| "{}".to_string())
    );

    Ok(())
}

/// Generate PTX for a named kernel from trueno-gpu.
fn generate_kernel_ptx(name: &str) -> Result<String> {
    // Known kernel patterns — generate PTX from trueno-gpu builders
    // trueno-gpu is available via realizar's re-export
    let _name_lower = name.to_lowercase();

    // For now, only file-based analysis is supported.
    // TODO: Wire trueno-gpu kernel builders when inference feature is enabled.
    Err(crate::error::CliError::Aprender(format!(
        "Kernel generation not yet supported for '{}'. Use a PTX file instead.\n\
         Hint: Run with DP4A_Q4K=1 to dump failing PTX to /tmp/failing_ptx.txt",
        name
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

.visible .entry vector_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
) {
    .reg .f32 %f<8>;
    .reg .u32 %r<6>;
    .reg .u64 %rd<8>;
    .reg .pred %p<2>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r3, %r1, %r2, %r0;
    ld.param.u32 %r4, [n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra exit;

    ld.param.u64 %rd0, [a_ptr];
    ld.param.u64 %rd1, [b_ptr];
    ld.param.u64 %rd2, [c_ptr];
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd6], %f2;
exit:
    ret;
}
"#;

    #[test]
    fn test_ptx_explain_inline() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let ptx_path = dir.path().join("test.ptx");
        std::fs::write(&ptx_path, SAMPLE_PTX).expect("write PTX");

        let result = run(Some(ptx_path.as_path()), None, false, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ptx_explain_json() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let ptx_path = dir.path().join("test.ptx");
        std::fs::write(&ptx_path, SAMPLE_PTX).expect("write PTX");

        let result = run(Some(ptx_path.as_path()), None, false, false, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ptx_explain_strict() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let ptx_path = dir.path().join("test.ptx");
        std::fs::write(&ptx_path, SAMPLE_PTX).expect("write PTX");

        let result = run(Some(ptx_path.as_path()), None, true, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ptx_explain_bugs_only() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let ptx_path = dir.path().join("test.ptx");
        std::fs::write(&ptx_path, SAMPLE_PTX).expect("write PTX");

        let result = run(Some(ptx_path.as_path()), None, false, true, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ptx_explain_no_args_errors() {
        let result = run(None, None, false, false, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_ptx_explain_unknown_kernel() {
        let result = run(None, Some("NonexistentKernel"), false, false, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_ptx_explain_missing_file() {
        let result = run(
            Some(Path::new("/tmp/nonexistent_ptx_file.ptx")),
            None,
            false,
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }
}
