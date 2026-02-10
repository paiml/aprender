//! Validation, summary, and header display

use crate::error::{CliError, Result};
use colored::Colorize;

use super::types::*;

pub(super) fn print_header(tier: ModelTier) {
    let tier_name = match tier {
        ModelTier::Tiny => "tiny (0.5B)",
        ModelTier::Small => "small (1.5B)",
        ModelTier::Medium => "medium (7B)",
        ModelTier::Large => "large (32B)",
    };
    println!();
    println!(
        "{}",
        "╔═══════════════════════════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        format!("║     Qwen2.5-Coder Showcase Demo [{tier_name}]").cyan()
    );
    println!(
        "{}",
        "║     PAIML Sovereign AI Stack                                  ║".cyan()
    );
    println!(
        "{}",
        "║     Iron Lotus Grade: Platinum                                ║".cyan()
    );
    println!(
        "{}",
        "╚═══════════════════════════════════════════════════════════════╝".cyan()
    );
    println!();
}

pub(super) fn print_summary(results: &ShowcaseResults, _config: &ShowcaseConfig) {
    println!();
    println!("{}", "═══ Showcase Summary ═══".cyan().bold());
    println!();

    let check = |passed: bool| if passed { "✓".green() } else { "✗".red() };

    println!("  {} HuggingFace Import", check(results.import));
    println!("  {} GGUF Inference", check(results.gguf_inference));
    println!("  {} APR Conversion", check(results.convert));
    println!("  {} APR Inference", check(results.apr_inference));

    if let Some(ref bench) = results.benchmark {
        let llama_pass = bench.speedup_vs_llama.is_some_and(|s| s >= 25.0);
        let ollama_pass = bench.speedup_vs_ollama.is_some_and(|s| s >= 25.0);
        println!(
            "  {} Benchmark vs llama.cpp (25%+ speedup)",
            check(llama_pass)
        );
        println!(
            "  {} Benchmark vs Ollama (25%+ speedup)",
            check(ollama_pass)
        );
    }

    println!("  {} Visualization", check(results.visualize));

    if let Some(ref zram) = results.zram_demo {
        let ratio_pass = zram.lz4_ratio > 2.0;
        let throughput_pass = zram.zero_page_gbps > 150.0;
        let context_pass = zram.context_extension >= 2.0;
        println!(
            "  {} ZRAM Compression (LZ4: {:.1}x @ {:.1} GB/s, ZSTD: {:.1}x, backend: {})",
            check(ratio_pass && throughput_pass),
            zram.lz4_ratio,
            zram.lz4_gbps,
            zram.zstd_ratio,
            zram.simd_backend
        );
        println!(
            "  {} Context Extension: {:.1}x (16K → {}K tokens)",
            check(context_pass),
            zram.context_extension,
            (16.0 * zram.context_extension) as u32
        );
    }

    if let Some(ref cuda) = results.cuda_demo {
        println!(
            "  {} CUDA GPU Detection ({} device(s): {}, {:.1}/{:.1} GB VRAM free)",
            check(cuda.cuda_available && cuda.device_count > 0),
            cuda.device_count,
            cuda.device_name,
            cuda.free_vram_gb,
            cuda.total_vram_gb
        );
    }
}

/// Check step completion and collect failures (Points 1-40)
fn validate_steps(
    results: &ShowcaseResults,
    is_full_run: bool,
    requested_step: Option<ShowcaseStep>,
) -> Vec<String> {
    let mut failures = Vec::new();
    let checks: &[(ShowcaseStep, bool, &str)] = &[
        (ShowcaseStep::Import, results.import, "Point 1: Import step failed"),
        (ShowcaseStep::GgufInference, results.gguf_inference, "Point 11: GGUF inference step failed"),
        (ShowcaseStep::Convert, results.convert, "Point 21: APR conversion step failed"),
        (ShowcaseStep::AprInference, results.apr_inference, "Point 31: APR inference step failed"),
    ];
    for &(step, passed, msg) in checks {
        if (is_full_run || matches!(requested_step, Some(s) if s == step)) && !passed {
            failures.push(msg.to_string());
        }
    }
    failures
}

/// Validate benchmark performance requirements (Points 41-50)
fn validate_benchmark(bench: &BenchmarkComparison) -> Vec<String> {
    let mut failures = Vec::new();

    if let Some(speedup) = bench.speedup_vs_llama {
        if speedup < 25.0 {
            failures.push(format!(
                "Point 41: APR speedup vs llama.cpp is {:.1}%, required ≥25%",
                speedup
            ));
        }
    }
    if let Some(speedup) = bench.speedup_vs_ollama {
        if speedup < 25.0 {
            failures.push(format!(
                "Point 42: APR speedup vs Ollama is {:.1}%, required ≥25%",
                speedup
            ));
        }
    }

    let cv = if bench.apr_tps > 0.0 {
        (bench.apr_tps_stddev / bench.apr_tps) * 100.0
    } else {
        100.0
    };
    if cv > 5.0 {
        failures.push(format!(
            "Point 49: Benchmark CV is {:.1}%, required <5%",
            cv
        ));
    }

    if bench.runs < 30 {
        failures.push(format!(
            "Point 50: Only {} benchmark runs, required ≥30",
            bench.runs
        ));
    }
    failures
}

/// Report failures and return error, or print success message
fn report_failures(failures: &[String], success_msg: &str) -> Result<()> {
    if failures.is_empty() {
        println!();
        println!("{}", success_msg.green().bold());
        return Ok(());
    }
    println!();
    println!("{}", "═══ Falsification Failures ═══".red().bold());
    for failure in failures {
        println!("  {} {}", "✗".red(), failure);
    }
    Err(CliError::ValidationFailed(format!(
        "{} falsification point(s) failed",
        failures.len()
    )))
}

pub(super) fn validate_falsification(
    results: &ShowcaseResults,
    config: &ShowcaseConfig,
) -> Result<()> {
    let is_full_run = config.auto_verify || matches!(config.step, Some(ShowcaseStep::All));
    let requested_step = config.step;

    // Standalone demos pass without requiring full pipeline
    if matches!(
        requested_step,
        Some(ShowcaseStep::CudaDemo | ShowcaseStep::ZramDemo | ShowcaseStep::BrickDemo)
    ) {
        println!();
        println!(
            "{}",
            "═══ Demo Complete (standalone mode) ═══".green().bold()
        );
        return Ok(());
    }

    let mut failures = validate_steps(results, is_full_run, requested_step);

    // Skip benchmark validation for single non-benchmark steps
    if !is_full_run && !matches!(requested_step, Some(ShowcaseStep::Benchmark)) {
        return report_failures(&failures, "═══ Step Complete ═══");
    }

    let Some(ref bench) = results.benchmark else {
        if is_full_run || matches!(requested_step, Some(ShowcaseStep::Benchmark)) {
            failures
                .push("Benchmark data missing - required for performance validation".to_string());
        }
        return report_failures(&failures, "═══ Step Complete ═══");
    };

    failures.extend(validate_benchmark(bench));
    report_failures(&failures, "═══ All Falsification Points Passed ═══")
}
