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

pub(super) fn validate_falsification(
    results: &ShowcaseResults,
    config: &ShowcaseConfig,
) -> Result<()> {
    let mut failures = Vec::new();

    // Determine which steps were requested
    let is_full_run = config.auto_verify || matches!(config.step, Some(ShowcaseStep::All));
    let requested_step = config.step;

    // For single-step runs (not auto-verify or --step all), skip validation of other steps
    // CUDA demo, ZRAM demo, and Brick demo are standalone demos that pass on their own
    let is_standalone_demo = matches!(
        requested_step,
        Some(ShowcaseStep::CudaDemo | ShowcaseStep::ZramDemo | ShowcaseStep::BrickDemo)
    );

    if is_standalone_demo {
        // Standalone demos pass without requiring full pipeline
        println!();
        println!(
            "{}",
            "═══ Demo Complete (standalone mode) ═══".green().bold()
        );
        return Ok(());
    }

    // Check step completion (Points 1-40) - only for full runs
    if (is_full_run || matches!(requested_step, Some(ShowcaseStep::Import))) && !results.import {
        failures.push("Point 1: Import step failed".to_string());
    }
    if (is_full_run || matches!(requested_step, Some(ShowcaseStep::GgufInference)))
        && !results.gguf_inference
    {
        failures.push("Point 11: GGUF inference step failed".to_string());
    }
    if (is_full_run || matches!(requested_step, Some(ShowcaseStep::Convert))) && !results.convert {
        failures.push("Point 21: APR conversion step failed".to_string());
    }
    if (is_full_run || matches!(requested_step, Some(ShowcaseStep::AprInference)))
        && !results.apr_inference
    {
        failures.push("Point 31: APR inference step failed".to_string());
    }

    // Point 41+: Benchmark required only for full runs or explicit bench step
    if !is_full_run && !matches!(requested_step, Some(ShowcaseStep::Benchmark)) {
        // Skip benchmark validation for single non-benchmark steps
        if failures.is_empty() {
            println!();
            println!("{}", "═══ Step Complete ═══".green().bold());
            return Ok(());
        }
    }

    let Some(ref bench) = results.benchmark else {
        if is_full_run || matches!(requested_step, Some(ShowcaseStep::Benchmark)) {
            failures
                .push("Benchmark data missing - required for performance validation".to_string());
        }
        if !failures.is_empty() {
            println!();
            println!("{}", "═══ Falsification Failures ═══".red().bold());
            for failure in &failures {
                println!("  {} {}", "✗".red(), failure);
            }
            return Err(CliError::ValidationFailed(format!(
                "{} falsification point(s) failed",
                failures.len()
            )));
        }
        println!();
        println!("{}", "═══ Step Complete ═══".green().bold());
        return Ok(());
    };

    // Points 41-42: 25% speedup requirement
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

    // Point 49: Coefficient of variation <5%
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

    // Point 50: Minimum 30 runs
    if bench.runs < 30 {
        failures.push(format!(
            "Point 50: Only {} benchmark runs, required ≥30",
            bench.runs
        ));
    }

    if !failures.is_empty() {
        println!();
        println!("{}", "═══ Falsification Failures ═══".red().bold());
        for failure in &failures {
            println!("  {} {}", "✗".red(), failure);
        }
        return Err(CliError::ValidationFailed(format!(
            "{} falsification point(s) failed",
            failures.len()
        )));
    }

    println!();
    println!(
        "{}",
        "═══ All Falsification Points Passed ═══".green().bold()
    );

    Ok(())
}
