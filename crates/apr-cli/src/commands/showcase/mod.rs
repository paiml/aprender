//! Qwen2.5-Coder-32B Showcase Demo (PMAT-201: split from monolithic showcase.rs)

pub mod benchmark;
pub mod demo;
pub mod pipeline;
pub mod types;
pub mod validation;

// Re-exports for backward compatibility
pub use types::*;

#[cfg(test)]
mod tests;

use crate::error::Result;
use colored::Colorize;

use benchmark::{export_benchmark_results, run_benchmark};
use validation::{print_header, print_summary, validate_falsification};

/// All showcase steps in execution order
fn all_steps() -> Vec<ShowcaseStep> {
    vec![
        ShowcaseStep::Import,
        ShowcaseStep::GgufInference,
        ShowcaseStep::Convert,
        ShowcaseStep::AprInference,
        ShowcaseStep::BrickDemo,
        ShowcaseStep::Benchmark,
        ShowcaseStep::Visualize,
        ShowcaseStep::ZramDemo,
        ShowcaseStep::CudaDemo,
    ]
}

/// Print available showcase steps when none specified
fn print_available_steps() {
    println!(
        "{}",
        "No step specified. Use --auto-verify or --step <step>".yellow()
    );
    println!();
    println!("Available steps:");
    println!("  import        - Download model from HuggingFace");
    println!("  gguf          - Run GGUF inference");
    println!("  convert       - Convert GGUF to APR format");
    println!("  apr           - Run APR inference");
    println!("  brick         - ComputeBrick timing with bottleneck detection");
    println!("  bench         - Run benchmark comparison");
    println!("  visualize     - Generate performance visualization");
    println!("  zram          - Run ZRAM compression demo");
    println!("  cuda          - CUDA Graph + DP4A brick demo (Sections 5.2/5.3)");
    println!("  all           - Run all steps");
}

/// Execute a single showcase step, updating results
fn execute_step(
    step: ShowcaseStep,
    config: &ShowcaseConfig,
    results: &mut ShowcaseResults,
) -> Result<()> {
    match step {
        ShowcaseStep::Import => results.import = pipeline::run_import(config)?,
        ShowcaseStep::GgufInference => {
            results.gguf_inference = pipeline::run_gguf_inference(config)?;
        }
        ShowcaseStep::Convert => results.convert = pipeline::run_convert(config)?,
        ShowcaseStep::AprInference => {
            results.apr_inference = pipeline::run_apr_inference(config)?;
        }
        ShowcaseStep::Benchmark => results.benchmark = Some(run_benchmark(config)?),
        ShowcaseStep::Visualize => {
            results.visualize = demo::run_visualize(config, results.benchmark.as_ref())?;
        }
        ShowcaseStep::Chat => results.chat = demo::run_chat(config)?,
        ShowcaseStep::ZramDemo => results.zram_demo = Some(demo::run_zram_demo(config)?),
        ShowcaseStep::CudaDemo => results.cuda_demo = Some(demo::run_cuda_demo(config)?),
        ShowcaseStep::BrickDemo => results.brick_demo = Some(demo::run_brick_demo(config)?),
        ShowcaseStep::All => unreachable!(),
    }
    Ok(())
}

/// Run the showcase demo
pub fn run(config: &ShowcaseConfig) -> Result<()> {
    print_header(config.tier);

    let steps = match config.step {
        Some(ShowcaseStep::All) => all_steps(),
        None if config.auto_verify => all_steps(),
        Some(step) => vec![step],
        None => {
            print_available_steps();
            return Ok(());
        }
    };

    let mut results = ShowcaseResults::default();

    for step in steps {
        execute_step(step, config, &mut results)?;
    }

    // Export benchmark results if requested (Point 85)
    if let Some(ref bench) = results.benchmark {
        export_benchmark_results(bench, config)?;
    }

    print_summary(&results, config);
    validate_falsification(&results, config)?;

    Ok(())
}
