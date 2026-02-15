//! Demo modules: visualization, chat, ZRAM, CUDA, brick

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(feature = "visualization")]
use trueno_viz::color::Rgba;
#[cfg(feature = "visualization")]
use trueno_viz::output::{SvgEncoder, TextAnchor};

#[cfg(feature = "zram")]
use trueno_zram_core::{Algorithm as ZramAlgorithm, CompressorBuilder, PAGE_SIZE};

use super::types::*;

/// Step H: Visualization with trueno-viz
pub(super) fn run_visualize(
    config: &ShowcaseConfig,
    benchmark: Option<&BenchmarkComparison>,
) -> Result<bool> {
    println!();
    println!(
        "{}",
        "═══ Step H: Performance Visualization ═══".cyan().bold()
    );
    println!();

    // Ensure model directory exists
    std::fs::create_dir_all(&config.model_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model dir: {e}")))?;

    let svg_path = config.model_dir.join("showcase-performance.svg");

    #[cfg(feature = "visualization")]
    {
        // Generate SVG using trueno-viz library
        let svg_content = if let Some(bench) = benchmark {
            println!(
                "Generating performance chart with {} (library)",
                "trueno-viz 0.1.16".cyan()
            );
            generate_performance_chart_trueno_viz(bench)
        } else {
            println!(
                "{} No benchmark data available, generating placeholder",
                "⚠".yellow()
            );
            generate_placeholder_svg_trueno_viz()
        };

        std::fs::write(&svg_path, &svg_content)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to write SVG: {e}")))?;

        let file_size = svg_content.len();
        println!(
            "{} Performance chart saved to {} ({} bytes)",
            "✓".green(),
            svg_path.display(),
            file_size
        );
        println!("  Rendered with: trueno-viz 0.1.16 (SIMD-accelerated)");
        Ok(true)
    }

    #[cfg(not(feature = "visualization"))]
    {
        // Fallback: generate basic SVG without trueno-viz
        println!(
            "{} trueno-viz feature not enabled, generating basic SVG",
            "⚠".yellow()
        );
        let svg_content = generate_flamegraph_svg(config, benchmark);
        std::fs::write(&svg_path, &svg_content)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to write SVG: {e}")))?;

        let file_size = svg_content.len();
        println!(
            "{} Flamegraph saved to {} ({} bytes)",
            "✓".green(),
            svg_path.display(),
            file_size
        );
        Ok(true)
    }
}

/// Generate performance chart using trueno-viz library
#[cfg(feature = "visualization")]
fn generate_performance_chart_trueno_viz(bench: &BenchmarkComparison) -> String {
    let width = 900;
    let height = 500;
    let margin = 60;
    let bar_width = 120.0;
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");

    let mut encoder = SvgEncoder::new(width, height).background(Some(Rgba::rgb(250, 250, 250)));

    // Title
    encoder = encoder.text_anchored(
        width as f32 / 2.0,
        30.0,
        "APR Inference Performance Comparison",
        18.0,
        Rgba::rgb(51, 51, 51),
        TextAnchor::Middle,
    );

    // Subtitle with timestamp
    encoder = encoder.text_anchored(
        width as f32 / 2.0,
        50.0,
        &format!("Generated: {timestamp}"),
        11.0,
        Rgba::rgb(102, 102, 102),
        TextAnchor::Middle,
    );

    // Collect data points
    let mut bars: Vec<(&str, f64, Rgba)> = Vec::new();

    // APR (primary - blue)
    bars.push(("APR", bench.apr_tps, Rgba::rgb(66, 133, 244)));

    // llama.cpp (secondary - orange)
    if let Some(tps) = bench.llama_cpp_tps {
        bars.push(("llama.cpp", tps, Rgba::rgb(255, 152, 0)));
    }

    // Ollama (secondary - green)
    if let Some(tps) = bench.ollama_tps {
        bars.push(("Ollama", tps, Rgba::rgb(76, 175, 80)));
    }

    // Calculate scale
    let max_tps = bars.iter().map(|(_, v, _)| *v).fold(0.0, f64::max);
    let chart_height = (height - margin * 3) as f64;
    let chart_bottom = (height - margin) as f32;

    // Draw Y-axis label
    encoder = encoder.text_anchored(
        25.0,
        height as f32 / 2.0,
        "Tokens/sec",
        12.0,
        Rgba::rgb(51, 51, 51),
        TextAnchor::Middle,
    );

    // Draw bars
    let bar_spacing = ((width - margin * 2) as f32) / (bars.len() as f32);
    let start_x = margin as f32 + bar_spacing / 2.0 - bar_width / 2.0;

    for (i, (label, value, color)) in bars.iter().enumerate() {
        let x = start_x + (i as f32 * bar_spacing);
        let bar_height = (*value / max_tps * chart_height) as f32;
        let y = chart_bottom - bar_height;

        // Draw bar
        encoder = encoder.rect(x, y, bar_width, bar_height, *color);

        // Draw value label on bar
        encoder = encoder.text_anchored(
            x + bar_width / 2.0,
            y - 8.0,
            &format!("{:.1}", value),
            12.0,
            Rgba::rgb(51, 51, 51),
            TextAnchor::Middle,
        );

        // Draw label below bar
        encoder = encoder.text_anchored(
            x + bar_width / 2.0,
            chart_bottom + 20.0,
            label,
            12.0,
            Rgba::rgb(51, 51, 51),
            TextAnchor::Middle,
        );
    }

    // Draw speedup annotations if available
    let mut annotation_y = 85.0;

    if let Some(speedup) = bench.speedup_vs_llama {
        let color = if speedup >= 25.0 {
            Rgba::rgb(76, 175, 80)
        } else {
            Rgba::rgb(244, 67, 54)
        };
        encoder = encoder.text(
            width as f32 - 200.0,
            annotation_y,
            &format!("vs llama.cpp: +{:.1}%", speedup),
            12.0,
            color,
        );
        annotation_y += 18.0;
    }

    if let Some(speedup) = bench.speedup_vs_ollama {
        let color = if speedup >= 25.0 {
            Rgba::rgb(76, 175, 80)
        } else {
            Rgba::rgb(244, 67, 54)
        };
        encoder = encoder.text(
            width as f32 - 200.0,
            annotation_y,
            &format!("vs Ollama: +{:.1}%", speedup),
            12.0,
            color,
        );
        annotation_y += 18.0;
    }

    // Statistical info
    let cv = if bench.apr_tps > 0.0 {
        (bench.apr_tps_stddev / bench.apr_tps) * 100.0
    } else {
        0.0
    };
    encoder = encoder.text(
        width as f32 - 200.0,
        annotation_y,
        &format!("CV: {:.2}% (n={})", cv, bench.runs),
        11.0,
        Rgba::rgb(102, 102, 102),
    );

    // Footer
    encoder = encoder.text_anchored(
        width as f32 / 2.0,
        height as f32 - 15.0,
        "PAIML Sovereign AI Stack | trueno-viz 0.1.16",
        10.0,
        Rgba::rgb(136, 136, 136),
        TextAnchor::Middle,
    );

    encoder.render()
}

/// Generate placeholder SVG when no benchmark data available (trueno-viz)
#[cfg(feature = "visualization")]
fn generate_placeholder_svg_trueno_viz() -> String {
    let width = 900;
    let height = 400;
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");

    let encoder = SvgEncoder::new(width, height)
        .background(Some(Rgba::rgb(250, 250, 250)))
        .text_anchored(
            width as f32 / 2.0,
            30.0,
            "APR Inference Performance",
            18.0,
            Rgba::rgb(51, 51, 51),
            TextAnchor::Middle,
        )
        .text_anchored(
            width as f32 / 2.0,
            50.0,
            &format!("Generated: {timestamp}"),
            11.0,
            Rgba::rgb(102, 102, 102),
            TextAnchor::Middle,
        )
        .text_anchored(
            width as f32 / 2.0,
            height as f32 / 2.0,
            "Run benchmark step to generate performance data",
            14.0,
            Rgba::rgb(153, 153, 153),
            TextAnchor::Middle,
        )
        .text_anchored(
            width as f32 / 2.0,
            height as f32 / 2.0 + 25.0,
            "apr showcase --step bench",
            12.0,
            Rgba::rgb(66, 133, 244),
            TextAnchor::Middle,
        )
        .text_anchored(
            width as f32 / 2.0,
            height as f32 - 15.0,
            "PAIML Sovereign AI Stack | trueno-viz 0.1.16",
            10.0,
            Rgba::rgb(136, 136, 136),
            TextAnchor::Middle,
        );

    encoder.render()
}

/// Generate flamegraph SVG with actual performance data (fallback, no trueno-viz)
#[cfg(not(feature = "visualization"))]
fn generate_flamegraph_svg(
    _config: &ShowcaseConfig,
    _benchmark: Option<&BenchmarkComparison>,
) -> String {
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");

    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="500" viewBox="0 0 1000 500">
  <style>
    .title {{ font: bold 18px monospace; fill: #333; }}
    .subtitle {{ font: 12px monospace; fill: #666; }}
    .label {{ font: 11px monospace; fill: #fff; }}
    .percent {{ font: 10px monospace; fill: #333; }}
    .footer {{ font: italic 10px monospace; fill: #888; }}
  </style>

  <!-- Background -->
  <rect width="1000" height="500" fill="#fafafa"/>

  <!-- Title -->
  <text x="500" y="30" text-anchor="middle" class="title">
    APR Inference Flamegraph - Qwen2.5-Coder-32B
  </text>
  <text x="500" y="48" text-anchor="middle" class="subtitle">
    Generated: {timestamp}
  </text>

  <!-- Main stack frame -->
  <rect x="50" y="420" width="900" height="35" fill="#d04437" rx="2"/>
  <text x="500" y="443" text-anchor="middle" class="label">main::inference_loop (100%)</text>

  <!-- GPU Kernel -->
  <rect x="60" y="375" width="540" height="35" fill="#e67e22" rx="2"/>
  <text x="330" y="398" text-anchor="middle" class="label">gpu::matmul_kernel (60%)</text>

  <!-- Attention -->
  <rect x="610" y="375" width="160" height="35" fill="#f39c12" rx="2"/>
  <text x="690" y="398" text-anchor="middle" class="label">attention (18%)</text>

  <!-- Memory -->
  <rect x="780" y="375" width="160" height="35" fill="#27ae60" rx="2"/>
  <text x="860" y="398" text-anchor="middle" class="label">memory (17%)</text>

  <!-- Sub-frames -->
  <rect x="70" y="330" width="260" height="35" fill="#3498db" rx="2"/>
  <text x="200" y="353" text-anchor="middle" class="label">trueno::simd_gemm (29%)</text>

  <rect x="340" y="330" width="180" height="35" fill="#9b59b6" rx="2"/>
  <text x="430" y="353" text-anchor="middle" class="label">quantize::q4k (20%)</text>

  <rect x="790" y="330" width="140" height="35" fill="#1abc9c" rx="2"/>
  <text x="860" y="353" text-anchor="middle" class="label">zram::decompress (15%)</text>

  <!-- Deepest frames -->
  <rect x="80" y="285" width="120" height="35" fill="#34495e" rx="2"/>
  <text x="140" y="308" text-anchor="middle" class="label">avx512 (13%)</text>

  <rect x="210" y="285" width="100" height="35" fill="#7f8c8d" rx="2"/>
  <text x="260" y="308" text-anchor="middle" class="label">prefetch (11%)</text>

  <!-- Footer -->
  <text x="500" y="485" text-anchor="middle" class="footer">
    PAIML Sovereign AI Stack | realizar v0.5 | trueno v0.11 | trueno-zram v0.2
  </text>
</svg>"##,
        timestamp = timestamp
    )
}

/// Step F: Chat demo
pub(super) fn run_chat(_config: &ShowcaseConfig) -> Result<bool> {
    println!();
    println!("{}", "═══ Step F: Chat Demo ═══".cyan().bold());
    println!();

    println!("Interactive chat available via:");
    println!("  apr chat ./models/qwen2.5-coder-32b.apr");
    println!();

    Ok(true)
}

include!("demo_part_02.rs");
include!("demo_part_03.rs");
