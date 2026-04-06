//! WASM Pixel GUI Demo Example (PROBAR-SPEC-009)
//!
//! Demonstrates GPU-accelerated random fill of a 1080p pixel grid
//! with real-time TUI visualization and Wilson confidence intervals.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example wasm_pixel_gui_demo
//! cargo run --example wasm_pixel_gui_demo -- --fast  # Fast convergence mode
//! cargo run --example wasm_pixel_gui_demo -- --small # Small test grid
//! ```
//!
//! # References
//!
//! - O'Neill (2014): PCG-XSH-RR random number generation
//! - Wilson (1927): Wilson score confidence intervals
//! - Nickolls et al. (2008): GPU parallel computing model

use jugar_probar::pixel_coverage::{
    ansi, wilson_confidence_interval, GpuPixelBuffer, PcgRng, WasmDemoConfig, WasmPixelDemo,
};
use std::io::{self, Write};
use std::time::Instant;

/// Demo mode for terminal output
#[derive(Debug, Clone, Copy)]
enum DemoMode {
    /// Standard 1080p demo
    Standard,
    /// Fast convergence mode (higher fill probability)
    Fast,
    /// Small test grid for quick verification
    Small,
}

impl DemoMode {
    fn config(&self) -> WasmDemoConfig {
        match self {
            Self::Standard => WasmDemoConfig::hd_1080p(),
            Self::Fast => WasmDemoConfig {
                fill_probability: 0.05,
                ..WasmDemoConfig::hd_1080p()
            },
            Self::Small => WasmDemoConfig::test_small(),
        }
    }
}

fn main() {
    println!("WASM Pixel GUI Demo (PROBAR-SPEC-009)");
    println!("=====================================\n");

    // Parse args
    let mode = std::env::args()
        .nth(1)
        .map(|arg| match arg.as_str() {
            "--fast" => DemoMode::Fast,
            "--small" => DemoMode::Small,
            _ => DemoMode::Standard,
        })
        .unwrap_or(DemoMode::Standard);

    let config = mode.config();
    println!(
        "Configuration: {}x{}, fill_probability={:.2}, target={:.0}%",
        config.width,
        config.height,
        config.fill_probability,
        config.target_coverage * 100.0
    );
    println!();

    // Phase 1: PCG RNG Verification
    println!("Phase 1: PCG-XSH-RR RNG Verification (O'Neill, 2014)");
    println!("----------------------------------------------------");
    verify_pcg_rng();
    println!();

    // Phase 2: GPU Buffer Creation
    println!("Phase 2: GPU Pixel Buffer Creation");
    println!("-----------------------------------");
    let mut demo = WasmPixelDemo::new(config);
    println!(
        "  Created {}x{} buffer ({} pixels)",
        demo.buffer.width,
        demo.buffer.height,
        demo.buffer.total_pixels()
    );

    // Show GPU status
    if demo.buffer.is_using_gpu() {
        if let Some(gpu_name) = GpuPixelBuffer::gpu_device_name() {
            println!(
                "  {}GPU: {} (ACCELERATED){}",
                ansi::PASS,
                gpu_name,
                ansi::RESET
            );
        } else {
            println!(
                "  {}GPU: Available (ACCELERATED){}",
                ansi::PASS,
                ansi::RESET
            );
        }
    } else {
        println!(
            "  {}GPU: Not available (CPU fallback){}",
            ansi::WARN,
            ansi::RESET
        );
        println!(
            "  {}Hint: Compile with --features gpu for GPU acceleration{}",
            ansi::DIM,
            ansi::RESET
        );
    }

    println!(
        "  Initial coverage: {:.2}%",
        demo.buffer.coverage_percentage() * 100.0
    );
    println!();

    // Phase 3: Random Fill Simulation
    println!("Phase 3: GPU Random Fill Simulation");
    println!("------------------------------------");
    run_fill_simulation(&mut demo);
    println!();

    // Phase 4: Wilson Confidence Intervals
    println!("Phase 4: Wilson Score Confidence Intervals (Wilson, 1927)");
    println!("----------------------------------------------------------");
    demonstrate_wilson_ci();
    println!();

    // Phase 5: Terminal Heatmap
    println!("Phase 5: Terminal Heatmap Visualization");
    println!("---------------------------------------");
    render_terminal_heatmap(&demo.buffer);
    println!();

    // Phase 6: Coverage Statistics
    println!("Phase 6: Final Coverage Statistics");
    println!("-----------------------------------");
    print_coverage_stats(&demo);

    println!(
        "\n{}✓ Demo completed successfully!{}",
        ansi::PASS,
        ansi::RESET
    );
}

fn verify_pcg_rng() {
    // Test determinism
    let mut rng1 = PcgRng::new(42);
    let mut rng2 = PcgRng::new(42);

    let vals1: Vec<u32> = (0..5).map(|_| rng1.next_u32()).collect();
    let vals2: Vec<u32> = (0..5).map(|_| rng2.next_u32()).collect();

    println!("  Determinism check (seed=42):");
    println!("    RNG1: {:?}", vals1);
    println!("    RNG2: {:?}", vals2);
    println!(
        "    {}Deterministic: {}{}",
        if vals1 == vals2 {
            ansi::PASS
        } else {
            ansi::FAIL
        },
        vals1 == vals2,
        ansi::RESET
    );

    // Test pixel hash
    let hash1 = PcgRng::hash_pixel(42, 1000, 5);
    let hash2 = PcgRng::hash_pixel(42, 1000, 5);
    println!(
        "  Pixel hash determinism: {} == {} = {}",
        hash1,
        hash2,
        hash1 == hash2
    );

    // Test distribution
    let mut rng = PcgRng::new(12345);
    let samples: Vec<f32> = (0..1000).map(|_| rng.next_f32()).collect();
    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    println!(
        "  Distribution mean (1000 samples): {:.4} (expected ~0.5)",
        mean
    );
}

fn run_fill_simulation(demo: &mut WasmPixelDemo) {
    let max_frames = 500;
    let target = demo.config.target_coverage;

    let start = Instant::now();
    let mut last_print = 0.0f32;

    for frame in 0..max_frames {
        demo.tick();
        let coverage = demo.buffer.coverage_percentage();

        // Print progress every 10%
        if coverage - last_print >= 0.1 || demo.is_complete() {
            print!(
                "\r  Frame {:4}: Coverage {:.1}% ",
                frame + 1,
                coverage * 100.0
            );
            print_progress_bar(coverage, target, 30);
            io::stdout().flush().ok();
            last_print = coverage;
        }

        if demo.is_complete() {
            break;
        }
    }

    println!();
    let elapsed = start.elapsed();
    println!(
        "  Converged in {} frames ({:.2}s)",
        demo.frame_count(),
        elapsed.as_secs_f64()
    );
    println!(
        "  Throughput: {:.1}M pixels/s",
        (demo.buffer.total_pixels() as f64 * demo.frame_count() as f64)
            / elapsed.as_secs_f64()
            / 1_000_000.0
    );
}

fn print_progress_bar(progress: f32, target: f32, width: usize) {
    let filled = ((progress * width as f32) as usize).min(width);
    let target_pos = ((target * width as f32) as usize).min(width);

    print!("[");
    for i in 0..width {
        if i < filled {
            if progress >= target {
                print!("{}█{}", ansi::PASS, ansi::RESET);
            } else {
                print!("{}█{}", ansi::WARN, ansi::RESET);
            }
        } else if i == target_pos {
            print!("{}|{}", ansi::INFO, ansi::RESET);
        } else {
            print!("░");
        }
    }
    print!("]");
}

fn demonstrate_wilson_ci() {
    let test_cases = [
        (50, 100, "50% coverage"),
        (5, 10, "Small sample"),
        (950, 1000, "High coverage"),
        (0, 100, "Zero coverage"),
        (100, 100, "Full coverage"),
    ];

    println!("  Wilson 95% Confidence Intervals:");
    for (successes, total, label) in test_cases {
        let ci = wilson_confidence_interval(successes, total, 0.95);
        let pct = if total > 0 {
            successes as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        println!(
            "    {}: {:.1}% [{:.1}%, {:.1}%]",
            label,
            pct,
            ci.lower * 100.0,
            ci.upper * 100.0
        );
    }

    // Show narrowing with sample size
    println!("\n  CI width narrows with sample size:");
    for n in [10, 100, 1000, 10000] {
        let ci = wilson_confidence_interval(n / 2, n, 0.95);
        let width = (ci.upper - ci.lower) * 100.0;
        println!("    n={:5}: width = {:.2}%", n, width);
    }
}

fn render_terminal_heatmap(buffer: &GpuPixelBuffer) {
    // Downsample to terminal size
    let term_width = 60;
    let term_height = 15;

    let downsampled = buffer.downsample(term_width, term_height);

    println!(
        "  {}x{} -> {}x{} downsampled:",
        buffer.width, buffer.height, term_width, term_height
    );
    println!();

    // Render using Unicode blocks
    print!("  ┌");
    for _ in 0..term_width {
        print!("─");
    }
    println!("┐");

    for y in 0..term_height {
        print!("  │");
        for x in 0..term_width {
            let idx = y * term_width + x;
            let value = downsampled[idx];

            // Map value to viridis-like color
            let (r, g, b) = value_to_viridis(value);
            let char = if value > 0.75 {
                '█'
            } else if value > 0.5 {
                '▓'
            } else if value > 0.25 {
                '▒'
            } else if value > 0.0 {
                '░'
            } else {
                ' '
            };

            print!("{}{}{}", ansi::rgb_fg(r, g, b), char, ansi::RESET);
        }
        println!("│");
    }

    print!("  └");
    for _ in 0..term_width {
        print!("─");
    }
    println!("┘");

    // Legend
    println!();
    println!(
        "  Legend: {} = 0%  {}░{} = 1-25%  {}▒{} = 26-50%  {}▓{} = 51-75%  {}█{} = 76-100%",
        ansi::DIM,
        ansi::rgb_fg(68, 1, 84),
        ansi::RESET,
        ansi::rgb_fg(59, 82, 139),
        ansi::RESET,
        ansi::rgb_fg(33, 145, 140),
        ansi::RESET,
        ansi::rgb_fg(253, 231, 37),
        ansi::RESET,
    );
}

fn value_to_viridis(value: f32) -> (u8, u8, u8) {
    // Simplified viridis palette
    let colors = [
        (68, 1, 84),    // 0.0 - dark purple
        (59, 82, 139),  // 0.25 - blue
        (33, 145, 140), // 0.5 - teal
        (93, 200, 99),  // 0.75 - green
        (253, 231, 37), // 1.0 - yellow
    ];

    let t = value.clamp(0.0, 1.0);
    let idx = (t * 4.0) as usize;
    let idx = idx.min(3);
    let frac = (t * 4.0) - idx as f32;

    let (r1, g1, b1) = colors[idx];
    let (r2, g2, b2) = colors[idx + 1];

    let r = (r1 as f32 * (1.0 - frac) + r2 as f32 * frac) as u8;
    let g = (g1 as f32 * (1.0 - frac) + g2 as f32 * frac) as u8;
    let b = (b1 as f32 * (1.0 - frac) + b2 as f32 * frac) as u8;

    (r, g, b)
}

fn print_coverage_stats(demo: &WasmPixelDemo) {
    let stats = demo.stats();

    println!(
        "  Coverage: {:.2}% ({}/{} pixels)",
        stats.percentage * 100.0,
        stats.covered,
        stats.total
    );
    println!(
        "  Wilson 95% CI: [{:.2}%, {:.2}%]",
        stats.wilson_ci.lower * 100.0,
        stats.wilson_ci.upper * 100.0
    );
    println!("  Gap regions: {}", stats.gaps.len());
    println!("  Max gap size: {} pixels", stats.max_gap_size());
    println!("  Frames: {}", demo.frame_count());
    println!("  Elapsed: {:.2}s", demo.elapsed().as_secs_f64());

    // Pass/fail summary
    let target = demo.config.target_coverage;
    let passed = stats.meets_threshold(target);
    println!();
    println!(
        "  Target: {:.0}% - {}{}{}",
        target * 100.0,
        if passed { ansi::PASS } else { ansi::FAIL },
        if passed { "PASSED" } else { "FAILED" },
        ansi::RESET
    );
}
