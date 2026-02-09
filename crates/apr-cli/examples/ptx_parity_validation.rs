//! PTX Parity Validation Example (GH-219, F-PTX-001)
//!
//! Demonstrates compile-time structural validation of batched GPU kernels against
//! their single-vector reference implementations. This is a Poka-Yoke (error-proofing)
//! system that catches PTX generation bugs before they reach runtime.
//!
//! # What It Validates
//!
//! For each of 6 batched kernel pairs:
//! 1. **Batch dispatch mechanism** — `ctaid.y` (grid_y) or `m_dim` (register_unroll)
//! 2. **No u64 shared memory addressing** — must use u32 for portability
//! 3. **Dispatch strategy** — elementwise kernels use grid_y, GEMV uses register_unroll
//!
//! # Kernel Pairs
//!
//! | # | Batched Kernel | Reference | Strategy |
//! |---|----------------|-----------|----------|
//! | 1 | BatchedRmsNorm | RmsNorm | grid_y |
//! | 2 | BatchedQ4KGemv | Q4KGemv | register_unroll |
//! | 3 | BatchedQ6KGemv | Q6KGemv | register_unroll |
//! | 4 | BatchedResidualAdd | ResidualAdd | grid_y |
//! | 5 | BatchedRoPE | RoPE | grid_y |
//! | 6 | BatchedSwiGLU | SwiGLU | grid_y |
//!
//! # Running
//!
//! ```bash
//! # Run with CUDA (validates actual PTX)
//! cargo run -p apr-cli --example ptx_parity_validation --features inference,cuda
//!
//! # Or run without CUDA (shows structure only)
//! cargo run -p apr-cli --example ptx_parity_validation --features inference
//! ```
//!
//! # Integration with `apr qa`
//!
//! ```bash
//! # Gate 6 runs this validation automatically against real model dimensions
//! apr qa model.gguf --verbose
//! ```

use realizar::ptx_parity::{KernelDimensions, PtxParityReport};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("     GH-219: PTX Parity Validation — Poka-Yoke for GPU Kernels");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Demo 1: Validate with Qwen2.5-Coder-1.5B dimensions
    demo_validate_1_5b();

    // Demo 2: Validate with Qwen2.5-Coder-7B dimensions
    demo_validate_7b();

    // Demo 3: Show dispatch strategy explanation
    demo_dispatch_strategies();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                         Demo Complete");
    println!("═══════════════════════════════════════════════════════════════════");
}

/// Demo 1: Validate PTX parity for 1.5B model dimensions
fn demo_validate_1_5b() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 1: Qwen2.5-Coder-1.5B (Q4K) — 6 Kernel Pairs             │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let dims = KernelDimensions {
        hidden_dim: 1536,
        intermediate_dim: 8960,
        num_heads: 12,
        head_dim: 128,
        rope_theta: 1_000_000.0,
        epsilon: 1e-6,
    };

    let report = realizar::ptx_parity::validate_all_kernel_pairs(&dims);
    print_report(&report, &dims);
}

/// Demo 2: Validate PTX parity for 7B model dimensions
fn demo_validate_7b() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 2: Qwen2.5-Coder-7B (Q4K) — 6 Kernel Pairs               │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let dims = KernelDimensions {
        hidden_dim: 3584,
        intermediate_dim: 18944,
        num_heads: 28,
        head_dim: 128,
        rope_theta: 1_000_000.0,
        epsilon: 1e-6,
    };

    let report = realizar::ptx_parity::validate_all_kernel_pairs(&dims);
    print_report(&report, &dims);
}

/// Demo 3: Explain the two dispatch strategies
fn demo_dispatch_strategies() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 3: Batch Dispatch Strategies                               │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    println!("  Two strategies for extending single-vector kernels to batched:");
    println!();
    println!("  1. grid_y (ctaid.y) — Elementwise kernels");
    println!("     Each batch element gets a separate grid Y index.");
    println!("     Used by: RmsNorm, ResidualAdd, RoPE, SwiGLU");
    println!("     PTX check: presence of %ctaid.y register");
    println!();
    println!("  2. register_unroll (m_dim) — Quantized GEMV kernels");
    println!("     Batch dimension folded into the M (output rows) dimension.");
    println!("     Each warp processes one output row across all batch elements.");
    println!("     Used by: Q4K GEMV, Q6K GEMV");
    println!("     PTX check: m_dim parameter in kernel signature");
    println!();
    println!("  Why two strategies?");
    println!("  - Elementwise ops are embarrassingly parallel per-element.");
    println!("    grid_y maps naturally to independent vectors.");
    println!("  - GEMV is memory-bandwidth-bound. Unrolling across batch");
    println!("    elements within the same warp improves memory coalescing");
    println!("    and shared memory reuse across the batch.");
    println!();
}

/// Print a parity report with detailed kernel-by-kernel results
fn print_report(report: &PtxParityReport, dims: &KernelDimensions) {
    println!("  Model dimensions:");
    println!("    hidden_dim:       {}", dims.hidden_dim);
    println!("    intermediate_dim: {}", dims.intermediate_dim);
    println!("    num_heads:        {}", dims.num_heads);
    println!("    head_dim:         {}", dims.head_dim);
    println!("    rope_theta:       {}", dims.rope_theta);
    println!("    epsilon:          {}\n", dims.epsilon);

    if report.total == 0 {
        println!("  (No CUDA feature — PTX validation requires --features cuda)\n");
        return;
    }

    println!("  ┌──────────────────────────────────┬──────────┬──────────────────┐");
    println!("  │ Kernel Pair                      │ Status   │ Dispatch         │");
    println!("  ├──────────────────────────────────┼──────────┼──────────────────┤");

    for result in &report.results {
        let status = if result.passed {
            "\x1b[32mPASS\x1b[0m"
        } else {
            "\x1b[31mFAIL\x1b[0m"
        };
        // Use fixed width for the status field to account for ANSI codes
        println!(
            "  │ {:<32} │ {}     │ {:<16} │",
            result.name, status, result.dispatch_strategy
        );

        // Show violations if any
        for violation in &result.violations {
            println!("  │   \x1b[31m{:<72}\x1b[0m │", truncate(violation, 72));
        }
    }

    println!("  └──────────────────────────────────┴──────────┴──────────────────┘");
    println!();

    if report.all_passed() {
        println!("  \x1b[32m{}\x1b[0m", report.summary());
    } else {
        println!("  \x1b[31m{}\x1b[0m", report.summary());
    }
    println!();
}

/// Truncate a string to max length
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}
