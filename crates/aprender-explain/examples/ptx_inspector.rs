//! PTX Inspector - Examine generated PTX for hidden bugs
//!
//! Run: `cargo run -p trueno-explain --example ptx_inspector`

use trueno_explain::{BugSeverity, PtxBugAnalyzer};
use trueno_gpu::kernels::{AttentionKernel, GemmKernel, Kernel, LayerNormKernel, SoftmaxKernel};

fn inspect_ptx(name: &str, ptx: &str) {
    println!("\n{}", "=".repeat(80));
    println!("INSPECTING: {}", name);
    println!("{}", "=".repeat(80));

    // Check for specific patterns
    let has_st_shared = ptx.contains("st.shared");
    let has_ld_shared = ptx.contains("ld.shared");
    let has_barrier = ptx.contains("bar.sync");
    let has_shared_decl = ptx.contains(".shared");
    let has_local = ptx.contains(".local");
    let has_wmma = ptx.contains("wmma.");
    let has_shfl = ptx.contains("shfl.");

    println!("\nPTX Features:");
    println!("  .shared declaration: {}", has_shared_decl);
    println!("  st.shared ops: {}", has_st_shared);
    println!("  ld.shared ops: {}", has_ld_shared);
    println!("  bar.sync: {}", has_barrier);
    println!("  .local (spills): {}", has_local);
    println!("  wmma (tensor core): {}", has_wmma);
    println!("  shfl (warp shuffle): {}", has_shfl);

    // Potential bugs
    println!("\nPotential Issues:");
    if has_shared_decl && (has_st_shared || has_ld_shared) && !has_barrier {
        println!("  ⚠️  MISSING BARRIER: Shared memory used but no bar.sync!");
    }
    if has_local {
        println!("  ⚠️  REGISTER SPILLS: .local memory detected");
    }

    // Run analyzer
    let result = PtxBugAnalyzer::strict().analyze(ptx);
    println!("\nBug Analyzer Results:");
    println!("  Total bugs: {}", result.bugs.len());
    println!(
        "  P0 Critical: {}",
        result.count_by_severity(BugSeverity::Critical)
    );
    println!("  P1 High: {}", result.count_by_severity(BugSeverity::High));
    println!(
        "  P2 Medium: {}",
        result.count_by_severity(BugSeverity::Medium)
    );

    for bug in &result.bugs {
        println!("  - [{}] {}", bug.class.code(), bug.message);
    }

    // Show PTX snippet with shared memory ops
    if has_st_shared || has_ld_shared {
        println!("\nShared Memory Operations Found:");
        for (i, line) in ptx.lines().enumerate() {
            if line.contains("st.shared") || line.contains("ld.shared") || line.contains("bar.sync")
            {
                println!("  Line {}: {}", i + 1, line.trim());
            }
        }
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                          PTX INSPECTOR - DEEP DIVE                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Inspect kernels that use shared memory
    inspect_ptx(
        "gemm_tiled_64",
        &GemmKernel::tiled(64, 64, 64, 16).emit_ptx(),
    );
    inspect_ptx(
        "gemm_tensor_core",
        &GemmKernel::tensor_core(64, 64, 64).emit_ptx(),
    );
    inspect_ptx(
        "gemm_wmma_fp16",
        &GemmKernel::wmma_fp16(64, 64, 64).emit_ptx(),
    );
    inspect_ptx("softmax_1024", &SoftmaxKernel::new(1024).emit_ptx());
    inspect_ptx("layernorm_256", &LayerNormKernel::new(256).emit_ptx());
    inspect_ptx("attention_64_32", &AttentionKernel::new(64, 32).emit_ptx());

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("INSPECTION COMPLETE");
    println!("{}", "=".repeat(80));
}
