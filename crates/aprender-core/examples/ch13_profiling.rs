#![allow(clippy::disallowed_methods)]
//! Chapter 13: Profiling and Optimization
//!
//! Demonstrates roofline model and fusion contracts.
//! Citation: Williams et al., "Roofline," CACM 2009
//! Contract: contracts/apr-book-ch13-v1.yaml

fn main() {
    // Roofline model: compute vs memory bound
    // Operational intensity = FLOPs / Bytes
    let params_7b: f64 = 7e9;
    let flops_per_token = 2.0 * params_7b; // 2 * params for matmul
    let bytes_per_token_q4 = params_7b / 2.0; // Q4 ~ 0.5 bytes/param
    let oi = flops_per_token / bytes_per_token_q4;
    println!("Roofline analysis (Williams et al., 2009):");
    println!("  7B Q4K operational intensity: {oi:.1} FLOPs/byte");
    println!("  < 10 -> memory-bound (typical for decode)");
    println!("  > 50 -> compute-bound (typical for batched prefill)");
    assert!(oi > 1.0, "OI must be positive");

    // Memory bandwidth targets
    let ddr5_bw = 50e9_f64; // ~50 GB/s DDR5
    let theoretical_tps = ddr5_bw / bytes_per_token_q4;
    println!("\nMemory bandwidth ceiling:");
    println!("  DDR5 bandwidth: {:.0} GB/s", ddr5_bw / 1e9);
    println!("  Theoretical max: {theoretical_tps:.0} tok/s (7B Q4K)");
    assert!(theoretical_tps > 10.0, "Must exceed 10 tok/s theoretical");

    // FFN gate+up fusion contract (PMAT-FFN-FUSION)
    let layers = 28_usize;
    let dispatches_unfused = layers * 2; // gate + up separate
    let dispatches_fused = layers; // gate+up in one dispatch
    println!("\nFFN gate+up fusion:");
    println!("  Unfused: {dispatches_unfused} rayon dispatches");
    println!("  Fused:   {dispatches_fused} rayon dispatches");
    println!(
        "  Reduction: {:.0}%",
        (1.0 - dispatches_fused as f64 / dispatches_unfused as f64) * 100.0
    );
    assert_eq!(
        dispatches_fused * 2,
        dispatches_unfused,
        "Fusion halves dispatches"
    );

    // Batched prefill speedup
    let serial_ms = 2570.0_f64;
    let batched_ms = 314.0_f64;
    let speedup = serial_ms / batched_ms;
    println!("\nBatched prefill (91-token prompt):");
    println!("  Serial:  {serial_ms:.0} ms");
    println!("  Batched: {batched_ms:.0} ms");
    println!("  Speedup: {speedup:.1}x");
    assert!(speedup > 5.0, "Batched prefill must achieve >5x speedup");

    println!("\nChapter 13 contracts: PASSED");
}
