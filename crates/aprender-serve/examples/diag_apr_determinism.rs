//! Test the parallel-reduction-nondeterminism hypothesis for SHIP-007.
//!
//! Per evidence/ship-007-layer3-bisection-2026-04-28/per-layer-accumulation.md,
//! the layer-3 sub-FFN divergence APR vs GGUF may stem from non-deterministic
//! f32 accumulation order in APR's parallel matmul (rayon).
//!
//! This diagnostic:
//! 1. Loads the canonical 7B teacher
//! 2. Runs forward() twice with the same prompt tokens
//! 3. Element-wise compares the final logits
//!
//! If logits differ across runs:
//!   → APR forward is non-deterministic
//!   → parallel reduction is the source
//!   → fix = deterministic reduction order
//!
//! If logits are identical:
//!   → APR is deterministic
//!   → the APR vs GGUF gap is structural (different kernel path)
//!   → next investigation = compare APR vs GGUF kernels on same synthetic input

use realizar::apr_transformer::AprTransformer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr";
    println!("Loading {}...", path);
    let t = AprTransformer::from_apr_file(path)?;

    // "What is 2+2?" prompt tokens (matches the apr trace evidence)
    let token_ids: Vec<u32> = vec![3838, 374, 220, 17, 10, 17, 30];

    println!("\n=== Run 1 ===");
    let logits1 = t.forward(&token_ids)?;
    let n = logits1.len();
    println!("logits.len() = {}", n);
    println!("first 5 = {:?}", &logits1[..5]);

    println!("\n=== Run 2 ===");
    let logits2 = t.forward(&token_ids)?;
    println!("first 5 = {:?}", &logits2[..5]);

    // Element-wise diff
    let mut max_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f32;
    let mut nonzero_diffs = 0usize;
    let mut first_diff_idx = None;
    for i in 0..n {
        let d = (logits1[i] - logits2[i]).abs();
        if d > 0.0 {
            nonzero_diffs += 1;
            if first_diff_idx.is_none() {
                first_diff_idx = Some(i);
            }
        }
        if d > max_diff {
            max_diff = d;
        }
        sum_sq_diff += d * d;
    }
    let rms = (sum_sq_diff / n as f32).sqrt();

    println!("\n=== VERDICT ===");
    println!("Total elements: {}", n);
    println!(
        "Non-zero diffs: {} ({:.3}%)",
        nonzero_diffs,
        100.0 * nonzero_diffs as f64 / n as f64
    );
    println!("Max abs diff:   {:.10}", max_diff);
    println!("RMS diff:       {:.10}", rms);
    if let Some(i) = first_diff_idx {
        println!(
            "First diff at idx {}: {:.10} vs {:.10} (diff = {:.10})",
            i,
            logits1[i],
            logits2[i],
            (logits1[i] - logits2[i]).abs()
        );
    }

    println!("\n=== HYPOTHESIS TEST ===");
    if max_diff == 0.0 {
        println!("APR forward is BYTE-IDENTICAL across runs.");
        println!("→ Parallel-reduction-nondeterminism hypothesis FALSIFIED.");
        println!("→ APR vs GGUF gap is structural (different kernel path).");
        println!("→ Next: compare APR vs GGUF kernels on same synthetic input.");
    } else {
        println!(
            "APR forward DIFFERS across runs by max abs diff = {:.6e}",
            max_diff
        );
        println!("→ Parallel-reduction-nondeterminism hypothesis CONFIRMED.");
        println!("→ Fix: enforce deterministic reduction order in APR matmul kernels.");
        println!("→ Once fixed, layer-3 ffn_swigl ratio should drop and 5 SHIP-007 PARTIALs flip to DISCHARGED.");
    }

    Ok(())
}
