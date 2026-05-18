//! M-FFN-GGUF-6 — REAL-TEACHER Path-A vs Path-B Q4K matvec comparison.
//!
//! After M91-M99 falsifier cascade closed all FOUR synthetic-testable
//! amplifier candidates (A1 RoPE, A2 softmax, A3 block-scale, A4 multi-
//! token batch), the residual SHIP-007 §27 magnitude gap is **78×**
//! (down from initial 3920×) — explained by:
//!   - 0.077% per-tensor mechanism (M94 confirmed)
//!   - 5.70× super-linear compounding (M95 confirmed)
//!   - 50× std-ratio measurement sensitivity (M99 confirmed)
//!   - 78× residual = real-weight + RMSNorm + cumulative-layer
//!
//! M-FFN-GGUF-6 directly tests A5 (real-weight non-uniformity) by
//! loading actual layer-3 down_proj Q4K bytes from the canonical 7B
//! Qwen2.5-Coder teacher .apr file and running both Path A (standalone
//! dequant + F32 matmul) and Path B (Q8K activation quant + fused
//! matvec) against a real activation vector.
//!
//! ## Hypothesis
//!
//! A5: real Qwen Q4K weights have heavy-tailed distributions (a few
//! large weights dominating per-tensor matvec); per-tensor rel_diff on
//! real weights may be 5-50× larger than synthetic uniform weights.
//!
//! ## Expected outcomes
//!
//! - rel_diff ≈ 0.077-0.092% (matches synthetic baseline): A5 FALSIFIED;
//!   real-weight non-uniformity does not amplify M94 mechanism. The 78×
//!   residual must come from A6 (RMSNorm rsqrt) or cumulative-layer
//!   interaction.
//! - rel_diff ≈ 0.5-5% (5-50× larger): A5 PARTIALLY CONFIRMED; real-
//!   weight non-uniformity contributes to §27 magnitude.
//! - rel_diff ≥ 5%: A5 CONFIRMED; real-weight non-uniformity is the
//!   dominant remaining amplifier.
//!
//! ## Run instructions
//!
//! ```text
//! cargo test -p aprender-serve --test ffn_gguf_real_teacher_q4k_matvec \
//!   -- --include-ignored --nocapture
//! ```
//!
//! Test is `#[ignore]`-gated and skips cleanly if the canonical 7B
//! teacher .apr is not present on the host.
//!
//! Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` v1.10.0 →
//! v1.11.0 amendment.

use std::path::Path;

const CANONICAL_QWEN25_CODER_7B_APR_PATHS: &[&str] = &[
    "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr",
    "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-apache-q4k-v1.apr",
    "/home/noah/.apr/models/qwen2.5-coder-7b-instruct-q4k.apr",
    "/home/noah/.apr/models/qwen2.5-coder-7b-apache-q4k-v1.apr",
];

/// Layer 3 is the §21 narrowed anomaly site. We test layer-3 down_proj
/// because it's the LAST matmul in the FFN block (after silu(gate)*up),
/// so per-tensor rel_diff there is the closest synthetic analog of
/// §27's measurement.
const ANOMALY_LAYER: usize = 3;

#[test]
#[ignore]
fn falsify_ffn_gguf_014_real_teacher_q4k_matvec_a5_test() {
    use realizar::quantize::{
        dequantize_q4_k_simd, fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
    };

    let Some(apr_path) = CANONICAL_QWEN25_CODER_7B_APR_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "M-FFN-GGUF-6 real-teacher: skipped — no canonical 7B APR \
             teacher in {CANONICAL_QWEN25_CODER_7B_APR_PATHS:?}"
        );
        return;
    };

    eprintln!("M-FFN-GGUF-6 / FALSIFY-FFN-GGUF-014: real-teacher A5 test");
    eprintln!("  apr_path: {apr_path}");
    eprintln!("  layer:    {ANOMALY_LAYER}");
    eprintln!();

    // Load APR transformer to access q4k_layers (Q4K raw bytes).
    let transformer = realizar::apr_transformer::AprTransformer::from_apr_file(apr_path)
        .expect("AprTransformer::from_apr_file failed");

    let q4k_layers = transformer
        .q4k_layers
        .as_ref()
        .expect("Q4K layers missing — model may not be Q4_K_M quantized");

    let layer = q4k_layers
        .get(ANOMALY_LAYER)
        .expect("layer-3 missing from q4k_layers");

    let raw_bytes = layer
        .ffn_down_weight
        .as_ref()
        .expect("ffn_down_weight is None at layer 3 — may be Q6_K instead");

    eprintln!("  layer-{ANOMALY_LAYER} ffn_down_weight:");
    eprintln!(
        "    bytes:  {} ({} KB)",
        raw_bytes.len(),
        raw_bytes.len() / 1024
    );

    // Q4K super-block size = 256 elements / 144 bytes.
    // For down_proj [hidden_dim=4096, intermediate_dim=11008]:
    //   total elements = 4096 * 11008 ≈ 45M
    //   total bytes    = 45M / 256 * 144 ≈ 25 MB
    //
    // Pick the FIRST 256-element super-block (one row's first super-
    // block) as the test fixture. Real Qwen weights have heavy-tailed
    // magnitude variance; a single super-block's f16 d (block scale)
    // tells us the magnitude range.
    let super_block_bytes = &raw_bytes[..144];
    let in_dim = 256;
    let out_dim = 1;

    // Build a realistic activation vector. Use deterministic synthetic
    // pattern that matches the M94 baseline (so rel_diff is comparable).
    let activation: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f32) - 128.0) * 0.05 + ((i % 7) as f32) * 0.01)
        .collect();

    // ---- Path A: standalone dequant + manual F32 dot ----
    let weights_f32 = dequantize_q4_k_simd(super_block_bytes)
        .expect("dequantize_q4_k_simd on real teacher block");
    let result_a: f32 = activation
        .iter()
        .zip(weights_f32.iter())
        .map(|(x, y)| x * y)
        .sum();

    // ---- Path B: Q8K activation quant + fused matvec ----
    let mut q8k_scales = vec![0.0f32; 1];
    let mut q8k_quants = vec![0i8; in_dim];
    quantize_activations_q8k_into(&activation, &mut q8k_scales, &mut q8k_quants)
        .expect("q8k_quant on real teacher");
    let mut result_b_buf = vec![0.0f32; out_dim];
    fused_q4k_q8k_parallel_matvec_into(
        super_block_bytes,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut result_b_buf,
    )
    .expect("fused_matvec on real teacher");
    let result_b = result_b_buf[0];

    let diff = (result_a - result_b).abs();
    let rel_diff = diff / result_a.abs().max(1e-9);

    // Block-scale info: read f16 d directly from bytes to log magnitude.
    let d_f16 = u16::from_le_bytes([super_block_bytes[0], super_block_bytes[1]]);
    let d_f32 = half::f16::from_bits(d_f16).to_f32();
    let dmin_f16 = u16::from_le_bytes([super_block_bytes[2], super_block_bytes[3]]);
    let dmin_f32 = half::f16::from_bits(dmin_f16).to_f32();

    // Stats on dequantized F32 weights.
    let weight_max = weights_f32
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let weight_min = weights_f32.iter().copied().fold(f32::INFINITY, f32::min);
    let weight_l2: f32 = weights_f32.iter().map(|x| x * x).sum::<f32>().sqrt();

    eprintln!();
    eprintln!("M-FFN-GGUF-6 first super-block of layer-3 down_proj:");
    eprintln!("  block scale f16 d:    {d_f32} (raw {d_f16:#06x})");
    eprintln!("  block scale f16 dmin: {dmin_f32} (raw {dmin_f16:#06x})");
    eprintln!("  dequantized weight stats:");
    eprintln!("    min:  {weight_min:.6}");
    eprintln!("    max:  {weight_max:.6}");
    eprintln!("    l2:   {weight_l2:.6}");
    eprintln!();
    eprintln!("M-FFN-GGUF-6 Path A (standalone) vs Path B (Q8K+fused):");
    eprintln!("  Path A: {result_a:.6} ({:#x})", result_a.to_bits());
    eprintln!("  Path B: {result_b:.6} ({:#x})", result_b.to_bits());
    eprintln!("  diff:   {diff:.6}");
    eprintln!("  rel_diff: {:.6}% ({:.6e})", rel_diff * 100.0, rel_diff);

    // Sanity bounds.
    assert!(
        rel_diff > 1e-7,
        "rel_diff essentially zero on real-teacher weights — fixture degenerate"
    );

    // Compare to synthetic baseline (0.077% per-tensor, M94).
    let synthetic_baseline = 0.00077;
    let real_amplification = rel_diff / synthetic_baseline;

    eprintln!(
        "  synthetic baseline (M94 single-tensor): {:.6}%",
        synthetic_baseline * 100.0
    );
    eprintln!("  real-teacher amplification:    {real_amplification:.4}×");
    eprintln!();

    // EMPIRICAL VERDICT:
    if real_amplification > 50.0 {
        eprintln!(
            "M-FFN-GGUF-6: amplification {real_amplification:.2}× > 50.0 — \
             A5 STRONGLY CONFIRMED. Real-weight non-uniformity is the \
             DOMINANT remaining amplifier for the §27 78× residual gap. \
             SHIP-007 §22 fix scope: Option-A (PROMOTE GGUF-PATH semantics \
             into APR forward) is the right path; the current synthetic-\
             chain's 22% upper bound underestimates real-teacher rel_diff \
             by 50×+. The M-FFN-GGUF-5 fix PR scope should expect the \
             real-teacher matvec to converge on Q8K-equivalent precision."
        );
    } else if real_amplification > 5.0 {
        eprintln!(
            "M-FFN-GGUF-6: amplification {real_amplification:.2}× ∈ (5, 50] — \
             A5 PARTIALLY CONFIRMED. Real-weight non-uniformity contributes \
             substantially to §27 magnitude but does not fully explain \
             the 78× residual; A6 (RMSNorm rsqrt) and/or cumulative-layer \
             interaction provide the remainder. SHIP-007 §22 fix may need \
             both Path A→B convergence AND post-norm precision care."
        );
    } else if real_amplification > 1.5 {
        eprintln!(
            "M-FFN-GGUF-6: amplification {real_amplification:.2}× ∈ (1.5, 5] — \
             A5 modest amplification only. Real-weight non-uniformity \
             contributes ~2-5× over synthetic baseline; remaining 16-50× \
             of the 78× residual must come from A6 + cumulative-layer."
        );
    } else {
        eprintln!(
            "M-FFN-GGUF-6: amplification {real_amplification:.2}× ≈ 1× — \
             A5 NOT CONFIRMED. Real-weight non-uniformity does NOT amplify \
             M94 mechanism beyond synthetic baseline. With A1, A2, A3, A4, \
             A5 all FALSIFIED, the §27 78× residual must come from A6 \
             (RMSNorm rsqrt non-linearity) or cumulative-layer interaction. \
             Multi-layer real-teacher chain test is the next deliverable."
        );
    }
}
