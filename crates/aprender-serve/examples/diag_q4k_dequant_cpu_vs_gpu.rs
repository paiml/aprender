//! §40.5 H1 falsifier: Q4K → F32 dequantization correctness on the GPU upload
//! path vs the CPU Q4K-fused reference.
//!
//! `apr run --no-gpu` produces correct output ("2 + 2 equals"); `apr run`
//! (default GPU) produces gibberish ("ampiezza = 1"). Per §40.4, all 4
//! FP8/CUDA env-var falsifiers FAIL — the bug is NOT in FP8 warming, FP8
//! cache, CUDA graph, or FP8 prefill/decode kernels.
//!
//! H1 hypothesis: the Q4K → F32 dequantization on the GPU upload path
//! produces F32 values that DIFFER from a reference Q4K dequantization beyond
//! Q4K rounding tolerance.
//!
//! H2 hypothesis (suspect site): `wgpu_adapter.rs:56-62`:
//!
//!     OwnedQKVWeights::Fused(tensor) => {
//!         let f32_data = dequant_tensor_public(tensor)?;
//!         let q_data = f32_data[..q_dim * hidden].to_vec();
//!         let k_data = f32_data[q_dim * hidden..(q_dim + kv_dim) * hidden].to_vec();
//!         let v_data = f32_data[(q_dim + kv_dim) * hidden..total_out * hidden].to_vec();
//!
//! This row-slices the dequantized fused QKV by [Q-rows | K-rows | V-rows].
//! If the underlying tensor is HEAD-INTERLEAVED instead, the slices pick up
//! wrong data — that's a layout bug.

use realizar::apr::MappedAprModel;
use realizar::gguf::{OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k, dequantize_q5_k, dequantize_q6_k};

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q5_K: u32 = 13;
const GGUF_TYPE_Q6_K: u32 = 14;
const GGUF_TYPE_F32: u32 = 0;

fn stats(label: &str, data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  {:40} n={:>10} mean={:>10.6} std={:>10.6} min={:>10.4} max={:>10.4}",
        label,
        data.len(),
        mean,
        std,
        min,
        max
    );
    (mean, std, min, max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr";

    println!("=== §40.5 H1+H2 falsifier: Q4K dequant + Fused-QKV layout ===");
    println!("Loading APR teacher: {}\n", apr_path);

    let apr_mapped = MappedAprModel::from_path(apr_path)?;
    let model = OwnedQuantizedModel::from_apr(&apr_mapped)?;
    let cfg = model.config();
    let hidden = cfg.hidden_dim;
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let head_dim = cfg.head_dim();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    println!(
        "Config: hidden={} num_heads={} num_kv_heads={} head_dim={} q_dim={} kv_dim={}",
        hidden, num_heads, num_kv_heads, head_dim, q_dim, kv_dim
    );

    println!("\n=== Step 1: QKV storage layout ===");
    let layer0 = &model.layers()[0];
    let q_data: Vec<f32>;
    let k_data: Vec<f32>;
    let v_data: Vec<f32>;
    match &layer0.qkv_weight {
        OwnedQKVWeights::Fused(tensor) => {
            println!("  Layer 0 QKV is FUSED");
            println!(
                "  Tensor: in_dim={} out_dim={} qtype={} data_len_bytes={}",
                tensor.in_dim,
                tensor.out_dim,
                tensor.qtype,
                tensor.data.len()
            );
            let total_out = q_dim + 2 * kv_dim;
            println!(
                "  Expected out_dim should be q_dim+2*kv_dim = {}+{} = {}",
                q_dim,
                2 * kv_dim,
                total_out
            );
            if tensor.out_dim != total_out {
                println!("  ❌ out_dim MISMATCH! Layout mapping needs verification");
            } else {
                println!("  ✓ out_dim matches expected");
            }

            let f32_data = match tensor.qtype {
                GGUF_TYPE_Q4_K => dequantize_q4_k(&tensor.data)?,
                GGUF_TYPE_Q5_K => dequantize_q5_k(&tensor.data)?,
                GGUF_TYPE_Q6_K => dequantize_q6_k(&tensor.data)?,
                _ => {
                    println!("  unsupported qtype: {}", tensor.qtype);
                    return Ok(());
                },
            };
            println!("  Dequantized {} f32 values", f32_data.len());
            q_data = f32_data[..q_dim * hidden].to_vec();
            k_data = f32_data[q_dim * hidden..(q_dim + kv_dim) * hidden].to_vec();
            v_data = f32_data[(q_dim + kv_dim) * hidden..total_out * hidden].to_vec();
        },
        OwnedQKVWeights::Separate { q, k, v } => {
            println!("  Layer 0 QKV is SEPARATE");
            println!(
                "  q tensor: in_dim={} out_dim={} qtype={} data_len_bytes={}",
                q.in_dim,
                q.out_dim,
                q.qtype,
                q.data.len()
            );
            println!(
                "  k tensor: in_dim={} out_dim={} qtype={} data_len_bytes={}",
                k.in_dim,
                k.out_dim,
                k.qtype,
                k.data.len()
            );
            println!(
                "  v tensor: in_dim={} out_dim={} qtype={} data_len_bytes={}",
                v.in_dim,
                v.out_dim,
                v.qtype,
                v.data.len()
            );
            let dq = |t: &realizar::gguf::OwnedQuantizedTensor| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                match t.qtype {
                    GGUF_TYPE_Q4_K => Ok(dequantize_q4_k(&t.data)?),
                    GGUF_TYPE_Q5_K => Ok(dequantize_q5_k(&t.data)?),
                    GGUF_TYPE_Q6_K => Ok(dequantize_q6_k(&t.data)?),
                    GGUF_TYPE_F32 => Ok(t
                        .data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()),
                    _ => Err(format!("unsupported qtype {}", t.qtype).into()),
                }
            };
            q_data = dq(q)?;
            k_data = dq(k)?;
            v_data = dq(v)?;
            println!("  → Separate layout — wgpu_adapter.rs:56-62 fused-slicing not exercised.");
        },
    }

    println!("\n=== Step 2: Per-projection statistics ===");
    stats("Q-projection (dequant)", &q_data);
    stats("K-projection (dequant)", &k_data);
    stats("V-projection (dequant)", &v_data);

    println!("\n=== Step 3: First 8 elements per projection ===");
    println!("  Q[0..8]: {:?}", &q_data[..8.min(q_data.len())]);
    println!("  K[0..8]: {:?}", &k_data[..8.min(k_data.len())]);
    println!("  V[0..8]: {:?}", &v_data[..8.min(v_data.len())]);

    println!("\n=== Step 4: Sanity bounds ===");
    let q_max_abs = q_data.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let k_max_abs = k_data.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let v_max_abs = v_data.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!(
        "  Max |Q| = {:.6}, Max |K| = {:.6}, Max |V| = {:.6}",
        q_max_abs, k_max_abs, v_max_abs
    );
    println!("  Expected for typical Q4K weights: |max| < ~10.0 (sane Gaussian-ish)");
    if q_max_abs > 100.0 || k_max_abs > 100.0 || v_max_abs > 100.0 {
        println!(
            "  ⚠ At least one projection has |max| > 100 — possible dequantization defect or layout bug"
        );
    } else {
        println!("  ✓ All projections within sane bounds");
    }

    println!("\n=== VERDICT ===");
    println!("  This script establishes the Q-projection's dequantized values ground-truth via");
    println!("  the same `dequantize_q4_k` function that the GPU upload path uses (per");
    println!("  wgpu_adapter.rs:54 / dequant_tensor_public). Stats and first-elements above are");
    println!("  the OBSERVABLE for H1+H2 falsification. Next: instrument the actual GPU upload");
    println!("  path to dump its uploaded F32 weight bytes, then diff vs the values above.");

    Ok(())
}
