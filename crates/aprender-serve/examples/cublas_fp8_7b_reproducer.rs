//! SPEC-CUBLAS-FP8-7B-FIX-001 Stage A — Deterministic reproducer.
//!
//! Outputs a single JSON line capturing the cuBLAS FP8 forward output on a
//! 7B Q4K GGUF for token_id=791 at position 0. Designed for bit-identity
//! comparison across consecutive runs.
//!
//! Run with:
//!
//! ```sh
//! MODEL_PATH=${APR_MODELS:?}/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
//!     cargo run --example cublas_fp8_7b_reproducer \
//!     --release -p aprender-serve --features cuda
//! ```
//!
//! Expected output (single line on stdout, all other diagnostics on stderr):
//!
//! ```json
//! {"cpu_argmax_idx":75311,"cpu_argmax_val":11.554419,"gpu_argmax_idx":1057,
//!  "gpu_argmax_val":11.132793,"correlation":0.986986,
//!  "gpu_logits_fnv1a":"<16-hex>","cpu_logits_fnv1a":"<16-hex>",
//!  "agrees_with_cpu":false}
//! ```
//!
//! Falsifier (see `contracts/cublas-fp8-7b-determinism-v1.yaml`):
//! running this binary 5 times in sequence MUST produce 5 bit-identical
//! JSON lines on stdout. If the bug is fixed, `agrees_with_cpu` will be `true`.
//! If still broken, `false` with reproducible argmax+correlation values.
//!
//! Context: #1864 cuBLAS FP8 7B Q4K gibberish. 2026-05-22 layer-by-layer
//! trace showed Layer 0 Q/K inputs differ between CPU and cuBLAS; logit
//! correlation 0.987 (high), linear fit GPU ≈ 0.96 × CPU + 0.12. This
//! reproducer locks that observation as a numerical signature.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    // Deterministic 64-bit FNV-1a fingerprint of a logit slice — avoids
    // taking on `sha2` or `hex` deps for what is in essence a checksum.
    fn fnv1a_64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        )
        .to_string()
    });

    // Deterministic probe: same token, same position. Token 791 = canonical
    // probe from CORRECTNESS-011 / layer_by_layer_trace.
    let token_id: u32 = 791;
    let position: usize = 0;

    eprintln!(
        "[cublas_fp8_7b_reproducer] model={} token={} pos={}",
        path, token_id, position
    );

    // Load model (CPU side).
    let mapped = MappedGGUFModel::from_path(&path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // CPU forward.
    let cpu_logits = model.forward(&[token_id])?;
    let (cpu_argmax_idx, cpu_argmax_val) =
        cpu_logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(idx, v), (i, &x)| {
                if x > v {
                    (i, x)
                } else {
                    (idx, v)
                }
            });

    // GPU forward via cuBLAS FP8 path.
    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;
    cuda_model.clear_decode_graph();

    let mut dummy_cache = realizar::gguf::OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        100,
    );
    let gpu_logits = cuda_model.forward_gpu_resident(token_id, &mut dummy_cache, position)?;

    let (gpu_argmax_idx, gpu_argmax_val) =
        gpu_logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(idx, v), (i, &x)| {
                if x > v {
                    (i, x)
                } else {
                    (idx, v)
                }
            });

    // Linear-fit correlation (matches layer_by_layer_trace's diagnostic).
    let n = cpu_logits.len().min(gpu_logits.len()) as f32;
    let mean_cpu: f32 = cpu_logits[..n as usize].iter().sum::<f32>() / n;
    let mean_gpu: f32 = gpu_logits[..n as usize].iter().sum::<f32>() / n;
    let (mut cov, mut var_cpu, mut var_gpu) = (0.0f32, 0.0f32, 0.0f32);
    for (c, g) in cpu_logits.iter().zip(gpu_logits.iter()) {
        let dc = c - mean_cpu;
        let dg = g - mean_gpu;
        cov += dc * dg;
        var_cpu += dc * dc;
        var_gpu += dg * dg;
    }
    let correlation = cov / (var_cpu.sqrt() * var_gpu.sqrt() + 1e-10);

    // FNV-1a fingerprint of logit bytes (LE f32) for bit-identity.
    let cpu_fp: u64 = {
        let mut all = Vec::with_capacity(cpu_logits.len() * 4);
        for v in &cpu_logits {
            all.extend_from_slice(&v.to_le_bytes());
        }
        fnv1a_64(&all)
    };
    let gpu_fp: u64 = {
        let mut all = Vec::with_capacity(gpu_logits.len() * 4);
        for v in &gpu_logits {
            all.extend_from_slice(&v.to_le_bytes());
        }
        fnv1a_64(&all)
    };

    let agrees = cpu_argmax_idx == gpu_argmax_idx;

    // Single-line JSON on stdout (all diagnostic prose on stderr).
    println!(
        "{{\"cpu_argmax_idx\":{},\"cpu_argmax_val\":{:.6},\
          \"gpu_argmax_idx\":{},\"gpu_argmax_val\":{:.6},\
          \"correlation\":{:.6},\
          \"cpu_logits_fnv1a\":\"{:016x}\",\"gpu_logits_fnv1a\":\"{:016x}\",\
          \"agrees_with_cpu\":{}}}",
        cpu_argmax_idx,
        cpu_argmax_val,
        gpu_argmax_idx,
        gpu_argmax_val,
        correlation,
        cpu_fp,
        gpu_fp,
        agrees,
    );

    // Exit 0 when GPU agrees with CPU (bug fixed); exit 1 when they disagree.
    // git bisect / CI invocations rely on this exit code.
    std::process::exit(if agrees { 0 } else { 1 });
}
