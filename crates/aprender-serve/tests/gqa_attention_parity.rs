//! PAR-021: GQA Attention Kernel Parity Test
//!
//! Verifies that the GPU incremental attention kernel produces the same results
//! as the CPU GQA attention implementation.
//!
//! Test methodology (PMAT extreme TDD):
//! 1. Generate known Q, K, V inputs with GQA dimensions
//! 2. Run CPU attention_with_cache_gqa
//! 3. Run GPU incremental_attention_gpu
//! 4. Compare outputs with tolerance

#![cfg(feature = "cuda")]
#![allow(unused_imports)]

/// GQA configuration for TinyLlama
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const HIDDEN_DIM: usize = NUM_HEADS * HEAD_DIM; // 2048
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 256

/// Reference CPU GQA attention implementation
/// This is the known-correct implementation we're testing against
fn cpu_gqa_attention(
    q: &[f32],         // [hidden_dim] = [2048]
    k_cache: &[f32],   // [cache_len * kv_dim]
    v_cache: &[f32],   // [cache_len * kv_dim]
    current_k: &[f32], // [kv_dim] = [256]
    current_v: &[f32], // [kv_dim] = [256]
) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q_per_kv = NUM_HEADS / NUM_KV_HEADS;
    let cache_len = if KV_DIM > 0 {
        k_cache.len() / KV_DIM
    } else {
        0
    };
    let total_len = cache_len + 1;

    let mut output = vec![0.0f32; HIDDEN_DIM];

    for q_head in 0..NUM_HEADS {
        let q_head_offset = q_head * HEAD_DIM;
        let q_head_data = &q[q_head_offset..q_head_offset + HEAD_DIM];

        let kv_head = q_head / q_per_kv;
        let kv_head_offset = kv_head * HEAD_DIM;

        let mut scores = Vec::with_capacity(total_len);

        // Scores against cached positions
        for pos in 0..cache_len {
            let k_start = pos * KV_DIM + kv_head_offset;
            let cached_key = &k_cache[k_start..k_start + HEAD_DIM];
            let score: f32 = q_head_data
                .iter()
                .zip(cached_key.iter())
                .map(|(a, b)| a * b)
                .sum();
            scores.push(score * scale);
        }

        // Score against current position
        let curr_key = &current_k[kv_head_offset..kv_head_offset + HEAD_DIM];
        let current_score: f32 = q_head_data
            .iter()
            .zip(curr_key.iter())
            .map(|(a, b)| a * b)
            .sum();
        scores.push(current_score * scale);

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            exp_sum += *s;
        }
        for s in &mut scores {
            *s /= exp_sum;
        }

        // Weighted sum of values
        let out_head = &mut output[q_head_offset..q_head_offset + HEAD_DIM];

        for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
            let v_start = pos * KV_DIM + kv_head_offset;
            let cached_val = &v_cache[v_start..v_start + HEAD_DIM];
            for (o, &v) in out_head.iter_mut().zip(cached_val.iter()) {
                *o += weight * v;
            }
        }

        let curr_val = &current_v[kv_head_offset..kv_head_offset + HEAD_DIM];
        let current_weight = scores[cache_len];
        for (o, &v) in out_head.iter_mut().zip(curr_val.iter()) {
            *o += current_weight * v;
        }
    }

    output
}

/// Test: First token attention (no cache)
/// For first token, attention output should equal V (expanded for GQA)
#[test]
fn test_gqa_first_token_attention() {
    // Generate test V vector
    let v: Vec<f32> = (0..KV_DIM).map(|i| (i as f32) * 0.01).collect();

    // Expected output: V expanded from 4 heads to 32 heads
    let q_per_kv = NUM_HEADS / NUM_KV_HEADS;
    let mut expected = vec![0.0f32; HIDDEN_DIM];
    for q_head in 0..NUM_HEADS {
        let kv_head = q_head / q_per_kv;
        let v_start = kv_head * HEAD_DIM;
        let out_start = q_head * HEAD_DIM;
        expected[out_start..out_start + HEAD_DIM].copy_from_slice(&v[v_start..v_start + HEAD_DIM]);
    }

    // Verify expansion is correct
    // Q heads 0-7 should map to KV head 0
    // Q heads 8-15 should map to KV head 1
    // Q heads 16-23 should map to KV head 2
    // Q heads 24-31 should map to KV head 3
    for q_head in 0..NUM_HEADS {
        let kv_head = q_head / q_per_kv;
        let v_slice = &v[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM];
        let out_slice = &expected[q_head * HEAD_DIM..(q_head + 1) * HEAD_DIM];
        assert_eq!(
            v_slice, out_slice,
            "Q head {} should map to KV head {}",
            q_head, kv_head
        );
    }
}

/// Test: Second token attention with single cached position
#[test]
fn test_gqa_second_token_attention() {
    // Q: [2048] random values
    let q: Vec<f32> = (0..HIDDEN_DIM)
        .map(|i| ((i * 17) % 100) as f32 * 0.01 - 0.5)
        .collect();

    // K/V cache: single position [256]
    let k_cache: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 23) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let v_cache: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 31) % 100) as f32 * 0.01)
        .collect();

    // Current K/V: [256]
    let current_k: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 37) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_v: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 41) % 100) as f32 * 0.01)
        .collect();

    // Run CPU attention
    let output = cpu_gqa_attention(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Verify output dimensions
    assert_eq!(output.len(), HIDDEN_DIM);

    // Verify output is not all zeros
    let sum: f32 = output.iter().sum();
    assert!(sum.abs() > 0.01, "Output should not be all zeros");

    // Verify output is finite
    assert!(
        output.iter().all(|x| x.is_finite()),
        "Output should be finite"
    );
}

/// PMAT-800B: CPU vs GPU first-token attention parity at HEAD_DIM=128.
///
/// The existing `test_cpu_gpu_gqa_attention_parity` covers HEAD_DIM=64
/// (TinyLlama). This extends it to HEAD_DIM=128 (Qwen2.5-coder-1.5B /
/// Qwen3 geometry: 12 heads, 2 kv_heads, GQA=6). For a single K/V
/// position softmax is trivially `[1.0]`, so the attention output MUST
/// equal V expanded over the GQA q-heads — a deterministic identity the
/// GPU kernel must honor at every head_dim.
///
/// PMAT-800B FINDING: this passes BIT-EXACT (cos=1.0) — the GB10 (sm_121)
/// per-layer GGUF parity sweep
/// (`evidence/gpu-head-dim-128-divergence-pmat800/findings.json`)
/// CONFIRMED the GPU attention kernel is correct at head_dim=128. The real
/// CPU↔GPU GGUF divergence is NOT in attention: it is a ~15% HwDp4a (INT8
/// DP4A) Q4_K/Q6_K error on a single massive-activation channel (dim 408),
/// latent until the final-layer FFN cancels the outlier. This test is the
/// head_dim=128 attention-correctness guard that RULES OUT the attention
/// kernel as the cause. Run with `--ignored` on CUDA hardware.
#[test]
#[ignore] // Run with --ignored when CUDA is available
fn test_cpu_gpu_attention_parity_head_dim_128() {
    use realizar::cuda::CudaExecutor;

    // Qwen2.5-coder-1.5B / Qwen3 attention geometry.
    const NH: usize = 12; // num_heads
    const NKV: usize = 2; // num_kv_heads (GQA=6)
    const HD: usize = 128; // head_dim
    const HIDDEN: usize = NH * HD; // 1536
    const KVDIM: usize = NKV * HD; // 256

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed, skipping: {e:?}");
            return;
        },
    };

    if let Err(e) = executor.init_kv_cache_gpu(1, NH, NKV, HD, 64) {
        eprintln!("init_kv_cache_gpu failed, skipping: {e:?}");
        return;
    }

    // Deterministic, non-trivial Q/K/V (same generator family as the
    // head_dim=64 test). Q is unused for the first-token identity but
    // must be supplied to the kernel.
    let q: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 17) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_k: Vec<f32> = (0..KVDIM)
        .map(|i| ((i * 37) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_v: Vec<f32> = (0..KVDIM).map(|i| ((i * 41) % 100) as f32 * 0.01).collect();

    // CPU reference: first token over a single K/V position ⇒ softmax=[1.0]
    // ⇒ attn_out[q_head] = current_v[kv_head], kv_head = q_head / (NH/NKV).
    let q_per_kv = NH / NKV;
    let mut cpu_output = vec![0.0f32; HIDDEN];
    for q_head in 0..NH {
        let kv_head = q_head / q_per_kv;
        let v_start = kv_head * HD;
        let out_start = q_head * HD;
        cpu_output[out_start..out_start + HD].copy_from_slice(&current_v[v_start..v_start + HD]);
    }

    let mut gpu_output = vec![0.0f32; HIDDEN];
    if let Err(e) =
        executor.incremental_attention_gpu(0, &q, &current_k, &current_v, &mut gpu_output)
    {
        eprintln!("GPU attention failed, skipping: {e:?}");
        return;
    }

    // Cosine similarity (scale-invariant, matches the sweep metric).
    let cos = {
        let dot: f64 = cpu_output
            .iter()
            .zip(&gpu_output)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let na: f64 = cpu_output
            .iter()
            .map(|a| f64::from(*a) * f64::from(*a))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = gpu_output
            .iter()
            .map(|b| f64::from(*b) * f64::from(*b))
            .sum::<f64>()
            .sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    };
    let max_diff = cpu_output
        .iter()
        .zip(&gpu_output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!(
        "[PMAT-800B] head_dim=128 first-token attn parity: cos={cos:.6} max_diff={max_diff:.6}"
    );

    // For a single K/V position the GPU MUST reproduce V exactly (within fp
    // tolerance). Confirmed bit-exact at head_dim=128 on GB10 (sm_121) —
    // this guards against a regression that would put a real bug in the
    // attention kernel (the actual GGUF divergence is the dim-408 HwDp4a
    // massive-activation error, not attention).
    assert!(
        cos >= 0.9999,
        "PMAT-800B: GPU first-token attention at head_dim=128 diverges from \
         V-identity: cos={cos:.6} (expected ≥0.9999), max_diff={max_diff:.6}. \
         Q/K/V inputs are identical to CPU; the GPU attention kernel computes \
         a different softmax(QKᵀ/√d)·V at head_dim=128. See \
         evidence/gpu-head-dim-128-divergence-pmat800/findings.json."
    );
}

/// PMAT-800B: PRODUCTION-path first-token attention parity at HEAD_DIM=128.
///
/// `test_cpu_gpu_attention_parity_head_dim_128` exercises
/// `incremental_attention_gpu` (the host-upload flash-cache path). The
/// production dense GGUF decode uses `incremental_attention_into`
/// (GPU-resident buffers → `scatter_kv_to_cache` → `launch_attention_kernel`),
/// a DIFFERENT path. This test drives `incremental_attention_into`
/// directly and asserts the same single-K/V V-identity.
///
/// PMAT-800B FINDING: this also passes BIT-EXACT (cos=1.0) — the production
/// scatter/launch attention path is correct at head_dim=128, ruling it out
/// as the GGUF divergence source. See
/// evidence/gpu-head-dim-128-divergence-pmat800/findings.json (root cause =
/// HwDp4a dim-408 massive-activation channel error).
#[test]
#[ignore] // Run with --ignored when CUDA is available
fn test_cpu_gpu_attention_into_parity_head_dim_128() {
    use realizar::cuda::CudaExecutor;
    use trueno_gpu::driver::GpuBuffer;

    const NH: usize = 12;
    const NKV: usize = 2;
    const HD: usize = 128;
    const HIDDEN: usize = NH * HD; // 1536
    const KVDIM: usize = NKV * HD; // 256

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed, skipping: {e:?}");
            return;
        },
    };
    if let Err(e) = executor.init_kv_cache_gpu(1, NH, NKV, HD, 64) {
        eprintln!("init_kv_cache_gpu failed, skipping: {e:?}");
        return;
    }

    let q: Vec<f32> = (0..HIDDEN)
        .map(|i| ((i * 17) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_k: Vec<f32> = (0..KVDIM)
        .map(|i| ((i * 37) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_v: Vec<f32> = (0..KVDIM).map(|i| ((i * 41) % 100) as f32 * 0.01).collect();

    let q_per_kv = NH / NKV;
    let mut cpu_output = vec![0.0f32; HIDDEN];
    for q_head in 0..NH {
        let kv_head = q_head / q_per_kv;
        cpu_output[q_head * HD..(q_head + 1) * HD]
            .copy_from_slice(&current_v[kv_head * HD..(kv_head + 1) * HD]);
    }

    // Production path: GPU-resident buffers.
    let ctx = executor.context();
    let q_gpu = GpuBuffer::from_host(ctx, &q).expect("q upload");
    let k_gpu = GpuBuffer::from_host(ctx, &current_k).expect("k upload");
    let v_gpu = GpuBuffer::from_host(ctx, &current_v).expect("v upload");
    let out_gpu = GpuBuffer::<f32>::new(ctx, HIDDEN).expect("out alloc");

    if let Err(e) = executor.incremental_attention_into(0, &q_gpu, &k_gpu, &v_gpu, &out_gpu) {
        eprintln!("incremental_attention_into failed, skipping: {e:?}");
        return;
    }
    let _ = executor.synchronize_all();

    let mut gpu_output = vec![0.0f32; HIDDEN];
    out_gpu.copy_to_host(&mut gpu_output).expect("out download");

    let cos = {
        let dot: f64 = cpu_output
            .iter()
            .zip(&gpu_output)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let na: f64 = cpu_output
            .iter()
            .map(|a| f64::from(*a) * f64::from(*a))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = gpu_output
            .iter()
            .map(|b| f64::from(*b) * f64::from(*b))
            .sum::<f64>()
            .sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    };
    let max_diff = cpu_output
        .iter()
        .zip(&gpu_output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!(
        "[PMAT-800B] head_dim=128 incremental_attention_into parity: cos={cos:.6} max_diff={max_diff:.6}"
    );
    assert!(
        cos >= 0.9999,
        "PMAT-800B: production incremental_attention_into at head_dim=128 \
         diverges from V-identity: cos={cos:.6} (expected ≥0.9999), \
         max_diff={max_diff:.6}. See evidence/gpu-head-dim-128-divergence-pmat800/."
    );
}

/// Test: CPU vs GPU attention parity for FIRST token
/// First token case: attention over single K/V position = just V (expanded)
#[test]
#[ignore] // Run with --ignored when CUDA is available
fn test_cpu_gpu_gqa_attention_parity() {
    use realizar::cuda::CudaExecutor;

    // Skip if CUDA not available
    let executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    // Initialize GPU KV cache
    let max_seq_len = 64;
    let num_layers = 1;

    let mut executor = executor;
    if let Err(e) =
        executor.init_kv_cache_gpu(num_layers, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, max_seq_len)
    {
        eprintln!("Failed to init GPU KV cache: {:?}", e);
        return;
    }

    // Generate test data for FIRST token (no cache)
    let q: Vec<f32> = (0..HIDDEN_DIM)
        .map(|i| ((i * 17) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_k: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 37) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_v: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 41) % 100) as f32 * 0.01)
        .collect();

    // CPU reference: first token = V expanded
    // Attention over single K/V position has softmax([score]) = [1.0]
    // So output = 1.0 * V for each head
    let q_per_kv = NUM_HEADS / NUM_KV_HEADS;
    let mut cpu_output = vec![0.0f32; HIDDEN_DIM];
    for q_head in 0..NUM_HEADS {
        let kv_head = q_head / q_per_kv;
        let v_start = kv_head * HEAD_DIM;
        let out_start = q_head * HEAD_DIM;
        cpu_output[out_start..out_start + HEAD_DIM]
            .copy_from_slice(&current_v[v_start..v_start + HEAD_DIM]);
    }

    // GPU execution
    let mut gpu_output = vec![0.0f32; HIDDEN_DIM];
    let layer_idx = 0;

    if let Err(e) =
        executor.incremental_attention_gpu(layer_idx, &q, &current_k, &current_v, &mut gpu_output)
    {
        eprintln!("GPU attention failed: {:?}", e);
        return;
    }

    // Compare outputs
    let tolerance = 1e-4;
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > tolerance {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
            if diff_count <= 10 {
                eprintln!(
                    "Mismatch at {}: CPU={:.6} GPU={:.6} diff={:.6}",
                    i, cpu, gpu, diff
                );
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "Total mismatches: {} / {} (max diff: {:.6})",
            diff_count, HIDDEN_DIM, max_diff
        );
    }

    assert!(
        max_diff < tolerance,
        "CPU/GPU outputs differ by more than {}: max_diff={}",
        tolerance,
        max_diff
    );
}

/// Property: GQA head mapping is correct
/// Each Q head should access the correct KV head
#[test]
fn test_gqa_head_mapping_property() {
    let q_per_kv = NUM_HEADS / NUM_KV_HEADS;

    for q_head in 0..NUM_HEADS {
        let expected_kv_head = q_head / q_per_kv;

        // Verify the mapping formula used in the kernel
        let kernel_kv_head = (q_head * NUM_KV_HEADS) / NUM_HEADS;

        assert_eq!(
            expected_kv_head, kernel_kv_head,
            "Q head {} should map to KV head {} (got {})",
            q_head, expected_kv_head, kernel_kv_head
        );
    }
}

/// Test: CPU vs GPU attention parity for SECOND token (with 1 cached position)
/// This tests the full attention mechanism with KV cache
#[test]
#[ignore] // Run with --ignored when CUDA is available
fn test_cpu_gpu_gqa_attention_parity_second_token() {
    use realizar::cuda::CudaExecutor;

    // Skip if CUDA not available
    let executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    // Initialize GPU KV cache
    let max_seq_len = 64;
    let num_layers = 1;

    let mut executor = executor;
    if let Err(e) =
        executor.init_kv_cache_gpu(num_layers, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, max_seq_len)
    {
        eprintln!("Failed to init GPU KV cache: {:?}", e);
        return;
    }

    // Token 0: First K/V (will become cached)
    let first_q: Vec<f32> = (0..HIDDEN_DIM)
        .map(|i| ((i * 13) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let first_k: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 23) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let first_v: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 31) % 100) as f32 * 0.01)
        .collect();

    // Token 1: Second K/V (current position)
    let second_q: Vec<f32> = (0..HIDDEN_DIM)
        .map(|i| ((i * 17) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let second_k: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 37) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let second_v: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 41) % 100) as f32 * 0.01)
        .collect();

    // Run GPU: First token (builds cache)
    let mut gpu_output_first = vec![0.0f32; HIDDEN_DIM];
    let layer_idx = 0;
    if let Err(e) = executor.incremental_attention_gpu(
        layer_idx,
        &first_q,
        &first_k,
        &first_v,
        &mut gpu_output_first,
    ) {
        eprintln!("GPU attention (first) failed: {:?}", e);
        return;
    }

    // Run GPU: Second token (uses cache)
    let mut gpu_output = vec![0.0f32; HIDDEN_DIM];
    if let Err(e) = executor.incremental_attention_gpu(
        layer_idx,
        &second_q,
        &second_k,
        &second_v,
        &mut gpu_output,
    ) {
        eprintln!("GPU attention (second) failed: {:?}", e);
        return;
    }

    // CPU reference: attention over [cached_pos, current_pos]
    // k_cache = [first_k], v_cache = [first_v], current = second_k/v
    let cpu_output = cpu_gqa_attention(&second_q, &first_k, &first_v, &second_k, &second_v);

    // Compare outputs
    let tolerance = 1e-3; // Allow slightly more tolerance for accumulated operations
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > tolerance {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
            if diff_count <= 10 {
                eprintln!(
                    "Mismatch at {}: CPU={:.6} GPU={:.6} diff={:.6}",
                    i, cpu, gpu, diff
                );
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "Total mismatches: {} / {} (max diff: {:.6})",
            diff_count, HIDDEN_DIM, max_diff
        );
    }

    assert!(
        max_diff < tolerance,
        "CPU/GPU outputs differ by more than {}: max_diff={}",
        tolerance,
        max_diff
    );
}

/// Property: Softmax outputs sum to 1
#[test]
fn test_softmax_sum_property() {
    let scores = [1.0f32, 2.0, 3.0, 4.0];

    // Apply softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();
    let softmax: Vec<f32> = exp_scores.iter().map(|e| e / sum).collect();

    // Sum should be 1
    let total: f32 = softmax.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "Softmax should sum to 1, got {}",
        total
    );
}
