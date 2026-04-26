//! SHIP-007 §15.4 falsifier — Qwen2.5-Coder-7B GQA-7:1 CPU vs GPU attention parity
//!
//! The existing `gqa_attention_parity.rs` test verifies CPU/GPU GQA parity for
//! TinyLlama's GQA-8:1 shape (NUM_HEADS=32, NUM_KV_HEADS=4, HEAD_DIM=64,
//! HIDDEN=2048). That test PASSES on RTX 4090 — the GQA kernel arithmetic is
//! provably correct for that shape.
//!
//! This test adds the **canonical SHIP-TWO-001 7B Qwen2.5-Coder shape**:
//! GQA-7:1 (NUM_HEADS=28, NUM_KV_HEADS=4, HEAD_DIM=128, HIDDEN=3584). The
//! ratio (7:1) is non-power-of-2 — different code path than 8:1.
//!
//! Per spec §15.4 (SPEC-SHIP-TWO-001 v2.59.0), this is the falsifiable next
//! investigation step:
//!   - If the test PASSES → GQA-7:1 attention kernel is correct;
//!     SHIP-007 root cause is upstream (Q/K/V projection) or downstream
//!     (o_proj, FFN, LM head, KV cache state) of the attention itself.
//!   - If the test FAILS → the divergent stage is localized to the
//!     incremental attention kernel for the 28:4 ratio specifically.
//!
//! Either outcome materially advances SHIP-007 root-cause analysis.

#![cfg(feature = "cuda")]
#![allow(unused_imports)]

/// Qwen2.5-Coder-7B-Instruct canonical GQA shape (per
/// `contracts/model-families/qwen2.yaml` and apr inspect on
/// `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr`).
const NUM_HEADS: usize = 28;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const HIDDEN_DIM: usize = NUM_HEADS * HEAD_DIM; // 3584
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 512

/// CPU reference GQA attention — single-token attention over a (cache_len + 1)
/// sequence. Mirrors the per-head dot-product softmax-weighted V aggregation
/// used by the GPU `incremental_attention_gpu` kernel, with the GQA-7:1
/// `q_per_kv = NUM_HEADS / NUM_KV_HEADS = 7` head-to-kv-head mapping.
fn cpu_gqa_attention(
    q: &[f32],         // [HIDDEN_DIM] = [3584]
    k_cache: &[f32],   // [cache_len * KV_DIM]
    v_cache: &[f32],   // [cache_len * KV_DIM]
    current_k: &[f32], // [KV_DIM] = [512]
    current_v: &[f32], // [KV_DIM] = [512]
) -> Vec<f32> {
    assert_eq!(q.len(), HIDDEN_DIM, "q must be [HIDDEN_DIM]");
    assert_eq!(current_k.len(), KV_DIM, "current_k must be [KV_DIM]");
    assert_eq!(current_v.len(), KV_DIM, "current_v must be [KV_DIM]");

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q_per_kv = NUM_HEADS / NUM_KV_HEADS; // 7
    let cache_len = if KV_DIM > 0 {
        k_cache.len() / KV_DIM
    } else {
        0
    };

    let mut output = vec![0.0f32; HIDDEN_DIM];

    for q_head in 0..NUM_HEADS {
        let q_head_offset = q_head * HEAD_DIM;
        let q_head_data = &q[q_head_offset..q_head_offset + HEAD_DIM];

        // GQA-7:1 head mapping: q_head 0..6 → kv_head 0; 7..13 → 1; 14..20 → 2; 21..27 → 3.
        let kv_head = q_head / q_per_kv;
        let kv_head_offset = kv_head * HEAD_DIM;

        let mut scores = Vec::with_capacity(cache_len + 1);

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

        let curr_key = &current_k[kv_head_offset..kv_head_offset + HEAD_DIM];
        let current_score: f32 = q_head_data
            .iter()
            .zip(curr_key.iter())
            .map(|(a, b)| a * b)
            .sum();
        scores.push(current_score * scale);

        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            exp_sum += *s;
        }
        for s in &mut scores {
            *s /= exp_sum;
        }

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

/// Property: GQA-7:1 head mapping is arithmetically correct.
/// q_head ∈ [0..28), kv_head = q_head / 7, kv_head ∈ [0..4).
/// Cross-check the kernel mapping formula
/// `(q_head * NUM_KV_HEADS) / NUM_HEADS` produces the same result for all 28 q_heads.
#[test]
fn ship_007_qwen2_gqa_7_1_head_mapping_property() {
    let q_per_kv = NUM_HEADS / NUM_KV_HEADS;
    assert_eq!(q_per_kv, 7, "Qwen2.5-Coder-7B GQA-7:1 ratio must be 7");

    for q_head in 0..NUM_HEADS {
        let expected_kv_head = q_head / q_per_kv;
        let kernel_kv_head = (q_head * NUM_KV_HEADS) / NUM_HEADS;

        assert_eq!(
            expected_kv_head, kernel_kv_head,
            "Q head {q_head} should map to KV head {expected_kv_head} (got {kernel_kv_head})",
        );

        // Spot-check: q_head 6 → kv_head 0, q_head 7 → kv_head 1
        match q_head {
            0..=6 => assert_eq!(expected_kv_head, 0),
            7..=13 => assert_eq!(expected_kv_head, 1),
            14..=20 => assert_eq!(expected_kv_head, 2),
            21..=27 => assert_eq!(expected_kv_head, 3),
            _ => unreachable!(),
        }
    }
}

/// SHIP-007 §15.4 falsifier — first token CPU vs GPU GQA-7:1 attention parity.
///
/// First token case: cache is empty. Attention over a single K/V position
/// reduces to softmax([single_score]) = [1.0], so output = current_v expanded
/// from 4 KV heads to 28 Q heads (each q_head gets the V slice for its
/// mapped kv_head).
///
/// Pass criteria: CPU and GPU outputs are element-wise within 1e-4 across
/// all 3584 elements. Failure surfaces as either:
/// - Mismatch at a specific kv_head (kernel head-mapping bug at 28:4)
/// - Mismatch at a specific element index (Q × K^T or scale bug at head_dim 128)
/// - Wholesale divergence (KV cache layout bug for 28:4 specifically)
#[test]
#[ignore] // Run with --ignored when CUDA is available (mirrors peer test pattern)
fn ship_007_qwen2_gqa_7_1_cpu_gpu_parity_first_token() {
    use realizar::cuda::CudaExecutor;

    let executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA not available ({e:?}), skipping GPU parity test");
            return;
        },
    };

    let max_seq_len = 64;
    let num_layers = 1;

    let mut executor = executor;
    if let Err(e) =
        executor.init_kv_cache_gpu(num_layers, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, max_seq_len)
    {
        eprintln!("Failed to init GPU KV cache for GQA-7:1 ({e:?}), skipping");
        return;
    }

    // Deterministic synthetic test data — distinct from peer GQA-8:1 test
    // multipliers so that test failures cannot be attributed to test-input
    // collision with the existing test bank.
    let q: Vec<f32> = (0..HIDDEN_DIM)
        .map(|i| ((i * 19) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_k: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 43) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let current_v: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 47) % 100) as f32 * 0.01)
        .collect();

    // CPU reference: first token = V expanded across the GQA-7:1 mapping.
    let q_per_kv = NUM_HEADS / NUM_KV_HEADS;
    let mut cpu_output = vec![0.0f32; HIDDEN_DIM];
    for q_head in 0..NUM_HEADS {
        let kv_head = q_head / q_per_kv;
        let v_start = kv_head * HEAD_DIM;
        let out_start = q_head * HEAD_DIM;
        cpu_output[out_start..out_start + HEAD_DIM]
            .copy_from_slice(&current_v[v_start..v_start + HEAD_DIM]);
    }

    // GPU execution.
    let mut gpu_output = vec![0.0f32; HIDDEN_DIM];
    let layer_idx = 0;

    if let Err(e) =
        executor.incremental_attention_gpu(layer_idx, &q, &current_k, &current_v, &mut gpu_output)
    {
        panic!("GPU GQA-7:1 attention kernel failed at first-token case: {e:?}");
    }

    let tolerance = 1e-4;
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;
    let mut first_mismatches = Vec::new();

    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > tolerance {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
            if first_mismatches.len() < 10 {
                let q_head = i / HEAD_DIM;
                let kv_head_for_q = q_head / q_per_kv;
                first_mismatches.push(format!(
                    "  idx {i:5} (q_head={q_head:2} kv_head={kv_head_for_q}): CPU={cpu:+.6} GPU={gpu:+.6} Δ={diff:.6}",
                ));
            }
        }
    }

    if diff_count > 0 {
        eprintln!("SHIP-007 §15.4 falsifier — first-token CPU/GPU GQA-7:1 mismatches:");
        for line in &first_mismatches {
            eprintln!("{line}");
        }
        eprintln!(
            "Total mismatches: {diff_count}/{HIDDEN_DIM} (max diff: {max_diff:.6})",
        );
    }

    assert!(
        max_diff < tolerance,
        "SHIP-007: GQA-7:1 first-token CPU/GPU outputs differ by > {tolerance:.0e}: \
         max_diff={max_diff:.6}, mismatches={diff_count}/{HIDDEN_DIM}. \
         Per spec §15.4, this localizes the SHIP-007 divergent stage to the GQA-7:1 \
         incremental attention kernel for the 28:4 ratio specifically.",
    );
}

/// SHIP-007 §15.4 falsifier — second token CPU vs GPU GQA-7:1 attention parity
/// with one cached K/V position.
///
/// This is the simplest non-trivial case: KV cache contains one position, then
/// attention is computed over the (cached + current) = 2-position context. Any
/// KV-cache-layout bug specific to the 28:4 ratio surfaces here but not in the
/// first-token case (which has no cache).
#[test]
#[ignore] // Run with --ignored when CUDA is available
fn ship_007_qwen2_gqa_7_1_cpu_gpu_parity_second_token() {
    use realizar::cuda::CudaExecutor;

    let executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA not available ({e:?}), skipping GPU parity test");
            return;
        },
    };

    let max_seq_len = 64;
    let num_layers = 1;
    let layer_idx = 0;

    let mut executor = executor;
    if let Err(e) =
        executor.init_kv_cache_gpu(num_layers, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, max_seq_len)
    {
        eprintln!("Failed to init GPU KV cache for GQA-7:1 ({e:?}), skipping");
        return;
    }

    // First token populates K/V cache (different from third-test inputs).
    let q_first: Vec<f32> = (0..HIDDEN_DIM)
        .map(|i| ((i * 23) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let cached_k: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 53) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let cached_v: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 59) % 100) as f32 * 0.01)
        .collect();

    let mut first_output = vec![0.0f32; HIDDEN_DIM];
    if let Err(e) = executor.incremental_attention_gpu(
        layer_idx, &q_first, &cached_k, &cached_v, &mut first_output,
    ) {
        panic!("GPU first-token populate failed: {e:?}");
    }

    // Second token Q + new K/V — distinct from first-token data.
    let q_second: Vec<f32> = (0..HIDDEN_DIM)
        .map(|i| ((i * 29) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let new_k: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 61) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let new_v: Vec<f32> = (0..KV_DIM)
        .map(|i| ((i * 67) % 100) as f32 * 0.01)
        .collect();

    let cpu_output = cpu_gqa_attention(&q_second, &cached_k, &cached_v, &new_k, &new_v);

    let mut gpu_output = vec![0.0f32; HIDDEN_DIM];
    if let Err(e) = executor.incremental_attention_gpu(
        layer_idx, &q_second, &new_k, &new_v, &mut gpu_output,
    ) {
        panic!("GPU second-token attention failed: {e:?}");
    }

    let tolerance = 1e-3; // Slightly looser than first-token (cumulative FP rounding)
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;
    let mut first_mismatches = Vec::new();
    let q_per_kv = NUM_HEADS / NUM_KV_HEADS;

    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > tolerance {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
            if first_mismatches.len() < 10 {
                let q_head = i / HEAD_DIM;
                let kv_head_for_q = q_head / q_per_kv;
                first_mismatches.push(format!(
                    "  idx {i:5} (q_head={q_head:2} kv_head={kv_head_for_q}): CPU={cpu:+.6} GPU={gpu:+.6} Δ={diff:.6}",
                ));
            }
        }
    }

    if diff_count > 0 {
        eprintln!("SHIP-007 §15.4 falsifier — second-token CPU/GPU GQA-7:1 mismatches:");
        for line in &first_mismatches {
            eprintln!("{line}");
        }
        eprintln!(
            "Total mismatches: {diff_count}/{HIDDEN_DIM} (max diff: {max_diff:.6})",
        );
    }

    assert!(
        max_diff < tolerance,
        "SHIP-007: GQA-7:1 second-token CPU/GPU outputs differ by > {tolerance:.0e}: \
         max_diff={max_diff:.6}, mismatches={diff_count}/{HIDDEN_DIM}. \
         Per spec §15.4, this localizes the SHIP-007 divergent stage to the GQA-7:1 \
         incremental attention kernel under KV-cache state.",
    );
}
