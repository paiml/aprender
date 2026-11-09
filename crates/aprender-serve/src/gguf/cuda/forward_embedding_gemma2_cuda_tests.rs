//! Tests for the EmbeddingGemma 2 GPU forward (APR-EMBED-001 EG-4, #4903).
//!
//! Tests cover:
//! - `eg2_attention` against the CPU's `parallel_multihead_attention_windowed`: global and
//!   windowed, grouped-query, a length that is not a multiple of a warp, and a score buffer
//!   small enough to force many passes
//! - The whole forward against the CPU forward on the unit-test file and the served file
//! - Long inputs at 2,000 and 8,000 tokens against the CPU
//! - I-5: an executor whose fp16 GEMM is armed is refused, naming `gemm_f16_to_f32`
//! - The CPU's refusals hold on the GPU: no tokens, more than the context length

use super::*;
use crate::gguf::forward_embedding_gemma2::tests::{long_fixture, served_fixture, small_fixture};
use crate::gguf::OwnedQuantizedModel;

fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn load(bytes: &[u8]) -> EmbeddingGemma2 {
    let model = GGUFModel::from_bytes(bytes).expect("the fixture parses");
    EmbeddingGemma2::from_gguf(&model, bytes).expect("the fixture loads")
}

/// The GPU and CPU vectors of `tokens`, which must agree to `tol` per value.
fn assert_parity(model: &EmbeddingGemma2, tokens: &[u32], tol: f32) {
    let executor = crate::cuda_executor_or_skip!(0);
    let mut gpu = EmbeddingGemma2Cuda::new(model, executor).expect("the encoder uploads");
    let want = model.embed(tokens).expect("the CPU embeds");
    let got = gpu.embed(tokens).expect("the GPU embeds");
    assert!(got.iter().all(|v| v.is_finite()), "I-5: a non-finite value");
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff <= tol,
        "{} tokens: GPU differs from CPU by {diff} (tol {tol})",
        tokens.len()
    );
    // The same call twice gives the same vector: no state carries between calls.
    assert_eq!(gpu.embed(tokens).expect("the GPU embeds again"), got);
}

fn attention_case(seq: u32, heads: u32, kv: u32, d: u32, window: Option<u32>, cap: usize) {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let (q_len, kv_len) = ((seq * heads * d) as usize, (seq * kv * d) as usize);
    let (q, k, v) = (ramp(q_len, 1), ramp(kv_len, 2), ramp(kv_len, 3));
    let want = OwnedQuantizedModel::parallel_multihead_attention_windowed(
        &q,
        &k,
        &v,
        seq as usize,
        heads as usize,
        kv as usize,
        d as usize,
        1.0,
        window.map(|w| w as usize),
    )
    .expect("the CPU attends");

    let up = |x: &[f32]| GpuBuffer::from_host(executor.context(), x).expect("upload");
    let (qb, kb, vb) = (up(&q), up(&k), up(&v));
    let out = GpuBuffer::<f32>::new(executor.context(), q_len).expect("alloc");
    let scores = GpuBuffer::<f32>::new(executor.context(), cap).expect("alloc");
    executor
        .eg2_attention(
            qb.as_ptr(),
            kb.as_ptr(),
            vb.as_ptr(),
            out.as_ptr(),
            scores.as_ptr(),
            cap,
            seq,
            heads,
            kv,
            d,
            1.0,
            window,
        )
        .expect("the GPU attends");
    executor.synchronize_all().expect("sync");
    let mut got = vec![0.0f32; q_len];
    out.copy_to_host(&mut got).expect("download");
    let diff = max_abs_diff(&got, &want);
    assert!(
        diff < 1e-5,
        "seq {seq} heads {heads}/{kv} d {d} window {window:?} cap {cap}: diff {diff}"
    );
}

#[test]
#[serial_test::serial]
fn attention_matches_the_cpu_global_and_windowed_eg4() {
    for window in [None, Some(0), Some(3), Some(100)] {
        // One pass holds every score.
        attention_case(37, 4, 2, 8, window, 1 << 20);
        // A pass of a few rows: the query rows run in many passes.
        attention_case(37, 4, 2, 8, window, 2 * 5 * 37);
        // Multi-head attention with no grouping, a head wider than a warp.
        attention_case(70, 2, 2, 64, window, 1 << 20);
    }
}

#[test]
#[serial_test::serial]
fn attention_refuses_a_buffer_too_small_for_one_row_eg4() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    let buf = GpuBuffer::<f32>::new(executor.context(), 4096).expect("alloc");
    let p = buf.as_ptr();
    let err = executor
        .eg2_attention(p, p, p, p, p, 7, 16, 2, 1, 4, 1.0, None)
        .expect_err("a 7-float buffer cannot hold one row of 2 × 16 scores");
    assert!(err.to_string().contains("one query row"), "{err}");
}

#[test]
#[serial_test::serial]
fn the_forward_matches_the_cpu_on_the_unit_file_eg4() {
    let (bytes, tokens) = small_fixture();
    let model = load(&bytes);
    assert_parity(&model, &tokens, 1e-5);
    assert_parity(&model, &tokens[..1], 1e-5);
}

#[test]
#[serial_test::serial]
fn the_forward_matches_the_cpu_on_the_served_file_eg4() {
    let model = load(&served_fixture(768));
    assert_parity(&model, &[2, 4, 5, 6, 7, 8, 9, 1], 1e-5);
}

#[test]
#[serial_test::serial]
fn long_inputs_match_the_cpu_at_2000_and_8000_tokens_eg4() {
    let model = load(&long_fixture(8192, 512));
    for seq in [2000u32, 8000] {
        let tokens: Vec<u32> = (0..seq).map(|i| (i * 7 + 3) % 5).collect();
        assert_parity(&model, &tokens, 1e-4);
    }
}

#[test]
#[serial_test::serial]
fn an_armed_fp16_gemm_is_refused_by_name_eg4() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    executor.set_qwen35_prefill_f16(true);
    let (bytes, _) = small_fixture();
    let Err(err) = EmbeddingGemma2Cuda::new(&load(&bytes), executor) else {
        panic!("I-5: an executor with the fp16 GEMM armed must be refused");
    };
    assert!(err.to_string().contains("gemm_f16_to_f32"), "{err}");
}

#[test]
#[serial_test::serial]
fn a_projection_refuses_when_the_fp16_gemm_is_armed_later_eg4() {
    let mut executor = crate::cuda_executor_or_skip!(0);
    executor.set_qwen35_prefill_f16(true);
    let buf = GpuBuffer::<f32>::new(executor.context(), 64).expect("alloc");
    let p = buf.as_ptr();
    let err = executor
        .eg2_project(WeightQuantType::F32, p, p, p, 1, 4, 4, 1.0)
        .expect_err("I-5: the projection must refuse");
    assert!(err.to_string().contains("gemm_f16_to_f32"), "{err}");
}

#[test]
#[serial_test::serial]
fn the_cpu_refusals_hold_on_the_gpu_eg4() {
    let executor = crate::cuda_executor_or_skip!(0);
    let (bytes, _) = small_fixture();
    let model = load(&bytes);
    let mut gpu = EmbeddingGemma2Cuda::new(&model, executor).expect("the encoder uploads");
    assert!(gpu.embed(&[]).is_err());
    let too_long = vec![1u32; 17];
    let cpu = model
        .embed(&too_long)
        .expect_err("the CPU refuses 17 tokens of 16");
    let got = gpu
        .embed(&too_long)
        .expect_err("the GPU refuses 17 tokens of 16");
    assert_eq!(got.to_string(), cpu.to_string());
    let outside = gpu
        .embed(&[99])
        .expect_err("a token outside the vocabulary");
    assert!(
        outside.to_string().contains("outside the vocabulary"),
        "{outside}"
    );
}
