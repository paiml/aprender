//! #5303: `batched_gemv_dispatch` against `gemv_dispatch` row by row, on the
//! projections of the real Qwen3.5-0.8B file.

use super::super::{CudaLayer, CudaQuantWeight};
use crate::cuda::types::WeightQuantType;
use crate::cuda::CudaExecutor;
use crate::gguf::cuda::Qwen35CudaModel;
use crate::gguf::forward_qwen35::Qwen35Model;
use trueno_gpu::driver::GpuBuffer;

/// The real hybrid file, as in the other Qwen3.5 CUDA tests.
const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Row counts the batch scheduler forms (c = 1, 2, 4, 8, 16).
const ROWS: [usize; 5] = [1, 2, 4, 8, 16];

/// Row `r` of an `m × k` activation matrix: bounded, distinct per row and
/// element, with both signs, so no row can equal another by accident.
fn activation(r: usize, k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let x = (r * 7919 + i * 104_729 + 13) % 2003;
            (x as f32 / 2003.0 - 0.5) * 4.0
        })
        .collect()
}

/// Row `r` of `buf` as a view; forget it, it owns nothing.
fn view(buf: &GpuBuffer<f32>, r: usize, width: u32) -> GpuBuffer<f32> {
    Qwen35CudaModel::view(buf, r as u32 * width, width)
}

/// The first projection of each distinct (qtype, n, k) in the model: every
/// FFN gate/up/down and the `lm_head`.
fn distinct_weights(
    gpu: &Qwen35CudaModel<'_>,
) -> Vec<(&'static str, WeightQuantType, u64, u32, u32)> {
    let mut seen: Vec<(&'static str, WeightQuantType, u64, u32, u32)> = Vec::new();
    let mut push = |what: &'static str, w: &CudaQuantWeight| {
        if !seen
            .iter()
            .any(|s| s.1 == w.qtype && s.3 == w.n && s.4 == w.k)
        {
            seen.push((what, w.qtype, w.ptr, w.n, w.k));
        }
    };
    for layer in &gpu.layers {
        match layer {
            CudaLayer::DeltaNet(w) => {
                push("attn_qkv", &w.attn_qkv);
                push("ssm_alpha", &w.ssm_alpha);
                push("ssm_beta", &w.ssm_beta);
                push("attn_gate", &w.attn_gate);
                push("ssm_out", &w.ssm_out);
            },
            CudaLayer::Attention(w) => {
                push("attn_q", &w.attn_q);
                push("attn_k", &w.attn_k);
                push("attn_v", &w.attn_v);
                push("attn_output", &w.attn_output);
            },
        }
        let f = layer.ffn();
        push("ffn_gate", f.gate);
        push("ffn_up", f.up);
        push("ffn_down", f.down);
    }
    push("lm_head", &gpu.lm_head);
    seen
}

/// One weight, one m: the max bit difference over rows, `None` if equal.
fn first_difference(
    ex: &mut CudaExecutor,
    (qtype, ptr, n, k): (WeightQuantType, u64, u32, u32),
    m: usize,
) -> Option<String> {
    let host: Vec<f32> = (0..m).flat_map(|r| activation(r, k as usize)).collect();
    let mut input = GpuBuffer::<f32>::new(ex.context(), m * k as usize).expect("input");
    input.copy_from_host_at(&host, 0).expect("upload");
    ex.invalidate_q8_activation();
    let batched = GpuBuffer::<f32>::new(ex.context(), m * n as usize).expect("batched out");
    let solo = GpuBuffer::<f32>::new(ex.context(), m * n as usize).expect("solo out");

    ex.batched_gemv_dispatch(qtype, ptr, &input, &batched, m as u32, n, k)
        .expect("batched");
    for r in 0..m {
        let x = view(&input, r, k);
        let y = view(&solo, r, n);
        // The Q8 activation cache is keyed on a buffer's base pointer.
        ex.invalidate_q8_activation();
        let run = ex.gemv_dispatch(qtype, ptr, &x, &y, n, k);
        std::mem::forget(x);
        std::mem::forget(y);
        run.expect("single row");
    }
    let mut a = vec![0f32; m * n as usize];
    let mut b = vec![0f32; m * n as usize];
    batched
        .copy_to_host_at(&mut a, 0)
        .expect("download batched");
    solo.copy_to_host_at(&mut b, 0).expect("download solo");
    let mut worst = 0f32;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        if x.to_bits() != y.to_bits() {
            worst = worst.max((x - y).abs());
            first.get_or_insert(i);
        }
    }
    first.map(|i| {
        format!(
            "row {} col {}: batched {} vs single {}, max |diff| {worst}",
            i / n as usize,
            i % n as usize,
            a[i],
            b[i]
        )
    })
}

/// FALSIFY (#5303): for m in {1, 2, 4, 8, 16} and every projection shape the
/// Qwen3.5 forward uses, each row of `batched_gemv_dispatch` is bitwise the
/// row `gemv_dispatch` gives alone. A qtype that is not equal is named with
/// its measured difference.
#[test]
#[serial_test::serial]
fn batched_gemv_rows_equal_the_single_row_kernel() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {MODEL_PATH} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base model");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");

    let weights = distinct_weights(&gpu);
    assert!(!weights.is_empty(), "no projection found");
    let mut unequal = Vec::new();
    for (what, qtype, ptr, n, k) in weights {
        for m in ROWS {
            let diff = first_difference(&mut gpu.executor, (qtype, ptr, n, k), m);
            eprintln!(
                "[gemv] {what} {qtype:?} n={n} k={k} m={m}: {}",
                diff.as_deref().unwrap_or("bitwise equal")
            );
            if let Some(d) = diff {
                unequal.push(format!("{what} {qtype:?} n={n} k={k} m={m}: {d}"));
            }
        }
    }
    assert!(
        unequal.is_empty(),
        "batched GEMV rows differ from the single-row kernel:\n{}",
        unequal.join("\n")
    );
}
