//! PMAT-3596 (#3596): the batched Qwen3.5 prefill against the per-token path at long
//! context — the "parity-checked … at 20k" evidence of the done_when.
//!
//! One prompt of `N` tokens (a file's tokens, or a deterministic id sequence) goes
//! through BOTH paths on the same device:
//!
//! - per-token: `N` calls of `forward_single` (what `qwen35_gpu_decode` did before #3596);
//! - batched: one `prefill_logits_at` (the batched prefill, with the `lm_head` run on
//!   the sampled rows so each sampled position has logits to compare).
//!
//! At every `STRIDE`-th position (and the last) it records the cosine and relative L∞
//! between the two logit vectors, both argmaxes, and the prompt's own next token — so
//! the JSON is also the input to the pinned-llama.cpp column (`llama_column.py` asks
//! llama-server for its top-1 at the same ids and positions). Both wall-clock times
//! are printed: the per-token path's IS the old TTFT.
//!
//! Usage:
//!   cargo run --release --features cuda -p aprender-serve --example qwen35_prefill_parity -- \
//!       <model.gguf> <n_tokens> <stride> <out.json> [prompt-token-ids.txt]
//!
//! Run it under the GPU lock (`flock /tmp/apr-gpu.lock choom -n 1000 -- …`): at 20k
//! positions on the 9B the per-token side takes tens of minutes.

use realizar::cuda::CudaExecutor;
use realizar::gguf::forward_qwen35::Qwen35Model;
use realizar::gguf::MappedGGUFModel;
use realizar::gguf::Qwen35CudaModel;
use std::time::Instant;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += f64::from(*x) * f64::from(*y);
        aa += f64::from(*x) * f64::from(*x);
        bb += f64::from(*y) * f64::from(*y);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn rel_linf(got: &[f32], want: &[f32]) -> f64 {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    f64::from(
        got.iter()
            .zip(want)
            .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()))
            / scale,
    )
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: qwen35_prefill_parity <model.gguf> <n_tokens> <stride> <out.json> [ids.txt]"
        );
        std::process::exit(2);
    }
    let (path, n, stride, out) = (
        &args[1],
        args[2].parse::<usize>().expect("n_tokens"),
        args[3].parse::<usize>().expect("stride").max(1),
        &args[4],
    );

    let mapped = MappedGGUFModel::from_path(path).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let vocab = base.config().vocab_size;

    // Token ids: from a file (one id per whitespace), or a deterministic sequence.
    let ids: Vec<u32> = if let Some(f) = args.get(5) {
        std::fs::read_to_string(f)
            .expect("ids file")
            .split_whitespace()
            .map(|t| t.parse().expect("token id"))
            .take(n)
            .collect()
    } else {
        let mut s = 0x3596_0200u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                1000 + (s >> 8) % (vocab as u32 - 2000)
            })
            .collect()
    };
    let n = ids.len();
    assert!(n >= 2, "need at least two tokens");

    let executor = CudaExecutor::new(0).expect("CUDA");
    let mut gpu = Qwen35CudaModel::with_max_seq_len(&qwen, executor, n + 1).expect("gpu model");
    let sampled: Vec<usize> = (0..n).filter(|p| p % stride == 0 || *p == n - 1).collect();

    // Per-token: keep only the sampled positions' logits.
    let mut state = gpu.new_state().expect("state");
    let t0 = Instant::now();
    let mut per_token = Vec::with_capacity(sampled.len());
    let mut next = 0;
    for (pos, &t) in ids.iter().enumerate() {
        let logits = gpu
            .forward_single(t, &mut state, pos)
            .expect("forward_single");
        if sampled.get(next) == Some(&pos) {
            per_token.push(logits);
            next += 1;
        }
    }
    let per_token_ms = t0.elapsed().as_secs_f64() * 1000.0;
    drop(state);

    // Batched, logits at the sampled positions (their lm_head GEMVs are part of this
    // timing, so the batched number here is an UPPER bound on what `prefill` costs).
    let mut state = gpu.new_state().expect("state");
    let t1 = Instant::now();
    let batched = gpu
        .prefill_logits_at(&ids, &mut state, 0, &sampled)
        .expect("prefill");
    let batched_ms = t1.elapsed().as_secs_f64() * 1000.0;
    drop(state);

    // And the production call: one prefill, last logits only.
    let mut state = gpu.new_state().expect("state");
    let t2 = Instant::now();
    let last = gpu.prefill(&ids, &mut state, 0).expect("prefill");
    let prefill_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let mut rows = Vec::new();
    let (mut worst_cos, mut worst_linf, mut argmax_same) = (1.0f64, 0.0f64, 0usize);
    for (i, &pos) in sampled.iter().enumerate() {
        let (b, p) = (&batched[i], &per_token[i]);
        let (c, l) = (cosine(b, p), rel_linf(b, p));
        let (ab, ap) = (argmax(b), argmax(p));
        worst_cos = worst_cos.min(c);
        worst_linf = worst_linf.max(l);
        argmax_same += usize::from(ab == ap);
        rows.push(serde_json::json!({
            "pos": pos,
            "next_token": ids.get(pos + 1),
            "batched_top1": ab,
            "per_token_top1": ap,
            "cosine": c,
            "rel_linf": l,
        }));
    }
    let last_agree = argmax(&last) == argmax(batched.last().expect("the last position is sampled"));
    let report = serde_json::json!({
        "model": path,
        "n_tokens": n,
        "stride": stride,
        "chunk_rows": gpu.prefill_chunk_rows(n),
        "per_token_ms": per_token_ms,
        "batched_every_row_ms": batched_ms,
        "prefill_ms": prefill_ms,
        "sampled_positions": sampled.len(),
        "argmax_agree": argmax_same,
        "worst_cosine": worst_cos,
        "worst_rel_linf": worst_linf,
        "prefill_last_argmax_equals_every_row_last": last_agree,
        "ids": ids,
        "rows": rows,
    });
    std::fs::write(out, serde_json::to_string(&report).expect("json")).expect("write");
    println!(
        "[3596] {path} n={n}: per-token {per_token_ms:.0} ms, batched prefill {prefill_ms:.0} ms \
         ({:.1}x); {argmax_same}/{} sampled argmax agree, worst cosine {worst_cos:.7}, worst rel L-inf \
         {worst_linf:.3e}",
        per_token_ms / prefill_ms.max(1e-9),
        sampled.len()
    );
}
