//! #4228: the layer-major batched prefill must be BIT-IDENTICAL to one
//! `forward_single_qwen35` per token — logits, every recurrent state, and the KV cache —
//! across chunk boundaries and a non-zero start position. Real Qwen3.5 GGUFs; each model
//! is skipped when absent, and `APR_QWEN35_REQUIRE_MODEL=1` turns a skip into a failure.

use super::{Qwen35Model, Qwen35State, QWEN35_PREFILL_CHUNK};

/// `$APR_QWEN35_MODEL_DIR/<file>`, else `$HOME/models/<file>` — never a literal home.
fn model_path(file: &str) -> String {
    let dir = std::env::var("APR_QWEN35_MODEL_DIR")
        .unwrap_or_else(|_| format!("{}/models", std::env::var("HOME").unwrap_or_default()));
    format!("{dir}/{file}")
}

fn load(path: &str) -> Option<crate::gguf::MappedGGUFModel> {
    if !std::path::Path::new(path).exists() {
        assert!(
            std::env::var("APR_QWEN35_REQUIRE_MODEL").as_deref() != Ok("1"),
            "APR_QWEN35_REQUIRE_MODEL=1 but {path} is absent"
        );
        eprintln!("SKIP: {path} is absent");
        return None;
    }
    Some(crate::gguf::MappedGGUFModel::from_path(path).expect("map the GGUF"))
}

/// Deterministic, in-vocab, non-repeating-ish prompt.
fn prompt(n: usize, vocab: usize) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u64 * 2_654_435_761 + 17) % (vocab as u64 - 1000) + 500) as u32)
        .collect()
}

fn assert_states_eq(a: &Qwen35State, b: &Qwen35State, what: &str) {
    assert_eq!(a.conv_states, b.conv_states, "{what}: conv states differ");
    assert_eq!(a.ssm_states, b.ssm_states, "{what}: ssm states differ");
    assert_eq!(
        a.kv_cache.len(),
        b.kv_cache.len(),
        "{what}: kv seq_len differs"
    );
    for il in 0..a.conv_states.len() {
        assert_eq!(
            a.kv_cache.get_k(il),
            b.kv_cache.get_k(il),
            "{what}: K cache layer {il}"
        );
        assert_eq!(
            a.kv_cache.get_v(il),
            b.kv_cache.get_v(il),
            "{what}: V cache layer {il}"
        );
    }
}

fn check_model(path: &str) {
    let Some(mapped) = load(path) else { return };
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base model");
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())
        .expect("hybrid layers");

    // 2 full chunks + a ragged tail, then a second prefill at a non-zero start position.
    let n1 = 2 * QWEN35_PREFILL_CHUNK + 7;
    let n2 = 5;
    let toks = prompt(n1 + n2, base.config.vocab_size);
    let max = n1 + n2 + 2;

    let mut per_token = qwen.new_state(max);
    let mut want = Vec::new();
    for (pos, &t) in toks[..n1].iter().enumerate() {
        want = qwen
            .forward_single_qwen35(t, &mut per_token, pos)
            .expect("forward");
    }
    let mut batched = qwen.new_state(max);
    let got = qwen
        .forward_prefill_qwen35(&toks[..n1], &mut batched, 0)
        .expect("prefill");
    assert_eq!(got, want, "{path}: prefill logits differ from per-token");
    assert_states_eq(&batched, &per_token, "after prefill");

    for (i, &t) in toks[n1..].iter().enumerate() {
        want = qwen
            .forward_single_qwen35(t, &mut per_token, n1 + i)
            .expect("forward");
    }
    let got = qwen
        .forward_prefill_qwen35(&toks[n1..], &mut batched, n1)
        .expect("second prefill");
    assert_eq!(got, want, "{path}: second prefill logits differ");
    assert_states_eq(&batched, &per_token, "after second prefill");

    // Decode continues identically from the batched state.
    let next = crate::gguf::ops::argmax(&want);
    let a = qwen
        .forward_single_qwen35(next, &mut per_token, n1 + n2)
        .expect("decode");
    let b = qwen
        .forward_single_qwen35(next, &mut batched, n1 + n2)
        .expect("decode");
    assert_eq!(a, b, "{path}: decode after batched prefill differs");
}

/// FALSIFY-4228-004: 0.8B (ratio-1 recurrence).
#[test]
fn falsify_4228_004_prefill_bit_identical_0_8b() {
    check_model(&model_path("Qwen3.5-0.8B-Q4_K_M.gguf"));
}

/// FALSIFY-4228-005: 4B (GQA recurrence, 32 value heads to 16 key heads).
#[test]
fn falsify_4228_005_prefill_bit_identical_4b() {
    check_model(&model_path("Qwen3.5-4B-Q4_K_M.gguf"));
}

/// Prefill throughput, batched vs per-token (#4228 acceptance). Not a gate: run with
/// `APR_QWEN35_PREFILL_MODEL=<gguf> APR_QWEN35_PREFILL_N=850 cargo test --release ... -- --ignored`.
#[test]
#[ignore = "perf probe, needs a real model"]
fn qwen35_prefill_throughput_probe() {
    let path = std::env::var("APR_QWEN35_PREFILL_MODEL")
        .unwrap_or_else(|_| model_path("Qwen3.5-4B-Q4_K_M.gguf"));
    let n: usize = std::env::var("APR_QWEN35_PREFILL_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(850);
    let reps: usize = std::env::var("APR_QWEN35_PREFILL_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let Some(mapped) = load(&path) else { return };
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base model");
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())
        .expect("hybrid layers");
    let toks = prompt(n, base.config.vocab_size);
    let skip_per_token = std::env::var("APR_QWEN35_PREFILL_SKIP_PER_TOKEN").as_deref() == Ok("1");
    for rep in 0..reps {
        let mut s = qwen.new_state(n + 1);
        let t0 = std::time::Instant::now();
        qwen.forward_prefill_qwen35(&toks, &mut s, 0)
            .expect("prefill");
        let batched = t0.elapsed().as_secs_f64();
        let per_token = if skip_per_token {
            f64::NAN
        } else {
            let mut s = qwen.new_state(n + 1);
            let t0 = std::time::Instant::now();
            for (pos, &t) in toks.iter().enumerate() {
                qwen.forward_single_qwen35(t, &mut s, pos).expect("forward");
            }
            t0.elapsed().as_secs_f64()
        };
        eprintln!(
            "rep {rep} n={n}: batched {:.1} tok/s, per-token {:.1} tok/s, speedup {:.2}x",
            n as f64 / batched,
            n as f64 / per_token,
            per_token / batched
        );
    }
}
