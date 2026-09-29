//! SRV-TIM-001 on the continuous-batch scheduler arm (PARITY-052/054) — the
//! last `/v1/completions` dispatch arm that answered with `timings: null`.
//!
//! The scheduler returned one whole-batch `latency_ms` and no per-request
//! first-token instant, so the arm could only omit the split. It now times
//! each request at its own samples: `PhaseClock` on the per-request path,
//! `BatchPhaseClock` on the lockstep path. Both paths run on CPU here — the
//! "GPU cache" warmup is a CPU dequantisation, and the lockstep loop only
//! reaches the GPU FFN at 32 active prompts.
//!
//! Columns asserted, per path: F1 (a timings block), F2 (its counts equal
//! `usage`), F3 (positive phases that fit inside the client's wall time).

#![cfg(feature = "gpu")]

use std::sync::Arc;

use axum::http::StatusCode;

use super::native_routes_2376::post;
use crate::api::{AppState, BatchConfig, BatchPhaseClock, PhaseTimings};
use crate::gguf::OwnedQuantizedModelCachedSync;

const VOCAB_SIZE: usize = 256;

fn cached_model() -> OwnedQuantizedModelCachedSync {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: VOCAB_SIZE,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    OwnedQuantizedModelCachedSync::new(create_test_quantized_model(&config))
}

fn vocab() -> Vec<String> {
    let mut vocab: Vec<String> = (0..VOCAB_SIZE).map(|i| format!("tok{i}")).collect();
    vocab[0] = "<unk>".to_string();
    vocab[3] = "Hello".to_string();
    vocab
}

/// A cached-model server with the batch scheduler switched on. `gpu_threshold`
/// picks the path: above the batch size it is per-request, at 1 it is lockstep.
fn batch_state(gpu_threshold: usize) -> AppState {
    let state = AppState::with_cached_model_and_vocab(cached_model(), vocab())
        .expect("build cached AppState");
    let model = Arc::clone(state.cached_model().expect("cached model is set"));
    if gpu_threshold <= 1 {
        model
            .warmup_gpu_cache()
            .expect("warm the batch cache (CPU dequant)");
    }
    let config = BatchConfig {
        window_ms: 1,
        min_batch: 1,
        optimal_batch: 1,
        gpu_threshold,
        ..BatchConfig::default()
    };
    let tx = crate::api::spawn_batch_processor(model, config.clone());
    state.with_batch_config(tx, config)
}

/// POST one completion; return the parsed body and the client's wall time.
async fn complete(state: AppState) -> (serde_json::Value, f64) {
    let body = serde_json::json!({
        "model": "t", "prompt": "Hello", "max_tokens": 4, "temperature": 0
    })
    .to_string();
    let t0 = std::time::Instant::now();
    let (status, body) = post(state, "/v1/completions", &body).await;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(status, StatusCode::OK, "/v1/completions failed: {body}");
    (
        serde_json::from_str(&body).expect("parse completion body"),
        wall_ms,
    )
}

/// F1 + F2 + F3 on one response from the batch arm.
fn assert_measured_split(v: &serde_json::Value, wall_ms: f64) {
    let id = v["id"].as_str().expect("id");
    assert!(
        id.starts_with("cmpl-batch"),
        "fixture check: the batch arm must answer this request, got id {id}"
    );
    let t = &v["timings"];
    assert!(t.is_object(), "F1: the batch arm answered with no timings: {v}");
    assert_eq!(t["prompt_n"], v["usage"]["prompt_tokens"], "F2: prompt_n != usage");
    assert_eq!(
        t["predicted_n"], v["usage"]["completion_tokens"],
        "F2: predicted_n != usage"
    );
    let prompt_ms = t["prompt_ms"].as_f64().expect("prompt_ms");
    let predicted_ms = t["predicted_ms"].as_f64().expect("predicted_ms");
    let predicted_n = t["predicted_n"].as_u64().expect("predicted_n");
    assert!(prompt_ms > 0.0, "F3: prompt_ms must be measured, got {prompt_ms}");
    assert!(
        predicted_n == 0 || predicted_ms > 0.0,
        "F3: {predicted_n} tokens decoded in {predicted_ms} ms"
    );
    assert!(
        prompt_ms + predicted_ms <= wall_ms + 1.0,
        "F3: {prompt_ms} + {predicted_ms} ms exceeds the client's wall {wall_ms} ms"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn falsify_srv_tim_009_batch_arm_per_request_path_reports_timings() {
    let (v, wall_ms) = complete(batch_state(usize::MAX)).await;
    assert_measured_split(&v, wall_ms);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn falsify_srv_tim_009_batch_arm_lockstep_path_reports_timings() {
    let (v, wall_ms) = complete(batch_state(1)).await;
    assert_measured_split(&v, wall_ms);
}

/// The lockstep engine reports every sample of every prompt, stop included.
#[test]
fn falsify_srv_tim_009_lockstep_engine_marks_each_prompt() {
    use crate::gguf::QuantizedGenerateConfig;

    let model = cached_model();
    model
        .warmup_gpu_cache()
        .expect("warm the batch cache (CPU dequant)");
    let prompts = vec![vec![3u32], vec![3u32, 4]];
    let config = QuantizedGenerateConfig {
        max_tokens: 3,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let mut samples = [0usize; 2];
    let out = model
        .batch_generate_gpu_observed(&prompts, &config, &mut |idx| samples[idx] += 1)
        .expect("lockstep generation");
    for (idx, seq) in out.iter().enumerate() {
        let generated = seq.len() - prompts[idx].len();
        assert!(
            samples[idx] >= generated && samples[idx] >= 1,
            "prompt {idx}: {generated} tokens generated but {} samples observed",
            samples[idx]
        );
    }
}

/// A request's decode ends at ITS last sample, not when the batch finishes.
#[test]
fn falsify_srv_tim_009_batch_clock_is_per_request() {
    let mut clock = BatchPhaseClock::start(2);
    std::thread::sleep(std::time::Duration::from_millis(2));
    clock.mark(0);
    clock.mark(1);
    std::thread::sleep(std::time::Duration::from_millis(2));
    clock.mark(0);
    // Request 1 stopped at its first sample; request 0 is done. Everything
    // after this is the batch finishing, which neither request decoded.
    std::thread::sleep(std::time::Duration::from_millis(30));

    let early = clock.finish(1);
    let late = clock.finish(0);
    assert!(early.prefill_ms.expect("measured") >= 2.0);
    assert_eq!(early.decode_ms, Some(0.0), "one sample: a measured 0 ms decode");
    let decode = late.decode_ms.expect("measured");
    assert!(
        (2.0..30.0).contains(&decode),
        "decode must end at request 0's last sample, got {decode} ms"
    );
}

/// No third state: a request the engine never sampled for has no split.
#[test]
fn falsify_srv_tim_009_batch_clock_unsampled_is_absent() {
    let mut clock = BatchPhaseClock::start(2);
    clock.mark(0);
    clock.mark(7); // out of range: ignored, not a panic
    assert_eq!(clock.finish(1), PhaseTimings::default());
    assert!(clock.finish(1).to_timings(1, 0).is_none());
    assert!(clock.finish(9).to_timings(1, 0).is_none());
    assert!(clock.finish(0).to_timings(1, 0).is_some());
}
