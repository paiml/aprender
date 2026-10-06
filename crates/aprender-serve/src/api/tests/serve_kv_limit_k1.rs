//! K1: `apr serve` sizes the device KV from the model's context, so a prompt at
//! the limit is answered whole and one past it is refused with a 400.
//!
//! Before K1 serve fixed the KV at 4096 whatever the model declared, so on a
//! model whose context is larger a prompt past 4096 tokens overflowed the cache
//! and came back as a 500. This builds an 8192-context model (above the old
//! 4096, so a return to the fixed size fails the KV assertion), serves it
//! through the real router and batch scheduler on the device, and checks three
//! things. At the limit it answers 200 with every requested token. At the limit
//! and one past it, it answers 400 and says the prompt was refused whole.
//!
//! The model is synthetic (no file), so the test runs on any CUDA runner.
//! Without a device it panics: a run that could not reach the device measured
//! nothing, so it is a failure, never a pass (L25). It is not `#[ignore]`d,
//! because `cuda-unit` runs the cuda-only lib tests without `--ignored`.

use crate::api::{create_router, AppState};
use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{
    ArchConstraints, GGUFConfig, OwnedQKVWeights, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    OwnedQuantizedTensor,
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

/// Above serve's pre-K1 fixed 4096, so a fixed size cannot pass.
const CONTEXT: usize = 8192;
const VOCAB: usize = 256;
/// Generation per request. At the limit, prompt + this == `CONTEXT`.
const MAX_TOKENS: usize = 4;

fn config() -> GGUFConfig {
    GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: VOCAB,
        rope_theta: 10000.0,
        context_length: CONTEXT,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

/// One token per `a`: the only single-character entries are the letters, and the
/// greedy matcher has no longer match for a run of `a`.
fn vocab() -> Vec<String> {
    let mut v: Vec<String> = vec!["<unk>".to_string()];
    v.extend(('a'..='z').map(String::from));
    v.extend((v.len()..VOCAB).map(|i| format!("<t{i}>")));
    v
}

/// Q4_K weights small and centred on zero (d = 2^-10, dmin = 2^-7, pseudo-random
/// scales and quants). The shared `create_q4k_test_data` pattern is all-positive
/// at d = 1.0, which saturates a forward pass until the device parity gate
/// (rightly) refuses it; this one keeps activations near unit scale.
fn q4k(in_dim: usize, out_dim: usize, seed: u64) -> OwnedQuantizedTensor {
    let blocks = in_dim.div_ceil(256);
    let mut data = vec![0u8; out_dim * blocks * 144];
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for block in data.chunks_exact_mut(144) {
        block[0..2].copy_from_slice(&0x1400_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x2000_u16.to_le_bytes());
        for b in &mut block[4..] {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *b = (x >> 56) as u8;
        }
    }
    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: 12,
    }
}

/// A llama-shaped model with separate Q/K/V, as a llama GGUF loads: the
/// GPU-resident path serve generates on takes nothing else
/// (`supports_gpu_resident`).
fn model() -> OwnedQuantizedModel {
    let c = config();
    let (h, kv, ffn) = (
        c.hidden_dim,
        c.num_kv_heads * (c.hidden_dim / c.num_heads),
        c.intermediate_dim,
    );
    let mut model = create_test_model_with_config(&c);
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let s = 10 * i as u64;
        layer.qkv_weight = OwnedQKVWeights::Separate {
            q: q4k(h, h, s + 1),
            k: q4k(h, kv, s + 2),
            v: q4k(h, kv, s + 3),
        };
        layer.attn_output_weight = q4k(h, h, s + 4);
        layer.ffn_up_weight = q4k(h, ffn, s + 5);
        layer.ffn_down_weight = q4k(ffn, h, s + 6);
        layer.ffn_gate_weight = Some(q4k(h, ffn, s + 7));
        // llama normalises before the FFN; the shared fixture leaves it out.
        layer.ffn_norm_weight = Some(vec![1.0; h]);
    }
    model.lm_head_weight = q4k(h, VOCAB, 99);
    model
}

/// The router `apr serve` builds for a CUDA GGUF: the model sized by
/// `for_serving`, behind the continuous-batching scheduler. Panics without a
/// CUDA device: an unreachable device is not measured, never a pass (L25).
fn served() -> (axum::Router, usize) {
    let cuda = match OwnedQuantizedModelCuda::for_serving(model(), 0) {
        Ok(m) => m,
        Err(e) => panic!("K1 not measured: CUDA model init failed: {}", e.error),
    };
    let kv = cuda.executor().max_kv_len();
    let state = AppState::with_cuda_model_and_vocab(cuda, vocab()).expect("state");
    let model = state.cuda_model().expect("the CUDA model").clone();
    let tx = crate::api::cuda_batch_scheduler::spawn_cuda_batch_scheduler(
        model,
        crate::api::cuda_batch_scheduler::CudaBatchConfig::default(),
        crate::api::InFlightCounter::new(),
    );
    (create_router(state.with_cuda_batch_tx(tx)), kv)
}

async fn complete(app: &axum::Router, prompt_tokens: usize) -> (StatusCode, serde_json::Value) {
    let body = serde_json::json!({
        "model": "k1",
        "prompt": "a".repeat(prompt_tokens),
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::json!({ "raw": String::from_utf8_lossy(&bytes) }));
    (status, json)
}

#[tokio::test(flavor = "multi_thread")]
async fn k1_serve_kv_is_the_model_context_and_the_limit_is_a_clean_400() {
    let (app, kv) = served();
    assert_eq!(
        kv, CONTEXT,
        "K1: the serving KV is the model's context, not a fixed size"
    );

    // At the limit: prompt + generation fill the context exactly. Answered
    // whole: 200, every requested token, the prompt counted as sent.
    let at_limit = CONTEXT - MAX_TOKENS;
    let (status, json) = complete(&app, at_limit).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "at the limit ({at_limit} + {MAX_TOKENS}): {json}"
    );
    assert_eq!(json["usage"]["prompt_tokens"], at_limit, "{json}");
    assert_eq!(
        json["usage"]["completion_tokens"], MAX_TOKENS,
        "full output: {json}"
    );
    assert!(json["choices"][0]["text"].is_string(), "{json}");
    eprintln!(
        "K1 at-limit: 200, prompt {at_limit}, completion {MAX_TOKENS}, finish {}",
        json["choices"][0]["finish_reason"]
    );

    // At and past the limit: refused before it reaches the device, as a 400
    // that says so. Not a 500, and not a truncated 200.
    for prompt in [CONTEXT, CONTEXT + 1] {
        let (status, json) = complete(&app, prompt).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "prompt {prompt}: {json}");
        let text = json.to_string();
        assert!(text.contains("context exceeds"), "prompt {prompt}: {text}");
        assert!(text.contains("refused whole"), "prompt {prompt}: {text}");
        eprintln!("K1 over-limit: prompt {prompt} → 400 {text}");
    }
}
