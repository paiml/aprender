//! #4269: `apr serve` SafeTensors CPU generation goes through
//! `realizar::session::Session`, which leaves a witness entry per turn.

use super::{st_context_budget, st_cpu_generate};
use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use realizar::session::{entries_for, EntryKind};

pub(super) fn tiny_transformer() -> AprTransformer {
    let (hidden_dim, intermediate_dim, vocab_size) = (8, 16, 12);
    let config = AprTransformerConfig {
        architecture: "safetensors-test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };
    AprTransformer {
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers: vec![AprTransformerLayer::empty(hidden_dim, intermediate_dim)],
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
        lm_head_tied: false,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
        config,
    }
}

#[test]
fn st_serve_generate_runs_through_session_and_leaves_a_safetensors_witness() {
    let model = tiny_transformer();
    // A prompt no other test in this binary uses, so the witness is this call's.
    let prompt = [7_u32, 5, 3, 9, 4269 % 12];
    let out = st_cpu_generate(&model, &prompt, 2, 0.0).expect("tiny model generates");
    assert_eq!(&out[..prompt.len()], &prompt, "prompt is echoed first");
    assert!(out.len() > prompt.len(), "at least one token was generated");
    let entries = entries_for(&prompt);
    assert!(
        entries
            .iter()
            .any(|e| e.arch == "safetensors" && e.kind == EntryKind::Generate),
        "serve ST CPU generate must go through Session, got {entries:?}"
    );
}

/// #3718: a prompt that fills the SafeTensors model's window is refused as a 400
/// `context_length_exceeded` before `Session` sees it (it used to surface as a 500).
#[tokio::test]
async fn st_prompt_that_fills_the_window_is_a_400_not_a_500() {
    let model = tiny_transformer(); // context_length 64
    let refusal = st_context_budget(&model, 64, 8).expect_err("64 tokens fill a 64 window");
    assert_eq!(refusal.status(), axum::http::StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(refusal.into_body(), usize::MAX)
        .await
        .expect("body");
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["error"]["code"], "context_length_exceeded");
    assert_eq!(v["error"]["prompt_tokens"], 64);
    assert_eq!(v["error"]["context_length"], 64);
}

/// #3718: near the end of the window the budget is the room left, and that clamped
/// budget is what generation runs with and what `finish_reason` is judged against,
/// so a reply cut by the window reports "length", never "stop".
#[test]
fn st_budget_is_clamped_to_the_room_left_and_generation_stays_inside_it() {
    let model = tiny_transformer();
    assert_eq!(st_context_budget(&model, 10, 8).ok(), Some(8));
    let budget = st_context_budget(&model, 60, 64).expect("60 of 64 fits");
    assert_eq!(budget, 4);
    let prompt: Vec<u32> = (0..60).map(|i| 1 + (i % 11)).collect();
    let out = st_cpu_generate(&model, &prompt, budget, 0.0).expect("generates inside the window");
    let generated = out.len() - prompt.len();
    assert!(generated <= budget, "{generated} > budget {budget}");
    if generated == budget {
        assert_eq!(
            super::super::handlers::finish_reason_for(generated, budget),
            "length"
        );
    }
}
