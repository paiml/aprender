//! #4269 (workstream M of #4263): `StCpuForward` is the only SafeTensors CPU
//! code a verb reaches, and only through a
//! [`Session`](crate::session::Session). This is the witness a verb port must
//! leave — the same contract `qwen35_session_tests.rs` checks for the GGUF
//! hybrid.

use super::StCpuForward;
use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::gguf::QuantizedGenerateConfig;
use crate::session::{entries_for, EntryKind, Session};

/// The smallest SafeTensors-shaped `AprTransformer` that can run a real
/// forward: one layer, tiny dims, non-GQA (`num_kv_heads == num_heads`).
fn tiny_transformer() -> AprTransformer {
    let hidden_dim = 8;
    let num_heads = 2;
    let intermediate_dim = 16;
    let vocab_size = 12;
    let config = AprTransformerConfig {
        architecture: "safetensors-test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads: num_heads,
        vocab_size,
        intermediate_dim,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: None,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
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

/// A turn through `Session<StCpuForward>::generate` leaves a `Generate`
/// witness entry for the `safetensors` arch — the guard
/// `tests_engine_identity` checks for every verb × arch (session.rs docs).
/// A verb that called `AprTransformer::generate_with_cache` directly, or ran
/// its own loop, leaves no entry here.
#[test]
fn safetensors_cpu_session_generate_leaves_a_witness_entry() {
    let model = tiny_transformer();
    let forward = StCpuForward::new(&model);
    let mut session = Session::new(forward);
    let prompt = vec![1_u32, 2, 3];
    let config = QuantizedGenerateConfig {
        max_tokens: 2,
        temperature: 0.0,
        top_k: 0,
        top_p: 0.9,
        seed: 42,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        stop_tokens: vec![0],
        trace: false,
        logprobs: false,
        cancel: crate::generate::CancelToken::never(),
    };
    let turn = session
        .generate(&prompt, &config, &mut |_tok| true)
        .expect("a tiny CPU forward must generate");
    assert!(turn.tokens.len() >= prompt.len());
    assert!(!turn.used_gpu, "StCpuForward never runs on the GPU");

    let entries = entries_for(&prompt);
    assert!(
        entries.iter().any(|e| e.arch == "safetensors" && e.kind == EntryKind::Generate),
        "Session::generate must leave a safetensors witness entry, got {entries:?}"
    );
}

/// `forward` before `reserve` is a caller bug (`Session` always reserves
/// first) — it must fail loudly, not panic or silently return empty logits.
#[test]
fn forward_before_reserve_is_an_error_not_a_panic() {
    let model = tiny_transformer();
    let mut forward = StCpuForward::new(&model);
    let err = ArchForwardOnly::forward(&mut forward, &[1, 2], 0);
    assert!(err.is_err(), "forward before reserve must error, not panic");
}

/// Local alias so the test above can name the trait method without pulling
/// the whole `ArchForward` surface into this file's imports twice.
use crate::session::ArchForward as ArchForwardOnly;
