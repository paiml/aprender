//! #4268: the dense CPU forward behind the engine answers what the loop it
//! replaces (`generate_with_cache`) answered, token for token.

use std::sync::Arc;

use super::*;
use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{GGUFConfig, QuantizedGenerateConfig};
use crate::session::ArchForward;

fn config() -> GGUFConfig {
    GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        rope_theta: 10000.0,
        context_length: 64,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

fn model() -> Arc<OwnedQuantizedModel> {
    Arc::new(create_test_model_with_config(&config()))
}

fn gen(max_tokens: usize) -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    }
}

fn session_tokens(
    model: &Arc<OwnedQuantizedModel>,
    prompt: &[u32],
    cfg: &QuantizedGenerateConfig,
) -> Vec<u32> {
    let mut session = DenseSession::new(DenseForward::cpu(Arc::clone(model)));
    session
        .generate(prompt, cfg, &mut |_| true)
        .expect("dense session generates")
        .tokens
}

#[test]
fn greedy_turn_matches_generate_with_cache() {
    let model = model();
    let prompt = [1, 7, 13, 21];
    let cfg = gen(12);
    let want = model
        .generate_with_cache(&prompt, &cfg)
        .expect("reference loop");
    assert_eq!(session_tokens(&model, &prompt, &cfg), want);
}

#[test]
fn sampled_turn_matches_generate_with_cache() {
    let model = model();
    let prompt = [3, 5, 8];
    let cfg = QuantizedGenerateConfig {
        max_tokens: 12,
        temperature: 1.5,
        top_k: 50,
        top_p: 0.95,
        seed: 42,
        ..Default::default()
    };
    let want = model
        .generate_with_cache(&prompt, &cfg)
        .expect("reference loop");
    assert_eq!(session_tokens(&model, &prompt, &cfg), want);
}

/// Logits that favour token 0, then token 1: argmax picks 0 every time unless
/// the repetition penalty pushes 0 below 1 once it is in the context.
struct Favours0Then1;

impl ArchForward for Favours0Then1 {
    fn arch(&self) -> &'static str {
        "favours"
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn context_length(&self) -> usize {
        64
    }
    fn batched_prefills(&self) -> usize {
        0
    }
    fn notices(&self) -> &[String] {
        &[]
    }
    fn reserve(&mut self, _positions: usize) -> crate::error::Result<bool> {
        Ok(false)
    }
    fn forward(&mut self, _tokens: &[u32], _start: usize) -> crate::error::Result<Vec<f32>> {
        Ok(vec![1.0, 0.9, 0.0, 0.0])
    }
}

/// The dense loops applied the repetition penalty before every choice; the
/// engine must too, or a `--repeat-penalty` run changes answer on the move.
#[test]
fn engine_applies_the_repeat_penalty() {
    let mut session = crate::session::Session::new(Favours0Then1);
    let plain = session
        .generate(&[3], &gen(2), &mut |_| true)
        .expect("plain turn");
    assert_eq!(plain.tokens, [3, 0, 0]);
    let penalised = QuantizedGenerateConfig {
        repeat_penalty: 2.0,
        repeat_last_n: 64,
        ..gen(2)
    };
    let turn = session
        .generate(&[3], &penalised, &mut |_| true)
        .expect("penalised turn");
    assert_eq!(turn.tokens, [3, 0, 1], "the penalty was not applied");
}

/// A second turn that extends the first prefills only its new suffix and
/// answers as a fresh session would.
#[test]
fn extending_turn_reuses_the_cache() {
    let model = model();
    let cfg = gen(6);
    let mut session = DenseSession::new(DenseForward::cpu(Arc::clone(&model)));
    let first = session
        .generate(&[1, 2, 3], &cfg, &mut |_| true)
        .expect("turn 1");
    let mut prompt = first.tokens.clone();
    prompt.extend([9, 10]);
    let second = session
        .generate(&prompt, &cfg, &mut |_| true)
        .expect("turn 2");
    assert!(
        second.reused > 0,
        "turn 2 re-prefilled the whole conversation"
    );
    assert_eq!(second.tokens, session_tokens(&model, &prompt, &cfg));
}

#[test]
fn cpu_forward_reports_its_route() {
    let forward = DenseForward::cpu(model());
    assert!(!forward.on_gpu());
    assert_eq!(forward.arch(), "llama");
    assert_eq!(forward.context_length(), 64);
    assert_eq!(forward.notices(), ["Backend: CPU".to_string()]);
}

#[test]
fn prompt_the_context_cannot_hold_is_refused() {
    let mut session = DenseSession::new(DenseForward::cpu(model()));
    let prompt: Vec<u32> = (0..64).collect();
    assert!(session.generate(&prompt, &gen(4), &mut |_| true).is_err());
}

/// The engine keeps the stop token that ended a turn; the loops it replaces
/// never did. `dense_turn` answers what `generate_with_cache` answered.
#[test]
fn dense_turn_drops_the_stop_token_like_generate_with_cache() {
    let model = model();
    let prompt = [2, 4, 6];
    let first = model
        .generate_with_cache(&prompt, &gen(1))
        .expect("reference loop")[prompt.len()];
    let cfg = QuantizedGenerateConfig {
        stop_tokens: vec![first],
        ..gen(8)
    };
    let want = model
        .generate_with_cache(&prompt, &cfg)
        .expect("reference loop");
    assert_eq!(want, prompt, "the fixture's first token must be the stop");
    let mut session = DenseSession::new(DenseForward::cpu(Arc::clone(&model)));
    let (tokens, used_gpu) = dense_turn(&mut session, &prompt, &cfg).expect("dense turn");
    assert_eq!(tokens, want);
    assert!(!used_gpu);
}

#[test]
fn dense_turn_keeps_the_old_context_error() {
    let mut session = DenseSession::new(DenseForward::cpu(model()));
    let prompt: Vec<u32> = (0..65).map(|t| t % 100).collect();
    let err = dense_turn(&mut session, &prompt, &gen(4)).expect_err("over the context");
    assert!(
        matches!(
            err,
            crate::error::RealizarError::ContextLimitExceeded { .. }
        ),
        "{err:?}"
    );
}
