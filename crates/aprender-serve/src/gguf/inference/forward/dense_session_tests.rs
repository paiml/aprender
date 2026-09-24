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

fn session_tokens(model: &Arc<OwnedQuantizedModel>, prompt: &[u32], cfg: &QuantizedGenerateConfig) -> Vec<u32> {
    let mut session = DenseSession::new(DenseForward::cpu(Arc::clone(model)));
    session.generate(prompt, cfg, &mut |_| true).expect("dense session generates").tokens
}

#[test]
fn greedy_turn_matches_generate_with_cache() {
    let model = model();
    let prompt = [1, 7, 13, 21];
    let cfg = gen(12);
    let want = model.generate_with_cache(&prompt, &cfg).expect("reference loop");
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
    let want = model.generate_with_cache(&prompt, &cfg).expect("reference loop");
    assert_eq!(session_tokens(&model, &prompt, &cfg), want);
}

/// The dense loops applied the repetition penalty before every choice; the
/// engine must too, or a `--repeat-penalty` run changes answer on the move.
#[test]
fn repeat_penalty_turn_matches_generate_with_cache() {
    let model = model();
    let prompt = [2, 4, 6];
    let plain = gen(16);
    let penalised = QuantizedGenerateConfig {
        repeat_penalty: 5.0,
        repeat_last_n: 64,
        ..gen(16)
    };
    let want = model.generate_with_cache(&prompt, &penalised).expect("reference loop");
    // The fixture must be one where the penalty changes the answer, or this
    // test cannot see the penalty go missing.
    assert_ne!(
        want,
        model.generate_with_cache(&prompt, &plain).expect("reference loop"),
        "fixture: the penalty changes nothing on this model"
    );
    assert_eq!(session_tokens(&model, &prompt, &penalised), want);
}

/// A second turn that extends the first prefills only its new suffix and
/// answers as a fresh session would.
#[test]
fn extending_turn_reuses_the_cache() {
    let model = model();
    let cfg = gen(6);
    let mut session = DenseSession::new(DenseForward::cpu(Arc::clone(&model)));
    let first = session.generate(&[1, 2, 3], &cfg, &mut |_| true).expect("turn 1");
    let mut prompt = first.tokens.clone();
    prompt.extend([9, 10]);
    let second = session.generate(&prompt, &cfg, &mut |_| true).expect("turn 2");
    assert!(second.reused > 0, "turn 2 re-prefilled the whole conversation");
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
