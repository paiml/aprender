//! #4269: `apr serve` SafeTensors CPU generation goes through
//! `realizar::session::Session`, which leaves a witness entry per turn.

use super::st_cpu_generate;
use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use realizar::session::{entries_for, EntryKind};

fn tiny_transformer() -> AprTransformer {
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
    let out = st_cpu_generate(&model, &prompt, 2, 0.0, None, false).expect("tiny model generates");
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

/// #2853: a model whose greedy pick is always token 0, the SafeTensors stop token.
fn eos_first_transformer() -> AprTransformer {
    let mut model = tiny_transformer();
    let mut bias = vec![0.0; model.config.vocab_size];
    bias[0] = 10.0;
    model.lm_head_bias = Some(bias);
    model
}

#[test]
fn st_generate_stops_at_eos_by_default_2853() {
    let prompt = [3_u32, 1, 4, 1, 5];
    let out =
        st_cpu_generate(&eos_first_transformer(), &prompt, 8, 0.0, None, false).expect("generates");
    assert!(
        out.len() - prompt.len() < 8,
        "the RED control: without ignore_eos the EOS-first model must stop early, got {out:?}"
    );
}

#[test]
fn st_generate_ignore_eos_runs_to_max_tokens_2853() {
    let prompt = [3_u32, 1, 4, 1, 5];
    let out =
        st_cpu_generate(&eos_first_transformer(), &prompt, 8, 0.0, None, true).expect("generates");
    assert_eq!(
        out.len() - prompt.len(),
        8,
        "ignore_eos must generate exactly max_tokens past EOS, got {out:?}"
    );
}

#[test]
fn st_generate_seed_is_honoured_2853() {
    // Uniform logits at temperature 1.0: the draw is the seed's alone.
    let model = tiny_transformer();
    let prompt = [2_u32, 7, 1, 8, 2];
    let run =
        |seed| st_cpu_generate(&model, &prompt, 16, 1.0, Some(seed), true).expect("generates");
    assert_eq!(run(11), run(11), "one seed reproduces its draw");
    assert!(
        (12..20).any(|s| run(s) != run(11)),
        "a request seed must change the draw, or it is still being dropped"
    );
}
