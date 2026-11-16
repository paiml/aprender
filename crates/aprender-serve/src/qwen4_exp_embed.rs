//! `qwen4_exp` gated norm, n-gram hashed embedding and per-layer embedding (PLE) (#7175).
//! STUB (RED): returns zeros.
#![allow(unused_variables, missing_docs)]

/// Activation applied to the gate in [`gated_rms_norm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAct {
    Sigmoid,
    Silu,
}

/// N-gram embedding hyperparameters.
#[derive(Debug, Clone)]
pub struct NgramConfig {
    pub vocab_size: u64,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: u64,
    pub seed: u64,
    pub eos_token_id: i64,
    pub embed_dim: usize,
    pub ple_layer_index: usize,
}

/// PLE parameters (borrowed row-major slices, torch layouts).
pub struct PleParams<'a> {
    pub ngram: &'a NgramConfig,
    pub hidden_size: usize,
    pub hc_count: usize,
    pub kernel: usize,
    pub eps: f32,
    pub table: &'a [f32],
    pub key_proj: &'a [f32],
    pub value_proj: &'a [f32],
    pub norm_key: &'a [f32],
    pub norm_query: &'a [f32],
    pub norm_conv: &'a [f32],
    pub conv_w: &'a [f32],
}

pub fn gated_rms_norm(x: &[f32], gate: &[f32], weight: &[f32], eps: f32, act: GateAct) -> Vec<f32> {
    vec![0.0; x.len()]
}

pub fn ngram_ids(cfg: &NgramConfig, ids: &[i64], ctx: Option<&[i64]>) -> Vec<i64> {
    vec![0; ids.len() * cfg.heads_per_ngram * (cfg.ngram_size - 1)]
}

pub fn ngram_embed(cfg: &NgramConfig, table: &[f32], ids: &[i64]) -> Vec<f32> {
    vec![0.0; ids.len() * cfg.embed_dim]
}

pub fn ple_forward(p: &PleParams<'_>, hidden: &[f32], ids: &[i64]) -> Vec<f32> {
    vec![0.0; hidden.len()]
}
