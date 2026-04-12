// PMAT-546: Architecture inference from tensor naming patterns
// Tests for RosettaStone::infer_architecture_from_tensors()

use super::*;

fn make_tensors(names: &[&str]) -> Vec<TensorInfo> {
    names
        .iter()
        .map(|n| TensorInfo {
            name: n.to_string(),
            dtype: "F32".to_string(),
            shape: vec![1],
            size_bytes: 4,
            stats: None,
        })
        .collect()
}

#[test]
fn infer_mamba_from_mixer_tensors() {
    let tensors = make_tensors(&[
        "backbone.embeddings.weight",
        "backbone.layers.0.mixer.in_proj.weight",
        "backbone.layers.0.mixer.out_proj.weight",
        "backbone.layers.0.norm.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("mamba".to_string())
    );
}

#[test]
fn infer_rwkv_from_blocks_tensors() {
    let tensors = make_tensors(&[
        "rwkv.blocks.0.att.key.weight",
        "rwkv.blocks.0.att.value.weight",
        "rwkv.blocks.0.ffn.key.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("rwkv".to_string())
    );
}

#[test]
fn infer_gpt_neox_from_gpt_neox_prefix() {
    let tensors = make_tensors(&[
        "gpt_neox.embed_in.weight",
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("gpt_neox".to_string())
    );
}

#[test]
fn infer_opt_from_model_decoder_prefix() {
    let tensors = make_tensors(&[
        "model.decoder.embed_tokens.weight",
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.0.fc1.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("opt".to_string())
    );
}

#[test]
fn infer_bert_from_bert_prefix() {
    let tensors = make_tensors(&[
        "bert.embeddings.word_embeddings.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("bert".to_string())
    );
}

#[test]
fn infer_gpt2_from_c_attn() {
    let tensors = make_tensors(&[
        "transformer.wte.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.mlp.c_fc.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("gpt2".to_string())
    );
}

#[test]
fn infer_qwen2_from_q_proj_with_bias() {
    let tensors = make_tensors(&[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.mlp.gate_proj.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("qwen2".to_string())
    );
}

#[test]
fn infer_llama_from_q_proj_gate_no_bias() {
    let tensors = make_tensors(&[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("llama".to_string())
    );
}

#[test]
fn infer_none_from_empty() {
    let tensors: Vec<TensorInfo> = vec![];
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        None
    );
}

#[test]
fn infer_transformer_fallback() {
    let tensors = make_tensors(&[
        "encoder.layers.0.self_attn.in_proj_weight",
        "encoder.layers.0.linear1.weight",
    ]);
    assert_eq!(
        RosettaStone::infer_architecture_from_tensors(&tensors),
        Some("transformer".to_string())
    );
}
