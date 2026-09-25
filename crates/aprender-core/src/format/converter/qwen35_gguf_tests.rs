//! #4418: the qwen35 GGUF name map, value transforms and Q4_K_M policy.
//!
//! The name-map fixture is the contract with llama.cpp `d1d3c3396`: each GGUF
//! name below was read from `convert_hf_to_gguf.py` output on a seeded tiny
//! Qwen3.5 checkpoint. Deleting or editing any row of `QWEN35_LAYER_NAME_MAP`
//! turns `qwen35_name_map_is_pinned` RED.

use super::*;

const P: &str = "model.language_model.";

/// (HF name, GGUF name) for every exported tensor kind, both layer types.
const PINNED: &[(&str, &str)] = &[
    (
        "model.language_model.embed_tokens.weight",
        "token_embd.weight",
    ),
    ("model.language_model.norm.weight", "output_norm.weight"),
    ("lm_head.weight", "output.weight"),
    (
        "model.language_model.layers.0.input_layernorm.weight",
        "blk.0.attn_norm.weight",
    ),
    (
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "blk.0.post_attention_norm.weight",
    ),
    (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "blk.0.ffn_gate.weight",
    ),
    (
        "model.language_model.layers.0.mlp.up_proj.weight",
        "blk.0.ffn_up.weight",
    ),
    (
        "model.language_model.layers.0.mlp.down_proj.weight",
        "blk.0.ffn_down.weight",
    ),
    (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "blk.0.attn_qkv.weight",
    ),
    (
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "blk.0.attn_gate.weight",
    ),
    (
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "blk.0.ssm_alpha.weight",
    ),
    (
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "blk.0.ssm_beta.weight",
    ),
    (
        "model.language_model.layers.0.linear_attn.A_log",
        "blk.0.ssm_a",
    ),
    (
        "model.language_model.layers.0.linear_attn.dt_bias",
        "blk.0.ssm_dt.bias",
    ),
    (
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "blk.0.ssm_conv1d.weight",
    ),
    (
        "model.language_model.layers.0.linear_attn.norm.weight",
        "blk.0.ssm_norm.weight",
    ),
    (
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "blk.0.ssm_out.weight",
    ),
    (
        "model.language_model.layers.31.self_attn.q_proj.weight",
        "blk.31.attn_q.weight",
    ),
    (
        "model.language_model.layers.31.self_attn.k_proj.weight",
        "blk.31.attn_k.weight",
    ),
    (
        "model.language_model.layers.31.self_attn.v_proj.weight",
        "blk.31.attn_v.weight",
    ),
    (
        "model.language_model.layers.31.self_attn.o_proj.weight",
        "blk.31.attn_output.weight",
    ),
    (
        "model.language_model.layers.31.self_attn.q_norm.weight",
        "blk.31.attn_q_norm.weight",
    ),
    (
        "model.language_model.layers.31.self_attn.k_norm.weight",
        "blk.31.attn_k_norm.weight",
    ),
    // text-only checkpoints drop the `language_model.` level
    (
        "model.layers.3.self_attn.q_proj.weight",
        "blk.3.attn_q.weight",
    ),
];

#[test]
fn qwen35_name_map_is_pinned() {
    for (hf, want) in PINNED {
        assert_eq!(qwen35_gguf_name(hf).as_deref(), Some(*want), "{hf}");
    }
    // Every row of the table is exercised by the fixture (no untested row).
    for (h, _) in QWEN35_LAYER_NAME_MAP {
        assert!(
            PINNED.iter().any(|(hf, _)| hf.ends_with(h)),
            "row {h} has no fixture"
        );
    }
    // No HF name leaks through: every mapped name is a GGUF name.
    for (hf, _) in PINNED {
        let g = qwen35_gguf_name(hf).expect("mapped");
        assert!(
            !g.contains("layers.") && !g.contains("proj") && !g.contains("linear_attn"),
            "{g}"
        );
    }
}

#[test]
fn qwen35_drops_mtp_visual_and_unknown() {
    for hf in [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.language_model.layers.0.linear_attn.mystery.weight",
    ] {
        assert_eq!(qwen35_gguf_name(hf), None, "{hf}");
    }
}

#[test]
fn reorder_v_heads_is_key_major_to_value_major() {
    // nk=2, nv=4 (2 value heads per key head), hd=1: [k0j0 k0j1 k1j0 k1j1] -> [k0j0 k1j0 k0j1 k1j1]
    assert_eq!(
        reorder_v_heads(&[0., 1., 2., 3.], 2, 4, 1, 1),
        vec![0., 2., 1., 3.]
    );
    // hd=2 keeps each head's elements together
    let src: Vec<f32> = (0..8).map(|v| v as f32).collect();
    assert_eq!(
        reorder_v_heads(&src, 2, 4, 2, 1),
        vec![0., 1., 4., 5., 2., 3., 6., 7.]
    );
    // item=2 moves whole rows
    assert_eq!(
        reorder_v_heads(&src, 2, 4, 1, 2),
        vec![0., 1., 4., 5., 2., 3., 6., 7.]
    );
    // identity when there is one value head per key head
    assert_eq!(reorder_v_heads(&src, 4, 4, 2, 1), src);
}

const D: LinearAttnDims = LinearAttnDims {
    nk: 2,
    nv: 4,
    dk: 1,
    dv: 1,
};

fn xf(suffix: &str, data: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    transform_qwen35_tensor(&format!("{P}layers.0.{suffix}"), data, shape, D).expect("transform")
}

#[test]
fn a_log_becomes_negative_exp_then_reordered() {
    let (v, _) = xf("linear_attn.A_log", &[0., 1., 2., 3.], &[4]);
    let e = |x: f32| -x.exp();
    assert_eq!(v, vec![e(0.), e(2.), e(1.), e(3.)]);
}

#[test]
fn zero_centred_norms_get_plus_one_but_ssm_norm_does_not() {
    for s in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
    ] {
        assert_eq!(xf(s, &[0.5], &[1]).0, vec![1.5], "{s}");
    }
    let (v, _) =
        transform_qwen35_tensor(&format!("{P}norm.weight"), &[0.25], &[1], D).expect("final norm");
    assert_eq!(v, vec![1.25]);
    assert_eq!(xf("linear_attn.norm.weight", &[0.5], &[1]).0, vec![0.5]);
}

#[test]
fn conv1d_is_squeezed_and_only_the_v_rows_move() {
    // qk rows = 2*nk*dk = 4 rows stay; the 4 v rows (nv*dv) are regrouped. K = 2.
    let src: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let (v, shape) = xf("linear_attn.conv1d.weight", &src, &[8, 1, 2]);
    assert_eq!(shape, vec![8, 2]);
    assert_eq!(&v[..8], &src[..8]);
    assert_eq!(&v[8..], &[8., 9., 12., 13., 10., 11., 14., 15.]);
}

#[test]
fn out_proj_regroups_columns() {
    let (v, _) = xf(
        "linear_attn.out_proj.weight",
        &[0., 1., 2., 3., 4., 5., 6., 7.],
        &[2, 4],
    );
    assert_eq!(v, vec![0., 2., 1., 3., 4., 6., 5., 7.]);
}

#[test]
fn use_more_bits_matches_published_q4_k_m() {
    // ffn_down Q6_K layers of the published Qwen3.5-4B Q4_K_M (unsloth, 32 layers)
    let q6: Vec<usize> = (0..32).filter(|&i| use_more_bits(i, 32)).collect();
    assert_eq!(
        q6,
        vec![0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31]
    );
    // its 8 attn_v tensors: Q6 Q6 Q6 Q6 Q4 Q4 Q6 Q4
    let v: Vec<GgmlType> = (0..8)
        .map(|r| qwen35_q4km_type("blk.3.attn_v.weight", Some(3), r, 32))
        .collect();
    use GgmlType::{Q4K, Q6K};
    assert_eq!(v, vec![Q6K, Q6K, Q6K, Q6K, Q4K, Q4K, Q6K, Q4K]);
}

#[test]
fn q4_k_m_policy_never_passes_embed_through_f32() {
    assert_eq!(
        qwen35_q4km_type("token_embd.weight", None, 0, 32),
        GgmlType::Q6K
    );
    assert_eq!(
        qwen35_q4km_type("blk.0.attn_qkv.weight", Some(0), 0, 32),
        GgmlType::Q5K
    );
    assert_eq!(
        qwen35_q4km_type("blk.0.ssm_out.weight", Some(0), 0, 32),
        GgmlType::Q5K
    );
    assert_eq!(
        qwen35_q4km_type("blk.7.ffn_down.weight", Some(7), 0, 32),
        GgmlType::Q4K
    );
    assert_eq!(
        qwen35_q4km_type("blk.0.attn_gate.weight", Some(0), 0, 32),
        GgmlType::Q4K
    );
}

#[test]
fn tokenizer_fixup_types_pads_and_ids() {
    let toks = [
        "a",
        "b",
        "<|endoftext|>",
        "<|im_end|>",
        "<think>",
        "<|pad_5|>",
    ];
    let mut md = vec![
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("bpe".into()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(toks.iter().map(|s| s.to_string()).collect()),
        ),
    ];
    fix_qwen35_tokenizer_metadata(&mut md);
    let get = |k: &str| md.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    assert!(matches!(get("tokenizer.ggml.model"), Some(GgufValue::String(s)) if s == "gpt2"));
    assert!(matches!(get("tokenizer.ggml.pre"), Some(GgufValue::String(s)) if s == "qwen35"));
    assert!(
        matches!(get("tokenizer.ggml.token_type"), Some(GgufValue::ArrayInt32(t)) if t == vec![1, 1, 3, 3, 4, 5])
    );
    assert!(matches!(
        get("tokenizer.ggml.eos_token_id"),
        Some(GgufValue::Uint32(3))
    ));
    assert!(matches!(
        get("tokenizer.ggml.padding_token_id"),
        Some(GgufValue::Uint32(2))
    ));
    assert!(
        matches!(get("tokenizer.ggml.tokens"), Some(GgufValue::ArrayString(t)) if t[5] == "[PAD5]")
    );
}

#[test]
fn hparams_refuse_a_config_that_disagrees_with_the_tensors() {
    let mut t = BTreeMap::new();
    t.insert(
        format!("{P}layers.0.linear_attn.A_log"),
        (vec![0.0; 4], vec![4]),
    );
    let h: serde_json::Map<String, serde_json::Value> = serde_json::from_str(
        r#"{"linear_num_key_heads":2,"linear_num_value_heads":8,"linear_key_head_dim":1,
            "linear_value_head_dim":1,"linear_conv_kernel_dim":4,"full_attention_interval":4}"#,
    )
    .expect("json");
    let base = Qwen35Base {
        n_layer: Some(4),
        hidden: Some(8),
        ffn: Some(8),
        n_head: Some(2),
        n_head_kv: Some(1),
        head_dim: Some(4),
        ..Default::default()
    };
    let err = resolve_qwen35_hparams(&base, &h, &t).expect_err("nv=8 vs A_log len 4");
    assert!(err.to_string().contains("A_log"), "{err}");
}

fn hybrid_names() -> Vec<String> {
    let mut v = Vec::new();
    for (i, mixer) in [(0, "linear_attn."), (1, "self_attn.")] {
        for (h, _) in QWEN35_LAYER_NAME_MAP {
            if h.starts_with(mixer)
                || !(h.starts_with("linear_attn.") || h.starts_with("self_attn."))
            {
                v.push(format!("{P}layers.{i}.{h}"));
            }
        }
    }
    v
}

#[test]
fn hybrid_completeness_accepts_alternating_mixers() {
    let names = hybrid_names();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    enforce_hybrid_layer_completeness(&refs, 2).expect("complete hybrid stack");
}

#[test]
fn hybrid_completeness_names_the_missing_tensor() {
    for drop in [
        "layers.0.linear_attn.dt_bias",
        "layers.1.self_attn.k_proj.weight",
        "layers.1.mlp.down_proj.weight",
    ] {
        let names = hybrid_names();
        let refs: Vec<&str> = names
            .iter()
            .map(String::as_str)
            .filter(|n| !n.ends_with(drop))
            .collect();
        let err = enforce_hybrid_layer_completeness(&refs, 2).expect_err(drop);
        let tail = drop
            .split_once('.')
            .and_then(|(_, r)| r.split_once('.'))
            .map_or(drop, |(_, r)| r);
        assert!(err.to_string().contains(tail), "{drop}: {err}");
    }
    // a declared layer that is absent entirely
    let names = hybrid_names();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert!(enforce_hybrid_layer_completeness(&refs, 3).is_err());
}

fn added(id: usize, content: &str, special: bool) -> AddedToken {
    AddedToken {
        id,
        content: content.to_string(),
        special,
    }
}

#[test]
fn added_token_holes_are_filled_but_real_tokens_are_not_overwritten() {
    let toks = ["a", "[PAD1]", "<|im_end|>", "<unk>", "<unk>_1", "<|pad_5|>"]
        .map(String::from)
        .to_vec();
    let mut md = vec![(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(toks),
    )];
    let add = [
        added(1, "<|audio_start|>", true),
        added(2, "clobber", true),
        added(3, "<tts_pad>", true),
        added(4, "<|audio_end|>", true),
        added(5, "<|audio_pad|>", true),
        added(9, "x", true),
    ];
    fill_added_token_holes(&mut md, &add);
    let want = [
        "a",
        "<|audio_start|>",
        "<|im_end|>",
        "<tts_pad>",
        "<|audio_end|>",
        "<|audio_pad|>",
    ];
    assert!(matches!(&md[0].1, GgufValue::ArrayString(t) if t == &want));
}

#[test]
fn added_token_types_follow_the_special_flag() {
    let toks = ["a", "<|im_end|>", "<tts_pad>", "<think>", "<|fim_pad|>"]
        .map(String::from)
        .to_vec();
    let mut md = vec![(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(toks),
    )];
    fix_qwen35_tokenizer_metadata(&mut md);
    // `<|fim_pad|>` is `special: false` in the config and still CONTROL
    let add = [
        added(1, "<|im_end|>", true),
        added(2, "<tts_pad>", true),
        added(3, "<think>", false),
        added(4, "<|fim_pad|>", false),
    ];
    apply_added_token_types(&mut md, &add);
    let ty = md
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.token_type")
        .map(|(_, v)| v.clone());
    assert!(
        matches!(&ty, Some(GgufValue::ArrayInt32(t)) if *t == vec![1, 3, 3, 4, 3]),
        "{ty:?}"
    );
}
