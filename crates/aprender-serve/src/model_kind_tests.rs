//! Falsifiers for `contracts/embeddinggemma2-load-v1.yaml` (FALSIFY-EG2L-001..004).

use super::*;

// FALSIFY-EG2L-001: the kind comes from an exact architecture match, never from the `gemma` prefix.
#[test]
fn falsify_eg2l_001_exact_arch_match_only() {
    assert!(is_embedding_gemma2("gemma-embedding2"));
    assert!(is_embedding_gemma2("Gemma-Embedding2"));
    for not_it in [
        "gemma",
        "gemma2",
        "gemma3",
        "gemma-embedding",
        "gemma-embedding3",
        "gemma-embedding2x",
        "embeddinggemma2",
        "",
    ] {
        assert!(
            !is_embedding_gemma2(not_it),
            "{not_it} must not be admitted as EmbeddingGemma 2"
        );
        assert_eq!(
            model_kind(not_it, Some(false)),
            Ok(ModelKind::Generative),
            "{not_it}"
        );
    }
}

#[test]
fn falsify_eg2l_001_kind_and_causal_flag() {
    assert_eq!(
        model_kind("gemma-embedding2", Some(false)),
        Ok(ModelKind::Embedding)
    );
    assert_eq!(
        model_kind("gemma-embedding2", None),
        Ok(ModelKind::Embedding)
    );
    let err = model_kind("gemma-embedding2", Some(true))
        .expect_err("causal embedding file must be refused");
    assert!(err.contains("attention.causal = true"), "{err}");
}

// FALSIFY-EG2L-002: per-layer KV heads are read from the array, not collapsed to num_heads.
#[test]
fn falsify_eg2l_002_kv_heads_array_read_per_layer() {
    let layers: Vec<GGUFValue> = (0..24u32)
        .map(|i| GGUFValue::UInt32(if i % 6 == 5 { 1 } else { 2 }))
        .collect();
    let got = per_layer_kv_heads(&GGUFValue::Array(layers), 24).expect("24-long array");
    assert_eq!(got.len(), 24);
    assert_eq!(got.iter().filter(|&&k| k == 1).count(), 4);
    assert_eq!(got[5], 1);
    assert_eq!(got[0], 2);
}

#[test]
fn falsify_eg2l_002_kv_heads_follow_the_file_not_a_pattern() {
    // Same counts, different positions: the reader must follow the file.
    let vals = [3u32, 1, 4, 1, 5];
    let arr = GGUFValue::Array(vals.iter().map(|&v| GGUFValue::Int32(v as i32)).collect());
    assert_eq!(per_layer_kv_heads(&arr, 5), Ok(vec![3, 1, 4, 1, 5]));
}

#[test]
fn falsify_eg2l_002_kv_heads_scalar_broadcasts() {
    assert_eq!(
        per_layer_kv_heads(&GGUFValue::UInt32(8), 3),
        Ok(vec![8, 8, 8])
    );
    assert_eq!(per_layer_kv_heads(&GGUFValue::UInt64(2), 2), Ok(vec![2, 2]));
    assert_eq!(per_layer_kv_heads(&GGUFValue::UInt8(1), 1), Ok(vec![1]));
    assert_eq!(per_layer_kv_heads(&GGUFValue::UInt16(4), 1), Ok(vec![4]));
}

#[test]
fn falsify_eg2l_002_kv_heads_refusals() {
    let short = GGUFValue::Array(vec![GGUFValue::UInt32(1); 23]);
    let err = per_layer_kv_heads(&short, 24).expect_err("length mismatch");
    assert!(err.contains("23 entries for 24 blocks"), "{err}");
    assert!(
        per_layer_kv_heads(&GGUFValue::UInt32(0), 2).is_err(),
        "zero heads"
    );
    assert!(
        per_layer_kv_heads(&GGUFValue::Int32(-1), 2).is_err(),
        "negative heads"
    );
    assert!(
        per_layer_kv_heads(&GGUFValue::Float32(2.0), 2).is_err(),
        "float heads"
    );
    assert!(
        per_layer_kv_heads(&GGUFValue::String("2".into()), 2).is_err(),
        "string heads"
    );
    let bad_entry = GGUFValue::Array(vec![GGUFValue::UInt32(1), GGUFValue::Bool(true)]);
    assert!(
        per_layer_kv_heads(&bad_entry, 2).is_err(),
        "non-integer entry"
    );
}

// FALSIFY-EG2L-003: layer attention kind from attn_q widths; swapping the shapes swaps the kinds.
#[test]
fn falsify_eg2l_003_layer_kind_from_shapes() {
    let widths: Vec<usize> = (0..24)
        .map(|i| {
            if [5, 11, 17, 23].contains(&i) {
                2048
            } else {
                1024
            }
        })
        .collect();
    let kinds = layer_attention_kinds(&widths).expect("two widths");
    let globals: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == LayerAttention::Global)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(globals, vec![5, 11, 17, 23]);
}

#[test]
fn falsify_eg2l_003_swapped_shapes_swap_kinds() {
    let widths = [2048, 1024, 1024, 2048, 1024];
    let kinds = layer_attention_kinds(&widths).expect("two widths");
    use LayerAttention::{Global, Local};
    assert_eq!(kinds, vec![Global, Local, Local, Global, Local]);
}

#[test]
fn falsify_eg2l_003_layer_kind_refusals() {
    assert!(layer_attention_kinds(&[]).is_err(), "no layers");
    let err = layer_attention_kinds(&[1024, 1024]).expect_err("single width");
    assert!(err.contains("no local/global split"), "{err}");
    assert!(layer_attention_kinds(&[0, 1024]).is_err(), "zero width");
}
