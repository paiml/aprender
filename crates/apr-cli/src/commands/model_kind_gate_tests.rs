// FALSIFY-EG2L-007: a generative verb on an embedding GGUF exits 6 (ModelLoadFailed) naming
// `apr embed`. Before EG-1, `apr chat` ran no gate on the CPU path and an empty stdin exited 0.

use super::*;

pub(crate) fn eg1_gguf(arch: &str, causal: Option<bool>) -> Vec<u8> {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    let tensors = vec![GgufTensor {
        name: "token_embd.weight".to_string(),
        shape: vec![4, 8],
        dtype: GgmlType::F32,
        data: vec![0u8; 4 * 8 * 4],
    }];
    let mut metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String(arch.to_string()),
    )];
    if let Some(c) = causal {
        metadata.push((format!("{arch}.attention.causal"), GgufValue::Bool(c)));
    }
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &tensors, &metadata).expect("write GGUF");
    bytes
}

#[test]
fn falsify_eg2l_007_embedding_refused_by_kind() {
    for causal in [None, Some(false)] {
        let err = refuse_non_generative_bytes(&eg1_gguf("gemma-embedding2", causal), "chat")
            .expect_err("an embedding file must not reach a generative verb");
        assert_eq!(err.exit_code_value(), 6, "{err}");
        let msg = err.to_string();
        assert!(msg.contains("Use `apr embed`"), "{msg}");
        assert!(msg.contains("`apr chat`"), "{msg}");
        assert!(!msg.contains("Gemma3"), "{msg}");
    }
}

#[test]
fn falsify_eg2l_007_causal_contradiction_refused() {
    let err = refuse_non_generative_bytes(&eg1_gguf("gemma-embedding2", Some(true)), "serve")
        .expect_err("a causal embedding file is refused");
    assert_eq!(err.exit_code_value(), 6, "{err}");
    assert!(err.to_string().contains("attention.causal = true"), "{err}");
}

#[test]
fn falsify_eg2l_007_generative_and_unparsed_files_pass_through() {
    for arch in ["llama", "gemma2", "gemma-embedding3"] {
        assert!(
            refuse_non_generative_bytes(&eg1_gguf(arch, None), "chat").is_ok(),
            "{arch}"
        );
    }
    let bytes = eg1_gguf("gemma-embedding2", None);
    assert!(refuse_non_generative_bytes(&bytes[..20], "chat").is_ok());
}
