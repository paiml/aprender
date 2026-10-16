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

fn write_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    file.write_all(bytes).expect("write");
    file
}

// EG-5 (I-3): `run`, `code` and `showcase` refuse an embedding file by path, by kind, naming
// `apr embed` and the verb that refused.
#[test]
fn falsify_eg5_path_refuses_embedding_for_each_verb() {
    let file = write_temp(&eg1_gguf("gemma-embedding2", Some(false)));
    for verb in ["run", "code", "showcase"] {
        let err = refuse_non_generative_path(file.path(), verb)
            .expect_err("an embedding file must not reach a generative verb");
        assert_eq!(err.exit_code_value(), 6, "{err}");
        let msg = err.to_string();
        assert!(msg.contains("Use `apr embed`"), "{msg}");
        assert!(msg.contains(&format!("`apr {verb}`")), "{msg}");
    }
}

#[test]
fn falsify_eg5_path_passes_generative_missing_and_non_gguf_files() {
    let generative = write_temp(&eg1_gguf("llama", None));
    assert!(refuse_non_generative_path(generative.path(), "run").is_ok());
    let apr = write_temp(b"APR2 not a gguf file");
    assert!(refuse_non_generative_path(apr.path(), "run").is_ok());
    let short = write_temp(b"GG");
    assert!(refuse_non_generative_path(short.path(), "run").is_ok());
    let missing = std::path::Path::new("/nonexistent/eg5/model.gguf");
    assert!(refuse_non_generative_path(missing, "run").is_ok());
}

// The header is found when the metadata outgrows the first 16 MiB prefix (the doubling loop).
#[test]
fn falsify_eg5_path_reads_a_header_past_the_first_prefix() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    let tensors = vec![GgufTensor {
        name: "token_embd.weight".to_string(),
        shape: vec![4, 8],
        dtype: GgmlType::F32,
        data: vec![0u8; 4 * 8 * 4],
    }];
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("gemma-embedding2".to_string()),
        ),
        (
            "general.description".to_string(),
            GgufValue::String("x".repeat(20 << 20)),
        ),
    ];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &tensors, &metadata).expect("write GGUF");
    assert!(bytes.len() > 16 << 20);
    let file = write_temp(&bytes);
    let err = refuse_non_generative_path(file.path(), "run").expect_err("found past 16 MiB");
    assert!(err.to_string().contains("Use `apr embed`"), "{err}");
}

// `apr rerank` serves neither kind: a GGUF file is refused naming the verb for its kind.
#[test]
fn falsify_eg5_rerank_refuses_gguf_by_kind() {
    let embedding = write_temp(&eg1_gguf("gemma-embedding2", None));
    let err = refuse_gguf_path(embedding.path(), "rerank").expect_err("embedding refused");
    assert_eq!(err.exit_code_value(), 6, "{err}");
    let msg = err.to_string();
    assert!(msg.contains("ModelKind::Embedding"), "{msg}");
    assert!(msg.contains("Use `apr embed`"), "{msg}");
    assert!(msg.contains("`apr rerank`"), "{msg}");

    let generative = write_temp(&eg1_gguf("llama", None));
    let err = refuse_gguf_path(generative.path(), "rerank").expect_err("generative refused");
    assert!(err.to_string().contains("Use `apr run`"), "{err}");

    let apr = write_temp(b"APR2 not a gguf file");
    assert!(refuse_gguf_path(apr.path(), "rerank").is_ok());
}
