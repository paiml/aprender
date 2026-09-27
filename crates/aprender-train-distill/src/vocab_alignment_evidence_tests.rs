//! FT-VOCAB-ALIGN-005/006: the Qwen3.5 27B↔4B cell of apr-distill-teacher-vocab-alignment-v1.
//!
//! The contract truncates teacher logits to the student's vocab and ASSUMES the first N
//! tokens of both models are the same tokens. `scripts/distill_vocab_identity.py` measured
//! that on the pinned GGUFs (evidence/crux/hf-sources.yaml); these tests bind the committed
//! measurement to the family contract, so a re-measurement that disagrees, or a family
//! contract whose vocab moves, is RED here rather than a silent change of what the
//! truncation aligns.

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn evidence(name: &str) -> serde_json::Value {
    let p = repo()
        .join("evidence/distill-qwen35-27b-4b-vocab-alignment")
        .join(name);
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn family_vocab(variant: &str) -> u64 {
    let p = repo().join("contracts/model-families/qwen3_5.yaml");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let y: serde_yaml::Value = serde_yaml::from_str(&s).expect("qwen3_5.yaml parses");
    y["size_variants"][variant]["vocab_size"]
        .as_u64()
        .unwrap_or_else(|| panic!("qwen3_5.yaml size_variants.{variant}.vocab_size"))
}

fn list<'a>(rep: &'a serde_json::Value, key: &str) -> &'a serde_json::Value {
    &rep["lists"][key]
}

/// FT-VOCAB-ALIGN-005: 27B teacher and 4B student carry the SAME token list, merges and
/// token types, at the family contract's vocab, so truncation is the identity.
#[test]
fn qwen35_27b_and_4b_share_one_vocab_so_truncation_is_the_identity() {
    let rep = evidence("identity.json");
    assert_eq!(rep["teacher"]["file"], "Qwen3.5-27B-Q4_K_M.gguf");
    assert_eq!(rep["student"]["file"], "Qwen3.5-4B-Q4_K_M.gguf");
    let vocab = family_vocab("27b");
    for key in [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
    ] {
        let l = list(&rep, key);
        assert_eq!(l["identical"], true, "{key} differs between 27B and 4B");
        assert_eq!(l["teacher_sha256"], l["student_sha256"], "{key} sha256");
        assert!(l["first_diff"].is_null(), "{key} first_diff");
    }
    let tok = list(&rep, "tokenizer.ggml.tokens");
    assert_eq!(tok["teacher_len"].as_u64(), Some(vocab));
    assert_eq!(tok["student_len"].as_u64(), Some(vocab));
    assert_eq!(rep["student_vocab_is_teacher_prefix"], true);
}

/// FT-VOCAB-ALIGN-006: vocab SIZE alone does not decide alignment. A Qwen2.5 student is
/// smaller than the Qwen3.5 teacher, so the size check would truncate and train, yet the
/// token lists part at index 280: the logits would be aligned across different tokens.
#[test]
fn a_smaller_student_vocab_is_not_a_shared_prefix() {
    let rep = evidence("identity-negative-qwen25-student.json");
    let tok = list(&rep, "tokenizer.ggml.tokens");
    let (t, s) = (
        tok["teacher_len"].as_u64().expect("teacher_len"),
        tok["student_len"].as_u64().expect("student_len"),
    );
    assert!(
        s < t,
        "negative control must pass the size-only check ({s} < {t})"
    );
    assert_eq!(tok["first_diff"].as_u64(), Some(280));
    assert_eq!(rep["student_vocab_is_teacher_prefix"], false);
}
