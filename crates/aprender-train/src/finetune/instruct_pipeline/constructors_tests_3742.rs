//! #3742: an `.apr`'s embedded tokenizer loads canonically, or the fine-tune stops with a
//! named error. It never proceeds as if the file had no tokenizer.

use super::InstructPipeline;
use aprender::serialization::apr::AprWriter;
use std::path::{Path, PathBuf};

/// The GPT-2 glyph of each byte (so `Ġ` and `Ċ` are tokens), then `extra`.
fn byte_level_vocab(extra: &[&str]) -> Vec<String> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut next = 0u32;
    let mut v: Vec<String> = (0..=255u8)
        .map(|b| {
            if printable(b) {
                char::from(b).to_string()
            } else {
                next += 1;
                char::from_u32(255 + next).expect("valid").to_string()
            }
        })
        .collect();
    v.extend(extra.iter().map(|s| (*s).to_string()));
    v
}

fn write_apr(dir: &Path, architecture: &str, vocab: &[String], merges: &[&str]) -> PathBuf {
    let mut w = AprWriter::new();
    w.set_metadata("architecture", serde_json::json!(architecture));
    if !vocab.is_empty() {
        w.set_metadata("tokenizer.vocabulary", serde_json::json!(vocab));
        w.set_metadata("tokenizer.merges", serde_json::json!(merges));
    }
    w.add_tensor_f32("probe", vec![1], &[0.0]);
    let path = dir.join(format!("{architecture}.apr"));
    w.write(&path).expect("write apr");
    path
}

#[test]
fn an_apr_without_an_embedded_vocabulary_is_none_so_the_sibling_is_tried() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_apr(dir.path(), "qwen2", &[], &[]);
    assert!(InstructPipeline::extract_embedded_tokenizer(&path).expect("no error").is_none());
}

/// The case the cop required a row for: the file DOES embed a tokenizer, a byte-level one
/// with no implemented pre-tokenizer. This used to be `.ok()`: `None`, so the fine-tune
/// reported "no embedded tokenizer" or silently took a sibling file instead.
#[test]
fn an_embedded_byte_level_vocabulary_that_cannot_be_encoded_is_a_named_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_apr(dir.path(), "llama", &byte_level_vocab(&["ab"]), &["a b"]);
    let err = InstructPipeline::extract_embedded_tokenizer(&path)
        .expect_err("an embedded tokenizer that cannot be loaded is an error");
    let msg = err.to_string();
    assert!(msg.contains("cannot be loaded") && msg.contains("#3742"), "{msg}");
}

#[cfg(feature = "realizar")]
#[test]
fn an_embedded_qwen2_byte_level_vocabulary_is_canonical() {
    let dir = tempfile::tempdir().expect("tempdir");
    let vocab = byte_level_vocab(&["ab"]);
    let path = write_apr(dir.path(), "qwen2", &vocab, &["a b"]);
    let tok =
        InstructPipeline::extract_embedded_tokenizer(&path).expect("loads").expect("embedded");
    assert_eq!(tok.encode("ab"), vec![vocab.len() as u32 - 1], "the merge is applied");
    assert_eq!(tok.decode(&tok.encode("ab é\n")), "ab é\n", "byte-level round trip");
}

#[test]
fn an_embedded_vocabulary_that_is_not_byte_level_loads_through_aprender() {
    let dir = tempfile::tempdir().expect("tempdir");
    let vocab: Vec<String> = ["a", "b", "ab"].iter().map(|s| (*s).to_string()).collect();
    let path = write_apr(dir.path(), "llama", &vocab, &["a b"]);
    let tok =
        InstructPipeline::extract_embedded_tokenizer(&path).expect("loads").expect("embedded");
    assert_eq!(tok.vocab_size(), 3);
}
