//! #3793: the dense CPU loop under a constraint, on a real (tiny, random-weight) GGUF model.
//!
//! The model's weights are noise, so what it "wants" to write is noise. Under a schema its
//! output must still be a conforming document: the mask, not the model, is what makes it so.
//! Make the mask a no-op and these rows go RED; that is the mutant #3793 names.

use crate::constrain::{ConstraintEnv, ConstraintRequest};
use crate::gguf::test_factory::build_executable_pygmy_gguf_with;
use crate::gguf::{ConstrainedStop, MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use std::io::Write;

/// The pygmy model's 32 tokens: control tokens, JSON punctuation, digits, and the letters of
/// `true`/`false`, so a small schema's documents can be spelled.
const VOCAB: [&str; 32] = [
    "<unk>", "</s>", "{", "}", "\"", ":", ",", "[", "]", "a", "b", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "-", ".", "t", "r", "u", "e", "f", "l", "s", "n", "x",
];
const EOS: u32 = 1;

fn pygmy_with_vocab() -> (tempfile::NamedTempFile, MappedGGUFModel) {
    let types: Vec<i32> = VOCAB
        .iter()
        .map(|t| match *t {
            "<unk>" => 2,
            "</s>" => 3,
            _ => 1,
        })
        .collect();
    let bytes = build_executable_pygmy_gguf_with(|b| {
        b.add_string("tokenizer.ggml.model", "llama")
            .add_string_array("tokenizer.ggml.tokens", &VOCAB)
            .add_i32_array("tokenizer.ggml.token_type", &types)
            .add_u32("tokenizer.ggml.eos_token_id", EOS)
    });
    let mut f = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp file");
    f.write_all(&bytes).expect("write");
    let mapped = MappedGGUFModel::from_path(f.path()).expect("map the pygmy model");
    (f, mapped)
}

fn config(max_tokens: usize) -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![EOS],
        ..Default::default()
    }
}

fn text_of(generated: &[u32]) -> String {
    generated.iter().map(|&t| VOCAB[t as usize]).collect()
}

/// `{"a": <integer>}` and nothing else.
fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "a": { "type": "integer" } },
        "required": ["a"],
        "additionalProperties": false
    })
}

fn constrained(
    mapped: &MappedGGUFModel,
    prompt: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, ConstrainedStop) {
    let model = OwnedQuantizedModel::from_mapped(mapped).expect("load");
    let env = ConstraintEnv::new(&mapped.model.constraint_vocab().expect("a vocabulary"))
        .expect("index the vocabulary");
    let mut c = ConstraintRequest::JsonSchema(schema())
        .compile(&env)
        .expect("compile the schema");
    let (tokens, stop) = model
        .generate_with_cache_constrained(prompt, &config(max_tokens), c.as_mut())
        .expect("generate");
    (tokens[prompt.len()..].to_vec(), stop)
}

#[test]
fn a_random_weight_model_under_a_schema_writes_a_conforming_document() {
    let (_f, mapped) = pygmy_with_vocab();
    // The unconstrained loop, for contrast: noise, not a document
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("load");
    let free = model
        .generate_with_cache(&[9], &config(12))
        .expect("generate");
    let free_text = text_of(&free[1..]);
    assert!(
        serde_json::from_str::<serde_json::Value>(&free_text).is_err(),
        "the pygmy model wrote JSON unconstrained ({free_text:?}), so this row would prove nothing"
    );

    let (generated, stop) = constrained(&mapped, &[9], 24);
    let text = text_of(&generated);
    assert_eq!(stop, ConstrainedStop::Complete, "{text:?}");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("one JSON document");
    let object = doc.as_object().expect("an object");
    assert_eq!(object.len(), 1, "{text:?}");
    assert!(object["a"].is_i64() || object["a"].is_u64(), "{text:?}");
    // The end-of-sequence token is never part of the returned text
    assert!(!generated.contains(&EOS));
}

#[test]
fn a_budget_that_ends_mid_document_is_length_never_complete() {
    let (_f, mapped) = pygmy_with_vocab();
    // `{"a":` needs five tokens before any value; two cannot finish it
    let (generated, stop) = constrained(&mapped, &[9], 2);
    assert_eq!(stop, ConstrainedStop::Length, "{:?}", text_of(&generated));
    assert_eq!(generated.len(), 2);
}

#[test]
fn greedy_constrained_decoding_is_deterministic() {
    let (_f, mapped) = pygmy_with_vocab();
    assert_eq!(
        constrained(&mapped, &[9], 24),
        constrained(&mapped, &[9], 24)
    );
}
