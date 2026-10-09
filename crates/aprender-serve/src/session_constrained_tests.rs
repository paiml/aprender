//! #3793: the one engine under a constraint, on a real (tiny, random-weight) GGUF model.
//!
//! The model's weights are noise, so what it "wants" to write is noise. Under a schema its
//! output must still be a conforming document: the mask, not the model, is what makes it so.
//! Make the mask a no-op and these rows go RED; that is the mutant #3793 names.

use crate::constrain::{ConstraintEnv, ConstraintRequest};
use crate::gguf::dense_session::{DenseForward, DenseSession};
use crate::gguf::test_factory::build_executable_pygmy_gguf_with;
use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use crate::session::{entries_of_session, ConstrainedStop, EntryKind};
use std::io::Write;
use std::sync::Arc;

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

/// The engine `apr run` takes on the CPU for a dense GGUF.
fn session(mapped: &MappedGGUFModel) -> DenseSession {
    let model = OwnedQuantizedModel::from_mapped(mapped).expect("load");
    DenseSession::new(DenseForward::cpu(Arc::new(model)))
}

fn constrained_on(
    session: &mut DenseSession,
    mapped: &MappedGGUFModel,
    prompt: &[u32],
    config: &QuantizedGenerateConfig,
) -> (Vec<u32>, ConstrainedStop) {
    let env = ConstraintEnv::new(&mapped.model.constraint_vocab().expect("a vocabulary"))
        .expect("index the vocabulary");
    let mut c = ConstraintRequest::JsonSchema(schema())
        .compile(&env)
        .expect("compile the schema");
    let (turn, stop) = session
        .generate_constrained(prompt, config, c.as_mut())
        .expect("generate");
    (turn.tokens[prompt.len()..].to_vec(), stop)
}

fn constrained(
    mapped: &MappedGGUFModel,
    prompt: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, ConstrainedStop) {
    constrained_on(&mut session(mapped), mapped, prompt, &config(max_tokens))
}

fn assert_conforms(text: &str) {
    let doc: serde_json::Value = serde_json::from_str(text).expect("one JSON document");
    let object = doc.as_object().expect("an object");
    assert_eq!(object.len(), 1, "{text:?}");
    assert!(object["a"].is_i64() || object["a"].is_u64(), "{text:?}");
}

#[test]
fn a_random_weight_model_under_a_schema_writes_a_conforming_document() {
    let (_f, mapped) = pygmy_with_vocab();
    // The same engine unconstrained, for contrast: noise, not a document
    let free = session(&mapped)
        .generate(&[9], &config(12), &mut |_| true)
        .expect("generate");
    let free_text = text_of(&free.tokens[1..]);
    assert!(
        serde_json::from_str::<serde_json::Value>(&free_text).is_err(),
        "the pygmy model wrote JSON unconstrained ({free_text:?}), so this row would prove nothing"
    );

    let (generated, stop) = constrained(&mapped, &[9], 24);
    let text = text_of(&generated);
    assert_eq!(stop, ConstrainedStop::Complete, "{text:?}");
    assert_conforms(&text);
    // The end-of-sequence token is never part of the returned text
    assert!(!generated.contains(&EOS));
}

#[test]
fn a_sampled_constrained_run_with_a_penalty_still_conforms() {
    // The choice the engine makes off the greedy path (seeded top-k/top-p after the
    // repetition penalty) is also taken after the mask.
    let (_f, mapped) = pygmy_with_vocab();
    let sampled = QuantizedGenerateConfig {
        temperature: 0.8,
        top_k: 8,
        top_p: 0.95,
        seed: 7,
        repeat_penalty: 1.3,
        repeat_last_n: 16,
        ..config(32)
    };
    let (generated, stop) = constrained_on(&mut session(&mapped), &mapped, &[9], &sampled);
    let text = text_of(&generated);
    assert_eq!(stop, ConstrainedStop::Complete, "{text:?}");
    assert_conforms(&text);
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

#[test]
fn a_constrained_run_enters_the_one_engine() {
    // #4263: the witness records the constrained entry like any other generate, so
    // `tests_engine_identity` sees a constrained `apr run` the way it sees the rest.
    let (_f, mapped) = pygmy_with_vocab();
    let mut s = session(&mapped);
    let id = s.id();
    let _ = constrained_on(&mut s, &mapped, &[9], &config(24));
    let entries = entries_of_session(id);
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert_eq!(entries[0].kind, EntryKind::Generate);
    assert!(!entries[0].on_gpu);
}

#[test]
fn a_second_constrained_turn_on_the_same_session_reuses_the_prefix() {
    // The prefill is the session's own: a prompt that extends what the state holds
    // forwards only the new suffix, and the answer is the one a fresh session gives.
    let (_f, mapped) = pygmy_with_vocab();
    let mut s = session(&mapped);
    let _ = constrained_on(&mut s, &mapped, &[9, 10], &config(24));
    let again = constrained_on(&mut s, &mapped, &[9, 10, 9], &config(24));
    let fresh = constrained_on(&mut session(&mapped), &mapped, &[9, 10, 9], &config(24));
    assert_eq!(again, fresh);
}
