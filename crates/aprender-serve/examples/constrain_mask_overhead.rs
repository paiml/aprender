//! #3568: what schema-constrained decoding costs per token, on a REAL vocabulary.
//!
//!   cargo run --release -p aprender-serve --features structured-output \
//!       --example constrain_mask_overhead -- <model.gguf> [schema | @schema.json]
//!
//! It indexes the model's vocabulary once and compiles the schema (default: RAH's quorum-lane
//! schema, #3716). Then it feeds a real verdict document, tokenized by the model's own
//! tokenizer, through `mask` + `accept` one token at a time, the way a generation loop will.
//! It reports the index time, the compile time, and the per-token mask cost (p50 / p95 / max).
//! No weights run. This is the constraint's own overhead, which is the same on CPU and CUDA; a
//! CUDA loop also pays one host-logits download per token (#3568 PR 3 measures that).

use std::time::Instant;

use realizar::constrain::{load_schema, ConstraintEnv};
use realizar::gguf::MappedGGUFModel;

const DOC: &str = r#"{"verdict":"PASS","summary":"The diff classifies every receipt with a strict RFC 8259 parse in awk and removes the UNMEASURED exit-3 path.","findings":[{"file":"scripts/parity_receipt_denominator.sh","line":42,"claim":"classify() parses rather than substring-matches","grounding":"cited"},{"file":"crates/aprender-contracts-cli/tests/ont4c3_parity_receipts.rs","line":213,"claim":"the exit-3 acceptance is gone","grounding":"measured","fix":"none needed"}]}"#;

fn rah_lane_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object", "required": ["verdict", "summary", "findings"],
        "properties": {
            "verdict": {"type": "string", "enum": ["PASS", "FAIL", "do-not-implement-as-written"]},
            "summary": {"type": "string"}, "tree_witness": {"type": "string"},
            "findings": {"type": "array", "items": {"type": "object", "required": ["file", "claim", "grounding"],
                "properties": {"file": {"type": "string"}, "line": {"type": "integer"}, "claim": {"type": "string"},
                    "grounding": {"type": "string", "enum": ["cited", "measured", "asserted"]}, "fix": {"type": "string"}}}}
        }
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: constrain_mask_overhead <model.gguf> [schema | @schema.json]");
    let mapped = MappedGGUFModel::from_path(path).expect("map the GGUF");
    let vocab = mapped
        .model
        .constraint_vocab()
        .expect("the GGUF has a vocabulary and an eos id");

    let t = Instant::now();
    let env = ConstraintEnv::new(&vocab).expect("index the vocabulary");
    let index_ms = t.elapsed().as_secs_f64() * 1e3;

    let schema = args.get(2).map_or_else(rah_lane_schema, |a| {
        load_schema(a).expect("the schema argument")
    });
    let t = Instant::now();
    let mut constraint = env.json_schema(&schema).expect("compile the schema");
    let compile_ms = t.elapsed().as_secs_f64() * 1e3;

    let tokens = mapped
        .model
        .encode(DOC)
        .expect("the model's tokenizer encodes the document");
    assert_eq!(
        mapped.model.decode(&tokens),
        DOC,
        "the tokenization round-trips"
    );

    let mut logits = vec![0.0f32; vocab.token_bytes.len()];
    let mut mask_us = Vec::with_capacity(tokens.len());
    for &token in &tokens {
        logits.iter_mut().for_each(|l| *l = 0.0);
        let t = Instant::now();
        constraint
            .mask(&mut logits)
            .expect("a mask at every position of a valid document");
        mask_us.push(t.elapsed().as_secs_f64() * 1e6);
        assert!(
            logits[token as usize].is_finite(),
            "the document's own token must be allowed"
        );
        constraint
            .accept(token)
            .expect("the document's own token is accepted");
    }
    let complete = constraint.is_complete();
    mask_us.sort_by(f64::total_cmp);
    let pct = |p: f64| mask_us[((mask_us.len() - 1) as f64 * p).round() as usize];
    println!(
        "vocab={} tokens={} index_ms={index_ms:.1} compile_ms={compile_ms:.2} mask_us p50={:.1} p95={:.1} max={:.1} complete={complete}",
        vocab.token_bytes.len(),
        tokens.len(),
        pct(0.50),
        pct(0.95),
        pct(1.0),
    );
}
