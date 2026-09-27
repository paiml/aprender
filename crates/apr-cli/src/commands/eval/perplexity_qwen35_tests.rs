//! #4270: `apr eval` perplexity on a Qwen3.5 hybrid goes through the one engine.

use super::super::{log_softmax_at, session_perplexity, Dataset, EvalConfig};
use super::run_evaluation;
use std::path::Path;

const MODEL: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

fn config(text: &str) -> EvalConfig {
    EvalConfig {
        dataset: Dataset::Custom,
        text: Some(text.to_string()),
        max_tokens: 64,
        threshold: 1.0e9,
    }
}

const PROSE: &str = "The quick brown fox jumps over the lazy dog. \
    The capital of France is Paris, and the capital of Germany is Berlin. \
    Water boils at one hundred degrees Celsius at sea level.";

/// The eval verb must enter the engine through `Session::score` — a qwen35 entry
/// of kind Score for exactly the sequence it scored. Before #4270 the verb built a
/// dense `OwnedQuantizedModel` and left no entry.
#[test]
fn eval_perplexity_on_qwen35_enters_session_score() {
    if !Path::new(MODEL).exists() {
        eprintln!("SKIP: {MODEL} is absent");
        return;
    }
    let result = run_evaluation(Path::new(MODEL), &config(PROSE), true).expect("eval runs");
    let mapped = realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map");
    let tokens: Vec<u32> = mapped
        .model
        .encode(PROSE)
        .expect("tokenizer")
        .into_iter()
        .take(64)
        .collect();
    let entries = realizar::session::entries_for(&tokens);
    assert!(
        entries.iter().any(|e| e.arch == "qwen35"
            && e.kind == realizar::session::EntryKind::Score
            && !e.on_gpu),
        "no qwen35 CPU Score entry for the scored sequence: {entries:?}"
    );
    assert_eq!(result.tokens_evaluated, tokens.len());
    assert!(
        result.perplexity.is_finite() && result.perplexity > 1.0,
        "{result:?}"
    );
}

/// The score reads the logits after `pos` against `tokens[pos + 1]`. English prose
/// must score far better than the same tokens reversed; an off-by-one target (or a
/// model that is not running) collapses that gap.
#[test]
fn qwen35_perplexity_separates_prose_from_its_reversal() {
    if !Path::new(MODEL).exists() {
        eprintln!("SKIP: {MODEL} is absent");
        return;
    }
    let mapped = realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map");
    let mut session =
        realizar::gguf::qwen35_session::Qwen35Session::load(&mapped, true).expect("load");
    let tokens: Vec<u32> = mapped.model.encode(PROSE).expect("tokenizer");
    let reversed: Vec<u32> = tokens.iter().rev().copied().collect();
    let (ppl, ce) = session_perplexity(&mut session, &tokens).expect("prose");
    let (ppl_rev, _) = session_perplexity(&mut session, &reversed).expect("reversed");
    assert!((ce.exp() - ppl).abs() <= 1e-3 * ppl);
    assert!(ppl < 60.0, "prose perplexity {ppl}");
    assert!(ppl_rev > 10.0 * ppl, "prose {ppl} vs reversed {ppl_rev}");
}

#[test]
fn log_softmax_at_matches_the_closed_form() {
    let logits = [1.0f32, 2.0, 3.0];
    let z: f64 = [1.0f64, 2.0, 3.0].iter().map(|x| x.exp()).sum();
    let got = log_softmax_at(&logits, 2).expect("in vocab");
    assert!((got - (3.0 - z.ln())).abs() < 1e-9);
    assert!(
        log_softmax_at(&logits, 3).is_none(),
        "out of vocab is skipped, not clamped"
    );
}
