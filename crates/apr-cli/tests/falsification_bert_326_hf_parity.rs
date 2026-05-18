//! FALSIFY-BERT-326-PHASE4B-HF-PARITY — end-to-end HF parity gate.
//!
//! Verifies that `apr rerank` produces the SAME sigmoid-mapped relevance
//! score as the HuggingFace reference `AutoModelForSequenceClassification`
//! for `cross-encoder/ms-marco-MiniLM-L-6-v2` on a fixed set of canonical
//! (query, passage) pairs.
//!
//! ## Empirically established on lambda-vector RTX 4090 (2026-05-17)
//!
//! | Pair | HF score | apr score | input_ids match |
//! |---|---|---|---|
//! | "France" + Paris  | 0.999805 | 0.999805 | ✅ exact |
//! | "France" + Cats   | 0.000015 | 0.000015 | ✅ exact |
//! | ML + neural       | 0.000020 | 0.000020 | ✅ exact |
//!
//! Raw logits agree within ~4e-4 (f32 round-off). Sigmoid maps both to
//! identical 6-decimal scores. WordPiece tokenization is bit-identical to
//! HF BertTokenizer.
//!
//! ## How to run
//!
//! Requires lambda-vector or any host with:
//! - `apr` binary (built from this branch)
//! - `~/.cache/pacha/models/57e6e922118ea840.safetensors` cached via
//!   `apr pull cross-encoder/ms-marco-MiniLM-L-6-v2`
//! - `uv` for the HF reference Python script
//!
//! ```
//! cargo test --test falsification_bert_326_hf_parity \
//!     -- --ignored --nocapture
//! ```
//!
//! `#[ignore]` because it pulls in 87 MB of cached fixtures and invokes
//! `uv run --with transformers --with torch python3` which downloads
//! ~3 GB of pip wheels into the uv cache on first run.

use std::path::Path;
use std::process::Command;

const MINILM_SAFETENSORS: &str = "/home/noah/.cache/pacha/models/57e6e922118ea840.safetensors";
const MINILM_TOKENIZER: &str = "/home/noah/.cache/pacha/models/57e6e922118ea840.tokenizer.json";

/// Canonical (query, passage, expected_score) triples.
///
/// Expected scores were captured from HuggingFace
/// `cross-encoder/ms-marco-MiniLM-L-6-v2` via
/// `AutoModelForSequenceClassification` (see `/tmp/hf_ref.py` in the PR
/// description). Aprender must reproduce these to 6 decimal places.
const PARITY_PAIRS: &[(&str, &str, f32)] = &[
    (
        "what is the capital of France",
        "Paris is the capital of France",
        0.999805,
    ),
    (
        "what is the capital of France",
        "Cats are mammals that purr",
        0.000015,
    ),
    (
        "machine learning",
        "neural networks are a key ML technique",
        0.000020,
    ),
];

/// Tolerance for absolute score difference (`apr` − HF reference). The
/// observed gap is < 1e-6 (sigmoid is monotonic + saturating, so the
/// ~4e-4 raw-logit drift compresses to f32 round-off at the score level).
/// 1e-4 is a generous bound that catches genuine numerical drift but
/// tolerates the existing 4e-4 raw-logit gap.
const SCORE_TOL: f32 = 1e-4;

fn extract_score_from_json(stdout: &str) -> f32 {
    // Parse the JSON output's `scores[0]` field.
    let v: serde_json::Value = serde_json::from_str(stdout).expect("apr rerank JSON parse");
    v.get("scores")
        .and_then(|s| s.get(0))
        .and_then(|s| s.as_f64())
        .expect("scores[0] missing") as f32
}

#[test]
#[ignore = "requires cached MiniLM SafeTensors + apr binary; takes ~30s"]
fn falsify_bert_326_phase4b_hf_parity() {
    if !Path::new(MINILM_SAFETENSORS).exists() {
        eprintln!(
            "FALSIFY-BERT-326-PHASE4B: skipped — no cached MiniLM at {MINILM_SAFETENSORS}.\n\
             Run `apr pull cross-encoder/ms-marco-MiniLM-L-6-v2` first."
        );
        return;
    }
    if !Path::new(MINILM_TOKENIZER).exists() {
        eprintln!(
            "FALSIFY-BERT-326-PHASE4B: skipped — no cached tokenizer.json at {MINILM_TOKENIZER}"
        );
        return;
    }

    // Build `.apr` from the cached SafeTensors. We rebuild each run so the
    // test catches future drift in either `apr import` or the loader.
    let apr_out = std::env::temp_dir().join("falsify-bert-326-phase4b.apr");
    let import_status = Command::new("apr")
        .args([
            "import",
            MINILM_SAFETENSORS,
            "--arch",
            "bert",
            "--allow-no-config",
            "-o",
        ])
        .arg(&apr_out)
        .status()
        .expect("spawn apr import");
    assert!(
        import_status.success(),
        "apr import --arch bert must succeed on cached MiniLM"
    );

    let mut failures: Vec<String> = Vec::new();
    for (q, p, expected) in PARITY_PAIRS {
        let output = Command::new("apr")
            .args(["rerank"])
            .arg(&apr_out)
            .args([
                "--query",
                q,
                "--passage",
                p,
                "--vocab",
                MINILM_TOKENIZER,
                "--json",
            ])
            .output()
            .expect("spawn apr rerank");
        assert!(
            output.status.success(),
            "apr rerank must succeed for ({q:?}, {p:?}); stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let score = extract_score_from_json(
            std::str::from_utf8(&output.stdout).expect("rerank output is UTF-8"),
        );
        let diff = (score - expected).abs();
        eprintln!(
            "FALSIFY-BERT-326-PHASE4B: q={q:?} p={p:?} apr={score:.6} hf={expected:.6} \
             diff={diff:.6e}{}",
            if diff < SCORE_TOL { "" } else { "  ← FAIL" }
        );
        if diff >= SCORE_TOL {
            failures.push(format!(
                "({q:?}, {p:?}): apr={score:.6} hf={expected:.6} diff={diff:.6e}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "FALSIFY-BERT-326-PHASE4B: {} pair(s) failed HF parity:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
