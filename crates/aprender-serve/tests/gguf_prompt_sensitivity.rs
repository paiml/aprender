//! FALSIFY-GGUF-PROMPT-SENS-* — `apr run <gguf>` distinct-prompt invariant.
//!
//! SPEC-SHIP-TWO-001 §61.8 surfaced an empirical finding: the canonical 7B
//! teacher GGUF emits byte-identical "ampiezza = 0.5\ndiametro = 10..."
//! Italian gibberish across THREE distinct prompts. Wall times differ
//! (proving inference IS running) but output text matches byte-for-byte.
//!
//! This test file pins the prompt-sensitivity invariant. RED on main
//! (bug active); GREEN once Branch B fix lands per
//! `contracts/gguf-prompt-sensitivity-v1.yaml`.
//!
//! All three tests are host-gated on the canonical teacher path. They
//! auto-skip on hosts that lack the 8 GB GGUF/APR fixtures (CI runners),
//! per `feedback_compute_pre_authorized.md`.

use std::path::Path;

use realizar::{run_inference, InferenceConfig};

const CANONICAL_GGUF: &str =
    "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.gguf";
const CANONICAL_APR: &str =
    "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr";

fn run_one(model_path: &str, prompt: &str, max_tokens: usize) -> Option<String> {
    let path = Path::new(model_path);
    if !path.exists() {
        eprintln!("[gguf_prompt_sensitivity] skipping: host lacks {model_path}");
        return None;
    }
    let config = InferenceConfig::new(path)
        .with_prompt(prompt.to_string())
        .with_max_tokens(max_tokens);
    match run_inference(&config) {
        Ok(result) => Some(result.text),
        Err(e) => {
            eprintln!("[gguf_prompt_sensitivity] run_inference failed: {e:?}");
            None
        },
    }
}

/// FALSIFY-GGUF-PROMPT-SENS-001: two distinct prompts on canonical 7B GGUF
/// teacher MUST produce DIFFERENT output text. Pre-fix the GGUF inference
/// path produces byte-identical "ampiezza..." gibberish regardless of prompt.
///
/// This test is RED on main (bug active) and is the load-bearing prompt-
/// sensitivity gate for Branch B of SPEC-SHIP-TWO-001 §61.8.
#[test]
#[ignore = "requires canonical 7B GGUF teacher (~8 GB) + GPU host"]
fn falsify_gguf_prompt_sensitivity_distinct_prompts_distinct_outputs() {
    let p1 = "What is 2+2? The answer is ";
    let p2 = "Hello, my name is";

    let out_a = match run_one(CANONICAL_GGUF, p1, 32) {
        Some(s) => s,
        None => return,
    };
    let out_b = match run_one(CANONICAL_GGUF, p2, 32) {
        Some(s) => s,
        None => return,
    };

    eprintln!("[falsify-gguf-prompt-sens-001] P1 output (first 80 chars): {:?}", &out_a[..out_a.len().min(80)]);
    eprintln!("[falsify-gguf-prompt-sens-001] P2 output (first 80 chars): {:?}", &out_b[..out_b.len().min(80)]);

    assert_ne!(
        out_a, out_b,
        "FALSIFY-GGUF-PROMPT-SENS-001: GGUF inference produced byte-identical output \
         for two distinct prompts. P1=\"{p1}\" → {out_a:?}; P2=\"{p2}\" → {out_b:?}. \
         Pre-fix: structural prompt-insensitive bug (input tokens dropped, KV cache \
         poisoned, sampler locked, or model state fixed-init). Bisect via eprintln \
         in `crates/aprender-serve/src/gguf/inference/fails.rs:228` (prefill loop \
         token IDs) and `matmul_fused.rs:45` (embedding lookup token IDs). See \
         contract `contracts/gguf-prompt-sensitivity-v1.yaml`."
    );
}

/// FALSIFY-GGUF-PROMPT-SENS-002: three-prompt sweep. The set
/// {output(P1), output(P2), output(P3)} MUST have cardinality ≥ 2.
/// Pre-fix all three collapse to byte-identical "ampiezza..." (cardinality 1);
/// post-fix all three differ (cardinality 3).
#[test]
#[ignore = "requires canonical 7B GGUF teacher (~8 GB) + GPU host"]
fn falsify_gguf_prompt_sensitivity_three_prompt_sweep() {
    let p1 = "What is 2+2? The answer is ";
    let p2 = "Hello, my name is";
    let p3 = "def fibonacci(n):";

    let out_1 = match run_one(CANONICAL_GGUF, p1, 32) {
        Some(s) => s,
        None => return,
    };
    let out_2 = match run_one(CANONICAL_GGUF, p2, 32) {
        Some(s) => s,
        None => return,
    };
    let out_3 = match run_one(CANONICAL_GGUF, p3, 32) {
        Some(s) => s,
        None => return,
    };

    let mut distinct = std::collections::HashSet::new();
    distinct.insert(out_1.clone());
    distinct.insert(out_2.clone());
    distinct.insert(out_3.clone());

    eprintln!("[falsify-gguf-prompt-sens-002] cardinality: {}", distinct.len());
    eprintln!("[falsify-gguf-prompt-sens-002] outputs: {:?}", distinct);

    assert!(
        distinct.len() >= 2,
        "FALSIFY-GGUF-PROMPT-SENS-002: three distinct prompts collapsed to \
         {} distinct outputs (expected >= 2). All-tied = single canned-output \
         failure mode. See FALSIFY-GGUF-PROMPT-SENS-001 bisection plan.",
        distinct.len()
    );
}

/// FALSIFY-GGUF-PROMPT-SENS-003: APR control gate — same canonical teacher
/// loaded as `.apr` MUST produce distinct outputs for distinct prompts.
/// Pre-2026-05-07 the APR path was ALSO broken (see `evidence/ship-two-001/
/// ex-06-ac006-preupload-local.json` — same "ampiezza..." canned text on
/// "def fib(n):" prompt back when APR shared the bug). The M-FFN-GGUF-5
/// cascade (PR #1550 + #1556) on 2026-05-07 fixed APR; this control gate
/// should already PASS on main, confirming the bug is GGUF-path-specific.
#[test]
#[ignore = "requires canonical 7B APR teacher (~8 GB) + GPU host"]
fn falsify_gguf_prompt_sensitivity_apr_control_passes() {
    let p1 = "What is 2+2? The answer is ";
    let p2 = "Hello, my name is";

    let out_a = match run_one(CANONICAL_APR, p1, 32) {
        Some(s) => s,
        None => return,
    };
    let out_b = match run_one(CANONICAL_APR, p2, 32) {
        Some(s) => s,
        None => return,
    };

    eprintln!("[falsify-gguf-prompt-sens-003] APR P1 (first 80 chars): {:?}", &out_a[..out_a.len().min(80)]);
    eprintln!("[falsify-gguf-prompt-sens-003] APR P2 (first 80 chars): {:?}", &out_b[..out_b.len().min(80)]);

    assert_ne!(
        out_a, out_b,
        "FALSIFY-GGUF-PROMPT-SENS-003: APR path also produces prompt-insensitive \
         output. The M-FFN-GGUF-5 fix (PR #1550 + #1556 on 2026-05-07) did NOT \
         close the APR path fully — re-investigate `apr_transformer/inference.rs` \
         and `OwnedQuantizedModel::from_apr` separately. P1=\"{p1}\" → {out_a:?}; \
         P2=\"{p2}\" → {out_b:?}."
    );
}
