// Integration tests: unwrap()/panic!() are idiomatic; strict workspace lints relaxed here.
#![allow(clippy::disallowed_methods, clippy::unwrap_used)]

//! M32c.2.2.2.1.4 — live falsifier pinning FALSIFY-QW3-MOE-FORWARD-003.
//!
//! Verifies that `apr run` (the user-facing CLI) emits at least one
//! non-whitespace character on stdout when invoked against the cached
//! Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf with a fresh prompt.
//!
//! This is the regression-prevention pin for the M32c.2.2.2.1.3 dispatch
//! flip (PR #1126, squash commit a902eea93). Before that flip the
//! qwen3_moe arch routed to `run_gguf_generate` whose dense FFN path
//! produced garbage on MoE weights; after the flip it routes to
//! `run_qwen3_moe_generate` whose forward emits real tokens.
//!
//! Token quality vs llama.cpp Q4_K is M32d (numerical parity). This
//! test asserts ONLY emit/exit-0 — the discharge gate for
//! FALSIFY-QW3-MOE-FORWARD-003.
//!
//! ## Skip path
//!
//! On hosts without the cached GGUF (CI runners, fresh dev boxes), the
//! test prints a SKIP marker and returns success. The lambda-vector
//! development host has the GGUF mmapped at
//! `/home/noah/.cache/pacha/models/2b88b180a790988f.gguf`.
//!
//! A SKIP measured nothing, so the lane that owns this falsifier
//! (cuda-nightly `falsifiers`, gx10, #4179) sets
//! `APR_QW3_MOE_REQUIRE_MODEL=1`: there a missing model FAILS instead of
//! skipping. `APR_QW3_MOE_GGUF` names the model ahead of the canonical paths.
//!
//! ## Why this is heavy
//!
//! `apr run --max-tokens 1` on the 17.3 GB Q4_K_M GGUF performs:
//! - mmap fault-in (~10 GB of pages)
//! - 48 × (RMSNorm + QKV proj + RoPE + causal attn + 128-expert
//!   softmax routing + top-8 × per-expert SwiGLU) per token
//! - lm_head over vocab=151936
//!
//! On lambda-vector RTX 4090 host this takes ~10s warm / ~130s cold.
//! KV-cache integration
//! is M32d follow-up — full prefill per token is acceptable here since
//! we only need ANY token to emit.

use std::path::Path;

use assert_cmd::Command;

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

/// Fresh prompt seed — date-tagged so the same-prompt fast-path can't
/// short-circuit the forward. Updated when this test is touched.
const FRESH_PROMPT: &str = "M32c.2.2.2.1.4 live falsifier 2026-04-29: write the letter q.";

/// The model to run: the override first, then the canonical paths. `Ok(None)`
/// is a SKIP, allowed only when the lane does not require the model (#4179).
fn resolve_model(
    override_path: Option<String>,
    candidates: &[&str],
    require: bool,
) -> Result<Option<String>, String> {
    let found = override_path
        .into_iter()
        .chain(candidates.iter().map(|p| (*p).to_string()))
        .find(|p| Path::new(p).exists());
    match found {
        Some(p) => Ok(Some(p)),
        None if require => Err(format!(
            "APR_QW3_MOE_REQUIRE_MODEL=1 and no Qwen3-Coder GGUF at any of {candidates:?} \
             (or APR_QW3_MOE_GGUF): a SKIP here would report a pass that measured nothing"
        )),
        None => Ok(None),
    }
}

#[test]
fn f_qw3_moe_c22214_000_require_model_fails_closed() {
    let missing = &["/nonexistent/qw3-moe-4179.gguf"];
    assert_eq!(resolve_model(None, missing, false), Ok(None));
    assert!(resolve_model(None, missing, true).is_err());
    assert!(resolve_model(Some("/nonexistent/o.gguf".into()), missing, true).is_err());
    let exe = std::env::current_exe().unwrap().display().to_string();
    assert_eq!(
        resolve_model(Some(exe.clone()), missing, true),
        Ok(Some(exe))
    );
}

#[test]
fn f_qw3_moe_c22214_001_apr_run_emits_at_least_one_non_whitespace_char() {
    let require = std::env::var("APR_QW3_MOE_REQUIRE_MODEL").as_deref() == Ok("1");
    let resolved = resolve_model(
        std::env::var("APR_QW3_MOE_GGUF").ok(),
        CANONICAL_QWEN3_CODER_GGUF_PATHS,
        require,
    );
    let Some(gguf_path) = resolved.unwrap_or_else(|e| panic!("F-QW3-MOE-C22214-001: {e}")) else {
        eprintln!(
            "F-QW3-MOE-C22214-001: SKIP — no cached Qwen3-Coder GGUF at any of {:?}",
            CANONICAL_QWEN3_CODER_GGUF_PATHS
        );
        return;
    };
    let gguf_path = gguf_path.as_str();

    eprintln!("F-QW3-MOE-C22214-001: live `apr run` against {gguf_path}");
    let start = std::time::Instant::now();

    let output = Command::cargo_bin("apr")
        .expect("cargo_bin(apr) must locate the binary")
        .args([
            "run",
            gguf_path,
            "--prompt",
            FRESH_PROMPT,
            "--max-tokens",
            "1",
        ])
        .output()
        .expect("apr binary must execute");

    let elapsed = start.elapsed();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!(
        "F-QW3-MOE-C22214-001: elapsed = {elapsed:?}\n  stdout (first 200B): {stdout:.200}\n  stderr (first 200B): {stderr:.200}",
    );

    assert!(
        output.status.success(),
        "F-QW3-MOE-C22214-001: `apr run` must exit 0 — got {:?}\nstderr:\n{stderr}",
        output.status,
    );

    assert!(
        stdout.chars().any(|c| !c.is_whitespace()),
        "F-QW3-MOE-C22214-001: `apr run` must emit ≥1 non-whitespace char on stdout — \
         got empty/whitespace-only stdout. \nstdout:\n{stdout}\nstderr:\n{stderr}",
    );

    eprintln!("F-QW3-MOE-C22214-001: PASS");
}
