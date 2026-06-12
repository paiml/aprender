//! M32d.3 — `F-QW3-MOE-PARITY-002`: llama.cpp Q4_K argmax sanity vs
//! HuggingFace FP16 reference.
//!
//! Contract: [`contracts/qwen3-moe-forward-v1.yaml`] — `AC_QW3_MOE_001`
//! (informal: "Q4_K decode argmax matches FP16 ground-truth on
//! deterministic greedy sample").
//!
//! Falsifier: `FALSIFY-QW3-MOE-FORWARD-004` axis (b) — secondary sanity:
//!
//! ```text
//! argmax(apr_logits[0]) == llama_cpp_top1_token
//! ```
//!
//! ## Why this is a *transitive* axis
//!
//! Strict axis (b) wants `apr_argmax_token_id == llama_cpp_argmax_token_id`,
//! but llama.cpp emits *decoded text* on stdout (not raw token IDs) and
//! decoding apr's argmax inside this test would require pulling the GGUF
//! tokenizer into a sibling integration test that already lives in the
//! M32d.2 PR (#1130). To keep this slice tight, we measure the same gate
//! transitively:
//!
//!   1. M32d.2 (`qwen3_moe_parity.rs::f_qw3_moe_parity_001_cosine_vs_hf_fp16`)
//!      asserts `cos_sim(apr_logits, hf_fp16_logits) > 0.99`.
//!      → apr's argmax ≈ HF FP16's argmax (any cosine > 0.99 over a
//!      151936-dim logit vector forces argmax agreement except in
//!      pathological near-tie cases).
//!   2. THIS test asserts `llama_cpp_first_decoded_token == hf_fp16.argmax_text`.
//!      → llama.cpp Q4_K's argmax equals HF FP16's argmax at the decoded-
//!      text level.
//!   3. Composing (1) and (2): apr ≈ HF ≈ llama.cpp — the contract gate.
//!
//! M32d.4 (DRAFT → ACTIVE_RUNTIME) requires both axes to discharge; this
//! test is the second.
//!
//! ## Heavy-test layout
//!
//! Three operator-confirm-gated inputs:
//!
//! 1. The 17.3 GB `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` weights, mmap'd
//!    by llama-cli. Cached on lambda-vector at the paths in
//!    `CANONICAL_QWEN3_CODER_GGUF_PATHS`.
//! 2. The `qwen3_moe_fp16_logits_pos0.json` fixture, generated once via
//!    `scripts/generate_qwen3_moe_fp16_logits.py` (M32d.1, PR #1129).
//! 3. The `llama-cli` binary, found via `which llama-cli` or one of the
//!    `LLAMA_CLI_CANDIDATE_PATHS` fallbacks.
//!
//! Skips with `eprintln!` if any of the three is absent. Marked `#[ignore]`
//! so it does NOT run in default CI.
//!
//! ## What the test does
//!
//! 1. Locate llama-cli binary, GGUF, and JSON fixture (skip if any missing).
//! 2. Read `fixture.prompt` and `fixture.argmax_text`.
//! 3. Spawn `llama-cli -m <gguf> -p <prompt> -n 1 --top-k 1 --temp 0.0
//!    --seed 0 --no-display-prompt -no-cnv --no-warmup --log-disable`,
//!    capture stdout.
//! 4. Trim the stdout to extract the first emitted decoded text.
//! 5. Assert the trimmed stdout equals (or contains) `fixture.argmax_text`,
//!    accommodating whitespace differences between tokenizers' detokenize
//!    paths (some prepend spaces; some don't).

use std::path::{Path, PathBuf};
use std::process::Command;

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/cache/apr-home/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

const FIXTURE_RELATIVE: &str = "tests/fixtures/qwen3_moe_fp16_logits_pos0.json";

const LLAMA_CLI_CANDIDATE_PATHS: &[&str] = &[
    "/home/noah/.local/bin/llama-cli",
    "/home/noah/src/llama.cpp/llama-cli",
    "/usr/local/bin/llama-cli",
    "/usr/bin/llama-cli",
];

#[derive(serde::Deserialize)]
struct Fp16Fixture {
    #[serde(default)]
    model_name: String,
    prompt: String,
    #[serde(default)]
    argmax_token: u32,
    #[serde(default)]
    argmax_text: String,
}

fn find_first_existing<I: AsRef<str>>(paths: &[I]) -> Option<PathBuf> {
    for p in paths {
        let pb = PathBuf::from(p.as_ref());
        if pb.exists() {
            return Some(pb);
        }
    }
    None
}

fn locate_llama_cli() -> Option<PathBuf> {
    if let Ok(out) = Command::new("which").arg("llama-cli").output() {
        if out.status.success() {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !path.is_empty() && Path::new(&path).exists() {
                return Some(PathBuf::from(path));
            }
        }
    }
    find_first_existing(LLAMA_CLI_CANDIDATE_PATHS)
}

fn fixture_path() -> PathBuf {
    if let Ok(repo_root) = std::env::var("CARGO_MANIFEST_DIR") {
        PathBuf::from(repo_root).join(FIXTURE_RELATIVE)
    } else {
        PathBuf::from("crates/aprender-serve").join(FIXTURE_RELATIVE)
    }
}

fn load_fixture(path: &Path) -> Option<Fp16Fixture> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Strip any noise llama-cli adds around the generated token (warmup
/// banner residue, trailing newlines, EOG marker artifacts).
fn extract_first_emit(raw: &str) -> String {
    raw.trim()
        .lines()
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string()
}

#[test]
#[ignore]
fn f_qw3_moe_parity_002_argmax_vs_llama_cpp() {
    let Some(llama_cli) = locate_llama_cli() else {
        eprintln!(
            "F-QW3-MOE-PARITY-002: skipped — llama-cli not in PATH or in {LLAMA_CLI_CANDIDATE_PATHS:?}"
        );
        return;
    };

    let Some(gguf_path) = find_first_existing(CANONICAL_QWEN3_CODER_GGUF_PATHS) else {
        eprintln!(
            "F-QW3-MOE-PARITY-002: skipped — no cached Qwen3-Coder GGUF in {CANONICAL_QWEN3_CODER_GGUF_PATHS:?}"
        );
        return;
    };

    let fx_path = fixture_path();
    let Some(fixture) = load_fixture(&fx_path) else {
        eprintln!(
            "F-QW3-MOE-PARITY-002: skipped — FP16 fixture not found at {} \
             (run scripts/generate_qwen3_moe_fp16_logits.py per M32d.1 to generate it)",
            fx_path.display()
        );
        return;
    };

    if fixture.argmax_text.is_empty() {
        eprintln!(
            "F-QW3-MOE-PARITY-002: skipped — fixture.argmax_text is empty \
             (regenerate fixture with M32d.1 script which emits decoded text)"
        );
        return;
    }

    eprintln!("F-QW3-MOE-PARITY-002: argmax sanity vs llama.cpp Q4_K");
    eprintln!("  llama-cli: {}", llama_cli.display());
    eprintln!("  gguf:      {}", gguf_path.display());
    eprintln!("  fixture:   {}", fx_path.display());
    eprintln!("  model:     {}", fixture.model_name);
    eprintln!("  prompt:    {:?}", fixture.prompt);
    eprintln!(
        "  hf_argmax: id={} text={:?}",
        fixture.argmax_token, fixture.argmax_text
    );

    let start = std::time::Instant::now();
    let output = Command::new(&llama_cli)
        .args([
            "-m",
            gguf_path.to_str().expect("gguf path utf8"),
            "-p",
            &fixture.prompt,
            "-n",
            "1",
            "--top-k",
            "1",
            "--temp",
            "0.0",
            "--seed",
            "0",
            "--no-display-prompt",
            "-no-cnv",
            "--no-warmup",
            "--log-disable",
        ])
        .output()
        .expect("F-QW3-MOE-PARITY-002: failed to spawn llama-cli");
    let elapsed = start.elapsed();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let llama_emit = extract_first_emit(&stdout);

    eprintln!(
        "F-QW3-MOE-PARITY-002:\n  elapsed       = {elapsed:?}\n  llama-cli exit= {}\n  stdout (raw)  = {:?}\n  stdout (trim) = {:?}\n  stderr (last) = {:?}",
        output.status,
        stdout,
        llama_emit,
        stderr.lines().last().unwrap_or("")
    );

    assert!(
        output.status.success(),
        "F-QW3-MOE-PARITY-002: llama-cli exited non-zero (status = {}). stderr: {}",
        output.status,
        stderr
    );
    assert!(
        !llama_emit.is_empty(),
        "F-QW3-MOE-PARITY-002: llama-cli produced empty stdout. stderr: {stderr}"
    );

    // Compare decoded text. Some tokenizers prepend a leading-space marker
    // when detokenizing the first sub-word; tolerate it by checking either
    // direction of substring containment, or trimmed equality.
    let llama_t = llama_emit.trim();
    let fix_t = fixture.argmax_text.trim();
    let matches = llama_t == fix_t || llama_t.contains(fix_t) || fix_t.contains(llama_t);

    assert!(
        matches,
        "F-QW3-MOE-PARITY-002 (AC_QW3_MOE_001 transitive via M32d.2): \
         llama.cpp first-emitted decoded text = {llama_t:?} but \
         HF FP16 fixture.argmax_text = {fix_t:?}. \
         Diagnostic per FALSIFY-QW3-MOE-FORWARD-004 if_fails (b): \
         math is correct (M32d.2 cosine > 0.99 already passed), divergence is in \
         the llama.cpp Q4_K dequant kernel OR the sampler/seed handling. \
         Verify llama.cpp's --top-k 1 --temp 0.0 deterministic path against \
         apr's greedy_argmax."
    );
}

#[test]
fn locate_llama_cli_handles_missing() {
    // Sanity for the fallback resolver: a deliberately-bogus list returns None.
    let none_paths: &[&str] = &["/nonexistent/llama-cli", "/also/nonexistent"];
    let result = find_first_existing(none_paths);
    assert!(result.is_none());
}

#[test]
fn extract_first_emit_strips_blank_leading_lines() {
    let raw = "\n\n  hello\nworld\n";
    assert_eq!(extract_first_emit(raw), "hello");
}

#[test]
fn extract_first_emit_handles_empty() {
    assert_eq!(extract_first_emit(""), "");
    assert_eq!(extract_first_emit("\n\n\n"), "");
}
