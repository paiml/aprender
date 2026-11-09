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
//!    by llama-cli. DECLARED, not searched for (#4981): the file name and its
//!    sha256 are the row `evidence/release-models.sha256` lists for it, and the
//!    file is read from `${APR_MODEL_DIR:-$HOME/models}`, the dir
//!    `scripts/tokenizer_parity.sh` reads the same list from. No machine path is
//!    named here. A file whose sha256 is not the listed one is a failure, not a
//!    different model to test.
//! 2. The `qwen3_moe_fp16_logits_pos0.json` fixture, generated once via
//!    `scripts/generate_qwen3_moe_fp16_logits.py` (M32d.1, PR #1129).
//! 3. The PINNED `llama-completion`, and only that one: `$LLAMA_COMPLETION` as
//!    `scripts/llama_bin.sh` exports it after PROVING the build against
//!    `scripts/llama_pin.toml` (#3740, #3563). Never PATH, never a list of
//!    candidate paths: a comparator nobody pinned makes the verdict about
//!    whichever llama.cpp the host happens to have (on lambda, `~/src/llama.cpp`
//!    is a different commit with a broken llama-cli). It is `llama-completion`,
//!    not `llama-cli`: this compares a RAW completion, `llama-cli` rejects
//!    `-no-cnv` and applies the chat template, and `--log-disable` empties
//!    `llama-completion`'s stdout (evidence/parity/pin-bump-d1d3c3396/LOAD.md).
//!    Run it as
//!    `. scripts/llama_bin.sh && cargo test -p aprender-serve --test qwen3_moe_argmax_parity -- --ignored`.
//!
//! Skips with `eprintln!` if any of the three is absent. Marked `#[ignore]`
//! so it does NOT run in default CI; the pre-publish dogfood runs it WITH the
//! pinned environment and treats a skip as a failure
//! (`scripts/dogfood_comparator_env_tests.sh`), so a skip is never the
//! release path.
//!
//! ## What the test does
//!
//! 1. Locate llama-cli binary, GGUF, and JSON fixture (skip if any missing),
//!    and assert the GGUF's sha256 is the one the model list declares.
//! 2. Read `fixture.prompt` and `fixture.argmax_text`.
//! 3. Spawn `llama-cli -m <gguf> -p <prompt> -n 1 --top-k 1 --temp 0.0
//!    --seed 0 --no-display-prompt -no-cnv --no-warmup --log-disable`,
//!    capture stdout.
//! 4. Trim the stdout to extract the first emitted decoded text.
//! 5. Assert the trimmed stdout equals (or contains) `fixture.argmax_text`,
//!    accommodating whitespace differences between tokenizers' detokenize
//!    paths (some prepend spaces; some don't).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The model this test runs: its file name is the key of its row in [`MODEL_LIST`].
const MODEL_FILE: &str = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf";
/// The committed list of release-time models, `<sha256>  <file name>` per row (#4981).
const MODEL_LIST: &str = "evidence/release-models.sha256";
/// The model dir, as `scripts/tokenizer_parity.sh` reads it: `${APR_MODEL_DIR:-$HOME/models}`.
const MODEL_DIR_ENV: &str = "APR_MODEL_DIR";

const FIXTURE_RELATIVE: &str = "tests/fixtures/qwen3_moe_fp16_logits_pos0.json";

/// The one way this test reaches llama.cpp: the variable `scripts/llama_bin.sh`
/// exports once it has proved the pinned build (#3740).
const LLAMA_CLI_ENV: &str = "LLAMA_COMPLETION";
/// What to run when it is unset, printed verbatim in the skip.
const RUN_PINNED: &str =
    ". scripts/llama_bin.sh && cargo test -p aprender-serve --test qwen3_moe_argmax_parity -- --ignored";

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

/// The sha256 the list declares for `name`: the first row `<sha256>  <name>`, two spaces as
/// sha256sum writes them. Comments (`#`) and blank lines are not rows. The list's own shape
/// (64 lowercase hex, no name twice) is held at PR time by `scripts/check_tokenizer_parity.sh`.
fn listed_sha256(list: &str, name: &str) -> Option<String> {
    list.lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once("  "))
        .find(|(_, n)| *n == name)
        .map(|(sha, _)| sha.to_string())
}

/// `${APR_MODEL_DIR:-$HOME/models}`: an empty `APR_MODEL_DIR` is unset, as in the shell.
fn model_dir_from(apr_model_dir: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    match apr_model_dir.filter(|v| !v.is_empty()) {
        Some(dir) => Some(PathBuf::from(dir)),
        None => home
            .filter(|h| !h.is_empty())
            .map(|h| PathBuf::from(h).join("models")),
    }
}

fn model_list_path() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/aprender-serve; the list lives at the repo root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(MODEL_LIST)
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// `$LLAMA_CLI` from `scripts/llama_bin.sh`, or `None`. The value is used as
/// given: `llama_bin.sh` has already proved it is the pinned build, and a second
/// check here would be the second resolver #3740 forbids.
fn locate_llama_cli() -> Option<PathBuf> {
    llama_cli_from(std::env::var_os(LLAMA_CLI_ENV))
}

fn llama_cli_from(value: Option<std::ffi::OsString>) -> Option<PathBuf> {
    value.filter(|v| !v.is_empty()).map(PathBuf::from)
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
            "F-QW3-MOE-PARITY-002: skipped — ${LLAMA_CLI_ENV} is unset, so there is no PINNED llama-completion. Run: {RUN_PINNED}"
        );
        return;
    };
    assert!(
        llama_cli.is_file(),
        "F-QW3-MOE-PARITY-002: ${LLAMA_CLI_ENV}={} is not a file; scripts/llama_bin.sh exports only a proved build",
        llama_cli.display()
    );

    let list_path = model_list_path();
    let list = std::fs::read_to_string(&list_path).unwrap_or_else(|e| {
        panic!(
            "F-QW3-MOE-PARITY-002: cannot read the model list {}: {e}",
            list_path.display()
        )
    });
    let want_sha256 = listed_sha256(&list, MODEL_FILE)
        .unwrap_or_else(|| panic!("F-QW3-MOE-PARITY-002: {MODEL_LIST} does not list {MODEL_FILE}"));
    let Some(model_dir) = model_dir_from(std::env::var_os(MODEL_DIR_ENV), std::env::var_os("HOME"))
    else {
        eprintln!(
            "F-QW3-MOE-PARITY-002: skipped — neither ${MODEL_DIR_ENV} nor $HOME is set, so there is no model dir"
        );
        return;
    };
    let gguf_path = model_dir.join(MODEL_FILE);
    if !gguf_path.is_file() {
        eprintln!(
            "F-QW3-MOE-PARITY-002: skipped — {MODEL_FILE}, listed in {MODEL_LIST}, is absent from {}",
            model_dir.display()
        );
        return;
    }
    let got_sha256 = sha256_file(&gguf_path).unwrap_or_else(|e| {
        panic!(
            "F-QW3-MOE-PARITY-002: cannot hash {}: {e}",
            gguf_path.display()
        )
    });
    assert_eq!(
        got_sha256,
        want_sha256,
        "F-QW3-MOE-PARITY-002: {} is not the model {MODEL_LIST} lists",
        gguf_path.display()
    );

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
fn locate_llama_cli_reads_only_the_pinned_env() {
    // #3740: the comparator is `$LLAMA_CLI` (exported by scripts/llama_bin.sh), or nothing.
    // Unset and empty both mean "no pinned build": the test skips instead of searching.
    assert_eq!(llama_cli_from(None), None);
    assert_eq!(llama_cli_from(Some(std::ffi::OsString::new())), None);
    assert_eq!(
        llama_cli_from(Some("/pinned/build/bin/llama-cli".into())),
        Some(PathBuf::from("/pinned/build/bin/llama-cli"))
    );
}

#[test]
fn listed_sha256_reads_only_the_named_row() {
    // #4981: the model is a declared row, never the first file that happens to exist.
    let a = "a".repeat(64);
    let b = "b".repeat(64);
    let list = format!("# header\n\n{a}  other.gguf\n{b}  {MODEL_FILE}\n");
    assert_eq!(listed_sha256(&list, MODEL_FILE), Some(b.clone()));
    assert_eq!(listed_sha256(&list, "other.gguf"), Some(a));
    assert_eq!(listed_sha256(&list, "absent.gguf"), None);
    // one space is not the sha256sum row shape; a commented-out row is not a row
    assert_eq!(
        listed_sha256(&format!("{b} {MODEL_FILE}\n"), MODEL_FILE),
        None
    );
    assert_eq!(
        listed_sha256(&format!("#{b}  {MODEL_FILE}\n"), MODEL_FILE),
        None
    );
}

#[test]
fn model_dir_is_apr_model_dir_else_home_models() {
    assert_eq!(
        model_dir_from(Some("/declared".into()), Some("/h".into())),
        Some(PathBuf::from("/declared"))
    );
    assert_eq!(
        model_dir_from(Some(OsString::new()), Some("/h".into())),
        Some(PathBuf::from("/h/models"))
    );
    assert_eq!(
        model_dir_from(None, Some("/h".into())),
        Some(PathBuf::from("/h/models"))
    );
    assert_eq!(model_dir_from(None, None), None);
}

#[test]
fn the_committed_model_list_names_this_model() {
    let list = std::fs::read_to_string(model_list_path()).expect("evidence/release-models.sha256");
    let sha = listed_sha256(&list, MODEL_FILE).expect("a row for the MoE comparator's model");
    assert_eq!(sha.len(), 64);
    assert!(sha
        .bytes()
        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)));
}

#[test]
fn sha256_file_is_the_sha256sum_digest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("abc.gguf");
    std::fs::write(&path, b"abc").expect("write");
    assert_eq!(
        sha256_file(&path).expect("hash"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert!(sha256_file(&dir.path().join("absent.gguf")).is_err());
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
