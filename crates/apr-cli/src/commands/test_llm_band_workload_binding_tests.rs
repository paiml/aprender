//! PMAT-973 / #2756 — `--workload` is a free-text label decoupled from the
//! prompts actually sent: `apr test llm bench --band --workload W1 --profile
//! short` sends ONE prompt 30 times and the receipt records `"workload":
//! "W1"` regardless — the labelled 256-prompt corpus never entered the
//! picture. `prompts-w1.jsonl`'s own `_meta.distinctness_rationale` says
//! identical prompts let prefix caching, not the scheduler, drive the number;
//! a label the receipt cannot falsify is not provenance.
//!
//! These tests pin `bind_workload`, the pure function that ties the label to
//! the bytes actually loaded (§ DAG row I-25): a claimed corpus is accepted
//! only when the loaded file's own `_meta.corpus` agrees, and the receipt
//! carries the sha256 of exactly those bytes as `corpus_sha256`.

use super::*;
use std::io::Write as _;

/// `crates/apr-cli`'s `CARGO_MANIFEST_DIR` sibling to
/// `crates/aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl`.
fn w1_corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl")
}

/// A private, per-process scratch file — never the real corpus, never
/// touched by anything else in the suite.
fn write_tmp(name: &str, content: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "pmat973-workload-binding-{}-{name}",
        std::process::id()
    ));
    let mut f = std::fs::File::create(&path).expect("create tmp corpus");
    f.write_all(content.as_bytes()).expect("write tmp corpus");
    path
}

#[test]
fn the_real_w1_corpus_binds_to_the_w1_label() {
    let path = w1_corpus_path();
    let binding = bind_workload(Workload::W1, &path)
        .unwrap_or_else(|e| panic!("the real W1 corpus must bind to the W1 label: {e}"));
    assert_eq!(binding.label, Workload::W1);
    assert_eq!(
        binding.prompt_count, 256,
        "prompts-w1.jsonl's own _meta.count is 256"
    );
    assert_eq!(binding.corpus_sha256.len(), 64, "{}", binding.corpus_sha256);
    assert!(
        binding
            .corpus_sha256
            .bytes()
            .all(|b: u8| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "{}",
        binding.corpus_sha256
    );
    let expected = sha256_file(&path).unwrap_or_else(|e| panic!("hashing {}: {e}", path.display()));
    assert_eq!(
        binding.corpus_sha256, expected,
        "corpus_sha256 must be the digest of exactly the bytes loaded"
    );
}

#[test]
fn a_corpus_labelled_w2_refuses_the_w1_label() {
    let path = write_tmp(
        "w2.jsonl",
        "{\"_meta\":{\"corpus\":\"W2\",\"count\":1}}\n{\"model\":\"m\",\"messages\":[]}\n",
    );
    let err = bind_workload(Workload::W1, &path)
        .expect_err("a file labelled W2 must refuse the W1 label");
    let msg = err.to_string();
    let _ = std::fs::remove_file(&path);
    assert!(msg.contains("W1"), "{msg}");
    assert!(msg.contains("W2"), "{msg}");
}

#[test]
fn a_prompt_set_with_no_meta_header_is_not_a_labelled_corpus() {
    // The `--profile short` shape: one prompt, no `_meta` header at all.
    let path = write_tmp("noeta.jsonl", "{\"model\":\"m\",\"messages\":[]}\n");
    let err = bind_workload(Workload::W1, &path)
        .expect_err("a prompt set with no _meta header is not a labelled corpus");
    let msg = err.to_string();
    let _ = std::fs::remove_file(&path);
    assert!(
        msg.to_lowercase().contains("not a labelled corpus"),
        "{msg}"
    );
}

#[test]
fn a_receipt_refuses_a_corpus_label_with_no_corpus_sha256() {
    assert!(
        !receipt_accepts_workload(Workload::W1, None),
        "a corpus label with no corpus_sha256 is a declaration, not provenance"
    );
    let digest = "a".repeat(64);
    assert!(
        receipt_accepts_workload(Workload::W1, Some(digest.as_str())),
        "a corpus label WITH a corpus_sha256 is accepted"
    );
}
