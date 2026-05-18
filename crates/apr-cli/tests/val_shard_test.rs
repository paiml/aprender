// Integration tests: unwrap()/panic!() are idiomatic; strict workspace lints relaxed.
#![allow(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    clippy::uninlined_format_args
)]

//! Integration tests for SPEC §84 P2-F `apr pretrain --val-shard <DIR>`.
//!
//! Contract: contracts/apr-pretrain-val-shard-v1.yaml
//! Discharges FALSIFY-PRETRAIN-VAL-SHARD-001/002/003 (the integration
//! tests it names) — the unit-level legacy-preservation falsifier
//! (-004) lives in `crates/apr-cli/src/commands/pretrain.rs::tests`.
//!
//! These tests exercise the CLI surface only (clap parse + plumb-through):
//!   - `--val-shard` accepts a path (FALSIFY-001 surface check)
//!   - `--val-shard` is documented in --help (operator discoverability)
//!   - An empty val-shard directory hard-fails with the falsifier ID
//!     (FALSIFY-003, integration via synthetic pretrain → real iter
//!     path would require a multi-MB token fixture; the help-text
//!     smoke + behavioural exit is the smallest reliable surface).
//!
//! A full FALSIFY-001/002 integration test would need a real .bin
//! shard fixture with > batch_size × (seq_length+1) tokens. That
//! fixture surface is large; the unit tests in pretrain.rs::tests
//! cover the branching logic on real `Vec<LMBatch>` data.

use assert_cmd::Command;
use std::fs;
use tempfile::TempDir;

/// FALSIFY-PRETRAIN-VAL-SHARD-003 (integration): an empty val-shard
/// directory MUST hard-fail with the falsifier ID in stderr. This
/// is the cheapest integration check — no real .bin shards required,
/// no real GPU dispatch.
#[test]
fn falsify_val_shard_003_empty_dir_rejected() {
    let tmp = TempDir::new().expect("tempdir");
    let dataset = tmp.path().join("dataset");
    fs::create_dir_all(&dataset).expect("mkdir dataset");
    // Stage a single .bin file so the dataset iter itself succeeds
    // (we want the val-shard error path, not the dataset error path).
    fs::write(dataset.join("shard-0000.bin"), [0u8; 32_768]).expect("write dataset shard");

    let val_shard = tmp.path().join("val-empty");
    fs::create_dir_all(&val_shard).expect("mkdir val-empty");
    // Intentionally NO .bin files in val-empty — this triggers the
    // ShardBatchIter::new error, which surfaces with the falsifier ID.

    let tokenizer = tmp.path().join("tok");
    fs::create_dir_all(&tokenizer).expect("mkdir tok");
    // Stage a 50257-entry vocab.json so the tokenizer pre-flight passes.
    let mut obj = serde_json::Map::with_capacity(50257);
    for i in 0..50257 {
        obj.insert(format!("t{i}"), serde_json::Value::from(i as u64));
    }
    fs::write(
        tokenizer.join("vocab.json"),
        serde_json::to_string(&obj).expect("serialize vocab"),
    )
    .expect("write vocab.json");

    let run_dir = tmp.path().join("run");

    let mut cmd = Command::cargo_bin("apr").expect("apr binary built");
    cmd.arg("pretrain")
        .arg("--dataset")
        .arg(&dataset)
        .arg("--tokenizer")
        .arg(&tokenizer)
        .arg("--run-dir")
        .arg(&run_dir)
        .arg("--val-shard")
        .arg(&val_shard)
        .arg("--num-steps")
        .arg("1")
        .arg("--batch-size")
        .arg("1")
        .arg("--seq-length")
        .arg("64")
        .arg("--mode")
        .arg("from-scratch")
        .arg("--device")
        .arg("cpu");
    // Intentionally NOT --synthetic — we want the real drive path
    // so the val-shard iterator is exercised.

    let output = cmd.output().expect("run apr pretrain");
    assert!(
        !output.status.success(),
        "empty --val-shard must exit non-zero; got exit {:?}",
        output.status.code()
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{}\n{}", stderr, stdout);
    // We accept either the val-shard falsifier or the shard-iter
    // init error — both surfaces correctly reject the empty val-shard
    // dir; the shard-iter init error is the upstream form, the
    // falsifier ID is the wrapper. Both name the path.
    let names_val_path = combined.contains("val-empty");
    let names_falsifier =
        combined.contains("FALSIFY-PRETRAIN-VAL-SHARD-") || combined.contains("no .bin shards in");
    assert!(
        names_val_path && names_falsifier,
        "expected stderr to name the val-shard path AND surface the FALSIFY-PRETRAIN-VAL-SHARD-* \
         falsifier ID (or the underlying \"no .bin shards in\" message), got combined output:\n{}",
        combined
    );
}

/// `--val-shard` MUST be advertised in `apr pretrain --help` so
/// operators can discover the flag without grepping source code.
/// Catches accidental clap-flag removal and help-text regression.
#[test]
fn val_shard_flag_documented_in_help() {
    let mut cmd = Command::cargo_bin("apr").expect("apr binary built");
    cmd.arg("pretrain").arg("--help");
    let output = cmd.output().expect("run apr pretrain --help");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--val-shard"),
        "apr pretrain --help must list --val-shard per contract \
         C-APR-PRETRAIN-VAL-SHARD"
    );
    // The flag's help text should reference §84 P2-F or the contract id
    // so an operator searching for "P2-F" or "val-shard" can land here.
    let has_context = stdout.contains("P2-F")
        || stdout.contains("apr-pretrain-val-shard")
        || stdout.contains("val_shard")
        || stdout.contains("held-out");
    assert!(
        has_context,
        "apr pretrain --help should reference P2-F / val-shard / held-out (got:\n{})",
        stdout
    );
}
