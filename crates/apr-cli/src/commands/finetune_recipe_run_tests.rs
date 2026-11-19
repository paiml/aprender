//! PMAT-7239: the recipe seed reaches the trainer; `receipt.json` is written after a run.

use super::*;
use crate::commands::train_recipe::{apr_engine, sha256_bytes, FINETUNE_NOT_APPLIED};
use std::path::PathBuf;

fn recipe_text(seed: u64) -> String {
    format!(
        "recipe: apr-recipe-v1\nname: tiny\nbase: base.apr\nmethod: qlora\n\
         data:\n  train: train.jsonl\neval:\n  heldout: heldout.jsonl\nseed: {seed}\n\
         training:\n  rank: 4\n  epochs: 2\n  learning_rate: 0.0002\n  max_seq_len: 128\n"
    )
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("apr-ft-run-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("scratch dir");
    std::fs::write(dir.join("base.apr"), b"base-bytes").expect("base");
    std::fs::write(dir.join("train.jsonl"), b"{\"a\":1}\n").expect("train");
    std::fs::write(dir.join("heldout.jsonl"), b"{\"a\":2}\n").expect("heldout");
    dir
}

fn receipt_for(dir: &Path, seed: u64) -> Value {
    let path = dir.join("r.yaml");
    std::fs::write(&path, recipe_text(seed)).expect("recipe");
    let (recipe, base) = crate::commands::finetune::preflight_recipe(&path, false).expect("ok");
    let resolved = ResolvedRun {
        epochs: 2,
        quantize_nf4: true,
        model_size: Some("0.8B"),
    };
    build_receipt(&recipe, &base, &apr_engine(), &resolved)
}

#[test]
fn trainer_seed_defaults_to_42_and_follows_the_guard() {
    assert_eq!(run_seed(), 42);
    {
        let _g = SeedGuard::set(7);
        assert_eq!(run_seed(), 7);
        {
            let _inner = SeedGuard::set(9);
            assert_eq!(run_seed(), 9);
        }
        assert_eq!(run_seed(), 7, "inner guard restores the outer seed");
    }
    assert_eq!(run_seed(), 42, "dropping the guard restores the default");
}

#[test]
fn seed_zero_is_applied_not_mistaken_for_unset() {
    let _g = SeedGuard::set(0);
    assert_eq!(run_seed(), 0);
}

#[test]
fn receipt_carries_hashes_version_commit_and_hyperparameters() {
    let dir = scratch("shape");
    let v = receipt_for(&dir, 11);
    assert_eq!(v["schema"], FINETUNE_RECEIPT_SCHEMA);
    assert_eq!(v["recipe_sha256"], sha256_bytes(recipe_text(11).as_bytes()));
    assert_eq!(v["base_model_sha256"], sha256_bytes(b"base-bytes"));
    assert_eq!(v["dataset_sha256"], sha256_bytes(b"{\"a\":1}\n"));
    assert_eq!(v["eval_sha256"], sha256_bytes(b"{\"a\":2}\n"));
    assert_eq!(v["apr"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(!v["apr"]["commit"].as_str().unwrap_or("").is_empty());
    let h = &v["hyperparameters"];
    assert_eq!(h["seed"], 11);
    assert_eq!(h["method"], "qlora");
    assert_eq!(h["rank"], 4);
    assert_eq!(h["epochs"], 2);
    assert_eq!(h["learning_rate"], 0.0002);
    assert_eq!(h["max_seq_len"], 128);
    assert_eq!(h["quantize_nf4"], true);
    assert_eq!(h["model_size"], "0.8B");
    assert_eq!(v["validated_not_applied"][0], FINETUNE_NOT_APPLIED[0]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_same_recipe_gives_byte_identical_receipt_and_another_seed_does_not() {
    let dir = scratch("determinism");
    let a = receipt_for(&dir, 5);
    let b = receipt_for(&dir, 5);
    let c = receipt_for(&dir, 6);
    let pa = write_receipt(&dir.join("a"), &a).expect("a");
    let pb = write_receipt(&dir.join("b"), &b).expect("b");
    let pc = write_receipt(&dir.join("c"), &c).expect("c");
    let ha = sha256_file(&pa);
    assert!(ha.is_some());
    assert_eq!(ha, sha256_file(&pb));
    assert_ne!(ha, sha256_file(&pc));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_missing_base_hash_is_null_never_invented() {
    let dir = scratch("nullbase");
    std::fs::remove_file(dir.join("base.apr")).expect("rm base");
    let path = dir.join("r.yaml");
    std::fs::write(&path, recipe_text(1)).expect("recipe");
    let (recipe, base) = crate::commands::finetune::preflight_recipe(&path, true).expect("plan");
    let resolved = ResolvedRun {
        epochs: 1,
        quantize_nf4: false,
        model_size: None,
    };
    let v = build_receipt(&recipe, &base, &apr_engine(), &resolved);
    assert!(v["base_model_sha256"].is_null());
    assert!(v["hyperparameters"]["model_size"].is_null());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn receipt_dir_is_the_adapter_directory_else_current() {
    assert_eq!(
        receipt_dir(Some(Path::new("out/x/adapter.apr"))),
        PathBuf::from("out/x")
    );
    assert_eq!(
        receipt_dir(Some(Path::new("adapter.apr"))),
        PathBuf::from(".")
    );
    assert_eq!(receipt_dir(None), PathBuf::from("."));
}

#[test]
fn write_receipt_creates_the_directory_and_ends_with_a_newline() {
    let dir = scratch("write");
    let out = dir.join("deep/er");
    let path = write_receipt(&out, &json!({"k": 1})).expect("write");
    assert_eq!(path, out.join(RECEIPT_FILE));
    let text = std::fs::read_to_string(&path).expect("read");
    assert!(text.ends_with("}\n"), "{text}");
    std::fs::remove_dir_all(&dir).ok();
}
