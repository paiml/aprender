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

#[test]
fn trainer_config_seed_equals_the_recipe_seed_zero_included() {
    use crate::commands::finetune::instruct_training_config;
    let out = Path::new("out/adapter.apr");
    for seed in [0u64, 1, 42, 7777] {
        let _g = SeedGuard::set(seed);
        assert_eq!(instruct_training_config(2, 10, out).seed, seed);
    }
    assert_eq!(instruct_training_config(2, 10, out).seed, DEFAULT_SEED);
}

#[test]
fn recipe_run_threads_the_recipe_seed_into_the_trainer_config() {
    // The same lowering `run_from_recipe` uses: the guard is set from `recipe.seed`.
    let dir = scratch("thread");
    let path = dir.join("r.yaml");
    std::fs::write(&path, recipe_text(0)).expect("recipe");
    let (recipe, _) = crate::commands::finetune::preflight_recipe(&path, false).expect("ok");
    let _g = SeedGuard::set(recipe.seed);
    let cfg = crate::commands::finetune::instruct_training_config(2, 10, Path::new("o/a.apr"));
    assert_eq!(cfg.seed, 0);
    std::fs::remove_dir_all(&dir).ok();
}

/// Train the smallest in-test fixture (tiny transformer, 12 samples, 2 epochs) on the CPU and
/// return the per-epoch losses plus the sha256 of every `model.safetensors` checkpoint written.
fn tiny_train(seed: u64, tag: &str) -> (Vec<(f32, f32)>, Vec<(String, String)>) {
    use entrenar::finetune::instruct_corpus::InstructSample;
    use entrenar::finetune::instruct_pipeline::{InstructConfig, InstructPipeline};
    use entrenar::finetune::instruct_trainer::InstructTrainer;
    use entrenar::transformer::TransformerConfig;
    let corpus: Vec<InstructSample> = (0..12)
        .map(|i| InstructSample {
            instruction: format!("Write function {i}"),
            response: format!("def func_{i}():\n    return {i}"),
            system: None,
            metadata: None,
        })
        .collect();
    let icfg = InstructConfig {
        lora_rank: 4,
        max_seq_len: 32,
        ..InstructConfig::default()
    };
    // The base weights come from a process-global init seed that the train-config loader tests
    // also set, without taking this lock. Hold the lock and rebuild until the seed read back
    // after construction is still the one asked for, so a racing setter cannot reach the fixture.
    let _init = entrenar::transformer::init::lock_init_seed(42);
    let pipeline = loop {
        entrenar::transformer::init::set_init_seed(42);
        let p = InstructPipeline::new(&TransformerConfig::tiny(), icfg.clone());
        if entrenar::transformer::init::get_init_seed() == 42 {
            break p;
        }
    };
    let _g = SeedGuard::set(seed);
    let ckpt = std::env::temp_dir().join(format!("apr-ft-det-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&ckpt).ok();
    let mut cfg =
        crate::commands::finetune::instruct_training_config(2, corpus.len(), Path::new("o/a.apr"));
    cfg.checkpoint_dir = ckpt.clone();
    let mut trainer = InstructTrainer::new(pipeline, corpus, cfg).expect("trainer");
    let result = trainer.train();
    let mut hashes = Vec::new();
    let mut stack = vec![ckpt.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "model.safetensors") {
                let rel = p
                    .strip_prefix(&ckpt)
                    .expect("under ckpt")
                    .display()
                    .to_string();
                hashes.push((rel, sha256_bytes(&std::fs::read(&p).expect("read ckpt"))));
            }
        }
    }
    hashes.sort();
    std::fs::remove_dir_all(&ckpt).ok();
    (
        result
            .epoch_metrics
            .iter()
            .map(|m| (m.train_loss, m.val_loss))
            .collect(),
        hashes,
    )
}

/// Real CPU run on the smallest in-test fixture. Measured limit, not hidden: two same-seed runs
/// of this fixture write checkpoints that alternate between two distinct sha256 values (epoch 0
/// included), so a byte-identical checkpoint claim is NOT made here. The seed is proven to reach
/// the trainer by the config tests above; this test proves the run completes end to end on CPU
/// with the recipe seed in force, writes a checkpoint per epoch, and gives identical losses.
#[test]
fn same_seed_cpu_run_completes_with_identical_losses_and_writes_checkpoints() {
    let (la, ha) = tiny_train(5, "a");
    let (lb, _) = tiny_train(5, "b");
    assert_eq!(la.len(), 2, "two epochs ran");
    assert!(
        ha.iter().any(|(p, _)| p.starts_with("epoch-0")),
        "epoch-0 checkpoint written"
    );
    assert!(
        ha.iter().any(|(p, _)| p.starts_with("epoch-1")),
        "epoch-1 checkpoint written"
    );
    assert_eq!(la, lb, "same seed: identical per-epoch losses");
}

#[test]
fn wgpu_run_keeps_seed_under_validated_not_applied() {
    let dir = scratch("wgpu");
    let path = dir.join("r.yaml");
    std::fs::write(&path, recipe_text(3)).expect("recipe");
    let mut flags = crate::commands::finetune::RecipeRunFlags {
        vram_gb: 24.0,
        plan_only: false,
        output_path: None,
        model_size: None,
        checkpoint_format: "safetensors",
        gpus: None,
        gpu_backend: "wgpu",
        wait_gpu: 0,
        json_output: false,
        experimental_mps: false,
        gpu_share: 0,
    };
    let (_, receipt) =
        crate::commands::finetune::recipe_receipt_for_run(&path, &flags).expect("receipt");
    let has_seed = receipt.validated_not_applied.contains(&"seed");
    assert_eq!(has_seed, cfg!(feature = "wgpu"));
    flags.gpu_backend = "cuda";
    let (_, receipt) =
        crate::commands::finetune::recipe_receipt_for_run(&path, &flags).expect("receipt");
    assert!(!receipt.validated_not_applied.contains(&"seed"));
    std::fs::remove_dir_all(&dir).ok();
}
