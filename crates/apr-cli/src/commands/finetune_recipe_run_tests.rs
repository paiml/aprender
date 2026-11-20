//! PMAT-7239: the recipe seed reaches the trainer; `receipt.json` is written after a run.

use super::*;
use crate::commands::train_recipe::{apr_engine, sha256_bytes};
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
        rank: 4,
        learning_rate: 0.0002,
        max_seq_len: 128,
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
    assert_eq!(
        v["validated_not_applied"],
        json!([]),
        "the trainer applies the eval: nothing is validated-only"
    );
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
        rank: 4,
        learning_rate: 0.0002,
        max_seq_len: 128,
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

fn run_flags<'a>(out: Option<&'a Path>, plan_only: bool, json: bool) -> RunFlags<'a> {
    crate::commands::finetune::RecipeRunFlags {
        vram_gb: 24.0,
        plan_only,
        output_path: out,
        model_size: None,
        checkpoint_format: "safetensors",
        gpus: None,
        gpu_backend: "cuda",
        wait_gpu: 0,
        json_output: json,
        experimental_mps: false,
        gpu_share: 0,
    }
}

type RunFlags<'a> = crate::commands::finetune::RecipeRunFlags<'a>;

/// Drive the real `run_recipe_with` seam (the code `run_from_recipe` runs) with an injected
/// trainer. Returns the result, what the trainer saw, and what the seam printed to stdout.
fn drive(
    dir: &Path,
    seed: u64,
    plan_only: bool,
    json: bool,
    trainer_ok: bool,
) -> (Result<()>, Vec<(u64, bool)>, String) {
    let path = dir.join("r.yaml");
    std::fs::write(&path, recipe_text(seed)).expect("recipe");
    let adapter = dir.join("out/adapter.apr");
    let flags = run_flags(Some(&adapter), plan_only, json);
    let (recipe, receipt) =
        crate::commands::finetune::recipe_receipt_for_run(&path, &flags).expect("preflight");
    let mut seen = Vec::new();
    let mut out: Vec<u8> = Vec::new();
    let r = crate::commands::finetune::run_recipe_with(
        &recipe,
        &receipt,
        &flags,
        &mut out,
        |_, _, _| {
            seen.push((run_seed(), plan_json_suppressed()));
            if trainer_ok {
                Ok(())
            } else {
                Err(CliError::ValidationFailed("trainer failed".to_string()))
            }
        },
    );
    (r, seen, String::from_utf8(out).expect("utf8"))
}

#[test]
fn recipe_run_hands_the_trainer_the_recipe_seed_zero_included() {
    for seed in [0u64, 1, 7777] {
        let dir = scratch("seam-seed");
        let (r, seen, _) = drive(&dir, seed, false, false, true);
        r.expect("run ok");
        assert_eq!(
            seen,
            vec![(seed, false)],
            "trainer ran under the recipe seed"
        );
        assert_eq!(run_seed(), DEFAULT_SEED, "guard released after the run");
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn recipe_run_writes_receipt_only_after_a_successful_real_run() {
    let dir = scratch("seam-receipt");
    let rpath = dir.join("out").join(RECEIPT_FILE);
    let (r, _, _) = drive(&dir, 3, false, false, false);
    assert!(r.is_err());
    assert!(!rpath.exists(), "a failed trainer leaves no receipt");
    let (r, _, _) = drive(&dir, 3, true, false, true);
    r.expect("plan ok");
    assert!(!rpath.exists(), "--plan leaves no receipt");
    let (r, _, _) = drive(&dir, 3, false, false, true);
    r.expect("run ok");
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(&rpath).expect("receipt")).expect("json");
    assert_eq!(v["hyperparameters"]["seed"], 3);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn json_recipe_run_prints_exactly_one_document_and_holds_back_the_plan() {
    let dir = scratch("seam-json");
    let (r, seen, out) = drive(&dir, 4, false, true, true);
    r.expect("run ok");
    assert_eq!(
        seen,
        vec![(4, true)],
        "plan JSON suppressed while the trainer runs"
    );
    let docs: Vec<Value> = serde_json::Deserializer::from_str(&out)
        .into_iter::<Value>()
        .collect::<std::result::Result<_, _>>()
        .expect("stdout is JSON");
    assert_eq!(docs.len(), 1, "exactly one document: {out}");
    assert_eq!(docs[0]["schema"], FINETUNE_RECEIPT_SCHEMA);
    let file: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("out").join(RECEIPT_FILE)).expect("receipt.json"),
    )
    .expect("receipt.json is JSON");
    assert_eq!(docs[0], file, "the --json document equals receipt.json");
    assert!(!plan_json_suppressed(), "guard released");
    let before = std::fs::read(dir.join("out").join(RECEIPT_FILE)).expect("receipt.json");
    let (_, seen, out) = drive(&dir, 4, true, true, true);
    assert_eq!(seen, vec![(4, false)], "--plan keeps its plan JSON");
    // The contract (recipe_dry_run) prints the resolved recipe on stdout under --plan; what
    // --plan never does is print or rewrite a receipt.
    let plan_doc: Value = serde_json::from_str(&out).expect("--plan stdout is the resolved recipe");
    assert_ne!(
        plan_doc["schema"], FINETUNE_RECEIPT_SCHEMA,
        "--plan prints no receipt"
    );
    let after = std::fs::read(dir.join("out").join(RECEIPT_FILE)).expect("receipt.json");
    assert_eq!(before, after, "--plan does not rewrite receipt.json");
    std::fs::remove_dir_all(&dir).ok();
}

/// One checkpoint file split the way a reader sees it: the header JSON parsed (key order
/// dropped), the sha256 of the tensor bytes after the header, and the header as written.
struct Ckpt {
    rel: String,
    raw_sha: String,
    header: Value,
    header_text: String,
    data_sha: String,
}

fn split_safetensors(rel: String, bytes: &[u8]) -> Ckpt {
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let header_text = String::from_utf8(bytes[8..8 + n].to_vec()).expect("utf8 header");
    Ckpt {
        rel,
        raw_sha: sha256_bytes(bytes),
        header: serde_json::from_str(&header_text).expect("header json"),
        header_text,
        data_sha: sha256_bytes(&bytes[8 + n..]),
    }
}

/// Run the recipe seam (`run_recipe_with`, the code `run_from_recipe` runs) with a trainer that
/// really trains the smallest fixture (tiny transformer, 6 samples, 3 heldout samples) on the
/// CPU. The trainer is built by `build_instruct_trainer` and its config by
/// `instruct_training_config`, the functions the real `apr finetune` run uses, so the seed and
/// the heldout file reach it only through the recipe. Returns the per-epoch losses, the train and
/// validation set sizes, and every `model.safetensors` checkpoint the run wrote.
fn tiny_train(seed: u64, tag: &str) -> (Vec<(f32, f32)>, (usize, usize), Vec<Ckpt>) {
    use entrenar::finetune::instruct_corpus::InstructSample;
    use entrenar::finetune::instruct_pipeline::{InstructConfig, InstructPipeline};
    use entrenar::transformer::TransformerConfig;
    let dir = scratch(&format!("det-{tag}"));
    let line = |i: usize| {
        format!("{{\"instruction\":\"Write function {i}\",\"response\":\"def func_{i}():\\n    return {i}\",\"system\":\"s\"}}\n")
    };
    std::fs::write(
        dir.join("train.jsonl"),
        (0..6).map(line).collect::<String>(),
    )
    .expect("train");
    std::fs::write(
        dir.join("heldout.jsonl"),
        (10..13).map(line).collect::<String>(),
    )
    .expect("heldout");
    let path = dir.join("r.yaml");
    std::fs::write(&path, recipe_text(seed)).expect("recipe");
    let adapter = dir.join("out/adapter.apr");
    let flags = run_flags(Some(&adapter), false, false);
    let (recipe, receipt) =
        crate::commands::finetune::recipe_receipt_for_run(&path, &flags).expect("preflight");
    let icfg = InstructConfig {
        lora_rank: 4,
        max_seq_len: 128,
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
    let mut losses = Vec::new();
    let mut sizes = (0, 0);
    let mut out: Vec<u8> = Vec::new();
    crate::commands::finetune::run_recipe_with(
        &recipe,
        &receipt,
        &flags,
        &mut out,
        |output, epochs, _| {
            let corpus: Vec<InstructSample> =
                crate::commands::finetune::load_instruct_corpus(&recipe.train)?;
            let cfg = crate::commands::finetune::instruct_training_config(
                epochs,
                corpus.len(),
                output.expect("recipe output path"),
            );
            let mut trainer =
                crate::commands::finetune::build_instruct_trainer(pipeline, corpus, cfg)?;
            sizes = (trainer.train_size(), trainer.val_size());
            let result = trainer.train();
            losses = result
                .epoch_metrics
                .iter()
                .map(|m| (m.train_loss, m.val_loss))
                .collect();
            Ok(())
        },
    )
    .expect("recipe run");
    let ckpt = dir.join("out/checkpoints");
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
                hashes.push(split_safetensors(
                    rel,
                    &std::fs::read(&p).expect("read ckpt"),
                ));
            }
        }
    }
    hashes.sort_by(|a, b| a.rel.cmp(&b.rel));
    std::fs::remove_dir_all(&dir).ok();
    (losses, sizes, hashes)
}

/// Real CPU training driven through the recipe run path (#7239 acceptance 1 and 2, #7255).
///
/// Same seed: identical per-epoch losses and a byte-identical checkpoint FILE (sha256 of the
/// whole `model.safetensors`, header included) for every epoch. Another seed: different losses
/// and different file sha256. The fixture's loss is non-zero (the chat template alone is ~100
/// bytes under the byte fallback, so `max_seq_len` must leave room for the response, or every
/// step returns loss 0.0 and nothing depends on the seed).
#[test]
fn same_seed_recipe_run_writes_identical_checkpoint_files_and_another_seed_differs() {
    let (la, sa, ha) = tiny_train(5, "a");
    let (lb, _, hb) = tiny_train(5, "b");
    let (lc, _, hc) = tiny_train(6, "c");
    assert_eq!(la.len(), 2, "two epochs ran");
    assert!(
        la.iter().all(|(t, v)| *t > 0.0 && *v > 0.0),
        "non-zero losses: {la:?}"
    );
    assert_eq!(
        sa,
        (6, 3),
        "train on the whole corpus, validate on the heldout file"
    );
    assert!(
        ha.iter().any(|c| c.rel.starts_with("epoch-0")),
        "epoch-0 checkpoint written"
    );
    assert!(
        ha.iter().any(|c| c.rel.starts_with("epoch-1")),
        "epoch-1 checkpoint written"
    );
    assert_eq!(la, lb, "same seed: identical per-epoch losses");
    assert_eq!(ha.len(), hb.len());
    for (x, y) in ha.iter().zip(&hb) {
        assert_eq!(x.rel, y.rel);
        assert_eq!(
            x.raw_sha, y.raw_sha,
            "same seed: identical checkpoint file sha256 for {} ({} vs {})",
            x.rel, x.header_text, y.header_text
        );
    }
    assert_ne!(la, lc, "another seed: different losses");
    assert!(
        ha.iter().zip(&hc).any(|(x, z)| x.raw_sha != z.raw_sha),
        "another seed: different checkpoint file sha256"
    );
}

/// `eval.heldout` reaches the trainer: it validates on that file and trains on all of the corpus,
/// where a flag-driven run splits the corpus.
#[test]
fn recipe_heldout_file_is_the_trainers_validation_set() {
    use entrenar::finetune::instruct_pipeline::{InstructConfig, InstructPipeline};
    use entrenar::transformer::TransformerConfig;
    let dir = scratch("heldout-seam");
    let sample = |i: usize| format!("{{\"instruction\":\"q{i}\",\"response\":\"a{i}\"}}\n");
    std::fs::write(
        dir.join("train.jsonl"),
        (0..8).map(sample).collect::<String>(),
    )
    .expect("t");
    std::fs::write(
        dir.join("heldout.jsonl"),
        (20..25).map(sample).collect::<String>(),
    )
    .expect("h");
    let path = dir.join("r.yaml");
    std::fs::write(&path, recipe_text(1)).expect("recipe");
    let adapter = dir.join("out/adapter.apr");
    let flags = run_flags(Some(&adapter), false, false);
    let (recipe, receipt) =
        crate::commands::finetune::recipe_receipt_for_run(&path, &flags).expect("preflight");
    let mk = || {
        InstructPipeline::new(
            &TransformerConfig::tiny(),
            InstructConfig {
                lora_rank: 4,
                max_seq_len: 32,
                ..InstructConfig::default()
            },
        )
    };
    let corpus = || crate::commands::finetune::load_instruct_corpus(&recipe.train).expect("corpus");
    let cfg =
        || crate::commands::finetune::instruct_training_config(1, 8, Path::new("out/adapter.apr"));
    let mut sizes = (0, 0);
    let mut out: Vec<u8> = Vec::new();
    crate::commands::finetune::run_recipe_with(&recipe, &receipt, &flags, &mut out, |_, _, _| {
        let t = crate::commands::finetune::build_instruct_trainer(mk(), corpus(), cfg())?;
        sizes = (t.train_size(), t.val_size());
        Ok(())
    })
    .expect("recipe run");
    assert_eq!(sizes, (8, 5), "recipe run: all train, heldout validates");
    assert!(run_heldout().is_none(), "guard released after the run");
    let t = crate::commands::finetune::build_instruct_trainer(mk(), corpus(), cfg()).expect("t");
    assert_ne!(
        (t.train_size(), t.val_size()),
        (8, 5),
        "a run outside a recipe splits the corpus"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// #7239 acceptance 4: no "not yet applied" on the recipe path, and the contract does not
/// claim the eval is unapplied.
#[test]
fn recipe_path_no_longer_says_not_yet_applied() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = [
        root.join("src/commands/finetune.rs"),
        root.join("src/commands/finetune_recipe_run.rs"),
        root.join("src/commands/train_recipe.rs"),
        root.join("../../contracts/apr-recipe-v1.yaml"),
    ];
    let needle = ["not yet", "applied"].join(" ");
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        assert!(
            !text.contains(&needle),
            "{} still says `{needle}`",
            f.display()
        );
        assert!(
            !text.contains("FINETUNE_NOT_APPLIED"),
            "{} still has the marker",
            f.display()
        );
    }
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
    assert_eq!(
        receipt.validated_not_applied.contains(&"eval"),
        cfg!(feature = "wgpu"),
        "the WGSL path reads no heldout file"
    );
    assert_eq!(
        receipt.validated_not_applied.contains(&"training.warmup"),
        cfg!(feature = "wgpu"),
        "the WGSL path reads no training.warmup"
    );
    flags.gpu_backend = "cuda";
    let (_, receipt) =
        crate::commands::finetune::recipe_receipt_for_run(&path, &flags).expect("receipt");
    assert!(receipt.validated_not_applied.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn receipt_records_the_values_the_run_used_when_the_recipe_omits_them() {
    let dir = scratch("resolved-hparams");
    let path = dir.join("r.yaml");
    std::fs::write(
        &path,
        "recipe: apr-recipe-v1\nname: tiny\nbase: base.apr\nmethod: qlora\n\
         model_size: 0.8B\ndata:\n  train: train.jsonl\neval:\n  heldout: heldout.jsonl\n\
         seed: 1\ntraining:\n  epochs: 2\n",
    )
    .expect("recipe");
    let adapter = dir.join("out/adapter.apr");
    let flags = run_flags(Some(&adapter), false, true);
    let (recipe, receipt) =
        crate::commands::finetune::recipe_receipt_for_run(&path, &flags).expect("preflight");
    assert!(
        recipe.rank.is_none() && recipe.learning_rate.is_none() && recipe.max_seq_len.is_none()
    );
    let mut out: Vec<u8> = Vec::new();
    crate::commands::finetune::run_recipe_with(&recipe, &receipt, &flags, &mut out, |_, _, _| {
        Ok(())
    })
    .expect("run ok");
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("out").join(RECEIPT_FILE)).expect("receipt"),
    )
    .expect("json");
    let h = &v["hyperparameters"];
    assert!(
        h["rank"].as_u64().is_some_and(|r| r > 0),
        "rank resolved: {h}"
    );
    assert!(
        h["learning_rate"].as_f64().is_some_and(|l| l > 0.0),
        "lr resolved: {h}"
    );
    assert!(
        h["max_seq_len"].as_u64().is_some_and(|m| m > 0),
        "max_seq_len resolved: {h}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn wgpu_run_trains_at_the_recipe_max_seq_len_not_a_hardcoded_512() {
    assert_eq!(
        crate::commands::finetune::wgpu_max_seq_len(Some(128)),
        128,
        "the recipe value reaches the WGPU pipeline"
    );
    assert_eq!(
        crate::commands::finetune::wgpu_max_seq_len(None),
        512,
        "no recipe value: entrenar's default"
    );
    let src = include_str!("finetune.rs");
    let call = src
        .split("WgpuInstructPipeline::new(")
        .nth(1)
        .and_then(|rest| rest.split(");").next())
        .expect("WgpuInstructPipeline::new call");
    assert!(
        !call.contains("512"),
        "the WGPU pipeline must take max_seq_len, not a literal 512"
    );
}
