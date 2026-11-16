//! `apr finetune --recipe` (CRUX-001, PMAT-5316; contracts/apr-recipe-v1.yaml).

use super::*;
use crate::commands::train_recipe::{sha256_bytes, RecipeMethod};
use std::path::PathBuf;

const VALID: &str = "\
recipe: apr-recipe-v1
name: qwen-lora-demo
base: models/base.apr
method: qlora
data:
  train: data/train.jsonl
eval:
  heldout: data/heldout.jsonl
seed: 42
training:
  rank: 8
  epochs: 1
";

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("apr-ft-recipe-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn write_recipe(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("r.yaml");
    std::fs::write(&path, body).expect("write recipe");
    path
}

fn flags(plan_only: bool) -> RecipeRunFlags<'static> {
    RecipeRunFlags {
        vram_gb: 16.0,
        plan_only,
        output_path: None,
        model_size: Some("1B"),
        checkpoint_format: "apr,safetensors",
        gpus: None,
        gpu_backend: "auto",
        wait_gpu: 0,
        json_output: true,
        experimental_mps: false,
        gpu_share: 50,
    }
}

/// FALSIFY-APR-RECIPE-001 at the verb: no `eval` block → refused, the text
/// names the field, and the refusal is the recipe's, not a model loader's
/// (the base does not exist, so reaching the loader would say so instead).
#[test]
fn recipe_without_eval_is_refused_before_any_model_is_touched() {
    let dir = scratch("noeval");
    let body = VALID.replace("eval:\n  heldout: data/heldout.jsonl\n", "");
    let path = write_recipe(&dir, &body);
    for plan_only in [true, false] {
        let msg = run_from_recipe(&path, &flags(plan_only))
            .expect_err("a recipe with no eval must be refused")
            .to_string();
        assert!(msg.contains("field `eval` is required"), "{msg}");
        assert!(!msg.contains("base.apr"), "the model was reached: {msg}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The positive control: the same recipe WITH its eval block plans cleanly.
#[test]
fn valid_recipe_plans() {
    let dir = scratch("plan");
    let path = write_recipe(&dir, VALID);
    run_from_recipe(&path, &flags(true)).expect("a valid recipe must plan");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn preflight_resolves_the_recipe_and_builds_the_receipt() {
    let dir = scratch("preflight");
    let path = write_recipe(&dir, VALID);
    let (recipe, receipt) = preflight_recipe(&path, true).expect("plan preflight");
    assert_eq!(recipe.method, RecipeMethod::Qlora);
    assert_eq!(recipe.rank, Some(8));
    assert_eq!(recipe.base, dir.join("models/base.apr"));
    assert_eq!(receipt.recipe_sha256, sha256_bytes(VALID.as_bytes()));
    assert_eq!(receipt.data_sha256, None, "the data file is absent");
    // Plan mode still hashes the files that exist.
    std::fs::create_dir_all(dir.join("data")).expect("data dir");
    std::fs::write(dir.join("data/train.jsonl"), b"abc").expect("train file");
    let (_, receipt) = preflight_recipe(&path, true).expect("plan preflight");
    assert_eq!(receipt.data_sha256, Some(sha256_bytes(b"abc")));
    assert_eq!(receipt.eval_sha256, None, "the eval file is absent");
    assert_eq!(receipt.validated_not_applied, vec!["seed", "eval"]);
    assert_eq!(receipt.engines.len(), 1);
    assert!(receipt.overrides.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

/// A real run (not `--plan`) refuses missing inputs by recipe field, before
/// the model loader can fail on the base.
#[test]
fn real_run_refuses_missing_inputs_by_field() {
    let dir = scratch("inputs");
    let path = write_recipe(&dir, VALID);
    let msg = run_from_recipe(&path, &flags(false))
        .expect_err("inputs are absent")
        .to_string();
    for field in ["`base`", "`data.train`", "`eval.heldout`"] {
        assert!(
            msg.contains(&format!("field {field}: no such file")),
            "{msg}"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// `model_size` in the recipe plans without `--model-size` and without the
/// base on disk; the flag wins when both are given; a bad size names the field.
#[test]
fn recipe_model_size_drives_the_plan() {
    let dir = scratch("size");
    let sized = VALID.replace("method: qlora", "model_size: 0.8B\nmethod: qlora");
    let path = write_recipe(&dir, &sized);
    let mut f = flags(true);
    f.model_size = None;
    run_from_recipe(&path, &f).expect("the recipe's own size plans");

    // Without either size the planner needs the base file, which is absent.
    let path = write_recipe(&dir, VALID);
    assert!(run_from_recipe(&path, &f).is_err());

    let path = write_recipe(&dir, &sized.replace("0.8B", "huge"));
    let msg = run_from_recipe(&path, &flags(true))
        .expect_err("a bad size is refused")
        .to_string();
    assert!(
        msg.contains("field `model_size` is not a model size"),
        "{msg}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// PMAT-5316 review: a size that is not a positive finite number is refused
/// by field name instead of saturating to 0 or u64::MAX.
#[test]
fn recipe_model_size_must_be_positive_and_finite() {
    let dir = scratch("size-bad");
    for bad in ["0B", "-1B", "NaNB", "infB", "0M"] {
        let body = VALID.replace(
            "method: qlora",
            &format!("model_size: {bad}\nmethod: qlora"),
        );
        let path = write_recipe(&dir, &body);
        let msg = run_from_recipe(&path, &flags(true))
            .expect_err(bad)
            .to_string();
        assert!(msg.contains("field `model_size`"), "{bad}: {msg}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The `--model-size` flag wins over the recipe's `model_size`.
#[test]
fn model_size_flag_wins_over_the_recipe() {
    let dir = scratch("size-win");
    let sized = VALID.replace("method: qlora", "model_size: 0.8B\nmethod: qlora");
    let (recipe, _) = preflight_recipe(&write_recipe(&dir, &sized), true).expect("valid");
    let mut f = flags(true);
    f.model_size = Some("7B");
    assert_eq!(recipe_model_size(&f, &recipe), Some("7B"));
    f.model_size = None;
    assert_eq!(recipe_model_size(&f, &recipe), Some("0.8B"));
    std::fs::remove_dir_all(&dir).ok();
}

/// `output.dir` is a directory: the adapter is written inside it, so the
/// checkpoints (`<adapter parent>/checkpoints`) stay inside it too. An
/// explicit `--output` is a file and is used as given.
#[test]
fn recipe_output_dir_holds_the_adapter_and_its_checkpoints() {
    let dir = scratch("outdir");
    let body = format!("{VALID}output:\n  dir: out/run1\n");
    let (recipe, _) = preflight_recipe(&write_recipe(&dir, &body), true).expect("valid");
    let f = flags(true);
    let out = recipe_output_path(&f, &recipe).expect("an output");
    assert_eq!(out, dir.join("out/run1").join("adapter.apr"));
    assert_eq!(
        out.parent().unwrap().join("checkpoints"),
        dir.join("out/run1/checkpoints")
    );
    let mut f = flags(true);
    let given = Path::new("x/mine.apr");
    f.output_path = Some(given);
    assert_eq!(recipe_output_path(&f, &recipe).as_deref(), Some(given));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn distill_recipe_is_refused_by_finetune_naming_method() {
    let dir = scratch("distill");
    let body = VALID
        .replace("method: qlora", "method: distill")
        .replace("data:", "teacher:\n  model: models/teacher.apr\ndata:");
    let path = write_recipe(&dir, &body);
    // A real run refuses on the method too, before it looks for the files.
    for plan_only in [true, false] {
        let msg = run_from_recipe(&path, &flags(plan_only))
            .expect_err("apr finetune does not run distill")
            .to_string();
        assert!(msg.contains("field `method` is distill"), "{msg}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn absent_recipe_file_is_named() {
    let msg = run_from_recipe(Path::new("/nonexistent/r.yaml"), &flags(true))
        .expect_err("no such recipe")
        .to_string();
    assert!(
        msg.contains("cannot read recipe /nonexistent/r.yaml"),
        "{msg}"
    );
}

/// The size bounds, at the function: one parameter is the smallest size, a
/// size under one parameter is refused, and so is one that does not fit u64.
#[test]
fn model_size_bounds_are_one_parameter_and_u64() {
    assert_eq!(parse_model_size("0.000001M").expect("one parameter"), 1);
    assert_eq!(parse_model_size("0.5M").expect("half a million"), 500_000);
    assert_eq!(parse_model_size("7b").expect("lower case"), 7_000_000_000);
    // 2^64 parameters is the first size refused; 18446744073B is under it
    // (the nearest f64 to 18446744073e9 is 512 below it).
    assert_eq!(
        parse_model_size("18446744073B").expect("just under 2^64"),
        18_446_744_072_999_999_488
    );
    for bad in [
        "0.0000005M",
        "0B",
        "-1B",
        "NaNB",
        "infB",
        "18446744073.709551616B",
        "1e11B",
        "1e300B",
    ] {
        let msg = parse_model_size(bad).expect_err(bad).to_string();
        assert!(msg.contains("at least 1 parameter"), "{bad}: {msg}");
    }
}

/// What the recipe decides by rule: epochs fall back to 3 when the recipe has
/// none, and only `qlora` asks for the NF4-quantized base.
#[test]
fn recipe_epochs_default_and_nf4_follow_the_method() {
    let dir = scratch("lower");
    let load = |body: &str| {
        let path = write_recipe(&dir, body);
        crate::commands::train_recipe::TrainRecipe::load(&path).expect("valid recipe")
    };
    assert_eq!(recipe_epochs_and_nf4(&load(VALID)), (1, true));
    let lora = VALID.replace("method: qlora", "method: lora");
    assert_eq!(recipe_epochs_and_nf4(&load(&lora)), (1, false));
    let no_epochs = VALID.replace("  epochs: 1\n", "");
    assert_eq!(recipe_epochs_and_nf4(&load(&no_epochs)), (3, true));
    std::fs::remove_dir_all(&dir).ok();
}
