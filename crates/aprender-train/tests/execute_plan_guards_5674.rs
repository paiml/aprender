//! `execute_plan` refuses to start when a precondition fails, and says which (#5674).
//! These guards run before any weight load, so they need no model or GPU.

use entrenar::finetune::{
    execute_plan, training_plan, ApplyConfig, PlanConfig, PlanVerdict, TrainingPlan,
};
use entrenar::Error;
use std::path::Path;

fn make_plan(dir: &Path) -> TrainingPlan {
    let data = dir.join("train.jsonl");
    std::fs::write(
        &data,
        "{\"input\":\"echo hi\",\"label\":0}\n{\"input\":\"rm -rf /\",\"label\":1}\n",
    )
    .expect("write corpus");
    let cfg = PlanConfig {
        task: "classify".to_string(),
        data_path: data,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: dir.join("out"),
        strategy: "manual".to_string(),
        budget: 1,
        scout: true,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(4),
        manual_batch_size: Some(2),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    training_plan(&cfg).expect("plan builds from a valid corpus")
}

fn apply(model: &Path, data: &Path, out: &Path) -> ApplyConfig {
    ApplyConfig {
        model_path: model.to_path_buf(),
        data_path: data.to_path_buf(),
        output_dir: out.to_path_buf(),
        on_trial_complete: None,
    }
}

fn config_msg(e: Error) -> String {
    match e {
        Error::ConfigError(m) | Error::Io(m) => m,
        other => format!("unexpected variant: {other}"),
    }
}

#[test]
fn blocked_plan_is_refused_before_touching_the_filesystem() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut plan = make_plan(tmp.path());
    plan.verdict = PlanVerdict::Blocked;
    let out = tmp.path().join("apply-out");
    let a = apply(tmp.path(), &tmp.path().join("train.jsonl"), &out);
    let err = execute_plan(&plan, &a).expect_err("blocked plan must not run");
    assert!(config_msg(err).contains("blocked plan"));
    assert!(!out.exists(), "no output dir for a refused plan");
}

#[test]
fn missing_model_dir_is_a_config_error_naming_the_path() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let plan = make_plan(tmp.path());
    let missing = tmp.path().join("no-such-model");
    let a = apply(&missing, &tmp.path().join("train.jsonl"), &tmp.path().join("o"));
    let err = execute_plan(&plan, &a).expect_err("missing model dir");
    assert!(matches!(err, Error::ConfigError(_)));
    assert!(config_msg(execute_plan(&plan, &a).expect_err("again")).contains("no-such-model"));
}

#[test]
fn missing_training_data_is_an_io_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let plan = make_plan(tmp.path());
    let a = apply(tmp.path(), &tmp.path().join("gone.jsonl"), &tmp.path().join("o"));
    let err = execute_plan(&plan, &a).expect_err("missing data");
    assert!(matches!(err, Error::Io(_)), "data absence is Io, not Config");
    assert!(config_msg(err).contains("gone.jsonl"));
}

#[test]
fn unknown_model_size_is_rejected_not_silently_defaulted() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut plan = make_plan(tmp.path());
    plan.model.size = "not-a-size".to_string();
    let out = tmp.path().join("o");
    let a = apply(tmp.path(), &tmp.path().join("train.jsonl"), &out);
    let err = execute_plan(&plan, &a).expect_err("GH-377: unknown size must error");
    assert!(matches!(err, Error::ConfigError(_)));
    assert!(out.is_dir(), "output dir is created before the size check");
}
