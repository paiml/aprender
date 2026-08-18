//! `aprender-train-distill validate` must reject a bad config.
//!
//! This binary is a thin clap wrapper over `ConfigValidator::validate`, the
//! same validator `apr distill` reaches through `entrenar_distill::run`. Both
//! directions are pinned: a config with an empty `teacher.model_id` must exit
//! nonzero, and a well-formed one must exit zero. Asserting only the rejection
//! would not exclude "validate rejects everything", and asserting only the
//! acceptance would not exclude "validate accepts everything".

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Write `yaml` into the per-target tmpdir under `name` and return its path.
fn config(name: &str, yaml: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("validate_rejects_bad_config");
    fs::create_dir_all(&dir).expect("create tmpdir");
    let path = dir.join(name);
    fs::write(&path, yaml).expect("write config");
    path
}

/// Run `aprender-train-distill validate --config <path>`.
fn validate(path: &PathBuf) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aprender-train-distill"))
        .args(["validate", "--config", path.to_str().expect("utf-8 path")])
        .output()
        .expect("run aprender-train-distill")
}

/// Parses as YAML, but `teacher.model_id` is empty — a validator error, not a
/// deserialization error, so this exercises `ConfigValidator` rather than serde.
const EMPTY_TEACHER: &str = r#"
teacher:
  model_id: ""
student:
  model_id: "TinyLlama/TinyLlama-1.1B"
distillation: {}
training: {}
"#;

const WELL_FORMED: &str = r#"
teacher:
  model_id: "meta-llama/Llama-2-7b"
student:
  model_id: "TinyLlama/TinyLlama-1.1B"
distillation: {}
training: {}
"#;

#[test]
fn validate_rejects_an_empty_teacher_model_id() {
    let out = validate(&config("empty_teacher.yaml", EMPTY_TEACHER));
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        !out.status.success(),
        "validate exited 0 on a config with an empty teacher.model_id — the \
         validator accepts anything. stdout:\n{}\nstderr:\n{stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains("teacher.model_id"),
        "expected the diagnostic to name the offending field, got:\n{stderr}"
    );
}

#[test]
fn validate_accepts_a_well_formed_config() {
    let out = validate(&config("well_formed.yaml", WELL_FORMED));

    assert!(
        out.status.success(),
        "validate rejected a well-formed config — the validator rejects \
         everything. stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
