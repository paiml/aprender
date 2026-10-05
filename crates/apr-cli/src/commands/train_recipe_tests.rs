//! Falsification tests for contracts/apr-recipe-v1.yaml (PMAT-4838).

use super::*;

const FINETUNE: &str = "\
recipe: apr.recipe/v1
kind: finetune
name: qwen35-lora
base: { model: Qwen/Qwen3.5-0.8B }
method: lora
lora: { rank: 16, alpha: 32 }
data: { train: data/train.jsonl, eval: data/eval.jsonl }
eval: { metric: accuracy }
output: { path: out/adapter.apr }
";

const DISTILL: &str = "\
recipe: apr.recipe/v1
kind: distill
name: qwen35-kd
base: { model: student.apr }
teacher: { model: teacher.apr }
data: { train: data/train.jsonl }
output: { path: out/student.apr }
";

const MERGE: &str = "\
recipe: apr.recipe/v1
kind: merge
name: two-way
merge: { strategy: weighted, models: [a.apr, b.apr], weights: [0.7, 0.3] }
output: { path: out/merged.apr }
";

fn violations(yaml: &str) -> Vec<Violation> {
    match Recipe::from_yaml(yaml) {
        Err(RecipeError::Invalid(v)) => v,
        other => panic!("expected Invalid, got {other:?}"),
    }
}

fn rules(yaml: &str) -> Vec<(&'static str, String)> {
    violations(yaml)
        .into_iter()
        .map(|v| (v.rule, v.field))
        .collect()
}

fn has(yaml: &str, rule: &str, field: &str) -> bool {
    rules(yaml).iter().any(|(r, f)| *r == rule && f == field)
}

#[test]
fn valid_recipes_of_every_kind_load() {
    for yaml in [FINETUNE, DISTILL, MERGE] {
        let r = Recipe::from_yaml(yaml).expect("valid recipe");
        assert!(r.validate().is_empty());
    }
}

#[test]
fn defaults_are_filled_for_kinds_that_train() {
    let d = Recipe::from_yaml(DISTILL).expect("valid");
    assert_eq!(d.training, Some(TrainingSpec::default()));
    assert_eq!(d.distillation, Some(DistillSpec::default()));
    let m = Recipe::from_yaml(MERGE).expect("valid");
    assert_eq!(m.training, None);
    assert_eq!(m.distillation, None);
}

#[test]
fn falsify_rcp_001_wrong_version() {
    let yaml = FINETUNE.replace("apr.recipe/v1", "apr.recipe/v2");
    assert!(has(&yaml, "RCP-001", "recipe"));
}

#[test]
fn falsify_rcp_002_unknown_key_is_a_parse_error() {
    let top = format!("{FINETUNE}epochs: 3\n");
    assert!(matches!(
        Recipe::from_yaml(&top),
        Err(RecipeError::Parse(_))
    ));
    let nested = FINETUNE.replace("rank: 16", "rnak: 16");
    assert!(matches!(
        Recipe::from_yaml(&nested),
        Err(RecipeError::Parse(_))
    ));
}

#[test]
fn falsify_rcp_002_unknown_kind_and_method_are_parse_errors() {
    let kind = FINETUNE.replace("kind: finetune", "kind: pretrain");
    assert!(matches!(
        Recipe::from_yaml(&kind),
        Err(RecipeError::Parse(_))
    ));
    let method = FINETUNE.replace("method: lora", "method: auto");
    assert!(matches!(
        Recipe::from_yaml(&method),
        Err(RecipeError::Parse(_))
    ));
}

#[test]
fn falsify_rcp_003_finetune_sections() {
    let no_base = FINETUNE.replace("base: { model: Qwen/Qwen3.5-0.8B }\n", "");
    assert!(has(&no_base, "RCP-003", "base"));
    let with_teacher = format!("{FINETUNE}teacher: {{ model: t.apr }}\n");
    assert!(has(&with_teacher, "RCP-003", "teacher"));
}

#[test]
fn falsify_rcp_003_distill_needs_teacher() {
    let yaml = DISTILL.replace("teacher: { model: teacher.apr }\n", "");
    assert!(has(&yaml, "RCP-003", "teacher"));
}

#[test]
fn falsify_rcp_003_merge_forbids_training_sections() {
    let yaml = format!("{MERGE}data: {{ train: d.jsonl }}\ntraining: {{ epochs: 1 }}\n");
    assert!(has(&yaml, "RCP-003", "data"));
    assert!(has(&yaml, "RCP-003", "training"));
    let no_merge = MERGE.replace(
        "merge: { strategy: weighted, models: [a.apr, b.apr], weights: [0.7, 0.3] }\n",
        "",
    );
    assert!(has(&no_merge, "RCP-003", "merge"));
}

#[test]
fn falsify_rcp_004_lora_section_iff_lora_method() {
    let missing = FINETUNE.replace("lora: { rank: 16, alpha: 32 }\n", "");
    assert!(has(&missing, "RCP-004", "lora"));
    let extra = FINETUNE.replace("method: lora", "method: full");
    assert!(has(&extra, "RCP-004", "lora"));
    let qlora = FINETUNE.replace("method: lora", "method: qlora");
    assert!(Recipe::from_yaml(&qlora).is_ok());
}

#[test]
fn falsify_rcp_004_lora_ranges() {
    let yaml = FINETUNE.replace(
        "lora: { rank: 16, alpha: 32 }",
        "lora: { rank: 0, alpha: 0, dropout: 1.0 }",
    );
    assert!(has(&yaml, "RCP-004", "lora.rank"));
    assert!(has(&yaml, "RCP-004", "lora.alpha"));
    assert!(has(&yaml, "RCP-004", "lora.dropout"));
}

#[test]
fn falsify_rcp_005_training_ranges() {
    let yaml = format!(
        "{FINETUNE}training: {{ epochs: 0, batch_size: 0, learning_rate: -1.0, max_seq_len: 0 }}\n"
    );
    for field in [
        "training.epochs",
        "training.batch_size",
        "training.learning_rate",
        "training.max_seq_len",
    ] {
        assert!(has(&yaml, "RCP-005", field), "{field}");
    }
    let nan = format!("{FINETUNE}training: {{ learning_rate: .nan }}\n");
    assert!(has(&nan, "RCP-005", "training.learning_rate"));
}

#[test]
fn falsify_rcp_006_distillation_ranges() {
    let yaml = format!("{DISTILL}distillation: {{ temperature: 0, alpha: 1.5 }}\n");
    assert!(has(&yaml, "RCP-006", "distillation.temperature"));
    assert!(has(&yaml, "RCP-006", "distillation.alpha"));
    let edges = format!("{DISTILL}distillation: {{ temperature: 1, alpha: 0 }}\n");
    assert!(Recipe::from_yaml(&edges).is_ok());
    let one = format!("{DISTILL}distillation: {{ alpha: 1 }}\n");
    assert!(Recipe::from_yaml(&one).is_ok());
}

#[test]
fn falsify_rcp_007_weighted_needs_matching_weights() {
    let none = MERGE.replace(", weights: [0.7, 0.3]", "");
    assert!(has(&none, "RCP-007", "merge.weights"));
    let short = MERGE.replace("[0.7, 0.3]", "[1.0]");
    assert!(has(&short, "RCP-007", "merge.weights"));
    let negative = MERGE.replace("[0.7, 0.3]", "[1.5, -0.5]");
    assert!(has(&negative, "RCP-007", "merge.weights"));
}

#[test]
fn falsify_rcp_007_average_rejects_weights_and_one_model() {
    let avg = MERGE.replace("strategy: weighted", "strategy: average");
    assert!(has(&avg, "RCP-007", "merge.weights"));
    let one = MERGE.replace("[a.apr, b.apr], weights: [0.7, 0.3]", "[a.apr]");
    assert!(has(&one, "RCP-007", "merge.models"));
    let slerp = MERGE.replace("strategy: weighted", "strategy: slerp");
    assert!(Recipe::from_yaml(&slerp).is_ok());
}

#[test]
fn falsify_rcp_008_blank_fields() {
    let yaml = FINETUNE
        .replace("name: qwen35-lora", "name: \"  \"")
        .replace("out/adapter.apr", "\"\"");
    assert!(has(&yaml, "RCP-008", "name"));
    assert!(has(&yaml, "RCP-008", "output.path"));
}

#[test]
fn falsify_rcp_009_every_violation_is_reported_at_once() {
    let yaml = FINETUNE
        .replace("apr.recipe/v1", "apr.recipe/v0")
        .replace("rank: 16", "rank: 0")
        .replace("name: qwen35-lora", "name: \"\"");
    let got: Vec<&str> = violations(&yaml).iter().map(|v| v.rule).collect();
    for rule in ["RCP-001", "RCP-004", "RCP-008"] {
        assert!(got.contains(&rule), "{rule} missing from {got:?}");
    }
    let msg = Recipe::from_yaml(&yaml).expect_err("invalid").to_string();
    assert!(msg.contains("RCP-001") && msg.contains("RCP-004") && msg.contains("RCP-008"));
}

#[test]
fn falsify_rcp_010_hash_ignores_layout_comments_and_written_defaults() {
    let a = Recipe::from_yaml(DISTILL).expect("valid").hash();
    let reordered = "\
# a comment
kind: distill
recipe: 'apr.recipe/v1'
output: { path: out/student.apr }
teacher:
  model: teacher.apr
base: { model: student.apr }
name: qwen35-kd
data: { train: data/train.jsonl }
training: { epochs: 3, batch_size: 16, learning_rate: 0.0002, max_seq_len: 512, seed: 42 }
distillation: { temperature: 4.0, alpha: 0.7 }
";
    let b = Recipe::from_yaml(reordered).expect("valid").hash();
    assert_eq!(a, b);
    assert_eq!(a.len(), 64);
    assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn falsify_rcp_010_hash_changes_when_any_value_changes() {
    let base = Recipe::from_yaml(FINETUNE).expect("valid").hash();
    for (from, to) in [
        ("rank: 16", "rank: 8"),
        ("alpha: 32", "alpha: 16"),
        ("data/train.jsonl", "data/train2.jsonl"),
        ("metric: accuracy", "metric: f1"),
        ("qwen35-lora", "qwen35-lora-b"),
    ] {
        let changed = Recipe::from_yaml(&FINETUNE.replace(from, to))
            .expect("valid")
            .hash();
        assert_ne!(base, changed, "{from} -> {to}");
    }
    let seed = format!("{FINETUNE}training: {{ seed: 7 }}\n");
    assert_ne!(base, Recipe::from_yaml(&seed).expect("valid").hash());
}

#[test]
fn canonical_json_sorts_keys_at_every_level() {
    let v = serde_json::json!({"b": {"z": 1, "a": [{"y": 2, "x": 3}]}, "a": "q\""});
    let mut out = String::new();
    write_canonical(&v, &mut out);
    assert_eq!(out, r#"{"a":"q\"","b":{"a":[{"x":3,"y":2}],"z":1}}"#);
}

#[test]
fn load_reads_a_file_and_reports_a_missing_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("r.yaml");
    std::fs::write(&path, MERGE).expect("write");
    assert_eq!(
        Recipe::load(&path).expect("load"),
        Recipe::from_yaml(MERGE).expect("valid")
    );
    assert!(matches!(
        Recipe::load(&dir.path().join("absent.yaml")),
        Err(RecipeError::Io { .. })
    ));
}
