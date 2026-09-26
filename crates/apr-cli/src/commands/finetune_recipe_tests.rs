use super::*;

/// A recipe directory with train/eval files; returns (dir, recipe text).
fn fixture(kind_block: &str) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("train.jsonl"), b"{\"x\":1}\n").expect("train");
    std::fs::write(dir.path().join("eval.jsonl"), b"{\"x\":2}\n").expect("eval");
    let train = sha256_file(&dir.path().join("train.jsonl")).expect("hash");
    let eval = sha256_file(&dir.path().join("eval.jsonl")).expect("hash");
    let text = format!(
        "recipe_version: 1
base:
  model: /models/does-not-exist.apr
data:
  train: train.jsonl
  sha256: {train}
method:
{kind_block}eval:
  held_out: eval.jsonl
  sha256: {eval}
  metric: loss
training:
  epochs: 2
  batch_size: 1
  learning_rate: 0.0001
  seed: 7
"
    );
    (dir, text)
}

const LORA: &str = "  kind: lora\n  rank: 8\n";

fn load_text(dir: &tempfile::TempDir, text: &str) -> Result<RecipeArgs> {
    let p = dir.path().join("r.yaml");
    std::fs::write(&p, text).expect("write recipe");
    load(&p)
}

fn refused_field(r: Result<RecipeArgs>) -> String {
    match r {
        Err(CliError::ValidationFailed(msg)) => msg,
        other => panic!("expected a recipe refusal, got {other:?}"),
    }
}

#[test]
fn a_valid_recipe_supplies_the_finetune_args() {
    let (dir, text) = fixture(LORA);
    let a = load_text(&dir, &text).expect("valid recipe");
    assert_eq!(a.method, "lora");
    assert_eq!(a.rank, Some(8));
    assert_eq!(a.epochs, 2);
    assert!((a.learning_rate - 1e-4).abs() < f64::EPSILON);
    // Relative data paths resolve against the recipe's own directory.
    assert_eq!(a.data, dir.path().join("train.jsonl"));
    // The model is never opened: a nonexistent model path still loads.
    assert_eq!(a.model, PathBuf::from("/models/does-not-exist.apr"));
    assert_eq!(a.seed, 7);
    assert_eq!(a.hash.len(), 64);
}

/// (case, edit, field the refusal must name)
#[test]
fn refusals_name_the_recipe_field() {
    let (dir, text) = fixture(LORA);
    let zeros = "0".repeat(64);
    let train_sha = sha256_file(&dir.path().join("train.jsonl")).expect("hash");
    let eval_sha = sha256_file(&dir.path().join("eval.jsonl")).expect("hash");
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "train data changed",
            text.replace(&train_sha, &zeros),
            "data.sha256",
        ),
        (
            "eval data changed",
            text.replace(&eval_sha, &zeros),
            "eval.sha256",
        ),
        (
            "train file missing",
            text.replace("train: train.jsonl", "train: gone.jsonl"),
            "data.sha256",
        ),
        (
            "distill routed to finetune",
            text.replace(LORA, "  kind: distill\n  teacher: /models/t.apr\n"),
            "method.kind",
        ),
        (
            "batch the trainer cannot honor",
            text.replace("batch_size: 1", "batch_size: 4"),
            "training.batch_size",
        ),
        (
            "schema refusal passes through",
            text.replace("  seed: 7\n", ""),
            "training",
        ),
    ];
    for (name, yaml, field) in cases {
        let msg = refused_field(load_text(&dir, &yaml));
        assert!(msg.contains(&format!("`{field}`")), "case {name}: {msg}");
    }
}

#[test]
fn an_unreadable_recipe_file_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let msg = refused_field(load(&dir.path().join("absent.yaml")));
    assert!(msg.contains("`<file>`"), "{msg}");
}
