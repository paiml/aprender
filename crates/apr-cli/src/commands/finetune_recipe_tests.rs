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

fn refused_field<T: std::fmt::Debug>(r: Result<T>) -> String {
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
    assert_eq!(a.held_out, dir.path().join("eval.jsonl"));
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
        (
            "metric the trainer does not report",
            text.replace("metric: loss", "metric: accuracy"),
            "eval.metric",
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

/// A distill recipe over a one-shard corpus; returns (dir, recipe text).
fn distill_fixture() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard: Vec<u8> = (0u32..64).flat_map(u32::to_le_bytes).collect();
    std::fs::write(dir.path().join("train.bin"), shard).expect("train");
    let eval: Vec<u8> = (100u32..132).flat_map(u32::to_le_bytes).collect();
    std::fs::write(dir.path().join("eval.bin"), eval).expect("eval");
    let train = sha256_file(&dir.path().join("train.bin")).expect("hash");
    let eval = sha256_file(&dir.path().join("eval.bin")).expect("hash");
    let text = format!(
        "recipe_version: 1
base:
  model: /models/student.apr
data:
  train: train.bin
  sha256: {train}
method:
  kind: distill
  teacher: /models/teacher.apr
  temperature: 2.5
eval:
  held_out: eval.bin
  sha256: {eval}
  metric: loss
training:
  epochs: 3
  batch_size: 8
  learning_rate: 0.0005
  seed: 11
"
    );
    (dir, text)
}

fn load_distill_text(dir: &tempfile::TempDir, text: &str) -> Result<DistillRecipeArgs> {
    let p = dir.path().join("r.yaml");
    std::fs::write(&p, text).expect("write recipe");
    load_distill(&p)
}

/// FALSIFY-RECIPE-008: every field a distill recipe declares reaches the
/// cuda distill arguments.
#[test]
fn a_valid_distill_recipe_supplies_the_distill_args() {
    let (dir, text) = distill_fixture();
    let a = load_distill_text(&dir, &text).expect("valid distill recipe");
    assert_eq!(a.student, PathBuf::from("/models/student.apr"));
    assert_eq!(a.teacher, PathBuf::from("/models/teacher.apr"));
    assert_eq!(a.data, dir.path().join("train.bin"));
    assert_eq!(a.held_out, dir.path().join("eval.bin"));
    assert!((a.temperature - 2.5).abs() < f64::EPSILON);
    assert_eq!((a.epochs, a.batch_size, a.seed), (3, 8, 11));
    assert!((a.learning_rate - 5e-4).abs() < f64::EPSILON);
    assert_eq!(a.hash.len(), 64);

    let no_t = text.replace("  temperature: 2.5\n", "");
    let a = load_distill_text(&dir, &no_t).expect("temperature is optional");
    assert!((a.temperature - DISTILL_DEFAULT_TEMPERATURE).abs() < f64::EPSILON);
}

/// (case, edit, field the refusal must name)
#[test]
fn distill_refusals_name_the_recipe_field() {
    let (dir, text) = distill_fixture();
    let zeros = "0".repeat(64);
    let train_sha = sha256_file(&dir.path().join("train.bin")).expect("hash");
    let eval_sha = sha256_file(&dir.path().join("eval.bin")).expect("hash");
    std::fs::write(dir.path().join("train.jsonl"), b"{}").expect("jsonl");
    let jsonl_sha = sha256_file(&dir.path().join("train.jsonl")).expect("hash");
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "train shard changed",
            text.replace(&train_sha, &zeros),
            "data.sha256",
        ),
        (
            "train is not a shard",
            text.replace("train: train.bin", "train: train.jsonl")
                .replace(&train_sha, &jsonl_sha),
            "data.train",
        ),
        (
            "lora routed to distill",
            text.replace(
                "  kind: distill\n  teacher: /models/teacher.apr\n  temperature: 2.5\n",
                "  kind: lora\n  rank: 8\n",
            ),
            "method.kind",
        ),
        (
            "held-out set is not a shard",
            text.replace("held_out: eval.bin", "held_out: train.jsonl")
                .replace(&eval_sha, &jsonl_sha),
            "eval.held_out",
        ),
        (
            "held-out shard changed",
            text.replace(&eval_sha, &zeros),
            "eval.sha256",
        ),
        (
            "metric the pipeline does not report",
            text.replace("metric: loss", "metric: accuracy"),
            "eval.metric",
        ),
        (
            "adapter alpha not honored",
            text.replace("  temperature: 2.5\n", "  temperature: 2.5\n  alpha: 16\n"),
            "method.alpha",
        ),
    ];
    for (name, yaml, field) in cases {
        let msg = refused_field(load_distill_text(&dir, &yaml));
        assert!(msg.contains(&format!("`{field}`")), "case {name}: {msg}");
    }
}

/// One pass over the held-out shard: whole `seq_len + 1` windows, whole
/// batches, never a wrapped-around partial one.
#[test]
fn held_out_batches_counts_whole_batches_in_one_pass() {
    // 64 tokens, windows of 8 → 8 windows → 2 batches of 4.
    assert_eq!(held_out_batches(64 * 4, 4, 7), 2);
    // 63 tokens → 7 windows → 1 whole batch of 4 (the partial one is dropped).
    assert_eq!(held_out_batches(63 * 4, 4, 7), 1);
    // Fewer tokens than one batch → 0 (the caller refuses the shard).
    assert_eq!(held_out_batches(31 * 4, 4, 7), 0);
    // A trailing partial token is not a token.
    assert_eq!(held_out_batches(64 * 4 + 3, 4, 7), 2);
}
