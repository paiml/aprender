use super::*;

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn good() -> String {
    format!(
        "recipe_version: 1
base:
  model: Qwen/Qwen3.5-0.8B
data:
  train: data/train.jsonl
  sha256: {SHA_A}
method:
  kind: lora
  rank: 16
  alpha: 32
eval:
  held_out: data/eval.jsonl
  sha256: {}
  metric: loss
training:
  epochs: 1
  batch_size: 4
  learning_rate: 0.0002
  seed: 42
",
        "b".repeat(64)
    )
}

fn refused(yaml: &str) -> RecipeError {
    Recipe::parse(yaml).expect_err("recipe should be refused")
}

#[test]
fn good_recipe_parses() {
    let r = Recipe::parse(&good()).expect("good recipe");
    assert_eq!(r.method.kind, MethodKind::Lora);
    assert_eq!(r.training.seed, 42);
}

/// (case, edit applied to the good recipe, field the refusal must name)
#[test]
fn refusals_name_the_field() {
    let g = good();
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "missing eval",
            g.replace(
                &format!(
                    "eval:\n  held_out: data/eval.jsonl\n  sha256: {}\n  metric: loss\n",
                    "b".repeat(64)
                ),
                "",
            ),
            "eval",
        ),
        ("missing seed", g.replace("  seed: 42\n", ""), "training"),
        ("unknown top key", format!("{g}severty: P1\n"), "severty"),
        ("unknown nested key", g.replace("  rank: 16\n", "  rank: 16\n  rnak: 2\n"), "method"),
        ("version 2", g.replace("recipe_version: 1", "recipe_version: 2"), "recipe_version"),
        ("bad data sha", g.replace(SHA_A, "HEAD"), "data.sha256"),
        ("eval sha is data sha", g.replace(&"b".repeat(64), SHA_A), "eval.sha256"),
        ("lora rank 0", g.replace("rank: 16", "rank: 0"), "method.rank"),
        ("lora no rank", g.replace("  rank: 16\n", ""), "method.rank"),
        (
            "distill no teacher",
            g.replace("kind: lora\n  rank: 16\n  alpha: 32\n", "kind: distill\n"),
            "method.teacher",
        ),
        (
            "distill self-teacher",
            g.replace(
                "kind: lora\n  rank: 16\n  alpha: 32\n",
                "kind: distill\n  teacher: Qwen/Qwen3.5-0.8B\n",
            ),
            "method.teacher",
        ),
        (
            "lora with teacher",
            g.replace("  alpha: 32\n", "  alpha: 32\n  teacher: Qwen/Qwen3.5-9B\n"),
            "method.teacher",
        ),
        ("full with rank", g.replace("kind: lora", "kind: full"), "method.rank"),
        ("unknown method", g.replace("kind: lora", "kind: dpo"), "method"),
        ("zero epochs", g.replace("epochs: 1", "epochs: 0"), "training.epochs"),
        (
            "nan lr",
            g.replace("learning_rate: 0.0002", "learning_rate: .nan"),
            "training.learning_rate",
        ),
        ("empty model", g.replace("model: Qwen/Qwen3.5-0.8B", "model: ''"), "base.model"),
        ("not a mapping", "- a\n- b\n".to_string(), "<root>"),
        ("uppercase data sha", g.replace(SHA_A, &SHA_A.to_uppercase()), "data.sha256"),
        ("63-char data sha", g.replace(SHA_A, &SHA_A[1..]), "data.sha256"),
        ("bad eval sha", g.replace(&"b".repeat(64), "g".repeat(64).as_str()), "eval.sha256"),
        (
            "empty held_out",
            g.replace("held_out: data/eval.jsonl", "held_out: ' '"),
            "eval.held_out",
        ),
        ("empty train", g.replace("train: data/train.jsonl", "train: ''"), "data.train"),
        ("zero alpha", g.replace("alpha: 32", "alpha: 0"), "method.alpha"),
        ("zero batch", g.replace("batch_size: 4", "batch_size: 0"), "training.batch_size"),
        (
            "negative lr",
            g.replace("learning_rate: 0.0002", "learning_rate: -0.1"),
            "training.learning_rate",
        ),
        (
            "zero temperature",
            g.replace(
                "kind: lora\n  rank: 16\n  alpha: 32\n",
                "kind: distill\n  teacher: Qwen/Qwen3.5-9B\n  temperature: 0\n",
            ),
            "method.temperature",
        ),
        ("qlora no rank", g.replace("kind: lora\n  rank: 16\n", "kind: qlora\n"), "method.rank"),
        (
            "full with alpha only",
            g.replace("kind: lora\n  rank: 16\n", "kind: full\n"),
            "method.rank",
        ),
    ];
    for (name, yaml, field) in cases {
        let e = refused(&yaml);
        assert_eq!(e.field, field, "case {name}: {e}");
        assert!(
            e.to_string().contains(&format!("`{field}`")),
            "case {name}: message must name the field: {e}"
        );
    }
}

#[test]
fn distill_with_distinct_teacher_parses() {
    let y = good().replace(
        "kind: lora\n  rank: 16\n  alpha: 32\n",
        "kind: distill\n  teacher: Qwen/Qwen3.5-9B\n  temperature: 4.0\n",
    );
    let r = Recipe::parse(&y).expect("distill recipe");
    assert_eq!(r.method.teacher.as_deref(), Some("Qwen/Qwen3.5-9B"));
}

#[test]
fn hash_ignores_formatting_and_tracks_values() {
    let a = Recipe::parse(&good()).expect("good");
    // Same values, different key order and quoting.
    let reordered = good()
        .replace("  epochs: 1\n  batch_size: 4\n", "  batch_size: 4\n  epochs: 1\n")
        .replace("model: Qwen/Qwen3.5-0.8B", "model: \"Qwen/Qwen3.5-0.8B\"");
    let b = Recipe::parse(&reordered).expect("reordered");
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.hash().len(), 64);
    let c = Recipe::parse(&good().replace("seed: 42", "seed: 43")).expect("seed 43");
    assert_ne!(a.hash(), c.hash(), "the seed is part of the recipe identity");
}
