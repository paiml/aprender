// Integration tests: unwrap()/expect() are idiomatic here.
#![allow(clippy::disallowed_methods, clippy::unwrap_used)]

//! End-to-end falsification of `apr finetune --recipe` LoRA keys and `--dry-run` (an alias of `--plan`)
//! (PMAT-7240; contracts/apr-recipe-v1.yaml, FALSIFY-APR-RECIPE-006..009).
//!
//! Every test drives the real `apr` binary on a recipe whose input files do not exist.

use assert_cmd::Command;
use std::path::PathBuf;

const RECIPE: &str = "\
recipe: apr-recipe-v1
name: e2e
base: models/none.apr
model_size: 1B
method: lora
data:
  train: data/t.jsonl
eval:
  heldout: data/h.jsonl
seed: 7
training:
  rank: 8
";

fn write_recipe(tag: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("apr-recipe-e2e-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("r.yaml");
    std::fs::write(&path, body).unwrap();
    path
}

fn apr(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("apr")
        .unwrap()
        .args(args)
        .output()
        .unwrap()
}

/// The first JSON object in a command's stdout (a header line may precede it).
fn first_json(stdout: &[u8]) -> serde_json::Value {
    let text = String::from_utf8_lossy(stdout);
    let start = text.find('{').expect("a JSON object on stdout");
    serde_json::Deserializer::from_str(&text[start..])
        .into_iter::<serde_json::Value>()
        .next()
        .expect("one value")
        .expect("valid JSON")
}

/// Every JSON object in a command's stdout, in order.
fn all_json(stdout: &[u8]) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(stdout);
    let start = text.find('{').expect("a JSON object on stdout");
    serde_json::Deserializer::from_str(&text[start..])
        .into_iter::<serde_json::Value>()
        .map(|v| v.expect("valid JSON"))
        .collect()
}

/// FALSIFY-APR-RECIPE-008, run path: `--dry-run` and `--plan` print the fully resolved recipe,
/// every default filled in, through the binary. A recipe that sets nothing prints the defaults;
/// one that sets values prints them. Every value is a number: the planner's rank, alpha and
/// learning rate are the ones the plan block above it prints, and `training.max_seq_len` is the
/// sequence length the selected training path uses (512 unset; the WGPU path ignores a set value).
#[test]
fn dry_run_prints_the_resolved_recipe_with_every_default_filled_in() {
    let bare = write_recipe("resolved", RECIPE);
    let p = bare.to_str().unwrap();
    for flag in ["--dry-run", "--plan"] {
        let out = apr(&["finetune", "--recipe", p, flag, "--json"]);
        assert_eq!(out.status.code(), Some(0), "{flag}");
        let docs = all_json(&out.stdout);
        assert_eq!(docs.len(), 2, "plan object then resolved recipe: {flag}");
        let r = &docs[1];
        assert_eq!(r["recipe"], "apr-recipe-v1");
        assert_eq!(r["name"], "e2e");
        assert_eq!(r["seed"], 7);
        assert_eq!(r["lora"]["dropout"], 0.0);
        assert_eq!(
            r["lora"]["target_modules"],
            serde_json::json!(["q_proj", "v_proj"])
        );
        let plan = &docs[0];
        assert_eq!(r["lora"]["alpha"], plan["recommended_alpha"], "{flag}");
        assert_eq!(r["lora"]["alpha"], 16.0, "2 x rank 8: {flag}");
        assert_eq!(
            r["training"]["learning_rate"], plan["learning_rate"],
            "{flag}"
        );
        assert!(
            r["training"]["learning_rate"]
                .as_f64()
                .is_some_and(|x| x > 0.0),
            "a real learning rate, not a label: {flag}"
        );
        assert_eq!(r["training"]["max_seq_len"], 512, "{flag}");
        assert_eq!(r["training"]["warmup"], 0.03);
        assert_eq!(r["training"]["weight_decay"], 0.01);
        assert_eq!(r["training"]["epochs"], 3);
        assert_eq!(r["training"]["rank"], 8);
        assert_eq!(r["training"]["rank"], plan["recommended_rank"], "{flag}");
    }
    let set = write_recipe(
        "resolved-set",
        &format!(
            "{RECIPE}  warmup: 0.1\n  weight_decay: 0.0\n  epochs: 2\n  max_seq_len: 256\n  learning_rate: 0.0005\nlora:\n  alpha: 32\n  dropout: 0.05\n  target_modules: [q_proj, k_proj]\n"
        ),
    );
    let out = apr(&[
        "finetune",
        "--recipe",
        set.to_str().unwrap(),
        "--dry-run",
        "--json",
    ]);
    assert_eq!(out.status.code(), Some(0));
    let r = all_json(&out.stdout).remove(1);
    assert_eq!(r["lora"]["alpha"], 32.0);
    assert_eq!(r["lora"]["dropout"], 0.05);
    assert_eq!(
        r["lora"]["target_modules"],
        serde_json::json!(["q_proj", "k_proj"])
    );
    assert_eq!(r["training"]["warmup"], 0.1);
    assert_eq!(r["training"]["weight_decay"], 0.0);
    assert_eq!(r["training"]["epochs"], 2);
    assert_eq!(r["training"]["learning_rate"], 0.0005);
    // The WGPU path builds its pipeline with a fixed 512 and never reads the set value.
    let expect_seq = if cfg!(feature = "wgpu") { 512 } else { 256 };
    assert_eq!(r["training"]["max_seq_len"], expect_seq);
    let out = apr(&["finetune", "--recipe", bare.to_str().unwrap(), "--dry-run"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains("Resolved recipe (defaults filled in):"));
}

/// FALSIFY-APR-RECIPE-007 (plan path): lora.alpha 32 shows as recommended_alpha 32.
/// The run path has its own falsifier, `recipe_alpha_reaches_the_config_the_run_trains_and_saves_with`.
#[test]
fn recipe_lora_alpha_reaches_the_plan() {
    let with = write_recipe("alpha32", &format!("{RECIPE}lora:\n  alpha: 32\n"));
    let out = apr(&[
        "finetune",
        "--recipe",
        with.to_str().unwrap(),
        "--plan",
        "--json",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(first_json(&out.stdout)["recommended_alpha"], 32.0);
    let without = write_recipe("alphadef", RECIPE);
    let out = apr(&[
        "finetune",
        "--recipe",
        without.to_str().unwrap(),
        "--plan",
        "--json",
    ]);
    assert!(out.status.success());
    assert_ne!(first_json(&out.stdout)["recommended_alpha"], 32.0);
}

/// FALSIFY-APR-RECIPE-008: `--dry-run` is `--plan`: same exit code, same stdout, with no
/// model, data or eval file on disk; refusals are the same (exit 5).
#[test]
fn dry_run_is_the_plan_flag() {
    let path = write_recipe("dry", &format!("{RECIPE}lora:\n  alpha: 32\n"));
    let p = path.to_str().unwrap();
    let plan = apr(&["finetune", "--recipe", p, "--plan", "--json"]);
    let dry = apr(&["finetune", "--recipe", p, "--dry-run", "--json"]);
    assert_eq!(
        dry.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&dry.stderr)
    );
    assert_eq!(plan.status.code(), dry.status.code());
    assert_eq!(
        plan.stdout, dry.stdout,
        "--dry-run must print what --plan prints"
    );
    for (tag, body, needle) in [
        ("unk", format!("{RECIPE}lora:\n  alfa: 32\n"), "alfa"),
        ("bad", format!("{RECIPE}lora:\n  alpha: 0\n"), "lora.alpha"),
    ] {
        let path = write_recipe(tag, &body);
        let p = path.to_str().unwrap();
        for flag in ["--plan", "--dry-run"] {
            let out = apr(&["finetune", "--recipe", p, flag]);
            assert_eq!(out.status.code(), Some(5), "{tag} {flag}");
            assert!(
                String::from_utf8_lossy(&out.stderr).contains(needle),
                "{tag} {flag}"
            );
        }
    }
}

/// FALSIFY-APR-RECIPE-007: a field validated but not applied is reported, never silent.
#[test]
fn validated_but_not_applied_fields_are_reported() {
    let path = write_recipe(
        "na",
        &format!("{RECIPE}  weight_decay: 0.0\nlora:\n  dropout: 0.1\n"),
    );
    let out = apr(&["finetune", "--recipe", path.to_str().unwrap(), "--dry-run"]);
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("lora.dropout"), "{text}");
    assert!(text.contains("training.weight_decay"), "{text}");
    let path = write_recipe("na0", RECIPE);
    let out = apr(&["finetune", "--recipe", path.to_str().unwrap(), "--dry-run"]);
    assert!(!String::from_utf8_lossy(&out.stderr).contains("lora.dropout"));
}

/// FALSIFY-APR-RECIPE-009: every shipped example recipe passes `--dry-run` through the binary.
#[test]
fn every_example_recipe_passes_dry_run() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes");
    let mut n = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|x| x == "yaml") {
            let out = apr(&["finetune", "--recipe", path.to_str().unwrap(), "--dry-run"]);
            assert_eq!(
                out.status.code(),
                Some(0),
                "{}: {}",
                path.display(),
                String::from_utf8_lossy(&out.stderr)
            );
            n += 1;
        }
    }
    assert!(
        n >= 7,
        "expected tiny plus six notebook examples, found {n}"
    );
}

/// #7240 acceptance: the resolved plan of the unmodified `examples/recipes/tiny.yaml` is
/// byte-identical before and after. `tests/fixtures/recipe_tiny_plan.json` is the exact stdout of
/// `apr finetune --recipe examples/recipes/tiny.yaml --plan --json` built from origin/main
/// (14fd3f7c0), before this change; the plan this branch prints is that text, byte for byte,
/// with the resolved recipe after it.
#[test]
fn tiny_recipe_plan_is_byte_identical_to_main() {
    let tiny = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes/tiny.yaml");
    let golden = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recipe_tiny_plan.json"),
    )
    .unwrap();
    assert!(!golden.is_empty(), "the golden plan is not empty");
    for flag in ["--plan", "--dry-run"] {
        let out = apr(&[
            "finetune",
            "--recipe",
            tiny.to_str().unwrap(),
            flag,
            "--json",
        ]);
        assert_eq!(out.status.code(), Some(0), "{flag}");
        assert!(
            out.stdout.starts_with(&golden),
            "{flag}: the plan object differs from main's:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
        let rest = &out.stdout[golden.len()..];
        let resolved: serde_json::Value = serde_json::from_slice(rest.trim_ascii())
            .expect("the resolved recipe follows the plan");
        let seq = if cfg!(feature = "wgpu") { 512 } else { 256 };
        assert_eq!(
            resolved["training"]["max_seq_len"], seq,
            "tiny sets 256: {flag}"
        );
        assert_eq!(resolved["training"]["rank"], 8);
    }
}
