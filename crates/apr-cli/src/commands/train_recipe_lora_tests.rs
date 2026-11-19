//! PMAT-7240: the LoRA/optimiser keys of `apr-recipe-v1` (contracts/apr-recipe-v1.yaml,
//! FALSIFY-APR-RECIPE-006 and 007).

use super::*;
use crate::commands::train_recipe::TrainRecipe;
use std::path::Path;

const HEAD: &str = "\
recipe: apr-recipe-v1
name: lora-fields
base: b.apr
method: lora
data:
  train: t.jsonl
eval:
  heldout: h.jsonl
seed: 1
";

fn parse(training: &str) -> crate::error::Result<TrainRecipe> {
    let src = format!("{HEAD}training:\n{training}");
    TrainRecipe::parse(&src, Path::new("."), "test")
}

fn refusal(training: &str) -> String {
    parse(training).expect_err("must be refused").to_string()
}

/// A recipe with none of the keys resolves to today's hard-coded values and applies none of them
/// differently: this is what keeps every existing recipe behaving identically.
#[test]
fn absent_keys_resolve_to_the_values_finetune_hard_coded() {
    let r = TrainRecipe::parse(HEAD, Path::new("."), "test").expect("valid");
    assert_eq!(r.lora, RecipeLora::default());
    assert_eq!(r.lora.alpha, None);
    assert_eq!(r.lora.dropout, 0.0);
    assert_eq!(r.lora.warmup_fraction, 0.03);
    assert_eq!(r.lora.weight_decay, 0.01);
    assert_eq!(r.lora.target_modules, ["q_proj", "v_proj"]);
    assert!(r.lora.not_applied().is_empty());
    assert_eq!(r.lora.alpha_or(16.0), 16.0, "no alpha: the planner's stays");
}

#[test]
fn every_key_is_read() {
    let r = parse(
        "  lora_alpha: 32\n  lora_dropout: 0.05\n  warmup_fraction: 0.1\n  weight_decay: 0.001\n  \
         target_modules: [q_proj, k_proj, v_proj, o_proj]\n",
    )
    .expect("valid");
    assert_eq!(r.lora.alpha, Some(32.0));
    assert_eq!(r.lora.alpha_or(16.0), 32.0, "the recipe's alpha wins");
    assert_eq!(r.lora.dropout, 0.05);
    assert_eq!(r.lora.warmup_fraction, 0.1);
    assert_eq!(r.lora.weight_decay, 0.001);
    assert_eq!(r.lora.target_modules.len(), 4);
}

/// Dropout, weight decay and target modules have no consumer in the training loop: when a recipe
/// moves one off its default the receipt must say it was not applied. Alpha and warmup are applied.
#[test]
fn only_values_the_loop_ignores_are_reported_as_not_applied() {
    let r = parse("  lora_alpha: 32\n  warmup_fraction: 0.1\n").expect("valid");
    assert!(r.lora.not_applied().is_empty(), "{:?}", r.lora);
    let r =
        parse("  lora_dropout: 0.05\n  weight_decay: 0.0\n  target_modules: [q_proj, k_proj]\n")
            .expect("valid");
    assert_eq!(
        r.lora.not_applied(),
        [
            "training.lora_dropout",
            "training.weight_decay",
            "training.target_modules"
        ]
    );
    let r =
        parse("  lora_dropout: 0.0\n  weight_decay: 0.01\n  target_modules: [q_proj, v_proj]\n")
            .expect("valid");
    assert!(r.lora.not_applied().is_empty(), "defaults are not 'set'");
}

#[test]
fn each_bad_value_is_refused_by_field() {
    let cases = [
        ("  lora_alpha: 0\n", "training.lora_alpha"),
        ("  lora_alpha: -4\n", "training.lora_alpha"),
        ("  lora_dropout: 1.0\n", "training.lora_dropout"),
        ("  lora_dropout: -0.1\n", "training.lora_dropout"),
        ("  warmup_fraction: 1\n", "training.warmup_fraction"),
        ("  warmup_fraction: -0.5\n", "training.warmup_fraction"),
        ("  weight_decay: -0.01\n", "training.weight_decay"),
        ("  target_modules: []\n", "training.target_modules"),
        (
            "  target_modules: [q_proj, '']\n",
            "training.target_modules",
        ),
    ];
    for (training, field) in cases {
        let msg = refusal(training);
        assert!(
            msg.contains(&format!("field `{field}`")),
            "{training}: {msg}"
        );
    }
}

#[test]
fn every_violation_is_listed_at_once() {
    let msg = refusal("  lora_alpha: 0\n  lora_dropout: 2\n  weight_decay: -1\n");
    for field in ["lora_alpha", "lora_dropout", "weight_decay"] {
        assert!(msg.contains(&format!("training.{field}")), "{msg}");
    }
}

#[test]
fn zero_dropout_zero_warmup_and_zero_decay_are_valid() {
    let r = parse("  lora_dropout: 0\n  warmup_fraction: 0\n  weight_decay: 0\n").expect("valid");
    assert_eq!(r.lora.warmup_fraction, 0.0);
    assert_eq!(r.lora.weight_decay, 0.0);
}

#[test]
fn a_misspelt_key_is_still_refused() {
    let msg = refusal("  lora_alfa: 32\n");
    assert!(msg.contains("lora_alfa"), "{msg}");
    assert!(msg.contains("apr-recipe-v1"), "{msg}");
}

#[test]
fn the_active_block_is_read_by_the_run_and_cleared_after() {
    assert_eq!(alpha_or(16.0), 16.0);
    assert_eq!(warmup_fraction(), 0.03);
    let r = parse("  lora_alpha: 32\n  warmup_fraction: 0.25\n").expect("valid");
    {
        let _guard = activate(&r.lora);
        assert_eq!(alpha_or(16.0), 32.0);
        assert_eq!(warmup_fraction(), 0.25);
    }
    assert_eq!(alpha_or(16.0), 16.0, "cleared when the run ends");
    assert_eq!(warmup_fraction(), 0.03);
}
