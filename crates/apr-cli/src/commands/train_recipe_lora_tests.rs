//! PMAT-7240: the LoRA/optimiser keys of `apr-recipe-v1` (contracts/apr-recipe-v1.yaml,
//! FALSIFY-APR-RECIPE-010 and 007).

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

/// `blocks` is YAML appended to the required keys: `lora:` and `training:` blocks.
fn parse(blocks: &str) -> crate::error::Result<TrainRecipe> {
    let src = format!("{HEAD}{blocks}");
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
        "lora:\n  alpha: 32\n  dropout: 0.05\n  target_modules: [q_proj, k_proj, v_proj, o_proj]\n\
         training:\n  warmup: 0.1\n  weight_decay: 0.001\n",
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
    let r = parse("lora:\n  alpha: 32\ntraining:\n  warmup: 0.1\n").expect("valid");
    assert!(r.lora.not_applied().is_empty(), "{:?}", r.lora);
    let r =
        parse("lora:\n  dropout: 0.05\n  target_modules: [q_proj, k_proj]\ntraining:\n  weight_decay: 0.0\n")
            .expect("valid");
    assert_eq!(
        r.lora.not_applied(),
        [
            "lora.dropout",
            "training.weight_decay",
            "lora.target_modules"
        ]
    );
    let r =
        parse("lora:\n  dropout: 0.0\n  target_modules: [q_proj, v_proj]\ntraining:\n  weight_decay: 0.01\n")
            .expect("valid");
    assert!(r.lora.not_applied().is_empty(), "defaults are not 'set'");
}

#[test]
fn each_bad_value_is_refused_by_field() {
    let cases = [
        ("lora:\n  alpha: 0\n", "lora.alpha"),
        ("lora:\n  alpha: -4\n", "lora.alpha"),
        ("lora:\n  dropout: 1.0\n", "lora.dropout"),
        ("lora:\n  dropout: -0.1\n", "lora.dropout"),
        ("training:\n  warmup: 1\n", "training.warmup"),
        ("training:\n  warmup: -0.5\n", "training.warmup"),
        (
            "training:\n  weight_decay: -0.01\n",
            "training.weight_decay",
        ),
        ("lora:\n  target_modules: []\n", "lora.target_modules"),
        (
            "lora:\n  target_modules: [q_proj, '']\n",
            "lora.target_modules",
        ),
    ];
    for (blocks, field) in cases {
        let msg = refusal(blocks);
        assert!(msg.contains(&format!("field `{field}`")), "{blocks}: {msg}");
    }
}

#[test]
fn every_violation_is_listed_at_once() {
    let msg = refusal("lora:\n  alpha: 0\n  dropout: 2\ntraining:\n  weight_decay: -1\n");
    for field in ["lora.alpha", "lora.dropout", "training.weight_decay"] {
        assert!(msg.contains(field), "{msg}");
    }
}

#[test]
fn zero_dropout_zero_warmup_and_zero_decay_are_valid() {
    let r =
        parse("lora:\n  dropout: 0\ntraining:\n  warmup: 0\n  weight_decay: 0\n").expect("valid");
    assert_eq!(r.lora.warmup_fraction, 0.0);
    assert_eq!(r.lora.weight_decay, 0.0);
}

/// The old flat names (`training.lora_alpha` ...) are not keys: the schema is the issue's.
#[test]
fn a_misspelt_or_old_key_is_still_refused() {
    for blocks in [
        "training:\n  lora_alpha: 32\n",
        "training:\n  warmup_fraction: 0.1\n",
        "lora:\n  alfa: 32\n",
    ] {
        let msg = refusal(blocks);
        assert!(msg.contains("apr-recipe-v1"), "{msg}");
    }
    let msg = refusal("lora:\n  alfa: 32\n");
    assert!(msg.contains("alfa"), "{msg}");
}

#[test]
fn the_active_block_is_read_by_the_run_and_cleared_after() {
    assert_eq!(alpha_or(16.0), 16.0);
    assert_eq!(warmup_fraction(), 0.03);
    let r = parse("lora:\n  alpha: 32\ntraining:\n  warmup: 0.25\n").expect("valid");
    {
        let _guard = activate(&r.lora);
        assert_eq!(alpha_or(16.0), 32.0);
        assert_eq!(warmup_fraction(), 0.25);
    }
    assert_eq!(alpha_or(16.0), 16.0, "cleared when the run ends");
    assert_eq!(warmup_fraction(), 0.03);
}
