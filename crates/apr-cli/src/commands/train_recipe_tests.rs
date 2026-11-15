//! `apr-recipe-v1` schema tests (contracts/apr-recipe-v1.yaml, PMAT-5316).
//!
//! Every refusal must name the recipe field at fault, and a valid recipe must
//! pass: a validator that refuses everything satisfies the negative cases.

use super::*;

const VALID: &str = "\
recipe: apr-recipe-v1
name: qwen-lora-demo
base: models/base.apr
method: lora
data:
  train: data/train.jsonl
eval:
  heldout: data/heldout.jsonl
seed: 42
";

fn parse(src: &str) -> Result<TrainRecipe> {
    TrainRecipe::parse(src, Path::new("/recipes"), "r.yaml")
}

#[test]
fn valid_minimal_recipe_parses_with_paths_anchored_at_the_recipe_dir() {
    let r = parse(VALID).expect("the positive control must parse");
    assert_eq!(r.name, "qwen-lora-demo");
    assert_eq!(r.method, RecipeMethod::Lora);
    assert_eq!(r.base, Path::new("/recipes/models/base.apr"));
    assert_eq!(r.train, Path::new("/recipes/data/train.jsonl"));
    assert_eq!(r.heldout, Path::new("/recipes/data/heldout.jsonl"));
    assert_eq!(r.seed, 42);
    assert_eq!(r.teacher, None);
    assert_eq!(
        (r.rank, r.epochs, r.learning_rate, r.max_seq_len),
        (None, None, None, None)
    );
    assert_eq!(r.output_dir, None);
    assert_eq!(r.benchmark, None);
    assert_eq!(r.model_size, None);
    assert_eq!(r.data_sha256_pin, None);
    assert_eq!(r.recipe_sha256, sha256_bytes(VALID.as_bytes()));
}

#[test]
fn method_spellings_round_trip() {
    for m in [
        RecipeMethod::Lora,
        RecipeMethod::Qlora,
        RecipeMethod::Distill,
    ] {
        assert_eq!(RecipeMethod::parse(m.as_str()), Some(m));
    }
    assert_eq!(
        RecipeMethod::parse("LoRA"),
        None,
        "the schema is lower-case"
    );
}
