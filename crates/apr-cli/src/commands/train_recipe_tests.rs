//! `apr-recipe-v1` schema tests (contracts/apr-recipe-v1.yaml, PMAT-5316).
//!
//! Every refusal must name the recipe field at fault, and a valid recipe must
//! pass: a validator that refuses everything satisfies the negative cases.

use super::*;

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
