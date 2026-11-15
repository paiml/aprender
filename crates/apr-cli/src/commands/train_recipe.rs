//! Declarative training recipe: `apr-recipe-v1` (CRUX-001, PMAT-5316).
//!
//! Contract: `contracts/apr-recipe-v1.yaml`. One file names everything a
//! fine-tune or distillation run depends on; `apr finetune --recipe r.yaml`
//! refuses an invalid recipe before any model is opened, and every refusal
//! names the recipe field at fault. Relative paths resolve against the
//! recipe file's directory.
// Removed when `apr finetune` wires the recipe in (last slice of the stack).
#![allow(dead_code)]

use crate::error::CliError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The value the recipe's `recipe:` key must carry.
pub(crate) const RECIPE_SCHEMA: &str = "apr-recipe-v1";

/// The training method a recipe selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum RecipeMethod {
    Lora,
    Qlora,
    Distill,
}

impl RecipeMethod {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "lora" => Some(Self::Lora),
            "qlora" => Some(Self::Qlora),
            "distill" => Some(Self::Distill),
            _ => None,
        }
    }

    /// The spelling used in the recipe and in `apr finetune --method`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::Qlora => "qlora",
            Self::Distill => "distill",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecipe {
    recipe: Option<String>,
    name: Option<String>,
    base: Option<String>,
    model_size: Option<String>,
    method: Option<String>,
    data: Option<RawData>,
    eval: Option<RawEval>,
    seed: Option<i64>,
    teacher: Option<RawTeacher>,
    #[serde(default)]
    training: RawTraining,
    output: Option<RawOutput>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawData {
    train: Option<String>,
    sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEval {
    heldout: Option<String>,
    benchmark: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTeacher {
    model: Option<String>,
    temperature: Option<f64>,
    alpha: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTraining {
    rank: Option<u32>,
    epochs: Option<u32>,
    learning_rate: Option<f64>,
    max_seq_len: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutput {
    dir: Option<String>,
}

/// The teacher half of a distillation recipe.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecipeTeacher {
    pub(crate) model: PathBuf,
    pub(crate) temperature: f64,
    pub(crate) alpha: f64,
}

/// A recipe that passed every rule of `apr-recipe-v1`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TrainRecipe {
    pub(crate) name: String,
    pub(crate) base: PathBuf,
    /// Optional planning size ("0.8B"), so `--plan` works before the base is
    /// on disk. `--model-size` on the command line wins over it.
    pub(crate) model_size: Option<String>,
    pub(crate) method: RecipeMethod,
    pub(crate) train: PathBuf,
    /// Optional pin: the sha256 the training file must have.
    pub(crate) data_sha256_pin: Option<String>,
    pub(crate) heldout: PathBuf,
    pub(crate) benchmark: Option<String>,
    pub(crate) seed: u64,
    pub(crate) teacher: Option<RecipeTeacher>,
    pub(crate) rank: Option<u32>,
    pub(crate) epochs: Option<u32>,
    pub(crate) learning_rate: Option<f64>,
    pub(crate) max_seq_len: Option<usize>,
    pub(crate) output_dir: Option<PathBuf>,
    /// sha256 of the recipe file's bytes. The eval block is part of it.
    pub(crate) recipe_sha256: String,
}

fn refuse(origin: &str, violations: &[String]) -> CliError {
    CliError::ValidationFailed(format!(
        "recipe {origin} is not a valid {RECIPE_SCHEMA}: {}",
        violations.join("; ")
    ))
}

#[cfg(test)]
#[path = "train_recipe_tests.rs"]
mod tests;
