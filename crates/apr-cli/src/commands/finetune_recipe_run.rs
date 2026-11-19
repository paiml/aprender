//! What a recipe run applies and what it leaves behind (PMAT-7239, contracts/apr-recipe-v1.yaml).
//!
//! * the recipe `seed` reaches the trainer through [`SeedGuard`] / [`run_seed`];
//! * after a real run `<output dir>/receipt.json` records the hashes of the recipe, the base
//!   model, the dataset and the heldout file, the `apr` version and commit, and the resolved
//!   hyperparameters. `--json` prints the same object.

use super::train_recipe::{sha256_file, EngineId, RecipeReceipt, TrainRecipe};
use crate::error::{CliError, Result};
use serde_json::{json, Value};
use std::cell::Cell;
use std::path::{Path, PathBuf};

/// The schema id of `receipt.json`.
pub(crate) const FINETUNE_RECEIPT_SCHEMA: &str = "apr-finetune-receipt-v1";
/// The file name written under the output directory.
pub(crate) const RECEIPT_FILE: &str = "receipt.json";
/// The seed of a run that did not come from a recipe.
pub(crate) const DEFAULT_SEED: u64 = 42;

thread_local! {
    static RUN_SEED: Cell<Option<u64>> = const { Cell::new(None) };
}

/// The trainer seed: the recipe's while a [`SeedGuard`] is alive on this thread, else 42.
pub(crate) fn run_seed() -> u64 {
    RUN_SEED.with(Cell::get).unwrap_or(DEFAULT_SEED)
}

/// Sets the trainer seed for this thread and restores the previous one on drop.
pub(crate) struct SeedGuard(Option<u64>);

impl SeedGuard {
    pub(crate) fn set(seed: u64) -> Self {
        Self(RUN_SEED.with(|c| c.replace(Some(seed))))
    }
}

impl Drop for SeedGuard {
    fn drop(&mut self) {
        RUN_SEED.with(|c| c.set(self.0));
    }
}

thread_local! {
    static PLAN_JSON_SUPPRESSED: Cell<bool> = const { Cell::new(false) };
}

/// True while a `--json` recipe run owns stdout: the plan object is not printed, because the
/// run prints exactly one document, the receipt.
pub(crate) fn plan_json_suppressed() -> bool {
    PLAN_JSON_SUPPRESSED.with(Cell::get)
}

/// Suppresses the plan JSON on this thread while alive; restores the previous state on drop.
pub(crate) struct PlanJsonGuard(bool);

impl PlanJsonGuard {
    pub(crate) fn suppress(on: bool) -> Self {
        Self(PLAN_JSON_SUPPRESSED.with(|c| c.replace(on)))
    }
}

impl Drop for PlanJsonGuard {
    fn drop(&mut self) {
        PLAN_JSON_SUPPRESSED.with(|c| c.set(self.0));
    }
}

/// The values the run resolved after recipe defaults and flags.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedRun<'a> {
    pub(crate) epochs: u32,
    pub(crate) quantize_nf4: bool,
    pub(crate) model_size: Option<&'a str>,
}

/// Build the receipt object, written to `receipt.json` and printed by `--json`.
pub(crate) fn build_receipt(
    recipe: &TrainRecipe,
    base: &RecipeReceipt,
    engine: &EngineId,
    resolved: &ResolvedRun<'_>,
) -> Value {
    json!({
        "schema": FINETUNE_RECEIPT_SCHEMA,
        "recipe_name": base.recipe_name,
        "recipe_sha256": base.recipe_sha256,
        "base_model_sha256": sha256_file(&recipe.base),
        "dataset_sha256": base.data_sha256,
        "eval_sha256": base.eval_sha256,
        "apr": { "version": engine.version, "commit": engine.sha },
        "hyperparameters": {
            "method": recipe.method.as_str(),
            "seed": recipe.seed,
            "rank": recipe.rank,
            "epochs": resolved.epochs,
            "learning_rate": recipe.learning_rate,
            "max_seq_len": recipe.max_seq_len,
            "quantize_nf4": resolved.quantize_nf4,
            "model_size": resolved.model_size,
            "eval_benchmark": recipe.benchmark,
        },
        "overrides": base.overrides,
        "validated_not_applied": base.validated_not_applied,
    })
}

/// Where `receipt.json` goes: the directory of the adapter path, else the current directory.
pub(crate) fn receipt_dir(output: Option<&Path>) -> PathBuf {
    match output.and_then(Path::parent) {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Write `receipt.json` (pretty, trailing newline) and return its path.
pub(crate) fn write_receipt(dir: &Path, receipt: &Value) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CliError::ValidationFailed(format!("cannot create {}: {e}", dir.display())))?;
    let path = dir.join(RECEIPT_FILE);
    let mut text = serde_json::to_string_pretty(receipt)
        .map_err(|e| CliError::ValidationFailed(format!("cannot serialize receipt: {e}")))?;
    text.push('\n');
    std::fs::write(&path, text)
        .map_err(|e| CliError::ValidationFailed(format!("cannot write {}: {e}", path.display())))?;
    Ok(path)
}

#[cfg(test)]
#[path = "finetune_recipe_run_tests.rs"]
mod tests;
