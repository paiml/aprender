//! Pre-flight gates for QLoRA distillation runs.
//!
//! Contract: `contracts/entrenar/qlora-distillation-v1.yaml`
//! Bindings: INV-DISTILL-001, INV-DISTILL-002, INV-DISTILL-004.
//!
//! MODEL-1 v2 autopsy identified three pre-flight gaps that each would
//! have caught the `ylkoylkoylko…` garbage on their own:
//!   (1) recipe.finetune.rank=32 vs metadata.lora_rank=16  → INV-DISTILL-001
//!   (2) recipe.target_tokens=500000 vs 99 rows on disk    → INV-DISTILL-002
//!   (3) recipe.temperature=4.0 with lr=2e-4               → INV-DISTILL-004
//!
//! Algorithm-level proof; full CLI wiring lands with PMAT-685 phase_1.

use serde::{Deserialize, Serialize};

/// Distillation recipe subset relevant to pre-flight (mirrors
/// `configs/distill/distill-32b-7b-text.yaml`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillRecipe {
    pub finetune: FinetuneSection,
    pub synthetic_data: SyntheticDataSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuneSection {
    pub rank: u32,
    pub learning_rate: f32,
    pub epochs: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_temperature() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticDataSection {
    pub target_tokens: u64,
}

/// Subset of `best/metadata.json` that the binding check inspects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub lora_rank: u32,
    pub lora_alpha: f32,
    pub learning_rate: f32,
    #[serde(default)]
    pub epochs_target: Option<u32>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PreflightError {
    #[error("config drift in field `{field}`: recipe={recipe}, run={run}")]
    ConfigDrift { field: &'static str, recipe: String, run: String },
    #[error("corpus undersized: have={have_tokens} tokens, need={need_tokens}")]
    CorpusUndersized { have_tokens: u64, need_tokens: u64 },
    #[error("learning rate {lr} too hot for distillation at temperature {temperature} (ceiling {ceiling})")]
    LrTooHotForDistillation { lr: f32, ceiling: f32, temperature: f32 },
}

/// Maximum LR permitted when soft-label temperature > 2.0.
pub const DISTILL_HOT_TEMPERATURE_LR_CEILING: f32 = 5e-5;

/// Temperature above which the tighter LR ceiling applies.
pub const DISTILL_HOT_TEMPERATURE_THRESHOLD: f32 = 2.0;

/// INV-DISTILL-001: recipe-to-run binding.
///
/// Rejects if any sensitive hyperparameter written to the run's
/// `metadata.json` disagrees with the recipe. `alpha = rank * 2` is a
/// QLoRA convention (see `qlora-hyperparameters-v1.yaml`).
pub fn check_recipe_binding(
    recipe: &DistillRecipe,
    run: &RunMetadata,
) -> Result<(), PreflightError> {
    if run.lora_rank != recipe.finetune.rank {
        return Err(PreflightError::ConfigDrift {
            field: "lora_rank",
            recipe: recipe.finetune.rank.to_string(),
            run: run.lora_rank.to_string(),
        });
    }
    let expected_alpha = recipe.finetune.rank as f32 * 2.0;
    if (run.lora_alpha - expected_alpha).abs() > f32::EPSILON {
        return Err(PreflightError::ConfigDrift {
            field: "lora_alpha",
            recipe: expected_alpha.to_string(),
            run: run.lora_alpha.to_string(),
        });
    }
    if (run.learning_rate - recipe.finetune.learning_rate).abs() > f32::EPSILON {
        return Err(PreflightError::ConfigDrift {
            field: "learning_rate",
            recipe: recipe.finetune.learning_rate.to_string(),
            run: run.learning_rate.to_string(),
        });
    }
    if let Some(run_epochs) = run.epochs_target {
        if run_epochs != recipe.finetune.epochs {
            return Err(PreflightError::ConfigDrift {
                field: "epochs",
                recipe: recipe.finetune.epochs.to_string(),
                run: run_epochs.to_string(),
            });
        }
    }
    Ok(())
}

/// INV-DISTILL-002: corpus-size pre-flight.
///
/// Caller passes a `token_counter` closure that runs the student
/// tokenizer over each row's `response` field. This avoids wiring the
/// real BPE tokenizer into the algorithm-level test surface.
pub fn check_corpus_size<F>(
    recipe: &DistillRecipe,
    corpus_rows: &[&str],
    token_counter: F,
) -> Result<u64, PreflightError>
where
    F: Fn(&str) -> u64,
{
    let have: u64 = corpus_rows.iter().map(|r| token_counter(r)).sum();
    let need = recipe.synthetic_data.target_tokens;
    if have < need {
        return Err(PreflightError::CorpusUndersized { have_tokens: have, need_tokens: need });
    }
    Ok(have)
}

/// INV-DISTILL-004: distillation-temperature-aware LR bound.
///
/// If recipe temperature exceeds the threshold, a tighter LR ceiling
/// (5e-5) applies regardless of model size. Below the threshold, the
/// classification path (`qlora-hyperparameters-v1` C-HP-001) governs
/// and this gate is Ok.
pub fn check_distill_lr_bound(recipe: &DistillRecipe) -> Result<(), PreflightError> {
    if recipe.finetune.temperature > DISTILL_HOT_TEMPERATURE_THRESHOLD
        && recipe.finetune.learning_rate > DISTILL_HOT_TEMPERATURE_LR_CEILING
    {
        return Err(PreflightError::LrTooHotForDistillation {
            lr: recipe.finetune.learning_rate,
            ceiling: DISTILL_HOT_TEMPERATURE_LR_CEILING,
            temperature: recipe.finetune.temperature,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe_v2() -> DistillRecipe {
        DistillRecipe {
            finetune: FinetuneSection {
                rank: 32,
                learning_rate: 2e-4,
                epochs: 3,
                temperature: 4.0,
            },
            synthetic_data: SyntheticDataSection { target_tokens: 500_000 },
        }
    }

    /// FALSIFY-DISTILL-001: the exact rank drift (16 vs 32) that MODEL-1
    /// v2 `best/metadata.json` contained.
    #[test]
    fn recipe_binding_rejects_rank_drift() {
        let recipe = recipe_v2();
        let run = RunMetadata {
            lora_rank: 16, // v2 actual
            lora_alpha: 32.0,
            learning_rate: 2e-4,
            epochs_target: Some(3),
        };
        let err = check_recipe_binding(&recipe, &run).unwrap_err();
        match err {
            PreflightError::ConfigDrift { field, recipe, run } => {
                assert_eq!(field, "lora_rank");
                assert_eq!(recipe, "32");
                assert_eq!(run, "16");
            }
            other => panic!("expected ConfigDrift on lora_rank, got {other:?}"),
        }
    }

    #[test]
    fn recipe_binding_accepts_matching_metadata() {
        let recipe = recipe_v2();
        let run = RunMetadata {
            lora_rank: 32,
            lora_alpha: 64.0,
            learning_rate: 2e-4,
            epochs_target: Some(3),
        };
        assert!(check_recipe_binding(&recipe, &run).is_ok());
    }

    #[test]
    fn recipe_binding_catches_alpha_drift() {
        let recipe = recipe_v2();
        let run = RunMetadata {
            lora_rank: 32,
            lora_alpha: 32.0, // should be 64.0 for rank=32
            learning_rate: 2e-4,
            epochs_target: Some(3),
        };
        let err = check_recipe_binding(&recipe, &run).unwrap_err();
        assert!(matches!(err, PreflightError::ConfigDrift { field: "lora_alpha", .. }));
    }

    /// FALSIFY-DISTILL-002: 99 teacher completions @ avg 500 tokens ≈
    /// 50K tokens, 10× below the 500K-token recipe target.
    #[test]
    fn corpus_preflight_rejects_99_samples() {
        let recipe = recipe_v2();
        let rows: Vec<&str> = (0..99).map(|_| "completion").collect();
        let ref_rows: Vec<&str> = rows.iter().copied().collect();
        let tokens_per_row = 500_u64;
        let err = check_corpus_size(&recipe, &ref_rows, |_| tokens_per_row).unwrap_err();
        match err {
            PreflightError::CorpusUndersized { have_tokens, need_tokens } => {
                assert_eq!(have_tokens, 99 * 500);
                assert_eq!(need_tokens, 500_000);
            }
            other => panic!("expected CorpusUndersized, got {other:?}"),
        }
    }

    #[test]
    fn corpus_preflight_accepts_500k_tokens() {
        let recipe = recipe_v2();
        let rows: Vec<&str> = (0..1000).map(|_| "completion").collect();
        let ref_rows: Vec<&str> = rows.iter().copied().collect();
        let ok = check_corpus_size(&recipe, &ref_rows, |_| 500).unwrap();
        assert_eq!(ok, 500_000);
    }

    /// FALSIFY-DISTILL-004: temperature=4.0 with LR=2e-4 (the v2 recipe)
    /// must be rejected; temperature=1.0 classification path with the
    /// same LR must pass (C-HP-001 territory).
    #[test]
    fn hot_temperature_rejects_2e4_lr() {
        let recipe = recipe_v2(); // temperature=4.0, lr=2e-4
        let err = check_distill_lr_bound(&recipe).unwrap_err();
        match err {
            PreflightError::LrTooHotForDistillation { lr, ceiling, temperature } => {
                assert!((lr - 2e-4).abs() < 1e-9);
                assert!((ceiling - 5e-5).abs() < 1e-9);
                assert!((temperature - 4.0).abs() < 1e-6);
            }
            other => panic!("expected LrTooHotForDistillation, got {other:?}"),
        }
    }

    #[test]
    fn hot_temperature_accepts_5e5_lr() {
        let mut recipe = recipe_v2();
        recipe.finetune.learning_rate = 5e-5;
        assert!(check_distill_lr_bound(&recipe).is_ok());
    }

    #[test]
    fn classification_temperature_accepts_2e4_lr() {
        let mut recipe = recipe_v2();
        recipe.finetune.temperature = 1.0;
        // lr=2e-4, temp=1.0 → classification path, C-HP-001 governs, this gate is Ok.
        assert!(check_distill_lr_bound(&recipe).is_ok());
    }
}
