//! LoRA and optimiser fields of `apr-recipe-v1` (PMAT-7240).
//!
//! Contract: `contracts/apr-recipe-v1.yaml` (`recipe_lora_fields`). Every field is optional and an
//! absent field means what `apr finetune` did before the field existed, so an existing recipe
//! behaves identically. `lora_alpha` and `warmup_fraction` reach the training run; the other three
//! are validated and named in the receipt's `validated_not_applied` while the training loop has no
//! consumer for them (`not_applied`).

use std::cell::RefCell;

/// Warmup fraction `apr finetune` hard-codes in its trainer config.
pub(crate) const DEFAULT_WARMUP_FRACTION: f64 = 0.03;
/// Weight decay of `AdamW::default_params`, the optimiser the instruct pipeline builds.
pub(crate) const DEFAULT_WEIGHT_DECAY: f64 = 0.01;
/// The instruct pipeline applies no LoRA dropout.
pub(crate) const DEFAULT_LORA_DROPOUT: f64 = 0.0;
/// The attention projections the LoRA pipelines adapt.
pub(crate) const DEFAULT_TARGET_MODULES: [&str; 2] = ["q_proj", "v_proj"];

/// The validated LoRA/optimiser block of a recipe.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecipeLora {
    /// `None`: the planner's value (2 x rank when `training.rank` is set).
    pub(crate) alpha: Option<f64>,
    pub(crate) dropout: f64,
    pub(crate) target_modules: Vec<String>,
    pub(crate) warmup_fraction: f64,
    pub(crate) weight_decay: f64,
    /// Fields the recipe set that the training loop does not read yet.
    not_applied: Vec<&'static str>,
}

impl Default for RecipeLora {
    fn default() -> Self {
        Self {
            alpha: None,
            dropout: DEFAULT_LORA_DROPOUT,
            target_modules: DEFAULT_TARGET_MODULES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            warmup_fraction: DEFAULT_WARMUP_FRACTION,
            weight_decay: DEFAULT_WEIGHT_DECAY,
            not_applied: Vec::new(),
        }
    }
}

/// The raw values of the five keys, as read from `training:`.
#[derive(Debug, Clone, Default)]
pub(crate) struct RawLoraFields {
    pub(crate) lora_alpha: Option<f64>,
    pub(crate) lora_dropout: Option<f64>,
    pub(crate) target_modules: Option<Vec<String>>,
    pub(crate) warmup_fraction: Option<f64>,
    pub(crate) weight_decay: Option<f64>,
}

impl RecipeLora {
    /// Validate the keys; every violation is recorded against its field.
    pub(crate) fn check(raw: &RawLoraFields, bad: &mut Vec<String>) -> Self {
        let mut out = Self::default();
        out.check_alpha(raw.lora_alpha, bad);
        out.check_dropout(raw.lora_dropout, bad);
        out.check_warmup(raw.warmup_fraction, bad);
        out.check_weight_decay(raw.weight_decay, bad);
        if let Some(t) = raw.target_modules.as_ref() {
            out.check_targets(t, bad);
        }
        out
    }

    fn check_alpha(&mut self, alpha: Option<f64>, bad: &mut Vec<String>) {
        let Some(a) = alpha else { return };
        if a.is_finite() && a > 0.0 {
            self.alpha = Some(a);
        } else {
            bad.push(format!(
                "field `training.lora_alpha` must be a positive number, got {a}"
            ));
        }
    }

    fn check_dropout(&mut self, dropout: Option<f64>, bad: &mut Vec<String>) {
        let Some(d) = dropout else { return };
        if (0.0..1.0).contains(&d) {
            self.dropout = d;
            self.mark_unapplied(d != DEFAULT_LORA_DROPOUT, "training.lora_dropout");
        } else {
            bad.push(format!(
                "field `training.lora_dropout` must be at least 0 and below 1, got {d}"
            ));
        }
    }

    fn check_warmup(&mut self, warmup: Option<f64>, bad: &mut Vec<String>) {
        let Some(w) = warmup else { return };
        if (0.0..1.0).contains(&w) {
            self.warmup_fraction = w;
        } else {
            bad.push(format!(
                "field `training.warmup_fraction` must be at least 0 and below 1, got {w}"
            ));
        }
    }

    fn check_weight_decay(&mut self, decay: Option<f64>, bad: &mut Vec<String>) {
        let Some(wd) = decay else { return };
        if wd.is_finite() && wd >= 0.0 {
            self.weight_decay = wd;
            self.mark_unapplied(wd != DEFAULT_WEIGHT_DECAY, "training.weight_decay");
        } else {
            bad.push(format!(
                "field `training.weight_decay` must be zero or a positive number, got {wd}"
            ));
        }
    }

    fn check_targets(&mut self, targets: &[String], bad: &mut Vec<String>) {
        if targets.is_empty() || targets.iter().any(|m| m.trim().is_empty()) {
            bad.push(
                "field `training.target_modules` must be a non-empty list of module names"
                    .to_string(),
            );
            return;
        }
        let cleaned: Vec<String> = targets.iter().map(|m| m.trim().to_string()).collect();
        let changed = cleaned
            .iter()
            .map(String::as_str)
            .ne(DEFAULT_TARGET_MODULES);
        self.target_modules = cleaned;
        self.mark_unapplied(changed, "training.target_modules");
    }

    fn mark_unapplied(&mut self, changed: bool, field: &'static str) {
        if changed {
            self.not_applied.push(field);
        }
    }

    /// Fields the recipe set to a non-default value that no training code reads.
    pub(crate) fn not_applied(&self) -> &[&'static str] {
        &self.not_applied
    }

    /// The alpha a run uses: the recipe's, else `planned`.
    pub(crate) fn alpha_or(&self, planned: f32) -> f32 {
        self.alpha.map_or(planned, |a| a as f32)
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<RecipeLora>> = const { RefCell::new(None) };
}

/// Holds a recipe's LoRA block active for the run on this thread; clears it on drop.
pub(crate) struct ActiveGuard;

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.with(|a| *a.borrow_mut() = None);
    }
}

/// Make `lora` the block `finetune::run` reads on this thread until the guard drops.
pub(crate) fn activate(lora: &RecipeLora) -> ActiveGuard {
    ACTIVE.with(|a| *a.borrow_mut() = Some(lora.clone()));
    ActiveGuard
}

/// The LoRA alpha for this run: the active recipe's, else `planned`.
pub(crate) fn alpha_or(planned: f32) -> f32 {
    ACTIVE.with(|a| a.borrow().as_ref().map_or(planned, |l| l.alpha_or(planned)))
}

/// The trainer's warmup fraction: the active recipe's, else the built-in 0.03.
pub(crate) fn warmup_fraction() -> f32 {
    ACTIVE.with(|a| {
        a.borrow()
            .as_ref()
            .map_or(DEFAULT_WARMUP_FRACTION, |l| l.warmup_fraction) as f32
    })
}

#[cfg(test)]
#[path = "train_recipe_lora_tests.rs"]
mod tests;
