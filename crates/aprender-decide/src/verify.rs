//! The gate verifier and the packer-for-serving (plan 08-09; D-06, D-07, D-17).
//!
//! [`crate::pack`] turns a run dir into bytes and applies no policy. This module is the
//! policy: it decides whether a run (or the exact `.apr` packed from it) may be served, and it
//! decides that on NOTHING the run dir merely asserts.
//!
//! # What is recomputed, and from what
//!
//! A `gate-report.json` is a file a training run wrote, so an edited report must not pass.
//! The pipeline both public doors run ([`pack_for_serving`] and [`verify_path`]: the cheap
//! checks, then the re-scores and the gate on the loaded bytes) therefore:
//!
//! 1. refuses any variant but `production` ([`check_variant`]), any recipe.json that differs
//!    from the contract's recipe block ([`check_recipe_block`]), and any base other than the
//!    contract's — every declared identity field, and the base dir's files against the
//!    contract's pins ([`check_base`], and the zero-shot load);
//! 2. re-hashes the data dir and the probability files against the report ([`check_inputs`]);
//! 3. re-derives text-level split disjointness from the data dir and the report's `slice_ids`
//!    ([`check_split`], the mirror of `scripts/laya_train/data.py`), and holds the report's
//!    calibration and device records to the relations the contract states for them
//!    ([`check_report_record`]);
//! 4. validates both probability files row by row ([`validate_probs`]), so the aprender-core
//!    metric functions' panicking preconditions can never fire on file data;
//! 5. re-scores EVERY eval row in Rust ([`rescore`]): the fine-tuned model is the
//!    [`Decider`] loaded from the PACKED bytes through the whole decide-apr-v1 ladder, the
//!    zero-shot model is the declared base ([`crate::pack::scorer_from_parts`] over the base
//!    dir's `model.safetensors`, re-hashed on the very bytes it builds from, and its tokenizer,
//!    bound to the run's `inputs_sha256.tokenizer_json`);
//!    every probability must agree within its set's bound with the argmax exact. The bound is
//!    laya-parity-v1 A1's `bound(c, s) = max(floor, k x noise(c, s))`, DERIVED here by
//!    [`rescore_bounds`] from the hash-bound float64 record `rescore-noise.json` (noise
//!    recomputed from its stored rows, reported values only cross-checked, a bound above the
//!    contract ceiling refused), and exactly the floor when the run carries no record;
//! 6. recomputes macro-F1 and ECE with aprender-core's ONE implementation of each (OPS-03,
//!    [`recompute_metrics`]; the eval NLL with its log loss, [`recompute_nll`]) on those
//!    verified probabilities, and decides the gate on the recomputed values ([`check_gate`]).
//!
//! Every leaf of recipe.json and gate-report.json is either bound by one of these checks or
//! listed report-only with its reason: laya-finetune-gate-v1 `run_field_bindings`, enforced by
//! the `every_run_field_is_bound_or_report_only` sweep (plan 08-21, class A verify side).
//!
//! Under laya-finetune-gate-v1 1.4.0 the cheap checks also RE-DERIVE the shipped seed
//! ([`check_seed_selection`], A3: every seed's metrics recomputed from its hash-bound
//! probability file, the median-ECE seed by the declared rank key and tie-break, the shipped
//! checkpoint and eval file bound by sha256) and RECOMPUTE the reported shift probe
//! ([`check_shift_probe`], A2: never re-scored, never a gate clause). A run whose recipe carries
//! no `seed_selection` is refused [`VerifyError::SeedPolicyMissing`] — but only AFTER the
//! re-scores and the gate, so a legacy run whose gate fails still reports
//! [`VerifyError::GateFailed`].
//!
//! The policy — thresholds, tolerances, the base block — is passed in as a [`VerifyPolicy`]
//! built by the ONE contract-to-policy mapping, [`VerifyPolicy::from_contract_views`], over the
//! typed views [`GateContractView`] and [`ParityContractView`]: the library stays YAML-free
//! (the caller parses the YAML into the views), and `examples/pack_laya.rs` and every test read
//! the contracts through that same mapping at run time.
//!
//! # Writing
//!
//! [`pack_for_serving`] writes only after that pipeline accepted the bytes, atomically (a
//! temp file in the target directory, then a rename). On any refusal nothing is created.
//! [`fixture_bytes`] is the one other writer's source, and it produces only
//! `synthetic-fixture` artifacts, which every verify refuses.

use crate::artifact::{self, within, ArtifactError};
use crate::digest::sha256_hex;
use crate::laya::argmax;
use crate::pack::{self, GateCalibration, GateReport, PackError, PackInputs, Recipe};
use crate::{DecideError, Decider, Decision, DecisionMethod, Task};
use aprender::calibration::expected_calibration_error_top_label_f64;
use aprender::metrics::classification::{macro_f1_f64, mean_f1_over_labels_f64};
use aprender::metrics::probabilistic::log_loss;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use unicode_normalization::UnicodeNormalization;

/// The recipe variant that may be packed for serving, verified and deployed.
pub const PRODUCTION_VARIANT: &str = "production";
/// The test-only variant ([`fixture_bytes`] writes nothing else).
pub const SYNTHETIC_FIXTURE_VARIANT: &str = "synthetic-fixture";
/// A probability row must sum to 1 within this (laya-finetune-gate-v1 `eval_probs_schema`,
/// plan 08-09 `validate_probs`).
pub const PROBS_ROW_SUM_ABS: f64 = 1.0e-5;

// ===========================================================================
// Policy, verdicts, errors
// ===========================================================================

/// Everything the verifier decides WITH, read from the contracts by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyPolicy {
    /// laya-finetune-gate-v1 `constants.gate_min_macro_f1_margin`.
    pub min_macro_f1_margin: f64,
    /// laya-finetune-gate-v1 `constants.gate_max_ece`.
    pub max_ece: f64,
    /// laya-finetune-gate-v1 `constants.ece_bins`.
    pub ece_bins: u64,
    /// laya-finetune-gate-v1 `constants.gate_metric_recompute_abs`.
    pub metric_recompute_abs: f64,
    /// laya-parity-v1 `equations.pack_rescore_probs_abs.float_tolerance`: the FLOOR of every
    /// re-score bound, and the whole bound for a run without a noise record (A1).
    pub rescore_probs_abs: f64,
    /// laya-parity-v1 `constants.pack_rescore_noise_k` (A1).
    pub rescore_noise_k: f64,
    /// laya-parity-v1 `constants.pack_rescore_bound_max_abs`: a derived bound above it is
    /// refused (A1).
    pub rescore_bound_max_abs: f64,
    /// laya-finetune-gate-v1 `constants.calibration_slice_min_per_class`.
    pub calibration_slice_min_per_class: u64,
    /// laya-finetune-gate-v1 `constants.calibration_slice_fraction`: every class keeps at least
    /// `max(calibration_slice_min_per_class, ceil(fraction x n_class))` slice rows (WR-08, the
    /// rule `scripts/laya_train/data.py` `calibration_split` applies).
    pub calibration_slice_fraction: f64,
    /// laya-finetune-gate-v1 `constants.calibration_temp_min` (`calibration_fit_bounded`).
    pub calibration_temp_min: f64,
    /// laya-finetune-gate-v1 `constants.calibration_temp_max` (`calibration_fit_bounded`).
    pub calibration_temp_max: f64,
    /// laya-finetune-gate-v1 `base`: the declared base's identity and pins (D-04).
    pub base: BasePins,
    /// laya-finetune-gate-v1 `recipe`, `early_stopping` and `seed_policy.declared_seed`: the
    /// recipe block a production recipe.json must carry (D-04, V6-d).
    pub recipe: RecipePins,
    /// laya-finetune-gate-v1 `seed_policy.selection` (`median_ece`, A3).
    pub seed_selection_policy: String,
    /// laya-finetune-gate-v1 `seed_policy.variance_seeds` (13, 17, 23).
    pub seed_selection_seeds: Vec<i64>,
    /// laya-finetune-gate-v1 `seed_policy.rank_scale`.
    pub seed_rank_scale: f64,
    /// laya-finetune-gate-v1 `seed_policy.tie_break` (`smaller_seed`).
    pub seed_tie_break: String,
    /// laya-finetune-gate-v1 `demo.criteria_order`: a task whose labels are exactly this list
    /// is the stance task, whose report carries `f_avg` over [`F_AVG_STANCE_LABELS`]; every
    /// other task's `f_avg` is null ([`f_avg_labels`], the rule `train.py` applies).
    pub stance_criteria_order: Vec<String>,
}

/// laya-finetune-gate-v1 `base`: what a production run must declare as its base, and what the
/// base dir must hold (D-04). Every identity field is compared with recipe.json `base`, so the
/// `model.base` string an artifact serves is the contract's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasePins {
    /// `base.family`.
    pub family: String,
    /// `base.checkpoint`.
    pub checkpoint: String,
    /// `base.repo`.
    pub repo: String,
    /// `base.revision`.
    pub revision: String,
    /// `base.model_safetensors_sha256`.
    pub model_safetensors_sha256: String,
    /// `base.encoder_config_sha256`: the base dir's `encoder/config.json`.
    pub encoder_config_sha256: String,
    /// `base.rl_agent_config_sha256`: the base dir's `rl_agent_config.json`.
    pub rl_agent_config_sha256: String,
    /// `base.tokenizer_json_sha256`: the base dir's `tokenizer/tokenizer.json`.
    pub tokenizer_json_sha256: String,
}

/// laya-finetune-gate-v1 `recipe` (D-04), `early_stopping` and `seed_policy.declared_seed`: the
/// values a production recipe.json must equal ([`check_recipe_block`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RecipePins {
    /// `recipe.optimizer`.
    pub optimizer: String,
    /// `recipe.encoder_lr`.
    pub encoder_lr: f64,
    /// `recipe.head_lr`.
    pub head_lr: f64,
    /// `recipe.eta_min`.
    pub eta_min: f64,
    /// `recipe.weight_decay`.
    pub weight_decay: f64,
    /// `recipe.grad_clip`.
    pub grad_clip: f64,
    /// `recipe.batch_size`.
    pub batch_size: u64,
    /// `recipe.proper_reward_w_sph`.
    pub proper_reward_w_sph: f64,
    /// `recipe.proper_reward_w_rps`.
    pub proper_reward_w_rps: f64,
    /// `recipe.schedule_literal`: the recipe.json `schedule` value.
    pub schedule: String,
    /// `recipe.epochs_at_most_16_per_class`.
    pub epochs_at_most_16_per_class: u64,
    /// `recipe.epochs_above_16_min`.
    pub epochs_above_16_min: u64,
    /// `recipe.epochs_above_16_max`.
    pub epochs_above_16_max: u64,
    /// `seed_policy.declared_seed`: recipe.json `seed` (the split seed and first gate seed).
    pub declared_seed: i64,
    /// The `early_stopping` block, as recipe.json must copy it when it declares the rule.
    pub early_stopping: pack::EarlyStoppingDecl,
}

// ===========================================================================
// The contracts, as typed views (the ONE contract-to-policy mapping)
// ===========================================================================

/// laya-finetune-gate-v1, as much of it as the verifier reads. NOT `deny_unknown_fields`: the
/// contract carries prose and other blocks; a view takes only what it needs, and every value it
/// takes is typed, so a malformed one (a non-integer `ece_bins`) fails to deserialize for every
/// caller alike. The caller parses the YAML (`serde_yaml`, a dev/example dependency) into this
/// type; the library stays YAML-free.
#[derive(Debug, Clone, Deserialize)]
pub struct GateContractView {
    /// `constants`.
    pub constants: GateConstantsView,
    /// `base`.
    pub base: BaseBlockView,
    /// `seed_policy`.
    pub seed_policy: SeedPolicyView,
    /// `recipe`.
    pub recipe: RecipeBlockView,
    /// `early_stopping`.
    pub early_stopping: EarlyStoppingBlockView,
    /// `demo`.
    pub demo: DemoBlockView,
}

/// laya-finetune-gate-v1 `demo` (the part the verifier reads).
#[derive(Debug, Clone, Deserialize)]
pub struct DemoBlockView {
    /// `criteria_order`.
    pub criteria_order: Vec<String>,
}

/// laya-finetune-gate-v1 `constants` (the part the verifier reads).
#[derive(Debug, Clone, Deserialize)]
pub struct GateConstantsView {
    /// `gate_min_macro_f1_margin`.
    pub gate_min_macro_f1_margin: f64,
    /// `gate_max_ece`.
    pub gate_max_ece: f64,
    /// `ece_bins` (an integer; `15.0` is refused).
    pub ece_bins: u64,
    /// `gate_metric_recompute_abs`.
    pub gate_metric_recompute_abs: f64,
    /// `calibration_slice_min_per_class`.
    pub calibration_slice_min_per_class: u64,
    /// `calibration_slice_fraction`.
    pub calibration_slice_fraction: f64,
    /// `calibration_temp_min`.
    pub calibration_temp_min: f64,
    /// `calibration_temp_max`.
    pub calibration_temp_max: f64,
}

/// laya-finetune-gate-v1 `base`.
#[derive(Debug, Clone, Deserialize)]
pub struct BaseBlockView {
    /// `family`.
    pub family: String,
    /// `repo`.
    pub repo: String,
    /// `revision`.
    pub revision: String,
    /// `checkpoint`.
    pub checkpoint: String,
    /// `model_safetensors_sha256`.
    pub model_safetensors_sha256: String,
    /// `encoder_config_sha256`.
    pub encoder_config_sha256: String,
    /// `rl_agent_config_sha256`.
    pub rl_agent_config_sha256: String,
    /// `tokenizer_json_sha256`.
    pub tokenizer_json_sha256: String,
}

/// laya-finetune-gate-v1 `recipe` (the part the verifier reads).
#[derive(Debug, Clone, Deserialize)]
pub struct RecipeBlockView {
    /// `optimizer`.
    pub optimizer: String,
    /// `encoder_lr`.
    pub encoder_lr: f64,
    /// `head_lr`.
    pub head_lr: f64,
    /// `eta_min`.
    pub eta_min: f64,
    /// `weight_decay`.
    pub weight_decay: f64,
    /// `grad_clip`.
    pub grad_clip: f64,
    /// `batch_size`.
    pub batch_size: u64,
    /// `proper_reward_w_sph`.
    pub proper_reward_w_sph: f64,
    /// `proper_reward_w_rps`.
    pub proper_reward_w_rps: f64,
    /// `schedule_literal`.
    pub schedule_literal: String,
    /// `epochs_at_most_16_per_class`.
    pub epochs_at_most_16_per_class: u64,
    /// `epochs_above_16_min`.
    pub epochs_above_16_min: u64,
    /// `epochs_above_16_max`.
    pub epochs_above_16_max: u64,
}

/// laya-finetune-gate-v1 `early_stopping` (the fields recipe.json copies).
#[derive(Debug, Clone, Deserialize)]
pub struct EarlyStoppingBlockView {
    /// `monitor`.
    pub monitor: String,
    /// `mode`.
    pub mode: String,
    /// `eval_every_epochs`.
    pub eval_every_epochs: u64,
    /// `first_candidate_epoch`.
    pub first_candidate_epoch: u64,
    /// `patience_epochs`.
    pub patience_epochs: u64,
    /// `min_delta`.
    pub min_delta: f64,
    /// `restore`.
    pub restore: String,
    /// `tie_break`.
    pub tie_break: String,
}

/// laya-finetune-gate-v1 `seed_policy` (the part the verifier reads).
#[derive(Debug, Clone, Deserialize)]
pub struct SeedPolicyView {
    /// `declared_seed`.
    pub declared_seed: i64,
    /// `selection`.
    pub selection: String,
    /// `variance_seeds`.
    pub variance_seeds: Vec<i64>,
    /// `rank_scale`.
    pub rank_scale: f64,
    /// `tie_break`.
    pub tie_break: String,
}

/// laya-parity-v1, as much of it as the verifier reads.
#[derive(Debug, Clone, Deserialize)]
pub struct ParityContractView {
    /// `constants`.
    pub constants: ParityConstantsView,
    /// `equations`.
    pub equations: ParityEquationsView,
}

/// laya-parity-v1 `constants` (the part the verifier reads).
#[derive(Debug, Clone, Deserialize)]
pub struct ParityConstantsView {
    /// `pack_rescore_noise_k`.
    pub pack_rescore_noise_k: f64,
    /// `pack_rescore_bound_max_abs`.
    pub pack_rescore_bound_max_abs: f64,
}

/// laya-parity-v1 `equations` (the part the verifier reads).
#[derive(Debug, Clone, Deserialize)]
pub struct ParityEquationsView {
    /// `pack_rescore_probs_abs`.
    pub pack_rescore_probs_abs: ToleranceView,
}

/// An equation's `float_tolerance`.
#[derive(Debug, Clone, Deserialize)]
pub struct ToleranceView {
    /// `float_tolerance`.
    pub float_tolerance: f64,
}

impl VerifyPolicy {
    /// THE contract-to-policy mapping. `examples/pack_laya.rs` and every test build their
    /// policy here, so no caller can verify under a mapping the CLI does not use.
    #[must_use]
    pub fn from_contract_views(gate: &GateContractView, parity: &ParityContractView) -> Self {
        let c = &gate.constants;
        let b = &gate.base;
        let s = &gate.seed_policy;
        let r = &gate.recipe;
        let es = &gate.early_stopping;
        Self {
            min_macro_f1_margin: c.gate_min_macro_f1_margin,
            max_ece: c.gate_max_ece,
            ece_bins: c.ece_bins,
            metric_recompute_abs: c.gate_metric_recompute_abs,
            rescore_probs_abs: parity.equations.pack_rescore_probs_abs.float_tolerance,
            rescore_noise_k: parity.constants.pack_rescore_noise_k,
            rescore_bound_max_abs: parity.constants.pack_rescore_bound_max_abs,
            calibration_slice_min_per_class: c.calibration_slice_min_per_class,
            calibration_slice_fraction: c.calibration_slice_fraction,
            calibration_temp_min: c.calibration_temp_min,
            calibration_temp_max: c.calibration_temp_max,
            base: BasePins {
                family: b.family.clone(),
                checkpoint: b.checkpoint.clone(),
                repo: b.repo.clone(),
                revision: b.revision.clone(),
                model_safetensors_sha256: b.model_safetensors_sha256.clone(),
                encoder_config_sha256: b.encoder_config_sha256.clone(),
                rl_agent_config_sha256: b.rl_agent_config_sha256.clone(),
                tokenizer_json_sha256: b.tokenizer_json_sha256.clone(),
            },
            recipe: RecipePins {
                optimizer: r.optimizer.clone(),
                encoder_lr: r.encoder_lr,
                head_lr: r.head_lr,
                eta_min: r.eta_min,
                weight_decay: r.weight_decay,
                grad_clip: r.grad_clip,
                batch_size: r.batch_size,
                proper_reward_w_sph: r.proper_reward_w_sph,
                proper_reward_w_rps: r.proper_reward_w_rps,
                schedule: r.schedule_literal.clone(),
                epochs_at_most_16_per_class: r.epochs_at_most_16_per_class,
                epochs_above_16_min: r.epochs_above_16_min,
                epochs_above_16_max: r.epochs_above_16_max,
                declared_seed: s.declared_seed,
                early_stopping: pack::EarlyStoppingDecl {
                    monitor: es.monitor.clone(),
                    mode: es.mode.clone(),
                    eval_every_epochs: es.eval_every_epochs,
                    first_candidate_epoch: es.first_candidate_epoch,
                    patience_epochs: es.patience_epochs,
                    min_delta: es.min_delta,
                    restore: es.restore.clone(),
                    tie_break: es.tie_break.clone(),
                },
            },
            seed_selection_policy: s.selection.clone(),
            seed_selection_seeds: s.variance_seeds.clone(),
            seed_rank_scale: s.rank_scale,
            seed_tie_break: s.tie_break.clone(),
            stance_criteria_order: gate.demo.criteria_order.clone(),
        }
    }
}

/// Which probability file (and which model re-scores it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbsWhich {
    /// `eval-probs.json`, re-scored by the packed artifact.
    FineTuned,
    /// `zero-shot-probs.json`, re-scored by the declared base.
    ZeroShot,
    /// `seeds/seed-<s>/eval-probs.json` (1.4.0): recomputed, never re-scored.
    Seed(i64),
    /// `shift-probs.json` (1.4.0 shift probe): recomputed, never re-scored.
    ShiftFineTuned,
    /// `shift-zero-shot-probs.json` (1.4.0 shift probe): recomputed, never re-scored.
    ShiftZeroShot,
}

impl fmt::Display for ProbsWhich {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FineTuned => f.write_str("fine_tuned"),
            Self::ZeroShot => f.write_str("zero_shot"),
            Self::Seed(s) => write!(f, "seed_{s}"),
            Self::ShiftFineTuned => f.write_str("shift_fine_tuned"),
            Self::ShiftZeroShot => f.write_str("shift_zero_shot"),
        }
    }
}

/// Which side of the base check disagreed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseWhich {
    /// The recipe's declared `base.family` differs from the contract's.
    Family,
    /// The recipe's declared `base.checkpoint` differs from the contract's.
    Checkpoint,
    /// The recipe's declared `base.repo` differs from the contract's.
    Repo,
    /// The recipe's declared `base.revision` differs from the contract's.
    Revision,
    /// The recipe's declared base sha256 differs from the contract's.
    Contract,
    /// The base dir's `model.safetensors` differs from the recipe's declared sha256.
    BaseDir,
    /// The base dir's `tokenizer/tokenizer.json` differs from the contract pin
    /// (`base.tokenizer_json_sha256`) or from the run's (`inputs_sha256.tokenizer_json`; the
    /// trainer copies the base tokenizer unchanged).
    BaseTokenizer,
    /// The base dir's `encoder/config.json` differs from `base.encoder_config_sha256`.
    EncoderConfig,
    /// The base dir's `rl_agent_config.json` differs from `base.rl_agent_config_sha256`.
    AgentConfig,
}

impl fmt::Display for BaseWhich {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Family => "family",
            Self::Checkpoint => "checkpoint",
            Self::Repo => "repo",
            Self::Revision => "revision",
            Self::Contract => "contract",
            Self::BaseDir => "base_dir",
            Self::BaseTokenizer => "base_tokenizer",
            Self::EncoderConfig => "encoder_config",
            Self::AgentConfig => "agent_config",
        })
    }
}

/// One clause of laya-finetune-gate-v1 `gate_pass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateClause {
    /// `ft.macro_f1 - zs.macro_f1 >= gate_min_macro_f1_margin`.
    Margin,
    /// `ft.ece_post <= gate_max_ece`.
    EcePost,
}

impl fmt::Display for GateClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Margin => "margin",
            Self::EcePost => "ece_post",
        })
    }
}

/// Macro-F1 and top-label ECE of one probability set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// aprender-core `macro_f1_f64` (f64, exactly-rounded mean).
    pub macro_f1: f64,
    /// aprender-core `expected_calibration_error_top_label_f64` (f64, exactly-rounded sums).
    pub ece: f64,
}

/// The gate metrics as RECOMPUTED in Rust.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recomputed {
    /// Zero-shot macro-F1.
    pub zs_macro_f1: f64,
    /// Zero-shot ECE.
    pub zs_ece: f64,
    /// Fine-tuned macro-F1.
    pub ft_macro_f1: f64,
    /// Fine-tuned ECE after calibration (the eval probabilities carry the applied T).
    pub ece_post: f64,
    /// Fine-tuned eval NLL, `mean -ln p[y]` of the calibrated probabilities (aprender-core
    /// `log_loss` of the true-class probability, the house implementation).
    pub ft_nll: f64,
    /// `ft_macro_f1 - zs_macro_f1`.
    pub margin: f64,
}

/// One re-score's agreement with its probability file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RescoreStats {
    /// `max |p_rust - p_file|` over every row and class (NaN-propagating).
    pub max_abs: f64,
    /// Rows whose argmax agrees.
    pub argmax_agree: usize,
    /// Rows re-scored.
    pub n: usize,
}

/// The re-score bound one (checkpoint, set) pair is held to (laya-parity-v1 A1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RescoreBound {
    /// Which re-score.
    pub which: ProbsWhich,
    /// `noise(c, s)` RECOMPUTED from the float64 record; `None` without a record.
    pub noise: Option<f64>,
    /// `max(floor, k x noise)`, or the floor without a record.
    pub bound: f64,
}

/// Evidence of an accepted run (decide-apr-v1 `deploy_eligibility`).
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyReport {
    /// sha256 of the whole verified artifact (D-11).
    pub artifact_sha256: String,
    /// Fine-tuned re-score maximum.
    pub rescore_max_abs: f64,
    /// Zero-shot re-score maximum.
    pub zs_rescore_max_abs: f64,
    /// The bound the fine-tuned re-score was held to (A1).
    pub rescore_bound: f64,
    /// The bound the zero-shot re-score was held to (A1).
    pub zs_rescore_bound: f64,
    /// The recomputed fine-tuned noise, `None` without a record.
    pub noise: Option<f64>,
    /// The recomputed zero-shot noise, `None` without a record.
    pub zs_noise: Option<f64>,
    /// The re-derived shipped (median) seed; always `Some` for an accepted run (1.4.0).
    pub shipped_seed: Option<i64>,
    /// Fine-tuned argmax agreement.
    pub argmax_agree: usize,
    /// Eval rows.
    pub n: usize,
    /// The recomputed gate metrics.
    pub recomputed: Recomputed,
    /// Always `true`: a report exists only for an accepted run.
    pub deploy_eligible: bool,
}

/// The evidence a gate refusal carries (boxed in [`VerifyError::GateFailed`]).
#[derive(Debug, Clone, PartialEq)]
pub struct GateFailure {
    /// The failed clauses, in `gate_pass` order.
    pub clauses: Vec<GateClause>,
    /// The recomputed metrics that decided it.
    pub recomputed: Recomputed,
    /// Fine-tuned re-score maximum.
    pub rescore_max_abs: f64,
    /// Zero-shot re-score maximum.
    pub zs_rescore_max_abs: f64,
    /// The bound the fine-tuned re-score was held to (A1).
    pub rescore_bound: f64,
    /// The bound the zero-shot re-score was held to (A1).
    pub zs_rescore_bound: f64,
    /// The recomputed fine-tuned noise, `None` without a record.
    pub noise: Option<f64>,
    /// The recomputed zero-shot noise, `None` without a record.
    pub zs_noise: Option<f64>,
    /// The re-derived shipped seed; `None` for a legacy run.
    pub shipped_seed: Option<i64>,
    /// Fine-tuned argmax agreement.
    pub argmax_agree: usize,
    /// Eval rows.
    pub n: usize,
    /// sha256 of the artifact that was verified and refused (never written by `pack`).
    pub artifact_sha256: String,
}

/// Why a run or artifact was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyError {
    /// The run dir could not be read or packed.
    Pack(PackError),
    /// The artifact was refused by the decide-apr-v1 load ladder.
    Artifact(ArtifactError),
    /// A file could not be read or written.
    Read {
        /// The path.
        path: PathBuf,
        /// The I/O error.
        reason: String,
    },
    /// A data-dir row is malformed.
    DataInvalid {
        /// `train_jsonl` or `eval_jsonl`.
        file: &'static str,
        /// 1-based line.
        line: usize,
        /// What is wrong.
        why: String,
    },
    /// A model refused to score the eval rows.
    Score {
        /// Which model.
        which: ProbsWhich,
        /// The refusal.
        reason: String,
    },
    /// The artifact's bytes are not the bytes the given run and data dirs pack to: a weight,
    /// a config blob or the manifest differs from the run's checkpoint (packing is
    /// byte-deterministic, so an honest file of this run is byte-identical).
    ArtifactNotFromRun {
        /// sha256 of the file.
        file_sha256: String,
        /// sha256 of the bytes the run packs to.
        packed_sha256: String,
    },
    /// The artifact's manifest does not describe the given run and data dirs.
    ManifestMismatch {
        /// The manifest field.
        field: &'static str,
    },
    /// The recipe variant is not `production` (laya-finetune-gate-v1 `synthetic_not_deployable`).
    SyntheticNotDeployable {
        /// The variant found.
        variant: String,
    },
    /// [`fixture_bytes`] was asked to pack a run that is not a synthetic fixture.
    NotSyntheticFixture {
        /// The variant found.
        variant: String,
    },
    /// The base is not the contract's.
    BaseMismatch {
        /// Which comparison failed.
        which: BaseWhich,
        /// The value it had to equal.
        expected: String,
        /// The value found.
        observed: String,
    },
    /// A file's sha256 disagrees with the report.
    InputHashMismatch {
        /// The file (`eval_jsonl`, `eval_probs_json`, ...).
        file: &'static str,
        /// Recorded in the report.
        recorded: String,
        /// Recomputed from the file.
        observed: String,
    },
    /// An eval text equals a train text after normalization.
    SplitOverlap {
        /// 0-based eval row.
        eval_row: usize,
        /// 0-based train row.
        train_row: usize,
    },
    /// One normalized train text carries two labels.
    ConflictingLabels {
        /// The 0-based train rows of that text.
        rows: Vec<usize>,
    },
    /// The calibration slice is malformed or not group-disjoint from fit.
    SliceInvalid {
        /// What is wrong.
        why: String,
    },
    /// A probability file does not hold exactly one row per eval row.
    ProbsRowCoverage {
        /// Which file.
        which: ProbsWhich,
        /// What is wrong.
        why: String,
    },
    /// A probability file row (or the file) is invalid.
    ProbsInvalid {
        /// Which file.
        which: ProbsWhich,
        /// The row index, or `None` for a file-level defect.
        row: Option<usize>,
        /// What is wrong.
        why: String,
    },
    /// A Rust re-score differs from the file by more than its set's bound (A1).
    RescoreDrift {
        /// Which re-score.
        which: ProbsWhich,
        /// The first row over the bar.
        row: usize,
        /// The maximum over every row.
        max_abs: f64,
        /// The bound used: the floor, or the noise-referenced bound.
        bound: f64,
    },
    /// `rescore-noise.json` is malformed or disagrees with the contract (A1).
    RescoreNoiseInvalid {
        /// The set, when the defect is inside one.
        which: Option<ProbsWhich>,
        /// The 0-based position, when the defect is a row.
        row: Option<usize>,
        /// The record field.
        field: &'static str,
        /// What is wrong.
        why: String,
    },
    /// A value `rescore-noise.json` REPORTS differs from its Rust recomputation (A1).
    RescoreNoiseMismatch {
        /// The set.
        which: ProbsWhich,
        /// `max_abs`, `bound` or `argmax_agree`.
        field: &'static str,
        /// The record's value.
        reported: f64,
        /// The Rust recomputation.
        recomputed: f64,
    },
    /// The derived bound exceeds laya-parity-v1 `pack_rescore_bound_max_abs` (A1).
    RescoreBoundCeiling {
        /// The set.
        which: ProbsWhich,
        /// The derived bound.
        bound: f64,
        /// The contract ceiling.
        ceiling: f64,
    },
    /// A float64 row's argmax differs from the stored float32 row's (A1).
    NoiseArgmaxFlip {
        /// The set.
        which: ProbsWhich,
        /// The 0-based eval row.
        row: usize,
        /// `argmax(p_torch32)`.
        torch: usize,
        /// `argmax(p_f64)`.
        reference: usize,
    },
    /// A Rust re-score's argmax differs from the file's.
    ArgmaxDrift {
        /// Which re-score.
        which: ProbsWhich,
        /// The first disagreeing row.
        row: usize,
    },
    /// A reported metric is further than `gate_metric_recompute_abs` from its recomputation.
    ReportedMetricMismatch {
        /// The seed, for a `seeds.per_seed` field (1.4.0); `None` for a top-level field.
        seed: Option<i64>,
        /// The report field (`fine_tuned.macro_f1`, `per_seed.ece_post`, ...).
        field: &'static str,
        /// The report's value.
        reported: f64,
        /// The Rust recomputation.
        recomputed: f64,
    },
    /// The report's thresholds are not the contract's.
    ThresholdMismatch {
        /// The threshold.
        field: &'static str,
        /// The report's value.
        report: f64,
        /// The contract's value.
        policy: f64,
    },
    /// The reported `pass` disagrees with the recomputed one.
    PassDisagrees {
        /// The report's `pass`.
        reported: bool,
        /// The recomputed `pass`.
        recomputed: bool,
    },
    /// The recomputed gate FAILED (the only variant with exit code 3).
    GateFailed(Box<GateFailure>),
    /// The report's seed selection is not the one the files support (A3): a non-median
    /// `shipped`, a per_seed row that does not bind the shipped checkpoint or eval file, a rank
    /// key or pass that differs from its recomputation, or a malformed seeds block.
    SeedPolicyViolated {
        /// The report field.
        field: &'static str,
        /// What is wrong, naming the seeds.
        why: String,
    },
    /// recipe.json `seed_selection` differs from the contract's `seed_policy` (A3).
    SeedPolicyMismatch {
        /// `policy`, `seeds`, `rank_scale` or `tie_break`.
        field: &'static str,
        /// The recipe's value.
        recipe: String,
        /// The contract's value.
        contract: String,
    },
    /// A legacy run (no `seed_selection`) whose gate PASSED: never deploy-eligible under 1.4.0
    /// (`seed_policy.legacy_rule`). Decided after the re-scores and the gate.
    SeedPolicyMissing,
    /// The shift probe differs from its recomputation or is malformed (A2).
    ShiftProbeMismatch {
        /// The report field or file.
        field: &'static str,
        /// What is wrong.
        why: String,
    },
    /// A gate-report record disagrees with the relation the contract states for it
    /// (`calibration_fit_bounded`: `t_applied = clamp(t_fitted)`, `clamp_hit = t_fitted at a
    /// bound`; `device_recorded`: `device_is_cpu = (device_used == "cpu")`).
    RecordMismatch {
        /// The report field.
        field: &'static str,
        /// What is wrong.
        why: String,
    },
    /// recipe.json differs from the contract's recipe block (D-04, V6-d): a recipe value, the
    /// schedule literal, the seed, the epoch rule, an `early_stopping` field, or
    /// `shots_per_class` against the data dir.
    RecipeMismatch {
        /// The recipe.json field (`encoder_lr`, `early_stopping.min_delta`, `epochs`, ...).
        field: &'static str,
        /// The recipe's value.
        recipe: String,
        /// What the contract (or the data dir) requires.
        contract: String,
    },
}

impl VerifyError {
    /// The back-office exit code: 3 for a failed gate, 2 for every other refusal.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        if matches!(self, Self::GateFailed(_)) {
            3
        } else {
            2
        }
    }

    /// The variant name, as the CLI prints it after `REFUSED`.
    #[must_use]
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::Pack(_) => "Pack",
            Self::Artifact(_) => "Artifact",
            Self::Read { .. } => "Read",
            Self::DataInvalid { .. } => "DataInvalid",
            Self::Score { .. } => "Score",
            Self::ArtifactNotFromRun { .. } => "ArtifactNotFromRun",
            Self::ManifestMismatch { .. } => "ManifestMismatch",
            Self::SyntheticNotDeployable { .. } => "SyntheticNotDeployable",
            Self::NotSyntheticFixture { .. } => "NotSyntheticFixture",
            Self::BaseMismatch { .. } => "BaseMismatch",
            Self::InputHashMismatch { .. } => "InputHashMismatch",
            Self::SplitOverlap { .. } => "SplitOverlap",
            Self::ConflictingLabels { .. } => "ConflictingLabels",
            Self::SliceInvalid { .. } => "SliceInvalid",
            Self::ProbsRowCoverage { .. } => "ProbsRowCoverage",
            Self::ProbsInvalid { .. } => "ProbsInvalid",
            Self::RescoreDrift { .. } => "RescoreDrift",
            Self::RescoreNoiseInvalid { .. } => "RescoreNoiseInvalid",
            Self::RescoreNoiseMismatch { .. } => "RescoreNoiseMismatch",
            Self::RescoreBoundCeiling { .. } => "RescoreBoundCeiling",
            Self::NoiseArgmaxFlip { .. } => "NoiseArgmaxFlip",
            Self::ArgmaxDrift { .. } => "ArgmaxDrift",
            Self::ReportedMetricMismatch { .. } => "ReportedMetricMismatch",
            Self::ThresholdMismatch { .. } => "ThresholdMismatch",
            Self::PassDisagrees { .. } => "PassDisagrees",
            Self::GateFailed(_) => "GateFailed",
            Self::SeedPolicyViolated { .. } => "SeedPolicyViolated",
            Self::SeedPolicyMismatch { .. } => "SeedPolicyMismatch",
            Self::SeedPolicyMissing => "SeedPolicyMissing",
            Self::ShiftProbeMismatch { .. } => "ShiftProbeMismatch",
            Self::RecipeMismatch { .. } => "RecipeMismatch",
            Self::RecordMismatch { .. } => "RecordMismatch",
        }
    }
}

/// An optional f64 as the CLI prints it (`null` when absent).
#[must_use]
pub fn opt_f64(v: Option<f64>) -> String {
    v.map_or_else(|| "null".to_string(), |x| x.to_string())
}

/// An optional seed as the CLI prints it (`null` when absent).
#[must_use]
pub fn opt_i64(v: Option<i64>) -> String {
    v.map_or_else(|| "null".to_string(), |x| x.to_string())
}

fn clause_list(clauses: &[GateClause]) -> String {
    clauses
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

impl fmt::Display for GateFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = &self.recomputed;
        write!(
            f,
            "clauses=[{}] zs_macro_f1={} ft_macro_f1={} margin={} ece_post={} \
             rescore_max_abs={} zs_rescore_max_abs={} rescore_bound={} zs_rescore_bound={} \
             noise={} zs_noise={} shipped_seed={} argmax={}/{} packed_sha256={}",
            clause_list(&self.clauses),
            r.zs_macro_f1,
            r.ft_macro_f1,
            r.margin,
            r.ece_post,
            self.rescore_max_abs,
            self.zs_rescore_max_abs,
            self.rescore_bound,
            self.zs_rescore_bound,
            opt_f64(self.noise),
            opt_f64(self.zs_noise),
            opt_i64(self.shipped_seed),
            self.argmax_agree,
            self.n,
            self.artifact_sha256
        )
    }
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pack(e) => write!(f, "{e}"),
            Self::Artifact(e) => write!(f, "load ladder: {e}"),
            Self::Read { path, reason } => write!(f, "{}: {reason}", path.display()),
            Self::DataInvalid { file, line, why } => write!(f, "{file} line {line}: {why}"),
            Self::Score { which, reason } => write!(f, "{which} re-score refused: {reason}"),
            Self::ArtifactNotFromRun {
                file_sha256,
                packed_sha256,
            } => write!(
                f,
                "the artifact (sha256 {file_sha256}) is not the bytes the given run/data dirs \
                 pack to (sha256 {packed_sha256}): a weight, config blob or manifest byte differs"
            ),
            Self::ManifestMismatch { field } => write!(
                f,
                "the artifact's manifest {field} does not describe the given run/data dirs"
            ),
            Self::SyntheticNotDeployable { variant } => {
                write!(f, "recipe variant {variant:?} is not deployable")
            }
            Self::NotSyntheticFixture { variant } => write!(
                f,
                "recipe variant {variant:?} is not {SYNTHETIC_FIXTURE_VARIANT:?}; pack-fixture writes nothing else"
            ),
            Self::BaseMismatch {
                which,
                expected,
                observed,
            } => write!(f, "which={which} expected={expected} observed={observed}"),
            Self::InputHashMismatch {
                file,
                recorded,
                observed,
            } => write!(f, "file={file} recorded={recorded} observed={observed}"),
            Self::SplitOverlap {
                eval_row,
                train_row,
            } => write!(
                f,
                "eval row {eval_row} equals train row {train_row} after NFC/trim/whitespace collapse"
            ),
            Self::ConflictingLabels { rows } => write!(
                f,
                "train rows {rows:?} carry one normalized text under different labels"
            ),
            Self::SliceInvalid { why } => write!(f, "{why}"),
            Self::ProbsRowCoverage { which, why } => write!(f, "{which}: {why}"),
            Self::ProbsInvalid { which, row, why } => match row {
                Some(r) => write!(f, "{which} row {r}: {why}"),
                None => write!(f, "{which}: {why}"),
            },
            Self::RescoreDrift {
                which,
                row,
                max_abs,
                bound,
            } => write!(f, "which={which} row={row} max_abs={max_abs} bound={bound}"),
            Self::RescoreNoiseInvalid {
                which,
                row,
                field,
                why,
            } => {
                write!(f, "rescore-noise.json")?;
                if let Some(w) = which {
                    write!(f, " set={w}")?;
                }
                if let Some(r) = row {
                    write!(f, " row={r}")?;
                }
                write!(f, " field={field}: {why}")
            }
            Self::RescoreNoiseMismatch {
                which,
                field,
                reported,
                recomputed,
            } => write!(
                f,
                "rescore-noise.json set={which} field={field} reported={reported} recomputed={recomputed}"
            ),
            Self::RescoreBoundCeiling {
                which,
                bound,
                ceiling,
            } => write!(
                f,
                "set={which} derived bound={bound} exceeds pack_rescore_bound_max_abs={ceiling}"
            ),
            Self::NoiseArgmaxFlip {
                which,
                row,
                torch,
                reference,
            } => write!(
                f,
                "rescore-noise.json set={which} row={row}: argmax(p_f64)={reference} but argmax(p_torch32)={torch}"
            ),
            Self::ArgmaxDrift { which, row } => write!(f, "which={which} row={row}"),
            Self::ReportedMetricMismatch {
                seed,
                field,
                reported,
                recomputed,
            } => {
                if let Some(s) = seed {
                    write!(f, "seed={s} ")?;
                }
                write!(f, "field={field} reported={reported} recomputed={recomputed}")
            }
            Self::ThresholdMismatch {
                field,
                report,
                policy,
            } => write!(f, "field={field} report={report} contract={policy}"),
            Self::PassDisagrees {
                reported,
                recomputed,
            } => write!(f, "reported pass={reported} recomputed pass={recomputed}"),
            Self::GateFailed(g) => write!(f, "{g}"),
            Self::SeedPolicyViolated { field, why }
            | Self::ShiftProbeMismatch { field, why }
            | Self::RecordMismatch { field, why } => {
                write!(f, "field={field}: {why}")
            }
            Self::SeedPolicyMismatch {
                field,
                recipe,
                contract,
            } => write!(
                f,
                "recipe.json seed_selection.{field} is {recipe}, the contract's seed_policy says {contract}"
            ),
            Self::RecipeMismatch {
                field,
                recipe,
                contract,
            } => write!(
                f,
                "recipe.json {field} is {recipe}, the contract requires {contract}"
            ),
            Self::SeedPolicyMissing => f.write_str(
                "recipe.json carries no seed_selection: a legacy (1.x) run is never deploy-eligible \
                 under laya-finetune-gate-v1 1.4.0 (seed_policy.legacy_rule); its re-scores and gate passed",
            ),
        }
    }
}

impl std::error::Error for VerifyError {}

/// The report file a [`PackError::ReportHashMismatch`] names, as an [`VerifyError::InputHashMismatch`]
/// `file` value.
fn hash_file_name(what: &str) -> &'static str {
    match what {
        "recipe_id" => "recipe_json",
        "probes_sha256" => "probes_json",
        "eval_probs_sha256" => "eval_probs_json",
        "zero_shot_probs_sha256" => "zero_shot_probs_json",
        "inputs_sha256.task_json" => "task_json",
        "inputs_sha256.train_jsonl" => "train_jsonl",
        "inputs_sha256.eval_jsonl" => "eval_jsonl",
        "inputs_sha256.tokenizer_json" => "tokenizer_json",
        "inputs_sha256.base_model" => "base_model",
        "rescore_noise_sha256" => "rescore_noise_json",
        "shift_probe.probs_sha256" => "shift_probs_json",
        "shift_probe.zero_shot_probs_sha256" => "shift_zero_shot_probs_json",
        _ => "unknown",
    }
}

impl From<PackError> for VerifyError {
    fn from(e: PackError) -> Self {
        match e {
            PackError::ReportHashMismatch {
                what,
                recorded,
                observed,
            } => Self::InputHashMismatch {
                file: hash_file_name(what),
                recorded,
                observed,
            },
            other => Self::Pack(other),
        }
    }
}

impl From<ArtifactError> for VerifyError {
    fn from(e: ArtifactError) -> Self {
        Self::Artifact(e)
    }
}

// ===========================================================================
// The data dir
// ===========================================================================

/// One `train.jsonl` / `eval.jsonl` row (decide-apr-v1 `train_row_schema`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RowJson {
    text: String,
    label: String,
}

/// A data-dir row with its label resolved to the task's label index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataRow {
    /// The text, exactly as stored.
    pub text: String,
    /// Label index in task criteria order (D-05).
    pub label: usize,
}

/// A parsed data dir (laya-finetune-gate-v1 `run_dir_layout.data_dir`).
#[derive(Debug, Clone)]
pub struct DataDir {
    /// `task.json`.
    pub task: Task,
    /// `train.jsonl`.
    pub train: Vec<DataRow>,
    /// `eval.jsonl`.
    pub eval: Vec<DataRow>,
    /// `shift.jsonl`, the OPTIONAL 1.4.0 shift probe (A2); `None` when the file is absent.
    pub shift: Option<Vec<DataRow>>,
    task_bytes: Vec<u8>,
    train_bytes: Vec<u8>,
    eval_bytes: Vec<u8>,
    shift_bytes: Option<Vec<u8>>,
}

fn read_path(path: &Path) -> Result<Vec<u8>, VerifyError> {
    std::fs::read(path).map_err(|e| VerifyError::Read {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })
}

fn parse_rows(
    file: &'static str,
    bytes: &[u8],
    labels: &[&str],
) -> Result<Vec<DataRow>, VerifyError> {
    let bad = |line: usize, why: String| VerifyError::DataInvalid { file, line, why };
    let text = std::str::from_utf8(bytes).map_err(|e| bad(0, e.to_string()))?;
    let mut rows = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        if raw.trim().is_empty() {
            return Err(bad(line, "blank line".into()));
        }
        let row: RowJson = serde_json::from_str(raw).map_err(|e| bad(line, e.to_string()))?;
        let label = labels
            .iter()
            .position(|l| *l == row.label)
            .ok_or_else(|| bad(line, format!("label {:?} is not a criterion", row.label)))?;
        rows.push(DataRow {
            text: row.text,
            label,
        });
    }
    if rows.is_empty() {
        return Err(bad(0, "no rows".into()));
    }
    Ok(rows)
}

/// Read and parse `task.json`, `train.jsonl`, `eval.jsonl` and, when present, the optional
/// `shift.jsonl` (1.4.0; hash-checked by [`check_shift_probe`] when the report names it).
///
/// # Errors
///
/// [`VerifyError::Read`], [`VerifyError::Pack`] (a task the D-05 parser refuses, as
/// [`PackError::Schema`]) or [`VerifyError::DataInvalid`].
pub fn read_data_dir(data_dir: &Path) -> Result<DataDir, VerifyError> {
    let task_bytes = read_path(&data_dir.join("task.json"))?;
    let task = Task::from_slice(&task_bytes).map_err(|e| {
        VerifyError::Pack(PackError::Schema {
            file: "task.json",
            reason: e.to_string(),
        })
    })?;
    let train_bytes = read_path(&data_dir.join("train.jsonl"))?;
    let eval_bytes = read_path(&data_dir.join("eval.jsonl"))?;
    let labels = task.labels();
    let train = parse_rows("train_jsonl", &train_bytes, &labels)?;
    let eval = parse_rows("eval_jsonl", &eval_bytes, &labels)?;
    let shift_path = data_dir.join("shift.jsonl");
    let shift_bytes = if shift_path.exists() {
        Some(read_path(&shift_path)?)
    } else {
        None
    };
    let shift = shift_bytes
        .as_deref()
        .map(|b| parse_rows("shift_jsonl", b, &labels))
        .transpose()?;
    Ok(DataDir {
        task,
        train,
        eval,
        shift,
        task_bytes,
        train_bytes,
        eval_bytes,
        shift_bytes,
    })
}

/// laya-finetune-gate-v1 `eval_set.generic_rule`: every criterion has at least one eval row.
/// The gate's macro-F1 averages over the labels present in `y ∪ pred`, so with a criterion
/// absent from eval the margin is decided by which model happens to predict it — a fine-tune
/// worse than zero-shot on every present class can pass. The mirror of `data.load_rows`'s
/// `eval-class-coverage`.
///
/// # Errors
///
/// [`VerifyError::DataInvalid`] on `eval_jsonl` naming the first missing criterion.
pub fn check_eval_coverage(eval: &[DataRow], labels: &[String]) -> Result<(), VerifyError> {
    match (0..labels.len()).find(|&c| !eval.iter().any(|r| r.label == c)) {
        None => Ok(()),
        Some(c) => Err(VerifyError::DataInvalid {
            file: "eval_jsonl",
            line: 0,
            why: format!(
                "criterion {:?} has no eval row: every criterion needs one for the gate's \
                 macro-F1 to compare the same labels",
                labels[c]
            ),
        }),
    }
}

// ===========================================================================
// Cheap checks
// ===========================================================================

/// laya-finetune-gate-v1 `synthetic_not_deployable`: only `production` may be served.
///
/// # Errors
///
/// [`VerifyError::SyntheticNotDeployable`] naming the variant.
#[provable_contracts_macros::contract(
    "laya-finetune-gate-v1",
    equation = "synthetic_not_deployable"
)]
pub fn check_variant(recipe: &Recipe) -> Result<(), VerifyError> {
    if recipe.variant == PRODUCTION_VARIANT {
        Ok(())
    } else {
        Err(VerifyError::SyntheticNotDeployable {
            variant: recipe.variant.clone(),
        })
    }
}

fn recipe_mismatch(field: &'static str, recipe: String, contract: String) -> VerifyError {
    VerifyError::RecipeMismatch {
        field,
        recipe,
        contract,
    }
}

/// `recipe == contract`, bit for bit.
fn recipe_f64(field: &'static str, recipe: f64, contract: f64) -> Result<(), VerifyError> {
    if recipe.to_bits() == contract.to_bits() {
        Ok(())
    } else {
        Err(recipe_mismatch(
            field,
            recipe.to_string(),
            contract.to_string(),
        ))
    }
}

/// `recipe == contract` for a string or integer field.
fn recipe_eq<T: PartialEq + fmt::Debug>(
    field: &'static str,
    recipe: &T,
    contract: &T,
) -> Result<(), VerifyError> {
    if recipe == contract {
        Ok(())
    } else {
        Err(recipe_mismatch(
            field,
            format!("{recipe:?}"),
            format!("{contract:?}"),
        ))
    }
}

/// recipe.json equals the contract's recipe block (D-04, V6-d): `optimizer`, every learning
/// rate, `eta_min`, `weight_decay`, `grad_clip`, `batch_size` and both reward weights (floats
/// bit for bit), the `schedule` literal, `seed == seed_policy.declared_seed`, the epoch rule
/// (`shots_per_class <= 16` -> exactly `epochs_at_most_16_per_class`; above 16 -> `epochs` in
/// `[epochs_above_16_min, epochs_above_16_max]`), and — when recipe.json declares
/// `early_stopping` — every field of it equal to the contract's `early_stopping` block (ABSENT is
/// the declared `fixed_epochs` rule, allowed). Files only: decided before any model is built.
///
/// # Errors
///
/// [`VerifyError::RecipeMismatch`] naming the first field that differs.
pub fn check_recipe_block(recipe: &Recipe, policy: &VerifyPolicy) -> Result<(), VerifyError> {
    let c = &policy.recipe;
    recipe_eq("optimizer", &recipe.optimizer, &c.optimizer)?;
    recipe_f64("encoder_lr", recipe.encoder_lr, c.encoder_lr)?;
    recipe_f64("head_lr", recipe.head_lr, c.head_lr)?;
    recipe_f64("eta_min", recipe.eta_min, c.eta_min)?;
    recipe_f64("weight_decay", recipe.weight_decay, c.weight_decay)?;
    recipe_f64("grad_clip", recipe.grad_clip, c.grad_clip)?;
    recipe_eq("batch_size", &recipe.batch_size, &c.batch_size)?;
    recipe_f64(
        "proper_reward_w_sph",
        recipe.proper_reward_w_sph,
        c.proper_reward_w_sph,
    )?;
    recipe_f64(
        "proper_reward_w_rps",
        recipe.proper_reward_w_rps,
        c.proper_reward_w_rps,
    )?;
    recipe_eq("schedule", &recipe.schedule, &c.schedule)?;
    recipe_eq("seed", &recipe.seed, &c.declared_seed)?;
    let (lo, hi) = if recipe.shots_per_class <= 16 {
        (c.epochs_at_most_16_per_class, c.epochs_at_most_16_per_class)
    } else {
        (c.epochs_above_16_min, c.epochs_above_16_max)
    };
    if !(lo..=hi).contains(&recipe.epochs) {
        return Err(recipe_mismatch(
            "epochs",
            recipe.epochs.to_string(),
            format!(
                "an epoch count in [{lo}, {hi}] at {} shots/class (recipe.epoch_rule)",
                recipe.shots_per_class
            ),
        ));
    }
    if let Some(es) = &recipe.early_stopping {
        let c = &c.early_stopping;
        recipe_eq("early_stopping.monitor", &es.monitor, &c.monitor)?;
        recipe_eq("early_stopping.mode", &es.mode, &c.mode)?;
        recipe_eq(
            "early_stopping.eval_every_epochs",
            &es.eval_every_epochs,
            &c.eval_every_epochs,
        )?;
        recipe_eq(
            "early_stopping.first_candidate_epoch",
            &es.first_candidate_epoch,
            &c.first_candidate_epoch,
        )?;
        recipe_eq(
            "early_stopping.patience_epochs",
            &es.patience_epochs,
            &c.patience_epochs,
        )?;
        recipe_f64("early_stopping.min_delta", es.min_delta, c.min_delta)?;
        recipe_eq("early_stopping.restore", &es.restore, &c.restore)?;
        recipe_eq("early_stopping.tie_break", &es.tie_break, &c.tie_break)?;
    }
    Ok(())
}

/// recipe.json `shots_per_class` is what the trainer derives from the data dir: the largest
/// class count of train.jsonl (`scripts/laya_train/train.py`), so the epoch rule is applied to
/// the shots the run actually had.
///
/// # Errors
///
/// [`VerifyError::RecipeMismatch`] on `shots_per_class`.
fn check_recipe_shots(recipe: &Recipe, train: &[DataRow], k: usize) -> Result<(), VerifyError> {
    let shots = (0..k)
        .map(|c| train.iter().filter(|r| r.label == c).count() as u64)
        .max()
        .unwrap_or(0);
    if recipe.shots_per_class == shots {
        Ok(())
    } else {
        Err(recipe_mismatch(
            "shots_per_class",
            recipe.shots_per_class.to_string(),
            format!("{shots}, the largest class count of train.jsonl"),
        ))
    }
}

/// Streamed sha256 of a file (the base `model.safetensors` is 0.8 GB; the HF cache's symlink
/// is followed).
fn sha256_file(path: &Path) -> Result<String, VerifyError> {
    let io = |e: std::io::Error| VerifyError::Read {
        path: path.to_path_buf(),
        reason: e.to_string(),
    };
    let mut file = std::fs::File::open(path).map_err(io)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(io)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// The declared base must be the contract's — every identity field (`family`, `checkpoint`,
/// `repo`, `revision`, in that order, then `sha256`), so the `model.base` string an artifact
/// serves is the contract's — and the base dir must hold exactly it.
///
/// # Errors
///
/// [`VerifyError::BaseMismatch`] naming the identity field (`family`, `checkpoint`, `repo`,
/// `revision`), `contract` (the recipe declares another sha256) or `base_dir` (the directory's
/// `model.safetensors` is not the declared one).
pub fn check_base(
    base_dir: &Path,
    recipe: &Recipe,
    policy: &VerifyPolicy,
) -> Result<(), VerifyError> {
    let d = &recipe.base;
    let p = &policy.base;
    for (which, expected, observed) in [
        (BaseWhich::Family, &p.family, &d.family),
        (BaseWhich::Checkpoint, &p.checkpoint, &d.checkpoint),
        (BaseWhich::Repo, &p.repo, &d.repo),
        (BaseWhich::Revision, &p.revision, &d.revision),
        (BaseWhich::Contract, &p.model_safetensors_sha256, &d.sha256),
    ] {
        if expected != observed {
            return Err(VerifyError::BaseMismatch {
                which,
                expected: expected.clone(),
                observed: observed.clone(),
            });
        }
    }
    let observed = sha256_file(&base_dir.join("model.safetensors"))?;
    if observed != recipe.base.sha256 {
        return Err(VerifyError::BaseMismatch {
            which: BaseWhich::BaseDir,
            expected: recipe.base.sha256.clone(),
            observed,
        });
    }
    Ok(())
}

/// One base-dir file's bytes against the sha256 it must have.
fn base_file_matches(which: BaseWhich, bytes: &[u8], expected: &str) -> Result<(), VerifyError> {
    let observed = sha256_hex(bytes);
    if observed == expected {
        Ok(())
    } else {
        Err(VerifyError::BaseMismatch {
            which,
            expected: expected.to_string(),
            observed,
        })
    }
}

/// The declared base as the zero-shot scorer, read ONCE and bound before it scores: each of the
/// four base-dir files is read once and those exact bytes are hashed and then built from, so
/// nothing can swap a file between the hash and the load (V6-c). `model.safetensors` must hash to
/// the declared (= contract) sha256, `encoder/config.json` and `rl_agent_config.json` to the
/// contract's `base` pins, and `tokenizer/tokenizer.json` BOTH to the contract pin and to the
/// run's `inputs_sha256.tokenizer_json` (the trainer copies the base tokenizer into the
/// checkpoint unchanged), so the baseline is not tokenized differently from the fine-tune it is
/// compared with.
///
/// # Errors
///
/// [`VerifyError::Read`], [`VerifyError::BaseMismatch`] (`base_dir`, `encoder_config`,
/// `agent_config`, `base_tokenizer`) or a [`VerifyError::Pack`] build refusal.
fn load_declared_base(
    base_dir: &Path,
    task: Task,
    model_sha256: &str,
    pins: &BasePins,
    run_tokenizer_sha256: &str,
) -> Result<crate::laya::Laya, VerifyError> {
    let model = read_path(&base_dir.join("model.safetensors"))?;
    base_file_matches(BaseWhich::BaseDir, &model, model_sha256)?;
    let encoder = read_path(&base_dir.join("encoder/config.json"))?;
    base_file_matches(
        BaseWhich::EncoderConfig,
        &encoder,
        &pins.encoder_config_sha256,
    )?;
    let agent = read_path(&base_dir.join("rl_agent_config.json"))?;
    base_file_matches(BaseWhich::AgentConfig, &agent, &pins.rl_agent_config_sha256)?;
    let tokenizer = read_path(&base_dir.join("tokenizer/tokenizer.json"))?;
    base_file_matches(
        BaseWhich::BaseTokenizer,
        &tokenizer,
        &pins.tokenizer_json_sha256,
    )?;
    base_file_matches(BaseWhich::BaseTokenizer, &tokenizer, run_tokenizer_sha256)?;
    Ok(pack::scorer_from_parts(
        &model, &encoder, &agent, &tokenizer, task,
    )?)
}

fn hash_matches(file: &'static str, recorded: &str, bytes: &[u8]) -> Result<(), VerifyError> {
    let observed = sha256_hex(bytes);
    if observed == recorded {
        Ok(())
    } else {
        Err(VerifyError::InputHashMismatch {
            file,
            recorded: recorded.to_string(),
            observed,
        })
    }
}

/// The data dir and the probability files hash to what the report records.
///
/// # Errors
///
/// [`VerifyError::InputHashMismatch`] naming the file.
pub fn check_inputs(inputs: &PackInputs, data: &DataDir) -> Result<(), VerifyError> {
    let r = &inputs.gate_report;
    hash_matches("task_json", &r.inputs_sha256.task_json, &data.task_bytes)?;
    hash_matches(
        "train_jsonl",
        &r.inputs_sha256.train_jsonl,
        &data.train_bytes,
    )?;
    hash_matches("eval_jsonl", &r.inputs_sha256.eval_jsonl, &data.eval_bytes)?;
    hash_matches(
        "eval_probs_json",
        &r.eval_probs_sha256,
        &inputs.eval_probs_json,
    )?;
    hash_matches(
        "zero_shot_probs_json",
        &r.zero_shot_probs_sha256,
        &inputs.zero_shot_probs_json,
    )?;
    match (&r.rescore_noise_sha256, &inputs.rescore_noise_json) {
        (Some(recorded), Some(bytes)) => hash_matches("rescore_noise_json", recorded, bytes),
        (None, None) => Ok(()),
        (Some(recorded), None) => Err(VerifyError::InputHashMismatch {
            file: "rescore_noise_json",
            recorded: recorded.clone(),
            observed: "absent".into(),
        }),
        (None, Some(bytes)) => Err(VerifyError::InputHashMismatch {
            file: "rescore_noise_json",
            recorded: "absent".into(),
            observed: sha256_hex(bytes),
        }),
    }
}

/// `nfc-trim-ws-v1`: NFC, then every Unicode White_Space run collapsed to one U+0020 with the
/// ends trimmed — the mirror of `scripts/laya_train/data.py` `normalize` (Rust's
/// `split_whitespace` IS the White_Space set that file spells out).
#[must_use]
pub fn normalize_text(text: &str) -> String {
    let composed: String = text.nfc().collect();
    composed.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// sha256 of [`normalize_text`].
#[must_use]
pub fn normalized_sha256(text: &str) -> String {
    sha256_hex(normalize_text(text).as_bytes())
}

/// Train rows grouped by normalized text, in first-seen order; a group with two labels is
/// refused.
fn group_train(train: &[DataRow]) -> Result<Vec<Vec<usize>>, VerifyError> {
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, row) in train.iter().enumerate() {
        let g = *index
            .entry(normalized_sha256(&row.text))
            .or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
        groups[g].push(i);
    }
    if let Some(g) = groups
        .iter()
        .find(|g| g.iter().any(|&i| train[i].label != train[g[0]].label))
    {
        return Err(VerifyError::ConflictingLabels { rows: g.clone() });
    }
    Ok(groups)
}

fn slice_invalid(why: String) -> VerifyError {
    VerifyError::SliceInvalid { why }
}

/// `slice_ids` sorted, unique, in range, equal to `slice_size` and to their recorded sha256.
fn check_slice_ids(calib: &GateCalibration, n_train: usize) -> Result<Vec<usize>, VerifyError> {
    let ids: Vec<usize> = calib.slice_ids.iter().map(|&i| i as usize).collect();
    if !ids.windows(2).all(|w| w[0] < w[1]) {
        return Err(slice_invalid("slice_ids are not sorted and unique".into()));
    }
    if let Some(&bad) = ids.iter().find(|&&i| i >= n_train) {
        return Err(slice_invalid(format!(
            "slice id {bad} is outside {n_train} train rows"
        )));
    }
    if ids.len() as u64 != calib.slice_size {
        return Err(slice_invalid(format!(
            "slice_size {} but {} slice_ids",
            calib.slice_size,
            ids.len()
        )));
    }
    let compact =
        serde_json::to_string(&calib.slice_ids).map_err(|e| slice_invalid(e.to_string()))?;
    let observed = sha256_hex(compact.as_bytes());
    if observed != calib.slice_ids_sha256 {
        return Err(slice_invalid(format!(
            "slice_ids hash to {observed}, the report records {}",
            calib.slice_ids_sha256
        )));
    }
    Ok(ids)
}

/// The slice rows a class of `in_class` train rows needs: `max(min_per_class,
/// ceil(fraction x in_class))`, exactly as `scripts/laya_train/data.py` `calibration_split`
/// computes it.
fn slice_need(in_class: usize, fraction: f64, min_per_class: u64) -> u64 {
    let by_fraction = (fraction * in_class as f64).ceil();
    // A class count times a fraction <= 1 is a small non-negative integer after ceil.
    min_per_class.max(by_fraction as u64)
}

/// Every class keeps at least `max(min_per_class, ceil(fraction x n_class))` slice rows (WR-08)
/// and at least one fit row.
fn check_slice_classes(
    ids: &[usize],
    train: &[DataRow],
    k: usize,
    fraction: f64,
    min_per_class: u64,
) -> Result<(), VerifyError> {
    for c in 0..k {
        let in_class = train.iter().filter(|r| r.label == c).count();
        let in_slice = ids.iter().filter(|&&i| train[i].label == c).count();
        let need = slice_need(in_class, fraction, min_per_class);
        if (in_slice as u64) < need || in_slice >= in_class {
            return Err(slice_invalid(format!(
                "class {c}: {in_slice} of {in_class} rows in the slice; need >= {need} \
                 (max(calibration_slice_min_per_class {min_per_class}, \
                 ceil(calibration_slice_fraction {fraction} x {in_class}))) and at least one fit row"
            )));
        }
    }
    Ok(())
}

/// laya-finetune-gate-v1 `split_disjointness`, re-derived from the data dir and the report's
/// `slice_ids` (row indices only): no eval text equals a train text after normalization, no
/// normalized train text carries two labels, and the calibration slice is well formed,
/// stratified to `max(min_per_class, ceil(fraction x n_class))` per class and group-disjoint
/// from the fit rows.
///
/// # Errors
///
/// [`VerifyError::SplitOverlap`], [`VerifyError::ConflictingLabels`] or
/// [`VerifyError::SliceInvalid`].
#[provable_contracts_macros::contract("laya-finetune-gate-v1", equation = "split_disjointness")]
pub fn check_split(
    train: &[DataRow],
    eval: &[DataRow],
    calib: &GateCalibration,
    k: usize,
    fraction: f64,
    min_per_class: u64,
) -> Result<(), VerifyError> {
    let train_hash: HashMap<String, usize> = train
        .iter()
        .enumerate()
        .rev()
        .map(|(i, r)| (normalized_sha256(&r.text), i))
        .collect();
    if let Some((eval_row, train_row)) = eval
        .iter()
        .enumerate()
        .find_map(|(e, r)| train_hash.get(&normalized_sha256(&r.text)).map(|&t| (e, t)))
    {
        return Err(VerifyError::SplitOverlap {
            eval_row,
            train_row,
        });
    }
    let groups = group_train(train)?;
    let ids = check_slice_ids(calib, train.len())?;
    check_slice_classes(&ids, train, k, fraction, min_per_class)?;
    let in_slice = |i: &usize| ids.binary_search(i).is_ok();
    if let Some(g) = groups
        .iter()
        .find(|g| g.iter().any(in_slice) && !g.iter().all(in_slice))
    {
        return Err(slice_invalid(format!(
            "train rows {g:?} share one normalized text but are split between calibration and fit"
        )));
    }
    Ok(())
}

fn record_mismatch(field: &'static str, why: String) -> VerifyError {
    VerifyError::RecordMismatch { field, why }
}

/// The gate-report records the contract states a relation for (plan 08-21, run_field_bindings):
/// `device_recorded` (`device_is_cpu == (device_used == "cpu")`) and `calibration_fit_bounded`
/// (`t_fitted` finite, `t_applied == clamp(t_fitted, calibration_temp_min,
/// calibration_temp_max)` bit for bit, `clamp_hit == (t_fitted <= min OR t_fitted >= max)`, as
/// `scripts/laya_train/gate.py` `fit_temperature` writes them). `t_applied` itself is bound to the
/// checkpoint's agent config by the load ladder (decide-apr-v1 rung 4); `t_fitted` cannot be
/// re-fitted here (the slice logits are not in the run dir), so this binds it through the clamp.
///
/// # Errors
///
/// [`VerifyError::RecordMismatch`] naming the field.
pub fn check_report_record(report: &GateReport, policy: &VerifyPolicy) -> Result<(), VerifyError> {
    let cpu = report.device_used == "cpu";
    if report.device_is_cpu != cpu {
        return Err(record_mismatch(
            "device_is_cpu",
            format!(
                "{} but device_used is {:?} (device_recorded: device_is_cpu == (device_used == \"cpu\"))",
                report.device_is_cpu, report.device_used
            ),
        ));
    }
    let c = &report.calibration;
    let (lo, hi) = (policy.calibration_temp_min, policy.calibration_temp_max);
    if !c.t_fitted.is_finite() {
        return Err(record_mismatch(
            "calibration.t_fitted",
            format!("{} is not finite", c.t_fitted),
        ));
    }
    let applied = hi.min(lo.max(c.t_fitted));
    if c.t_applied.to_bits() != applied.to_bits() {
        return Err(record_mismatch(
            "calibration.t_applied",
            format!(
                "{} but clamp(t_fitted {}, {lo}, {hi}) = {applied} (calibration_fit_bounded)",
                c.t_applied, c.t_fitted
            ),
        ));
    }
    let at_bound = c.t_fitted <= lo || c.t_fitted >= hi;
    if c.clamp_hit != at_bound {
        return Err(record_mismatch(
            "calibration.clamp_hit",
            format!(
                "{} but t_fitted {} is {}at a bound of [{lo}, {hi}] (calibration_fit_bounded)",
                c.clamp_hit,
                c.t_fitted,
                if at_bound { "" } else { "not " }
            ),
        ));
    }
    Ok(())
}

// ===========================================================================
// Probability files
// ===========================================================================

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbsFileJson {
    labels: Vec<String>,
    rows: Vec<ProbsRowJson>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbsRowJson {
    row: usize,
    text_sha256: String,
    probabilities: Vec<f64>,
}

fn check_probs_row(
    which: ProbsWhich,
    row: &ProbsRowJson,
    eval_text: &str,
    k: usize,
) -> Result<Vec<f32>, VerifyError> {
    let bad = |why: String| VerifyError::ProbsInvalid {
        which,
        row: Some(row.row),
        why,
    };
    if row.text_sha256 != sha256_hex(eval_text.as_bytes()) {
        return Err(bad("text_sha256 is not the eval row's".into()));
    }
    if row.probabilities.len() != k {
        return Err(bad(format!(
            "{} probabilities, the task has {k}",
            row.probabilities.len()
        )));
    }
    if let Some(p) = row
        .probabilities
        .iter()
        .find(|p| !(p.is_finite() && (0.0..=1.0).contains(*p)))
    {
        return Err(bad(format!(
            "probability {p} is not a finite value in [0, 1]"
        )));
    }
    let sum: f64 = row.probabilities.iter().sum();
    if !within((sum - 1.0).abs(), PROBS_ROW_SUM_ABS) {
        return Err(bad(format!(
            "probabilities sum to {sum}, not 1 within {PROBS_ROW_SUM_ABS}"
        )));
    }
    Ok(row.probabilities.iter().map(|&p| p as f32).collect())
}

/// Validate a probability file against the eval rows: exactly one row per eval row, unique
/// indices `0..N-1`, each row's `text_sha256` equal to its eval row's, and `K` finite
/// probabilities in `[0, 1]` summing to 1 within [`PROBS_ROW_SUM_ABS`]. Returns the rows in
/// eval order.
///
/// # Errors
///
/// [`VerifyError::ProbsRowCoverage`] or [`VerifyError::ProbsInvalid`].
pub fn validate_probs(
    which: ProbsWhich,
    bytes: &[u8],
    eval: &[DataRow],
    labels: &[String],
) -> Result<Vec<Vec<f32>>, VerifyError> {
    let file: ProbsFileJson =
        serde_json::from_slice(bytes).map_err(|e| VerifyError::ProbsInvalid {
            which,
            row: None,
            why: e.to_string(),
        })?;
    if file.labels != labels {
        return Err(VerifyError::ProbsInvalid {
            which,
            row: None,
            why: format!("labels {:?} are not the task's {labels:?}", file.labels),
        });
    }
    let coverage = |why: String| VerifyError::ProbsRowCoverage { which, why };
    if file.rows.len() != eval.len() {
        return Err(coverage(format!(
            "{} rows for {} eval rows",
            file.rows.len(),
            eval.len()
        )));
    }
    let mut out: Vec<Option<Vec<f32>>> = vec![None; eval.len()];
    for r in &file.rows {
        let slot = out
            .get_mut(r.row)
            .ok_or_else(|| coverage(format!("row index {} is outside 0..{}", r.row, eval.len())))?;
        if slot.is_some() {
            return Err(coverage(format!("row index {} appears twice", r.row)));
        }
        *slot = Some(check_probs_row(which, r, &eval[r.row].text, labels.len())?);
    }
    out.into_iter()
        .enumerate()
        .map(|(i, p)| p.ok_or_else(|| coverage(format!("row index {i} is missing"))))
        .collect()
}

// ===========================================================================
// Re-score, recompute, gate
// ===========================================================================

/// `max |a - b|`, NaN-propagating; NaN on a length mismatch.
pub(crate) fn row_max_abs(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    a.iter().zip(b).fold(0.0f64, |m, (&x, &y)| {
        let d = (f64::from(x) - f64::from(y)).abs();
        if d.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(d)
        }
    })
}

/// Re-score every eval row with `classify` (the model's own prepare + forward) and compare
/// with the file's probabilities: every component within `tol` (NaN-visible) and the argmax
/// exact (laya-parity-v1 `pack_rescore_probs_abs`). `tol` is the set's [`RescoreBound`].
///
/// # Errors
///
/// [`VerifyError::Score`] when the model refuses, [`VerifyError::RescoreDrift`] naming the
/// first row over `tol` and the overall maximum, or [`VerifyError::ArgmaxDrift`].
#[provable_contracts_macros::contract("laya-parity-v1", equation = "pack_rescore_probs_abs")]
pub fn rescore<F>(
    which: ProbsWhich,
    classify: F,
    texts: &[String],
    probs: &[Vec<f32>],
    tol: f64,
) -> Result<RescoreStats, VerifyError>
where
    F: FnOnce(&[String]) -> Result<Vec<Decision>, DecideError>,
{
    let decisions = classify(texts).map_err(|e| VerifyError::Score {
        which,
        reason: e.to_string(),
    })?;
    if decisions.len() != probs.len() {
        return Err(VerifyError::RescoreDrift {
            which,
            row: decisions.len().min(probs.len()),
            max_abs: f64::NAN,
            bound: tol,
        });
    }
    let mut max_abs = 0.0f64;
    let mut first_drift = None;
    let mut first_argmax = None;
    let mut argmax_agree = 0;
    for (row, (d, p)) in decisions.iter().zip(probs).enumerate() {
        let delta = row_max_abs(&d.probabilities, p);
        max_abs = if delta.is_nan() || max_abs.is_nan() {
            f64::NAN
        } else {
            max_abs.max(delta)
        };
        if first_drift.is_none() && !within(delta, tol) {
            first_drift = Some(row);
        }
        if argmax(&d.probabilities) == argmax(p) {
            argmax_agree += 1;
        } else if first_argmax.is_none() {
            first_argmax = Some(row);
        }
    }
    if let Some(row) = first_drift {
        return Err(VerifyError::RescoreDrift {
            which,
            row,
            max_abs,
            bound: tol,
        });
    }
    if let Some(row) = first_argmax {
        return Err(VerifyError::ArgmaxDrift { which, row });
    }
    Ok(RescoreStats {
        max_abs,
        argmax_agree,
        n: probs.len(),
    })
}

// ===========================================================================
// The noise-referenced re-score bound (laya-parity-v1 A1)
// ===========================================================================

/// The float64 reference name `rescore-noise.json` must carry.
const NOISE_REFERENCE: &str = "float64";
/// The control forwards the first `min(CONTROL_ROWS, n)` eval rows.
const CONTROL_ROWS: usize = 5;

fn noise_invalid(
    which: Option<ProbsWhich>,
    row: Option<usize>,
    field: &'static str,
    why: String,
) -> VerifyError {
    VerifyError::RescoreNoiseInvalid {
        which,
        row,
        field,
        why,
    }
}

/// The record-level fields: schema, reference, k and floor equal to the contract, the control
/// exactly 0.0 on the first `min(5, n)` rows, and exactly the two sets in order.
fn check_noise_header(
    rec: &pack::RescoreNoise,
    n: usize,
    policy: &VerifyPolicy,
) -> Result<(), VerifyError> {
    let bad = |field, why| noise_invalid(None, None, field, why);
    if rec.schema != pack::RESCORE_NOISE_SCHEMA {
        return Err(bad(
            "schema",
            format!(
                "{:?}, expected {:?}",
                rec.schema,
                pack::RESCORE_NOISE_SCHEMA
            ),
        ));
    }
    if rec.reference != NOISE_REFERENCE {
        return Err(bad(
            "reference",
            format!("{:?}, expected {NOISE_REFERENCE:?}", rec.reference),
        ));
    }
    if rec.k.to_bits() != policy.rescore_noise_k.to_bits() {
        return Err(bad(
            "k",
            format!(
                "{} but laya-parity-v1 pack_rescore_noise_k is {}",
                rec.k, policy.rescore_noise_k
            ),
        ));
    }
    if rec.floor_abs.to_bits() != policy.rescore_probs_abs.to_bits() {
        return Err(bad(
            "floor_abs",
            format!(
                "{} but laya-parity-v1 pack_rescore_probs_abs is {}",
                rec.floor_abs, policy.rescore_probs_abs
            ),
        ));
    }
    if rec.control_max_abs.to_bits() != 0.0f64.to_bits() {
        return Err(bad(
            "control_max_abs",
            format!(
                "{}: the manual fp32 forward must reproduce the Scorer exactly (0.0)",
                rec.control_max_abs
            ),
        ));
    }
    let want: Vec<usize> = (0..n.min(CONTROL_ROWS)).collect();
    if rec.control_rows != want {
        return Err(bad(
            "control_rows",
            format!("{:?}, expected {want:?}", rec.control_rows),
        ));
    }
    let names: Vec<&str> = rec.sets.iter().map(|s| s.which.as_str()).collect();
    if names != ["fine_tuned", "zero_shot"] {
        return Err(bad(
            "sets",
            format!("{names:?}, expected exactly [\"fine_tuned\", \"zero_shot\"]"),
        ));
    }
    Ok(())
}

/// `noise(c, s)` of one set, RECOMPUTED: every eval row once and in order, `K` finite f64
/// components in `[0, 1]` summing to 1, the float64 argmax equal to the stored float32 argmax,
/// and the maximum of `|f64(p_torch32) - p_f64|` over every row and component.
fn recompute_noise(
    which: ProbsWhich,
    set: &pack::NoiseSet,
    probs: &[Vec<f32>],
) -> Result<f64, VerifyError> {
    let n = probs.len();
    let bad = |row, field, why| noise_invalid(Some(which), row, field, why);
    if set.scored != "eval" {
        return Err(bad(
            None,
            "scored",
            format!("{:?}, expected \"eval\"", set.scored),
        ));
    }
    if set.n != n || set.rows.len() != n {
        return Err(bad(
            None,
            "rows",
            format!("n {} and {} rows for {n} eval rows", set.n, set.rows.len()),
        ));
    }
    let mut noise = 0.0f64;
    for (i, (r, p32)) in set.rows.iter().zip(probs).enumerate() {
        if r.row != i {
            return Err(bad(
                Some(i),
                "rows",
                format!(
                    "position {i} holds row {}: every eval row once, in order",
                    r.row
                ),
            ));
        }
        let p64 = &r.probabilities_f64;
        if p64.len() != p32.len() {
            return Err(bad(
                Some(i),
                "probabilities_f64",
                format!("{} components, the task has {}", p64.len(), p32.len()),
            ));
        }
        if let Some(p) = p64
            .iter()
            .find(|p| !(p.is_finite() && (0.0..=1.0).contains(*p)))
        {
            return Err(bad(
                Some(i),
                "probabilities_f64",
                format!("{p} is not a finite value in [0, 1]"),
            ));
        }
        let sum: f64 = p64.iter().sum();
        if !within((sum - 1.0).abs(), PROBS_ROW_SUM_ABS) {
            return Err(bad(
                Some(i),
                "probabilities_f64",
                format!("sum {sum}, not 1 within {PROBS_ROW_SUM_ABS}"),
            ));
        }
        let (torch, reference) = (argmax(p32), argmax(p64));
        if torch != reference {
            return Err(VerifyError::NoiseArgmaxFlip {
                which,
                row: i,
                torch,
                reference,
            });
        }
        for (&a, &b) in p32.iter().zip(p64) {
            let d = (f64::from(a) - b).abs();
            noise = if d.is_nan() || noise.is_nan() {
                f64::NAN
            } else {
                noise.max(d)
            };
        }
    }
    let agree = n as f64;
    if (set.argmax_agree as f64).to_bits() != agree.to_bits() {
        return Err(VerifyError::RescoreNoiseMismatch {
            which,
            field: "argmax_agree",
            reported: set.argmax_agree as f64,
            recomputed: agree,
        });
    }
    Ok(noise)
}

/// laya-parity-v1 A1: the bound each re-score is held to, DERIVED in Rust.
///
/// Without a record (`rescore_noise_sha256` absent) both sets get the floor
/// (`rescore_probs_abs`) — omitting the record can only tighten the check. With one, the
/// record is parsed ([`pack::RescoreNoise`]), its header checked against the contract
/// ([`VerifyPolicy::rescore_noise_k`], the floor, a 0.0 control), `noise(c, s)` RECOMPUTED from
/// its stored float64 rows against the already-validated probability rows (`ft_probs`,
/// `zs_probs`), the reported `max_abs` and `bound` required to equal the recomputation
/// bit-for-bit, and `bound = max(floor, k x noise)` refused above
/// [`VerifyPolicy::rescore_bound_max_abs`]. The reported numbers are never used.
///
/// # Errors
///
/// [`VerifyError::RescoreNoiseInvalid`] (malformed record, a field unequal to the contract, a
/// missing or repeated row), [`VerifyError::NoiseArgmaxFlip`],
/// [`VerifyError::RescoreNoiseMismatch`] (a reported value that differs from the
/// recomputation) or [`VerifyError::RescoreBoundCeiling`].
#[provable_contracts_macros::contract("laya-parity-v1", equation = "rescore_noise_reference")]
pub fn rescore_bounds(
    inputs: &PackInputs,
    ft_probs: &[Vec<f32>],
    zs_probs: &[Vec<f32>],
    policy: &VerifyPolicy,
) -> Result<[RescoreBound; 2], VerifyError> {
    let floor = policy.rescore_probs_abs;
    let mut out = [ProbsWhich::FineTuned, ProbsWhich::ZeroShot].map(|which| RescoreBound {
        which,
        noise: None,
        bound: floor,
    });
    let Some(bytes) = inputs.rescore_noise_json.as_deref() else {
        return Ok(out);
    };
    let rec: pack::RescoreNoise = serde_json::from_slice(bytes)
        .map_err(|e| noise_invalid(None, None, "json", e.to_string()))?;
    check_noise_header(&rec, ft_probs.len(), policy)?;
    for (slot, (set, probs)) in out
        .iter_mut()
        .zip(rec.sets.iter().zip([ft_probs, zs_probs]))
    {
        let which = slot.which;
        let noise = recompute_noise(which, set, probs)?;
        if set.max_abs.to_bits() != noise.to_bits() {
            return Err(VerifyError::RescoreNoiseMismatch {
                which,
                field: "max_abs",
                reported: set.max_abs,
                recomputed: noise,
            });
        }
        let bound = floor.max(policy.rescore_noise_k * noise);
        if set.bound.to_bits() != bound.to_bits() {
            return Err(VerifyError::RescoreNoiseMismatch {
                which,
                field: "bound",
                reported: set.bound,
                recomputed: bound,
            });
        }
        if !within(bound, policy.rescore_bound_max_abs) {
            return Err(VerifyError::RescoreBoundCeiling {
                which,
                bound,
                ceiling: policy.rescore_bound_max_abs,
            });
        }
        *slot = RescoreBound {
            which,
            noise: Some(noise),
            bound,
        };
    }
    Ok(out)
}

/// Macro-F1 and top-label ECE with aprender-core's ONE implementation of each (OPS-03).
///
/// Both are aprender-core's f64 paths with exactly-rounded sums — `macro_f1_f64` and
/// `expected_calibration_error_top_label_f64` — bit-identical to `scripts/laya_train/metrics.py`
/// `macro_f1` and `ece_top_label` (laya-finetune-gate-v1 `numeric_agreement`, replayed by
/// `verify::tests::gate_numeric_cases_agree_bit_for_bit`), so the margin, the ECE clause and the
/// rank key are decided on the same bits in both languages.
///
/// `probs` must already be validated ([`validate_probs`]): non-empty, `k >= 2` columns, rows
/// summing to 1 and labels `< k` — the metric functions' panicking preconditions.
#[must_use]
#[provable_contracts_macros::contract("laya-finetune-gate-v1", equation = "ece_top_label")]
pub fn recompute_metrics(
    probs: &[Vec<f32>],
    labels: &[usize],
    k: usize,
    ece_bins: usize,
) -> Metrics {
    let flat: Vec<f32> = probs.iter().flatten().copied().collect();
    let pred: Vec<usize> = probs.iter().map(|p| argmax(p)).collect();
    Metrics {
        macro_f1: macro_f1_f64(&pred, labels),
        ece: expected_calibration_error_top_label_f64(&flat, k, labels, ece_bins),
    }
}

/// The gate metrics of one run from its two probability sets — the ONE place the verifier
/// derives them: [`recompute_metrics`] on each set, [`recompute_nll`] on the fine-tuned set,
/// and `margin = ft.macro_f1 - zs.macro_f1` in f64 (the subtraction `scripts/laya_train/gate.py`
/// `evaluate_gate` does on the same two f64 values).
///
/// Both sets must already be validated ([`validate_probs`]) against `labels` and `k`.
#[must_use]
pub fn recompute_gate(
    zs_probs: &[Vec<f32>],
    ft_probs: &[Vec<f32>],
    labels: &[usize],
    k: usize,
    ece_bins: usize,
) -> Recomputed {
    let zs = recompute_metrics(zs_probs, labels, k, ece_bins);
    let ft = recompute_metrics(ft_probs, labels, k, ece_bins);
    Recomputed {
        zs_macro_f1: zs.macro_f1,
        zs_ece: zs.ece,
        ft_macro_f1: ft.macro_f1,
        ece_post: ft.ece,
        ft_nll: recompute_nll(ft_probs, labels),
        margin: ft.macro_f1 - zs.macro_f1,
    }
}

/// The eval NLL `mean_i -ln p_i[y_i]` with aprender-core's ONE log-loss implementation
/// (OPS-03): the binary `log_loss` of each row's true-class probability with label 1 is exactly
/// `-ln p[y]` (clipped to `[eps, 1 - eps]`; `scripts/laya_train/metrics.py` `nll` clips at 1e-12,
/// which differs only for a true-class probability below 1e-12).
///
/// `probs` must already be validated ([`validate_probs`]) and `labels < k`.
#[must_use]
pub fn recompute_nll(probs: &[Vec<f32>], labels: &[usize]) -> f64 {
    let p_true: Vec<f32> = probs.iter().zip(labels).map(|(p, &y)| p[y]).collect();
    f64::from(log_loss(&vec![1; p_true.len()], &p_true))
}

fn threshold_eq(field: &'static str, report: f64, policy: f64) -> Result<(), VerifyError> {
    if report.to_bits() == policy.to_bits() {
        Ok(())
    } else {
        Err(VerifyError::ThresholdMismatch {
            field,
            report,
            policy,
        })
    }
}

fn metric_close(
    field: &'static str,
    reported: f64,
    recomputed: f64,
    tol: f64,
) -> Result<(), VerifyError> {
    metric_close_for(None, field, reported, recomputed, tol)
}

/// [`metric_close`] naming the seed of a `seeds.per_seed` field.
fn metric_close_for(
    seed: Option<i64>,
    field: &'static str,
    reported: f64,
    recomputed: f64,
    tol: f64,
) -> Result<(), VerifyError> {
    if within((reported - recomputed).abs(), tol) {
        Ok(())
    } else {
        Err(VerifyError::ReportedMetricMismatch {
            seed,
            field,
            reported,
            recomputed,
        })
    }
}

/// The report's thresholds equal the contract's, bit for bit (D-07). Called by [`check_gate`]
/// and, since the per-seed pass verdicts are recomputed under the contract's thresholds, once
/// more before [`check_seed_selection`].
///
/// # Errors
///
/// [`VerifyError::ThresholdMismatch`] naming the threshold.
pub fn check_thresholds(report: &GateReport, policy: &VerifyPolicy) -> Result<(), VerifyError> {
    let t = &report.thresholds;
    threshold_eq(
        "min_macro_f1_margin",
        t.min_macro_f1_margin,
        policy.min_macro_f1_margin,
    )?;
    threshold_eq("max_ece", t.max_ece, policy.max_ece)?;
    threshold_eq("ece_bins", t.ece_bins as f64, policy.ece_bins as f64)
}

/// `a >= b`, NaN-visible (a NaN fails).
fn at_least(a: f64, b: f64) -> bool {
    matches!(a.partial_cmp(&b), Some(Ordering::Greater | Ordering::Equal))
}

/// laya-finetune-gate-v1 `gate_pass` on RECOMPUTED metrics, in order: the report's thresholds
/// equal the contract's; every reported metric (and row count) is within
/// `metric_recompute_abs` of its recomputation; the recomputed pass equals the reported one.
/// Returns the FAILED clauses — empty exactly when the gate passes; the caller turns a
/// non-empty list into [`VerifyError::GateFailed`] with its evidence.
///
/// # Errors
///
/// [`VerifyError::ThresholdMismatch`], [`VerifyError::ReportedMetricMismatch`] or
/// [`VerifyError::PassDisagrees`].
#[provable_contracts_macros::contract("laya-finetune-gate-v1", equation = "gate_pass")]
pub fn check_gate(
    report: &GateReport,
    recomputed: &Recomputed,
    n: usize,
    policy: &VerifyPolicy,
) -> Result<Vec<GateClause>, VerifyError> {
    check_thresholds(report, policy)?;
    let tol = policy.metric_recompute_abs;
    let r = recomputed;
    metric_close("zero_shot.n", report.zero_shot.n as f64, n as f64, 0.0)?;
    metric_close("fine_tuned.n", report.fine_tuned.n as f64, n as f64, 0.0)?;
    metric_close(
        "zero_shot.macro_f1",
        report.zero_shot.macro_f1,
        r.zs_macro_f1,
        tol,
    )?;
    metric_close("zero_shot.ece", report.zero_shot.ece, r.zs_ece, tol)?;
    metric_close(
        "fine_tuned.macro_f1",
        report.fine_tuned.macro_f1,
        r.ft_macro_f1,
        tol,
    )?;
    metric_close(
        "fine_tuned.ece_post",
        report.fine_tuned.ece_post,
        r.ece_post,
        tol,
    )?;
    metric_close("margin", report.margin, r.margin, tol)?;
    metric_close("fine_tuned.nll", report.fine_tuned.nll, r.ft_nll, tol)?;
    let mut failed = Vec::new();
    if !at_least(r.margin, policy.min_macro_f1_margin) {
        failed.push(GateClause::Margin);
    }
    if !within(r.ece_post, policy.max_ece) {
        failed.push(GateClause::EcePost);
    }
    let pass = failed.is_empty();
    if pass != report.pass {
        return Err(VerifyError::PassDisagrees {
            reported: report.pass,
            recomputed: pass,
        });
    }
    Ok(failed)
}

// ===========================================================================
// The seed selection (A3) and the shift probe (A2), laya-finetune-gate-v1 1.4.0
// ===========================================================================

fn seed_violated(field: &'static str, why: String) -> VerifyError {
    VerifyError::SeedPolicyViolated { field, why }
}

/// `seed_policy.rank_rule`: `floor(ece_post x rank_scale)` as an integer. A non-finite ECE (or
/// key) has no rank and is refused.
///
/// # Errors
///
/// [`VerifyError::SeedPolicyViolated`] naming the seed.
pub fn rank_key(seed: i64, ece_post: f64, rank_scale: f64) -> Result<i64, VerifyError> {
    let key = (ece_post * rank_scale).floor();
    if !key.is_finite() || key.abs() >= i64::MAX as f64 {
        return Err(seed_violated(
            "per_seed.ece_post",
            format!("seed {seed}: ece_post {ece_post} has no rank key"),
        ));
    }
    Ok(key as i64)
}

/// The SHIPPED seed under `seed_policy.selection` `median_ece`: `rows` are `(seed, ece_post)`,
/// ordered by (`rank_key` ascending, seed ascending — `tie_break` `smaller_seed`); the seed at
/// 0-based index `(N - 1) / 2`. The mirror of `scripts/laya_train/gate.py`
/// `select_median_seed`; this one is authoritative.
///
/// # Errors
///
/// [`VerifyError::SeedPolicyViolated`] for an empty or even N, a repeated seed or a
/// non-finite ECE (no median to ship).
pub fn select_median_seed(rows: &[(i64, f64)], rank_scale: f64) -> Result<i64, VerifyError> {
    let n = rows.len();
    if n == 0 || n % 2 == 0 {
        return Err(seed_violated(
            "per_seed",
            format!("the median rule needs an odd, non-empty number of seeds, got {n}"),
        ));
    }
    let mut seeds: Vec<i64> = rows.iter().map(|r| r.0).collect();
    seeds.sort_unstable();
    if seeds.windows(2).any(|w| w[0] == w[1]) {
        return Err(seed_violated(
            "per_seed",
            format!("a seed appears more than once in {seeds:?}"),
        ));
    }
    // (rank_key ascending, seed ascending): tie_break smaller_seed.
    let mut keyed = rows
        .iter()
        .map(|&(seed, ece)| rank_key(seed, ece, rank_scale).map(|k| (k, seed)))
        .collect::<Result<Vec<_>, _>>()?;
    keyed.sort_unstable();
    Ok(keyed[(n - 1) / 2].1)
}

/// recipe.json `seed_selection` equals the contract's `seed_policy`.
fn check_seed_decl(
    decl: &pack::SeedSelectionDecl,
    policy: &VerifyPolicy,
) -> Result<(), VerifyError> {
    let mismatch = |field, recipe: String, contract: String| {
        Err(VerifyError::SeedPolicyMismatch {
            field,
            recipe,
            contract,
        })
    };
    if decl.policy != policy.seed_selection_policy {
        return mismatch(
            "policy",
            format!("{:?}", decl.policy),
            format!("{:?}", policy.seed_selection_policy),
        );
    }
    if decl.seeds != policy.seed_selection_seeds {
        return mismatch(
            "seeds",
            format!("{:?}", decl.seeds),
            format!("{:?}", policy.seed_selection_seeds),
        );
    }
    if decl.rank_scale.to_bits() != policy.seed_rank_scale.to_bits() {
        return mismatch(
            "rank_scale",
            decl.rank_scale.to_string(),
            policy.seed_rank_scale.to_string(),
        );
    }
    if decl.tie_break != policy.seed_tie_break {
        return mismatch(
            "tie_break",
            format!("{:?}", decl.tie_break),
            format!("{:?}", policy.seed_tie_break),
        );
    }
    Ok(())
}

/// The seeds block's structure: declared seed, policy, n, the per_seed seeds in order, and
/// the shipped row binding the shipped checkpoint and the top-level eval file. Returns
/// `(per_seed, shipped)`.
fn check_seeds_block<'a>(
    inputs: &'a PackInputs,
    decl: &pack::SeedSelectionDecl,
) -> Result<(&'a [pack::PerSeedRow], i64), VerifyError> {
    let sb = &inputs.gate_report.seeds;
    let declared = decl.seeds.first().copied();
    if declared != Some(inputs.recipe.seed) || sb.declared != inputs.recipe.seed {
        return Err(seed_violated(
            "declared",
            format!(
                "recipe seed {}, report seeds.declared {}, first gate seed {declared:?} must be one seed",
                inputs.recipe.seed, sb.declared
            ),
        ));
    }
    if sb.policy.as_deref() != Some(decl.policy.as_str()) {
        return Err(seed_violated(
            "seeds.policy",
            format!("{:?}, the recipe declares {:?}", sb.policy, decl.policy),
        ));
    }
    let per = sb
        .per_seed
        .as_deref()
        .ok_or_else(|| seed_violated("per_seed", "absent under seed_selection".into()))?;
    let seeds: Vec<i64> = per.iter().map(|r| r.seed).collect();
    if seeds != decl.seeds || sb.n != seeds.len() as u64 {
        return Err(seed_violated(
            "per_seed",
            format!(
                "seeds {seeds:?} (n {}), the recipe declares {:?}",
                sb.n, decl.seeds
            ),
        ));
    }
    let label = format!("median-ECE seed of {} seeds", seeds.len());
    if sb.label != label {
        return Err(seed_violated(
            "seeds.label",
            format!(
                "{:?}, the contract's literal under median_ece is {label:?} (seed_policy.rule)",
                sb.label
            ),
        ));
    }
    let shipped = sb
        .shipped
        .ok_or_else(|| seed_violated("shipped", "absent under seed_selection".into()))?;
    let row = per.iter().find(|r| r.seed == shipped).ok_or_else(|| {
        seed_violated(
            "shipped",
            format!("seed {shipped} is not one of the gate seeds {seeds:?}"),
        )
    })?;
    if row.eval_probs_sha256 != inputs.gate_report.eval_probs_sha256 {
        return Err(seed_violated(
            "eval_probs_sha256",
            format!(
                "the shipped seed {shipped}'s per_seed eval_probs_sha256 {} is not the report's eval-probs.json {}",
                row.eval_probs_sha256, inputs.gate_report.eval_probs_sha256
            ),
        ));
    }
    let t_applied = inputs.gate_report.calibration.t_applied;
    if row.t_applied.to_bits() != t_applied.to_bits() {
        return Err(seed_violated(
            "per_seed.t_applied",
            format!(
                "the shipped seed {shipped}'s per_seed t_applied {} is not calibration.t_applied {t_applied}",
                row.t_applied
            ),
        ));
    }
    if row.model_safetensors_sha256 != inputs.checkpoint_sha256 {
        return Err(seed_violated(
            "model_safetensors_sha256",
            format!(
                "the shipped seed {shipped}'s per_seed model sha256 {} is not checkpoint/model.safetensors {}",
                row.model_safetensors_sha256, inputs.checkpoint_sha256
            ),
        ));
    }
    Ok((per, shipped))
}

/// laya-finetune-gate-v1 1.4.0 `seed_selection_median`, RE-DERIVED from the files.
///
/// `Ok(None)` for a legacy recipe (no `seed_selection`; its report must carry no seed
/// selection either) — the pipeline refuses such a run [`VerifyError::SeedPolicyMissing`]
/// only after its gate. For a recipe with `seed_selection`: the declaration equals the
/// contract's; the seeds block names the declared seeds in order and ships one of them; the
/// shipped row binds `checkpoint/model.safetensors` and `eval-probs.json` by sha256; every
/// seed's `seeds/seed-<s>/eval-probs.json` (hash-bound by [`PackInputs::from_run_dir`]) is
/// validated and its macro-F1, ECE and margin (against the shared zero-shot baseline)
/// RECOMPUTED with the house functions, each reported value within `metric_recompute_abs`, its
/// pass and rank key equal to the recomputation (a rank key Python and Rust disagree on — an
/// ECE on a grid line — is refused rather than guessed, `why_quantized`); and `shipped` must be
/// the median of the RECOMPUTED ECEs ([`select_median_seed`]). Returns the shipped seed.
///
/// # Errors
///
/// [`VerifyError::SeedPolicyMismatch`], [`VerifyError::SeedPolicyViolated`],
/// [`VerifyError::ReportedMetricMismatch`] (naming the seed), or a probability-file refusal.
#[provable_contracts_macros::contract("laya-finetune-gate-v1", equation = "seed_selection_median")]
pub fn check_seed_selection(
    inputs: &PackInputs,
    data: &DataDir,
    policy: &VerifyPolicy,
) -> Result<Option<i64>, VerifyError> {
    let sb = &inputs.gate_report.seeds;
    let Some(decl) = &inputs.recipe.seed_selection else {
        if sb.policy.is_some() || sb.shipped.is_some() || sb.per_seed.is_some() {
            return Err(seed_violated(
                "seeds",
                "the report carries a seed selection the recipe does not declare".into(),
            ));
        }
        return Ok(None);
    };
    check_seed_decl(decl, policy)?;
    let (per, shipped) = check_seeds_block(inputs, decl)?;
    let labels = data.task.owned_labels();
    let y: Vec<usize> = data.eval.iter().map(|r| r.label).collect();
    let (k, bins, tol) = (
        labels.len(),
        policy.ece_bins as usize,
        policy.metric_recompute_abs,
    );
    let zs = validate_probs(
        ProbsWhich::ZeroShot,
        &inputs.zero_shot_probs_json,
        &data.eval,
        &labels,
    )?;
    let zs_f1 = recompute_metrics(&zs, &y, k, bins).macro_f1;
    let mut eces = Vec::with_capacity(per.len());
    for (row, (seed, bytes)) in per.iter().zip(&inputs.seed_eval_probs) {
        if row.seed != *seed {
            return Err(seed_violated(
                "per_seed.seed",
                format!("row seed {} read with seed {seed}'s file", row.seed),
            ));
        }
        let probs = validate_probs(ProbsWhich::Seed(*seed), bytes, &data.eval, &labels)?;
        let m = recompute_metrics(&probs, &y, k, bins);
        let margin = m.macro_f1 - zs_f1;
        let s = Some(*seed);
        metric_close_for(s, "per_seed.macro_f1", row.macro_f1, m.macro_f1, tol)?;
        metric_close_for(s, "per_seed.ece_post", row.ece_post, m.ece, tol)?;
        metric_close_for(s, "per_seed.margin", row.margin, margin, tol)?;
        let pass = at_least(margin, policy.min_macro_f1_margin) && within(m.ece, policy.max_ece);
        if pass != row.pass {
            return Err(seed_violated(
                "per_seed.pass",
                format!(
                    "seed {seed}: reported pass={} recomputed pass={pass}",
                    row.pass
                ),
            ));
        }
        let key = rank_key(*seed, m.ece, decl.rank_scale)?;
        if key != row.rank_key {
            return Err(seed_violated(
                "per_seed.rank_key",
                format!(
                    "seed {seed}: reported rank_key {} but floor({} x {}) = {key}; refused rather than guessed (seed_policy.why_quantized)",
                    row.rank_key, m.ece, decl.rank_scale
                ),
            ));
        }
        eces.push((*seed, m.ece));
    }
    if eces.len() != per.len() {
        return Err(seed_violated(
            "per_seed",
            format!("{} per_seed rows but {} seed files", per.len(), eces.len()),
        ));
    }
    let median = select_median_seed(&eces, decl.rank_scale)?;
    if shipped != median {
        return Err(seed_violated(
            "shipped",
            format!(
                "the report ships seed {shipped}, but the median-ECE seed of the recomputed per-seed ECEs is seed {median}"
            ),
        ));
    }
    Ok(Some(shipped))
}

fn shift_mismatch(field: &'static str, why: String) -> VerifyError {
    VerifyError::ShiftProbeMismatch { field, why }
}

fn shift_close(
    field: &'static str,
    reported: f64,
    recomputed: f64,
    tol: f64,
) -> Result<(), VerifyError> {
    if within((reported - recomputed).abs(), tol) {
        Ok(())
    } else {
        Err(shift_mismatch(
            field,
            format!("reported {reported}, recomputed {recomputed}"),
        ))
    }
}

/// laya-finetune-gate-v1 1.4.0 `shift_probe_reported`: when the report names a shift probe,
/// `shift.jsonl` must exist and hash to `inputs_sha256.shift_jsonl`, share no normalized text
/// with train.jsonl, and both shift probability files (hash-bound by
/// [`PackInputs::from_run_dir`]) must validate against it; every reported shift metric is then
/// RECOMPUTED and must agree within `metric_recompute_abs`. The probe is never re-scored and
/// its numbers never reach [`check_gate`].
///
/// # Errors
///
/// [`VerifyError::ShiftProbeMismatch`] naming the field or file, or a probability-file refusal.
#[provable_contracts_macros::contract("laya-finetune-gate-v1", equation = "shift_probe_reported")]
pub fn check_shift_probe(
    inputs: &PackInputs,
    data: &DataDir,
    policy: &VerifyPolicy,
) -> Result<(), VerifyError> {
    let r = &inputs.gate_report;
    let (probe, recorded) = match (&r.shift_probe, &r.inputs_sha256.shift_jsonl) {
        // Present EXACTLY when the data dir carries shift.jsonl: a report may not drop the
        // probe (and with it the shift evidence A2 keeps visible) while the data has one.
        (None, None) if data.shift.is_some() => {
            return Err(shift_mismatch(
                "shift_probe",
                "the data dir carries shift.jsonl but the report names no shift probe".into(),
            ))
        }
        (None, None) => return Ok(()),
        (Some(p), Some(h)) => (p, h),
        _ => {
            return Err(shift_mismatch(
                "inputs_sha256.shift_jsonl",
                "shift_probe and inputs_sha256.shift_jsonl must be present together".into(),
            ))
        }
    };
    if probe.gate_clause {
        return Err(shift_mismatch(
            "shift_probe.gate_clause",
            "true: the shift probe is never a gate clause".into(),
        ));
    }
    let (Some(rows), Some(bytes)) = (&data.shift, &data.shift_bytes) else {
        return Err(shift_mismatch(
            "shift_jsonl",
            "the report names a shift probe but the data dir has no shift.jsonl".into(),
        ));
    };
    let observed = sha256_hex(bytes);
    if &observed != recorded {
        return Err(shift_mismatch(
            "shift_jsonl",
            format!("shift.jsonl hashes to {observed}, the report records {recorded}"),
        ));
    }
    let train: HashMap<String, usize> = data
        .train
        .iter()
        .enumerate()
        .rev()
        .map(|(i, t)| (normalized_sha256(&t.text), i))
        .collect();
    if let Some((i, j)) = rows
        .iter()
        .enumerate()
        .find_map(|(i, row)| train.get(&normalized_sha256(&row.text)).map(|&j| (i, j)))
    {
        return Err(shift_mismatch(
            "shift_jsonl",
            format!("shift row {i} equals train row {j} after normalization"),
        ));
    }
    shift_close("shift_probe.n", probe.n as f64, rows.len() as f64, 0.0)?;
    let labels = data.task.owned_labels();
    let missing = || {
        shift_mismatch(
            "shift_probe",
            "a shift probability file was not read".into(),
        )
    };
    let ft = validate_probs(
        ProbsWhich::ShiftFineTuned,
        inputs.shift_probs_json.as_deref().ok_or_else(missing)?,
        rows,
        &labels,
    )?;
    let zs = validate_probs(
        ProbsWhich::ShiftZeroShot,
        inputs
            .shift_zero_shot_probs_json
            .as_deref()
            .ok_or_else(missing)?,
        rows,
        &labels,
    )?;
    let y: Vec<usize> = rows.iter().map(|r| r.label).collect();
    let (k, bins, tol) = (
        labels.len(),
        policy.ece_bins as usize,
        policy.metric_recompute_abs,
    );
    let f = recompute_metrics(&ft, &y, k, bins);
    let z = recompute_metrics(&zs, &y, k, bins);
    shift_close(
        "shift_probe.zero_shot.macro_f1",
        probe.zero_shot.macro_f1,
        z.macro_f1,
        tol,
    )?;
    shift_close("shift_probe.zero_shot.ece", probe.zero_shot.ece, z.ece, tol)?;
    shift_close(
        "shift_probe.fine_tuned.macro_f1",
        probe.fine_tuned.macro_f1,
        f.macro_f1,
        tol,
    )?;
    shift_close(
        "shift_probe.fine_tuned.ece_post",
        probe.fine_tuned.ece_post,
        f.ece,
        tol,
    )?;
    shift_close(
        "shift_probe.margin",
        probe.margin,
        f.macro_f1 - z.macro_f1,
        tol,
    )
}

/// The stance pair `scripts/laya_train/train.py` averages F1 over for `f_avg`
/// (`[labels.index("against"), labels.index("favor")]`).
pub const F_AVG_STANCE_LABELS: [&str; 2] = ["against", "favor"];

/// The trainer's `f_avg` rule: the indices of [`F_AVG_STANCE_LABELS`] exactly when the task's
/// labels ARE `demo.criteria_order` (the stance task), else `None` (every `f_avg` is null).
///
/// # Errors
///
/// [`VerifyError::RecordMismatch`] when the contract's criteria order lacks a stance label (the
/// trainer's `labels.index` would raise on it).
pub fn f_avg_labels(
    task_labels: &[String],
    criteria_order: &[String],
) -> Result<Option<Vec<usize>>, VerifyError> {
    if task_labels != criteria_order {
        return Ok(None);
    }
    F_AVG_STANCE_LABELS
        .iter()
        .map(|name| {
            task_labels.iter().position(|l| l == name).ok_or_else(|| {
                record_mismatch(
                    "f_avg",
                    format!("demo.criteria_order {criteria_order:?} has no {name:?} label"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// The probability sets whose `f_avg` a report records, validated: the eval sets, every gate
/// seed's eval set, and the shift probe's two sets (1.4.0).
#[derive(Debug, Clone, Default)]
pub struct FAvgSets {
    /// The eval rows' true labels.
    pub eval_labels: Vec<usize>,
    /// `zero-shot-probs.json`.
    pub zero_shot: Vec<Vec<f32>>,
    /// `eval-probs.json`.
    pub fine_tuned: Vec<Vec<f32>>,
    /// `(seed, seeds/seed-<s>/eval-probs.json)`, in the report's per_seed order.
    pub per_seed: Vec<(i64, Vec<Vec<f32>>)>,
    /// `(shift labels, shift-probs.json, shift-zero-shot-probs.json)`.
    pub shift: Option<(Vec<usize>, Vec<Vec<f32>>, Vec<Vec<f32>>)>,
}

/// One reported `f_avg` against the rule and its recomputation.
fn f_avg_close(
    seed: Option<i64>,
    field: &'static str,
    reported: Option<f64>,
    probs: &[Vec<f32>],
    y: &[usize],
    rule: Option<&[usize]>,
    tol: f64,
) -> Result<(), VerifyError> {
    let who = seed.map_or_else(String::new, |s| format!("seed {s}: "));
    match (rule, reported) {
        (None, None) => Ok(()),
        (None, Some(v)) => Err(record_mismatch(
            field,
            format!(
                "{who}reported {v}, but the task's labels are not demo.criteria_order, so the trainer writes null"
            ),
        )),
        (Some(_), None) => Err(record_mismatch(
            field,
            format!(
                "{who}null, but the task's labels are demo.criteria_order, so the trainer writes the stance f_avg"
            ),
        )),
        (Some(labels), Some(v)) => {
            let pred: Vec<usize> = probs.iter().map(|p| argmax(p)).collect();
            let recomputed = mean_f1_over_labels_f64(&pred, y, labels);
            metric_close_for(seed, field, v, recomputed, tol)
        }
    }
}

/// laya-finetune-gate-v1 `numeric_agreement.f_avg` (plan 08-27, V6-e): every `f_avg` the
/// report records — zero-shot, fine-tuned, each `seeds.per_seed` row and both shift-probe sets
/// — follows the trainer's null rule ([`f_avg_labels`]) and, when set, is within
/// `metric_recompute_abs` of its RECOMPUTATION (aprender-core `mean_f1_over_labels_f64`, the
/// f64 exactly-rounded mean `metrics.py` `f_avg` equals bit for bit). `f_avg` enters no gate
/// clause; it is bound so a report cannot carry a number verify never checked.
///
/// # Errors
///
/// [`VerifyError::RecordMismatch`] (the null rule, or a per_seed / shift set the report and the
/// run dir disagree on) or [`VerifyError::ReportedMetricMismatch`] (a forged value), naming the
/// field.
pub fn check_f_avg(
    report: &GateReport,
    task_labels: &[String],
    sets: &FAvgSets,
    policy: &VerifyPolicy,
) -> Result<(), VerifyError> {
    let rule = f_avg_labels(task_labels, &policy.stance_criteria_order)?;
    let rule = rule.as_deref();
    let tol = policy.metric_recompute_abs;
    let y = &sets.eval_labels;
    f_avg_close(
        None,
        "zero_shot.f_avg",
        report.zero_shot.f_avg,
        &sets.zero_shot,
        y,
        rule,
        tol,
    )?;
    f_avg_close(
        None,
        "fine_tuned.f_avg",
        report.fine_tuned.f_avg,
        &sets.fine_tuned,
        y,
        rule,
        tol,
    )?;
    let rows = report.seeds.per_seed.as_deref().unwrap_or_default();
    if rows.len() != sets.per_seed.len() {
        return Err(record_mismatch(
            "per_seed.f_avg",
            format!(
                "{} per_seed rows but {} seed probability sets",
                rows.len(),
                sets.per_seed.len()
            ),
        ));
    }
    for (row, (seed, probs)) in rows.iter().zip(&sets.per_seed) {
        if row.seed != *seed {
            return Err(record_mismatch(
                "per_seed.f_avg",
                format!("row seed {} read with seed {seed}'s file", row.seed),
            ));
        }
        f_avg_close(
            Some(*seed),
            "per_seed.f_avg",
            row.f_avg,
            probs,
            y,
            rule,
            tol,
        )?;
    }
    match (&report.shift_probe, &sets.shift) {
        (None, None) => Ok(()),
        (Some(probe), Some((shift_y, ft, zs))) => {
            f_avg_close(
                None,
                "shift_probe.zero_shot.f_avg",
                probe.zero_shot.f_avg,
                zs,
                shift_y,
                rule,
                tol,
            )?;
            f_avg_close(
                None,
                "shift_probe.fine_tuned.f_avg",
                probe.fine_tuned.f_avg,
                ft,
                shift_y,
                rule,
                tol,
            )
        }
        _ => Err(record_mismatch(
            "shift_probe.f_avg",
            "the report and the run dir disagree on whether a shift probe exists".into(),
        )),
    }
}

/// The [`FAvgSets`] of a run: the two validated eval sets plus every seed file and the shift
/// probe's files, validated again here (each was already validated, hash-bound and recomputed
/// by [`check_seed_selection`] / [`check_shift_probe`]).
fn f_avg_sets(
    inputs: &PackInputs,
    data: &DataDir,
    labels: &[String],
    ft_probs: &[Vec<f32>],
    zs_probs: &[Vec<f32>],
) -> Result<FAvgSets, VerifyError> {
    let per_seed = inputs
        .seed_eval_probs
        .iter()
        .map(|(seed, bytes)| {
            validate_probs(ProbsWhich::Seed(*seed), bytes, &data.eval, labels).map(|p| (*seed, p))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shift = match (
        &data.shift,
        &inputs.shift_probs_json,
        &inputs.shift_zero_shot_probs_json,
    ) {
        (Some(rows), Some(ft), Some(zs)) => Some((
            rows.iter().map(|r| r.label).collect(),
            validate_probs(ProbsWhich::ShiftFineTuned, ft, rows, labels)?,
            validate_probs(ProbsWhich::ShiftZeroShot, zs, rows, labels)?,
        )),
        _ => None,
    };
    Ok(FAvgSets {
        eval_labels: data.eval.iter().map(|r| r.label).collect(),
        zero_shot: zs_probs.to_vec(),
        fine_tuned: ft_probs.to_vec(),
        per_seed,
        shift,
    })
}

// ===========================================================================
// The pipeline
// ===========================================================================

/// Everything the cheap checks established, carried into the re-scores.
struct Checked {
    texts: Vec<String>,
    labels: Vec<usize>,
    task: Task,
    ft_probs: Vec<Vec<f32>>,
    zs_probs: Vec<Vec<f32>>,
    bounds: [RescoreBound; 2],
    shipped_seed: Option<i64>,
}

/// Steps 1-4 of the module docs: variant, recipe block, base, hashes, shots, split, probability
/// files, the per-set
/// re-score bounds (A1), the thresholds, the re-derived seed selection (A3) and the shift probe
/// (A2) — files only, so all of it is decided before any model is built.
fn cheap_checks(
    inputs: &PackInputs,
    data_dir: &Path,
    base_dir: &Path,
    policy: &VerifyPolicy,
) -> Result<Checked, VerifyError> {
    check_variant(&inputs.recipe)?;
    check_recipe_block(&inputs.recipe, policy)?;
    check_base(base_dir, &inputs.recipe, policy)?;
    let data = read_data_dir(data_dir)?;
    check_inputs(inputs, &data)?;
    let labels = data.task.owned_labels();
    check_eval_coverage(&data.eval, &labels)?;
    let k = labels.len();
    check_recipe_shots(&inputs.recipe, &data.train, k)?;
    check_split(
        &data.train,
        &data.eval,
        &inputs.gate_report.calibration,
        k,
        policy.calibration_slice_fraction,
        policy.calibration_slice_min_per_class,
    )?;
    check_report_record(&inputs.gate_report, policy)?;
    let ft_probs = validate_probs(
        ProbsWhich::FineTuned,
        &inputs.eval_probs_json,
        &data.eval,
        &labels,
    )?;
    let zs_probs = validate_probs(
        ProbsWhich::ZeroShot,
        &inputs.zero_shot_probs_json,
        &data.eval,
        &labels,
    )?;
    let bounds = rescore_bounds(inputs, &ft_probs, &zs_probs, policy)?;
    check_thresholds(&inputs.gate_report, policy)?;
    let shipped_seed = check_seed_selection(inputs, &data, policy)?;
    check_shift_probe(inputs, &data, policy)?;
    let sets = f_avg_sets(inputs, &data, &labels, &ft_probs, &zs_probs)?;
    check_f_avg(&inputs.gate_report, &labels, &sets, policy)?;
    Ok(Checked {
        texts: data.eval.iter().map(|r| r.text.clone()).collect(),
        labels: data.eval.iter().map(|r| r.label).collect(),
        task: data.task,
        ft_probs,
        zs_probs,
        bounds,
        shipped_seed,
    })
}

/// Steps 5-6: both re-scores (the fine-tuned one on `decider`, dropped before the base is
/// built), the recomputed metrics and the gate.
fn verify_loaded(
    inputs: &PackInputs,
    checked: Checked,
    decider: Decider,
    base_dir: &Path,
    policy: &VerifyPolicy,
) -> Result<VerifyReport, VerifyError> {
    let artifact_sha256 = decider.identity().artifact_sha256.clone();
    let [ft_bound, zs_bound] = checked.bounds;
    let ft = rescore(
        ProbsWhich::FineTuned,
        |t| decider.classify(t),
        &checked.texts,
        &checked.ft_probs,
        ft_bound.bound,
    )?;
    drop(decider);
    let base = load_declared_base(
        base_dir,
        checked.task.clone(),
        &inputs.recipe.base.sha256,
        &policy.base,
        &inputs.gate_report.inputs_sha256.tokenizer_json,
    )?;
    let zs = rescore(
        ProbsWhich::ZeroShot,
        |t| base.classify(t),
        &checked.texts,
        &checked.zs_probs,
        zs_bound.bound,
    )?;
    drop(base);
    let k = checked.task.criteria().len();
    let bins = policy.ece_bins as usize;
    let recomputed = recompute_gate(
        &checked.zs_probs,
        &checked.ft_probs,
        &checked.labels,
        k,
        bins,
    );
    let n = checked.labels.len();
    let clauses = check_gate(&inputs.gate_report, &recomputed, n, policy)?;
    if !clauses.is_empty() {
        return Err(VerifyError::GateFailed(Box::new(GateFailure {
            clauses,
            recomputed,
            rescore_max_abs: ft.max_abs,
            zs_rescore_max_abs: zs.max_abs,
            rescore_bound: ft_bound.bound,
            zs_rescore_bound: zs_bound.bound,
            noise: ft_bound.noise,
            zs_noise: zs_bound.noise,
            shipped_seed: checked.shipped_seed,
            argmax_agree: ft.argmax_agree,
            n,
            artifact_sha256,
        })));
    }
    // seed_policy.legacy_rule: AFTER the re-scores and the gate, so a legacy run whose gate
    // fails still reports GateFailed (the 1.x fail-closed vectors keep their refusals).
    if inputs.recipe.seed_selection.is_none() {
        return Err(VerifyError::SeedPolicyMissing);
    }
    Ok(VerifyReport {
        artifact_sha256,
        rescore_max_abs: ft.max_abs,
        zs_rescore_max_abs: zs.max_abs,
        rescore_bound: ft_bound.bound,
        zs_rescore_bound: zs_bound.bound,
        noise: ft_bound.noise,
        zs_noise: zs_bound.noise,
        shipped_seed: checked.shipped_seed,
        argmax_agree: ft.argmax_agree,
        n,
        recomputed,
        deploy_eligible: true,
    })
}

/// Verify `packed` (bytes packed from `inputs`) against its data dir and declared base, in
/// the order of the module docs: cheap checks first, the two full re-scores last, the
/// fine-tuned one on [`Decider::load_bytes`] of `packed` — the whole ladder, probes included.
///
/// NOT an eligibility door (WR-02): the bytes here are the caller's claim of what the run packs
/// to, so it is test-only (`cfg(test)`, the unit tests' way to verify induced run copies). The
/// only eligibility doors are [`pack_for_serving`] (packs the bytes itself) and [`verify_path`]
/// (binds an existing file to a fresh pack of its run).
///
/// # Errors
///
/// The first refusal, as a [`VerifyError`]; [`VerifyError::GateFailed`] (exit code 3) when
/// every input verified and the recomputed gate failed.
#[cfg(test)]
pub(crate) fn verify_run(
    inputs: &PackInputs,
    packed: &[u8],
    data_dir: &Path,
    base_dir: &Path,
    policy: &VerifyPolicy,
) -> Result<VerifyReport, VerifyError> {
    let checked = cheap_checks(inputs, data_dir, base_dir, policy)?;
    let decider = Decider::load_bytes(packed)?;
    verify_loaded(inputs, checked, decider, base_dir, policy)
}

/// A process-wide counter that makes every [`write_atomic`] temp name unique.
static WRITE_SEQ: AtomicU64 = AtomicU64::new(0);
/// Fresh temp names [`write_atomic`] tries before giving up.
const WRITE_TMP_ATTEMPTS: usize = 8;

/// Write `bytes` to `out` atomically: a temp file in `out`'s directory, then a rename.
///
/// The temp file is opened `create_new` (O_CREAT | O_EXCL) under a per-call unique name (pid
/// plus a process-wide counter), so an existing path or a planted symlink at that name is never
/// followed or truncated (V8-c): an occupied name is skipped for the next, and after
/// [`WRITE_TMP_ATTEMPTS`] occupied names the write is refused. On any failure only a file THIS
/// call created is removed.
fn write_atomic(out: &Path, bytes: &[u8]) -> Result<(), VerifyError> {
    let dir = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let io = |path: &Path, e: std::io::Error| VerifyError::Read {
        path: path.to_path_buf(),
        reason: e.to_string(),
    };
    let name = out
        .file_name()
        .map_or_else(|| "out".into(), |n| n.to_string_lossy().into_owned());
    let mut last = None;
    for _ in 0..WRITE_TMP_ATTEMPTS {
        let seq = WRITE_SEQ.fetch_add(1, AtomicOrdering::Relaxed);
        let tmp = dir.join(format!(".{name}.tmp-{}-{seq}", std::process::id()));
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                last = Some((tmp, e));
                continue;
            }
            Err(e) => return Err(io(&tmp, e)),
        };
        // From here the temp file is ours: remove it (and only it) on any failure.
        let result = file
            .write_all(bytes)
            .and_then(|()| file.sync_all())
            .and_then(|()| {
                drop(file);
                std::fs::rename(&tmp, out)
            });
        if let Err(e) = result {
            let _ = std::fs::remove_file(&tmp);
            return Err(io(out, e));
        }
        return Ok(());
    }
    Err(match last {
        Some((tmp, e)) => io(&tmp, e),
        None => io(out, std::io::Error::other("no temp name was tried")),
    })
}

/// Pack a run dir FOR SERVING: read it, run the cheap checks, pack the bytes in memory,
/// verify them (the pipeline's steps on those exact bytes), and ONLY on acceptance write
/// them to `out` atomically. On any refusal nothing is created in `out`'s directory.
///
/// # Errors
///
/// The first refusal ([`VerifyError::exit_code`] 3 for a failed gate, 2 otherwise).
pub fn pack_for_serving(
    run_dir: &Path,
    data_dir: &Path,
    base_dir: &Path,
    out: &Path,
    policy: &VerifyPolicy,
) -> Result<VerifyReport, VerifyError> {
    let inputs = PackInputs::from_run_dir(run_dir, data_dir)?;
    let checked = cheap_checks(&inputs, data_dir, base_dir, policy)?;
    let packed = artifact::write_decide_apr(&inputs)?;
    let decider = Decider::load_bytes(&packed)?;
    let report = verify_loaded(&inputs, checked, decider, base_dir, policy)?;
    write_atomic(out, &packed)?;
    Ok(report)
}

/// Deployment eligibility of the EXACT file `apr` (decide-apr-v1 `deploy_eligibility`): read
/// it once (bounded, rung 1) and load those bytes through the whole ladder, require its
/// manifest to describe the given run and data dirs (recipe_id, report sha256, input hashes,
/// labels), require the file to BE the bytes that run packs to, then every step of
/// the pipeline with the fine-tuned re-score on that loaded file.
///
/// The byte binding is what makes the re-score sufficient: the re-score only exercises the
/// weights the eval rows and probes reach, so without it a file whose unexercised weights
/// (e.g. embedding rows of tokens no eval row contains) differ from the run's checkpoint would
/// verify. Packing is byte-deterministic (FALSIFY-DECIDE-APR-002), so an honest file is
/// identical to a fresh pack of its run.
///
/// # Errors
///
/// [`VerifyError::Artifact`] for a ladder refusal, [`VerifyError::ManifestMismatch`],
/// [`VerifyError::ArtifactNotFromRun`], or any refusal of the pipeline.
pub fn verify_path(
    apr: &Path,
    run_dir: &Path,
    data_dir: &Path,
    base_dir: &Path,
    policy: &VerifyPolicy,
) -> Result<VerifyReport, VerifyError> {
    let io = |e: std::io::Error| ArtifactError::Read {
        reason: e.to_string(),
    };
    let file = std::fs::File::open(apr).map_err(io)?;
    let declared = file.metadata().map_err(io)?.len();
    let bytes = artifact::read_decide_apr_bytes_bounded(file, Some(declared))?;
    let decider = Decider::load_bytes(&bytes)?;
    let inputs = PackInputs::from_run_dir(run_dir, data_dir)?;
    let m = decider.manifest();
    let report_sha = sha256_hex(&inputs.gate_report_json);
    if m.recipe_id != inputs.gate_report.recipe_id {
        return Err(VerifyError::ManifestMismatch { field: "recipe_id" });
    }
    if m.gate.report_sha256 != report_sha {
        return Err(VerifyError::ManifestMismatch {
            field: "gate.report_sha256",
        });
    }
    if m.inputs_sha256 != inputs.inputs_sha256 {
        return Err(VerifyError::ManifestMismatch {
            field: "inputs_sha256",
        });
    }
    let checked = cheap_checks(&inputs, data_dir, base_dir, policy)?;
    if m.labels != checked.task.owned_labels() {
        return Err(VerifyError::ManifestMismatch { field: "labels" });
    }
    let packed = artifact::write_decide_apr(&inputs)?;
    if packed != bytes {
        return Err(VerifyError::ArtifactNotFromRun {
            file_sha256: decider.identity().artifact_sha256.clone(),
            packed_sha256: sha256_hex(&packed),
        });
    }
    drop(packed);
    drop(bytes);
    verify_loaded(&inputs, checked, decider, base_dir, policy)
}

/// The `pack_run_dir` bytes of a `synthetic-fixture` run — the ONE way to write a test
/// artifact, and one every verify refuses ([`check_variant`]).
///
/// # Errors
///
/// [`VerifyError::NotSyntheticFixture`] for any other variant, or a pack refusal.
pub fn fixture_bytes(run_dir: &Path, data_dir: &Path) -> Result<Vec<u8>, VerifyError> {
    let inputs = PackInputs::from_run_dir(run_dir, data_dir)?;
    if inputs.recipe.variant != SYNTHETIC_FIXTURE_VARIANT {
        return Err(VerifyError::NotSyntheticFixture {
            variant: inputs.recipe.variant,
        });
    }
    Ok(artifact::write_decide_apr(&inputs)?)
}

/// [`fixture_bytes`], then an atomic write to `out`. Returns the artifact sha256.
///
/// # Errors
///
/// As [`fixture_bytes`], or [`VerifyError::Read`] for the write.
pub fn pack_fixture(run_dir: &Path, data_dir: &Path, out: &Path) -> Result<String, VerifyError> {
    let bytes = fixture_bytes(run_dir, data_dir)?;
    write_atomic(out, &bytes)?;
    Ok(artifact::artifact_sha256_hex(&bytes))
}

#[cfg(test)]
mod tests;
