//! The packer: a Laya run dir (plus its data dir) -> ONE `decide-apr-v1` `.apr` (D-17).
//!
//! [`PackInputs::from_run_dir`] reads the `run_dir_layout` of
//! `contracts/laya-finetune-gate-v1.yaml` and refuses a run dir that contradicts itself
//! (a task copy that differs from the data dir's, a report whose hashes do not match the
//! files beside it) rather than laundering it into an artifact. [`pack_run_dir`] then
//! hands the inputs to [`crate::artifact::write_decide_apr`].
//!
//! # What this module deliberately does NOT do
//!
//! It applies no gate or variant POLICY: a failing gate report and a `synthetic-fixture`
//! recipe both pack, because the tiny test fixture is exactly that. Refusing a failing
//! report, and requiring the report thresholds to equal the contract, belong to plan
//! 08-09's pack CLI and verifier. Deployability is decide-apr-v1 `deploy_eligibility`,
//! never "it packed".
//!
//! # SafeTensors scope
//!
//! Production code reads SafeTensors only in src/pack.rs (this file), on the PACK-time and
//! verify-time back-office path: [`PackInputs::from_run_dir`] parses the run
//! dir's checkpoint, and `verify` hands the base checkpoint's bytes to
//! [`scorer_from_parts`] here rather than parsing them itself. The one other parser in the
//! crate is the `#[cfg(test)]` fixture builder `src/test_support.rs`, which parses the tiny
//! checkpoint to build test artifacts and is never compiled into a server.
//!
//! The LOAD path (`artifact`, `task`, `laya`) reads only `.apr` bytes. It uses this
//! module's serde record TYPES ([`Recipe`], [`GateReport`]) to parse the blobs the `.apr`
//! carries, but calls no function here that reads a file, and it hashes through
//! [`crate::digest`], not through this module. So the SafeTensors carve-out in CLAUDE.md
//! is not widened. Enforced by the lib test `safetensors_is_read_only_by_pack`, which
//! scans every `.rs` file under `src/`.

use crate::artifact::{self, ArtifactError, BaseDecl, InputsSha256, ProbeRecord};
use crate::digest::sha256_hex;
use crate::laya::{Laya, LayaError};
use crate::Task;
use aprender::format::v2::{AprV2Metadata, AprV2ReaderRef, AprV2Writer, TensorDType};
use serde::Deserialize;
use std::fmt;
use std::path::{Path, PathBuf};

/// One checkpoint tensor exactly as stored: raw little-endian bytes, never re-rounded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointTensor {
    /// The verbatim Laya tensor name.
    pub name: String,
    /// `F16` for every weight, `F32` for the `temperature` buffer.
    pub dtype: TensorDType,
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Raw little-endian bytes copied from `model.safetensors`.
    pub bytes: Vec<u8>,
}

/// `recipe.json` (laya-finetune-gate-v1 `recipe_json_schema`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// `production` or `synthetic-fixture`.
    pub variant: String,
    /// `adamw`.
    pub optimizer: String,
    /// Encoder learning rate.
    pub encoder_lr: f64,
    /// Head learning rate.
    pub head_lr: f64,
    /// Cosine floor.
    pub eta_min: f64,
    /// AdamW weight decay.
    pub weight_decay: f64,
    /// Gradient clip norm.
    pub grad_clip: f64,
    /// Batch size.
    pub batch_size: u64,
    /// Spherical-score reward weight.
    pub proper_reward_w_sph: f64,
    /// RPS reward weight.
    pub proper_reward_w_rps: f64,
    /// `cosine`.
    pub schedule: String,
    /// Shots per class.
    pub shots_per_class: u64,
    /// Epochs.
    pub epochs: u64,
    /// Training seed.
    pub seed: i64,
    /// The declared base (D-04).
    pub base: BaseDecl,
    /// The early-stopping rule (laya-finetune-gate-v1 1.1.0 `early_stopping`); absent for the
    /// `fixed_epochs` rule, in which case `epochs` is exact rather than a maximum.
    #[serde(default)]
    pub early_stopping: Option<EarlyStoppingDecl>,
    /// The median-ECE seed selection (laya-finetune-gate-v1 1.4.0 `seed_policy`, A3); ABSENT =
    /// the legacy 1.x declared-seed rule, which is never deploy-eligible under 1.4.0.
    #[serde(default)]
    pub seed_selection: Option<SeedSelectionDecl>,
}

/// `recipe.json` `seed_selection` (laya-finetune-gate-v1 `recipe_json_schema.seed_selection`),
/// equal to the contract's `seed_policy` and written before training.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedSelectionDecl {
    /// `median_ece` (`seed_policy.selection`).
    pub policy: String,
    /// The gate seeds, in order (`seed_policy.variance_seeds`).
    pub seeds: Vec<i64>,
    /// `seed_policy.rank_scale`.
    pub rank_scale: f64,
    /// `smaller_seed` (`seed_policy.tie_break`).
    pub tie_break: String,
}

/// `recipe.json` `early_stopping` (laya-finetune-gate-v1 `early_stopping` block), copied into
/// the recipe before training so the recipe_id names the stopping rule.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EarlyStoppingDecl {
    /// `calibration_nll_at_fitted_t`.
    pub monitor: String,
    /// `min`.
    pub mode: String,
    /// Epochs between monitor evaluations.
    pub eval_every_epochs: u64,
    /// The first epoch that may be restored (epoch 0 is the untrained base).
    pub first_candidate_epoch: u64,
    /// Epochs without improvement before stopping.
    pub patience_epochs: u64,
    /// The improvement an epoch must exceed.
    pub min_delta: f64,
    /// `best`.
    pub restore: String,
    /// `earliest`.
    pub tie_break: String,
}

/// `gate-report.json` `thresholds`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateThresholds {
    /// Required macro-F1 margin over zero-shot.
    pub min_macro_f1_margin: f64,
    /// ECE ceiling after calibration.
    pub max_ece: f64,
    /// ECE bins.
    pub ece_bins: u64,
}

/// `gate-report.json` `zero_shot`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZeroShotMetrics {
    /// Macro-F1.
    pub macro_f1: f64,
    /// Stance F_avg; `null` for non-stance tasks.
    pub f_avg: Option<f64>,
    /// Top-label ECE.
    pub ece: f64,
    /// Eval rows.
    pub n: u64,
}

/// `gate-report.json` `fine_tuned`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FineTunedMetrics {
    /// Macro-F1.
    pub macro_f1: f64,
    /// Stance F_avg; `null` for non-stance tasks.
    pub f_avg: Option<f64>,
    /// ECE before calibration.
    pub ece_pre: f64,
    /// ECE after calibration.
    pub ece_post: f64,
    /// Eval NLL.
    pub nll: f64,
    /// Eval rows.
    pub n: u64,
}

/// `gate-report.json` `calibration`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateCalibration {
    /// Temperature bucket key (`choice:3-5`).
    pub bucket: String,
    /// Fitted temperature.
    pub t_fitted: f64,
    /// Applied (clamped) temperature.
    pub t_applied: f64,
    /// Whether the fit hit a bound.
    pub clamp_hit: bool,
    /// Calibration slice size.
    pub slice_size: u64,
    /// Sorted 0-based `train.jsonl` row indices of the slice.
    pub slice_ids: Vec<u64>,
    /// sha256 of the compact JSON of `slice_ids`.
    pub slice_ids_sha256: String,
}

/// `gate-report.json` `seeds`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateSeeds {
    /// The declared seed.
    pub declared: i64,
    /// Seeds run.
    pub n: u64,
    /// `single seed`, `median-ECE seed of N seeds` (1.4.0 median rule) or the legacy
    /// multi-seed `mean ± sd over N seeds` (laya-finetune-gate-v1 `seed_policy.rule`).
    pub label: String,
    /// `median_ece` under seed selection (1.4.0); absent for a legacy run.
    #[serde(default)]
    pub policy: Option<String>,
    /// The shipped (median) seed under seed selection (1.4.0).
    #[serde(default)]
    pub shipped: Option<i64>,
    /// Every gate seed's run, in seed order (1.4.0).
    #[serde(default)]
    pub per_seed: Option<Vec<PerSeedRow>>,
}

/// One `seeds.per_seed` row (laya-finetune-gate-v1 `gate_report_schema.seeds`, 1.4.0).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerSeedRow {
    /// The seed.
    pub seed: i64,
    /// Macro-F1 on the gate eval rows.
    pub macro_f1: f64,
    /// Stance F_avg; `null` for non-stance tasks.
    pub f_avg: Option<f64>,
    /// ECE after that seed's own calibration.
    pub ece_post: f64,
    /// `macro_f1 - zero_shot.macro_f1`.
    pub margin: f64,
    /// That seed's run's gate verdict.
    pub pass: bool,
    /// That seed's applied temperature.
    pub t_applied: f64,
    /// `floor(ece_post x rank_scale)`.
    pub rank_key: i64,
    /// sha256 of `seeds/seed-<s>/eval-probs.json`.
    pub eval_probs_sha256: String,
    /// sha256 of that seed's `model.safetensors` (only the shipped one is kept).
    pub model_safetensors_sha256: String,
}

/// `gate-report.json` `inputs_sha256`: the five hashes the artifact manifest also carries, plus
/// the OPTIONAL 1.4.0 `shift_jsonl` (present exactly when the data dir carries `shift.jsonl`).
/// A separate type from [`InputsSha256`] so the decide-apr-v1 manifest schema is untouched.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportInputsSha256 {
    /// `task.json`.
    pub task_json: String,
    /// `train.jsonl`.
    pub train_jsonl: String,
    /// `eval.jsonl`.
    pub eval_jsonl: String,
    /// The declared base `model.safetensors`.
    pub base_model: String,
    /// `tokenizer.json`.
    pub tokenizer_json: String,
    /// `shift.jsonl` (1.4.0 shift probe).
    #[serde(default)]
    pub shift_jsonl: Option<String>,
}

/// `gate-report.json` `shift_probe` (1.4.0, A2): REPORTED, recomputed by the verifier, never
/// re-scored and never a gate clause.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftProbe {
    /// Always `false`: the probe is never a gate clause.
    pub gate_clause: bool,
    /// `shift.jsonl` rows.
    pub n: u64,
    /// The declared base on the shift rows.
    pub zero_shot: ShiftZeroShot,
    /// The shipped checkpoint on the shift rows.
    pub fine_tuned: ShiftFineTuned,
    /// `fine_tuned.macro_f1 - zero_shot.macro_f1` on the shift rows.
    pub margin: f64,
    /// sha256 of `shift-probs.json`.
    pub probs_sha256: String,
    /// sha256 of `shift-zero-shot-probs.json`.
    pub zero_shot_probs_sha256: String,
}

/// `shift_probe.zero_shot`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftZeroShot {
    /// Macro-F1.
    pub macro_f1: f64,
    /// Stance F_avg; `null` for non-stance tasks.
    pub f_avg: Option<f64>,
    /// Top-label ECE.
    pub ece: f64,
}

/// `shift_probe.fine_tuned`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftFineTuned {
    /// Macro-F1.
    pub macro_f1: f64,
    /// Stance F_avg; `null` for non-stance tasks.
    pub f_avg: Option<f64>,
    /// Top-label ECE at the shipped seed's applied T.
    pub ece_post: f64,
}

/// `gate-report.json` (laya-finetune-gate-v1 `gate_report_schema`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateReport {
    /// `laya-gate-report-v1`.
    pub schema: String,
    /// The trainer's verdict (08-09 recomputes it; this module does not judge it).
    pub pass: bool,
    /// Thresholds the trainer used.
    pub thresholds: GateThresholds,
    /// Zero-shot metrics on the eval rows.
    pub zero_shot: ZeroShotMetrics,
    /// Fine-tuned metrics on the eval rows.
    pub fine_tuned: FineTunedMetrics,
    /// `fine_tuned.macro_f1 - zero_shot.macro_f1`.
    pub margin: f64,
    /// The temperature fit.
    pub calibration: GateCalibration,
    /// Seed record.
    pub seeds: GateSeeds,
    /// Device read back from the parameters.
    pub device_used: String,
    /// Whether that device is the CPU.
    pub device_is_cpu: bool,
    /// torch version.
    pub torch_version: String,
    /// sha256 of `recipe.json`.
    pub recipe_id: String,
    /// Input hashes.
    pub inputs_sha256: ReportInputsSha256,
    /// sha256 of `eval-probs.json`.
    pub eval_probs_sha256: String,
    /// sha256 of `zero-shot-probs.json`.
    pub zero_shot_probs_sha256: String,
    /// sha256 of `probes.json`.
    pub probes_sha256: String,
    /// sha256 of `rescore-noise.json` (1.4.0, laya-parity-v1 A1). ABSENT = the 1.0e-5 floor for
    /// both re-scores; every 1.x report parses unchanged.
    #[serde(default)]
    pub rescore_noise_sha256: Option<String>,
    /// The shift probe (1.4.0, A2); present exactly when the data dir carries `shift.jsonl`.
    #[serde(default)]
    pub shift_probe: Option<ShiftProbe>,
}

/// The gate report `schema` value this packer reads.
pub const GATE_REPORT_SCHEMA: &str = "laya-gate-report-v1";

/// The `rescore-noise.json` `schema` value (laya-finetune-gate-v1 `rescore_noise_schema`).
pub const RESCORE_NOISE_SCHEMA: &str = "laya-rescore-noise-v1";

/// `rescore-noise.json` (laya-finetune-gate-v1 `rescore_noise_schema`; laya-parity-v1
/// `rescore_noise_reference`): torch's own float64 answer for every eval row of the shipped
/// checkpoint and of the declared base. Only its stored ROWS are evidence — the verifier
/// recomputes the noise and the bound from them and cross-checks the reported values.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RescoreNoise {
    /// `laya-rescore-noise-v1`.
    pub schema: String,
    /// `float64`.
    pub reference: String,
    /// Copy of laya-parity-v1 `constants.pack_rescore_noise_k`.
    pub k: f64,
    /// Copy of laya-parity-v1 `pack_rescore_probs_abs.float_tolerance` (the floor).
    pub floor_abs: f64,
    /// `max |dz|` of the manual fp32 forward against the Scorer's logits; must be 0.0.
    pub control_max_abs: f64,
    /// The first `min(5, n)` eval rows the control forwarded.
    pub control_rows: Vec<usize>,
    /// Exactly `fine_tuned` then `zero_shot`.
    pub sets: Vec<NoiseSet>,
}

/// One scored set of a [`RescoreNoise`] record.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseSet {
    /// `fine_tuned` (the shipped checkpoint) or `zero_shot` (the declared base).
    pub which: String,
    /// `eval`.
    pub scored: String,
    /// The temperature the scorer applied.
    pub t_applied: f64,
    /// Eval rows.
    pub n: usize,
    /// Rows whose float64 argmax equals the stored float32 argmax (reported).
    pub argmax_agree: usize,
    /// Reported `max |p_torch32 - p_f64|` (cross-checked, never used).
    pub max_abs: f64,
    /// Reported `max(floor, k x max_abs)` (cross-checked, never used).
    pub bound: f64,
    /// Every eval row once, in order.
    pub rows: Vec<NoiseRow>,
}

/// One float64 row of a [`NoiseSet`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseRow {
    /// 0-based eval row.
    pub row: usize,
    /// `softmax(z_f64 / T)` in criteria order.
    pub probabilities_f64: Vec<f64>,
}

/// `probes.json` (laya-finetune-gate-v1 `probes_json_schema`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbesFile {
    /// One record per decide-apr-v1 probe input, in order.
    pub probes: Vec<ProbeRecord>,
}

/// Everything one artifact is packed from, already read and cross-checked.
#[derive(Debug, Clone, PartialEq)]
pub struct PackInputs {
    /// Every checkpoint tensor, sorted by name.
    pub tensors: Vec<CheckpointTensor>,
    /// `checkpoint/encoder/config.json`, byte-exact.
    pub encoder_config: Vec<u8>,
    /// `checkpoint/rl_agent_config.json`, byte-exact.
    pub agent_config: Vec<u8>,
    /// `checkpoint/tokenizer/tokenizer.json`, byte-exact.
    pub tokenizer: Vec<u8>,
    /// `task.json`, byte-exact (equal to the data dir's).
    pub task_json: Vec<u8>,
    /// `recipe.json`, byte-exact (its sha256 is the recipe_id).
    pub recipe_json: Vec<u8>,
    /// `gate-report.json`, byte-exact.
    pub gate_report_json: Vec<u8>,
    /// Parsed `recipe.json`.
    pub recipe: Recipe,
    /// Parsed `gate-report.json`.
    pub gate_report: GateReport,
    /// Parsed `probes.json`: Laya's OWN probe probabilities (Python values).
    pub probes: Vec<ProbeRecord>,
    /// `eval-probs.json`, byte-exact (hash-checked against the report; plan 08-09's verifier
    /// validates and re-scores it).
    pub eval_probs_json: Vec<u8>,
    /// `zero-shot-probs.json`, byte-exact (hash-checked against the report).
    pub zero_shot_probs_json: Vec<u8>,
    /// `rescore-noise.json`, byte-exact and hash-checked, present exactly when the report names
    /// it (`rescore_noise_sha256`); the verifier's `rescore_bounds` parses and recomputes it.
    pub rescore_noise_json: Option<Vec<u8>>,
    /// sha256 of `checkpoint/model.safetensors` (the shipped seed's per_seed row must name it).
    pub checkpoint_sha256: String,
    /// `(seed, seeds/seed-<s>/eval-probs.json)` for every `seeds.per_seed` row, in row order,
    /// each hash-checked against its row (empty for a report without per_seed).
    pub seed_eval_probs: Vec<(i64, Vec<u8>)>,
    /// `shift-probs.json`, hash-checked, present exactly when the report carries `shift_probe`.
    pub shift_probs_json: Option<Vec<u8>>,
    /// `shift-zero-shot-probs.json`, hash-checked, present exactly with `shift_probe`.
    pub shift_zero_shot_probs_json: Option<Vec<u8>>,
    /// Input hashes recomputed from the data dir, tokenizer and declared base.
    pub inputs_sha256: InputsSha256,
}

/// Why a run dir could not be packed.
#[derive(Debug, Clone, PartialEq)]
pub enum PackError {
    /// A run-dir or data-dir file could not be read.
    Read {
        /// The path.
        path: PathBuf,
        /// The I/O error.
        reason: String,
    },
    /// `model.safetensors` could not be parsed.
    SafeTensors(String),
    /// A checkpoint tensor has a dtype the artifact does not carry.
    UnsupportedDtype {
        /// Tensor name.
        name: String,
        /// The safetensors dtype.
        dtype: String,
    },
    /// A run-dir JSON file did not parse under its contract schema.
    Schema {
        /// File name.
        file: &'static str,
        /// The serde error.
        reason: String,
    },
    /// The gate report names a schema this packer does not read.
    GateReportSchema {
        /// The `schema` value found.
        observed: String,
    },
    /// The run dir's `task.json` is not a byte copy of the data dir's.
    TaskCopyDiffers,
    /// A hash the gate report records disagrees with the file beside it.
    ReportHashMismatch {
        /// Which recorded hash.
        what: &'static str,
        /// Recorded in the report.
        recorded: String,
        /// Recomputed from the file.
        observed: String,
    },
    /// A `seeds/seed-<s>/eval-probs.json` disagrees with its `seeds.per_seed` row's
    /// `eval_probs_sha256` (1.4.0, A3).
    SeedProbsHashMismatch {
        /// The seed.
        seed: i64,
        /// Recorded in the per_seed row.
        recorded: String,
        /// Recomputed from the file.
        observed: String,
    },
    /// The artifact writer refused.
    Artifact(ArtifactError),
    /// A checkpoint could not be rebuilt into a Laya model for scoring
    /// ([`load_checkpoint_for_scoring`]).
    Rebuild(LayaError),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, reason } => write!(f, "pack: read {}: {reason}", path.display()),
            Self::SafeTensors(e) => write!(f, "pack: model.safetensors: {e}"),
            Self::UnsupportedDtype { name, dtype } => {
                write!(f, "pack: tensor {name} has dtype {dtype}; only F16 (and F32 temperature) are carried")
            }
            Self::Schema { file, reason } => write!(f, "pack: {file}: {reason}"),
            Self::GateReportSchema { observed } => write!(
                f,
                "pack: gate-report.json schema is {observed:?}, expected {GATE_REPORT_SCHEMA:?}"
            ),
            Self::TaskCopyDiffers => write!(
                f,
                "pack: the run dir's task.json is not a byte copy of the data dir's task.json"
            ),
            Self::ReportHashMismatch {
                what,
                recorded,
                observed,
            } => write!(
                f,
                "pack: gate report records {what} = {recorded}, the file hashes to {observed}"
            ),
            Self::SeedProbsHashMismatch {
                seed,
                recorded,
                observed,
            } => write!(
                f,
                "pack: seeds/seed-{seed}/eval-probs.json hashes to {observed}, its per_seed row records {recorded}"
            ),
            Self::Artifact(e) => write!(f, "pack: {e}"),
            Self::Rebuild(e) => write!(f, "pack: rebuild for scoring: {e}"),
        }
    }
}

impl std::error::Error for PackError {}

impl From<ArtifactError> for PackError {
    fn from(e: ArtifactError) -> Self {
        Self::Artifact(e)
    }
}

fn read_file(dir: &Path, rel: &str) -> Result<Vec<u8>, PackError> {
    let path = dir.join(rel);
    std::fs::read(&path).map_err(|e| PackError::Read {
        path,
        reason: e.to_string(),
    })
}

fn parse<T: serde::de::DeserializeOwned>(file: &'static str, bytes: &[u8]) -> Result<T, PackError> {
    serde_json::from_slice(bytes).map_err(|e| PackError::Schema {
        file,
        reason: e.to_string(),
    })
}

fn check_hash(what: &'static str, recorded: &str, bytes: &[u8]) -> Result<(), PackError> {
    let observed = sha256_hex(bytes);
    if observed == recorded {
        Ok(())
    } else {
        Err(PackError::ReportHashMismatch {
            what,
            recorded: recorded.to_string(),
            observed,
        })
    }
}

/// Every tensor of `model.safetensors` as raw bytes, sorted by name. F16 stays F16 and
/// F32 stays F32; any other dtype is a typed refusal.
fn read_checkpoint_tensors(bytes: &[u8]) -> Result<Vec<CheckpointTensor>, PackError> {
    let st = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|e| PackError::SafeTensors(e.to_string()))?;
    let mut out = st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let dtype = match view.dtype() {
                safetensors::Dtype::F16 => TensorDType::F16,
                safetensors::Dtype::F32 => TensorDType::F32,
                other => {
                    return Err(PackError::UnsupportedDtype {
                        name,
                        dtype: format!("{other:?}"),
                    })
                }
            };
            Ok(CheckpointTensor {
                name,
                dtype,
                shape: view.shape().to_vec(),
                bytes: view.data().to_vec(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

impl PackInputs {
    /// Read and cross-check a run dir (laya-finetune-gate-v1 `run_dir_layout`) and its
    /// data dir.
    ///
    /// Refuses when the run-dir `task.json` differs from the data dir's, when the gate
    /// report's `recipe_id`, `inputs_sha256`, `eval_probs_sha256`,
    /// `zero_shot_probs_sha256`, `probes_sha256` or (when present) `rescore_noise_sha256`,
    /// a `seeds.per_seed` row's `eval_probs_sha256` or the `shift_probe` file hashes disagree
    /// with the files, or when any
    /// JSON file carries a field its contract schema does not declare.
    ///
    /// # Errors
    ///
    /// A [`PackError`] naming the file or hash that failed.
    pub fn from_run_dir(run_dir: &Path, data_dir: &Path) -> Result<Self, PackError> {
        let task_json = read_file(run_dir, "task.json")?;
        let data_task = read_file(data_dir, "task.json")?;
        if task_json != data_task {
            return Err(PackError::TaskCopyDiffers);
        }
        let train = read_file(data_dir, "train.jsonl")?;
        let eval = read_file(data_dir, "eval.jsonl")?;
        let tokenizer = read_file(run_dir, "checkpoint/tokenizer/tokenizer.json")?;
        let encoder_config = read_file(run_dir, "checkpoint/encoder/config.json")?;
        let agent_config = read_file(run_dir, "checkpoint/rl_agent_config.json")?;
        let recipe_json = read_file(run_dir, "recipe.json")?;
        let gate_report_json = read_file(run_dir, "gate-report.json")?;
        let probes_json = read_file(run_dir, "probes.json")?;
        let eval_probs = read_file(run_dir, "eval-probs.json")?;
        let zero_shot_probs = read_file(run_dir, "zero-shot-probs.json")?;

        let recipe: Recipe = parse("recipe.json", &recipe_json)?;
        let gate_report: GateReport = parse("gate-report.json", &gate_report_json)?;
        let probes: ProbesFile = parse("probes.json", &probes_json)?;
        if gate_report.schema != GATE_REPORT_SCHEMA {
            return Err(PackError::GateReportSchema {
                observed: gate_report.schema,
            });
        }

        check_hash("recipe_id", &gate_report.recipe_id, &recipe_json)?;
        check_hash("probes_sha256", &gate_report.probes_sha256, &probes_json)?;
        check_hash(
            "eval_probs_sha256",
            &gate_report.eval_probs_sha256,
            &eval_probs,
        )?;
        check_hash(
            "zero_shot_probs_sha256",
            &gate_report.zero_shot_probs_sha256,
            &zero_shot_probs,
        )?;
        let rescore_noise_json = match &gate_report.rescore_noise_sha256 {
            Some(recorded) => {
                let bytes = read_file(run_dir, "rescore-noise.json")?;
                check_hash("rescore_noise_sha256", recorded, &bytes)?;
                Some(bytes)
            }
            None => None,
        };
        let mut seed_eval_probs = Vec::new();
        for row in gate_report.seeds.per_seed.iter().flatten() {
            let bytes = read_file(run_dir, &format!("seeds/seed-{}/eval-probs.json", row.seed))?;
            let observed = sha256_hex(&bytes);
            if observed != row.eval_probs_sha256 {
                return Err(PackError::SeedProbsHashMismatch {
                    seed: row.seed,
                    recorded: row.eval_probs_sha256.clone(),
                    observed,
                });
            }
            seed_eval_probs.push((row.seed, bytes));
        }
        let (shift_probs_json, shift_zero_shot_probs_json) = match &gate_report.shift_probe {
            Some(probe) => {
                let ft = read_file(run_dir, "shift-probs.json")?;
                check_hash("shift_probe.probs_sha256", &probe.probs_sha256, &ft)?;
                let zs = read_file(run_dir, "shift-zero-shot-probs.json")?;
                check_hash(
                    "shift_probe.zero_shot_probs_sha256",
                    &probe.zero_shot_probs_sha256,
                    &zs,
                )?;
                (Some(ft), Some(zs))
            }
            None => (None, None),
        };
        let recorded = &gate_report.inputs_sha256;
        check_hash("inputs_sha256.task_json", &recorded.task_json, &data_task)?;
        check_hash("inputs_sha256.train_jsonl", &recorded.train_jsonl, &train)?;
        check_hash("inputs_sha256.eval_jsonl", &recorded.eval_jsonl, &eval)?;
        check_hash(
            "inputs_sha256.tokenizer_json",
            &recorded.tokenizer_json,
            &tokenizer,
        )?;
        if recorded.base_model != recipe.base.sha256 {
            return Err(PackError::ReportHashMismatch {
                what: "inputs_sha256.base_model",
                recorded: recorded.base_model.clone(),
                observed: recipe.base.sha256.clone(),
            });
        }
        let inputs_sha256 = InputsSha256 {
            task_json: sha256_hex(&data_task),
            train_jsonl: sha256_hex(&train),
            eval_jsonl: sha256_hex(&eval),
            base_model: recipe.base.sha256.clone(),
            tokenizer_json: sha256_hex(&tokenizer),
        };

        let model = read_file(run_dir, "checkpoint/model.safetensors")?;
        let checkpoint_sha256 = sha256_hex(&model);
        let tensors = read_checkpoint_tensors(&model)?;
        Ok(Self {
            tensors,
            encoder_config,
            agent_config,
            tokenizer,
            task_json,
            recipe_json,
            gate_report_json,
            recipe,
            gate_report,
            probes: probes.probes,
            eval_probs_json: eval_probs,
            zero_shot_probs_json: zero_shot_probs,
            rescore_noise_json,
            checkpoint_sha256,
            seed_eval_probs,
            shift_probs_json,
            shift_zero_shot_probs_json,
            inputs_sha256,
        })
    }
}

/// Pack a run dir into `decide-apr-v1` bytes: [`PackInputs::from_run_dir`] then
/// [`artifact::write_decide_apr`]. No gate or variant policy (see the module docs).
///
/// # Errors
///
/// A [`PackError`] from reading, cross-checking or writing.
#[provable_contracts_macros::contract("decide-apr-v1", equation = "determinism")]
pub fn pack_run_dir(run_dir: &Path, data_dir: &Path) -> Result<Vec<u8>, PackError> {
    let inputs = PackInputs::from_run_dir(run_dir, data_dir)?;
    Ok(artifact::write_decide_apr(&inputs)?)
}

/// Build an in-memory Laya scorer from a checkpoint DIRECTORY (`model.safetensors`,
/// `encoder/config.json`, `rl_agent_config.json`, `tokenizer/tokenizer.json` — the layout of
/// both a run dir's `checkpoint/` and the declared base snapshot), bound to `task`.
///
/// Every tensor goes through the SAME F16 path the packer uses: raw bytes into an in-memory
/// `.apr`, widened by core's loader. This is the back-office zero-shot scorer of plan 08-09's
/// verifier (the declared base re-scored against `zero-shot-probs.json`). It returns a
/// [`Laya`], never a [`crate::Decider`]: nothing built here is servable, and no load ladder,
/// probe replay or manifest applies to it.
///
/// # Errors
///
/// [`PackError::Read`] for a missing file, [`PackError::SafeTensors`] /
/// [`PackError::UnsupportedDtype`] for the weights, [`PackError::Artifact`] for the in-memory
/// container, and [`PackError::Rebuild`] when Laya refuses the parts.
pub fn load_checkpoint_for_scoring(checkpoint_dir: &Path, task: Task) -> Result<Laya, PackError> {
    scorer_from_parts(
        &read_file(checkpoint_dir, "model.safetensors")?,
        &read_file(checkpoint_dir, "encoder/config.json")?,
        &read_file(checkpoint_dir, "rl_agent_config.json")?,
        &read_file(checkpoint_dir, "tokenizer/tokenizer.json")?,
        task,
    )
}

/// The scorer [`load_checkpoint_for_scoring`] builds, from checkpoint files the caller has
/// ALREADY read (and hashed): what is scored is exactly the bytes the caller checked, with no
/// window between a hash and a second read of the same path.
///
/// # Errors
///
/// As [`load_checkpoint_for_scoring`], minus the file reads.
pub fn scorer_from_parts(
    model_safetensors: &[u8],
    encoder_config: &[u8],
    agent_config: &[u8],
    tokenizer: &[u8],
    task: Task,
) -> Result<Laya, PackError> {
    let tensors = read_checkpoint_tensors(model_safetensors)?;
    let apr = {
        let mut w = AprV2Writer::new(AprV2Metadata::default());
        for t in tensors {
            w.add_tensor(t.name, t.dtype, t.shape, t.bytes);
        }
        w.write().map_err(|e| {
            PackError::Artifact(ArtifactError::Write {
                reason: e.to_string(),
            })
        })?
    };
    let reader = AprV2ReaderRef::from_bytes(&apr).map_err(|e| {
        PackError::Artifact(ArtifactError::Container {
            reason: e.to_string(),
        })
    })?;
    Laya::from_parts(&reader, "", encoder_config, agent_config, tokenizer, task)
        .map_err(PackError::Rebuild)
}

#[cfg(test)]
mod tests {
    use super::{PackError, PackInputs};
    use crate::test_support::fixture_dir;
    use std::path::{Path, PathBuf};

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).expect("create dir");
        for entry in std::fs::read_dir(from).expect("read dir") {
            let entry = entry.expect("dir entry");
            let target = to.join(entry.file_name());
            if entry.file_type().expect("file type").is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).expect("copy file");
            }
        }
    }

    /// A private copy of the tiny run dir, so a test can corrupt one file.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("decide-0805-{name}-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("clear scratch");
        }
        copy_dir(&fixture_dir(), &dir);
        dir
    }

    fn append(path: &Path, extra: &[u8]) {
        let mut bytes = std::fs::read(path).expect("read");
        bytes.extend_from_slice(extra);
        std::fs::write(path, bytes).expect("write");
    }

    #[test]
    fn task_copy_must_match() {
        let dir = scratch("task-copy");
        append(&dir.join("task.json"), b" ");
        let e = PackInputs::from_run_dir(&dir, &dir.join("data")).expect_err("copy differs");
        assert_eq!(e, PackError::TaskCopyDiffers);
        std::fs::remove_dir_all(&dir).expect("clean up");
    }

    #[test]
    fn report_hash_must_match() {
        let dir = scratch("probes-hash");
        append(&dir.join("probes.json"), b"\n");
        let e = PackInputs::from_run_dir(&dir, &dir.join("data")).expect_err("hash differs");
        assert!(
            matches!(
                &e,
                PackError::ReportHashMismatch {
                    what: "probes_sha256",
                    ..
                }
            ),
            "{e}"
        );
        std::fs::remove_dir_all(&dir).expect("clean up");
    }

    /// The early-stopping recipe (laya-finetune-gate-v1 1.1.0): `early_stopping` is the ONE
    /// optional recipe.json key. Absent = fixed_epochs (every pre-1.1.0 run dir still parses);
    /// present = a strictly-typed object; an unknown key inside it is still refused.
    #[test]
    fn recipe_early_stopping_is_optional_and_strict() {
        let fixed = std::fs::read_to_string(fixture_dir().join("recipe.json")).expect("recipe");
        let r: super::Recipe = serde_json::from_str(&fixed).expect("fixed_epochs recipe parses");
        assert_eq!(r.early_stopping, None);

        let es = r#""early_stopping":{"eval_every_epochs":1,"first_candidate_epoch":1,"min_delta":0.001,"mode":"min","monitor":"calibration_nll_at_fitted_t","patience_epochs":3,"restore":"best","tie_break":"earliest"},"#;
        let with = fixed.replacen('{', &format!("{{{es}"), 1);
        let r: super::Recipe = serde_json::from_str(&with).expect("early_stopping recipe parses");
        let got = r.early_stopping.expect("early_stopping present");
        assert_eq!(got.monitor, "calibration_nll_at_fitted_t");
        assert_eq!(got.mode, "min");
        assert_eq!(
            (
                got.eval_every_epochs,
                got.first_candidate_epoch,
                got.patience_epochs
            ),
            (1, 1, 3)
        );
        assert!((got.min_delta - 0.001).abs() < 1e-15);
        assert_eq!(
            (got.restore.as_str(), got.tie_break.as_str()),
            ("best", "earliest")
        );

        let bad = with.replacen("\"mode\"", "\"extra\":1,\"mode\"", 1);
        let e = serde_json::from_str::<super::Recipe>(&bad).expect_err("unknown nested key");
        assert!(e.to_string().contains("extra"), "{e}");
    }

    #[test]
    fn unknown_recipe_key_is_refused() {
        let dir = scratch("recipe-key");
        let path = dir.join("recipe.json");
        let text = std::fs::read_to_string(&path).expect("recipe");
        let text = text.replacen('{', "{\n  \"extra\": 1,", 1);
        std::fs::write(&path, text).expect("write recipe");
        let e = PackInputs::from_run_dir(&dir, &dir.join("data")).expect_err("unknown key");
        assert!(
            matches!(&e, PackError::Schema { file: "recipe.json", reason } if reason.contains("extra")),
            "{e}"
        );
        std::fs::remove_dir_all(&dir).expect("clean up");
    }
}
