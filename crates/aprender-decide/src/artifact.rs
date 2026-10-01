//! The `decide-apr-v1` artifact (D-17): what a decision-model `.apr` contains, how it
//! is written, and the load LADDER that is the only way to a verified
//! [`Decider`](crate::Decider).
//!
//! Contract: `contracts/decide-apr-v1.yaml` (manifest, blob set, load ladder, probe
//! policy, identity, determinism). Every constant below restates that contract and is
//! asserted equal to it by a test.
//!
//! # Layout
//!
//! - every checkpoint weight under its verbatim Laya name, F16 raw little-endian bytes
//!   copied from `model.safetensors` (never re-rounded through f32); the `temperature`
//!   buffer stays F32; `act_head.*` is stored so the tensor set is a bijection with the
//!   checkpoint, and is unused at inference;
//! - six U8 blob tensors ([`BLOB_TENSORS`]): tokenizer, task, encoder config, agent
//!   config, recipe, gate report — raw bytes, sha256 recorded in the manifest;
//! - `model_type` [`MODEL_TYPE_TAG`] and EXACTLY ONE custom metadata key
//!   [`CUSTOM_METADATA_KEY`], whose value is the manifest as a JSON STRING serialized
//!   from typed structs (arrays, never maps), so its bytes do not depend on the
//!   serde_json map backing a build links;
//! - no timestamp or host field anywhere.
//!
//! # Probes store PYTHON values
//!
//! The manifest's probe expectations are copied from the run dir's `probes.json`
//! (Laya's own predict path). The packer runs its Rust probe and checks it against
//! them within [`PROBE_PROBABILITIES_ABS`], but never serializes its own values: the
//! Rust probe bits come from the packing machine's GEMM kernel (NEON on the dev box,
//! AVX on x86 CI), and an artifact that embedded them could not have one golden hash.
//!
//! # The ladder (load)
//!
//! 1. bounded read ([`read_decide_apr_bytes_bounded`]; in-memory length check)
//! 2. header (APR v2 version, CRC, row-major) and index extent, BEFORE any index parser runs
//! 3. `model_type`, exactly one custom key, the manifest (`deny_unknown_fields`)
//! 4. structure: unique index names, blob hashes, the architecture-derived tensor set, per-entry sizes,
//!    task labels == manifest labels, and every other manifest field equal to the
//!    sha-bound source decide-apr-v1 `manifest.bindings` names (recipe, gate report,
//!    blob digests, the task's bucket and the agent config's applied temperature)
//! 5. non-finite scan
//! 6. rebuild ([`Laya::from_parts`] over the zero-copy [`AprV2ReaderRef`])
//! 7. probe replay against the stored Python values
//! 8. mint — [`Decider`] has private fields, and this module is its only constructor.

use crate::digest::sha256_hex;
use crate::laya::temperature::{applied_temperature_f64, bucket_key};
use crate::laya::{AgentConfig, Laya, QType};
use crate::pack::{CheckpointTensor, PackInputs};
use crate::{DecideError, Decider, DecisionMethod, ModelIdentity, Task, TaskError};
use aprender::format::v2::{
    AprV2Header, AprV2Metadata, AprV2ReaderRef, AprV2Writer, TensorDType, HEADER_SIZE_V2,
    VERSION_V2,
};
use aprender::models::modernbert::{expected_modernbert_tensor_names, ModernBertConfig};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fmt;

// ===========================================================================
// Contract-resident constants (each asserted equal to decide-apr-v1 by a test)
// ===========================================================================

/// The manifest `schema` value.
pub const ARTIFACT_SCHEMA: &str = "decide-apr-v1";
/// The manifest `schema_version` value.
pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;
/// The APR `model_type` of a decision artifact.
pub const MODEL_TYPE_TAG: &str = "decide";
/// The ONE custom metadata key; its value is the manifest JSON string.
pub const CUSTOM_METADATA_KEY: &str = "decide";
/// The only method this schema version knows.
pub const METHOD_LAYA: &str = "laya";

/// Raw `tokenizer.json` (U8).
pub const TOKENIZER_BLOB: &str = "tokenizer.blob";
/// Raw `task.json` (U8).
pub const TASK_BLOB: &str = "decide.task_json";
/// Raw encoder `config.json` (U8).
pub const ENCODER_CONFIG_BLOB: &str = "decide.encoder_config";
/// Raw `rl_agent_config.json` (U8).
pub const AGENT_CONFIG_BLOB: &str = "decide.agent_config";
/// Raw `recipe.json` (U8); its sha256 is the recipe_id.
pub const RECIPE_BLOB: &str = "decide.recipe_json";
/// Raw `gate-report.json` (U8).
pub const GATE_REPORT_BLOB: &str = "decide.gate_report_json";
/// The blob set, in contract order (decide-apr-v1 `blob_tensors`).
pub const BLOB_TENSORS: [&str; 6] = [
    TOKENIZER_BLOB,
    TASK_BLOB,
    ENCODER_CONFIG_BLOB,
    AGENT_CONFIG_BLOB,
    RECIPE_BLOB,
    GATE_REPORT_BLOB,
];

/// 1.25 GiB (decide-apr-v1 `constants.max_artifact_bytes`).
pub const MAX_ARTIFACT_BYTES: u64 = 1_342_177_280;
/// decide-apr-v1 `constants.max_tensor_count`.
pub const MAX_TENSOR_COUNT: u32 = 4096;
/// The most metadata bytes (the manifest section) a header may declare (decide-apr-v1
/// `constants.max_metadata_bytes`). Laya-en's is ~2.4 KB. The container reader parses the
/// section into a JSON tree at 40-50x its size, so rung 2 bounds the declared length BEFORE
/// the reader may.
pub const MAX_METADATA_BYTES: u32 = 1_048_576;
/// The smallest encodable tensor-index entry: u16 name length + u8 dtype + u8 ndim +
/// u64 offset + u64 size (decide-apr-v1 `constants.min_index_entry_bytes`). The value
/// is apr-format's own (the reader's index-capacity bound), widened to u64 here.
pub const MIN_INDEX_ENTRY_BYTES: u64 = aprender::format::v2::MIN_INDEX_ENTRY_BYTES as u64;
/// A probe row longer than this is refused at pack (decide-apr-v1
/// `constants.probe_max_row_tokens`).
pub const PROBE_MAX_ROW_TOKENS: usize = 48;
/// Probe probability tolerance (decide-apr-v1 `constants.probe_probabilities_abs`).
pub const PROBE_PROBABILITIES_ABS: f64 = 1.0e-5;
/// The two synthetic probe inputs (decide-apr-v1 `probe_policy.inputs`), verbatim.
pub const PROBE_INPUTS: [&str; 2] = [
    "the quick brown fox jumps over the lazy dog",
    "Stance check: I firmly support this position!!! #debate @user123",
];
/// The synthetic probe task (decide-apr-v1 `probe_policy.probe_task`), as `task.json`.
pub const PROBE_TASK: &str =
    r#"{"type": "choice", "instructions": "probe", "criteria": {"yes": null, "no": null}}"#;

// ===========================================================================
// The manifest (decide-apr-v1 `manifest.fields`)
// ===========================================================================

/// The declared base checkpoint (D-04), copied from `recipe.json` `base`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseDecl {
    /// Model family (`laya`).
    pub family: String,
    /// Checkpoint name (`en-root`; `tiny-synthetic` for the fixture).
    pub checkpoint: String,
    /// Hub repo.
    pub repo: String,
    /// Pinned revision.
    pub revision: String,
    /// sha256 of the base `model.safetensors`.
    pub sha256: String,
}

impl BaseDecl {
    /// `laya-en-root@55cf4c4e`-style display string: family, checkpoint and the first
    /// eight characters of the revision.
    #[must_use]
    pub fn display(&self) -> String {
        let rev: String = self.revision.chars().take(8).collect();
        format!("{}-{}@{rev}", self.family, self.checkpoint)
    }
}

/// `manifest.agent`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentDecl {
    /// Decision-head layers.
    pub head_layers: usize,
    /// Row cap in tokens.
    pub max_len: usize,
    /// Head + options budget.
    pub head_max_len: usize,
}

impl From<&AgentConfig> for AgentDecl {
    fn from(a: &AgentConfig) -> Self {
        Self {
            head_layers: a.head_layers,
            max_len: a.max_len,
            head_max_len: a.head_max_len,
        }
    }
}

/// One `manifest.blobs` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlobHash {
    /// Blob tensor name.
    pub name: String,
    /// Lowercase-hex sha256 of its bytes.
    pub sha256: String,
}

/// `manifest.calibration`, copied from the gate report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationDecl {
    /// Bucket key.
    pub bucket: String,
    /// Fitted temperature.
    pub t_fitted: f64,
    /// Applied temperature.
    pub t_applied: f64,
    /// Whether the fit hit a bound.
    pub clamp_hit: bool,
    /// sha256 of the calibration slice ids.
    pub slice_ids_sha256: String,
}

/// `manifest.gate`: an identity summary of the embedded report, NEVER an eligibility
/// verdict (decide-apr-v1 `deploy_eligibility`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateSummary {
    /// The trainer's verdict as recorded.
    pub pass: bool,
    /// Recorded macro-F1 margin.
    pub margin: f64,
    /// Recorded post-calibration ECE.
    pub ece_post: f64,
    /// sha256 of the gate-report blob.
    pub report_sha256: String,
}

/// Input hashes (`manifest.inputs_sha256`, gate report `inputs_sha256`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputsSha256 {
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
}

/// One probe expectation: Laya's OWN probabilities from `probes.json` (Python values,
/// f32 bit patterns in hex), stored verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeRecord {
    /// Index into [`PROBE_INPUTS`].
    pub input_index: usize,
    /// Built-row length.
    pub tokens: usize,
    /// The probe task label Laya chose.
    pub label: String,
    /// One f32 bit pattern (8 hex digits) per probe-task criterion.
    pub probabilities_f32_hex: Vec<String>,
}

/// The ONE manifest document (the value of [`CUSTOM_METADATA_KEY`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// [`ARTIFACT_SCHEMA`].
    pub schema: String,
    /// [`ARTIFACT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// [`METHOD_LAYA`].
    pub method: String,
    /// The recipe variant (`production` / `synthetic-fixture`), carried so every
    /// reader sees it; deployability is 08-09's `verify`, not this field.
    pub variant: String,
    /// The declared base (D-04).
    pub base: BaseDecl,
    /// Criterion names in task.json document order (the label index, D-05).
    pub labels: Vec<String>,
    /// Agent geometry.
    pub agent: AgentDecl,
    /// Blob hashes, in [`BLOB_TENSORS`] order.
    pub blobs: Vec<BlobHash>,
    /// sha256 of the recipe blob (D-11).
    pub recipe_id: String,
    /// Calibration record.
    pub calibration: CalibrationDecl,
    /// Gate summary (identity only).
    pub gate: GateSummary,
    /// Input hashes.
    pub inputs_sha256: InputsSha256,
    /// Device the trainer used.
    pub device_used: String,
    /// Probe expectations (Python values).
    pub probes: Vec<ProbeRecord>,
}

// ===========================================================================
// Limits
// ===========================================================================

/// The artifact's resource bounds as one value.
///
/// The fields are private and the only value a shipped build can name is
/// [`Self::CONTRACTED`]; the shrinking constructor is `#[cfg(test)]`, so the MECHANISM
/// of each bound can be shown biting on a tiny real artifact while its VALUE is
/// asserted against the contract separately (the setfit-apr-v1 `ArtifactLimits`
/// shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactLimits {
    max_artifact_bytes: u64,
    probe_max_row_tokens: usize,
}

impl ArtifactLimits {
    /// The contracted bounds.
    pub const CONTRACTED: Self = Self {
        max_artifact_bytes: MAX_ARTIFACT_BYTES,
        probe_max_row_tokens: PROBE_MAX_ROW_TOKENS,
    };

    /// The byte cap.
    #[must_use]
    pub fn max_artifact_bytes(&self) -> u64 {
        self.max_artifact_bytes
    }

    /// The probe row cap.
    #[must_use]
    pub fn probe_max_row_tokens(&self) -> usize {
        self.probe_max_row_tokens
    }

    /// Deliberately shrunk bounds, so each cap can be shown biting on the tiny fixture.
    #[cfg(test)]
    pub(crate) const fn tiny(max_artifact_bytes: u64, probe_max_row_tokens: usize) -> Self {
        Self {
            max_artifact_bytes,
            probe_max_row_tokens,
        }
    }
}

// ===========================================================================
// Errors
// ===========================================================================

/// Why an artifact was refused — at pack, or at a named load rung.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactError {
    /// Rung 1 (or pack): over [`MAX_ARTIFACT_BYTES`]. `what` names the check:
    /// `declared_length`, `stream`, `input_bytes` or `packed`.
    ArtifactTooLarge {
        /// Which check fired.
        what: &'static str,
        /// Observed length.
        observed: u64,
        /// The cap.
        cap: u64,
    },
    /// Rung 1: the read failed.
    Read {
        /// The I/O error.
        reason: String,
    },
    /// Rung 2: the 64-byte header did not parse (too short, bad magic), or its version is not
    /// APR v2's (`VERSION_V2`, checked by rung 2 itself: the container reader does not).
    Header {
        /// The container error.
        reason: String,
    },
    /// Rung 2: the header CRC does not match.
    HeaderChecksum,
    /// Rung 2: the header carries the column-major flag (LAYOUT-002).
    ColumnMajor,
    /// Rung 2: `tensor_count` over [`MAX_TENSOR_COUNT`].
    TensorCountOverCap {
        /// Declared count.
        declared: u32,
        /// The cap.
        cap: u32,
    },
    /// Rung 2: `tensor_count x 20` does not fit between the index and data offsets.
    IndexExtentTooSmall {
        /// Declared count.
        declared: u32,
        /// Bytes between `tensor_index_offset` and `data_offset` (0 when inverted).
        extent: u64,
    },
    /// Rung 2: `data_offset` points past the end of the file.
    IndexPastEnd {
        /// Declared data offset.
        data_offset: u64,
        /// File length.
        file_len: u64,
    },
    /// Rung 2: `metadata_size` over [`MAX_METADATA_BYTES`].
    MetadataOverCap {
        /// Declared metadata length.
        declared: u32,
        /// The cap.
        cap: u32,
    },
    /// Rung 3: the container reader refused the metadata or index.
    Container {
        /// The container error.
        reason: String,
    },
    /// Rung 3: `model_type` is not [`MODEL_TYPE_TAG`].
    WrongModelType {
        /// Observed tag.
        observed: String,
    },
    /// Rung 3: the custom keys are not exactly `{decide}`.
    CustomKeys {
        /// Observed keys, sorted.
        observed: Vec<String>,
    },
    /// Rung 3: the manifest value is not a JSON string.
    ManifestNotString,
    /// Rung 3: the manifest string did not parse (`deny_unknown_fields`).
    ManifestParse {
        /// The serde error.
        reason: String,
    },
    /// Rung 3: `schema` is not [`ARTIFACT_SCHEMA`].
    WrongSchema {
        /// Observed schema.
        observed: String,
    },
    /// Rung 3: `schema_version` is not [`ARTIFACT_SCHEMA_VERSION`].
    SchemaVersion {
        /// Observed version.
        observed: u32,
    },
    /// Rung 3: a method this schema version does not know.
    UnknownMethod {
        /// Observed method.
        observed: String,
    },
    /// Rung 4: `manifest.blobs` is not exactly [`BLOB_TENSORS`] in order.
    ManifestBlobs {
        /// Names found.
        observed: Vec<String>,
    },
    /// Rung 4: a blob's bytes do not hash to the manifest's value.
    BlobHashMismatch {
        /// The blob.
        blob: String,
    },
    /// Rung 4 (or pack): a config blob did not parse.
    ConfigBlob {
        /// The blob.
        blob: &'static str,
        /// Why.
        reason: String,
    },
    /// Rung 4: the tensor index names a tensor twice (WR-01, decide side). Rungs 4 (c) and 5
    /// look entries up by name (first match), so a second entry would escape both.
    DuplicateTensor {
        /// The first repeated name, in index order.
        name: String,
    },
    /// Rung 4 (or pack): an expected tensor is absent.
    MissingTensor {
        /// Tensor name.
        name: String,
    },
    /// Rung 4 (or pack): a tensor the architecture does not derive is present.
    UnexpectedTensor {
        /// Tensor name.
        name: String,
    },
    /// Rung 4 (or pack): a tensor has the wrong dtype.
    DtypeMismatch {
        /// Tensor name.
        name: String,
        /// Expected dtype.
        expected: String,
        /// Stored dtype.
        observed: String,
    },
    /// Rung 4 (or pack): a tensor's byte size is not `product(shape) x width`.
    SizeMismatch {
        /// Tensor name.
        name: String,
        /// `product(shape) x width` (`u64::MAX` when that overflows).
        expected: u64,
        /// Stored size.
        observed: u64,
    },
    /// Rung 4: a tensor's data range lies outside the file.
    DataOutOfBounds {
        /// Tensor name.
        name: String,
    },
    /// Rung 4 (or pack): the task blob was refused (D-05).
    Task(TaskError),
    /// Rung 4: the task blob's labels differ from `manifest.labels`.
    LabelsDisagreeWithTask {
        /// Manifest labels.
        manifest: Vec<String>,
        /// Labels parsed from the task blob.
        task: Vec<String>,
    },
    /// Rung 4: `manifest.recipe_id` is not the recipe blob's sha256.
    RecipeIdMismatch {
        /// Manifest value.
        manifest: String,
        /// sha256 of the recipe blob.
        blob: String,
    },
    /// Rung 4: `manifest.agent` disagrees with the agent config blob.
    AgentMismatch,
    /// Rung 4: a manifest field disagrees with the sha-bound source it summarises (an
    /// embedded blob, a value derived from one, or the task / agent config the model runs
    /// with) — decide-apr-v1 `manifest.bindings`. `field` is the dotted manifest field.
    ManifestDisagreesWithBlob {
        /// The dotted manifest field (`base`, `variant`, `calibration.t_applied`, ...).
        field: &'static str,
    },
    /// Rung 5 (or pack): a weight holds a NaN or an infinity.
    NonFiniteWeight {
        /// Tensor name.
        name: String,
    },
    /// Rung 6 (or pack): the model could not be rebuilt.
    Rebuild(DecideError),
    /// Pack: a built probe row exceeds [`PROBE_MAX_ROW_TOKENS`].
    ProbeRowOverBudget {
        /// Probe index.
        index: usize,
        /// Built-row tokens.
        tokens: usize,
        /// The cap.
        cap: usize,
    },
    /// Pack: the Rust probe disagrees with `probes.json` (`component` names what).
    ProbeDisagreesWithOracle {
        /// Probe index.
        index: usize,
        /// `count`, `input_index`, `tokens`, `label`, `hex` or `probabilities`.
        component: &'static str,
    },
    /// Rung 7: the model could not classify a probe during load-time replay (V9-b: a replay
    /// failure, not a rebuild one — the model was already built at rung 6).
    ProbeReplay {
        /// The classify error.
        reason: String,
    },
    /// Rung 7: the replayed probe disagrees with the stored expectation.
    ProbeMismatch {
        /// Probe index.
        index: usize,
        /// `count`, `input_index`, `tokens`, `label`, `hex` or `probabilities`.
        component: &'static str,
    },
    /// Pack: the container writer refused.
    Write {
        /// The container error.
        reason: String,
    },
}

impl ArtifactError {
    /// The decide-apr-v1 load rung (or `pack`) this refusal belongs to.
    #[must_use]
    pub fn rung(&self) -> &'static str {
        match self {
            Self::ArtifactTooLarge { .. } | Self::Read { .. } => "1 bounded_read",
            Self::Header { .. }
            | Self::HeaderChecksum
            | Self::ColumnMajor
            | Self::TensorCountOverCap { .. }
            | Self::IndexExtentTooSmall { .. }
            | Self::IndexPastEnd { .. }
            | Self::MetadataOverCap { .. } => "2 header_and_index_extent",
            Self::Container { .. }
            | Self::WrongModelType { .. }
            | Self::CustomKeys { .. }
            | Self::ManifestNotString
            | Self::ManifestParse { .. }
            | Self::WrongSchema { .. }
            | Self::SchemaVersion { .. }
            | Self::UnknownMethod { .. } => "3 manifest",
            Self::ManifestBlobs { .. }
            | Self::BlobHashMismatch { .. }
            | Self::ConfigBlob { .. }
            | Self::DuplicateTensor { .. }
            | Self::MissingTensor { .. }
            | Self::UnexpectedTensor { .. }
            | Self::DtypeMismatch { .. }
            | Self::SizeMismatch { .. }
            | Self::DataOutOfBounds { .. }
            | Self::Task(_)
            | Self::LabelsDisagreeWithTask { .. }
            | Self::RecipeIdMismatch { .. }
            | Self::AgentMismatch
            | Self::ManifestDisagreesWithBlob { .. } => "4 structural",
            Self::NonFiniteWeight { .. } => "5 non_finite_scan",
            Self::Rebuild(_) => "6 rebuild",
            Self::ProbeReplay { .. } | Self::ProbeMismatch { .. } => "7 probe_replay",
            Self::ProbeRowOverBudget { .. }
            | Self::ProbeDisagreesWithOracle { .. }
            | Self::Write { .. } => "pack",
        }
    }
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "decide-apr-v1 rung {}: ", self.rung())?;
        match self {
            Self::ArtifactTooLarge {
                what,
                observed,
                cap,
            } => write!(f, "{what} {observed} bytes exceeds the {cap}-byte cap"),
            Self::Read { reason } => write!(f, "read failed: {reason}"),
            Self::Header { reason } => write!(f, "header refused: {reason}"),
            Self::HeaderChecksum => write!(f, "header CRC mismatch"),
            Self::ColumnMajor => write!(f, "column-major layout flag set (LAYOUT-002)"),
            Self::TensorCountOverCap { declared, cap } => {
                write!(f, "tensor_count {declared} exceeds {cap}")
            }
            Self::IndexExtentTooSmall { declared, extent } => write!(
                f,
                "tensor_count {declared} x {MIN_INDEX_ENTRY_BYTES} does not fit the {extent}-byte index extent"
            ),
            Self::IndexPastEnd {
                data_offset,
                file_len,
            } => write!(f, "data_offset {data_offset} is past the {file_len}-byte file"),
            Self::MetadataOverCap { declared, cap } => {
                write!(f, "metadata_size {declared} exceeds {cap}")
            }
            Self::Container { reason } => write!(f, "container refused: {reason}"),
            Self::WrongModelType { observed } => {
                write!(f, "model_type {observed:?}, expected {MODEL_TYPE_TAG:?}")
            }
            Self::CustomKeys { observed } => write!(
                f,
                "custom metadata keys {observed:?}, expected exactly [{CUSTOM_METADATA_KEY:?}]"
            ),
            Self::ManifestNotString => write!(f, "the manifest is not a JSON string"),
            Self::ManifestParse { reason } => write!(f, "manifest refused: {reason}"),
            Self::WrongSchema { observed } => {
                write!(f, "schema {observed:?}, expected {ARTIFACT_SCHEMA:?}")
            }
            Self::SchemaVersion { observed } => write!(
                f,
                "schema_version {observed}, expected {ARTIFACT_SCHEMA_VERSION}"
            ),
            Self::UnknownMethod { observed } => write!(f, "unknown method {observed:?}"),
            Self::ManifestBlobs { observed } => {
                write!(f, "manifest blobs {observed:?}, expected {BLOB_TENSORS:?}")
            }
            Self::BlobHashMismatch { blob } => {
                write!(f, "blob {blob} does not hash to the manifest value")
            }
            Self::ConfigBlob { blob, reason } => write!(f, "{blob} refused: {reason}"),
            Self::DuplicateTensor { name } => {
                write!(f, "the tensor index names {name} more than once")
            }
            Self::MissingTensor { name } => write!(f, "missing tensor {name}"),
            Self::UnexpectedTensor { name } => write!(f, "unexpected tensor {name}"),
            Self::DtypeMismatch {
                name,
                expected,
                observed,
            } => write!(f, "tensor {name} is {observed}, expected {expected}"),
            Self::SizeMismatch {
                name,
                expected,
                observed,
            } => write!(
                f,
                "tensor {name} holds {observed} bytes, its shape needs {expected}"
            ),
            Self::DataOutOfBounds { name } => {
                write!(f, "tensor {name}'s data lies outside the file")
            }
            Self::Task(e) => write!(f, "task blob: {e}"),
            Self::LabelsDisagreeWithTask { manifest, task } => write!(
                f,
                "manifest labels {manifest:?} differ from the task blob's {task:?}"
            ),
            Self::RecipeIdMismatch { manifest, blob } => write!(
                f,
                "recipe_id {manifest} is not the recipe blob's sha256 {blob}"
            ),
            Self::AgentMismatch => {
                write!(f, "manifest agent disagrees with the agent config blob")
            }
            Self::ManifestDisagreesWithBlob { field } => write!(
                f,
                "manifest {field} disagrees with the sha-bound source it summarises (manifest.bindings)"
            ),
            Self::NonFiniteWeight { name } => write!(f, "tensor {name} holds a non-finite value"),
            Self::Rebuild(e) => write!(f, "rebuild: {e}"),
            Self::ProbeRowOverBudget { index, tokens, cap } => {
                write!(f, "probe {index} builds a {tokens}-token row, cap {cap}")
            }
            Self::ProbeDisagreesWithOracle { index, component } => write!(
                f,
                "probe {index}: the Rust {component} disagrees with probes.json"
            ),
            Self::ProbeReplay { reason } => write!(f, "probe replay failed: {reason}"),
            Self::ProbeMismatch { index, component } => write!(
                f,
                "probe {index}: the replayed {component} disagrees with the stored expectation"
            ),
            Self::Write { reason } => write!(f, "container write failed: {reason}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

// ===========================================================================
// Shared structural rules (pack step 1-2 and load rungs 4-5 use the SAME code)
// ===========================================================================

/// Lowercase-hex sha256 of the whole artifact: the D-11 identity.
#[must_use]
pub fn artifact_sha256_hex(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

/// Artifact bytes hashed ONCE, by this crate: the digest a caller checks its pin against
/// BEFORE any parse, and the digest [`crate::Decider::load_hashed`] mints the identity from
/// instead of hashing the same bytes a second time (a ~0.85 GB software sha256 is ~3 s of a
/// Lambda cold start).
///
/// The fields are private and the only constructor hashes exactly the bytes it holds, so an
/// identity minted from it is still the ladder's own (decide-apr-v1 `private_mint`).
#[derive(Debug, Clone)]
pub struct HashedArtifact<'a> {
    bytes: &'a [u8],
    sha256: String,
}

impl<'a> HashedArtifact<'a> {
    /// Hash `bytes` — the whole file, the D-11 identity.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            sha256: artifact_sha256_hex(bytes),
        }
    }

    /// Lowercase-hex sha256 of the whole artifact.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The bytes that were hashed.
    #[must_use]
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Bytes per element for the dtypes this artifact carries; `None` for any other.
fn dtype_width(d: TensorDType) -> Option<u64> {
    matches!(d, TensorDType::F32 | TensorDType::F16 | TensorDType::U8)
        .then(|| d.bytes_per_element() as u64)
}

/// `observed <= cap`, or the rung-1 / pack refusal naming `what`.
fn over_cap(what: &'static str, observed: u64, cap: u64) -> Result<(), ArtifactError> {
    if observed > cap {
        return Err(ArtifactError::ArtifactTooLarge {
            what,
            observed,
            cap,
        });
    }
    Ok(())
}

/// `product(shape) x width`, exact: `Some(0)` whenever any dimension is 0 (even after
/// a dimension whose running product would overflow), otherwise checked arithmetic
/// with `None` when the size exceeds u64. `None` for a dtype this artifact does not
/// carry (KANI-DECIDE-APR-003's rule).
pub(crate) fn expected_bytes(shape: &[usize], dtype: TensorDType) -> Option<u64> {
    let width = dtype_width(dtype)?;
    if shape.contains(&0) {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(width, |acc, &d| acc.checked_mul(u64::try_from(d).ok()?))
}

fn check_size(
    name: &str,
    shape: &[usize],
    dtype: TensorDType,
    observed: u64,
) -> Result<(), ArtifactError> {
    match expected_bytes(shape, dtype) {
        Some(expected) if expected == observed => Ok(()),
        expected => Err(ArtifactError::SizeMismatch {
            name: name.to_string(),
            expected: expected.unwrap_or(u64::MAX),
            observed,
        }),
    }
}

/// Every weight the architecture derives, with its dtype: the encoder names from
/// core's ModernBERT contract (under `encoder.`), the head layers, `type_emb`, the
/// scorer, the F32 `temperature` buffer and the (unused) `act_head.` family.
pub(crate) fn expected_weights(
    encoder_config: &ModernBertConfig,
    agent: &AgentConfig,
) -> BTreeMap<String, TensorDType> {
    let mut out: BTreeMap<String, TensorDType> =
        expected_modernbert_tensor_names(encoder_config, "encoder.")
            .into_iter()
            .map(|n| (n, TensorDType::F16))
            .collect();
    const HEAD: [&str; 12] = [
        "norm1.weight",
        "norm1.bias",
        "self_attn.in_proj_weight",
        "self_attn.in_proj_bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "norm2.weight",
        "norm2.bias",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
    ];
    for i in 0..agent.head_layers {
        for leaf in HEAD {
            out.insert(format!("head.layers.{i}.{leaf}"), TensorDType::F16);
        }
    }
    for n in [
        "type_emb.weight",
        "scorer.0.weight",
        "scorer.0.bias",
        "scorer.1.weight",
        "scorer.1.bias",
        "scorer.3.weight",
        "scorer.3.bias",
        "act_head.0.weight",
        "act_head.0.bias",
        "act_head.2.weight",
        "act_head.2.bias",
    ] {
        out.insert(n.to_string(), TensorDType::F16);
    }
    out.insert("temperature".to_string(), TensorDType::F32);
    out
}

/// The first name that repeats in `names`, in iteration order: a set walk, so it does not
/// rely on the container keeping its index sorted (an adjacent-pair check would). `None`
/// when every name is unique. Its memory is bounded by rung 2's `max_tensor_count`.
pub(crate) fn first_repeated_tensor_name<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
    let mut seen = HashSet::new();
    names.into_iter().find(|n| !seen.insert(*n))
}

/// The set rule: `observed` names equal `expected` names exactly. Missing is reported
/// before unexpected, each naming the first offender in name order.
fn check_name_set<'a>(
    expected: &BTreeMap<String, TensorDType>,
    observed: impl Iterator<Item = &'a str> + Clone,
) -> Result<(), ArtifactError> {
    if let Some(name) = expected
        .keys()
        .find(|n| !observed.clone().any(|o| o == n.as_str()))
    {
        return Err(ArtifactError::MissingTensor { name: name.clone() });
    }
    let mut extra: Vec<&str> = observed.filter(|o| !expected.contains_key(*o)).collect();
    extra.sort_unstable();
    if let Some(name) = extra.first() {
        return Err(ArtifactError::UnexpectedTensor {
            name: (*name).to_string(),
        });
    }
    Ok(())
}

fn check_dtype(
    name: &str,
    expected: TensorDType,
    observed: TensorDType,
) -> Result<(), ArtifactError> {
    if expected == observed {
        Ok(())
    } else {
        Err(ArtifactError::DtypeMismatch {
            name: name.to_string(),
            expected: format!("{expected:?}"),
            observed: format!("{observed:?}"),
        })
    }
}

/// True when every element of `bytes` (little-endian `dtype`) is finite, scanned on
/// the raw bit patterns: an F16 is non-finite iff its exponent is all ones
/// (`0x7C00`), an F32 iff `0x7F80_0000`. No widening, no allocation.
pub(crate) fn all_finite(dtype: TensorDType, bytes: &[u8]) -> bool {
    match dtype {
        TensorDType::F16 => bytes
            .chunks_exact(2)
            .all(|c| u16::from_le_bytes([c[0], c[1]]) & 0x7C00 != 0x7C00),
        TensorDType::F32 => bytes
            .chunks_exact(4)
            .all(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) & 0x7F80_0000 != 0x7F80_0000),
        _ => false,
    }
}

fn parse_encoder_config(bytes: &[u8]) -> Result<ModernBertConfig, ArtifactError> {
    ModernBertConfig::from_json_bytes(bytes).map_err(|e| ArtifactError::ConfigBlob {
        blob: ENCODER_CONFIG_BLOB,
        reason: e.to_string(),
    })
}

fn parse_agent_config(bytes: &[u8]) -> Result<AgentConfig, ArtifactError> {
    AgentConfig::from_json_bytes(bytes).map_err(|e| ArtifactError::ConfigBlob {
        blob: AGENT_CONFIG_BLOB,
        reason: e.to_string(),
    })
}

/// NaN-visible `delta <= bound`: a NaN on either side never passes. The crate's one
/// definition (the parity tests use it through `test_support`).
pub(crate) fn within(delta: f64, bound: f64) -> bool {
    matches!(
        delta.partial_cmp(&bound),
        Some(Ordering::Less | Ordering::Equal)
    )
}

/// An 8-hex-digit big-endian f32 bit pattern (the probes.json convention).
fn f32_from_hex(h: &str) -> Option<f32> {
    if h.len() != 8 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(h, 16).ok().map(f32::from_bits)
}

/// One Rust probe result.
struct RustProbe {
    tokens: usize,
    label: String,
    probabilities: Vec<f32>,
}

/// Run the contract's synthetic probe task over [`PROBE_INPUTS`] with `laya`'s weights.
///
/// Every probe row is BUILT and checked against `cap` (`probe_max_row_tokens`) before any
/// forward pass: the replay is the only forward a load runs, and its cost must be bounded by
/// the contract whatever the artifact's `max_len` or tokenizer say. `over_budget(index,
/// tokens, cap)` names that refusal; `classify_failed` names the refusal a builder or classify
/// error becomes: at pack the model is built and probed as one step
/// ([`ArtifactError::Rebuild`]); at load it is [`replay_failure`].
fn run_probes(
    laya: &Laya,
    cap: usize,
    over_budget: fn(usize, usize, usize) -> ArtifactError,
    classify_failed: fn(DecideError) -> ArtifactError,
) -> Result<Vec<RustProbe>, ArtifactError> {
    let task = Task::from_slice(PROBE_TASK.as_bytes()).map_err(ArtifactError::Task)?;
    let builder = laya.builder();
    let prefix = builder
        .prefix(
            QType::Choice.name(),
            task.instructions(),
            &task.render_options(),
        )
        .map_err(|e| classify_failed(e.into()))?;
    for (index, text) in PROBE_INPUTS.iter().enumerate() {
        let row = builder
            .finish(&prefix, text)
            .map_err(|e| classify_failed(e.into()))?;
        if row.tokens > cap {
            return Err(over_budget(index, row.tokens, cap));
        }
    }
    let texts: Vec<String> = PROBE_INPUTS.iter().map(|s| (*s).to_string()).collect();
    let decisions = laya
        .classify_for_task(&task, &texts)
        .map_err(classify_failed)?;
    let labels = task.labels();
    let probes = decisions
        .into_iter()
        .map(|d| RustProbe {
            tokens: d.tokens,
            label: labels
                .get(d.label_index)
                .map_or_else(String::new, |l| (*l).to_string()),
            probabilities: d.probabilities,
        })
        .collect();
    Ok(probes)
}

/// Load: a classify failure while replaying the probes is a rung-7 refusal (V9-b). The model
/// was already rebuilt at rung 6; the replay is its first forward pass.
fn replay_failure(e: DecideError) -> ArtifactError {
    ArtifactError::ProbeReplay {
        reason: e.to_string(),
    }
}

/// Compare Rust probes with stored expectations; `err(index, component)` builds the
/// refusal (pack: `ProbeDisagreesWithOracle`; load: `ProbeMismatch`). The row budget
/// is checked first, before any comparison ([`run_probes`] has already refused an
/// over-budget row before the forward; this re-check is on the scored rows).
fn compare_probes(
    rust: &[RustProbe],
    stored: &[ProbeRecord],
    cap: usize,
    err: fn(usize, &'static str) -> ArtifactError,
) -> Result<(), ArtifactError> {
    if let Some(index) = rust.iter().position(|r| r.tokens > cap) {
        return Err(err(index, "tokens"));
    }
    if stored.len() != rust.len() {
        return Err(err(stored.len().min(rust.len()), "count"));
    }
    for (index, (r, s)) in rust.iter().zip(stored).enumerate() {
        if s.input_index != index {
            return Err(err(index, "input_index"));
        }
        if s.tokens != r.tokens {
            return Err(err(index, "tokens"));
        }
        if s.label != r.label {
            return Err(err(index, "label"));
        }
        if s.probabilities_f32_hex.len() != r.probabilities.len() {
            return Err(err(index, "probabilities"));
        }
        for (h, &p) in s.probabilities_f32_hex.iter().zip(&r.probabilities) {
            let want = f32_from_hex(h).ok_or_else(|| err(index, "hex"))?;
            let delta = (f64::from(p) - f64::from(want)).abs();
            if !within(delta, PROBE_PROBABILITIES_ABS) {
                return Err(err(index, "probabilities"));
            }
        }
    }
    Ok(())
}

// ===========================================================================
// Write
// ===========================================================================

/// Write `inputs` as `decide-apr-v1` bytes.
///
/// # Order is load-bearing (steps, not rungs)
///
/// 1. tensor-set bijection with the architecture-derived set, dtypes and byte sizes;
/// 2. every weight finite;
/// 3. task.json parses (D-05); its labels become `manifest.labels`;
/// 4. the Laya method is built in memory from these inputs and runs the contract's two
///    probes; a probe row over [`PROBE_MAX_ROW_TOKENS`] is refused, and each Rust
///    probe must agree with `probes.json` within [`PROBE_PROBABILITIES_ABS`];
/// 5. the ONE manifest, serialized from typed structs, storing the PYTHON probe values;
/// 6. only now the container: `model_type` `decide`, exactly one custom key, every
///    weight as its raw bytes, each blob as a U8 tensor, no timestamp.
///
/// Nothing is returned until every step has passed: there is no partial artifact.
///
/// # Errors
///
/// An [`ArtifactError`] naming the tensor, blob or probe that failed.
#[provable_contracts_macros::contract("decide-apr-v1", equation = "determinism")]
pub fn write_decide_apr(inputs: &PackInputs) -> Result<Vec<u8>, ArtifactError> {
    write_decide_apr_within(inputs, &ArtifactLimits::CONTRACTED)
}

/// [`write_decide_apr`] at caller-chosen bounds (module-private; only tests shrink them).
pub(crate) fn write_decide_apr_within(
    inputs: &PackInputs,
    limits: &ArtifactLimits,
) -> Result<Vec<u8>, ArtifactError> {
    let encoder_config = parse_encoder_config(&inputs.encoder_config)?;
    let agent = parse_agent_config(&inputs.agent_config)?;

    // (1) THE TENSOR SET, derived from the configs — the same rule rung 4 applies.
    let expected = expected_weights(&encoder_config, &agent);
    check_name_set(&expected, inputs.tensors.iter().map(|t| t.name.as_str()))?;
    for t in &inputs.tensors {
        if let Some(&want) = expected.get(&t.name) {
            check_dtype(&t.name, want, t.dtype)?;
        }
        check_size(&t.name, &t.shape, t.dtype, t.bytes.len() as u64)?;
    }

    // (2) EVERY WEIGHT FINITE.
    if let Some(t) = inputs
        .tensors
        .iter()
        .find(|t| !all_finite(t.dtype, &t.bytes))
    {
        return Err(ArtifactError::NonFiniteWeight {
            name: t.name.clone(),
        });
    }

    // (3) THE TASK (D-05): document order is the label index.
    let task = Task::from_slice(&inputs.task_json).map_err(ArtifactError::Task)?;
    let labels = task.owned_labels();

    // (4) PROBES: the Rust result is CHECKED against probes.json, never stored.
    check_pack_probes(inputs, task, limits)?;

    // (5) THE ONE MANIFEST.
    let manifest = build_manifest(inputs, &agent, labels);
    let manifest_json = serde_json::to_string(&manifest).map_err(|e| ArtifactError::Write {
        reason: e.to_string(),
    })?;

    // (6) THE CONTAINER. Nothing was written before this point.
    let bytes = write_container(inputs, manifest_json)?;
    over_cap("packed", bytes.len() as u64, limits.max_artifact_bytes)?;
    Ok(bytes)
}

/// Pack step 4: build Laya over a weights-only in-memory `.apr` of `inputs`, run the
/// probe task, refuse an over-budget row, and compare with the Python expectations.
fn check_pack_probes(
    inputs: &PackInputs,
    task: Task,
    limits: &ArtifactLimits,
) -> Result<(), ArtifactError> {
    let weights = {
        let mut w = AprV2Writer::new(AprV2Metadata::default());
        add_weights(&mut w, inputs);
        w.write().map_err(|e| ArtifactError::Write {
            reason: e.to_string(),
        })?
    };
    let reader = AprV2ReaderRef::from_bytes(&weights).map_err(|e| ArtifactError::Container {
        reason: e.to_string(),
    })?;
    let laya = Laya::from_parts(
        &reader,
        "",
        &inputs.encoder_config,
        &inputs.agent_config,
        &inputs.tokenizer,
        task,
    )
    .map_err(|e| ArtifactError::Rebuild(e.into()))?;
    let rust = run_probes(
        &laya,
        limits.probe_max_row_tokens,
        |index, tokens, cap| ArtifactError::ProbeRowOverBudget { index, tokens, cap },
        ArtifactError::Rebuild,
    )?;
    compare_probes(
        &rust,
        &inputs.probes,
        limits.probe_max_row_tokens,
        |index, component| ArtifactError::ProbeDisagreesWithOracle { index, component },
    )
}

/// Every checkpoint weight as its raw bytes, in `inputs` order (the probe container
/// and the artifact itself write the SAME tensors the same way).
fn add_weights(w: &mut AprV2Writer, inputs: &PackInputs) {
    for CheckpointTensor {
        name,
        dtype,
        shape,
        bytes,
    } in &inputs.tensors
    {
        w.add_tensor(name.clone(), *dtype, shape.clone(), bytes.clone());
    }
}

fn blob_bytes(inputs: &PackInputs) -> [(&'static str, &[u8]); 6] {
    [
        (TOKENIZER_BLOB, inputs.tokenizer.as_slice()),
        (TASK_BLOB, inputs.task_json.as_slice()),
        (ENCODER_CONFIG_BLOB, inputs.encoder_config.as_slice()),
        (AGENT_CONFIG_BLOB, inputs.agent_config.as_slice()),
        (RECIPE_BLOB, inputs.recipe_json.as_slice()),
        (GATE_REPORT_BLOB, inputs.gate_report_json.as_slice()),
    ]
}

fn build_manifest(inputs: &PackInputs, agent: &AgentConfig, labels: Vec<String>) -> Manifest {
    let report = &inputs.gate_report;
    Manifest {
        schema: ARTIFACT_SCHEMA.to_string(),
        schema_version: ARTIFACT_SCHEMA_VERSION,
        method: METHOD_LAYA.to_string(),
        variant: inputs.recipe.variant.clone(),
        base: inputs.recipe.base.clone(),
        labels,
        agent: AgentDecl::from(agent),
        blobs: blob_bytes(inputs)
            .iter()
            .map(|(name, bytes)| BlobHash {
                name: (*name).to_string(),
                sha256: sha256_hex(bytes),
            })
            .collect(),
        recipe_id: sha256_hex(&inputs.recipe_json),
        calibration: CalibrationDecl {
            bucket: report.calibration.bucket.clone(),
            t_fitted: report.calibration.t_fitted,
            t_applied: report.calibration.t_applied,
            clamp_hit: report.calibration.clamp_hit,
            slice_ids_sha256: report.calibration.slice_ids_sha256.clone(),
        },
        gate: GateSummary {
            pass: report.pass,
            margin: report.margin,
            ece_post: report.fine_tuned.ece_post,
            report_sha256: sha256_hex(&inputs.gate_report_json),
        },
        inputs_sha256: inputs.inputs_sha256.clone(),
        device_used: report.device_used.clone(),
        probes: inputs.probes.clone(),
    }
}

fn write_container(inputs: &PackInputs, manifest_json: String) -> Result<Vec<u8>, ArtifactError> {
    let mut metadata = AprV2Metadata::new(MODEL_TYPE_TAG);
    metadata.custom.insert(
        CUSTOM_METADATA_KEY.to_string(),
        serde_json::Value::String(manifest_json),
    );
    let mut w = AprV2Writer::new(metadata);
    add_weights(&mut w, inputs);
    for (name, bytes) in blob_bytes(inputs) {
        w.add_tensor(name, TensorDType::U8, vec![bytes.len()], bytes.to_vec());
    }
    w.write().map_err(|e| ArtifactError::Write {
        reason: e.to_string(),
    })
}

// ===========================================================================
// Rung 1: the bounded read
// ===========================================================================

/// Read an artifact from `reader`, bounded (rung 1).
///
/// (a) A `declared_len` over the cap is refused WITHOUT TOUCHING the reader, so a
/// caller passing `fs::metadata(path)?.len()` never reads a hostile file. (b) The read
/// is capped at `cap + 1` anyway, because metadata can lie; the `+ 1` distinguishes a
/// legal artifact of exactly the cap from a longer stream cut at the cap.
///
/// # Errors
///
/// [`ArtifactError::ArtifactTooLarge`] (`declared_length` or `stream`) or
/// [`ArtifactError::Read`].
#[provable_contracts_macros::contract("decide-apr-v1", equation = "size_cap")]
pub fn read_decide_apr_bytes_bounded<R: std::io::Read>(
    reader: R,
    declared_len: Option<u64>,
) -> Result<Vec<u8>, ArtifactError> {
    read_bounded_within(reader, declared_len, &ArtifactLimits::CONTRACTED)
}

pub(crate) fn read_bounded_within<R: std::io::Read>(
    reader: R,
    declared_len: Option<u64>,
    limits: &ArtifactLimits,
) -> Result<Vec<u8>, ArtifactError> {
    use std::io::Read as _;
    let cap = limits.max_artifact_bytes;
    if let Some(declared) = declared_len {
        over_cap("declared_length", declared, cap)?;
    }
    let reserve = usize::try_from(declared_len.unwrap_or(0).min(cap)).unwrap_or(0);
    let mut bytes = Vec::with_capacity(reserve);
    reader
        .take(cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| ArtifactError::Read {
            reason: e.to_string(),
        })?;
    over_cap("stream", bytes.len() as u64, cap)?;
    Ok(bytes)
}

// ===========================================================================
// Rung 2: header and index extent, before any index parser
// ===========================================================================

/// The rung-2 index-extent predicate (KANI-DECIDE-APR-001's property), pure so it can
/// be checked exhaustively: `tensor_count <= MAX_TENSOR_COUNT`, `data_offset <=
/// file_len`, and, in checked arithmetic, `tensor_count x 20 <= data_offset -
/// tensor_index_offset`.
///
/// # Errors
///
/// [`ArtifactError::TensorCountOverCap`], [`ArtifactError::IndexPastEnd`] or
/// [`ArtifactError::IndexExtentTooSmall`].
#[provable_contracts_macros::contract("decide-apr-v1", equation = "bounded_read_and_index_extent")]
pub fn check_index_extent(
    tensor_count: u32,
    tensor_index_offset: u64,
    data_offset: u64,
    file_len: u64,
) -> Result<(), ArtifactError> {
    if tensor_count > MAX_TENSOR_COUNT {
        return Err(ArtifactError::TensorCountOverCap {
            declared: tensor_count,
            cap: MAX_TENSOR_COUNT,
        });
    }
    if data_offset > file_len {
        return Err(ArtifactError::IndexPastEnd {
            data_offset,
            file_len,
        });
    }
    let extent = data_offset.checked_sub(tensor_index_offset);
    let needed = u64::from(tensor_count).checked_mul(MIN_INDEX_ENTRY_BYTES);
    match (extent, needed) {
        (Some(extent), Some(needed)) if needed <= extent => Ok(()),
        (extent, _) => Err(ArtifactError::IndexExtentTooSmall {
            declared: tensor_count,
            extent: extent.unwrap_or(0),
        }),
    }
}

/// Rung 2: parse the 64-byte header ourselves and bound the index BEFORE
/// [`AprV2ReaderRef::from_bytes`] may allocate for it.
fn rung2_header(bytes: &[u8]) -> Result<(), ArtifactError> {
    if bytes.len() < HEADER_SIZE_V2 {
        return Err(ArtifactError::Header {
            reason: format!(
                "{} bytes is shorter than the {HEADER_SIZE_V2}-byte header",
                bytes.len()
            ),
        });
    }
    let header = AprV2Header::from_bytes(bytes).map_err(|e| ArtifactError::Header {
        reason: e.to_string(),
    })?;
    if !header.verify_checksum() {
        return Err(ArtifactError::HeaderChecksum);
    }
    // V9-d: the container parses any version with v2's layout; only v2 is this schema's.
    if header.version != VERSION_V2 {
        return Err(ArtifactError::Header {
            reason: format!(
                "version {}.{}, expected APR v2 {}.{}",
                header.version.0, header.version.1, VERSION_V2.0, VERSION_V2.1
            ),
        });
    }
    if !header.flags.is_layout_valid() {
        return Err(ArtifactError::ColumnMajor);
    }
    // The reader parses the whole declared metadata section into a JSON tree; bound it first.
    if header.metadata_size > MAX_METADATA_BYTES {
        return Err(ArtifactError::MetadataOverCap {
            declared: header.metadata_size,
            cap: MAX_METADATA_BYTES,
        });
    }
    check_index_extent(
        header.tensor_count,
        header.tensor_index_offset,
        header.data_offset,
        bytes.len() as u64,
    )
}

// ===========================================================================
// Rung 3: the manifest
// ===========================================================================

fn rung3_manifest(reader: &AprV2ReaderRef<'_>) -> Result<Manifest, ArtifactError> {
    let metadata = reader.metadata();
    if metadata.model_type != MODEL_TYPE_TAG {
        return Err(ArtifactError::WrongModelType {
            observed: metadata.model_type.clone(),
        });
    }
    let value = match metadata.custom.get(CUSTOM_METADATA_KEY) {
        Some(v) if metadata.custom.len() == 1 => v,
        _ => {
            let mut observed: Vec<String> = metadata.custom.keys().cloned().collect();
            observed.sort_unstable();
            return Err(ArtifactError::CustomKeys { observed });
        }
    };
    let text = value.as_str().ok_or(ArtifactError::ManifestNotString)?;
    let manifest: Manifest =
        serde_json::from_str(text).map_err(|e| ArtifactError::ManifestParse {
            reason: e.to_string(),
        })?;
    if manifest.schema != ARTIFACT_SCHEMA {
        return Err(ArtifactError::WrongSchema {
            observed: manifest.schema,
        });
    }
    if manifest.schema_version != ARTIFACT_SCHEMA_VERSION {
        return Err(ArtifactError::SchemaVersion {
            observed: manifest.schema_version,
        });
    }
    if manifest.method != METHOD_LAYA {
        return Err(ArtifactError::UnknownMethod {
            observed: manifest.method,
        });
    }
    Ok(manifest)
}

/// Rungs 1-3 over bytes already in memory.
fn open_within<'a>(
    bytes: &'a [u8],
    limits: &ArtifactLimits,
) -> Result<(AprV2ReaderRef<'a>, Manifest), ArtifactError> {
    over_cap("input_bytes", bytes.len() as u64, limits.max_artifact_bytes)?;
    rung2_header(bytes)?;
    let reader = AprV2ReaderRef::from_bytes(bytes).map_err(|e| ArtifactError::Container {
        reason: e.to_string(),
    })?;
    let manifest = rung3_manifest(&reader)?;
    Ok((reader, manifest))
}

/// The manifest of an artifact, through rungs 1-4 (identity for 08-09's `inspect`).
///
/// Rung 4 binds every manifest leaf to its sha-bound source (decide-apr-v1
/// `manifest.bindings`, plan 08-19), so the identity returned is the one the embedded blobs
/// say, never a free-standing claim (IN-02, A2-3). The weights are not scanned (rung 5), the
/// model is not rebuilt (rung 6) and the probes are not replayed (rung 7): this is IDENTITY,
/// NOT an eligibility check. decide-apr-v1 `deploy_eligibility` is `pack_laya verify` on the
/// exact file.
///
/// # Errors
///
/// A rung 1-4 [`ArtifactError`].
pub fn inspect_manifest(bytes: &[u8]) -> Result<Manifest, ArtifactError> {
    let (reader, manifest) = open_within(bytes, &ArtifactLimits::CONTRACTED)?;
    rung4_structure(&reader, &manifest)?;
    Ok(manifest)
}

// ===========================================================================
// Rung 4: structure
// ===========================================================================

/// The verified blobs rung 4 hands on, with the task it already parsed.
struct Blobs<'a> {
    tokenizer: &'a [u8],
    task: Task,
    encoder_config: &'a [u8],
    agent_config: &'a [u8],
}

fn rung4_structure<'r>(
    reader: &'r AprV2ReaderRef<'_>,
    manifest: &Manifest,
) -> Result<Blobs<'r>, ArtifactError> {
    // (0) Every index name is unique (WR-01, decide side), BEFORE any lookup by name: (a),
    //     (c) and rung 5 resolve a name to its FIRST entry, so a repeated name would hide its
    //     second entry from every check. The apr-format reader refuses a repeat at rung 3
    //     (plan 08-20); this walk does not depend on that, or on the index's sort order.
    if let Some(name) =
        first_repeated_tensor_name(reader.tensor_index().iter().map(|e| e.name.as_str()))
    {
        return Err(ArtifactError::DuplicateTensor {
            name: name.to_string(),
        });
    }
    // (a) Blobs first: present and hashing to the manifest — before any config they
    //     carry is trusted to derive the weight set. Their dtype / size / bounds are
    //     checked with every other entry in (c); a hash over the in-bounds bytes is
    //     what makes the configs trustworthy here.
    let names: Vec<&str> = manifest.blobs.iter().map(|b| b.name.as_str()).collect();
    if names != BLOB_TENSORS {
        return Err(ArtifactError::ManifestBlobs {
            observed: names.iter().map(|n| (*n).to_string()).collect(),
        });
    }
    // In BLOB_TENSORS order (just checked equal to the manifest's). Each blob is hashed
    // exactly once, here; the later bindings reuse these digests.
    let mut data: [&[u8]; 6] = [&[]; 6];
    let mut sha: [String; 6] = Default::default();
    for ((slot, digest), b) in data.iter_mut().zip(sha.iter_mut()).zip(&manifest.blobs) {
        if reader.get_tensor(&b.name).is_none() {
            return Err(ArtifactError::MissingTensor {
                name: b.name.clone(),
            });
        }
        *slot = reader
            .get_tensor_data(&b.name)
            .ok_or_else(|| ArtifactError::DataOutOfBounds {
                name: b.name.clone(),
            })?;
        *digest = sha256_hex(slot);
        if *digest != b.sha256 {
            return Err(ArtifactError::BlobHashMismatch {
                blob: b.name.clone(),
            });
        }
    }
    let [tokenizer, task_bytes, encoder_config_bytes, agent_config_bytes, recipe, gate_report] =
        data;
    let [tokenizer_sha, task_sha, _, _, recipe_sha, gate_report_sha] = sha;
    if recipe_sha != manifest.recipe_id {
        return Err(ArtifactError::RecipeIdMismatch {
            manifest: manifest.recipe_id.clone(),
            blob: recipe_sha,
        });
    }

    // (b) The weight set, derived from the verified configs.
    let encoder_config = parse_encoder_config(encoder_config_bytes)?;
    let agent = parse_agent_config(agent_config_bytes)?;
    if manifest.agent != AgentDecl::from(&agent) {
        return Err(ArtifactError::AgentMismatch);
    }
    let mut expected = expected_weights(&encoder_config, &agent);
    for name in BLOB_TENSORS {
        expected.insert(name.to_string(), TensorDType::U8);
    }
    let index = reader.tensor_names();
    check_name_set(&expected, index.iter().copied())?;

    // (c) Every entry: dtype, size == product(shape) x width, data inside the file.
    for name in &index {
        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| ArtifactError::MissingTensor {
                name: (*name).to_string(),
            })?;
        if let Some(&want) = expected.get(*name) {
            check_dtype(name, want, entry.dtype)?;
        }
        check_size(name, &entry.shape, entry.dtype, entry.size)?;
        if reader.get_tensor_data(name).is_none() {
            return Err(ArtifactError::DataOutOfBounds {
                name: (*name).to_string(),
            });
        }
    }

    // (d) The task blob's labels ARE the manifest labels (D-05).
    let task = Task::from_slice(task_bytes).map_err(ArtifactError::Task)?;
    let task_labels = task.owned_labels();
    if task_labels != manifest.labels {
        return Err(ArtifactError::LabelsDisagreeWithTask {
            manifest: manifest.labels.clone(),
            task: task_labels,
        });
    }

    // (e) Every other manifest field equals the sha-bound source it summarises
    //     (decide-apr-v1 `manifest.bindings`).
    check_manifest_bindings(
        manifest,
        &BoundSources {
            recipe,
            gate_report,
            gate_report_sha: &gate_report_sha,
            task_sha: &task_sha,
            tokenizer_sha: &tokenizer_sha,
            task: &task,
            agent: &agent,
        },
    )?;
    Ok(Blobs {
        tokenizer,
        task,
        encoder_config: encoder_config_bytes,
        agent_config: agent_config_bytes,
    })
}

/// The sha-bound sources rung 4 has verified by the time the manifest bindings run: the
/// raw recipe and gate-report blobs, the digests (a) computed (never recomputed), and the
/// task and agent config the model will run with.
struct BoundSources<'a> {
    recipe: &'a [u8],
    gate_report: &'a [u8],
    gate_report_sha: &'a str,
    task_sha: &'a str,
    tokenizer_sha: &'a str,
    task: &'a Task,
    agent: &'a AgentConfig,
}

/// f64 equality on the bit pattern: what the report records is what the manifest says,
/// exactly; a NaN is never equal to anything.
fn same_f64(a: f64, b: f64) -> bool {
    !a.is_nan() && a.to_bits() == b.to_bits()
}

/// Rung 4 (e): every manifest field that is not already bound by rungs 3, 4 (a)-(d) or 7
/// equals the value of the sha-bound source decide-apr-v1 `manifest.bindings` names. The
/// first disagreement, in the order below, is refused naming the dotted field.
///
/// `ModelIdentity.base` is minted from `manifest.base`; this is what binds it.
fn check_manifest_bindings(m: &Manifest, src: &BoundSources<'_>) -> Result<(), ArtifactError> {
    let recipe: crate::pack::Recipe =
        serde_json::from_slice(src.recipe).map_err(|e| ArtifactError::ConfigBlob {
            blob: RECIPE_BLOB,
            reason: e.to_string(),
        })?;
    let report: crate::pack::GateReport =
        serde_json::from_slice(src.gate_report).map_err(|e| ArtifactError::ConfigBlob {
            blob: GATE_REPORT_BLOB,
            reason: e.to_string(),
        })?;
    let (inputs, r_in) = (&m.inputs_sha256, &report.inputs_sha256);
    let (cal, r_cal) = (&m.calibration, &report.calibration);
    // The task this model answers: `choice` (the only D-05 type) with K = its criteria.
    let k = src.task.criteria().len();
    let bucket = bucket_key(QType::Choice, k);
    let applied = applied_temperature_f64(src.agent, QType::Choice, k);
    // One entry per comparison, so each can be disabled on its own (plan 08-19 mutation
    // proof); the first disagreement in this order is the refusal.
    let checks: [(&'static str, bool); 23] = [
        ("variant", m.variant == recipe.variant),
        ("base", m.base == recipe.base),
        (
            "gate.report_sha256",
            m.gate.report_sha256 == src.gate_report_sha,
        ),
        ("inputs_sha256.task_json", inputs.task_json == src.task_sha),
        (
            "inputs_sha256.tokenizer_json",
            inputs.tokenizer_json == src.tokenizer_sha,
        ),
        (
            "inputs_sha256.task_json",
            inputs.task_json == r_in.task_json,
        ),
        (
            "inputs_sha256.train_jsonl",
            inputs.train_jsonl == r_in.train_jsonl,
        ),
        (
            "inputs_sha256.eval_jsonl",
            inputs.eval_jsonl == r_in.eval_jsonl,
        ),
        (
            "inputs_sha256.base_model",
            inputs.base_model == r_in.base_model,
        ),
        (
            "inputs_sha256.tokenizer_json",
            inputs.tokenizer_json == r_in.tokenizer_json,
        ),
        ("base.sha256", m.base.sha256 == r_in.base_model),
        ("recipe_id", m.recipe_id == report.recipe_id),
        ("gate.pass", m.gate.pass == report.pass),
        ("gate.margin", same_f64(m.gate.margin, report.margin)),
        (
            "gate.ece_post",
            same_f64(m.gate.ece_post, report.fine_tuned.ece_post),
        ),
        ("calibration.bucket", cal.bucket == r_cal.bucket),
        (
            "calibration.t_fitted",
            same_f64(cal.t_fitted, r_cal.t_fitted),
        ),
        (
            "calibration.t_applied",
            same_f64(cal.t_applied, r_cal.t_applied),
        ),
        ("calibration.clamp_hit", cal.clamp_hit == r_cal.clamp_hit),
        (
            "calibration.slice_ids_sha256",
            cal.slice_ids_sha256 == r_cal.slice_ids_sha256,
        ),
        ("calibration.bucket", cal.bucket == bucket),
        ("calibration.t_applied", same_f64(cal.t_applied, applied)),
        ("device_used", m.device_used == report.device_used),
    ];
    match checks.iter().find(|(_, agrees)| !agrees) {
        Some(&(field, _)) => Err(ArtifactError::ManifestDisagreesWithBlob { field }),
        None => Ok(()),
    }
}

// ===========================================================================
// Rung 5: non-finite scan
// ===========================================================================

fn rung5_finite(reader: &AprV2ReaderRef<'_>) -> Result<(), ArtifactError> {
    for entry in reader.tensor_index() {
        if entry.dtype == TensorDType::U8 {
            continue;
        }
        let data =
            reader
                .get_tensor_data(&entry.name)
                .ok_or_else(|| ArtifactError::DataOutOfBounds {
                    name: entry.name.clone(),
                })?;
        if !all_finite(entry.dtype, data) {
            return Err(ArtifactError::NonFiniteWeight {
                name: entry.name.clone(),
            });
        }
    }
    Ok(())
}

// ===========================================================================
// The ladder
// ===========================================================================

/// Rungs 1-8 over bytes already in memory: the ONLY constructor of [`Decider`].
#[provable_contracts_macros::contract("decide-apr-v1", equation = "private_mint")]
pub(crate) fn load_verified(bytes: &[u8]) -> Result<Decider, ArtifactError> {
    load_verified_within(bytes, &ArtifactLimits::CONTRACTED)
}

pub(crate) fn load_verified_within(
    bytes: &[u8],
    limits: &ArtifactLimits,
) -> Result<Decider, ArtifactError> {
    load_rungs(bytes, limits, || artifact_sha256_hex(bytes))
}

/// Rungs 1-8 over bytes this crate already hashed ([`HashedArtifact::new`]): the rung-8
/// identity is that digest, so the bytes are hashed once rather than twice.
pub(crate) fn load_verified_hashed(hashed: &HashedArtifact<'_>) -> Result<Decider, ArtifactError> {
    load_rungs(hashed.bytes, &ArtifactLimits::CONTRACTED, || {
        hashed.sha256.clone()
    })
}

/// The ladder itself; `whole_file_sha256` supplies the rung-8 identity (always the sha256
/// of exactly `bytes`, computed by this crate) and runs only once every rung has passed.
fn load_rungs(
    bytes: &[u8],
    limits: &ArtifactLimits,
    whole_file_sha256: impl FnOnce() -> String,
) -> Result<Decider, ArtifactError> {
    // Rungs 1-3.
    let (reader, manifest) = open_within(bytes, limits)?;
    // Rung 4.
    let blobs = rung4_structure(&reader, &manifest)?;
    // Rung 5.
    rung5_finite(&reader)?;
    // Rung 6: F16 widened to f32 by core's loader over the zero-copy reader.
    let laya = Laya::from_parts(
        &reader,
        "",
        blobs.encoder_config,
        blobs.agent_config,
        blobs.tokenizer,
        blobs.task,
    )
    .map_err(|e| ArtifactError::Rebuild(e.into()))?;
    // Rung 7: replay against the stored PYTHON values. Rows over probe_max_row_tokens are
    // refused before any forward; a classify failure here is rung 7's.
    let rust = run_probes(
        &laya,
        limits.probe_max_row_tokens,
        |index, _, _| ArtifactError::ProbeMismatch {
            index,
            component: "tokens",
        },
        replay_failure,
    )?;
    compare_probes(
        &rust,
        &manifest.probes,
        limits.probe_max_row_tokens,
        |index, component| ArtifactError::ProbeMismatch { index, component },
    )?;
    // Rung 8: mint.
    let identity = ModelIdentity {
        artifact_sha256: whole_file_sha256(),
        recipe_id: manifest.recipe_id.clone(),
        method: manifest.method.clone(),
        base: manifest.base.display(),
        base_decl: manifest.base.clone(),
    };
    let method: Box<dyn DecisionMethod> = Box::new(laya);
    Ok(Decider {
        method,
        identity,
        manifest,
    })
}

#[cfg(test)]
mod determinism;
#[cfg(test)]
mod ladder;
#[cfg(test)]
mod tests;
