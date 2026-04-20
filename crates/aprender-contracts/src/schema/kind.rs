//! Contract kinds — declare which validation rules apply to a YAML file.

use serde::{Deserialize, Serialize};

/// The kind of contract artifact. Determines which validation rules apply.
///
/// - `Kernel` (default): a mathematical kernel contract — the provability
///   invariant applies (must have `proof_obligations`, `falsification_tests`,
///   `kani_harnesses`).
/// - `Registry`: a data registry (lookup tables, enum definitions, config
///   bounds) — exempt from provability, validated for `metadata` + entries.
/// - `ModelFamily`: architecture metadata (`HuggingFace` family descriptors,
///   size variants, vendor) — exempt from provability, validated for
///   `metadata` fields. Custom top-level fields are preserved but not
///   enforced by the kernel schema.
/// - `ModelFamilyVariant`: a concrete size variant of a model family
///   (e.g. Llama 370M sovereign). Freezes hyperparameters (vocab, hidden
///   dim, layer count) and delta-dispatches invariants from the parent
///   family. Exempt from provability.
/// - `Tokenizer`: a concrete tokenizer contract — vocab bounds, required
///   special tokens, round-trip gate, normalization form. Exempt from
///   provability (gates are byte-exact tests, not Kani harnesses).
/// - `TrainingLoop`: a training-loop contract — loss schedule, optimizer
///   config, gradient-clipping policy, checkpoint cadence. Exempt from
///   provability; validated for `metadata` + schedule fields.
/// - `PretrainingCorpus`: a pretraining-corpus contract — dataset source,
///   license, total-bytes bound, shard layout. Exempt from provability.
/// - `Pattern`: a cross-cutting verification pattern (threading safety,
///   async safety, compute parity) that applies across multiple kernels.
///   Exempt from the kernel provability invariant but still validated for
///   metadata and any proof/falsification data present.
/// - `Schema`: a generic reference/schema document — exempt from provability,
///   validated only for `metadata.id`, `metadata.version`, `metadata.description`,
///   and `metadata.references`.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum ContractKind {
    #[default]
    Kernel,
    Registry,
    ModelFamily,
    ModelFamilyVariant,
    Tokenizer,
    TrainingLoop,
    PretrainingCorpus,
    Pattern,
    Schema,
}

impl std::fmt::Display for ContractKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Kernel => "kernel",
            Self::Registry => "registry",
            Self::ModelFamily => "model-family",
            Self::ModelFamilyVariant => "model-family-variant",
            Self::Tokenizer => "tokenizer",
            Self::TrainingLoop => "training-loop",
            Self::PretrainingCorpus => "pretraining-corpus",
            Self::Pattern => "pattern",
            Self::Schema => "schema",
        };
        write!(f, "{s}")
    }
}
