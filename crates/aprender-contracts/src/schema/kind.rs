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
/// - `TrainingPreconditionGate`: a hard precondition gate that must be
///   satisfied before training starts (e.g. Chinchilla compute-optimal
///   token/parameter ratio, GPU memory floor, dataset checksum). Exempt
///   from provability; validated for `metadata` + gate-formula fields.
/// - `CorpusAssembly`: a multi-source corpus assembly pipeline contract —
///   declares input shards, dedup strategy, license-merge policy, and
///   output manifest. Exempt from provability.
/// - `Pattern`: a cross-cutting verification pattern (threading safety,
///   async safety, compute parity) that applies across multiple kernels.
///   Exempt from the kernel provability invariant but still validated for
///   metadata and any proof/falsification data present.
/// - `Schema`: a generic reference/schema document — exempt from provability,
///   validated only for `metadata.id`, `metadata.version`, `metadata.description`,
///   and `metadata.references`.
/// - `CompetitiveParity`: the competitive-parity ledger — one dated, verdict-
///   bearing row per in-scope entry point. **Carries proof obligations and is
///   NOT exemptible by `metadata.registry: true`** (see [`ContractKind`] docs
///   on the registry escape hatch below).
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
    TrainingPreconditionGate,
    CorpusAssembly,
    Pattern,
    Schema,
    /// A head-to-head BEAT benchmark: a committed, CI-wired claim that aprender
    /// beats an incumbent (sklearn / PyTorch / Unsloth / Ollama) on a canonical
    /// task, with a pinned baseline that fails CI on regression. The measurement
    /// backbone for the four-pillar "replace AND beat" mission (PMAT-741).
    BeatBenchmark,
    /// The competitive-parity LEDGER: one row per in-scope entry point, each
    /// naming a competitor, a PINNED competitor version, both invocations, a
    /// verdict from the closed vocabulary
    /// (`BETTER`/`PARITY`/`WORSE`/`NOT_COMPARABLE`/`UNMEASURED`), the date it
    /// was measured, the date the measurement EXPIRES, an owner, and an
    /// evidence pointer.
    ///
    /// WHY IT IS ITS OWN KIND, AND WHY IT CANNOT BE REGISTRY-EXEMPTED.
    /// Before this kind existed there were exactly two roads and both led to a
    /// contract that proves nothing:
    ///
    ///   * pick `kernel` (the `#[default]`) and set `registry: true` — then
    ///     [`Contract::kind`] rewrites the kind to `Registry` and every
    ///     provability check is skipped. 481 contracts in `contracts/` already
    ///     sit on this road.
    ///   * pick any NON-kernel kind — then `requires_proofs()` is false, so the
    ///     obligation never applied in the first place.
    ///
    /// `CompetitiveParity` closes both: `requires_proofs()` returns true for it
    /// (`types.rs`), the rewrite in `Contract::kind` only fires when the
    /// declared kind is `Kernel` so it cannot reach here, and validator rule
    /// PARITY-000 REJECTS `metadata.registry: true` on this kind outright — so
    /// the escape hatch is not merely bypassed, it is a hard error.
    CompetitiveParity,
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
            Self::TrainingPreconditionGate => "training-precondition-gate",
            Self::CorpusAssembly => "corpus-assembly",
            Self::Pattern => "pattern",
            Self::Schema => "schema",
            Self::BeatBenchmark => "beat-benchmark",
            Self::CompetitiveParity => "competitive-parity",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod beat_benchmark_tests {
    use super::*;
    use crate::error::Severity;
    use crate::schema::{parse_contract_str, validate_contract};

    /// BeatBenchmark serde round-trips through its kebab-case "beat-benchmark".
    #[test]
    fn beat_benchmark_kind_round_trips() {
        assert_eq!(ContractKind::BeatBenchmark.to_string(), "beat-benchmark");
        let k: ContractKind = serde_yaml::from_str("beat-benchmark").unwrap();
        assert_eq!(k, ContractKind::BeatBenchmark);
    }

    /// The pilot beat contract (contracts/beat-sklearn-iris-v1.yaml) parses as a
    /// BeatBenchmark and validates with zero Error-severity violations (PMAT-741).
    #[test]
    fn pilot_beat_contract_validates() {
        let yaml = include_str!("../../../../contracts/beat-sklearn-iris-v1.yaml");
        let contract = parse_contract_str(yaml).expect("pilot beat contract parses");
        assert_eq!(contract.kind(), ContractKind::BeatBenchmark);
        let errors: Vec<_> = validate_contract(&contract)
            .into_iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "pilot beat contract has errors: {errors:?}"
        );
    }
}
