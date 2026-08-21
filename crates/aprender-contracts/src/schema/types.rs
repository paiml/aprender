use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub use super::composition::{ShapeContract, ShapeExpr};
pub use super::kind::ContractKind;
pub use super::parity::ParityLedger;

/// A complete YAML kernel contract.
///
/// This is the root type for the contract schema defined in
/// `docs/specifications/pv-spec.md` Section 3.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Contract {
    pub metadata: Metadata,
    /// Equations are optional — kaizen, pipeline, and registry contracts
    /// may define only `proof_obligations` without mathematical equations.
    ///
    /// Accepts both map form (`equations: { silu: { formula: ... } }`, the
    /// canonical schema) and sequence form (`equations: [{ id: silu,
    /// formula: ... }]`, used by several diagnostic/methodology contracts
    /// predating APR-MONO). The sequence form promotes each item's `id`
    /// field to the map key.
    #[serde(default, deserialize_with = "deserialize_equations")]
    pub equations: BTreeMap<String, Equation>,
    #[serde(default)]
    pub proof_obligations: Vec<ProofObligation>,
    #[serde(default)]
    pub kernel_structure: Option<KernelStructure>,
    #[serde(default)]
    pub simd_dispatch: BTreeMap<String, BTreeMap<String, String>>,
    #[serde(default)]
    pub enforcement: BTreeMap<String, EnforcementRule>,
    #[serde(default)]
    pub falsification_tests: Vec<FalsificationTest>,
    #[serde(default)]
    pub kani_harnesses: Vec<KaniHarness>,
    #[serde(default)]
    pub qa_gate: Option<QaGate>,
    /// Phase 7: Lean 4 verification summary across all obligations.
    #[serde(default)]
    pub verification_summary: Option<VerificationSummary>,
    /// Type-level invariants (Meyer's class invariants).
    #[serde(default)]
    pub type_invariants: Vec<TypeInvariant>,
    /// Coq verification specification.
    #[serde(default)]
    pub coq_spec: Option<CoqSpec>,
    /// BEAT-benchmark parameters (PMAT-741) — present on `metadata.kind:
    /// beat-benchmark` contracts; pins a machine-measured incumbent baseline so
    /// CI fails when aprender regresses below it on the incumbent's canonical task.
    #[serde(default)]
    pub beat: Option<Beat>,
    /// The competitive-parity LEDGER — present on `metadata.kind:
    /// competitive-parity` contracts. See [`crate::schema::parity`] for why the
    /// gate checks that a fresh dated verdict EXISTS rather than that it says
    /// `BETTER`.
    #[serde(default)]
    pub parity: Option<ParityLedger>,
}

/// Parameters of a head-to-head BEAT benchmark (`metadata.kind: beat-benchmark`,
/// PMAT-741): a falsifiable, CI-wired claim that aprender meets-or-beats an
/// incumbent (scikit-learn / PyTorch / Unsloth / Ollama·llama.cpp) on the
/// incumbent's own canonical task — the measurement backbone of the four-pillar
/// "replace AND beat" mission. Required-shape is enforced by
/// `validate_beat_benchmark` in the validator (BEAT-001..007).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Beat {
    /// Which pillar (1=sklearn, 2=PyTorch, 3=Unsloth, 4=Ollama/llama.cpp).
    #[serde(default)]
    pub pillar: Option<u8>,
    /// The incumbent being beaten — must name one of the four pillars.
    #[serde(default)]
    pub incumbent: String,
    /// How/when the baseline was pinned (free-form provenance).
    #[serde(default)]
    pub incumbent_pinned: Option<String>,
    /// The canonical task on which the beat is measured (apples-to-apples).
    #[serde(default)]
    pub canonical_task: Option<String>,
    /// The measured metric (e.g. `accuracy`, `wall_clock_ms`, `tokens_per_sec`, `mse`).
    #[serde(default)]
    pub metric: String,
    /// `higher_is_better` or `lower_is_better` — fixes the regression direction.
    #[serde(default)]
    pub direction: String,
    /// The incumbent's pinned baseline value.
    #[serde(default)]
    pub baseline_value: Option<f64>,
    /// Optional worst-case incumbent value (e.g. sklearn min over seeds).
    #[serde(default)]
    pub baseline_floor: Option<f64>,
    /// The threshold aprender must meet/beat; CI fails on regression past it.
    #[serde(default)]
    pub beat_threshold: Option<f64>,
    /// When the baseline was sourced (ISO date).
    #[serde(default)]
    pub baseline_sourced_date: Option<String>,
    /// `CPU` or `GPU` — the compute approved for this gate.
    #[serde(default)]
    pub approved_compute: Option<String>,
    /// The CI test/gate name that enforces this beat.
    #[serde(default)]
    pub ci_gate_name: String,
}

/// The outcome of evaluating a measured value against a [`Beat`]'s pinned
/// threshold — the falsifiable verdict at the heart of `apr beat-run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BeatOutcome {
    /// aprender meets-or-beats the incumbent: measured is on the winning side of
    /// `beat_threshold` per `direction`.
    Won,
    /// aprender regressed below the pinned threshold — CI must fail.
    Regressed,
}

impl Beat {
    /// Evaluate a measured value against this beat's pinned `beat_threshold`,
    /// honoring `direction`:
    /// - `higher_is_better` ⇒ `Won` iff `measured >= beat_threshold`
    /// - `lower_is_better`  ⇒ `Won` iff `measured <= beat_threshold`
    ///
    /// Returns `None` when the contract is too malformed to judge (no
    /// `beat_threshold`, a non-finite threshold/measurement, or an unknown
    /// `direction`) — the caller should treat that as a hard error, not a pass.
    /// The validator's BEAT-004/BEAT-005 rules reject such contracts up front,
    /// so a well-formed contract always yields `Some`.
    #[must_use]
    pub fn evaluate(&self, measured: f64) -> Option<BeatOutcome> {
        let threshold = self.beat_threshold?;
        if !threshold.is_finite() || !measured.is_finite() {
            return None;
        }
        match self.direction.trim() {
            "higher_is_better" => Some(if measured >= threshold {
                BeatOutcome::Won
            } else {
                BeatOutcome::Regressed
            }),
            "lower_is_better" => Some(if measured <= threshold {
                BeatOutcome::Won
            } else {
                BeatOutcome::Regressed
            }),
            _ => None,
        }
    }

    /// Convenience: `true` iff [`evaluate`](Self::evaluate) returns
    /// [`BeatOutcome::Won`]. A malformed contract (`None`) is **not** a win.
    #[must_use]
    pub fn is_won(&self, measured: f64) -> bool {
        self.evaluate(measured) == Some(BeatOutcome::Won)
    }
}

impl Contract {
    /// Back-compat: `metadata.registry: true` OR `metadata.kind: registry`.
    pub fn is_registry(&self) -> bool {
        self.metadata.registry || self.metadata.kind == ContractKind::Registry
    }

    /// The effective kind, honoring the legacy `registry: true` flag.
    pub fn kind(&self) -> ContractKind {
        if self.metadata.registry && self.metadata.kind == ContractKind::Kernel {
            ContractKind::Registry
        } else {
            self.metadata.kind
        }
    }

    /// True iff this contract must satisfy PROVABILITY-001.
    ///
    /// Two kinds carry it: [`ContractKind::Kernel`] and
    /// [`ContractKind::CompetitiveParity`].
    ///
    /// The second is the point. Before it existed, every road led to a contract
    /// that proves nothing: `kind()` below rewrites `kernel` + `registry: true`
    /// into `Registry` (481 contracts already sit on that road, all with
    /// `proven: 0`), and any OTHER kind escapes the exemption only by escaping
    /// the obligation — `requires_proofs()` was false for all of them. So the
    /// registry exemption keyed on Kernel and provability keyed on Kernel, and
    /// a contract could be neither provable nor exempt-with-a-reason.
    ///
    /// `CompetitiveParity` cannot take either road: the rewrite in [`Self::kind`]
    /// fires only when the DECLARED kind is `Kernel`, so `registry: true` cannot
    /// reach it, and validator rule PARITY-000 rejects `registry: true` on this
    /// kind outright so the intent cannot even be expressed.
    pub fn requires_proofs(&self) -> bool {
        matches!(
            self.kind(),
            ContractKind::Kernel | ContractKind::CompetitiveParity
        )
    }

    /// Enforce the provability invariant: kernel contracts MUST have
    /// `proof_obligations`, `falsification_tests`, and `kani_harnesses`.
    /// Returns a list of violations. Empty list = contract is valid.
    pub fn provability_violations(&self) -> Vec<String> {
        if !self.requires_proofs() {
            return vec![];
        }
        // A competitive-parity LEDGER carries obligations and falsifiers but no
        // Kani harnesses: there is no mathematical kernel to model-check, and
        // demanding one would push authors back onto `registry: true` — the
        // exact escape this kind exists to close. The obligation it DOES carry
        // is the one that matters: every claim must have a falsifier, so a
        // ledger cannot be a list of assertions.
        if self.kind() == ContractKind::CompetitiveParity {
            let mut violations = Vec::new();
            if self.proof_obligations.is_empty() {
                violations.push("Competitive-parity contract has no proof_obligations".into());
            }
            if self.falsification_tests.is_empty() {
                violations.push("Competitive-parity contract has no falsification_tests".into());
            }
            if self.falsification_tests.len() < self.proof_obligations.len() {
                violations.push(format!(
                    "falsification_tests ({}) < proof_obligations ({})",
                    self.falsification_tests.len(),
                    self.proof_obligations.len(),
                ));
            }
            return violations;
        }
        let mut violations = Vec::new();
        if self.proof_obligations.is_empty() {
            violations.push("Kernel contract has no proof_obligations".into());
        }
        if self.falsification_tests.is_empty() {
            violations.push("Kernel contract has no falsification_tests".into());
        }
        if self.kani_harnesses.is_empty() {
            violations.push("Kernel contract has no kani_harnesses".into());
        }
        if self.falsification_tests.len() < self.proof_obligations.len() {
            violations.push(format!(
                "falsification_tests ({}) < proof_obligations ({})",
                self.falsification_tests.len(),
                self.proof_obligations.len(),
            ));
        }
        violations
    }
}

/// Contract metadata block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub version: String,
    #[serde(default)]
    pub created: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    pub description: String,
    #[serde(default)]
    pub references: Vec<String>,
    /// Contract dependencies — other contracts this one composes.
    /// Values are contract stems (e.g. "silu-kernel-v1").
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Legacy registry flag — prefer `metadata.kind: registry` for new contracts.
    #[serde(default)]
    pub registry: bool,
    /// Contract kind. Defaults to [`ContractKind::Kernel`].
    #[serde(default)]
    pub kind: ContractKind,
    /// Per-contract enforcement level (Section 17, Gap 1).
    /// `basic` → schema valid; `standard` → + falsification + kani;
    /// `strict` → + all bindings implemented; `proven` → + Lean 4 proved.
    #[serde(default)]
    pub enforcement_level: Option<EnforcementLevel>,
    /// Once set, the contract cannot drop below this verification level
    /// without an explicit `pv unlock` (Section 17, Gap 5).
    #[serde(default)]
    pub locked_level: Option<String>,
}

/// Per-contract enforcement level (gradual enforcement, Section 17).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnforcementLevel {
    /// Schema valid, has equations.
    Basic,
    /// + falsification tests + Kani harnesses.
    Standard,
    /// + all bindings implemented + `#[contract]` annotations.
    Strict,
    /// + Lean 4 proved (no sorry).
    Proven,
}

/// A mathematical equation extracted from a paper (Phase 1 output).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Equation {
    /// Default-empty so diagnostic/methodology contracts that use prose
    /// requirements instead of a formula (e.g.
    /// `decode-hot-path-prefix-cache-diagnostic-v1`) still parse.
    #[serde(default)]
    pub formula: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub codomain: Option<String>,
    #[serde(default)]
    pub invariants: Vec<String>,
    /// Rust preconditions — compiled to `debug_assert!()` by `build.rs`.
    #[serde(default)]
    pub preconditions: Vec<String>,
    /// Rust postconditions — compiled to `debug_assert!()` by `build.rs`.
    #[serde(default)]
    pub postconditions: Vec<String>,
    /// Lean 4 theorem name that proves this equation correct.
    /// Example: "ProvableContracts.Theorems.Softmax.PartitionOfUnity"
    #[serde(default)]
    pub lean_theorem: Option<String>,
    /// IEEE 754 tolerance: codegen emits `>=` instead of `>` for boundaries (GH-67).
    #[serde(default)]
    pub float_tolerance: Option<f64>,
    /// Compositional verification: what this equation requires from upstream.
    /// References a guarantees block from another contract/equation.
    #[serde(default)]
    pub assumes: Option<ShapeContract>,
    /// Compositional verification: what this equation provides to downstream.
    /// Must be satisfiable by any downstream equation that assumes it.
    #[serde(default)]
    pub guarantees: Option<ShapeContract>,
}

/// A proof obligation derived from an equation.
///
/// 26 obligation types: 19 property types plus 7 Design by Contract
/// types (`precondition`, `postcondition`, `frame`, `loop_invariant`,
/// `loop_variant`, `old_state`, `subcontract`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofObligation {
    /// Obligation category. Defaults to `Invariant` for legacy contracts
    /// that predate the DbC split (e.g. `eval-harness-humaneval-v1`,
    /// `publish-manifest-v1`) which ship with just `property:`/`formal:`.
    #[serde(rename = "type", default)]
    pub obligation_type: ObligationType,
    /// Human-readable statement of what must hold. Alias `statement`
    /// accepted for legacy diagnostic contracts (e.g.
    /// `decode-hot-path-prefix-cache-diagnostic-v1`) whose POs predate
    /// the canonical `property:` naming.
    #[serde(default, alias = "statement")]
    pub property: String,
    /// Formal predicate (Rust/Lean syntax). Alias `verification` accepted
    /// for legacy contracts that ship a shell/pmat-query check instead of
    /// a formal predicate.
    #[serde(default, alias = "verification")]
    pub formal: Option<String>,
    #[serde(default)]
    pub tolerance: Option<f64>,
    #[serde(default)]
    pub applies_to: Option<AppliesTo>,
    /// Phase 7: Lean 4 theorem proving metadata.
    #[serde(default)]
    pub lean: Option<LeanProof>,
    /// Postcondition only: links to a precondition obligation ID.
    #[serde(default)]
    pub requires: Option<String>,
    /// Loop invariant/variant only: references a `kernel_structure.phases[]` name.
    #[serde(default)]
    pub applies_to_phase: Option<String>,
    /// Subcontract only: contract stem being refined (must be in `metadata.depends_on`).
    #[serde(default)]
    pub parent_contract: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ObligationType {
    #[default]
    Invariant,
    Equivalence,
    Bound,
    Monotonicity,
    Idempotency,
    Linearity,
    Symmetry,
    Associativity,
    Conservation,
    Ordering,
    Completeness,
    Soundness,
    Involution,
    Determinism,
    Roundtrip,
    #[serde(rename = "state_machine")]
    StateMachine,
    Classification,
    Independence,
    Termination,
    /// Memory/IO safety obligation (bounds checks, non-null, etc.). Legacy
    /// pre-APR-MONO contracts (e.g. `apr-cli-publish-extra-v1`) used this
    /// spelling; kept for back-compat alongside the 26 other types.
    Safety,
    /// Liveness property (eventually-happens). Same legacy contract
    /// (`apr-cli-publish-extra-v1`) uses this for progress obligations;
    /// kept for back-compat.
    Liveness,
    // Eiffel DbC types (Meyer 1997)
    Precondition,
    Postcondition,
    Frame,
    #[serde(rename = "loop_invariant")]
    LoopInvariant,
    #[serde(rename = "loop_variant")]
    LoopVariant,
    #[serde(rename = "old_state")]
    OldState,
    Subcontract,
}

impl std::fmt::Display for ObligationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Invariant => "invariant",
            Self::Equivalence => "equivalence",
            Self::Bound => "bound",
            Self::Monotonicity => "monotonicity",
            Self::Idempotency => "idempotency",
            Self::Linearity => "linearity",
            Self::Symmetry => "symmetry",
            Self::Associativity => "associativity",
            Self::Conservation => "conservation",
            Self::Ordering => "ordering",
            Self::Completeness => "completeness",
            Self::Soundness => "soundness",
            Self::Involution => "involution",
            Self::Determinism => "determinism",
            Self::Roundtrip => "roundtrip",
            Self::StateMachine => "state_machine",
            Self::Classification => "classification",
            Self::Independence => "independence",
            Self::Termination => "termination",
            Self::Safety => "safety",
            Self::Liveness => "liveness",
            Self::Precondition => "precondition",
            Self::Postcondition => "postcondition",
            Self::Frame => "frame",
            Self::LoopInvariant => "loop_invariant",
            Self::LoopVariant => "loop_variant",
            Self::OldState => "old_state",
            Self::Subcontract => "subcontract",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AppliesTo {
    All,
    Scalar,
    Simd,
    Converter,
    /// Algorithm-specific target (e.g., "degree", "bce", "huber").
    #[serde(other)]
    Other,
}

/// Kernel phase decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelStructure {
    pub phases: Vec<KernelPhase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPhase {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub invariant: Option<String>,
}

/// An enforcement rule from the contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementRule {
    pub description: String,
    #[serde(default)]
    pub check: Option<String>,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub reference: Option<String>,
}

/// A Popperian falsification test.
///
/// Each makes a falsifiable prediction about the implementation.
/// If the prediction is wrong, the test identifies root cause.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FalsificationTest {
    pub id: String,
    /// What the test asserts. Alias `description` accepted for legacy
    /// pre-APR-MONO contracts that used the `description:` field name.
    /// `name:` is NOT aliased because several legacy contracts (e.g.
    /// `publish-manifest-v1`) ship both `name:` (a slug) and
    /// `description:` (prose) side-by-side; aliasing both collapses to
    /// a `duplicate field` error.
    #[serde(default, alias = "description")]
    pub rule: String,
    /// The predicted outcome if the rule holds. Alias `expected` accepted
    /// for legacy contracts (e.g. `expected: exit 0`, `expected: "PASS"`).
    /// Defaulted because diagnostic contracts often encode prediction
    /// inside the rule text alone.
    #[serde(default, alias = "expected")]
    pub prediction: String,
    /// How to run the test. Alias `command` accepted for legacy contracts
    /// (e.g. shell snippets under `command: |`).
    #[serde(default, alias = "command")]
    pub test: Option<String>,
    /// How to run the test, in the `test_harness:` spelling. 619 entries in
    /// `contracts/` use this field INSTEAD of `test:` — 94 of them holding a
    /// real `cargo test` invocation, the rest a shell harness (`grep -q …`,
    /// `test -f …`, `bash …`).
    ///
    /// #2465: this field did not exist on the struct, and `FalsificationTest`
    /// is not `deny_unknown_fields`, so serde dropped it silently. Every one
    /// of those 619 entries reached `strict_test_binding` with `test: None`
    /// and was skipped — the gate reported them as neither bound nor broken.
    #[serde(default)]
    pub test_harness: Option<String>,
    /// The bare test-fn name, when the contract names it here rather than in
    /// an invocation. Deliberately NOT a serde `alias` of `rule`: several
    /// legacy contracts (e.g. `publish-manifest-v1`) ship `name:` (a slug)
    /// and `description:` (prose) side by side, and aliasing both onto one
    /// field collapses to a `duplicate field` parse error. Consumed as a
    /// binding source of last resort — see `strict_test_binding`.
    #[serde(default)]
    pub name: Option<String>,
    /// What failure means. Alias `fails_if` accepted for legacy contracts.
    /// Defaulted because several legacy diagnostic contracts omit it.
    #[serde(default, alias = "fails_if")]
    pub if_fails: String,
}

/// A Kani bounded model checking harness definition.
///
/// Corresponds to Phase 6 (Verify) of the pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KaniHarness {
    pub id: String,
    pub obligation: String,
    #[serde(default)]
    pub property: Option<String>,
    #[serde(default)]
    pub bound: Option<u32>,
    #[serde(default)]
    pub strategy: Option<KaniStrategy>,
    #[serde(default)]
    pub solver: Option<String>,
    #[serde(default)]
    pub harness: Option<String>,
    /// GH-1595: When `true`, the harness has been verified by a green
    /// `cargo kani` run in CI (e.g. apr-cookbook `kani-gate`). Lifts the
    /// D3 strategy weight to 1.0 for non-exhaustive strategies because
    /// the runtime witness supplants the static-readiness signal.
    #[serde(default)]
    pub actually_verified: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KaniStrategy {
    Exhaustive,
    StubFloat,
    Compositional,
    BoundedInt,
}

impl std::fmt::Display for KaniStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Exhaustive => "exhaustive",
            Self::StubFloat => "stub_float",
            Self::Compositional => "compositional",
            Self::BoundedInt => "bounded_int",
        };
        write!(f, "{s}")
    }
}

/// Phase 7: Lean 4 theorem proving metadata for a proof obligation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeanProof {
    /// Lean 4 theorem name (e.g., `Softmax.partition_of_unity`).
    pub theorem: String,
    /// Lean 4 module path (e.g., `ProvableContracts.Softmax`).
    #[serde(default)]
    pub module: Option<String>,
    /// Current status of the Lean proof.
    #[serde(default)]
    pub status: LeanStatus,
    /// Lean-level theorem dependencies.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Mathlib import paths required.
    #[serde(default)]
    pub mathlib_imports: Vec<String>,
    /// Free-form notes (e.g., "Proof over reals; f32 gap addressed separately").
    #[serde(default)]
    pub notes: Option<String>,
}

/// Status of a Lean 4 proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LeanStatus {
    /// Proof is complete and type-checks.
    Proved,
    /// Proof uses `sorry` (axiomatized, not yet proved).
    #[default]
    Sorry,
    /// Work in progress.
    Wip,
    /// Obligation is not amenable to Lean proof (e.g., performance).
    NotApplicable,
}

impl std::fmt::Display for LeanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Proved => "proved",
            Self::Sorry => "sorry",
            Self::Wip => "wip",
            Self::NotApplicable => "not-applicable",
        };
        write!(f, "{s}")
    }
}

/// Phase 7: Verification summary across all obligations in a contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSummary {
    pub total_obligations: u32,
    #[serde(default)]
    pub l2_property_tested: u32,
    #[serde(default)]
    pub l3_kani_proved: u32,
    #[serde(default)]
    pub l4_lean_proved: u32,
    #[serde(default)]
    pub l4_sorry_count: u32,
    #[serde(default)]
    pub l4_not_applicable: u32,
}

/// QA gate definition for certeza integration.
///
/// Legacy diagnostic contracts (e.g.
/// `decode-hot-path-prefix-cache-diagnostic-v1`) ship a `qa_gate:` block
/// with only `must_pass:` / `integration:` / `regression_protection:` — no
/// `id:` or `name:`. All schema fields default so those parse cleanly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QaGate {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub checks: Vec<String>,
    #[serde(default)]
    pub pass_criteria: Option<String>,
    #[serde(default)]
    pub falsification: Option<String>,
}

/// A type-level invariant (Meyer's class invariant).
///
/// Asserts a predicate that must hold for every instance of `type_name`
/// at every stable state — after construction and after every public method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInvariant {
    pub name: String,
    /// Rust type name (e.g., `ValidatedTensor`).
    #[serde(rename = "type")]
    pub type_name: String,
    /// Rust boolean expression over `self` (e.g., `!self.dims.is_empty()`).
    pub predicate: String,
    #[serde(default)]
    pub description: Option<String>,
}

/// Coq verification specification for a contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoqSpec {
    /// Coq module name (e.g., `SoftmaxSpec`).
    pub module: String,
    /// Coq import statements.
    #[serde(default)]
    pub imports: Vec<String>,
    /// Coq definitions generated from equations.
    #[serde(default)]
    pub definitions: Vec<CoqDefinition>,
    /// Links from proof obligations to Coq lemmas.
    #[serde(default)]
    pub obligations: Vec<CoqObligation>,
}

/// A Coq definition derived from a contract equation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoqDefinition {
    pub name: String,
    pub statement: String,
}

/// A link between a proof obligation and a Coq lemma.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoqObligation {
    /// References a proof obligation property or ID.
    pub links_to: String,
    /// Coq lemma name.
    pub coq_lemma: String,
    /// Current status of the Coq proof.
    #[serde(default = "coq_status_default")]
    pub status: String,
}

fn coq_status_default() -> String {
    "stub".to_string()
}

/// Accepts `equations:` as either a map (canonical) or a list-of-dicts
/// with an `id` field (legacy pre-APR-MONO diagnostic contracts like
/// `decode-hot-path-prefix-cache-diagnostic-v1`). The list form promotes
/// each entry's `id` to the map key; entries without `id` fall back to
/// `equation_{N}` so parsing never silently drops data.
fn deserialize_equations<'de, D>(d: D) -> Result<BTreeMap<String, Equation>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    use serde_yaml::Value;

    let value = Value::deserialize(d)?;
    match value {
        Value::Null => Ok(BTreeMap::new()),
        Value::Mapping(_) => serde_yaml::from_value(value).map_err(D::Error::custom),
        Value::Sequence(items) => {
            let mut out = BTreeMap::new();
            for (i, mut item) in items.into_iter().enumerate() {
                let key = match &mut item {
                    Value::Mapping(m) => m
                        .remove(Value::String("id".into()))
                        .and_then(|v| v.as_str().map(ToString::to_string))
                        .unwrap_or_else(|| format!("equation_{i}")),
                    _ => format!("equation_{i}"),
                };
                let eq: Equation = serde_yaml::from_value(item).map_err(D::Error::custom)?;
                out.insert(key, eq);
            }
            Ok(out)
        }
        other => Err(D::Error::custom(format!(
            "`equations:` must be a map or a list; got {other:?}"
        ))),
    }
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
