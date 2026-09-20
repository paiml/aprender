use std::path::Path;

use crate::error::ContractError;
use crate::schema::types::{Contract, ContractKind, KaizenRecord, CONTRACT_TOP_LEVEL_FIELDS};

/// Parse a YAML contract file into a [`Contract`] struct.
///
/// This is the entry point for Phase 2 validation. The parser
/// deserializes the YAML and performs structural checks.
///
/// # Errors
///
/// Returns [`ContractError::Io`] if the file cannot be read,
/// or [`ContractError::Yaml`] if the YAML is malformed.
pub fn parse_contract(path: &Path) -> Result<Contract, ContractError> {
    let content = std::fs::read_to_string(path)?;
    parse_contract_str(&content)
}

/// Files under `contracts/` that are NOT `Contract` documents.
///
/// `contracts/binding.yaml` is a `BindingRegistry` (equation → implementing
/// function), not a contract. It has no `metadata:` block, so parsing it as a
/// `Contract` fails with ``missing field `metadata` `` — which is exactly how
/// `cargo test -p aprender-contracts --test validate_contracts` failed 3 of its
/// 10 tests on `main` while `pv lint contracts/` reported zero errors: the two
/// walkers disagreed about what a contract file is.
/// Files under `contracts/` that are NOT contracts, and would fail to parse as
/// one: the binding registry, and ONT-001 ONT-1's declaration of corpora this
/// tree does NOT hold (`contracts/external-corpora.yaml`). Adding a file here is
/// how the corpus keeps ONE definition of "a contract file" — the census, the
/// linter and the validator all read this list.
const NON_CONTRACT_FILENAMES: [&str; 4] = [
    "binding.yaml",
    "binding.yml",
    "external-corpora.yaml",
    // ONT-2b: Σ (`contracts/ontology.yaml`) declares what contracts may SAY; it is not one of them and has
    // no `metadata:`. Measured before this line existed: adding Σ took `pv lint contracts/` from 1792
    // contracts / 0 errors to 1793 / 1, i.e. the corpus failed because its own vocabulary was parsed as a
    // member of it.
    "ontology.yaml",
];

/// Is `path` a `.yaml` file the contract schema owns?
///
/// The single source of truth for "which files under `contracts/` are
/// contracts". `pv lint`'s directory walker and the `validate_contracts`
/// integration test both call it, so neither can drift into walking a file the
/// other skips. Directory-level exclusions (`kaizen/`, `legacy/`,
/// `pipelines/`, `publish-manifests/`) are a separate concern and stay with the
/// recursive walker in `lint::gates`.
#[must_use]
pub fn is_contract_yaml(path: &Path) -> bool {
    if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
        return false;
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    !name.starts_with('.') && !NON_CONTRACT_FILENAMES.contains(&name)
}

/// Parse a YAML contract from a string.
///
/// Two passes on purpose. `Contract` is intentionally NOT
/// `#[serde(deny_unknown_fields)]` — 1224 of the 1726 contracts `pv lint`
/// walks carry a downstream-owned top-level block (`family:`, `sections:`,
/// `surface:`, …) and denying them would stop ~71% of the corpus from parsing
/// in one commit. But serde's tolerance is also what let a top-level `kind:`
/// (119 contracts) and a misspelled `falsification:` block vanish without a
/// sound. So the second pass reads the raw mapping and records which top-level
/// keys serde did not consume, in [`Contract::unknown_top_level_keys`]; the
/// validator turns the never-legitimate ones into errors (SCHEMA-018 /
/// SCHEMA-019) and leaves the rest alone.
///
/// # Errors
///
/// Returns [`ContractError::Yaml`] if the YAML is malformed or does not match
/// the contract schema.
pub fn parse_contract_str(yaml: &str) -> Result<Contract, ContractError> {
    let mut contract: Contract = serde_yaml::from_str(yaml)?;
    contract.unknown_top_level_keys = unknown_top_level_keys(yaml);
    contract.kind_declared = kind_is_declared(yaml);
    contract.strict_yaml_error = strict_yaml_error(yaml);
    contract.kaizen_record = kaizen_record(yaml, contract.kind())?;
    Ok(contract)
}

/// Third pass, run ONLY for `metadata.kind: kaizen`: read the top-level blocks
/// a kaizen improvement record carries (`contract:`, `kaizen:`, `status:`,
/// `baseline:`, `target:`, …) that `Contract` deliberately does not model.
///
/// Gated on the kind rather than run unconditionally because those key names
/// are not owned by the kaizen schema: `status:`, `version:` and `invariants:`
/// appear at top level across the corpus with shapes that would make this
/// deserialize fail. On a non-kaizen contract that failure would turn a
/// perfectly good contract into a parse error; scoping the pass means it can
/// only ever be reached by a document that has declared itself a kaizen record.
///
/// When it IS reached, a type error is returned rather than swallowed — a
/// kaizen record whose `baseline:` is unreadable must be REPORTED, not
/// silently validated as if it had none.
fn kaizen_record(yaml: &str, kind: ContractKind) -> Result<Option<KaizenRecord>, ContractError> {
    if kind != ContractKind::Kaizen {
        return Ok(None);
    }
    Ok(Some(serde_yaml::from_str::<KaizenRecord>(yaml)?))
}

/// Top-level mapping keys of `yaml` that are not fields of [`Contract`].
///
/// Deliberately deserializes into `BTreeMap<String, IgnoredAny>` rather than
/// `serde_yaml::Value`: `IgnoredAny` drains each value without building it, so
/// this pass reads ONLY the top-level key names and cannot be derailed by
/// anything nested. That matters — `contracts/apr-cli-commands-v1.yaml` defines
/// `subcommands:` twice inside `commands:`, which makes a strict
/// `serde_yaml::Value` parse fail; capturing keys through `Value` silently
/// returned "no unknown keys" for that file and its top-level
/// `kind: CLICommandContract` went unreported. A `BTreeMap` also just
/// overwrites a duplicate top-level key instead of erroring.
///
/// Returns an empty list when the document is not a mapping at all — that case
/// is already a hard parse error above, so this never masks one.
fn unknown_top_level_keys(yaml: &str) -> Vec<String> {
    use serde::de::IgnoredAny;
    use std::collections::BTreeMap;

    let Ok(map) = serde_yaml::from_str::<BTreeMap<String, IgnoredAny>>(yaml) else {
        return Vec::new();
    };
    map.into_keys()
        .filter(|k| !CONTRACT_TOP_LEVEL_FIELDS.contains(&k.as_str()))
        .collect()
}

/// Was `metadata.kind:` written in the document? (ONT-6b, infra#751)
///
/// Read from the raw mapping, not from the deserialized `Contract`: serde has
/// already replaced an absent `kind` with [`ContractKind::Kernel`] by then, and
/// that substitution is precisely what this has to observe.
///
/// Only `metadata`'s KEYS are built; every other top-level value is drained
/// into `IgnoredAny` by the derived deserializer. This is not a style choice —
/// the reasons `unknown_top_level_keys` gives above apply here with teeth.
/// Deserializing the whole document into a `BTreeMap<String, serde_yaml::Value>`
/// would FAIL on `contracts/apr-cli-commands-v1.yaml`, which defines
/// `subcommands:` twice inside `commands:`, and the `false` returned for that
/// reason would be this function silently reporting "no kind declared" about a
/// file that declares one — the same shape of wrong answer ONT-6b exists to
/// remove. Draining unknown top-level values means a duplicate key nested under
/// one is never built and cannot derail the probe.
///
/// A document whose `metadata` is absent or is not a mapping answers `false` —
/// it declared no kind, which is the truth, and the missing-`metadata` case is
/// a parse error the caller has already surfaced.
fn kind_is_declared(yaml: &str) -> bool {
    use serde::de::IgnoredAny;
    use std::collections::BTreeMap;

    #[derive(serde::Deserialize)]
    struct KindProbe {
        metadata: BTreeMap<String, IgnoredAny>,
    }

    serde_yaml::from_str::<KindProbe>(yaml).is_ok_and(|probe| probe.metadata.contains_key("kind"))
}

/// The error a STRICT reader gets on YAML the contract schema accepted.
///
/// `Contract`'s derived deserializer walks only the fields it knows and skips
/// the rest, so a document can be well-formed to it and malformed to anyone
/// else. The one shape this catches today is a duplicate mapping key: YAML
/// requires keys to be unique, and every consumer that builds a real map (a
/// `serde_yaml::Value`, `yq`, a Python `dict`) keeps exactly one of them and
/// throws the other away without a word.
///
/// `None` means the document round-trips through a strict reader.
fn strict_yaml_error(yaml: &str) -> Option<String> {
    serde_yaml::from_str::<serde_yaml::Value>(yaml)
        .err()
        .map(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_CONTRACT: &str = r#"
metadata:
  version: "1.0.0"
  description: "Test contract"
  references:
    - "Test paper (2024)"
equations:
  test_eq:
    formula: "f(x) = x + 1"
proof_obligations: []
falsification_tests: []
"#;

    /// ONT-6b: the probe answers about `metadata.kind` and nothing else.
    ///
    /// The three cases that decide it, including the one that is a trap: a
    /// document carrying a DUPLICATE key nested under an unknown top-level
    /// block. `contracts/apr-cli-commands-v1.yaml` is that shape in the real
    /// corpus (`subcommands:` appears 20 times under `commands:`), and a probe
    /// that deserialized the whole document into `serde_yaml::Value` would fail
    /// on it and report "no kind declared" about a file that declares one.
    /// `KindProbe` drains unknown top-level values instead of building them, so
    /// the duplicate is never constructed.
    #[test]
    fn kind_is_declared_reads_metadata_and_survives_a_duplicate_nested_key() {
        assert!(
            !kind_is_declared(MINIMAL_CONTRACT),
            "no metadata.kind is written, so none was declared"
        );
        let declared = MINIMAL_CONTRACT.replace(
            "  description: \"Test contract\"",
            "  description: \"Test contract\"\n  kind: pattern",
        );
        assert!(
            kind_is_declared(&declared),
            "metadata.kind is written, whatever its value"
        );

        let with_duplicate =
            format!("{declared}commands:\n  root:\n    subcommands: [a]\n    subcommands: [b]\n");
        assert!(
            serde_yaml::from_str::<std::collections::BTreeMap<String, serde_yaml::Value>>(
                &with_duplicate
            )
            .is_err(),
            "anti-vacuity: this fixture must be a document a whole-file Value parse REFUSES, \
             otherwise the next assertion proves nothing"
        );
        assert!(
            kind_is_declared(&with_duplicate),
            "a duplicate key under an unknown top-level block must not make a declared kind \
             read as absent"
        );

        assert!(
            !kind_is_declared("not: a contract\n"),
            "a document with no metadata declared no kind"
        );
    }

    #[test]
    fn parse_minimal_contract() {
        let contract = parse_contract_str(MINIMAL_CONTRACT).unwrap();
        assert_eq!(contract.metadata.version, "1.0.0");
        assert_eq!(contract.metadata.description, "Test contract");
        assert_eq!(contract.equations.len(), 1);
        assert!(contract.equations.contains_key("test_eq"));
    }

    #[test]
    fn parse_contract_with_all_fields() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  created: "2026-02-18"
  author: "Test Author"
  description: "Full contract"
  references:
    - "Paper A (2024)"
    - "Paper B (2025)"
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))"
    domain: "x ∈ ℝ^n, n ≥ 1"
    codomain: "σ(x) ∈ (0,1)^n"
    invariants:
      - "sum(output) = 1.0"
      - "output_i > 0"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|sum(softmax(x)) - 1.0| < ε"
    tolerance: 1.0e-6
    applies_to: all
  - type: equivalence
    property: "SIMD matches scalar"
    tolerance: 8.0
    applies_to: simd
kernel_structure:
  phases:
    - name: find_max
      description: "Find max element"
      invariant: "max >= all elements"
    - name: exp_subtract
      description: "Compute exp(x_i - max)"
      invariant: "all values in (0, 1]"
simd_dispatch:
  softmax:
    scalar: "softmax_scalar"
    avx2: "softmax_avx2"
enforcement:
  normalization:
    description: "Output sums to 1.0"
    check: "contract_tests::FALSIFY-SM-001"
    severity: "ERROR"
falsification_tests:
  - id: FALSIFY-SM-001
    rule: "Normalization"
    prediction: "sum(output) ≈ 1.0"
    test: "proptest with random vectors"
    if_fails: "Missing max-subtraction trick"
kani_harnesses:
  - id: KANI-SM-001
    obligation: SM-INV-001
    property: "Softmax sums to 1.0"
    bound: 16
    strategy: stub_float
    solver: cadical
    harness: verify_softmax_normalization
qa_gate:
  id: F-SM-001
  name: "Softmax Contract"
  checks:
    - "normalization"
  pass_criteria: "All falsification tests pass"
  falsification: "Introduce off-by-one in max reduction"
"#;

        let contract = parse_contract_str(yaml).unwrap();
        assert_eq!(contract.metadata.version, "1.0.0");
        assert_eq!(contract.metadata.references.len(), 2);
        assert_eq!(contract.equations.len(), 1);
        assert_eq!(contract.proof_obligations.len(), 2);
        assert!(contract.kernel_structure.is_some());
        let ks = contract.kernel_structure.unwrap();
        assert_eq!(ks.phases.len(), 2);
        assert_eq!(contract.simd_dispatch.len(), 1);
        assert_eq!(contract.enforcement.len(), 1);
        assert_eq!(contract.falsification_tests.len(), 1);
        assert_eq!(contract.falsification_tests[0].id, "FALSIFY-SM-001");
        assert_eq!(contract.kani_harnesses.len(), 1);
        assert_eq!(contract.kani_harnesses[0].bound, Some(16));
        assert!(contract.qa_gate.is_some());
    }

    #[test]
    fn parse_invalid_yaml_returns_error() {
        let result = parse_contract_str("not: [valid: yaml: {{");
        assert!(result.is_err());
    }

    #[test]
    fn parse_missing_metadata_returns_error() {
        let yaml = r#"
equations:
  test:
    formula: "f(x) = x"
"#;
        let result = parse_contract_str(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn parse_obligation_types() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "type test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "test"
    if_fails: ""
  - type: equivalence
    property: "test"
  - type: bound
    property: "test"
  - type: monotonicity
    property: "test"
  - type: idempotency
    property: "test"
  - type: linearity
    property: "test"
  - type: symmetry
    property: "test"
  - type: associativity
    property: "test"
  - type: conservation
    property: "test"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        assert_eq!(contract.proof_obligations.len(), 9);
    }

    #[test]
    fn parse_dbc_obligation_types() {
        use crate::schema::types::ObligationType;

        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "DbC type test"
  depends_on: ["parent-v1"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: precondition
    property: "input finite"
    formal: "isFinite(x)"
  - type: postcondition
    property: "output bounded"
    requires: "PRE-001"
  - type: frame
    property: "input unchanged"
  - type: loop_invariant
    property: "max tracks true max"
    applies_to_phase: "find_max"
  - type: loop_variant
    property: "remaining decreasing"
    applies_to_phase: "accumulate"
  - type: old_state
    property: "cache grows"
  - type: subcontract
    property: "refines parent"
    parent_contract: "parent-v1"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        assert_eq!(contract.proof_obligations.len(), 7);
        assert_eq!(
            contract.proof_obligations[0].obligation_type,
            ObligationType::Precondition
        );
        assert_eq!(
            contract.proof_obligations[1].obligation_type,
            ObligationType::Postcondition
        );
        assert_eq!(
            contract.proof_obligations[1].requires.as_deref(),
            Some("PRE-001")
        );
        assert_eq!(
            contract.proof_obligations[2].obligation_type,
            ObligationType::Frame
        );
        assert_eq!(
            contract.proof_obligations[3].obligation_type,
            ObligationType::LoopInvariant
        );
        assert_eq!(
            contract.proof_obligations[3].applies_to_phase.as_deref(),
            Some("find_max")
        );
        assert_eq!(
            contract.proof_obligations[4].obligation_type,
            ObligationType::LoopVariant
        );
        assert_eq!(
            contract.proof_obligations[5].obligation_type,
            ObligationType::OldState
        );
        assert_eq!(
            contract.proof_obligations[6].obligation_type,
            ObligationType::Subcontract
        );
        assert_eq!(
            contract.proof_obligations[6].parent_contract.as_deref(),
            Some("parent-v1")
        );
    }

    #[test]
    fn parse_contract_with_kind_model_family() {
        use crate::schema::types::ContractKind;

        // A realistic aprender model-family YAML: metadata + custom
        // top-level fields. No equations, no proof obligations — should
        // parse and validate cleanly as kind: model-family.
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Google BERT architecture family metadata"
  kind: model-family
  references:
    - "https://arxiv.org/abs/1810.04805"
    - "https://huggingface.co/google-bert"
# Custom top-level fields ignored by the kernel schema,
# consumed by the downstream crate that owns the file.
family: bert
display_name: "Google BERT"
vendor: Google
architectures:
  - BertModel
  - BertForMaskedLM
size_variants:
  base:
    parameters: "110M"
    hidden_dim: 768
"#;
        let contract = parse_contract_str(yaml).unwrap();
        assert_eq!(contract.kind(), ContractKind::ModelFamily);
        assert!(!contract.requires_proofs());
        assert!(!contract.is_registry());
        // Validates cleanly — no kernel-specific checks fire.
        let violations = crate::schema::validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == crate::error::Severity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "model-family YAML should validate with no errors, got: {errors:?}",
        );
    }

    #[test]
    fn parse_contract_defaults_to_kernel_kind() {
        use crate::schema::types::ContractKind;

        let contract = parse_contract_str(MINIMAL_CONTRACT).unwrap();
        assert_eq!(contract.kind(), ContractKind::Kernel);
        assert!(contract.requires_proofs());
    }

    #[test]
    fn parse_kani_strategies() {
        use crate::schema::types::KaniStrategy;

        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "kani test"
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: K1
    obligation: OBL-1
    strategy: exhaustive
  - id: K2
    obligation: OBL-2
    strategy: stub_float
  - id: K3
    obligation: OBL-3
    strategy: compositional
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        assert_eq!(contract.kani_harnesses.len(), 3);
        assert_eq!(
            contract.kani_harnesses[0].strategy,
            Some(KaniStrategy::Exhaustive)
        );
        assert_eq!(
            contract.kani_harnesses[1].strategy,
            Some(KaniStrategy::StubFloat)
        );
        assert_eq!(
            contract.kani_harnesses[2].strategy,
            Some(KaniStrategy::Compositional)
        );
    }
}
