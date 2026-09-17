//! ONT-001 §5 ONT-2b — the `sigma` gate: the corpus may only say what Σ declares.
//!
//! Two corpus rules live here, both `reject:` (exit 1) because the corpus is what is wrong:
//!
//! - a contract whose `entity.type` Σ's `entity_types` does not declare (PV-ONT-001);
//! - a `relations:` key Σ's `roles` does not declare (PV-ONT-002).
//!
//! Two answers are NOT corpus verdicts: a malformed Σ is the declaration's fault ([`SigmaOutcome::Malformed`],
//! exit 3 `error:`), and a corpus with no Σ at all measured nothing ([`SigmaOutcome::NoSigma`], exit 2 `decline:` —
//! ONT R-2: zero is a decline, never an accept).
//!
//! **This reads RAW YAML, not the parsed `Contract`.** `entity:` and `relations:` are §4.2 additive keys that the
//! `Contract` struct does not carry, and serde drops what it does not know — the same reason ONT-1's census reads
//! raw. A gate that read the typed struct would find zero `entity:` blocks in a corpus that had them.
//!
//! **Both rules are vacuous on aprender's corpus today** (0 of 1792 contracts carry `entity:` or `relations:`), so
//! `tests/fixtures/ont/sigma-entity-type-unknown/` and `…/sigma-undeclared-role/` are the only witnesses that they
//! can fire at all. A rule that cannot fire is this fleet's signature defect; those fixtures are the answer to it.

use std::path::Path;
use std::time::Instant;

use crate::ontology::sigma::{Sigma, SigmaError};

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult, Verdict};

/// What one `sigma` run answers. Only [`SigmaOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum SigmaOutcome {
    /// No `ontology.yaml` under the corpus: nothing was measured.
    NoSigma,
    /// Σ does not parse, or does not satisfy its own integrity rules.
    Malformed(SigmaError),
    /// Σ was read and the corpus was checked against it.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// Run the gate over `contract_dir`, reading Σ from `<contract_dir>/ontology.yaml`.
#[must_use]
pub fn run_sigma_gate(contract_dir: &Path) -> SigmaOutcome {
    let start = Instant::now();
    let path = contract_dir.join("ontology.yaml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return SigmaOutcome::NoSigma;
    };
    let sigma = match Sigma::from_yaml(&text) {
        Ok(s) => s,
        Err(e) => return SigmaOutcome::Malformed(e),
    };
    if let Err(e) = sigma.check_integrity() {
        return SigmaOutcome::Malformed(e);
    }

    let mut files = Vec::new();
    super::collect_yaml_files(contract_dir, &mut files);
    let mut findings = Vec::new();
    let mut checked = 0usize;
    for file in &files {
        if file == &path {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(file) else {
            continue;
        };
        let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&raw) else {
            continue; // a file that does not parse is the `validate` gate's business, not Σ's
        };
        checked += 1;
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        findings.extend(check_entity_type(&sigma, &doc, &stem, file));
        findings.extend(check_roles(&sigma, &doc, &stem, file));
    }

    let violations = findings.len();
    let passed = violations == 0;
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let result = GateResult {
        name: "sigma".into(),
        passed,
        skipped: false,
        verdict: Verdict::from_gate(passed, false),
        duration_ms: duration,
        // `GateDetail` is FROZEN at the 0.3.1 variants, so the shape is borrowed the way `duplicate-stems`
        // borrows it and the real payload rides in `GateExtra` (which is `#[non_exhaustive]`).
        detail: GateDetail::Validate {
            contracts: checked,
            errors: violations,
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Sigma {
            entity_types: sigma.entity_types.len(),
            roles: sigma.roles.len(),
            symbols: sigma.symbols.len(),
            contracts_checked: checked,
            violations,
        }),
    };
    SigmaOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

/// PV-ONT-001 — `entity.type` must be one Σ declares.
fn check_entity_type(
    sigma: &Sigma,
    doc: &serde_yaml::Value,
    stem: &str,
    file: &Path,
) -> Vec<LintFinding> {
    let Some(ty) = doc
        .get("entity")
        .and_then(|e| e.get("type"))
        .and_then(serde_yaml::Value::as_str)
    else {
        return Vec::new();
    };
    if sigma.declares_entity_type(ty) {
        return Vec::new();
    }
    let mut f = LintFinding::new(
        "PV-ONT-001",
        RuleSeverity::Error,
        format!("entity.type `{ty}` is not in Σ entity_types (contracts/ontology.yaml)"),
        file.display().to_string(),
    );
    f.contract_stem = Some(stem.to_string());
    vec![f]
}

/// PV-ONT-002 — every `relations:` key must be a role Σ declares.
fn check_roles(
    sigma: &Sigma,
    doc: &serde_yaml::Value,
    stem: &str,
    file: &Path,
) -> Vec<LintFinding> {
    let Some(map) = doc.get("relations").and_then(serde_yaml::Value::as_mapping) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for key in map.keys() {
        let Some(role) = key.as_str() else { continue };
        if sigma.declares_role(role) {
            continue;
        }
        let mut f = LintFinding::new(
            "PV-ONT-002",
            RuleSeverity::Error,
            format!("relation `{role}` is not a role Σ declares (contracts/ontology.yaml)"),
            file.display().to_string(),
        );
        f.contract_stem = Some(stem.to_string());
        out.push(f);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont")
            .join(name)
    }

    #[test]
    fn a_well_formed_corpus_passes() {
        match run_sigma_gate(&fixture("sigma-ok")) {
            SigmaOutcome::Ran { result, findings } => {
                assert!(result.passed, "{findings:?}");
                assert_eq!(result.verdict, Verdict::Pass);
                assert!(findings.is_empty());
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn an_undeclared_entity_type_is_a_finding() {
        match run_sigma_gate(&fixture("sigma-entity-type-unknown")) {
            SigmaOutcome::Ran { result, findings } => {
                assert!(!result.passed);
                assert_eq!(findings.len(), 1);
                assert_eq!(findings[0].rule_id, "PV-ONT-001");
                assert!(findings[0].message.contains("ghost-entity"));
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn an_undeclared_role_is_a_finding() {
        match run_sigma_gate(&fixture("sigma-undeclared-role")) {
            SigmaOutcome::Ran { result, findings } => {
                assert!(!result.passed);
                assert_eq!(findings.len(), 1);
                assert_eq!(findings[0].rule_id, "PV-ONT-002");
                assert!(findings[0].message.contains("ghost_role"));
            }
            other => panic!("expected Ran, got {other:?}"),
        }
    }

    #[test]
    fn a_malformed_sigma_is_not_a_corpus_verdict() {
        match run_sigma_gate(&fixture("sigma-malformed")) {
            SigmaOutcome::Malformed(e) => {
                assert!(e.to_string().contains("pv_contract"), "{e}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn no_sigma_measures_nothing() {
        assert!(matches!(
            run_sigma_gate(&fixture("sigma-absent")),
            SigmaOutcome::NoSigma
        ));
    }

    #[test]
    fn sigma_itself_is_not_counted_as_a_contract() {
        // `ontology.yaml` lives in the corpus directory; counting it as a contract would make the gate
        // report a number it did not check.
        match run_sigma_gate(&fixture("sigma-ok")) {
            SigmaOutcome::Ran { result, .. } => match result.extra {
                Some(GateExtra::Sigma {
                    contracts_checked, ..
                }) => assert_eq!(contracts_checked, 1),
                other => panic!("expected the Sigma payload, got {other:?}"),
            },
            other => panic!("expected Ran, got {other:?}"),
        }
    }
}
