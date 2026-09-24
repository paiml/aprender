//! ONT-001 §5 ONT-7 — the `valid-under` gate: a kernel-kind contract carries a WORLD INDEX.
//!
//! A kernel contract's claim ("row sums to 1 within ε") holds in some world: a toolchain, a host class,
//! a backend, a feature set. Until this gate, nothing said which, so a claim measured on `cuda` read the
//! same as one that holds everywhere. `metadata.valid_under` names it:
//!
//! ```yaml
//! metadata:
//!   valid_under:
//!     world: committed                      # a key of Σ's `worlds:` (the index); omitted = `committed`
//!     toolchain: { rust: "1.93" }           # optional qualifiers, closed set
//!     host_class: [x86_64-linux]
//!     backend: [cpu, cuda]
//!     features: [cuda]
//! ```
//!
//! Rules, all `reject:` (exit 1) because the corpus is what is wrong:
//!
//! - PV-ONT-013 — `valid_under` is present but not a mapping, is empty, or carries a key outside the closed set;
//! - PV-ONT-014 — `world` is not a string, or names a world Σ does not declare (an omitted `world` reads
//!   [`DEFAULT_WORLD`], which Σ must declare — so the spec's Appendix B example is admitted as written);
//! - PV-ONT-015 — a qualifier has the wrong shape (`toolchain` a map of strings; the others non-empty lists
//!   of non-empty strings);
//! - PV-ONT-016 — the `contracts_without_valid_under` ratchet ROSE. The debt is the kernel-kind contracts
//!   (non-registry) that carry no `valid_under`; it is recorded at the TOP LEVEL of
//!   `contracts/lint-baseline.json` (the row's probe reads it there) and is shrink-only, the
//!   `formal_prose` pattern. Without a baseline the count is reported, never treated as satisfied.
//!
//! The rules apply to `valid_under` wherever it appears; only the ratchet is scoped to kernel-kind, because
//! that is the class the row obliges. Non-verdict answers: no Σ → decline, malformed Σ → error, and a corpus
//! with no kernel-kind contract AND no `valid_under` anywhere measured nothing → decline (ONT R-2: zero is a
//! decline, never an accept). The gate is computed in every `pv lint` run (gate 13, R-8) and armed per repo.
//!
//! **Reads RAW YAML for the key** (the `Contract` struct does not carry it, and serde drops what it does
//! not know — sigma_gate's reason), and the PARSED contract for the kind, so "kernel" means exactly what
//! `pv validate` means by it, including the absent-kind default.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use crate::ontology::sigma::{Sigma, SigmaError};
use crate::schema::{parse_contract, ContractKind};

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult, Verdict};

/// The keys `metadata.valid_under` may carry. Closed world: anything else is a reject.
pub const VALID_UNDER_KEYS: [&str; 5] = ["world", "toolchain", "host_class", "backend", "features"];

/// The world a `valid_under` without `world:` is read in. Σ declares it, in its own words: `committed` is
/// "the world every contract is read in unless it says otherwise" (contracts/ontology.yaml). So the spec's
/// Appendix B example — qualifiers and no `world` — is admitted, and indexes `committed`. It must still be a
/// world Σ declares: a Σ without `committed` gives an omitted `world` nothing to default to (PV-ONT-014).
pub const DEFAULT_WORLD: &str = "committed";

/// The top-level key of `lint-baseline.json` that records the debt.
pub const BASELINE_KEY: &str = "contracts_without_valid_under";

/// What one `valid-under` run answers. Only [`ValidUnderOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum ValidUnderOutcome {
    /// No `ontology.yaml` under the corpus: there is no world index to resolve against.
    NoSigma,
    /// Σ does not parse, or does not satisfy its own integrity rules.
    Malformed(SigmaError),
    /// The corpus holds no kernel-kind contract: nothing the row obliges was measured.
    NoKernels {
        /// Contract files read.
        contracts_checked: usize,
    },
    /// Σ was read and the corpus was checked against it.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// Run the gate over `contract_dir`, reading Σ from `<contract_dir>/ontology.yaml`.
#[must_use]
pub fn run_valid_under_gate(contract_dir: &Path) -> ValidUnderOutcome {
    let start = Instant::now();
    let sigma_path = contract_dir.join("ontology.yaml");
    let Ok(text) = std::fs::read_to_string(&sigma_path) else {
        return ValidUnderOutcome::NoSigma;
    };
    let sigma = match Sigma::from_yaml(&text) {
        Ok(s) => s,
        Err(e) => return ValidUnderOutcome::Malformed(e),
    };
    if let Err(e) = sigma.check_integrity() {
        return ValidUnderOutcome::Malformed(e);
    }

    let mut c = Census::default();
    let mut files = Vec::new();
    super::collect_yaml_files(contract_dir, &mut files);
    files.sort();
    for file in files.iter().filter(|f| **f != sigma_path) {
        c.observe(&sigma, file);
    }
    // Zero is a decline only when NOTHING was measured: a malformed `valid_under` on a non-kernel contract is
    // still a reject (the rules apply wherever the key appears — #4076 re-review, lane 2).
    if c.kernels == 0 && c.carrying == 0 {
        return ValidUnderOutcome::NoKernels {
            contracts_checked: c.checked,
        };
    }
    let baseline = baseline_without_valid_under(contract_dir);
    c.findings
        .extend(ratchet_finding(baseline, c.kernels_without));
    let Census {
        findings,
        checked,
        kernels,
        kernels_without,
        carrying,
        by_world,
    } = c;

    let violations = findings.len();
    let passed = violations == 0;
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let result = GateResult {
        name: "valid-under".into(),
        passed,
        skipped: false,
        verdict: Verdict::from_gate(passed, false),
        duration_ms: duration,
        // `GateDetail` is FROZEN at the 0.3.1 variants (see `GateExtra`), so the shape is borrowed as the
        // `sigma` gate borrows it and the real payload rides in `GateExtra`.
        detail: GateDetail::Validate {
            contracts: checked,
            errors: violations,
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::ValidUnder {
            worlds: sigma.worlds.keys().cloned().collect(),
            contracts_checked: checked,
            kernel_contracts: kernels,
            contracts_with_valid_under: carrying,
            contracts_without_valid_under: kernels_without,
            baseline,
            by_world: by_world.iter().map(|(w, n)| format!("{w}={n}")).collect(),
            violations,
        }),
    };
    ValidUnderOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

/// What one pass over the corpus counted. Split out of [`run_valid_under_gate`] under the complexity ratchet
/// (cognitive 40 → below 25); the arithmetic is unchanged.
#[derive(Default)]
struct Census {
    findings: Vec<LintFinding>,
    checked: usize,
    kernels: usize,
    kernels_without: usize,
    carrying: usize,
    by_world: BTreeMap<String, usize>,
}

impl Census {
    /// Read one file. Not YAML, or not a contract, is the `validate` gate's business and is skipped here;
    /// the kind is read through the one parser `pv validate` uses.
    fn observe(&mut self, sigma: &Sigma, file: &Path) {
        let Some(doc) = std::fs::read_to_string(file)
            .ok()
            .and_then(|raw| serde_yaml::from_str::<serde_yaml::Value>(&raw).ok())
        else {
            return;
        };
        let Ok(contract) = parse_contract(file) else {
            return;
        };
        self.checked += 1;
        let is_kernel = contract.kind() == ContractKind::Kernel && !contract.is_registry();
        self.kernels += usize::from(is_kernel);
        let Some(v) = doc.get("metadata").and_then(|m| m.get("valid_under")) else {
            self.kernels_without += usize::from(is_kernel);
            return;
        };
        self.carrying += 1;
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let before = self.findings.len();
        check_valid_under(sigma, v, &stem, file, &mut self.findings);
        if self.findings.len() == before {
            let w = v
                .get("world")
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or(DEFAULT_WORLD);
            *self.by_world.entry(w.to_string()).or_default() += 1;
        }
    }
}

/// PV-ONT-016: the debt may fall, never rise. No baseline → nothing to compare against, reported only.
fn ratchet_finding(baseline: Option<usize>, without: usize) -> Option<LintFinding> {
    let b = baseline?;
    if without <= b {
        return None;
    }
    let mut f = LintFinding::new(
        "PV-ONT-016",
        RuleSeverity::Error,
        format!(
            "{BASELINE_KEY} rose {b} -> {without}: a kernel-kind contract without `metadata.valid_under` was added. The baseline in contracts/lint-baseline.json is shrink-only — give the new contract a world"
        ),
        "contracts/lint-baseline.json".to_string(),
    );
    f.contract_stem = Some("lint-baseline".to_string());
    Some(f)
}

fn finding(rule: &str, msg: String, stem: &str, file: &Path) -> LintFinding {
    let mut f = LintFinding::new(rule, RuleSeverity::Error, msg, file.display().to_string());
    f.contract_stem = Some(stem.to_string());
    f
}

/// Every rule a present `metadata.valid_under` must satisfy.
fn check_valid_under(
    sigma: &Sigma,
    v: &serde_yaml::Value,
    stem: &str,
    file: &Path,
    out: &mut Vec<LintFinding>,
) {
    let Some(map) = v.as_mapping() else {
        out.push(finding(
            "PV-ONT-013",
            "`metadata.valid_under` must be a mapping — `world:` (omitted = committed) and/or the qualifiers toolchain, host_class, backend, features".to_string(),
            stem,
            file,
        ));
        return;
    };
    for key in map.keys() {
        let name = key.as_str().unwrap_or("<non-string key>");
        if !VALID_UNDER_KEYS.contains(&name) {
            out.push(finding(
                "PV-ONT-013",
                format!(
                    "`metadata.valid_under.{name}` is not a valid_under key (closed set: {})",
                    VALID_UNDER_KEYS.join(", ")
                ),
                stem,
                file,
            ));
        }
    }
    if map.is_empty() {
        out.push(finding(
            "PV-ONT-013",
            "`metadata.valid_under` is empty — name a `world:` or a qualifier, or remove the key"
                .to_string(),
            stem,
            file,
        ));
        return;
    }
    match map.get("world").map(|w| w.as_str()) {
        None if !sigma.worlds.contains_key(DEFAULT_WORLD) => out.push(finding(
            "PV-ONT-014",
            format!("`metadata.valid_under` names no `world:` and Σ declares no `{DEFAULT_WORLD}` world to default to"),
            stem,
            file,
        )),
        None => {}
        Some(None) => out.push(finding(
            "PV-ONT-014",
            "`metadata.valid_under.world` must be a string naming a world in contracts/ontology.yaml".to_string(),
            stem,
            file,
        )),
        Some(Some(w)) if !sigma.worlds.contains_key(w) => out.push(finding(
            "PV-ONT-014",
            format!(
                "`metadata.valid_under.world: {w}` is not a world Σ declares (worlds: {}) — declare it in contracts/ontology.yaml",
                sigma.worlds.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
            stem,
            file,
        )),
        Some(Some(_)) => {}
    }
    if let Some(tc) = map.get("toolchain") {
        let ok = tc.as_mapping().is_some_and(|m| {
            !m.is_empty()
                && m.iter().all(|(k, v)| {
                    k.as_str().is_some_and(|s| !s.is_empty())
                        && v.as_str().is_some_and(|s| !s.is_empty())
                })
        });
        if !ok {
            out.push(finding(
                "PV-ONT-015",
                "`metadata.valid_under.toolchain` must be a non-empty map of tool -> version strings (e.g. `{ rust: \"1.93\" }`)".to_string(),
                stem,
                file,
            ));
        }
    }
    for key in ["host_class", "backend", "features"] {
        if let Some(list) = map.get(key) {
            let ok = list.as_sequence().is_some_and(|s| {
                !s.is_empty() && s.iter().all(|x| x.as_str().is_some_and(|s| !s.is_empty()))
            });
            if !ok {
                out.push(finding(
                    "PV-ONT-015",
                    format!("`metadata.valid_under.{key}` must be a non-empty list of non-empty strings"),
                    stem,
                    file,
                ));
            }
        }
    }
}

/// The top-level `contracts_without_valid_under` from `<contract_dir>/lint-baseline.json`, when recorded.
fn baseline_without_valid_under(contract_dir: &Path) -> Option<usize> {
    let raw = std::fs::read_to_string(contract_dir.join("lint-baseline.json")).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&raw).ok()?;
    doc.get(BASELINE_KEY)?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
}

#[cfg(test)]
#[path = "valid_under_gate_tests.rs"]
mod tests;
