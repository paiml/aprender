//! ONT-001 §5 ONT-8 — the `evidence` gate: one evidence block, PROV-O names, one L-enum, every entity type.
//!
//! A claim says how it was produced. Until this gate, a contract could say it in any words it liked — `author:`,
//! `tool:`, `level: L0`, `proved_by:` — and every one of them read the same to `pv`: not at all. One block,
//! one vocabulary:
//!
//! ```yaml
//! evidence:
//!   level: L2                                   # ProofLevel — the ONE L-enum (PVL-001 EV-3)
//!   mark: C                                     # optional: V verified · C cited · A directive · U unknown
//!   provenance:
//!     wasGeneratedBy: { command: "pv extract README.md", git_sha: <40-hex> }
//!     wasAttributedTo: pv                       # a Σ `agents` entry
//!     generatedAtTime: "2026-09-24T00:00:00Z"
//! ```
//!
//! Rules, all `reject:` (exit 1) because the corpus is what is wrong:
//!
//! - PV-ONT-017 — a key outside the closed PROV-O set, at any of the three levels (`evidence`, `provenance`,
//!   `wasGeneratedBy`), or one of those levels is not a mapping. F-16: `author:` is RED.
//! - PV-ONT-018 — `level` is missing or is not a [`ProofLevel`]. It is parsed THROUGH the enum, never matched
//!   against a list of strings kept here, so `levels_source` is `enum` by construction — a second list is the
//!   drift EV-3 removed (`L0` is not a level).
//! - PV-ONT-019 — `mark` is not one of V C A U, or a `[C]`/`[V]` claim carries no `wasGeneratedBy.command`.
//! - PV-ONT-020 — a `[V]` claim's `wasGeneratedBy.git_sha` does not bind: missing, not 40 lowercase hex, not
//!   equal to `census.git_sha` when the census records one, and — because the census deliberately records
//!   `git_sha: null` (a commit cannot name its own sha; `id_set_sha256` is the content address) — otherwise
//!   not a commit the corpus's own repository holds. A sha git could not LOOK for (no git, not a repository,
//!   or a shallow clone missing it) is not a finding: it is `Unknown{ToolAbsent}`, because "did not look" is
//!   not "found nothing".
//! - PV-ONT-021 — `provenance` or `wasAttributedTo` is missing, or `wasAttributedTo` is not a Σ `agents` entry;
//!   `generatedAtTime`, when present, is not a non-empty string.
//!
//! **No rule reads `entity.type`** (R-17): a README, a model file and a code contract meet the same rules. The
//! type is read only to COUNT the kinds of entity the gate judged (`entity_types_checked`), which is how the
//! row's probe proves the rules were applied across types rather than asserted to be. Non-verdict answers: no Σ
//! → decline, malformed Σ → error, and no evidence block anywhere measured nothing → decline (R-2). Computed in
//! every `pv lint` run (gate 16, R-8) and armed per repo.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use crate::ontology::sigma::{Sigma, SigmaError};
use crate::ontology::verdict::{Reason, Verdict};
use crate::proof_status::ProofLevel;

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult};

/// The keys `evidence` may carry.
pub const EVIDENCE_KEYS: [&str; 3] = ["level", "mark", "provenance"];
/// The PROV-O keys `evidence.provenance` may carry.
pub const PROVENANCE_KEYS: [&str; 3] = ["wasGeneratedBy", "wasAttributedTo", "generatedAtTime"];
/// The keys `provenance.wasGeneratedBy` may carry.
pub const GENERATED_BY_KEYS: [&str; 2] = ["command", "git_sha"];
/// The provenance marks (ONT-001 §0).
pub const MARKS: [&str; 4] = ["V", "C", "A", "U"];
/// What `levels_source` reports: the levels are read through [`ProofLevel`], not a list kept here.
pub const LEVELS_SOURCE: &str = "enum";

/// What one `evidence` run answers. Only [`EvidenceOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum EvidenceOutcome {
    /// No `ontology.yaml` under the corpus: there is no `agents` set to attribute to.
    NoSigma,
    /// Σ does not parse, or does not satisfy its own integrity rules.
    Malformed(SigmaError),
    /// No contract carries an `evidence` block: nothing was measured.
    NoEvidence {
        /// Contract files read.
        contracts_checked: usize,
    },
    /// Σ was read and the corpus was checked against it.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// A `[V]` claim's sha, waiting on the repository to say whether it holds it.
struct PendingSha {
    sha: String,
    stem: String,
    file: String,
}

/// Run the gate over `contract_dir`, reading Σ from `<contract_dir>/ontology.yaml`.
#[must_use]
pub fn run_evidence_gate(contract_dir: &Path) -> EvidenceOutcome {
    let start = Instant::now();
    let sigma_path = contract_dir.join("ontology.yaml");
    let Ok(text) = std::fs::read_to_string(&sigma_path) else {
        return EvidenceOutcome::NoSigma;
    };
    let sigma = match Sigma::from_yaml(&text) {
        Ok(s) => s,
        Err(e) => return EvidenceOutcome::Malformed(e),
    };
    if let Err(e) = sigma.check_integrity() {
        return EvidenceOutcome::Malformed(e);
    }
    let census_sha = census_git_sha(contract_dir);

    let mut c = Census::default();
    let mut files = Vec::new();
    super::collect_yaml_files(contract_dir, &mut files);
    files.sort();
    for file in files.iter().filter(|f| **f != sigma_path) {
        c.observe(&sigma, census_sha.as_deref(), file);
    }
    if c.carrying == 0 {
        return EvidenceOutcome::NoEvidence {
            contracts_checked: c.checked,
        };
    }
    let unresolved = resolve_pending(
        contract_dir,
        std::mem::take(&mut c.pending),
        &mut c.findings,
    );
    let Census {
        findings,
        checked,
        carrying,
        entity_types,
        by_level,
        ..
    } = c;

    let violations = findings.len();
    let passed = violations == 0 && unresolved == 0;
    let verdict = if violations > 0 {
        Verdict::Fail
    } else if unresolved > 0 {
        Verdict::Unknown(Reason::ToolAbsent)
    } else {
        Verdict::Pass
    };
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let result = GateResult {
        name: "evidence".into(),
        passed,
        skipped: false,
        verdict,
        duration_ms: duration,
        // `GateDetail` is FROZEN at the 0.3.1 variants (see `GateExtra`); the shape is borrowed as `sigma` and
        // `valid-under` borrow it, and the real payload rides in `GateExtra`.
        detail: GateDetail::Validate {
            contracts: checked,
            errors: violations,
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Evidence {
            contracts_checked: checked,
            contracts_with_evidence: carrying,
            entity_types_checked: entity_types.len(),
            entity_types: entity_types.into_iter().collect(),
            levels_source: LEVELS_SOURCE.to_string(),
            by_level: by_level.iter().map(|(l, n)| format!("{l}={n}")).collect(),
            census_git_sha: census_sha,
            unresolved_git_shas: unresolved,
            violations,
        }),
    };
    EvidenceOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

/// What one pass over the corpus counted.
#[derive(Default)]
struct Census {
    findings: Vec<LintFinding>,
    pending: Vec<PendingSha>,
    checked: usize,
    carrying: usize,
    entity_types: BTreeSet<String>,
    by_level: BTreeMap<String, usize>,
}

impl Census {
    /// Read one file. Not YAML, or not a mapping, is the `validate` gate's business and is skipped here.
    fn observe(&mut self, sigma: &Sigma, census_sha: Option<&str>, file: &Path) {
        let Some(doc) = std::fs::read_to_string(file)
            .ok()
            .and_then(|raw| serde_yaml::from_str::<serde_yaml::Value>(&raw).ok())
            .filter(serde_yaml::Value::is_mapping)
        else {
            return;
        };
        self.checked += 1;
        let Some(ev) = doc.get("evidence") else {
            return;
        };
        self.carrying += 1;
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let before = self.findings.len();
        let pending = check_evidence(sigma, census_sha, ev, &stem, file, &mut self.findings);
        self.pending.extend(pending);
        // Counted, never branched on (R-17): the type decides nothing above.
        if let Some(t) = doc
            .get("entity")
            .and_then(|e| e.get("type"))
            .and_then(serde_yaml::Value::as_str)
        {
            self.entity_types.insert(t.to_string());
        }
        if self.findings.len() == before {
            if let Some(l) = ev.get("level").and_then(serde_yaml::Value::as_str) {
                *self.by_level.entry(l.to_string()).or_default() += 1;
            }
        }
    }
}

fn finding(rule: &str, msg: String, stem: &str, file: &Path) -> LintFinding {
    let mut f = LintFinding::new(rule, RuleSeverity::Error, msg, file.display().to_string());
    f.contract_stem = Some(stem.to_string());
    f
}

/// PV-ONT-017: `v` at `path` must be a mapping whose keys are all in `allowed`. Returns the mapping when it is one.
fn closed_mapping<'a>(
    v: &'a serde_yaml::Value,
    path: &str,
    allowed: &[&str],
    stem: &str,
    file: &Path,
    out: &mut Vec<LintFinding>,
) -> Option<&'a serde_yaml::Mapping> {
    let Some(map) = v.as_mapping() else {
        out.push(finding(
            "PV-ONT-017",
            format!("`{path}` must be a mapping of {}", allowed.join(", ")),
            stem,
            file,
        ));
        return None;
    };
    for key in map.keys() {
        let name = key.as_str().unwrap_or("<non-string key>");
        if !allowed.contains(&name) {
            out.push(finding(
                "PV-ONT-017",
                format!(
                    "`{path}.{name}` is not a PROV-O evidence key (closed set: {})",
                    allowed.join(", ")
                ),
                stem,
                file,
            ));
        }
    }
    Some(map)
}

/// Every rule a present `evidence` block must satisfy. Returns the `[V]` sha still to be resolved in git, if any.
fn check_evidence(
    sigma: &Sigma,
    census_sha: Option<&str>,
    ev: &serde_yaml::Value,
    stem: &str,
    file: &Path,
    out: &mut Vec<LintFinding>,
) -> Option<PendingSha> {
    let map = closed_mapping(ev, "evidence", &EVIDENCE_KEYS, stem, file, out)?;
    check_level(map.get("level"), stem, file, out);
    let mark = check_mark(map.get("mark"), stem, file, out);
    let Some(prov) = map.get("provenance") else {
        out.push(finding(
            "PV-ONT-021",
            "`evidence.provenance` is missing — an evidence block names who produced the claim (`wasAttributedTo`)".to_string(),
            stem,
            file,
        ));
        return None;
    };
    let prov = closed_mapping(
        prov,
        "evidence.provenance",
        &PROVENANCE_KEYS,
        stem,
        file,
        out,
    )?;
    check_attribution(sigma, prov, stem, file, out);
    let generated_by = prov.get("wasGeneratedBy").and_then(|g| {
        closed_mapping(
            g,
            "evidence.provenance.wasGeneratedBy",
            &GENERATED_BY_KEYS,
            stem,
            file,
            out,
        )
    });
    let command = generated_by
        .and_then(|g| g.get("command"))
        .and_then(serde_yaml::Value::as_str)
        .filter(|s| !s.trim().is_empty());
    if matches!(mark, Some("C" | "V")) && command.is_none() {
        out.push(finding(
            "PV-ONT-019",
            format!("a `[{}]` claim must name the command that produced it (`evidence.provenance.wasGeneratedBy.command`)", mark.unwrap_or_default()),
            stem,
            file,
        ));
    }
    if mark != Some("V") {
        return None;
    }
    let sha = generated_by
        .and_then(|g| g.get("git_sha"))
        .and_then(serde_yaml::Value::as_str);
    check_verified_sha(census_sha, sha, stem, file, out)
}

/// PV-ONT-018: the level is parsed through [`ProofLevel`] — the one definition — or it is not a level.
fn check_level(
    level: Option<&serde_yaml::Value>,
    stem: &str,
    file: &Path,
    out: &mut Vec<LintFinding>,
) {
    let ok = level.is_some_and(|l| serde_yaml::from_value::<ProofLevel>(l.clone()).is_ok());
    if !ok {
        let shown = level.map_or_else(|| "missing".to_string(), |l| format!("{l:?}"));
        out.push(finding(
            "PV-ONT-018",
            format!("`evidence.level` ({shown}) is not a ProofLevel — L1, L2, L3, L4 or L5"),
            stem,
            file,
        ));
    }
}

/// PV-ONT-019: the mark, when present, is one of V C A U. Returns it when it is.
fn check_mark<'a>(
    mark: Option<&'a serde_yaml::Value>,
    stem: &str,
    file: &Path,
    out: &mut Vec<LintFinding>,
) -> Option<&'a str> {
    let m = mark?;
    match m.as_str() {
        Some(s) if MARKS.contains(&s) => Some(s),
        _ => {
            out.push(finding(
                "PV-ONT-019",
                format!("`evidence.mark` ({m:?}) is not a provenance mark — V, C, A or U"),
                stem,
                file,
            ));
            None
        }
    }
}

/// PV-ONT-021: attributed to a Σ agent; `generatedAtTime`, when given, is a non-empty string.
fn check_attribution(
    sigma: &Sigma,
    prov: &serde_yaml::Mapping,
    stem: &str,
    file: &Path,
    out: &mut Vec<LintFinding>,
) {
    match prov.get("wasAttributedTo").and_then(serde_yaml::Value::as_str) {
        Some(a) if sigma.agents.iter().any(|x| x == a) => {}
        other => out.push(finding(
            "PV-ONT-021",
            format!(
                "`evidence.provenance.wasAttributedTo` ({}) is not an agent Σ declares (agents: {}) — declare it in contracts/ontology.yaml",
                other.unwrap_or("missing or not a string"),
                sigma.agents.join(", ")
            ),
            stem,
            file,
        )),
    }
    if let Some(t) = prov.get("generatedAtTime") {
        if t.as_str().is_none_or(|s| s.trim().is_empty()) {
            out.push(finding(
                "PV-ONT-021",
                "`evidence.provenance.generatedAtTime` must be a non-empty timestamp string"
                    .to_string(),
                stem,
                file,
            ));
        }
    }
}

fn is_full_sha(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// PV-ONT-020, the parts decidable without git. A census that records a sha is the binding, as the row wrote it;
/// a null census (the 2026-09-16 ruling) leaves the repository as the binding, resolved in [`resolve_pending`].
fn check_verified_sha(
    census_sha: Option<&str>,
    sha: Option<&str>,
    stem: &str,
    file: &Path,
    out: &mut Vec<LintFinding>,
) -> Option<PendingSha> {
    let Some(sha) = sha.filter(|s| is_full_sha(s)) else {
        out.push(finding(
            "PV-ONT-020",
            "a `[V]` claim must pin `evidence.provenance.wasGeneratedBy.git_sha` to a full 40-hex commit sha".to_string(),
            stem,
            file,
        ));
        return None;
    };
    match census_sha {
        Some(c) if c != sha => {
            out.push(finding(
                "PV-ONT-020",
                format!("a `[V]` claim's git_sha {sha} is not the census's git_sha {c}"),
                stem,
                file,
            ));
            None
        }
        Some(_) => None,
        None => Some(PendingSha {
            sha: sha.to_string(),
            stem: stem.to_string(),
            file: file.display().to_string(),
        }),
    }
}

/// `git_sha` from `<contract_dir>/census.json`, when it records one.
fn census_git_sha(contract_dir: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(contract_dir.join("census.json")).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&raw).ok()?;
    doc.get("git_sha")?.as_str().map(str::to_string)
}

/// Ask the corpus's repository whether it holds each `[V]` sha. A sha it does not hold is PV-ONT-020; a sha git
/// could not look for is counted and returned — the caller makes that `Unknown{ToolAbsent}`, never `Pass`.
fn resolve_pending(
    contract_dir: &Path,
    pending: Vec<PendingSha>,
    out: &mut Vec<LintFinding>,
) -> usize {
    if pending.is_empty() {
        return 0;
    }
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(contract_dir)
            .args(args)
            .output()
            .ok()
    };
    let shallow = match git(&["rev-parse", "--is-shallow-repository"]) {
        Some(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim() == "true",
        _ => return pending.len(),
    };
    let mut unresolved = 0;
    for p in pending {
        let spec = format!("{}^{{commit}}", p.sha);
        match git(&["cat-file", "-e", &spec]) {
            Some(o) if o.status.success() => {}
            Some(_) if !shallow => {
                let mut f = LintFinding::new(
                    "PV-ONT-020",
                    RuleSeverity::Error,
                    format!(
                        "a `[V]` claim's git_sha {} is not a commit this repository holds",
                        p.sha
                    ),
                    p.file,
                );
                f.contract_stem = Some(p.stem);
                out.push(f);
            }
            _ => unresolved += 1,
        }
    }
    unresolved
}

#[cfg(test)]
#[path = "evidence_gate_tests.rs"]
mod tests;
