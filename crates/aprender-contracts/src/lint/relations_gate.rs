//! ONT-001 §5 ONT-4 — the `relations` gate: a contract's `relations:` block is typed, resolved and acyclic.
//!
//! A `relations:` block relates one contract to others by a role Σ declares (`refines`, `supersedes`,
//! `contradicts`, `depends_on` — the four with `Contract` on both sides). Five corpus rules, all `reject:`
//! (exit 1), because the corpus is what is wrong:
//!
//! - PV-ONT-005 — a `relations:` key that is not a role Σ declares (the `sigma` gate's PV-ONT-002 says the same;
//!   this gate runs alone under `--gate relations`, so it says it too rather than assume the other ran);
//! - PV-ONT-006 — a role whose `domain` or `range` is not `Contract`: a relations block relates contracts, and a
//!   `binds`-shaped role (range `Code`) has no business in it;
//! - PV-ONT-007 — a value that is not a list of ids;
//! - PV-ONT-008 — an id that resolves to no `contracts/<id>.yaml` (dangling), named with contract, role and target;
//! - PV-ONT-009 — a cycle through a role Σ marks `acyclic` (`refines`, `supersedes`, `depends_on`), named as the
//!   path that closes it.
//!
//! `contradicts` is `symmetric` in Σ: the gate materializes the reverse edge, so `A contradicts B` is also
//! `B contradicts A` in the report, and a corpus that declares both directions declares nothing twice.
//!
//! **`metadata.depends_on` is read, counted and NEVER rewritten** (R-5: additive, no bulk rewrite). 188 contracts
//! carry it at `3409b29d`, 111 distinct targets, 98 of which resolve. Those edges are reported as
//! `legacy_depends_on` and their unresolved count as `legacy_unresolved_depends_on`, which the baseline ratchets
//! shrink-only (PV-ONT-010, the `formal_prose` pattern) — a legacy target that dangles is measured, not rejected,
//! because rejecting it would be a bulk migration by another name.
//!
//! **Zero typed relations is a decline, never a Pass** (R-2): [`RelationsOutcome::NoRelations`] → exit 2. The
//! real corpus carries typed relations on the ONT contracts themselves (`ont-relations-v1` depends on the Σ and
//! lattice contracts it reads and emits), which is why `pv lint contracts/ --gate relations` has focus nodes at
//! all; `supersedes` and `contradicts` have no true instance in the corpus today and are exercised on
//! `tests/fixtures/ont/relations-*` — as ONT-2b did for rules the corpus cannot fire.
//!
//! **Reads RAW YAML, not the parsed `Contract`** — `relations:` is a §4.2 additive key serde would drop.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use crate::ontology::sigma::{Sigma, SigmaError};

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult, Verdict};

/// What one `relations` run answers. Only [`RelationsOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum RelationsOutcome {
    /// No `ontology.yaml` under the corpus: no roles to type against, nothing measured.
    NoSigma,
    /// Σ does not parse, or does not satisfy its own integrity rules.
    Malformed(SigmaError),
    /// Σ was read, the corpus was walked, and not one contract carries a typed relation (R-2).
    NoRelations {
        contracts_checked: usize,
        legacy_depends_on: usize,
    },
    /// Σ was read and the corpus's typed relations were checked against it.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// One typed edge, as read from a contract's `relations:` block.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    from: String,
    role: String,
    to: String,
}

/// Run the gate over `contract_dir`, reading Σ from `<contract_dir>/ontology.yaml`.
#[must_use]
pub fn run_relations_gate(contract_dir: &Path) -> RelationsOutcome {
    let start = Instant::now();
    let sigma_path = contract_dir.join("ontology.yaml");
    let Ok(text) = std::fs::read_to_string(&sigma_path) else {
        return RelationsOutcome::NoSigma;
    };
    let sigma = match Sigma::from_yaml(&text) {
        Ok(s) => s,
        Err(e) => return RelationsOutcome::Malformed(e),
    };
    if let Err(e) = sigma.check_integrity() {
        return RelationsOutcome::Malformed(e);
    }

    let (docs, stems) = read_corpus(contract_dir, &sigma_path);

    // Pass 2: the typed edges, with the five rules; and the legacy count.
    let mut findings = Vec::new();
    let mut edges: BTreeSet<Edge> = BTreeSet::new();
    let mut contracts_with_relations = 0usize;
    let mut legacy_depends_on = 0usize;
    let mut legacy_unresolved = 0usize;
    for (stem, file, doc) in &docs {
        let (n_legacy, n_unresolved) = legacy_depends_on_edges(doc, &stems);
        legacy_depends_on += n_legacy;
        legacy_unresolved += n_unresolved;
        let Some(map) = doc.get("relations").and_then(serde_yaml::Value::as_mapping) else {
            continue;
        };
        contracts_with_relations += 1;
        findings.extend(check_block(&sigma, stem, file, map, &stems, &mut edges));
    }

    let relations_n = edges.len();
    if relations_n == 0 && findings.is_empty() {
        return RelationsOutcome::NoRelations {
            contracts_checked: docs.len(),
            legacy_depends_on,
        };
    }

    let (cycle_findings, cycles) = cycle_sweep(&sigma, &edges);
    findings.extend(cycle_findings);

    // The legacy ratchet (R-6): a RISE in unresolved legacy targets is a violation; a fall passes.
    if let Some(baseline) = baseline_legacy_unresolved(contract_dir) {
        if legacy_unresolved > baseline {
            findings.push(LintFinding::new(
                "PV-ONT-010",
                RuleSeverity::Error,
                format!(
                    "legacy_unresolved_depends_on rose {baseline} -> {legacy_unresolved}: a `metadata.depends_on` naming no contract was added. The baseline in contracts/lint-baseline.json is shrink-only"
                ),
                "contracts/lint-baseline.json".to_string(),
            ));
        }
    }
    let mut roles_used: BTreeMap<String, usize> = BTreeMap::new();
    for e in &edges {
        *roles_used.entry(e.role.clone()).or_default() += 1;
    }
    let violations = findings.len();
    let passed = violations == 0;
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let result = GateResult {
        name: "relations".into(),
        passed,
        skipped: false,
        verdict: Verdict::from_gate(passed, false),
        duration_ms: duration,
        detail: GateDetail::Validate {
            contracts: docs.len(),
            errors: violations,
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Relations {
            relations_n,
            contracts_with_relations,
            roles_used: roles_used.iter().map(|(r, n)| format!("{r}={n}")).collect(),
            cycles,
            legacy_depends_on,
            legacy_unresolved_depends_on: legacy_unresolved,
            violations,
        }),
    };
    RelationsOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

type Doc = (String, std::path::PathBuf, serde_yaml::Value);

/// Pass 1: every raw document the corpus parses, and every stem — so a target can be resolved against the set.
fn read_corpus(contract_dir: &Path, sigma_path: &Path) -> (Vec<Doc>, BTreeSet<String>) {
    let mut files = Vec::new();
    super::collect_yaml_files(contract_dir, &mut files);
    let mut docs: Vec<Doc> = Vec::new();
    let mut stems: BTreeSet<String> = BTreeSet::new();
    for file in &files {
        if file == sigma_path {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(file) else {
            continue;
        };
        let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&raw) else {
            continue; // a file that does not parse is the `validate` gate's business
        };
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        stems.insert(stem.clone());
        docs.push((stem, file.clone(), doc));
    }
    (docs, stems)
}

/// Rules 1–4 over one contract's `relations:` block; the well-formed edges (and their symmetric twins) go into `edges`.
fn check_block(
    sigma: &Sigma,
    stem: &str,
    file: &Path,
    map: &serde_yaml::Mapping,
    stems: &BTreeSet<String>,
    edges: &mut BTreeSet<Edge>,
) -> Vec<LintFinding> {
    let mut out = Vec::new();
    for (key, value) in map {
        let Some(role) = key.as_str() else { continue };
        let Some(decl) = sigma.roles.get(role) else {
            out.push(finding(
                "PV-ONT-005",
                format!("relation `{role}` is not a role Σ declares (contracts/ontology.yaml)"),
                file,
                stem,
            ));
            continue;
        };
        if decl.domain != "Contract" || decl.range != "Contract" {
            out.push(finding(
                "PV-ONT-006",
                format!(
                    "relation `{role}` relates {} to {} in Σ, not Contract to Contract — it does not belong in a `relations:` block",
                    decl.domain, decl.range
                ),
                file,
                stem,
            ));
            continue;
        }
        let Some(targets) = value.as_sequence() else {
            out.push(finding(
                "PV-ONT-007",
                format!("relation `{role}` must be a list of contract ids"),
                file,
                stem,
            ));
            continue;
        };
        out.extend(check_targets(
            stem,
            file,
            role,
            decl.symmetric,
            targets,
            stems,
            edges,
        ));
    }
    out
}

/// Rule 5: no cycle through a role Σ marks `acyclic`. Returns the findings and the cycles as `role: a -> b -> a`.
fn cycle_sweep(sigma: &Sigma, edges: &BTreeSet<Edge>) -> (Vec<LintFinding>, Vec<String>) {
    let mut findings = Vec::new();
    let mut cycles = Vec::new();
    for (role, decl) in &sigma.roles {
        if !decl.acyclic {
            continue;
        }
        let adj: BTreeMap<&str, Vec<&str>> =
            edges
                .iter()
                .filter(|e| e.role == *role)
                .fold(BTreeMap::new(), |mut m, e| {
                    m.entry(e.from.as_str()).or_default().push(e.to.as_str());
                    m
                });
        if let Some(path) = first_cycle(&adj) {
            let shown = path.join(" -> ");
            findings.push(LintFinding::new(
                "PV-ONT-009",
                RuleSeverity::Error,
                format!("`{role}` is acyclic in Σ, and the corpus closes a cycle: {shown}"),
                format!("contracts/{}.yaml", path[0]),
            ));
            cycles.push(format!("{role}: {shown}"));
        }
    }
    (findings, cycles)
}

/// One role's targets: each must be a string naming a stem the corpus has; a symmetric role adds the reverse edge.
fn check_targets(
    stem: &str,
    file: &Path,
    role: &str,
    symmetric: bool,
    targets: &[serde_yaml::Value],
    stems: &BTreeSet<String>,
    edges: &mut BTreeSet<Edge>,
) -> Vec<LintFinding> {
    let mut out = Vec::new();
    for t in targets {
        let Some(to) = t.as_str() else {
            out.push(finding(
                "PV-ONT-007",
                format!("relation `{role}` carries a non-string target"),
                file,
                stem,
            ));
            continue;
        };
        let to = normalize_id(to);
        if !stems.contains(&to) {
            out.push(finding(
                "PV-ONT-008",
                format!("`{stem}` {role} `{to}`, and no contracts/{to}.yaml exists — a dangling relation"),
                file,
                stem,
            ));
            continue;
        }
        edges.insert(Edge {
            from: stem.to_string(),
            role: role.to_string(),
            to: to.clone(),
        });
        if symmetric {
            edges.insert(Edge {
                from: to,
                role: role.to_string(),
                to: stem.to_string(),
            });
        }
    }
    out
}

fn finding(rule: &str, message: String, file: &Path, stem: &str) -> LintFinding {
    let mut f = LintFinding::new(
        rule,
        RuleSeverity::Error,
        message,
        file.display().to_string(),
    );
    f.contract_stem = Some(stem.to_string());
    f
}

/// A target may be written as a stem (`apr-cli-v1`) or as a path (`contracts/apr-cli-v1.yaml`); both name the stem.
fn normalize_id(raw: &str) -> String {
    let s = raw.trim();
    let s = s.strip_prefix("contracts/").unwrap_or(s);
    let s = s.strip_suffix(".yaml").unwrap_or(s);
    s.to_string()
}

/// `metadata.depends_on` — counted and resolved, never a finding. Returns (edges, unresolved).
fn legacy_depends_on_edges(doc: &serde_yaml::Value, stems: &BTreeSet<String>) -> (usize, usize) {
    let Some(list) = doc
        .get("metadata")
        .and_then(|m| m.get("depends_on"))
        .and_then(serde_yaml::Value::as_sequence)
    else {
        return (0, 0);
    };
    let mut n = 0usize;
    let mut unresolved = 0usize;
    for item in list {
        let Some(s) = item.as_str() else { continue };
        n += 1;
        if !stems.contains(&normalize_id(s)) {
            unresolved += 1;
        }
    }
    (n, unresolved)
}

/// `ont.legacy_unresolved_depends_on` from `<contract_dir>/lint-baseline.json`, when it is recorded.
fn baseline_legacy_unresolved(contract_dir: &Path) -> Option<usize> {
    let raw = std::fs::read_to_string(contract_dir.join("lint-baseline.json")).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&raw).ok()?;
    doc.get("ont")?
        .get("legacy_unresolved_depends_on")?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
}

/// DFS colouring: unvisited, on the stack, finished.
#[derive(Clone, Copy, PartialEq)]
enum Mark {
    White,
    Grey,
    Black,
}

/// The first cycle found by DFS over `adj`, as the closed path `[a, b, …, a]`; `None` when the graph is acyclic.
/// Deterministic: nodes and neighbours are visited in `BTreeMap` (byte) order, so two runs name the same cycle.
fn first_cycle<'a>(adj: &BTreeMap<&'a str, Vec<&'a str>>) -> Option<Vec<String>> {
    let mut mark: BTreeMap<&str, Mark> = adj.keys().map(|k| (*k, Mark::White)).collect();
    for k in adj.values().flatten() {
        mark.entry(k).or_insert(Mark::White);
    }
    let mut stack: Vec<&str> = Vec::new();
    let nodes: Vec<&str> = mark.keys().copied().collect();
    for n in nodes {
        if mark.get(n).copied() != Some(Mark::White) {
            continue;
        }
        if let Some(c) = visit(n, adj, &mut mark, &mut stack) {
            return Some(c);
        }
    }
    None
}

/// Grey-marks the DFS stack; a grey neighbour closes a cycle, returned as the stack from it plus itself.
fn visit<'a>(
    node: &'a str,
    adj: &BTreeMap<&'a str, Vec<&'a str>>,
    mark: &mut BTreeMap<&'a str, Mark>,
    stack: &mut Vec<&'a str>,
) -> Option<Vec<String>> {
    mark.insert(node, Mark::Grey);
    stack.push(node);
    for &n in adj.get(node).map(Vec::as_slice).unwrap_or_default() {
        match mark.get(n).copied().unwrap_or(Mark::White) {
            Mark::Grey => {
                let at = stack.iter().position(|s| *s == n).unwrap_or(0);
                let mut path: Vec<String> = stack[at..].iter().map(|s| (*s).to_string()).collect();
                path.push(n.to_string());
                return Some(path);
            }
            Mark::White => {
                if let Some(c) = visit(n, adj, mark, stack) {
                    return Some(c);
                }
            }
            Mark::Black => {}
        }
    }
    stack.pop();
    mark.insert(node, Mark::Black);
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont")
            .join(name)
    }

    fn ran(name: &str) -> (Box<GateResult>, Vec<LintFinding>) {
        match run_relations_gate(&fixture(name)) {
            RelationsOutcome::Ran { result, findings } => (result, findings),
            other => panic!("expected Ran for {name}, got {other:?}"),
        }
    }

    fn rules(findings: &[LintFinding]) -> Vec<&str> {
        findings.iter().map(|f| f.rule_id.as_str()).collect()
    }

    #[test]
    fn four_relations_are_accepted_and_contradicts_is_symmetric() {
        let (result, findings) = ran("relations-ok");
        assert!(result.passed, "{findings:?}");
        assert_eq!(result.verdict, Verdict::Pass);
        match result.extra {
            Some(GateExtra::Relations {
                relations_n,
                contracts_with_relations,
                ref roles_used,
                ..
            }) => {
                // a refines b; b supersedes c; a depends_on c; a contradicts d → and d contradicts a, materialized
                assert_eq!(relations_n, 5, "{roles_used:?}");
                assert_eq!(contracts_with_relations, 2);
                assert!(
                    roles_used.contains(&"contradicts=2".to_string()),
                    "{roles_used:?}"
                );
            }
            other => panic!("expected Relations extra, got {other:?}"),
        }
    }

    #[test]
    fn a_dangling_id_is_rejected_naming_contract_role_and_target() {
        let (result, findings) = ran("relations-dangling");
        assert!(!result.passed);
        assert_eq!(rules(&findings), vec!["PV-ONT-008"]);
        assert!(
            findings[0].message.contains("`a` depends_on `ghost`"),
            "{}",
            findings[0].message
        );
    }

    #[test]
    fn a_cycle_through_an_acyclic_role_is_rejected_naming_the_path() {
        let (result, findings) = ran("relations-cycle");
        assert!(!result.passed);
        assert_eq!(rules(&findings), vec!["PV-ONT-009"]);
        assert!(
            findings[0].message.contains("a -> b -> c -> a"),
            "{}",
            findings[0].message
        );
    }

    #[test]
    fn a_role_outside_contract_to_contract_is_rejected() {
        let (result, findings) = ran("relations-domain");
        assert!(!result.passed);
        assert_eq!(rules(&findings), vec!["PV-ONT-006"]);
        assert!(
            findings[0].message.contains("`binds`"),
            "{}",
            findings[0].message
        );
    }

    #[test]
    fn an_undeclared_role_and_a_non_list_are_rejected() {
        let (result, findings) = ran("relations-malformed");
        assert!(!result.passed);
        let mut got = rules(&findings);
        got.sort_unstable();
        assert_eq!(got, vec!["PV-ONT-005", "PV-ONT-007"]);
    }

    #[test]
    fn zero_typed_relations_is_a_decline_not_a_pass() {
        match run_relations_gate(&fixture("sigma-ok")) {
            RelationsOutcome::NoRelations {
                contracts_checked, ..
            } => assert!(contracts_checked > 0),
            other => panic!("expected NoRelations (R-2), got {other:?}"),
        }
    }

    #[test]
    fn legacy_depends_on_is_counted_never_rejected_and_ratcheted() {
        // the fixture carries metadata.depends_on with one resolving and one dangling target, and a baseline of 1
        let (result, findings) = ran("relations-legacy");
        assert!(result.passed, "{findings:?}");
        match result.extra {
            Some(GateExtra::Relations {
                legacy_depends_on,
                legacy_unresolved_depends_on,
                ..
            }) => {
                assert_eq!(legacy_depends_on, 2);
                assert_eq!(legacy_unresolved_depends_on, 1);
            }
            other => panic!("expected Relations extra, got {other:?}"),
        }
        // and a rise above the baseline is a violation
        let (result, findings) = ran("relations-legacy-rise");
        assert!(!result.passed);
        assert_eq!(rules(&findings), vec!["PV-ONT-010"]);
    }

    #[test]
    fn no_sigma_is_not_a_corpus_verdict() {
        assert!(matches!(
            run_relations_gate(&fixture("sigma-absent")),
            RelationsOutcome::NoSigma
        ));
    }

    #[test]
    fn first_cycle_is_deterministic_and_names_the_closing_path() {
        let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        adj.insert("x", vec!["y"]);
        adj.insert("y", vec!["z"]);
        adj.insert("z", vec!["y"]);
        assert_eq!(
            first_cycle(&adj),
            Some(vec!["y".into(), "z".into(), "y".into()])
        );
        let mut dag: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        dag.insert("x", vec!["y", "z"]);
        dag.insert("y", vec!["z"]);
        assert_eq!(first_cycle(&dag), None);
    }
}
