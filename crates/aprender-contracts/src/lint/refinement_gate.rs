//! ONT-001 ONT-3b (#4073) — the `refinement` gate: every `models:` entry in `formalization.yaml` names a Rust item
//! the workspace resolves (through ONT-3a's `syn` walk), and two counts that must never get worse are checked HEAD
//! against BASE in the same run, by the same scanner: the theorem-bearing Lean modules with no L4 model
//! (`unrefined`), and the contract-bound theorems outside the root's import cone (`orphaned_roots`). See
//! [`crate::discharge::refinement`] for what a model earns.
//!
//! There is no stored limit. The operator's ruling (2026-09-27, relayed by the cop): "counts that must never get
//! worse are checked head against base, not against a number committed to the repo". BASE is
//! merge-base(HEAD, origin/main), else the origin/main tip (`lib_baseline_ratchet.sh`'s order); its Lean tree and
//! contracts are read out of git (`git archive`), never out of the work tree.
//!
//! Rules (`reject:`, exit 1):
//!
//! - PV-ONT-031 — a `model_of` does not resolve: a ghost formalization, named;
//! - PV-ONT-032 — a `models:` entry is malformed (no module/model_of/evidence, unknown `relation.kind`, a module
//!   modelled twice) or names a module outside the root's import cone;
//! - PV-ONT-033 — a module is unrefined at HEAD and was not at BASE (a new unproven module, or one that lost its
//!   model), named;
//! - PV-ONT-034 — a contract-bound theorem is outside the cone at HEAD and was not at BASE (ORPHANED-ROOT), named.
//!
//! Declines (exit 2): no Lean tree or `formalization.yaml` beside the corpus (the fixture corpora), no BASE to
//! compare with, or the resolver's positive control did not fire.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::Serialize;

use crate::discharge::{self, refinement, summary, Tree};
use crate::ontology::extract::code::positive_control;
use crate::ontology::verdict::Verdict;

use super::finding::LintFinding;
use super::ratchet_gates::RatchetOutcome;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult};

/// The gate's name, under `--gate` and in a full run.
pub const GATE: &str = "refinement";

/// The Lean tree, relative to the corpus's parent.
pub const LEAN_DIR: &str = "crates/aprender-contracts-staging/lean";

/// How many modules or roots a finding names before it says "and N more".
const NAMED: usize = 12;

/// The counters `--gate refinement` reports, flattened into [`GateExtra::Refinement`]'s JSON.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RefinementCounters {
    /// Well-formed `models:` entries.
    pub models: usize,
    pub resolved: usize,
    /// Models granting L4 (`extraction`/`simulation`, resolving, in the cone).
    pub l4_models: usize,
    /// Models granting L3 (`test_witnessed`, resolving, in the cone).
    pub l3_models: usize,
    /// `model_of` that does not resolve (PV-ONT-031).
    pub ghosts: usize,
    /// HEAD: modules in the root's import cone with at least one theorem.
    pub theorem_modules: usize,
    /// HEAD: of those, the ones no L4 model covers.
    pub unrefined: usize,
    /// HEAD: contract-bound theorems outside the cone.
    pub orphaned_roots: usize,
    /// What HEAD was compared with (`merge-base(HEAD, origin/main) <sha>`).
    pub base: String,
    pub base_theorem_modules: usize,
    pub base_unrefined: usize,
    /// Same measured value as `base_unrefined`, under the name the ONT-3b spec probe reads. Measured at the
    /// merge-base in this run; never a stored count.
    pub unrefined_baseline: usize,
    pub base_orphaned_roots: usize,
    /// Unrefined at HEAD, not at BASE (PV-ONT-033).
    pub new_unrefined: usize,
    /// Orphaned at HEAD, not at BASE (PV-ONT-034).
    pub new_orphaned_roots: usize,
    /// `fired` — the resolver found a present item and refused an absent one this run.
    pub pc_resolver: String,
    pub violations: usize,
}

/// One side (HEAD or BASE), measured from source by one scanner.
#[derive(Debug, Clone, Default)]
pub struct Side {
    /// Modules in the root's import cone (paths relative to the lean dir).
    pub cone: BTreeSet<String>,
    /// Theorem-bearing modules in the cone, with their theorems.
    pub theorem_modules: BTreeMap<String, Vec<String>>,
    /// Theorem-bearing modules no L4 model covers.
    pub unrefined: BTreeSet<String>,
    /// Contract-bound theorems (fully qualified) whose module is outside the cone.
    pub orphaned: BTreeSet<String>,
    pub judged: Vec<refinement::Judged>,
    pub malformed: Vec<String>,
}

/// Measure the tree at `lean_dir` against the contracts at `contract_dir`, with `rec`'s models. `Err` is a decline.
pub fn measure(
    lean_dir: &Path,
    contract_dir: &Path,
    rec: refinement::Record,
    resolve: impl FnMut(&str) -> Result<(), String>,
) -> Result<Side, String> {
    let tree = Tree::load(lean_dir)?;
    let modules = summary::modules(&tree);
    let cone: BTreeSet<String> = modules.iter().map(|m| m.path.clone()).collect();
    let theorem_modules: BTreeMap<String, Vec<String>> = modules
        .into_iter()
        .filter(|m| !m.theorems.is_empty())
        .map(|m| (m.path, m.theorems))
        .collect();
    let judged = refinement::judge(&rec.models, &cone, resolve);
    let l4 = refinement::l4_modules(&judged);
    let unrefined = refinement::unrefined(&theorem_modules, &l4)
        .into_iter()
        .map(str::to_string)
        .collect();
    let cone_mods = tree.cone();
    let orphaned = discharge::bind(&tree, contract_dir)
        .roots
        .into_iter()
        .filter(|r| !cone_mods.contains(&r.module))
        .map(|r| r.fqn)
        .collect();
    Ok(Side {
        cone,
        theorem_modules,
        unrefined,
        orphaned,
        judged,
        malformed: rec.malformed,
    })
}

fn finding(rule: &str, msg: String) -> LintFinding {
    LintFinding::new(
        rule,
        RuleSeverity::Error,
        msg,
        format!("{LEAN_DIR}/formalization.yaml"),
    )
}

/// `a, b, c and N more`.
fn named(items: &[&String]) -> String {
    let shown: Vec<&str> = items.iter().take(NAMED).map(|s| s.as_str()).collect();
    let rest = items.len().saturating_sub(NAMED);
    if rest == 0 {
        shown.join(", ")
    } else {
        format!("{} and {rest} more", shown.join(", "))
    }
}

/// Run the gate over `contract_dir`: HEAD is the work tree, BASE comes out of git.
#[must_use]
pub fn run_refinement_gate(contract_dir: &Path) -> RatchetOutcome {
    let root = crate::ontology::extract::repo_root(contract_dir);
    if !root.join(LEAN_DIR).join("formalization.yaml").is_file() {
        return RatchetOutcome::Declined(format!(
            "no {LEAN_DIR}/formalization.yaml beside the corpus"
        ));
    }
    let base = match BaseTree::extract(&root, contract_dir) {
        Ok(b) => b,
        Err(e) => return RatchetOutcome::Declined(format!("no BASE to compare with: {e}")),
    };
    run_with(
        &root.join(LEAN_DIR),
        contract_dir,
        &base.lean,
        &base.contracts,
        &base.label,
        refinement::workspace_resolver(&root),
    )
}

/// The gate: HEAD's tree at `lean_dir` + `contract_dir` against BASE's at `base_lean` + `base_contracts`, one
/// scanner and one resolver for both.
pub fn run_with(
    lean_dir: &Path,
    contract_dir: &Path,
    base_lean: &Path,
    base_contracts: &Path,
    base_label: &str,
    mut resolve: impl FnMut(&str) -> Result<(), String>,
) -> RatchetOutcome {
    let start = Instant::now();
    if !positive_control() {
        return RatchetOutcome::Declined(
            "positive control pc_resolver did not fire: the resolver cannot tell a present item from an absent one"
                .into(),
        );
    }
    let head = match load_head_side(lean_dir, contract_dir, &mut resolve) {
        Ok(s) => s,
        Err(e) => return RatchetOutcome::Declined(format!("HEAD: {e}")),
    };
    let base = match load_base_side(base_lean, base_contracts, &mut resolve) {
        Ok(s) => s,
        Err(e) => return RatchetOutcome::Declined(format!("BASE {base_label}: {e}")),
    };

    let mut findings: Vec<LintFinding> = head
        .malformed
        .iter()
        .map(|m| finding("PV-ONT-032", m.clone()))
        .collect();
    let mut c = RefinementCounters {
        models: head.judged.len(),
        theorem_modules: head.theorem_modules.len(),
        unrefined: head.unrefined.len(),
        orphaned_roots: head.orphaned.len(),
        base: base_label.to_string(),
        base_theorem_modules: base.theorem_modules.len(),
        base_unrefined: base.unrefined.len(),
        unrefined_baseline: base.unrefined.len(),
        base_orphaned_roots: base.orphaned.len(),
        pc_resolver: crate::ontology::witness::FIRED.to_string(),
        ..RefinementCounters::default()
    };
    judge_findings(&head.judged, &mut findings, &mut c);
    diff_findings(&head, &base, base_label, &mut findings, &mut c);
    c.violations = findings.len();
    build_result(c, findings, start)
}

/// Load and measure HEAD's side: `Err` is a decline reason (no "HEAD:" prefix yet).
fn load_head_side(
    lean_dir: &Path,
    contract_dir: &Path,
    resolve: &mut impl FnMut(&str) -> Result<(), String>,
) -> Result<Side, String> {
    refinement::load(lean_dir)
        .map_err(|e| format!("no formalization.yaml: {e}"))
        .and_then(|rec| measure(lean_dir, contract_dir, rec, resolve))
}

/// Load and measure BASE's side. A BASE with no `formalization.yaml` yet has no models: every
/// theorem-bearing module there is unrefined.
fn load_base_side(
    base_lean: &Path,
    base_contracts: &Path,
    resolve: &mut impl FnMut(&str) -> Result<(), String>,
) -> Result<Side, String> {
    let rec = if base_lean.join("formalization.yaml").exists() {
        refinement::load(base_lean)?
    } else {
        refinement::Record::default()
    };
    measure(base_lean, base_contracts, rec, resolve)
}

/// Score each judged model into `c`'s counters, and raise PV-ONT-031/032 for ghosts and out-of-cone models.
fn judge_findings(
    judged: &[refinement::Judged],
    findings: &mut Vec<LintFinding>,
    c: &mut RefinementCounters,
) {
    for j in judged {
        let m = &j.model;
        match &j.resolved {
            Ok(()) => c.resolved += 1,
            Err(why) => {
                c.ghosts += 1;
                findings.push(finding(
                    "PV-ONT-031",
                    format!(
                        "{}: model_of `{}` does not resolve (ghost formalization): {why}",
                        m.module, m.model_of
                    ),
                ));
            }
        }
        if !j.in_tree {
            findings.push(finding(
                "PV-ONT-032",
                format!("{}: not a module in the root's import cone", m.module),
            ));
        }
        match j.level() {
            Some(4) => c.l4_models += 1,
            Some(3) => c.l3_models += 1,
            _ => {}
        }
    }
}

/// Raise PV-ONT-033/034 for anything unrefined or orphaned at HEAD that BASE did not have, and count them.
fn diff_findings(
    head: &Side,
    base: &Side,
    base_label: &str,
    findings: &mut Vec<LintFinding>,
    c: &mut RefinementCounters,
) {
    let new_unrefined: Vec<&String> = head.unrefined.difference(&base.unrefined).collect();
    c.new_unrefined = new_unrefined.len();
    if !new_unrefined.is_empty() {
        findings.push(finding(
            "PV-ONT-033",
            format!(
                "{} unrefined at HEAD vs {} at BASE {base_label}: {} module(s) with no L4 model that BASE did not have: {}",
                c.unrefined,
                c.base_unrefined,
                new_unrefined.len(),
                named(&new_unrefined)
            ),
        ));
    }
    let new_orphaned: Vec<&String> = head.orphaned.difference(&base.orphaned).collect();
    c.new_orphaned_roots = new_orphaned.len();
    if !new_orphaned.is_empty() {
        findings.push(finding(
            "PV-ONT-034",
            format!(
                "{} ORPHANED-ROOT at HEAD vs {} at BASE {base_label}: {} contract-bound theorem(s) outside the root's import cone that BASE did not have: {}",
                c.orphaned_roots,
                c.base_orphaned_roots,
                new_orphaned.len(),
                named(&new_orphaned)
            ),
        ));
    }
}

/// Assemble the final [`RatchetOutcome`] from the counters and findings collected for both sides.
fn build_result(
    c: RefinementCounters,
    findings: Vec<LintFinding>,
    start: Instant,
) -> RatchetOutcome {
    let verdict = if findings.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Fail
    };
    let result = GateResult {
        name: GATE.into(),
        passed: verdict == Verdict::Pass,
        skipped: false,
        verdict,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Validate {
            contracts: c.models,
            errors: findings.len(),
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Refinement(Box::new(c))),
    };
    RatchetOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

/// BASE's Lean tree and contracts, extracted from git into a scratch dir that is removed on drop.
struct BaseTree {
    dir: PathBuf,
    lean: PathBuf,
    contracts: PathBuf,
    label: String,
}

impl BaseTree {
    fn extract(root: &Path, contract_dir: &Path) -> Result<Self, String> {
        let (commit, label) = default_commit(root)
            .ok_or("neither merge-base(HEAD, origin/main) nor origin/main resolves")?;
        let top = git(root, &["rev-parse", "--show-toplevel"]).ok_or("not a git work tree")?;
        let top = PathBuf::from(top);
        let rel = |p: &Path| -> Result<String, String> {
            let abs = p
                .canonicalize()
                .map_err(|e| format!("{}: {e}", p.display()))?;
            let top = top
                .canonicalize()
                .map_err(|e| format!("{}: {e}", top.display()))?;
            abs.strip_prefix(&top)
                .map(|r| r.to_string_lossy().replace('\\', "/"))
                .map_err(|_| format!("{} is outside the work tree", abs.display()))
        };
        let lean_rel = rel(&root.join(LEAN_DIR))?;
        let contracts_rel = rel(contract_dir)?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir =
            std::env::temp_dir().join(format!("pv-refinement-base-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let me = Self {
            lean: dir.join(&lean_rel),
            contracts: dir.join(&contracts_rel),
            dir,
            label,
        };
        me.archive(&top, &commit, &[&lean_rel, &contracts_rel])?;
        Ok(me)
    }

    /// `git archive <commit> -- <paths> | tar -x -C <dir>`. A path BASE does not have is not an error: that
    /// side is then empty.
    fn archive(&self, top: &Path, commit: &str, paths: &[&str]) -> Result<(), String> {
        for p in paths {
            let present = git(top, &["cat-file", "-e", &format!("{commit}:{p}")]).is_some();
            if !present {
                continue;
            }
            let mut git_archive = Command::new("git")
                .arg("-C")
                .arg(top)
                .args(["archive", "--format=tar", commit, "--", p])
                .env_remove("GIT_DIR")
                .env_remove("GIT_WORK_TREE")
                .env_remove("GIT_INDEX_FILE")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("git archive: {e}"))?;
            let out = git_archive.stdout.take().ok_or("git archive: no stdout")?;
            let tar = Command::new("tar")
                .arg("-x")
                .arg("-C")
                .arg(&self.dir)
                .stdin(out)
                .stderr(Stdio::null())
                .status()
                .map_err(|e| format!("tar: {e}"))?;
            let ga = git_archive
                .wait()
                .map_err(|e| format!("git archive: {e}"))?;
            if !ga.success() || !tar.success() {
                return Err(format!("git archive {commit} -- {p} | tar -x failed"));
            }
        }
        Ok(())
    }
}

impl Drop for BaseTree {
    fn drop(&mut self) {
        if self
            .dir
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("pv-refinement-base-"))
        {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

/// merge-base(HEAD, origin/main), else the origin/main tip — `lib_baseline_ratchet.sh`'s order.
fn default_commit(root: &Path) -> Option<(String, String)> {
    if let Some(commit) = git(root, &["merge-base", "HEAD", "origin/main"]) {
        let label = format!("merge-base(HEAD, origin/main) {}", short(&commit));
        return Some((commit, label));
    }
    let commit = git(
        root,
        &["rev-parse", "--verify", "--quiet", "origin/main^{commit}"],
    )?;
    let label = format!("origin/main {}", short(&commit));
    Some((commit, label))
}

fn short(commit: &str) -> &str {
    commit.get(..12).unwrap_or(commit)
}

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .trim_end_matches('\n')
            .to_string(),
    )
}

#[cfg(test)]
#[path = "refinement_gate_tests.rs"]
mod tests;
