//! PVL-001 EV-8a (#4202) — `discharge-summary.json`, the one summary the tree can carry.
//!
//! `pv discharge run <lean-dir>` writes it as a SIBLING of `<lean-dir>` (`<staging-crate>/discharge-summary.json`),
//! never inside it: a file inside the tree it hashes cannot carry that tree's sha, its own bytes being an input to
//! it. It holds no timestamp and no commit sha, so a re-run over the same tree reproduces it byte for byte and a
//! squash merge keeps it valid (`tree_sha` is `git rev-parse HEAD:<lean-dir>`, content-addressed). It is written
//! on failure too: a RED run is a summary with RED fields, never a missing file.
//!
//! Its one consumer here is the `proved-is-derived` lint gate: a YAML `lean.status: proved` claim is DERIVED only
//! when a GREEN summary lists its theorem. The summary is a claim until EV-9's job regenerates it and diffs
//! (summary-fresh); a hand edit toward pass is caught there, not here.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{escapes, is_theorem, Report, Tree};

/// The tracked summary's file name, beside `<lean-dir>`.
pub const SUMMARY_FILE: &str = "discharge-summary.json";
/// The full log's file name, inside `<lean-dir>`: untracked (`.gitignore`d).
pub const LOG_FILE: &str = "discharge.json";

/// One module the check covered: a file in the root's import cone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Module {
    /// Relative to `<lean-dir>`.
    pub path: String,
    /// blake3 of the file's bytes, 64 hex digits.
    pub blake3: String,
    /// Its public theorems whose own declaration holds no escape (`sorry`, `admit`, …), as full names.
    pub theorems: Vec<String>,
}

/// `discharge-summary.json`. Field order is the file's; `None` is `null` (the step never ran).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    /// `git rev-parse HEAD:<lean-dir>`.
    pub tree_sha: Option<String>,
    /// `<lean-dir>/lean-toolchain`, trimmed.
    pub toolchain: Option<String>,
    /// The `mathlib` package's `rev` in `<lean-dir>/lake-manifest.json`.
    pub mathlib_rev: Option<String>,
    /// `build.sh`'s exit (rc 2 is its cache-miss decline). The Lean steps run only after a 0.
    pub build_exit: Option<i32>,
    /// `lake env lean Axioms.lean`.
    pub lake_exit: Option<i32>,
    /// `lake env leanchecker ProvableContracts` (124/137 on a timeout).
    pub leanchecker_exit: Option<i32>,
    /// `Axioms.lean` is its regeneration AND elaborated with rc 0.
    pub axioms_ok: bool,
    /// The escape scan, `--strict`, found nothing unlisted, stale, malformed or pending.
    pub escapes_ok: bool,
    /// `n/m` challenges the comparator closed; `None` when it never judged a row set.
    pub challenges_closed: Option<String>,
    /// The modules in the root's import cone, by path.
    pub modules: Vec<Module>,
}

impl Summary {
    /// Every Lean step ran and passed. Only a green summary derives anything.
    #[must_use]
    pub fn is_green(&self) -> bool {
        self.build_exit == Some(0)
            && self.lake_exit == Some(0)
            && self.leanchecker_exit == Some(0)
            && self.axioms_ok
            && self.escapes_ok
    }

    /// The theorems this summary derives: every listed one when green, none otherwise.
    #[must_use]
    pub fn derived(&self) -> BTreeSet<&str> {
        if !self.is_green() {
            return BTreeSet::new();
        }
        self.modules
            .iter()
            .flat_map(|m| m.theorems.iter().map(String::as_str))
            .collect()
    }

    /// The file's bytes: pretty JSON and a trailing newline, so a regeneration diffs clean.
    #[must_use]
    pub fn render(&self) -> String {
        let mut s = serde_json::to_string_pretty(self)
            .unwrap_or_else(|e| format!("{{\"error\": {:?}}}", e.to_string()));
        s.push('\n');
        s
    }
}

/// `<lean-dir>`'s sibling `discharge-summary.json` (`${LEAN%/lean}/discharge-summary.json`).
#[must_use]
pub fn summary_path(lean_dir: &Path) -> PathBuf {
    match lean_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(SUMMARY_FILE),
        _ => PathBuf::from(SUMMARY_FILE),
    }
}

/// Read a summary. `Err` names why (absent, unreadable, not a summary); the gate reads any `Err` as "derives nothing".
pub fn load(path: &Path) -> Result<Summary, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|e| format!("{}: not a discharge summary: {e}", path.display()))
}

/// Does `derived` hold the claim's `lean.theorem`? The YAML names it exactly or without the `ProvableContracts.`
/// root namespace (`Softmax.partition_of_unity` for `ProvableContracts.Softmax.partition_of_unity`); nothing looser.
#[must_use]
pub fn claim_derived(derived: &BTreeSet<&str>, theorem: &str) -> bool {
    derived.contains(theorem) || derived.contains(format!("ProvableContracts.{theorem}").as_str())
}

/// The modules the root's import cone reaches, each with its blake3 and its escape-free public theorems. A file
/// that cannot be read is listed with an empty hash: the summary records it, and it derives nothing.
#[must_use]
pub fn modules(tree: &Tree) -> Vec<Module> {
    let cone = tree.cone();
    let escaped: BTreeSet<(String, String)> = escapes(tree)
        .into_iter()
        .map(|e| (e.file, e.decl))
        .collect();
    let mut out: Vec<Module> = tree
        .files
        .iter()
        .filter(|f| cone.contains(&f.module))
        .map(|f| {
            let bytes = std::fs::read(tree.dir.join(&f.rel));
            let (blake3, readable) = match &bytes {
                Ok(b) => (blake3::hash(b).to_hex().to_string(), true),
                Err(_) => (String::new(), false),
            };
            let theorems = f
                .decls
                .iter()
                .filter(|d| readable && is_theorem(d))
                .filter(|d| !escaped.contains(&(f.rel.clone(), d.fqn.clone())))
                .map(|d| d.fqn.clone())
                .collect();
            Module {
                path: f.rel.clone(),
                blake3,
                theorems,
            }
        })
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// `toolchain` and `mathlib_rev` from `<lean-dir>`'s own files.
#[must_use]
pub fn pins(lean_dir: &Path) -> (Option<String>, Option<String>) {
    let toolchain = std::fs::read_to_string(lean_dir.join("lean-toolchain"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let mathlib = std::fs::read_to_string(lean_dir.join("lake-manifest.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|doc| {
            doc.get("packages")?
                .as_array()?
                .iter()
                .find(|p| p.get("name").and_then(|n| n.as_str()) == Some("mathlib"))?
                .get("rev")?
                .as_str()
                .map(str::to_string)
        });
    (toolchain, mathlib)
}

/// The summary of one run: `r` is the finished report (Lean steps included), `tree` the tree it judged (`None`
/// when it could not be loaded — no module is then listed).
#[must_use]
pub fn summarize(
    r: &Report,
    tree: Option<&Tree>,
    lean_dir: &Path,
    tree_sha: Option<String>,
    build_exit: Option<i32>,
) -> Summary {
    let (toolchain, mathlib_rev) = pins(lean_dir);
    Summary {
        tree_sha,
        toolchain,
        mathlib_rev,
        build_exit,
        lake_exit: r.lake_exit,
        leanchecker_exit: r.leanchecker_exit,
        axioms_ok: r.axioms_fresh == Some(true) && r.lake_exit == Some(0),
        escapes_ok: r.escapes_ok == Some(true),
        challenges_closed: r.challenges.map(|c| format!("{}/{}", c.closed, c.total)),
        modules: tree.map(modules).unwrap_or_default(),
    }
}

/// Where L4 credit comes from (EV-8b): a discharge summary beside the Lean base, or — with none to read — the
/// ONT-2a scan of the tree's own `.lean` text, which the report labels `self-declared`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum L4Source {
    Discharge,
    #[default]
    SelfDeclared,
}

/// What a summary grants: the theorems it discharges, or none and the first reason why.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Discharged {
    pub theorems: BTreeSet<String>,
    /// `None` when the summary grants; otherwise why it grants nothing (`stale discharge`, `lake_exit 1`, …).
    pub withheld: Option<String>,
}

impl Discharged {
    /// Does this grant the equation's `lean_theorem`? Same naming rule as [`claim_derived`].
    #[must_use]
    pub fn grants(&self, theorem: &str) -> bool {
        let t = theorem.trim().trim_matches('"');
        self.theorems.contains(t) || self.theorems.contains(&format!("ProvableContracts.{t}"))
    }
}

/// `n/m` with `n == m` and `m > 0`: every pinned statement was compared and closed.
fn all_challenges_closed(closed: Option<&str>) -> bool {
    let Some((n, m)) = closed.and_then(|c| c.split_once('/')) else {
        return false;
    };
    matches!((n.trim().parse::<u32>(), m.trim().parse::<u32>()), (Ok(n), Ok(m)) if m > 0 && n == m)
}

/// L4 credit from a summary: its theorems, only when every Lean step exited 0, its `tree_sha` is the CURRENT Lean
/// tree's, and every challenge closed. Anything else grants nothing, and says why — a stale or red summary is the
/// andon, never a partial credit.
#[must_use]
pub fn discharged(s: &Summary, current_tree_sha: Option<&str>) -> Discharged {
    let withheld = if s.tree_sha.is_none() || s.tree_sha.as_deref() != current_tree_sha {
        Some(format!(
            "stale discharge: summary tree_sha {} != current lean tree {}",
            s.tree_sha.as_deref().unwrap_or("null"),
            current_tree_sha.unwrap_or("unknown")
        ))
    } else if !s.is_green() {
        Some(format!(
            "red discharge: build_exit {:?}, lake_exit {:?}, leanchecker_exit {:?}, axioms_ok {}, escapes_ok {}",
            s.build_exit, s.lake_exit, s.leanchecker_exit, s.axioms_ok, s.escapes_ok
        ))
    } else if !all_challenges_closed(s.challenges_closed.as_deref()) {
        Some(format!(
            "challenges not closed: {}",
            s.challenges_closed.as_deref().unwrap_or("never judged")
        ))
    } else {
        None
    };
    let theorems = if withheld.is_none() {
        s.derived().into_iter().map(str::to_string).collect()
    } else {
        BTreeSet::new()
    };
    Discharged { theorems, withheld }
}

/// `git rev-parse HEAD:./` in `lean_dir`: the tree a fresh summary must name. `None` outside a git checkout, or
/// when the dir is not in HEAD.
#[must_use]
pub fn current_tree_sha(lean_dir: &Path) -> Option<String> {
    let o = std::process::Command::new("git")
        .args(["rev-parse", "HEAD:./"])
        .current_dir(lean_dir)
        .output()
        .ok()?;
    let sha = String::from_utf8_lossy(&o.stdout).trim().to_string();
    (o.status.success() && !sha.is_empty()).then_some(sha)
}

#[cfg(test)]
#[path = "summary_tests.rs"]
mod tests;
