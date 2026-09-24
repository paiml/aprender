//! PVL-001 EV-6a (#4139): `pv discharge` — what the Lean tree's proofs actually rest on.
//!
//! - **Escapes.** A token-level scan (comments and strings excluded, [`lex`]) of `ProvableContracts/**` and the
//!   root for `sorry admit axiom native_decide implemented_by extern unsafe partial`. Each one must be listed in
//!   `escape-allowlist.yaml` as `{file, decl, kind, reason, ticket, confirmed_by}`; `confirmed_by: pending` is
//!   PENDING, and RED only under `--strict` — the scanner and its exemptions are never confirmed by the same run.
//! - **Roots.** A theorem is contract-bound when a contract's `lean_theorem:` names it: an EXACT name
//!   (`ProvableContracts.<…>.<decl>`) must name a declaration or it is MISSING-ROOT; a LABEL is matched the way
//!   ONT-4b2 matches it ([`crate::ontology::extract::lean`]), and an unresolved label is held by a non-increasing
//!   ratchet keyed on the SET in `unresolved-labels.json`: only a label NOT in it fails (by name). The gate never
//!   writes that file; `make label-ratchet` only shrinks it (PVL-001 infra#992; cop ruling on #4139).
//! - **Axioms.lean**, generated: every bound theorem in the root's import cone gets a SUBSET pin over
//!   `Lean.collectAxioms` (a proof needing fewer axioms stays green); `capstones:` in `formalization.yaml` get an
//!   exact `#guard_msgs in #print axioms`. Bound theorems outside the cone are ORPHANED-ROOT: `lake env lean`
//!   cannot see a module `lake build` never built (EV-5a's orphans; EV-5c drains them).

pub mod lex;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::ontology::extract::lean::{reference_matches, references_of, Statement};

/// The escape kinds the scan reports.
pub const ESCAPE_KINDS: &[&str] = &[
    "sorry",
    "admit",
    "axiom",
    "native_decide",
    "implemented_by",
    "extern",
    "unsafe",
    "partial",
];
/// `status.axioms` when `formalization.yaml` does not say otherwise, in the order `#print axioms` prints them.
pub const DEFAULT_AXIOMS: &[&str] = &["propext", "Classical.choice", "Quot.sound"];
pub const ALLOWLIST: &str = "escape-allowlist.yaml";
pub const LABELS: &str = "unresolved-labels.json";
pub const AXIOMS_FILE: &str = "Axioms.lean";
const ROOT_MODULE: &str = "ProvableContracts";

/// One parsed `.lean` file of the tree.
#[derive(Debug, Clone)]
pub struct LeanFile {
    /// Relative to the lean dir, `/`-separated.
    pub rel: String,
    pub module: String,
    pub toks: Vec<lex::Token>,
    pub decls: Vec<lex::Decl>,
    pub imports: Vec<String>,
}

/// The Lean tree under a lean dir: the root `ProvableContracts.lean` and every file under `ProvableContracts/`.
#[derive(Debug, Clone)]
pub struct Tree {
    pub dir: PathBuf,
    pub files: Vec<LeanFile>,
}

impl Tree {
    /// `Err` (a decline) when the dir has no `ProvableContracts.lean`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let root = dir.join(format!("{ROOT_MODULE}.lean"));
        if !root.is_file() {
            return Err(format!("no {ROOT_MODULE}.lean under {}", dir.display()));
        }
        let mut paths = vec![root];
        collect_lean(&dir.join(ROOT_MODULE), &mut paths);
        paths.sort();
        let files = paths.iter().filter_map(|p| parse_file(dir, p)).collect();
        Ok(Self {
            dir: dir.to_path_buf(),
            files,
        })
    }

    /// The modules the root reaches through imports, transitively (itself included).
    #[must_use]
    pub fn cone(&self) -> BTreeSet<String> {
        let by_mod: BTreeMap<&str, &LeanFile> =
            self.files.iter().map(|f| (f.module.as_str(), f)).collect();
        let mut seen = BTreeSet::new();
        let mut todo = vec![ROOT_MODULE.to_string()];
        while let Some(m) = todo.pop() {
            let Some(f) = by_mod.get(m.as_str()) else {
                continue;
            };
            if seen.insert(m.clone()) {
                todo.extend(f.imports.iter().cloned());
            }
        }
        seen
    }
}

fn collect_lean(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_lean(&p, out);
        } else if p.extension().is_some_and(|x| x == "lean") {
            out.push(p);
        }
    }
}

fn parse_file(dir: &Path, p: &Path) -> Option<LeanFile> {
    let src = std::fs::read_to_string(p).ok()?;
    let rel = p
        .strip_prefix(dir)
        .ok()?
        .to_string_lossy()
        .replace('\\', "/");
    let blanked = lex::blank(&src);
    let toks = lex::tokens(&blanked);
    Some(LeanFile {
        module: rel.trim_end_matches(".lean").replace('/', "."),
        decls: lex::decls(&toks),
        imports: lex::imports(&blanked),
        toks,
        rel,
    })
}

/// One escape the scan found.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Escape {
    pub file: String,
    pub line: usize,
    pub kind: String,
    /// The declaration it belongs to: the one it sits in (`sorry admit native_decide`), the one it declares
    /// (`axiom`), or the one it modifies (`implemented_by extern unsafe partial` precede their declaration).
    pub decl: String,
}

/// Every escape in the tree, in file then line order.
#[must_use]
pub fn escapes(tree: &Tree) -> Vec<Escape> {
    let mut out = Vec::new();
    for f in &tree.files {
        for (i, t) in f.toks.iter().enumerate() {
            if ESCAPE_KINDS.contains(&t.text.as_str()) {
                out.push(Escape {
                    file: f.rel.clone(),
                    line: t.line,
                    kind: t.text.clone(),
                    decl: owner(&f.decls, i, &t.text),
                });
            }
        }
    }
    out
}

fn owner(decls: &[lex::Decl], i: usize, kind: &str) -> String {
    let pick = match kind {
        "sorry" | "admit" | "native_decide" => decls.iter().rev().find(|d| d.at < i),
        "axiom" => decls.iter().find(|d| d.at == i),
        _ => decls.iter().find(|d| d.at > i),
    };
    pick.map_or_else(|| "<none>".to_string(), |d| d.fqn.clone())
}

/// One `escape-allowlist.yaml` entry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Allowed {
    pub file: String,
    pub decl: String,
    pub kind: String,
    pub reason: String,
    pub ticket: String,
    pub confirmed_by: String,
}

impl Allowed {
    fn key(&self) -> (String, String, String) {
        (self.file.clone(), self.decl.clone(), self.kind.clone())
    }

    /// The fields this entry lacks (an entry without its reason or its ticket exempts nothing).
    fn missing(&self) -> Vec<&'static str> {
        [
            ("file", &self.file),
            ("decl", &self.decl),
            ("kind", &self.kind),
            ("reason", &self.reason),
            ("ticket", &self.ticket),
            ("confirmed_by", &self.confirmed_by),
        ]
        .into_iter()
        .filter(|(_, v)| v.trim().is_empty())
        .map(|(k, _)| k)
        .collect()
    }
}

/// The allowlist; a missing file is an empty list (every escape is then unlisted), an unreadable one an error.
pub fn load_allowlist(dir: &Path) -> Result<Vec<Allowed>, String> {
    let p = dir.join(ALLOWLIST);
    if !p.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let doc: serde_yaml::Value =
        serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
    let entries = match &doc {
        serde_yaml::Value::Null => return Ok(Vec::new()),
        serde_yaml::Value::Sequence(s) => s,
        _ => return Err(format!("{}: not a list of entries", p.display())),
    };
    let field = |e: &serde_yaml::Value, k: &str| {
        e.get(k).map(|v| match v {
            serde_yaml::Value::String(s) => s.clone(),
            other => serde_yaml::to_string(other)
                .unwrap_or_default()
                .trim()
                .to_string(),
        })
    };
    Ok(entries
        .iter()
        .map(|e| Allowed {
            file: field(e, "file").unwrap_or_default(),
            decl: field(e, "decl").unwrap_or_default(),
            kind: field(e, "kind").unwrap_or_default(),
            reason: field(e, "reason").unwrap_or_default(),
            ticket: field(e, "ticket").unwrap_or_default(),
            confirmed_by: field(e, "confirmed_by").unwrap_or_default(),
        })
        .collect())
}

/// `formalization.yaml`'s `status.axioms` and `capstones`, or the defaults when it is absent (EV-8b lands it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formalization {
    pub axioms: Vec<String>,
    pub capstones: Vec<String>,
}

pub fn load_formalization(dir: &Path) -> Result<Formalization, String> {
    let p = dir.join("formalization.yaml");
    let strings = |v: Option<&serde_yaml::Value>| -> Vec<String> {
        v.and_then(serde_yaml::Value::as_sequence)
            .map(|s| {
                s.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let doc: serde_yaml::Value = if p.exists() {
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?
    } else {
        serde_yaml::Value::Null
    };
    let mut axioms = strings(doc.get("status").and_then(|s| s.get("axioms")));
    if axioms.is_empty() {
        axioms = DEFAULT_AXIOMS.iter().map(|s| (*s).to_string()).collect();
    }
    Ok(Formalization {
        axioms,
        capstones: strings(doc.get("capstones")),
    })
}

/// A contract-bound theorem.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Root {
    pub fqn: String,
    pub module: String,
}

/// What the contracts' `lean_theorem:` references bind.
#[derive(Debug, Clone, Default)]
pub struct Binding {
    pub roots: BTreeSet<Root>,
    /// `(contract stem, exact-name reference)` naming no declaration.
    pub missing: BTreeSet<(String, String)>,
    /// `(contract stem, label reference)` matching no theorem, file or domain.
    pub unresolved_labels: BTreeSet<(String, String)>,
}

/// An exact-name reference: the fully qualified form, `ProvableContracts.<…>.<decl>`.
#[must_use]
pub fn is_exact_name(reference: &str) -> bool {
    reference.starts_with("ProvableContracts.") && !reference.contains(char::is_whitespace)
}

/// The theorems of `Theorems/<Domain>/*.lean` (ONT-4b2's scope), non-private, as ONT statements plus their roots.
fn statements(tree: &Tree) -> Vec<(Statement, Root)> {
    let mut out = Vec::new();
    for f in &tree.files {
        let parts: Vec<&str> = f.rel.split('/').collect();
        let [_, "Theorems", domain, file] = parts.as_slice() else {
            continue;
        };
        for d in f.decls.iter().filter(|d| is_theorem(d)) {
            let s = Statement {
                domain: (*domain).to_string(),
                stem: file.trim_end_matches(".lean").to_string(),
                name: d.name.clone(),
                file: f.rel.clone(),
                sorry_free: true,
            };
            out.push((
                s,
                Root {
                    fqn: d.fqn.clone(),
                    module: f.module.clone(),
                },
            ));
        }
    }
    out
}

fn theorem_fqns(tree: &Tree) -> BTreeSet<&str> {
    tree.files
        .iter()
        .flat_map(|f| {
            f.decls
                .iter()
                .filter(|d| is_theorem(d))
                .map(|d| d.fqn.as_str())
        })
        .collect()
}

fn is_theorem(d: &lex::Decl) -> bool {
    matches!(d.keyword.as_str(), "theorem" | "lemma") && !d.private
}

/// Join every contract under `contract_dir` to the tree.
#[must_use]
pub fn bind(tree: &Tree, contract_dir: &Path) -> Binding {
    let stmts = statements(tree);
    let accepted: Vec<BTreeSet<String>> = stmts.iter().map(|(s, _)| s.accepted_names()).collect();
    let theorems: BTreeMap<String, Root> = tree
        .files
        .iter()
        .flat_map(|f| {
            f.decls
                .iter()
                .filter(|d| is_theorem(d))
                .map(move |d| (d.fqn.clone(), f.module.clone()))
        })
        .map(|(fqn, module)| (fqn.clone(), Root { fqn, module }))
        .collect();
    let mut b = Binding::default();
    for (stem, _rel, doc) in crate::ontology::extract::pv_contract::documents(contract_dir) {
        for r in references_of(&doc) {
            bind_one(&mut b, &stem, &r, &theorems, &stmts, &accepted);
        }
    }
    b
}

fn bind_one(
    b: &mut Binding,
    stem: &str,
    r: &str,
    theorems: &BTreeMap<String, Root>,
    stmts: &[(Statement, Root)],
    accepted: &[BTreeSet<String>],
) {
    if is_exact_name(r) {
        match theorems.get(r) {
            Some(root) => {
                b.roots.insert(root.clone());
            }
            None => {
                b.missing.insert((stem.to_string(), r.to_string()));
            }
        }
        return;
    }
    let hits: Vec<&Root> = stmts
        .iter()
        .zip(accepted)
        .filter(|(_, names)| reference_matches(r, names))
        .map(|((_, root), _)| root)
        .collect();
    if hits.is_empty() {
        b.unresolved_labels
            .insert((stem.to_string(), r.to_string()));
    }
    b.roots.extend(hits.into_iter().cloned());
}

/// `unresolved-labels.json` `{command, labels: [{contract, label}]}`: the SET the label ratchet is keyed on.
/// `None` when absent (every unresolved label is then new). The gate never writes it (PVL-001 infra#992).
pub fn load_labels(dir: &Path) -> Result<Option<BTreeSet<(String, String)>>, String> {
    let p = dir.join(LABELS);
    if !p.exists() {
        return Ok(None);
    }
    let bad = |e: &dyn std::fmt::Display| format!("{}: {e}", p.display());
    let text = std::fs::read_to_string(&p).map_err(|e| bad(&e))?;
    let doc: serde_json::Value = serde_json::from_str(&text).map_err(|e| bad(&e))?;
    let labels = doc
        .get("labels")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| bad(&"no `labels` array"))?;
    labels
        .iter()
        .map(|l| {
            let f = |k: &str| {
                l.get(k)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            };
            f("contract")
                .zip(f("label"))
                .ok_or_else(|| bad(&format!("entry without contract/label: {l}")))
        })
        .collect::<Result<BTreeSet<_>, _>>()
        .map(Some)
}

/// The bytes `make label-ratchet` writes.
#[must_use]
pub fn render_labels(set: &BTreeSet<(String, String)>) -> String {
    use serde_json::{Map, Value};
    let obj = |pairs: [(&str, Value); 2]| {
        Value::Object(
            pairs
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect::<Map<_, _>>(),
        )
    };
    let labels = set
        .iter()
        .map(|(c, l)| {
            obj([
                ("contract", Value::from(c.as_str())),
                ("label", Value::from(l.as_str())),
            ])
        })
        .collect();
    let doc = obj([
        ("command", Value::from("make label-ratchet")),
        ("labels", Value::Array(labels)),
    ]);
    format!(
        "{}\n",
        serde_json::to_string_pretty(&doc).unwrap_or_default()
    )
}

/// A Lean name literal: `` `A.b ``, with `«»` around any component that is not a plain identifier.
fn name_lit(fqn: &str) -> String {
    let parts: Vec<String> = fqn
        .split('.')
        .map(|p| {
            let plain = p
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
                && p.chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '\'');
            if plain {
                p.to_string()
            } else {
                format!("«{p}»")
            }
        })
        .collect();
    format!("`{}", parts.join("."))
}

/// The pinned set: `status.axioms` plus every axiom `escape-allowlist.yaml` exempts (the allowlist IS the
/// exemption; an axiom removed from it is RED twice — in the scan and in its dependents' pins).
#[must_use]
pub fn pinned_axioms(form: &Formalization, allow: &[Allowed]) -> Vec<String> {
    let mut out = form.axioms.clone();
    for a in allow
        .iter()
        .filter(|a| a.kind == "axiom" && !a.decl.is_empty())
    {
        if !out.contains(&a.decl) {
            out.push(a.decl.clone());
        }
    }
    out
}

/// `Axioms.lean`: the roots in `cone` get subset pins, capstones an exact pin; the rest are counted, not pinned.
#[must_use]
pub fn render_axioms(
    roots: &BTreeSet<Root>,
    cone: &BTreeSet<String>,
    pinned: &[String],
    form: &Formalization,
) -> String {
    let in_cone: Vec<&Root> = roots.iter().filter(|r| cone.contains(&r.module)).collect();
    let orphaned = roots.len() - in_cone.len();
    let mut s = format!(
        "-- GENERATED by `pv discharge gen-axioms` (PVL-001 EV-6a, #4139). Do not edit: regenerate.\n\
         -- Every contract-bound theorem in the root's import cone may use only the pinned axioms. It is a SUBSET pin,\n\
         -- so a proof that needs fewer stays green; `sorryAx`, or any axiom outside the set, fails elaboration.\n\
         -- {} pinned; {orphaned} bound outside the root's import cone (ORPHANED-ROOT, pinned once EV-5c imports them).\n\
         import {ROOT_MODULE}\n\n\
         open Lean Elab Command in\n\
         /-- `n` exists and every axiom it depends on is in `pinned`. -/\n\
         def pvlAxiomsSubset (n : Name) (pinned : List Name) : CommandElabM Unit := do\n\
         \x20 unless (← getEnv).contains n do\n\
         \x20   throwError \"MISSING-ROOT {{n}}\"\n\
         \x20 let axs ← collectAxioms n\n\
         \x20 let extra := axs.toList.filter (fun a => !pinned.contains a)\n\
         \x20 unless extra.isEmpty do\n\
         \x20   throwError \"AXIOMS {{n}}: {{extra}} outside the pinned set\"\n\n\
         open Lean in\n\
         /-- formalization.yaml `status.axioms`, plus every axiom escape-allowlist.yaml exempts. -/\n\
         def pvlPinned : List Name := [{}]\n\n",
        in_cone.len(),
        pinned.iter().map(|a| name_lit(a)).collect::<Vec<_>>().join(", "),
    );
    for r in &in_cone {
        s.push_str(&format!(
            "run_cmd pvlAxiomsSubset {} pvlPinned\n",
            name_lit(&r.fqn)
        ));
    }
    for c in &form.capstones {
        s.push_str(&format!(
            "\n/-- info: '{c}' depends on axioms: [{}] -/\n#guard_msgs in #print axioms {c}\n",
            form.axioms.join(", ")
        ));
    }
    s
}

/// Everything one generation read from disk, and the `Axioms.lean` it renders.
#[derive(Debug, Clone)]
pub struct Generated {
    pub text: String,
    pub tree: Tree,
    pub binding: Binding,
    pub allow: Vec<Allowed>,
    pub form: Formalization,
}

/// The whole generation, from disk. `Err` (a decline: nothing can be judged) when the root file is missing or
/// `escape-allowlist.yaml` / `formalization.yaml` cannot be read.
pub fn generate(lean_dir: &Path, contract_dir: &Path) -> Result<Generated, String> {
    let tree = Tree::load(lean_dir)?;
    let binding = bind(&tree, contract_dir);
    let allow = load_allowlist(lean_dir)?;
    let form = load_formalization(lean_dir)?;
    let text = render_axioms(
        &binding.roots,
        &tree.cone(),
        &pinned_axioms(&form, &allow),
        &form,
    );
    Ok(Generated {
        text,
        tree,
        binding,
        allow,
        form,
    })
}

/// A `pv discharge` verdict: printed lines, and whether it rejects (rc 1) or declines (rc 2).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub lines: Vec<String>,
    pub reject: bool,
    pub decline: Option<String>,
}

impl Report {
    fn fail(&mut self, line: String) {
        self.lines.push(format!("FAIL  {line}"));
        self.reject = true;
    }
}

/// `check`'s switches.
#[derive(Debug, Clone, Copy, Default)]
pub struct CheckOpts {
    pub strict: bool,
}

/// Escapes against the allowlist: unlisted, malformed and stale are RED; pending is RED only under `strict`.
pub fn judge_escapes(found: &[Escape], allow: &[Allowed], strict: bool, r: &mut Report) {
    let listed: BTreeSet<_> = allow.iter().map(Allowed::key).collect();
    for e in found
        .iter()
        .filter(|e| !listed.contains(&(e.file.clone(), e.decl.clone(), e.kind.clone())))
    {
        r.fail(format!(
            "ESCAPE {}:{} `{}` in {} -- not in {ALLOWLIST}",
            e.file, e.line, e.kind, e.decl
        ));
    }
    let present: BTreeSet<_> = found
        .iter()
        .map(|e| (e.file.clone(), e.decl.clone(), e.kind.clone()))
        .collect();
    for a in allow {
        let miss = a.missing();
        if !miss.is_empty() {
            r.fail(format!(
                "{ALLOWLIST} entry {}/{}/{} has no {}",
                a.file,
                a.decl,
                a.kind,
                miss.join("/")
            ));
        } else if !present.contains(&a.key()) {
            r.fail(format!(
                "STALE {ALLOWLIST} entry {}/{}/{} -- no such escape: remove it",
                a.file, a.decl, a.kind
            ));
        }
    }
    let pending: Vec<&Allowed> = allow
        .iter()
        .filter(|a| a.confirmed_by.trim() == "pending")
        .collect();
    if !pending.is_empty() {
        r.lines.push(format!("PENDING ({})", pending.len()));
        for a in &pending {
            r.lines.push(format!(
                "  {} {} in {} -- {}",
                a.kind, a.decl, a.file, a.ticket
            ));
        }
        if strict {
            r.fail(format!(
                "--strict: {} allowlist entr(ies) still confirmed_by: pending",
                pending.len()
            ));
        }
    }
}

/// The label ratchet: an unresolved label NOT in the set fails BY NAME; a listed one that resolves now is reported
/// (`make label-ratchet` removes it) and is not a failure.
pub fn judge_labels(
    current: &BTreeSet<(String, String)>,
    base: Option<&BTreeSet<(String, String)>>,
    r: &mut Report,
) {
    let empty = BTreeSet::new();
    let base = base.unwrap_or(&empty);
    for (s, l) in current.difference(base) {
        r.fail(format!(
            "NEW-UNRESOLVED-LABEL {s}: {l} -- names no theorem, file or domain in the tree"
        ));
    }
    let resolved = base.difference(current).count();
    if resolved > 0 {
        r.lines.push(format!(
            "RESOLVED-LABEL ({resolved}) still listed in {LABELS} -- `make label-ratchet` removes them"
        ));
    }
    r.lines.push(format!(
        "UNRESOLVED-LABEL ({}) (listed {})",
        current.len(),
        base.len()
    ));
}

/// `pv discharge check` without the Lean elaboration (the caller runs `lake env lean Axioms.lean` unless
/// `--no-lake`). Failures are judged before the vacuity decline: a failure never hides behind "not a verdict".
pub fn check(lean_dir: &Path, contract_dir: &Path, opts: CheckOpts) -> Report {
    let mut r = Report::default();
    let g = match generate(lean_dir, contract_dir) {
        Ok(g) => g,
        Err(e) => {
            r.decline = Some(e);
            return r;
        }
    };
    judge_escapes(&escapes(&g.tree), &g.allow, opts.strict, &mut r);
    judge_roots(&g, &mut r);
    judge_ratchet(lean_dir, &g.binding, &mut r);
    judge_axioms_file(lean_dir, &g.text, &mut r);
    let cone = g.tree.cone();
    let roots = &g.binding.roots;
    let pinned = roots.iter().filter(|x| cone.contains(&x.module)).count();
    r.lines.push(format!(
        "ROOTS {pinned} pinned, {} ORPHANED-ROOT",
        roots.len() - pinned
    ));
    if pinned == 0 && !r.reject {
        r.decline = Some(format!(
            "0 contract-bound theorems in the root's import cone under {}",
            lean_dir.display()
        ));
    }
    r
}

/// MISSING-ROOT: an exact-name reference naming no theorem (saying what it names instead, if anything), and a
/// capstone naming no theorem.
fn judge_roots(g: &Generated, r: &mut Report) {
    let kinds: BTreeMap<&str, &str> = g
        .tree
        .files
        .iter()
        .flat_map(|f| f.decls.iter().map(|d| (d.fqn.as_str(), d.keyword.as_str())))
        .collect();
    for (s, x) in &g.binding.missing {
        let what = kinds.get(x.as_str()).map_or_else(
            || "no such theorem in the tree".to_string(),
            |k| format!("it names an `{k}`, not a proved theorem"),
        );
        r.fail(format!("MISSING-ROOT contract {s}: {x} -- {what}"));
    }
    let theorems = theorem_fqns(&g.tree);
    for c in g
        .form
        .capstones
        .iter()
        .filter(|c| !theorems.contains(c.as_str()))
    {
        r.fail(format!(
            "MISSING-ROOT capstone {c} -- no such theorem in the tree"
        ));
    }
}

fn judge_ratchet(lean_dir: &Path, b: &Binding, r: &mut Report) {
    match load_labels(lean_dir) {
        Ok(base) => judge_labels(&b.unresolved_labels, base.as_ref(), r),
        Err(e) => r.fail(format!("{LABELS} unreadable: {e}")),
    }
}

/// `make label-ratchet` (`pv discharge label-ratchet`): the set keeps only labels still unresolved; it never gains
/// one (a new label is reported and left OUT). A missing file is seeded from what is measured.
pub fn ratchet_labels(lean_dir: &Path, contract_dir: &Path) -> Report {
    let mut r = Report::default();
    let tree = match Tree::load(lean_dir) {
        Ok(t) => t,
        Err(e) => {
            r.decline = Some(e);
            return r;
        }
    };
    let current = bind(&tree, contract_dir).unresolved_labels;
    let next: BTreeSet<_> = match load_labels(lean_dir) {
        Ok(Some(base)) => {
            judge_labels(&current, Some(&base), &mut r);
            base.intersection(&current).cloned().collect()
        }
        Ok(None) => current,
        Err(e) => {
            r.fail(format!("{LABELS} unreadable: {e}"));
            return r;
        }
    };
    let p = lean_dir.join(LABELS);
    match std::fs::write(&p, render_labels(&next)) {
        Ok(()) => r
            .lines
            .push(format!("wrote {} ({} label(s))", p.display(), next.len())),
        Err(e) => r.fail(format!("cannot write {}: {e}", p.display())),
    }
    r
}

fn judge_axioms_file(lean_dir: &Path, text: &str, r: &mut Report) {
    match std::fs::read_to_string(lean_dir.join(AXIOMS_FILE)) {
        Ok(on_disk) if on_disk == text => {}
        Ok(_) => r.fail(format!(
            "STALE {AXIOMS_FILE} -- regeneration differs: run `pv discharge gen-axioms`"
        )),
        Err(_) => r.fail(format!("no {AXIOMS_FILE} -- run `pv discharge gen-axioms`")),
    }
}

#[cfg(test)]
mod tests;
