//! README ontology — published docs name the monorepo, never a retired repo (#5382, contract
//! `contracts/readme-ontology-v1.yaml`).
//!
//! APR-MONO merged 20 repos into `paiml/aprender` and archived every one of them. crates.io renders a
//! crate's README and its manifest's `repository` / `homepage` / `documentation` at publish time, so a
//! link to an archived repo, or to the crates.io / docs.rs page of a crate that no longer ships, is
//! frozen into the published page. Measured on 2026-10-15: 232 such links across 38 READMEs.
//!
//! The ontology fact this module checks: the crate names a published doc may point a reader at are the
//! packages in this tree. A name in [`ARCHIVED_REPOS`] is retired as a repository always. As a crate, it
//! and every old crate name the tree still uses as a `[lib] name` (`trueno_gpu` was `trueno-gpu`) are
//! retired unless a package still carries the name (the `provable-contracts*` facades do).
//!
//! `pv ontology readme [ROOT]` runs [`audit_tree`]; `crates/aprender-core/tests/readme_contract.rs` gates
//! on it, and also checks [`ARCHIVED_REPOS`] against the archive list in
//! `docs/specifications/aprender-monorepo-consolidation.md`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;

use crate::doc_integrity::DocViolation;

/// The 20 repos APR-MONO archived (consolidation spec, Phase 5b), lowercase.
pub const ARCHIVED_REPOS: &[&str] = &[
    "alimentar",
    "batuta",
    "batuta-common",
    "certeza",
    "entrenar",
    "pacha",
    "presentar",
    "probar",
    "provable-contracts",
    "realizar",
    "renacer",
    "repartir",
    "simular",
    "trueno",
    "trueno-db",
    "trueno-graph",
    "trueno-rag",
    "trueno-viz",
    "trueno-zram",
    "verificar",
];

/// A link to an archived repository (`github.com/paiml/<old>`, `raw.githubusercontent.com/paiml/<old>`,
/// `paiml.github.io/<old>`).
pub const RULE_ARCHIVED_REPO: &str = "readme-archived-repo";
/// A crates.io / docs.rs / lib.rs / shields.io crate page for a name no package in the tree carries.
pub const RULE_RETIRED_CRATE: &str = "readme-retired-crate";
/// `cargo install|add <old>` or a `<old> = "x.y"` dependency line for a retired crate.
pub const RULE_RETIRED_INSTALL: &str = "readme-retired-install";

fn is_archived(name: &str) -> bool {
    ARCHIVED_REPOS.contains(&name.to_ascii_lowercase().as_str())
}

fn retired_crate(name: &str, retired: &BTreeSet<String>) -> bool {
    retired.contains(&name.to_ascii_lowercase().replace('_', "-"))
}

/// The crate names a published doc must not point at: every [`ARCHIVED_REPOS`] name, plus every `[lib] name`
/// that differs from its package (`trueno_gpu` in `aprender-gpu` was the crates.io crate `trueno-gpu`),
/// minus any name a package in the tree still carries.
pub fn retired_crates(manifests: &[(String, Option<String>)]) -> BTreeSet<String> {
    let packages: BTreeSet<&str> = manifests.iter().map(|(p, _)| p.as_str()).collect();
    let libs = manifests
        .iter()
        .filter_map(|(_, lib)| lib.as_ref().map(|l| l.replace('_', "-")));
    ARCHIVED_REPOS
        .iter()
        .map(|s| (*s).to_string())
        .chain(libs)
        .filter(|n| !packages.contains(n.as_str()))
        .collect()
}

struct Patterns {
    repo: Regex,
    crate_page: Regex,
    install: Regex,
    dep_line: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| Patterns {
        repo: Regex::new(r"(?i)(?:github\.com/paiml|raw\.githubusercontent\.com/paiml|paiml\.github\.io)/([A-Za-z0-9_.-]+)")
            .expect("static regex"),
        crate_page: Regex::new(
            r"(?i)(?:crates\.io/crates|docs\.rs(?:/crate)?|lib\.rs/crates|shields\.io/(?:crates/[a-z]+|docsrs))/([A-Za-z0-9_-]+)",
        )
        .expect("static regex"),
        install: Regex::new(r"cargo\s+(?:install|add)\s+(?:--?[A-Za-z-]+\s+)*([A-Za-z0-9_-]+)")
            .expect("static regex"),
        dep_line: Regex::new(r#"^\s*([A-Za-z0-9_-]+)\s*=\s*("[^"]*"|\{.*\})"#).expect("static regex"),
    })
}

/// Trim what a URL path segment picks up around a repo name: `trueno.git`, `trueno.` at a sentence end.
fn repo_segment(raw: &str) -> &str {
    let s = raw.trim_end_matches('.');
    s.strip_suffix(".git").unwrap_or(s).trim_end_matches('.')
}

/// Scan one published text (a README) for pointers at retired repos and crates.
///
/// `retired` is [`retired_crates`] of the tree, lowercase.
pub fn scan_text(text: &str, retired: &BTreeSet<String>) -> Vec<DocViolation> {
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let hit = |rule, message| DocViolation {
            line: idx + 1,
            rule,
            message,
        };
        out.extend(archived_repo_hits(line).map(|m| hit(RULE_ARCHIVED_REPO, m)));
        out.extend(crate_page_hits(line, retired).map(|m| hit(RULE_RETIRED_CRATE, m)));
        out.extend(install_hits(line, retired).map(|m| hit(RULE_RETIRED_INSTALL, m)));
    }
    out
}

/// One message per `github.com/paiml/<archived>`-style link on `line`.
fn archived_repo_hits(line: &str) -> impl Iterator<Item = String> + '_ {
    patterns().repo.captures_iter(line).filter_map(|c| {
        let name = repo_segment(&c[1]);
        is_archived(name).then(|| {
            format!(
                "`{}` points at archived repo `{name}`; use github.com/paiml/aprender",
                &c[0]
            )
        })
    })
}

/// One message per crates.io / docs.rs / lib.rs / badge page naming a retired crate on `line`.
fn crate_page_hits<'a>(
    line: &'a str,
    retired: &'a BTreeSet<String>,
) -> impl Iterator<Item = String> + 'a {
    patterns()
        .crate_page
        .captures_iter(line)
        .filter_map(move |c| {
            retired_crate(&c[1], retired).then(|| {
                format!(
                    "`{}` names retired crate `{}`; use the package name from its Cargo.toml",
                    &c[0], &c[1]
                )
            })
        })
}

/// `cargo install|add <retired>` hits, then a crates.io dependency line on a retired crate.
fn install_hits(line: &str, retired: &BTreeSet<String>) -> std::vec::IntoIter<String> {
    let p = patterns();
    let mut out: Vec<String> = p
        .install
        .captures_iter(line)
        .filter(|c| retired_crate(&c[1], retired))
        .map(|c| format!("`{}` installs retired crate `{}`", c[0].trim(), &c[1]))
        .collect();
    let dep = p.dep_line.captures(line).filter(|c| {
        retired_crate(&c[1], retired) && !c[2].contains("package") && !c[2].contains("path")
    });
    if let Some(c) = dep {
        out.push(format!(
            "`{}` depends on retired crate `{}` from crates.io",
            line.trim(),
            &c[1]
        ));
    }
    out.into_iter()
}

/// Scan a Cargo.toml's crates.io-rendered URL keys (`repository`, `homepage`, `documentation`).
pub fn scan_manifest(toml: &str, retired: &BTreeSet<String>) -> Vec<DocViolation> {
    let mut out = Vec::new();
    for (idx, line) in toml.lines().enumerate() {
        let key = line.trim_start();
        if ["repository", "homepage", "documentation"]
            .iter()
            .any(|k| key.starts_with(k) && key[k.len()..].trim_start().starts_with('='))
        {
            out.extend(scan_text(line, retired).into_iter().map(|mut v| {
                v.line = idx + 1;
                v
            }));
        }
    }
    out
}

/// The first `name = "..."` under `section` (`[package]`, `[lib]`), lowercase.
fn section_name(toml: &str, section: &str) -> Option<String> {
    let mut inside = false;
    for line in toml.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            inside = t == section;
            continue;
        }
        if inside {
            if let Some(rest) = t.strip_prefix("name") {
                if let Some(v) = rest.trim_start().strip_prefix('=') {
                    return Some(v.trim().trim_matches('"').to_ascii_lowercase());
                }
            }
        }
    }
    None
}

/// `([package] name, [lib] name)`, lowercase; `None` for a manifest with no package (a workspace shell).
pub fn package_and_lib(toml: &str) -> Option<(String, Option<String>)> {
    Some((
        section_name(toml, "[package]")?,
        section_name(toml, "[lib]"),
    ))
}

/// What [`audit_tree`] read and found.
#[derive(Debug, Default)]
pub struct TreeAudit {
    pub readmes: Vec<PathBuf>,
    pub manifests: Vec<PathBuf>,
    pub packages: BTreeSet<String>,
    pub retired: BTreeSet<String>,
    pub findings: Vec<(PathBuf, DocViolation)>,
}

fn skip_dir(name: &str) -> bool {
    name.starts_with('.') || matches!(name, "target" | "node_modules")
}

fn walk(dir: &Path, readmes: &mut Vec<PathBuf>, manifests: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if path.is_dir() {
            if !skip_dir(&name) {
                walk(&path, readmes, manifests);
            }
        } else if name == "README.md" {
            readmes.push(path);
        } else if name == "Cargo.toml" {
            manifests.push(path);
        }
    }
}

/// Audit `root/README.md`, every `README.md` and every `Cargo.toml` under `root/crates/`.
///
/// Paths in the result are relative to `root`.
pub fn audit_tree(root: &Path) -> TreeAudit {
    let mut audit = TreeAudit::default();
    let root_readme = root.join("README.md");
    if root_readme.is_file() {
        audit.readmes.push(root_readme);
    }
    walk(
        &root.join("crates"),
        &mut audit.readmes,
        &mut audit.manifests,
    );
    audit.manifests.insert(0, root.join("Cargo.toml"));
    let manifest_texts: Vec<(PathBuf, String)> = audit
        .manifests
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok().map(|t| (p.clone(), t)))
        .collect();
    let named: Vec<(String, Option<String>)> = manifest_texts
        .iter()
        .filter_map(|(_, t)| package_and_lib(t))
        .collect();
    audit.packages = named.iter().map(|(p, _)| p.clone()).collect();
    audit.retired = retired_crates(&named);
    let rel = |p: &Path| p.strip_prefix(root).unwrap_or(p).to_path_buf();
    for path in &audit.readmes {
        if let Ok(text) = std::fs::read_to_string(path) {
            for v in scan_text(&text, &audit.retired) {
                audit.findings.push((rel(path), v));
            }
        }
    }
    for (path, text) in &manifest_texts {
        for v in scan_manifest(text, &audit.retired) {
            audit.findings.push((rel(path), v));
        }
    }
    audit
}

#[cfg(test)]
#[path = "readme_tests.rs"]
mod tests;
