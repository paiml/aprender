//! no-mock-named-real-v1 (TRACE-001 TR-04, rule R-5: a mock is named as a mock).
//!
//! A module or type declared outside a workspace crate but named after that
//! crate's library (`mod renacer`, `mod trueno`, `struct Realizar`, …) reads as
//! the real crate. `tests/modality_matrix/common.rs` carried `pub mod renacer`:
//! a thread-local span recorder that traced nothing, whose QA-A08 test printed
//! "renacer::capture() API works correctly".
//!
//! Std only, compiled by `scripts/check_no_mock_named_real.sh` with `rustc`
//! (TRACE-001 R-3: guards are Rust). Input on stdin, NUL-separated: `git
//! ls-files -z` of the tree. Library names are DERIVED from every tracked
//! `Cargo.toml` below the root (`[package] name` and `[lib] name`, `-` → `_`);
//! there is no hand-kept list of crates.
//!
//! The declarations that predate the rule are frozen in a shrink-only baseline
//! (`--baseline FILE`, one `path kind ident` per line). A declaration not in it
//! is NEW; a baseline line that no longer matches is STALE. Either fails.
//!
//! Exit: 0 clean, 1 NEW or STALE lines, 2 cannot judge.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

/// `mod:<lib>` and `type:<Lib>` → the crate directories whose library carries that name.
type Owners = BTreeMap<String, Vec<PathBuf>>;

/// One module or type declared outside the crate whose library it is named after.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Hit {
    path: String,
    kind: &'static str,
    ident: String,
    line: usize,
    owner: String,
}

impl Hit {
    /// Line-number-free identity, so an edit above the declaration does not churn the baseline.
    fn key(&self) -> String {
        format!("{} {} {}", self.path, self.kind, self.ident)
    }
}

/// `name = "x"` values under `[package]` and `[lib]` of one manifest.
fn manifest_names(text: &str) -> Vec<String> {
    let mut section = String::new();
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            section = line.to_string();
            continue;
        }
        if section != "[package]" && section != "[lib]" {
            continue;
        }
        let Some(rest) = line.strip_prefix("name") else {
            continue;
        };
        let Some(value) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"');
        if !value.is_empty() {
            out.push(value.replace('-', "_"));
        }
    }
    out
}

/// `trueno_graph` → `TruenoGraph`.
fn camel(snake: &str) -> String {
    snake
        .split('_')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut c = p.chars();
            c.next()
                .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        })
        .collect()
}

/// The `(kind, identifier)` a line declares with `mod`/`struct`/`enum`/`trait`/`type`, if any.
fn declared_item(line: &str) -> Option<(&'static str, &str)> {
    let mut s = line.trim_start();
    if s.starts_with("//") {
        return None;
    }
    if let Some(rest) = s.strip_prefix("pub") {
        s = rest.trim_start();
        if s.starts_with('(') {
            s = s[s.find(')')? + 1..].trim_start();
        }
    }
    let (kind, rest) = [("mod", "mod"), ("struct", "type"), ("enum", "type"), ("trait", "type"), ("type", "type")]
        .into_iter()
        .find_map(|(kw, kind)| s.strip_prefix(kw).map(|r| (kind, r)))?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    let ident = &rest[..end];
    let tail = rest[end..].trim_start();
    let declares = if kind == "mod" {
        tail.starts_with('{') || tail.starts_with(';')
    } else {
        tail.starts_with(['{', ';', '(', '<', '=', ':'])
    };
    (!ident.is_empty() && declares).then_some((kind, ident))
}

/// Every module or type in one source file that wears the name of a library it is outside of.
fn scan_file(path: &Path, text: &str, owners: &Owners) -> Vec<Hit> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let Some((kind, ident)) = declared_item(line) else {
            continue;
        };
        let Some(dirs) = owners.get(&format!("{kind}:{ident}")) else {
            continue;
        };
        if dirs.iter().any(|d| path.starts_with(d)) {
            continue;
        }
        out.push(Hit {
            path: path.display().to_string(),
            kind,
            ident: ident.to_string(),
            line: i + 1,
            owner: dirs.iter().map(|d| format!("{}/", d.display())).collect::<Vec<_>>().join(", "),
        });
    }
    out
}

/// Library names declared by the manifests among `files` (relative paths, read under `root`).
fn owners(root: &Path, files: &[PathBuf]) -> Result<Owners, String> {
    let mut map = Owners::new();
    for f in files.iter().filter(|f| f.file_name().is_some_and(|n| n == "Cargo.toml")) {
        // The root facade contains every path, so it cannot own a name: its library shares
        // its name with the crate that implements it, and that crate is the owner.
        let Some(dir) = f.parent().filter(|d| !d.as_os_str().is_empty()) else {
            continue;
        };
        let text = std::fs::read_to_string(root.join(f)).map_err(|e| format!("{}: {e}", f.display()))?;
        for name in manifest_names(&text) {
            for key in [format!("mod:{name}"), format!("type:{}", camel(&name))] {
                let dirs = map.entry(key).or_default();
                if !dirs.contains(&dir.to_path_buf()) {
                    dirs.push(dir.to_path_buf());
                }
            }
        }
    }
    Ok(map)
}

/// NEW hits (not in the baseline) and STALE baseline keys (no longer hit).
fn compare(hits: &[Hit], baseline: &BTreeSet<String>) -> (Vec<Hit>, Vec<String>) {
    let keys: BTreeSet<String> = hits.iter().map(Hit::key).collect();
    let new = hits.iter().filter(|h| !baseline.contains(&h.key())).cloned().collect();
    let stale = baseline.iter().filter(|k| !keys.contains(*k)).cloned().collect();
    (new, stale)
}

fn read_baseline(path: &Path) -> Result<BTreeSet<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect())
}

fn run() -> Result<bool, String> {
    let mut args = std::env::args_os().skip(1);
    let root = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: no_mock_named_real <repo-root> [--baseline FILE] [--print-baseline] < git-ls-files-z")?;
    let mut baseline = BTreeSet::new();
    let mut print_baseline = false;
    while let Some(a) = args.next() {
        match a.to_str() {
            Some("--baseline") => {
                let p = args.next().ok_or("--baseline needs a FILE")?;
                baseline = read_baseline(Path::new(&p))?;
            }
            Some("--print-baseline") => print_baseline = true,
            _ => return Err(format!("unknown argument {a:?}")),
        }
    }
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).map_err(|e| e.to_string())?;
    let files: Vec<PathBuf> = input
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| PathBuf::from(String::from_utf8_lossy(s).into_owned()))
        .collect();
    let owners = owners(&root, &files)?;
    if owners.is_empty() {
        return Err("no crate Cargo.toml in the file list: nothing to judge against".into());
    }
    let mut hits = Vec::new();
    for f in files.iter().filter(|f| f.extension().is_some_and(|e| e == "rs")) {
        if let Ok(text) = std::fs::read_to_string(root.join(f)) {
            hits.extend(scan_file(f, &text, &owners));
        }
    }
    hits.sort();
    if print_baseline {
        for h in &hits {
            println!("{}", h.key());
        }
        return Ok(true);
    }
    let (new, stale) = compare(&hits, &baseline);
    for h in &hits {
        let tag = if new.contains(h) { "NEW  " } else { "KNOWN" };
        println!(
            "{tag} {}:{}: {} {} — `{}` is the library of {}; outside it, name a mock as a mock (e.g. mock_{})",
            h.path, h.line, h.kind, h.ident, h.ident, h.owner, h.ident.to_ascii_lowercase()
        );
    }
    for k in &stale {
        println!("STALE {k} — no longer declared; delete the baseline line (the baseline only shrinks)");
    }
    println!(
        "no-mock-named-real-v1: {} declared, {} known, {} new, {} stale",
        hits.len(),
        hits.len() - new.len(),
        new.len(),
        stale.len()
    );
    Ok(new.is_empty() && stale.is_empty())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("no-mock-named-real-v1: cannot judge: {e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned() -> Owners {
        let mut o = Owners::new();
        for (name, dir) in [("trueno", "crates/aprender-compute"), ("renacer", "crates/aprender-profile")] {
            o.insert(format!("mod:{name}"), vec![PathBuf::from(dir)]);
            o.insert(format!("type:{}", camel(name)), vec![PathBuf::from(dir)]);
        }
        o
    }

    #[test]
    fn manifest_names_reads_package_and_lib_only() {
        let m = "[package]\nname = \"aprender-compute\"\n[lib]\nname = \"trueno\"\n[dependencies]\nname = \"x\"\n";
        assert_eq!(manifest_names(m), vec!["aprender_compute", "trueno"]);
        assert_eq!(camel("trueno_graph"), "TruenoGraph");
    }

    /// Must-match / must-not-match table for the declaration matcher (CLAUDE.md rule 7).
    #[test]
    fn declared_item_case_table() {
        let must_match = [
            ("mod trueno {", ("mod", "trueno")),
            ("pub mod trueno {", ("mod", "trueno")),
            ("    pub(crate) mod trueno;", ("mod", "trueno")),
            ("pub(in crate::a) mod trueno{", ("mod", "trueno")),
            ("mod renacer;", ("mod", "renacer")),
            ("pub struct Trueno {", ("type", "Trueno")),
            ("struct Trueno;", ("type", "Trueno")),
            ("pub(crate) struct Trueno(u8);", ("type", "Trueno")),
            ("enum Trueno {", ("type", "Trueno")),
            ("pub trait Trueno: Send {", ("type", "Trueno")),
            ("type Trueno = u8;", ("type", "Trueno")),
            ("struct Trueno<T> {", ("type", "Trueno")),
        ];
        for (line, want) in must_match {
            assert_eq!(declared_item(line), Some(want), "must match: {line}");
        }
        let must_not = [
            "// pub mod trueno {",
            "use trueno::Vector;",
            "modtrueno {",
            "mod trueno_mock {",
            "let mod_trueno = 1;",
            "pub fn mod trueno",
            "structure Trueno {",
            "impl Trueno for X {",
            "let x: Trueno = y;",
        ];
        for line in must_not {
            let got = declared_item(line).map(|(_, i)| i);
            assert!(got != Some("trueno") && got != Some("Trueno"), "must not match: {line} -> {got:?}");
        }
    }

    /// FALSIFY-NMR-001: a planted `mod trueno` in a test outside aprender-compute is NEW (RED).
    #[test]
    fn planted_mod_trueno_in_a_test_is_red() {
        let hits = scan_file(
            Path::new("crates/aprender-serve/tests/planted.rs"),
            "pub mod trueno {\n    pub fn matmul() {}\n}\n",
            &owned(),
        );
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].key(), "crates/aprender-serve/tests/planted.rs mod trueno");
        let (new, stale) = compare(&hits, &BTreeSet::new());
        assert_eq!((new.len(), stale.len()), (1, 0));
    }

    /// FALSIFY-NMR-002: the renamed mock and the owning crate's own module are GREEN.
    #[test]
    fn mock_named_as_mock_and_owner_module_are_green() {
        let o = owned();
        assert!(scan_file(Path::new("crates/aprender-serve/tests/c.rs"), "pub mod mock_trace {\n}\n", &o).is_empty());
        assert!(scan_file(Path::new("crates/aprender-compute/src/lib.rs"), "mod trueno;\n", &o).is_empty());
    }

    /// FALSIFY-NMR-003: the baseline only shrinks — a removed declaration leaves a STALE line.
    #[test]
    fn baseline_line_without_a_declaration_is_stale() {
        let baseline: BTreeSet<String> = ["crates/x/src/lib.rs mod trueno".to_string()].into();
        let (new, stale) = compare(&[], &baseline);
        assert!(new.is_empty());
        assert_eq!(stale, vec!["crates/x/src/lib.rs mod trueno".to_string()]);
    }
}
