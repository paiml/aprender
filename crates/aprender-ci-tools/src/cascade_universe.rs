//! Port of `scripts/lib/cascade_universe.py`: the set of crates a release cascade must ship,
//! read from EVERY workspace this repository publishes out of (aprender#2559).
//!
//! One TSV row per publishable crate, sorted by name:
//! `<name>\t<version>\t<manifest path>\t<workspace root>`, or the names alone with `--names`.
//! A crate whose `publish` is `[]` (`publish = false`) is not in the universe.
//!
//! Exit codes, as the original's:
//! - 0: the universe printed.
//! - 2: `cargo metadata` failed or printed nothing for a workspace, a name is publishable
//!   from two workspaces, or fewer than [`MIN_CRATES`] crates were enumerated.
//! - 1: anything the original raised on (cargo missing, bad JSON, a package without
//!   `name`/`version`/`manifest_path`, `packages` not a list). Stdout is empty.
//!
//! `cargo` is run from `PATH`, as the original runs it, so a test can plant a stub.
//!
//! Known divergences, none reachable from cargo's metadata schema:
//! - JSON that `json.loads` takes and `serde_json` refuses (`NaN`, `Infinity`, a lone
//!   surrogate, UTF-16/32 or a BOM) exits 1 here.
//! - A `name` that is not a string exits 1 here. Python hashes it, so `1` and `1.0` and
//!   `True` would be one key there.
//! - `manifest_path` values are compared exactly, so `1` and `1.0` differ here.

use crate::dag_status::{py_str, Py};
use std::collections::BTreeMap;
use std::process::Command;

/// Every workspace this repository publishes out of, relative to the repo root.
/// `crates/facades` is `exclude`d from the root workspace, so cargo never volunteers it.
pub const WORKSPACES: [&str; 2] = [".", "crates/facades"];

/// A universe smaller than this means the enumeration broke, not that the repo shrank.
pub const MIN_CRATES: usize = 70;

/// What the original would have printed and returned.
#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub stdout: String,
    pub stderr: String,
    pub code: u8,
}

fn fail(code: u8, stderr: String) -> Outcome {
    Outcome {
        stdout: String::new(),
        stderr,
        code,
    }
}

/// `os.path.join(a, b)` (posix).
fn py_join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        b.to_owned()
    } else if a.is_empty() || a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

/// `os.path.normpath(p)` (posix): lexical, symlinks are not resolved.
pub fn py_normpath(p: &str) -> String {
    let lead = lead_of(p);
    let mut parts: Vec<&str> = Vec::new();
    for c in p.split('/') {
        push_part(&mut parts, c, lead.is_empty());
    }
    let out = format!("{lead}{}", parts.join("/"));
    if out.is_empty() {
        ".".to_owned()
    } else {
        out
    }
}

/// The slashes `normpath` keeps: POSIX gives exactly two leading slashes their own meaning.
fn lead_of(p: &str) -> &'static str {
    if p.starts_with("//") && !p.starts_with("///") {
        "//"
    } else if p.starts_with('/') {
        "/"
    } else {
        ""
    }
}

/// One component of `normpath`'s walk. `..` climbs out of a relative path, never past `/`.
fn push_part<'a>(parts: &mut Vec<&'a str>, c: &'a str, relative: bool) {
    match c {
        "" | "." => {}
        ".." if parts.last().is_some_and(|l| *l != "..") => {
            parts.pop();
        }
        ".." if !relative => {}
        other => parts.push(other),
    }
}

/// `os.path.abspath(p)`: `normpath(join(getcwd(), p))`.
fn py_abspath(p: &str) -> Result<String, String> {
    if p.starts_with('/') {
        return Ok(py_normpath(p));
    }
    let cwd = std::env::current_dir().map_err(|e| format!("getcwd: {e}"))?;
    Ok(py_normpath(&py_join(&cwd.to_string_lossy(), p)))
}

/// `bytes.strip()` is empty: only ASCII whitespace.
fn blank(b: &[u8]) -> bool {
    b.iter()
        .all(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c))
}

/// `metadata()`: `Ok(None)` is the original's `None` (exit 2), `Err` is a raise (exit 1).
fn metadata(repo_root: &str, ws: &str) -> Result<Result<Py, String>, String> {
    let manifest = py_join(&py_join(repo_root, ws), "Cargo.toml");
    let out = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .arg("--manifest-path")
        .arg(&manifest)
        .output()
        .map_err(|e| format!("cascade_universe: cannot run cargo: {e}"))?;
    if !out.status.success() || blank(&out.stdout) {
        return Ok(Err(format!(
            "cascade_universe: cargo metadata failed for {manifest}\n{}\n",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    serde_json::from_slice::<Py>(&out.stdout)
        .map(Ok)
        .map_err(|e| format!("cascade_universe: cargo metadata for {manifest}: {e}"))
}

/// The packages `for pkg in doc["packages"]` visits. An empty dict or string iterates
/// nothing; a non-empty one yields keys or characters, whose `.get` raises.
fn packages(doc: &Py) -> Result<&[Py], String> {
    let Some(p) = doc.get("packages") else {
        return Err("cascade_universe: metadata has no `packages`".to_owned());
    };
    match p {
        Py::List(items) => Ok(items),
        Py::Dict(d) if d.is_empty() => Ok(&[]),
        Py::Str(s) if s.is_empty() => Ok(&[]),
        _ => Err("cascade_universe: `packages` is not a list of objects".to_owned()),
    }
}

fn key<'a>(pkg: &'a Py, k: &str) -> Result<&'a Py, String> {
    pkg.get(k)
        .ok_or_else(|| format!("cascade_universe: a package has no `{k}`"))
}

type Row = (Py, Py, String);

/// `rows()`: `Ok(Err(msg))` is the original's `None`.
fn rows(repo_root: &str) -> Result<Result<BTreeMap<String, Row>, String>, String> {
    let mut seen: BTreeMap<String, Row> = BTreeMap::new();
    for ws in WORKSPACES {
        let doc = match metadata(repo_root, ws)? {
            Ok(doc) => doc,
            Err(msg) => return Ok(Err(msg)),
        };
        let ws_root = py_normpath(&py_join(&py_abspath(repo_root)?, ws));
        for pkg in packages(&doc)? {
            if let Err(msg) = add(&mut seen, pkg, &ws_root)? {
                return Ok(Err(msg));
            }
        }
    }
    Ok(Ok(seen))
}

/// One iteration of the original's package loop.
fn add(
    seen: &mut BTreeMap<String, Row>,
    pkg: &Py,
    ws_root: &str,
) -> Result<Result<(), String>, String> {
    if !matches!(pkg, Py::Dict(_)) {
        return Err("cascade_universe: a package is not an object".to_owned());
    }
    if matches!(pkg.get("publish"), Some(Py::List(l)) if l.is_empty()) {
        return Ok(Ok(()));
    }
    let Py::Str(name) = key(pkg, "name")? else {
        return Err("cascade_universe: a package name is not a string".to_owned());
    };
    if let Some((_, prev, _)) = seen.get(name) {
        let manifest = key(pkg, "manifest_path")?;
        if prev != manifest {
            return Ok(Err(format!(
                "cascade_universe: `{name}` is publishable from two workspaces:\n  {}\n  {}\n",
                py_str(prev),
                py_str(manifest)
            )));
        }
    }
    let row = (
        key(pkg, "version")?.clone(),
        key(pkg, "manifest_path")?.clone(),
        ws_root.to_owned(),
    );
    seen.insert(name.clone(), row);
    Ok(Ok(()))
}

/// `main(argv)` minus the program name.
pub fn run(args: &[String]) -> Outcome {
    let names_only = args.iter().any(|a| a == "--names");
    let repo_root = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map_or(".", String::as_str);
    let got = match rows(repo_root) {
        Err(raised) => return fail(1, raised),
        Ok(Err(msg)) => return fail(2, msg),
        Ok(Ok(got)) => got,
    };
    if got.len() < MIN_CRATES {
        return fail(
            2,
            format!(
                "cascade_universe: enumerated only {} publishable crate(s), expected at least \
                 {MIN_CRATES}. The ENUMERATION is broken, not the repo.\n",
                got.len()
            ),
        );
    }
    let mut stdout = String::new();
    for (name, (version, manifest, ws_root)) in &got {
        if names_only {
            stdout.push_str(&format!("{name}\n"));
        } else {
            stdout.push_str(&format!(
                "{name}\t{}\t{}\t{ws_root}\n",
                py_str(version),
                py_str(manifest)
            ));
        }
    }
    Outcome {
        stdout,
        stderr: String::new(),
        code: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normpath_matches_posixpath() {
        let cases = [
            ("", "."),
            (".", "."),
            ("/", "/"),
            ("//", "//"),
            ("///", "/"),
            ("//a/./b", "//a/b"),
            ("/a/b/../c", "/a/c"),
            ("/../a", "/a"),
            ("a/../..", ".."),
            ("../a/..", ".."),
            ("a/b/", "a/b"),
            ("/a/./.", "/a"),
            ("a//b", "a/b"),
        ];
        for (p, want) in cases {
            assert_eq!(py_normpath(p), want, "normpath({p:?})");
        }
    }

    #[test]
    fn join_matches_posixpath() {
        assert_eq!(py_join("", "."), ".");
        assert_eq!(py_join("a/", "."), "a/.");
        assert_eq!(py_join("a", "crates/facades"), "a/crates/facades");
        assert_eq!(py_join("a", "/x"), "/x");
    }

    #[test]
    fn packages_shapes() {
        let doc = |j: &str| -> Py { serde_json::from_str(j).expect("json") };
        assert_eq!(packages(&doc(r#"{"packages":[]}"#)).map(<[Py]>::len), Ok(0));
        assert_eq!(packages(&doc(r#"{"packages":{}}"#)).map(<[Py]>::len), Ok(0));
        assert_eq!(packages(&doc(r#"{"packages":""}"#)).map(<[Py]>::len), Ok(0));
        assert!(packages(&doc(r#"{"packages":{"a":1}}"#)).is_err());
        assert!(packages(&doc(r#"{"packages":null}"#)).is_err());
        assert!(packages(&doc(r#"{"x":[]}"#)).is_err());
    }

    #[test]
    fn blank_is_bytes_strip() {
        assert!(blank(b""));
        assert!(blank(b" \t\r\n\x0b\x0c"));
        assert!(!blank(b" {}"));
        assert!(!blank("\u{a0}".as_bytes()));
    }
}
