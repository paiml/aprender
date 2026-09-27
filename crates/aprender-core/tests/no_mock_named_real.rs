//! `no-mock-named-real-v1` (TRACE-001 TR-04, aprender#4559).
//!
//! A module or type named after a real workspace crate, defined outside that
//! crate, reads as the real thing. `aprender-serve/tests/modality_matrix/common.rs`
//! carried `pub mod renacer { … }`, a thread-local span recorder, and its test
//! printed "QA-A08 PASS: renacer::capture() API works correctly" — renacer (the
//! in-tree `aprender-profile` syscall tracer) was never run.
//!
//! The rule, per `.rs` file tracked by git:
//! - `mod NAME` where NAME is the `[lib]` name of a workspace package other than
//!   the file's own, and the module body never names the real crate
//!   (`NAME::…`, `use NAME`, `extern crate NAME`) → violation. A module that
//!   does name the real crate is an adapter, not a mock.
//! - `struct|enum|trait|type NAME` where NAME is the capitalised form of a
//!   single-word lib name (`Trueno`, `Renacer`) → violation (0 at HEAD, so no
//!   adapter exemption). Multi-word libs are skipped: `apr_format` would flag
//!   the domain enum `AprFormat`.
//! - Hits under `tests/` can never be allowlisted. `src/` hits that predate the
//!   rule are listed in [`ALLOW`] with the reason each stays; an entry that no
//!   longer hits is itself a failure, so the list can only shrink.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Pre-existing `src/` modules named after a workspace crate that never name it.
/// Renaming any of them changes a public path, so each is its own ticket; the
/// reason says why it is not a test-side mock.
const ALLOW: &[(&str, &str, &str)] = &[
    (
        "crates/apr-cli/src/commands/mod.rs",
        "cbtop",
        "the `apr cbtop` subcommand module (CLI verb name)",
    ),
    (
        "crates/aprender-orchestrate/src/agent/memory/mod.rs",
        "trueno",
        "memory substrate built on trueno_rag, not trueno",
    ),
    (
        "crates/aprender-present-yaml/src/lib.rs",
        "pacha",
        "pacha:// URI loader (protocol name), public path",
    ),
    (
        "crates/aprender-train/src/ecosystem/mod.rs",
        "batuta",
        "ENT-030/031 pricing integration, public path entrenar::ecosystem::batuta",
    ),
    (
        "crates/aprender-train/src/ecosystem/mod.rs",
        "realizar",
        "ENT-032 GGUF export integration, public path entrenar::ecosystem::realizar",
    ),
    (
        "crates/aprender-train/src/storage/mod.rs",
        "trueno",
        "ENT-001 TruenoDB backend, public path entrenar::storage::trueno",
    ),
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Hit {
    file: String,
    line: usize,
    name: String,
    kind: &'static str,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root must resolve from crates/aprender-core")
}

/// Workspace packages as (package-dir relative to root, lib name).
fn workspace_libs(root: &Path) -> Vec<(String, String)> {
    let out = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .expect("cargo metadata must run");
    assert!(out.status.success(), "cargo metadata failed");
    let meta: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("cargo metadata emits JSON");
    let mut libs = Vec::new();
    for pkg in meta["packages"].as_array().expect("packages array") {
        let manifest = PathBuf::from(pkg["manifest_path"].as_str().expect("manifest_path"));
        let dir = manifest.parent().expect("manifest has a parent");
        let rel = dir
            .strip_prefix(root)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        for t in pkg["targets"].as_array().expect("targets array") {
            let kinds = t["kind"].as_array().expect("kind array");
            if kinds.iter().any(|k| k == "lib" || k == "proc-macro") {
                libs.push((rel.clone(), t["name"].as_str().expect("name").to_string()));
            }
        }
    }
    libs
}

/// The package dir owning `file`: the longest package-dir prefix.
fn owner_of<'a>(file: &str, libs: &'a [(String, String)]) -> Option<&'a str> {
    libs.iter()
        .map(|(dir, _)| dir.as_str())
        .filter(|dir| dir.is_empty() || file.starts_with(&format!("{dir}/")))
        .max_by_key(|dir| dir.len())
}

fn upper_camel(lib: &str) -> String {
    lib.split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next()
                .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        })
        .collect()
}

/// `mod NAME {` / `mod NAME;` (any visibility) at the start of a trimmed line.
fn decl<'a>(line: &'a str, keyword: &str) -> Option<&'a str> {
    let mut rest = line.trim_start();
    if let Some(r) = rest.strip_prefix("pub") {
        rest = r.trim_start();
        if let Some(r) = rest.strip_prefix('(') {
            rest = r.split_once(')')?.1.trim_start();
        }
    }
    let rest = rest
        .strip_prefix(keyword)?
        .strip_prefix(char::is_whitespace)?;
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

/// True if `text` names crate `name` as a path root: `name::`, `use name`, or
/// `extern crate name`, and not as `self::name` / `super::name` / `crate::name`.
fn names_crate(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let mut from = 0;
    while let Some(i) = text[from..].find(name).map(|i| i + from) {
        let after = &text[i + name.len()..];
        let before_ok = i == 0 || {
            let b = bytes[i - 1];
            !(b.is_ascii_alphanumeric() || b == b'_' || b == b':')
        };
        let prefix = text[..i].trim_end();
        let is_path = after.starts_with("::")
            || ((prefix.ends_with("use") || prefix.ends_with("extern crate"))
                && !after.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_'));
        if before_ok && is_path {
            return true;
        }
        from = i + name.len();
    }
    false
}

/// Text of the inline module opened on `lines[start]` (brace-matched).
fn inline_body(lines: &[&str], start: usize) -> String {
    let mut depth = 0i64;
    let mut seen_open = false;
    let mut body = String::new();
    for l in &lines[start..] {
        body.push_str(l);
        body.push('\n');
        for c in l.chars() {
            match c {
                '{' => {
                    depth += 1;
                    seen_open = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if seen_open && depth <= 0 {
            break;
        }
    }
    body
}

/// Every `.rs` text under the out-of-line module `name` declared in `file`.
fn file_module_body(root: &Path, file: &str, name: &str) -> String {
    let path = root.join(file);
    let parent = path.parent().expect("file has a parent");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let base = if matches!(stem, "mod" | "lib" | "main") {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    };
    let mut body = String::new();
    let single = base.join(format!("{name}.rs"));
    if let Ok(s) = std::fs::read_to_string(&single) {
        body.push_str(&s);
    }
    let dir = base.join(name);
    if dir.is_dir() {
        let mut stack = vec![dir];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).expect("read module dir").flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    body.push_str(&std::fs::read_to_string(&p).unwrap_or_default());
                }
            }
        }
    }
    body
}

/// Scan one file. `module_body` resolves an out-of-line `mod NAME;` to its text.
fn scan_source(
    file: &str,
    src: &str,
    libs: &[(String, String)],
    module_body: &dyn Fn(&str) -> String,
) -> Vec<Hit> {
    let Some(own) = owner_of(file, libs) else {
        return Vec::new();
    };
    let foreign: HashSet<&str> = libs
        .iter()
        .filter(|(dir, _)| dir != own)
        .map(|(_, lib)| lib.as_str())
        .collect();
    let own_libs: Vec<&str> = libs
        .iter()
        .filter(|(dir, _)| dir == own)
        .map(|(_, l)| l.as_str())
        .collect();
    let camel: HashSet<String> = foreign
        .iter()
        .filter(|l| !own_libs.contains(l) && !l.contains('_'))
        .map(|l| upper_camel(l))
        .collect();
    let lines: Vec<&str> = src.lines().collect();
    let mut hits = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if let Some(name) = decl(l, "mod") {
            if foreign.contains(name) && !own_libs.contains(&name) {
                let rest = l.trim_end();
                let body = if rest.ends_with(';') {
                    module_body(name)
                } else {
                    inline_body(&lines, i)
                };
                if !names_crate(&body, name) {
                    hits.push(Hit {
                        file: file.to_string(),
                        line: i + 1,
                        name: name.to_string(),
                        kind: "mod",
                    });
                }
            }
        }
        for kw in ["struct", "enum", "trait", "type"] {
            if let Some(name) = decl(l, kw) {
                if camel.contains(name) {
                    hits.push(Hit {
                        file: file.to_string(),
                        line: i + 1,
                        name: name.to_string(),
                        kind: "type",
                    });
                }
            }
        }
    }
    hits
}

fn scan_head(root: &Path, libs: &[(String, String)]) -> Vec<Hit> {
    let out = Command::new("git")
        .args(["ls-files", "-z", "--", "*.rs"])
        .current_dir(root)
        .output()
        .expect("git ls-files must run");
    assert!(out.status.success(), "git ls-files failed");
    let files = String::from_utf8(out.stdout).expect("utf-8 paths");
    let mut hits = Vec::new();
    let mut scanned = 0usize;
    for file in files.split('\0').filter(|f| !f.is_empty()) {
        let Ok(src) = std::fs::read_to_string(root.join(file)) else {
            continue;
        };
        scanned += 1;
        hits.extend(scan_source(file, &src, libs, &|name| {
            file_module_body(root, file, name)
        }));
    }
    assert!(scanned > 1000, "scanned only {scanned} .rs files — vacuous");
    hits
}

fn fixture_libs() -> Vec<(String, String)> {
    [
        ("crates/aprender-compute", "trueno"),
        ("crates/aprender-serve", "realizar"),
        ("crates/aprender-profile", "renacer"),
    ]
    .iter()
    .map(|(d, l)| ((*d).to_string(), (*l).to_string()))
    .collect()
}

fn no_file(_: &str) -> String {
    String::new()
}

/// FALSIFY-NMNR-001: a planted `mod trueno` mock in a test turns RED.
#[test]
fn falsify_nmnr_001_planted_mock_is_red() {
    let src = "use std::cell::RefCell;\n\
               pub mod trueno {\n    pub fn capture() -> u32 { 7 }\n}\n";
    let hits = scan_source(
        "crates/aprender-serve/tests/x.rs",
        src,
        &fixture_libs(),
        &no_file,
    );
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!((hits[0].name.as_str(), hits[0].line), ("trueno", 2));

    let out_of_line = scan_source(
        "crates/aprender-serve/tests/x.rs",
        "mod renacer;\n",
        &fixture_libs(),
        &|_| "pub fn capture() {}\n".to_string(),
    );
    assert_eq!(out_of_line.len(), 1, "{out_of_line:?}");

    let typ = scan_source(
        "crates/aprender-serve/tests/x.rs",
        "pub(crate) struct Renacer;\n",
        &fixture_libs(),
        &no_file,
    );
    assert_eq!(typ.len(), 1, "{typ:?}");
}

/// FALSIFY-NMNR-002: what must stay GREEN — the honest name, an adapter that
/// names the real crate, a crate's own name, and a self-path look-alike.
#[test]
fn falsify_nmnr_002_honest_names_are_green() {
    let libs = fixture_libs();
    let cases = [
        (
            "crates/aprender-serve/tests/x.rs",
            "pub mod mock_trace {\n}\n",
        ),
        (
            "crates/aprender-serve/src/a.rs",
            "pub mod trueno {\n    use trueno::Vector;\n}\n",
        ),
        (
            "crates/aprender-serve/src/a.rs",
            "mod renacer {\n    pub fn f() { renacer::run(); }\n}\n",
        ),
        (
            "crates/aprender-compute/src/lib.rs",
            "pub mod trueno {\n}\n",
        ),
        ("crates/aprender-compute/src/lib.rs", "pub struct Trueno;\n"),
        ("crates/aprender-serve/src/a.rs", "// mod trueno {}\n"),
    ];
    for (file, src) in cases {
        let hits = scan_source(file, src, &libs, &no_file);
        assert!(hits.is_empty(), "{file}: {src:?} -> {hits:?}");
    }
    let self_path = scan_source(
        "crates/aprender-serve/tests/x.rs",
        "mod trueno {\n    use super::trueno::X;\n    fn f() { crate::trueno::g(); }\n}\n",
        &libs,
        &no_file,
    );
    assert_eq!(
        self_path.len(),
        1,
        "self/super/crate paths are not the real crate"
    );
}

/// FALSIFY-NMNR-003: 0 un-allowlisted hits at HEAD, none under `tests/`, and no
/// stale allowlist entry.
#[test]
fn falsify_nmnr_003_zero_hits_at_head() {
    let root = workspace_root();
    let libs = workspace_libs(&root);
    assert!(
        libs.len() > 50,
        "only {} workspace libs — vacuous",
        libs.len()
    );
    let hits = scan_head(&root, &libs);

    let in_tests: Vec<&Hit> = hits.iter().filter(|h| h.file.contains("/tests/")).collect();
    assert!(
        in_tests.is_empty(),
        "no-mock-named-real-v1: test code defines a module/type named after a real \
         workspace crate — rename it (e.g. `mock_trace`): {in_tests:#?}"
    );
    let unlisted: Vec<&Hit> = hits
        .iter()
        .filter(|h| !ALLOW.iter().any(|(f, n, _)| *f == h.file && *n == h.name))
        .collect();
    assert!(
        unlisted.is_empty(),
        "no-mock-named-real-v1: module/type named after a real workspace crate that \
         never names it: {unlisted:#?}"
    );
    let stale: Vec<&(&str, &str, &str)> = ALLOW
        .iter()
        .filter(|(f, n, _)| !hits.iter().any(|h| h.file == *f && h.name == *n))
        .collect();
    assert!(
        stale.is_empty(),
        "stale ALLOW entries — delete them: {stale:#?}"
    );
}
