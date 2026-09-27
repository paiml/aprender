//! `no-mock-named-real-v1` (TRACE-001 TR-04, R-5, aprender#4559).
//!
//! A mock must be named as a mock. `tests/modality_matrix/common.rs` defined an
//! in-process `mod renacer`, and its test printed "renacer::capture() API works"
//! while renacer never ran: a reader, and a grep for tracer coverage, saw the real
//! tracer. That is F5 (false telemetry) by naming.
//!
//! RED when a module or type named after a workspace **lib crate** is defined
//! outside that crate's directory in a mock shape:
//! - an inline module `mod <crate> {` anywhere, or
//! - a module `mod <crate>` or a type `struct|enum|trait|type|union <Crate>` in
//!   test code (a path under `tests/` or `benches/`).
//!
//! Exempt, on purpose: a FILE module in `src/` (`pub mod realizar;` backed by
//! `driver/realizar.rs`). Those are adapters that call the real crate, not stand-ins
//! for it; eleven exist at HEAD (batuta's `driver::realizar`, entrenar's
//! `ecosystem::batuta`, …) and none is a mock.
//!
//! R-6: the scan of the tree is REPORT-ONLY until first green plus 7 green nights;
//! it prints its hits and passes. Set `NO_MOCK_NAMED_REAL_ENFORCE=1` to make it
//! fail. The case table below is always enforced: it tests the lint, not the tree.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A workspace lib crate: its `[lib] name` and its directory relative to the root.
struct LibCrate {
    name: String,
    dir: String,
}

fn camel(name: &str) -> String {
    name.split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next()
                .map(|f| f.to_ascii_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        })
        .collect()
}

fn is_test_code(rel: &str) -> bool {
    let p = format!("/{rel}");
    p.contains("/tests/") || p.contains("/benches/")
}

/// `pub(crate) mod x {` -> ("mod", "x", rest). Strips a leading visibility.
fn item<'a>(line: &'a str, kw: &str) -> Option<(&'a str, &'a str)> {
    let mut t = line.trim_start();
    if let Some(r) = t.strip_prefix("pub") {
        t = r.trim_start();
        if t.starts_with('(') {
            t = t[t.find(')')? + 1..].trim_start();
        }
    }
    let rest = t.strip_prefix(kw)?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(rest.len());
    Some((&rest[..end], rest[end..].trim_start()))
}

/// Every mock-shaped definition in `src` (at repo path `rel`) named after a lib crate
/// that does not live at `rel`.
fn violations(rel: &str, src: &str, libs: &[LibCrate]) -> Vec<String> {
    let test_code = is_test_code(rel);
    let foreign = |lib: &LibCrate| {
        !(rel.starts_with(&format!("{}/", lib.dir))
            || (lib.dir.is_empty() && rel.starts_with("src/")))
    };
    let mut out = Vec::new();
    for (n, line) in src.lines().enumerate() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        // Parse the line once; only a line that defines an item is matched against
        // the crate list.
        if let Some((name, rest)) = item(line, "mod") {
            if rest.starts_with('{') || test_code {
                for lib in libs.iter().filter(|l| l.name == name && foreign(l)) {
                    out.push(format!(
                        "{rel}:{}: mod {name} (crate lives in {})",
                        n + 1,
                        lib.dir
                    ));
                }
            }
        }
        if !test_code {
            continue;
        }
        for kw in ["struct", "enum", "trait", "type", "union"] {
            let Some((name, _)) = item(line, kw) else {
                continue;
            };
            for lib in libs.iter().filter(|l| camel(&l.name) == name && foreign(l)) {
                out.push(format!("{rel}:{}: {kw} {name} (crate {})", n + 1, lib.name));
            }
        }
    }
    out
}

fn workspace_libs(root: &Path) -> Vec<LibCrate> {
    let out = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .expect("cargo metadata must run");
    assert!(out.status.success(), "cargo metadata failed");
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout).expect("metadata json");
    let mut libs = Vec::new();
    for pkg in meta["packages"].as_array().expect("packages") {
        let manifest = PathBuf::from(pkg["manifest_path"].as_str().expect("manifest_path"));
        let dir = manifest
            .parent()
            .and_then(|d| d.strip_prefix(root).ok())
            .map(|d| d.to_string_lossy().into_owned())
            .expect("member under the workspace root");
        for t in pkg["targets"].as_array().expect("targets") {
            let kinds = t["kind"].as_array().expect("kind");
            if kinds
                .iter()
                .any(|k| k == "lib" || k == "proc-macro" || k == "rlib")
            {
                let name = t["name"].as_str().expect("name").replace('-', "_");
                libs.push(LibCrate {
                    name,
                    dir: dir.clone(),
                });
            }
        }
    }
    libs
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        if p.is_dir() {
            rust_files(&p, out);
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

fn libs_fixture() -> Vec<LibCrate> {
    [
        ("trueno", "crates/aprender-compute"),
        ("renacer", "crates/aprender-profile"),
        ("trueno_graph", "crates/aprender-graph"),
        ("realizar", "crates/aprender-serve"),
    ]
    .iter()
    .map(|(n, d)| LibCrate {
        name: (*n).into(),
        dir: (*d).into(),
    })
    .collect()
}

/// The lint's case table: every must-be-RED row is RED, every must-stay-GREEN row is
/// GREEN. Always enforced (it does not read the tree).
#[test]
fn falsify_tr04_no_mock_named_real_case_table() {
    let libs = libs_fixture();
    let red = [
        ("crates/aprender-serve/tests/x.rs", "mod trueno {\n}"),
        (
            "crates/aprender-serve/tests/x.rs",
            "pub mod trueno { pub fn f() {} }",
        ),
        (
            "crates/aprender-serve/tests/modality_matrix/common.rs",
            "pub mod renacer {",
        ),
        ("crates/aprender-core/tests/y.rs", "mod trueno;"),
        (
            "crates/aprender-core/benches/b.rs",
            "pub(crate) mod realizar {",
        ),
        ("crates/aprender-core/src/lib.rs", "    mod trueno {"),
        ("crates/aprender-core/tests/y.rs", "struct Trueno;"),
        (
            "crates/aprender-core/tests/y.rs",
            "pub enum TruenoGraph { A }",
        ),
        ("crates/aprender-core/tests/y.rs", "trait Renacer {}"),
        ("crates/aprender-core/tests/y.rs", "type Realizar = u8;"),
    ];
    for (rel, src) in red {
        assert!(
            !violations(rel, src, &libs).is_empty(),
            "must be RED: {rel}: {src}"
        );
    }
    let green = [
        (
            "crates/aprender-serve/tests/modality_matrix/common.rs",
            "pub mod mock_trace {",
        ),
        (
            "crates/aprender-orchestrate/src/agent/driver/mod.rs",
            "pub mod realizar;",
        ),
        ("crates/aprender-compute/tests/x.rs", "mod trueno {"),
        ("crates/aprender-core/tests/y.rs", "// mod trueno {"),
        ("crates/aprender-core/tests/y.rs", "mod trueno_mock {"),
        ("crates/aprender-core/tests/y.rs", "use trueno::Vector;"),
        ("crates/aprender-core/tests/y.rs", "struct TruenoMock;"),
        ("crates/aprender-core/tests/y.rs", "let modtrueno = 1;"),
        ("crates/aprender-core/src/lib.rs", "struct Trueno;"),
    ];
    for (rel, src) in green {
        assert!(
            violations(rel, src, &libs).is_empty(),
            "must stay GREEN: {rel}: {src}"
        );
    }
}

/// The tree at HEAD. REPORT-ONLY (R-6) unless `NO_MOCK_NAMED_REAL_ENFORCE=1`; the
/// scan must still reach the files it claims to scan.
#[test]
fn tr04_no_mock_named_real_at_head() {
    let root = super::workspace_root();
    let libs = workspace_libs(&root);
    assert!(
        libs.iter().any(|l| l.name == "renacer"),
        "renacer must be a lib crate"
    );
    let mut files = Vec::new();
    rust_files(&root.join("crates"), &mut files);
    rust_files(&root.join("src"), &mut files);
    let target = root.join("crates/aprender-serve/tests/modality_matrix/common.rs");
    assert!(
        files.contains(&target),
        "scan must reach {}",
        target.display()
    );
    let mut hits = Vec::new();
    for f in &files {
        let rel = f
            .strip_prefix(&root)
            .expect("under root")
            .to_string_lossy()
            .into_owned();
        if let Ok(src) = std::fs::read_to_string(f) {
            hits.extend(violations(&rel, &src, &libs));
        }
    }
    eprintln!(
        "no-mock-named-real-v1: {} files, {} hits (report-only, R-6)",
        files.len(),
        hits.len()
    );
    for h in &hits {
        eprintln!("  {h}");
    }
    if std::env::var("NO_MOCK_NAMED_REAL_ENFORCE").is_ok_and(|v| v == "1") {
        assert!(
            hits.is_empty(),
            "{} mock(s) named after a real crate",
            hits.len()
        );
    }
}
