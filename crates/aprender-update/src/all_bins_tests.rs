//! EPIC #4232 drift gate: every binary target of every workspace member runs
//! the update check. A new `[[bin]]` (or `src/main.rs`, `src/bin/*.rs`) that
//! does not call it turns CI's `--lib` run RED, naming the file.

use std::path::{Path, PathBuf};

/// What counts as wiring: the hook, or an entry point that reaches
/// `update_main` + `startup` (apr's two entry points go through apr-cli).
const WIRED: &[&str] = &[
    "sovereign_update::hook!(",
    "sovereign_update::entry(",
    "sovereign_update::update_main(",
    "apr_cli::update_or_check(",
    "apr_cli::cli_main()",
];

/// Binaries that are not a product: never installed, so never updated.
const NOT_A_PRODUCT: &[(&str, &str)] = &[(
    "crates/aprender-compute-xtask/src/main.rs",
    "`cargo xtask` build tooling, publish = false",
)];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The binary source files cargo builds for one member.
/// The path of one explicit `[[bin]]` entry: its `path`, else cargo's
/// `src/bin/<name>.rs` default.
fn explicit_bin_path(member: &Path, bin: &toml::Value) -> PathBuf {
    if let Some(p) = bin.get("path").and_then(toml::Value::as_str) {
        return member.join(p);
    }
    let name = bin
        .get("name")
        .and_then(toml::Value::as_str)
        .expect("bin name");
    member.join(format!("src/bin/{name}.rs"))
}

/// The targets cargo's autobins discovery finds: `src/main.rs`, plus every
/// `src/bin/*.rs` and `src/bin/*/main.rs`.
fn auto_bin_paths(member: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let main = member.join("src/main.rs");
    if main.is_file() {
        out.push(main);
    }
    let Ok(rd) = std::fs::read_dir(member.join("src/bin")) else {
        return out;
    };
    for p in rd.flatten().map(|e| e.path()) {
        if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        } else if p.join("main.rs").is_file() {
            out.push(p.join("main.rs"));
        }
    }
    out
}

fn bin_sources(member: &Path) -> Vec<PathBuf> {
    let text = std::fs::read_to_string(member.join("Cargo.toml")).expect("member manifest");
    let manifest: toml::Value = toml::from_str(&text).expect("member manifest parses");
    let mut out: Vec<PathBuf> = manifest
        .get("bin")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .map(|b| explicit_bin_path(member, b))
        .collect();
    let autobins = manifest
        .get("package")
        .and_then(|p| p.get("autobins"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(true);
    if autobins {
        out.extend(auto_bin_paths(member));
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn every_workspace_binary_runs_the_update_check() {
    let root = repo_root();
    let text = std::fs::read_to_string(root.join("Cargo.toml")).expect("root manifest");
    let ws: toml::Value = toml::from_str(&text).expect("root manifest parses");
    let members = ws["workspace"]["members"].as_array().expect("members");
    let (mut seen, mut missing) = (0usize, Vec::new());
    for m in members {
        let member = root.join(m.as_str().expect("member path"));
        for src in bin_sources(&member) {
            let rel = src
                .strip_prefix(&root)
                .expect("under root")
                .to_string_lossy()
                .replace('\\', "/");
            let rel = rel.trim_start_matches("./").to_string();
            seen += 1;
            if NOT_A_PRODUCT.iter().any(|(p, _)| *p == rel) {
                continue;
            }
            let body = std::fs::read_to_string(&src).unwrap_or_default();
            if !WIRED.iter().any(|w| body.contains(w)) {
                missing.push(rel);
            }
        }
    }
    // Vacuity: 29 binary targets on 2026-09-25. A walk that finds almost
    // none read the wrong tree, and must not pass as "all wired".
    assert!(
        seen >= 25,
        "found only {seen} binary targets — the walk is broken"
    );
    assert!(
        missing.is_empty(),
        "binaries without the update check (add `sovereign_update::hook!(\"<bin>\");` \
         as the first line of main, EPIC #4232): {missing:#?}"
    );
}
