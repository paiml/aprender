//! #5478 / FALSIFY-AUTH-001: no serve starter builds realizar's router outside
//! the Bearer gate.
//!
//! The GPU-batched, CUDA and APR/SafeTensors GPU starters need weights and a
//! device, so no unit test can send them a request. The behaviour of the gate
//! itself is proven over a model-free state in `server.rs`
//! (`gguf_router_auth_tests`, 401 without the key, 200 with it); this guard
//! proves every starter goes through that same gate, by reading the source of
//! the serve module: each `create_router_with_config(` code line must sit
//! inside an `auth::layer(` call (within the three lines before it).

#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

const NEEDLE: &str = "create_router_with_config(";
const GATE: &str = "auth::layer(";
const WINDOW: usize = 3;

/// 1-based line numbers of `create_router_with_config(` calls in `src` that
/// are not wrapped in `auth::layer(`. Comment lines and the `use` imports are
/// not calls.
fn ungated_sites(src: &str) -> Vec<usize> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let code = line.trim_start();
        if code.starts_with("//") || !code.contains(NEEDLE) {
            continue;
        }
        let start = i.saturating_sub(WINDOW);
        if !lines[start..i].iter().any(|l| l.contains(GATE)) {
            out.push(i + 1);
        }
    }
    out
}

/// The case table: the detector must flag a bare build and pass a gated one,
/// or its green over the real tree means nothing.
#[test]
fn ungated_sites_case_table() {
    let cases: &[(&str, &str, usize)] = &[
        ("bare let", "let app = create_router_with_config(state, cfg);", 1),
        (
            "bare full path",
            "let r = realizar::api::create_router_with_config(s, c);",
            1,
        ),
        (
            "gated helper",
            "super::auth::layer(\n    gate,\n    realizar::api::create_router_with_config(s, c),\n)",
            0,
        ),
        ("comment", "// built straight from `create_router_with_config(`", 0),
        (
            "use import",
            "use realizar::api::{create_router_with_config, AppState};",
            0,
        ),
        (
            "gate too far away",
            "auth::layer(\n1\n2\n3\nlet a = create_router_with_config(s, c);",
            1,
        ),
    ];
    for (name, src, want) in cases {
        assert_eq!(ungated_sites(src).len(), *want, "case `{name}`: {src}");
    }
}

#[test]
fn every_serve_router_is_built_behind_the_auth_gate() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/commands/serve");
    let mut scanned = 0usize;
    let mut found_gated_call = false;
    let mut bad = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read serve dir") {
        let path = entry.expect("dir entry").path();
        // This file names the needle in its own case table.
        if path.extension().and_then(|e| e.to_str()) != Some("rs")
            || path.file_name().and_then(|n| n.to_str()) == Some("tests_router_auth_5478.rs")
        {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read source");
        scanned += 1;
        found_gated_call |= src.contains(NEEDLE) && src.contains(GATE);
        for line in ungated_sites(&src) {
            bad.push(format!("{}:{line}", path.display()));
        }
    }
    // Refuse to pass vacuously: the scan must have read the module and seen
    // at least the gated helper.
    assert!(scanned > 10, "scanned only {scanned} files in {dir}");
    assert!(
        found_gated_call,
        "no gated create_router_with_config call found"
    );
    assert!(
        bad.is_empty(),
        "realizar router built without auth::layer (FALSIFY-AUTH-001, #5478):\n{}",
        bad.join("\n")
    );
}
