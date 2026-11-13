//! Provable-contracts enforcement (L1 binding verification), and the plot verb's proven set.
//!
//! Validates that binding.yaml entries match source implementations.
//!
//! PLOT-GG-001 (#6991): `trueno_viz::plot_cli` draws only what a ggplot2 conformance row proves.
//! A row is the `ggplot2_row:` block of a `contracts/trueno-viz/ggplot2-*.yaml` contract; the
//! block names the geom, stat, position, scales, coord, facet, theme and aesthetics that the
//! row's R oracle was generated with. This script turns those blocks into
//! `src/plot_cli/proven_rows.rs`, which the verb `include!`s, so the verb's supported set is
//! derived from the contracts and never typed by hand.
//!
//! In the workspace the contracts are present: the script regenerates the table and fails the
//! build when the committed file differs (`APRENDER_VIZ_REGEN_PROVEN=1` rewrites it). A row
//! whose oracle directory has no `expected*.json`, or whose contract has no falsification
//! test, fails the build too: a binding without its evidence is not a proof. So does an
//! oracle that draws anything but one layer of the block's geom (`src/plot_cli/proven_gen.rs`).
//! In a packaged crate the contracts are absent and the committed file, checked when it was
//! published, is used as it is.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

include!("src/plot_cli/proven_gen.rs");

const GENERATED: &str = "src/plot_cli/proven_rows.rs";

fn main() {
    // AllImplemented policy: build.rs exists with contract enforcement keywords
    // This enables L1 binding verification via pmat comply CB-1208
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=contracts/");
    println!("cargo:rerun-if-changed={GENERATED}");
    println!("cargo:rerun-if-changed=src/plot_cli/proven_gen.rs");
    println!("cargo:rerun-if-env-changed=APRENDER_VIZ_REGEN_PROVEN");

    let root = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let workspace = root.join("../..");
    let dir = workspace.join("contracts/trueno-viz");
    if !dir.is_dir() {
        return; // packaged crate: the committed table is used as published
    }
    println!("cargo:rerun-if-changed={}", dir.display());
    let rows = read_rows(&dir, &workspace);
    let text = render(&rows);
    let path = root.join(GENERATED);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if committed == text {
        return;
    }
    if std::env::var_os("APRENDER_VIZ_REGEN_PROVEN").is_some() {
        std::fs::write(&path, text).unwrap_or_else(|e| fail(&format!("write {GENERATED}: {e}")));
        return;
    }
    fail(&format!(
        "{GENERATED} differs from the ggplot2_row blocks in contracts/trueno-viz; \
         regenerate it with APRENDER_VIZ_REGEN_PROVEN=1 cargo build -p aprender-viz and commit it"
    ));
}

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| fail(&format!("{key} is not set")))
}

fn fail(msg: &str) -> ! {
    panic!("aprender-viz build.rs: {msg}")
}

/// Every `ggplot2_row:` block, keyed by contract id, each checked for its evidence.
fn read_rows(dir: &Path, workspace: &Path) -> BTreeMap<String, Row> {
    let mut rows = BTreeMap::new();
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| fail(&format!("read {dir:?}: {e}")));
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let Some(id) = name.strip_prefix("ggplot2-").and_then(|n| n.strip_suffix(".yaml")) else {
            continue;
        };
        let id = format!("ggplot2-{id}");
        println!("cargo:rerun-if-changed={}", path.display());
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| fail(&format!("{name}: {e}")));
        let Some(row) = parse_row(name, &text).unwrap_or_else(|e| fail(&e)) else { continue };
        let oracles = expected_jsons(&workspace.join(&row.oracle));
        check_oracle(name, &row, &oracles).unwrap_or_else(|e| fail(&e));
        rows.insert(id, row);
    }
    rows
}

/// `(file name, text)` of each `expected*.json` in an oracle directory, by name.
fn expected_jsons(dir: &Path) -> Vec<(String, String)> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<(String, String)> = entries
        .flatten()
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .filter(|(n, _)| n.starts_with("expected") && n.ends_with(".json"))
        .map(|(n, p)| {
            println!("cargo:rerun-if-changed={}", p.display());
            let text = std::fs::read_to_string(&p).unwrap_or_else(|e| fail(&format!("{n}: {e}")));
            (n, text)
        })
        .collect();
    out.sort();
    out
}
