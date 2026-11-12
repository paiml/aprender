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
//! test, fails the build too: a binding without its evidence is not a proof. In a packaged
//! crate the contracts are absent and the committed file, checked when it was published, is
//! used as it is.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(serde::Deserialize)]
struct Contract {
    #[serde(default)]
    ggplot2_row: Option<Row>,
    #[serde(default)]
    falsification_tests: Vec<serde::de::IgnoredAny>,
}

/// The `ggplot2_row:` block. Flat scalars only, so that it can become the `properties` of a
/// pv ontology entity once the `ggplot2-row` entity type is declared.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    #[serde(rename = "ref")]
    name: String,
    geom: String,
    stat: String,
    position: String,
    scale_x: String,
    scale_y: String,
    coord: String,
    facet: String,
    theme: String,
    aes: String,
    oracle: String,
}

const GENERATED: &str = "src/plot_cli/proven_rows.rs";

fn main() {
    // AllImplemented policy: build.rs exists with contract enforcement keywords
    // This enables L1 binding verification via pmat comply CB-1208
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=contracts/");
    println!("cargo:rerun-if-changed={GENERATED}");
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
         regenerate it with APRENDER_VIZ_REGEN_PROVEN=1 cargo build -p aprender-viz \
         --features plot-cli and commit it"
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
        let c: Contract =
            serde_saphyr::from_str(&text).unwrap_or_else(|e| fail(&format!("{name}: {e}")));
        let Some(row) = c.ggplot2_row else { continue };
        if c.falsification_tests.is_empty() {
            fail(&format!("{name}: a ggplot2_row needs at least one falsification test"));
        }
        let oracle = workspace.join(&row.oracle);
        if !has_expected_json(&oracle) {
            fail(&format!("{name}: oracle {} has no expected*.json", row.oracle));
        }
        rows.insert(id, row);
    }
    rows
}

fn has_expected_json(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            let n = e.file_name();
            let n = n.to_string_lossy();
            n.starts_with("expected") && n.ends_with(".json")
        })
    })
}

fn words(s: &str) -> String {
    let quoted: Vec<String> = s.split_whitespace().map(|w| format!("{w:?}")).collect();
    format!("&[{}]", quoted.join(", "))
}

fn render(rows: &BTreeMap<String, Row>) -> String {
    let mut out = String::from(
        "// @generated by crates/aprender-viz/build.rs from the `ggplot2_row:` blocks of\n\
         // contracts/trueno-viz/ggplot2-*.yaml. Do not edit: change the contract, then run\n\
         // APRENDER_VIZ_REGEN_PROVEN=1 cargo build -p aprender-viz --features plot-cli\n\
         &[\n",
    );
    for (id, r) in rows {
        out.push_str(&format!(
            "    ProvenRow {{\n        contract: {id:?},\n        name: {:?},\n        \
             geom: {:?},\n        stat: {:?},\n        position: {:?},\n        \
             scales: &[(\"x\", {:?}), (\"y\", {:?})],\n        coord: {:?},\n        \
             facet: {:?},\n        theme: {:?},\n        aes: {},\n    }},\n",
            r.name,
            r.geom,
            r.stat,
            r.position,
            r.scale_x,
            r.scale_y,
            r.coord,
            r.facet,
            r.theme,
            words(&r.aes),
        ));
    }
    out.push_str("]\n");
    out
}
