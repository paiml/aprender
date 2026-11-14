//! Provable-contracts enforcement (L1 binding verification), and the plot verb's proven set.
//!
//! Validates that binding.yaml entries match source implementations.
//!
//! PLOT-GG-001 (#6991): `trueno_viz::plot_cli` draws only what a ggplot2 conformance row proves.
//! A row is the `ggplot2_row:` block of a `contracts/trueno-viz/ggplot2-*.yaml` contract. The
//! generator (`src/plot_cli/proven_gen.rs`, whose tests are the plot_cli unit tests) turns those
//! blocks into `src/plot_cli/proven_rows.rs`, which the verb `include!`s, and fails the build
//! when the committed file differs (`APRENDER_VIZ_REGEN_PROVEN=1` rewrites it), or when a row
//! has no falsification test or an oracle that does not back its block. In a packaged crate the
//! contracts are absent and the committed file, checked when it was published, is used as it is.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

include!("src/plot_cli/proven_gen.rs");

fn main() {
    // AllImplemented policy: build.rs exists with contract enforcement keywords
    // This enables L1 binding verification via pmat comply CB-1208
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=contracts/");
    println!("cargo:rerun-if-changed=src/plot_cli/proven_rows.rs");
    println!("cargo:rerun-if-changed=src/plot_cli/proven_gen.rs");
    println!("cargo:rerun-if-env-changed=APRENDER_VIZ_REGEN_PROVEN");
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let workspace = root.join("../..");
    let table = root.join("src/plot_cli/proven_rows.rs");
    let regen = std::env::var_os("APRENDER_VIZ_REGEN_PROVEN").is_some();
    if let Err(e) = sync(&workspace.join("contracts/trueno-viz"), &workspace, &table, regen) {
        panic!("aprender-viz build.rs: {e}");
    }
}
