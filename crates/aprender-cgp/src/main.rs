//! Standalone `aprender-cgp` binary.
//!
//! The entire command surface lives in `cgp::cli` so that a host binary (`apr`)
//! can dispatch into it; this target is only a shim.

use anyhow::Result;

/// Printed before anything runs: this binary is one of the duplicates `apr`
/// absorbed (#4060, EPIC #4057). It still works, through the same library code
/// `apr cgp` calls, so scripts that call it keep running until its consumers
/// have moved; then the bin target is removed.
const DEPRECATED: &str = "warning: `aprender-cgp` is deprecated and will be removed; run `apr cgp` instead (the same code path). See paiml/aprender#4057.";

fn main() -> Result<()> {
    eprintln!("{DEPRECATED}");
    cgp::cli::run()
}
