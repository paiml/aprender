//! Standalone `aprender-cgp` binary.
//!
//! The entire command surface lives in `cgp::cli` so that a host binary (`apr`)
//! can dispatch into it; this target is only a shim.

use anyhow::Result;

fn main() -> Result<()> {
    sovereign_update::hook!("aprender-cgp"); // EPIC #4232: `aprender-cgp update`, and the startup notice
    cgp::cli::run()
}
