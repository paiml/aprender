//! Standalone `aprender-cgp` binary.
//!
//! The entire command surface lives in `cgp::cli` so that a host binary (`apr`)
//! can dispatch into it; this target is only a shim.

use anyhow::Result;

fn main() -> Result<()> {
    cgp::cli::run()
}
