//! TRACE-001 TR-10: print `crux-trace-bind-receipt.json` for the in-tree contracts.
//!
//! ```bash
//! cargo run -p aprender-contracts --example crux_trace_bind -- contracts \
//!     > evidence/trace-001/crux-trace-bind-receipt.json
//! ```
//!
//! Exits 1 when G14 names any contract (active with no discharge record).

use std::path::PathBuf;
use std::process::ExitCode;

use provable_contracts::crux_trace_bind::build;

fn main() -> ExitCode {
    let dir = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("contracts"), PathBuf::from);
    let receipt = match build(&dir) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("crux_trace_bind: cannot list {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    match serde_json::to_string_pretty(&receipt) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("crux_trace_bind: serialize: {e}");
            return ExitCode::from(2);
        }
    }
    if receipt.g14_active_without_discharge.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
