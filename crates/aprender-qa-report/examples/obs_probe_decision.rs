//! OBS-12 (APR-OBS-001 §7): write the hourly-probe decision report.
//!
//! ```text
//! cargo run -p aprender-qa-report --example obs_probe_decision -- \
//!     --ledger perf.jsonl --red-reports red.jsonl [--lane lane.jsonl] [--md report.md]
//! ```
//!
//! Prints the `apr-obs-probe-decision-v1` JSON on stdout. Exit 0 = ready (a
//! recommendation was made), 10 = insufficient nights, 11 = refused, 2 = usage/IO.

use aprender_qa_report::probe_decision::{build_report, Status};
use std::process::ExitCode;

fn read(path: Option<&String>) -> Result<String, String> {
    path.map_or_else(
        || Ok(String::new()),
        |p| std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}")),
    )
}

fn run() -> Result<ExitCode, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let ledger = flag("--ledger").ok_or("--ledger is required")?;
    let red = flag("--red-reports").ok_or("--red-reports is required")?;
    let report = build_report(
        &read(Some(ledger))?,
        &read(Some(red))?,
        &read(flag("--lane"))?,
    );
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    println!("{json}");
    if let Some(md) = flag("--md") {
        std::fs::write(md, report.to_markdown()).map_err(|e| format!("{md}: {e}"))?;
    }
    Ok(match report.status {
        Status::Ready => ExitCode::SUCCESS,
        Status::InsufficientNights => ExitCode::from(10),
        Status::Refused => ExitCode::from(11),
    })
}

fn main() -> ExitCode {
    run().unwrap_or_else(|e| {
        eprintln!("obs_probe_decision: {e}");
        ExitCode::from(2)
    })
}
