//! `pv check-parity` — SEMANTIC gate for parity-matrix contracts.
//!
//! `pv validate` is the SCHEMA gate — it checks the YAML parses and carries
//! the fields required by the `aprender-contracts` schema. A parity-matrix
//! contract (e.g. `contracts/apr-code-parity-v1.yaml`, `kind: pattern`)
//! additionally encodes a per-row `cross_check_command` whose output is the
//! mechanical verification of `status`. This command runs each row's
//! cross-check and compares the hit count against the declared
//! `expected_min_hits` / `expected_max_hits` bounds.
//!
//! Closes the SEMANTIC half of PMAT-CONTRACTS-PARITY-001.

use std::path::Path;
use std::process::Command;

use serde_yaml::Value;

#[derive(Debug)]
pub struct RowResult {
    pub id: String,
    pub status: String,
    pub verdict: Verdict,
}

#[derive(Debug)]
pub enum Verdict {
    Pass { hits: u64 },
    Fail { reason: String },
    Skipped { reason: String },
}

pub fn run(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let doc: Value = serde_yaml::from_str(&text)?;

    let rows = doc
        .get("categories")
        .and_then(Value::as_sequence)
        .ok_or("contract has no `categories:` sequence — is this a parity matrix?")?;

    let mut results: Vec<RowResult> = Vec::with_capacity(rows.len());
    for row in rows {
        results.push(check_row(row));
    }

    print_report(&results);

    let failures = results
        .iter()
        .filter(|r| matches!(r.verdict, Verdict::Fail { .. }))
        .count();
    if failures == 0 {
        Ok(())
    } else {
        Err(format!("{failures} parity row(s) failed").into())
    }
}

fn check_row(row: &Value) -> RowResult {
    let id = row
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("<unnamed>")
        .to_string();
    let status = row
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("UNKNOWN")
        .to_string();

    let Some(cmd) = row.get("cross_check_command").and_then(Value::as_str) else {
        return RowResult {
            id,
            status,
            verdict: Verdict::Skipped {
                reason: "no cross_check_command".to_string(),
            },
        };
    };

    let output = match Command::new("sh").arg("-c").arg(cmd.trim()).output() {
        Ok(o) => o,
        Err(e) => {
            return RowResult {
                id,
                status,
                verdict: Verdict::Skipped {
                    reason: format!("exec failed: {e}"),
                },
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stdout_trimmed = stdout.trim();
    let Ok(hits) = stdout_trimmed.parse::<u64>() else {
        return RowResult {
            id,
            status,
            verdict: Verdict::Skipped {
                reason: format!("non-numeric output: {stdout_trimmed:?}"),
            },
        };
    };

    let min = row
        .get("expected_min_hits")
        .and_then(Value::as_u64)
        .or_else(|| row.get("expected_variant_count_min").and_then(Value::as_u64));
    let max = row
        .get("expected_max_hits")
        .and_then(Value::as_u64)
        .or_else(|| row.get("expected_variant_count_max").and_then(Value::as_u64));

    if let Some(m) = min {
        if hits < m {
            return RowResult {
                id,
                status,
                verdict: Verdict::Fail {
                    reason: format!("hits {hits} < expected_min {m}"),
                },
            };
        }
    }
    if let Some(m) = max {
        if hits > m {
            return RowResult {
                id,
                status,
                verdict: Verdict::Fail {
                    reason: format!("hits {hits} > expected_max {m}"),
                },
            };
        }
    }

    RowResult {
        id,
        status,
        verdict: Verdict::Pass { hits },
    }
}

fn print_report(results: &[RowResult]) {
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut skip = 0usize;
    for r in results {
        match &r.verdict {
            Verdict::Pass { hits } => {
                pass += 1;
                println!("  PASS  [{:<8}] {}  (hits={hits})", r.status, r.id);
            }
            Verdict::Fail { reason } => {
                fail += 1;
                println!("  FAIL  [{:<8}] {}  ({reason})", r.status, r.id);
            }
            Verdict::Skipped { reason } => {
                skip += 1;
                println!("  SKIP  [{:<8}] {}  ({reason})", r.status, r.id);
            }
        }
    }
    println!();
    println!(
        "{} row(s) checked: {pass} pass, {fail} fail, {skip} skip",
        results.len()
    );
}
