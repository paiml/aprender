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

use serde_json::Value;

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
    let doc: Value = crate::yaml::from_str(&text)?;

    let rows = doc
        .get("categories")
        .and_then(Value::as_array)
        .ok_or("contract has no `categories:` sequence — is this a parity matrix?")?;

    let mut results: Vec<RowResult> = Vec::with_capacity(rows.len());
    for row in rows {
        results.push(check_row(row));
    }

    print_report(&results);

    let headline_errors = check_headline(&doc, rows);
    for e in &headline_errors {
        println!("  FAIL  [HEADLINE] {e}");
    }

    let failures = results
        .iter()
        .filter(|r| matches!(r.verdict, Verdict::Fail { .. }))
        .count();
    let total_failures = failures + headline_errors.len();
    if total_failures == 0 {
        Ok(())
    } else {
        Err(format!("{total_failures} parity check(s) failed").into())
    }
}

/// FALSIFY-CODE-PARITY-002: aggregate invariant — the `headline.counts`
/// block must match the actual status distribution across `categories[]`.
fn check_headline(doc: &Value, rows: &[Value]) -> Vec<String> {
    let mut errors = Vec::new();

    let (mut shipped, mut partial, mut missing) = (0u64, 0u64, 0u64);
    for row in rows {
        match row.get("status").and_then(Value::as_str) {
            Some("SHIPPED") => shipped += 1,
            Some("PARTIAL") => partial += 1,
            Some("NONE" | "MISSING") => missing += 1,
            _ => {}
        }
    }
    let actual_total = rows.len() as u64;

    let Some(headline) = doc.get("headline") else {
        return errors;
    };

    if let Some(declared) = headline.get("total_rows").and_then(Value::as_u64) {
        if declared != actual_total {
            errors.push(format!(
                "headline.total_rows {declared} ≠ actual {actual_total}"
            ));
        }
    }

    let Some(counts) = headline.get("counts") else {
        return errors;
    };
    for (name, actual) in &[
        ("shipped", shipped),
        ("partial", partial),
        ("missing", missing),
    ] {
        if let Some(declared) = counts.get(*name).and_then(Value::as_u64) {
            if declared != *actual {
                errors.push(format!(
                    "headline.counts.{name} {declared} ≠ actual {actual}"
                ));
            }
        }
    }

    errors
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
        .or_else(|| {
            row.get("expected_variant_count_min")
                .and_then(Value::as_u64)
        });
    let max = row
        .get("expected_max_hits")
        .and_then(Value::as_u64)
        .or_else(|| {
            row.get("expected_variant_count_max")
                .and_then(Value::as_u64)
        });

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

#[cfg(test)]
mod tests {
    use super::*;

    fn row(yaml: &str) -> Value {
        crate::yaml::from_str(yaml).expect("fixture row is valid YAML")
    }

    fn pass_hits(r: &RowResult) -> Option<u64> {
        match r.verdict {
            Verdict::Pass { hits } => Some(hits),
            _ => None,
        }
    }

    fn fail_reason(r: &RowResult) -> Option<&str> {
        match &r.verdict {
            Verdict::Fail { reason } => Some(reason),
            _ => None,
        }
    }

    fn skip_reason(r: &RowResult) -> Option<&str> {
        match &r.verdict {
            Verdict::Skipped { reason } => Some(reason),
            _ => None,
        }
    }

    #[test]
    fn check_row_without_command_is_skipped_with_defaults() {
        let r = check_row(&row("{}"));
        assert_eq!(r.id, "<unnamed>");
        assert_eq!(r.status, "UNKNOWN");
        assert_eq!(skip_reason(&r), Some("no cross_check_command"));
    }

    #[test]
    fn check_row_carries_id_and_status() {
        let r = check_row(&row(
            "{id: MCP, status: SHIPPED, cross_check_command: 'echo 3'}",
        ));
        assert_eq!(r.id, "MCP");
        assert_eq!(r.status, "SHIPPED");
        assert_eq!(pass_hits(&r), Some(3));
    }

    #[test]
    fn check_row_non_numeric_output_is_skipped() {
        let r = check_row(&row("{id: X, cross_check_command: 'echo abc'}"));
        assert_eq!(skip_reason(&r), Some("non-numeric output: \"abc\""));
    }

    #[test]
    fn check_row_below_min_fails() {
        let r = check_row(&row(
            "{id: X, cross_check_command: 'echo 2', expected_min_hits: 5}",
        ));
        assert_eq!(fail_reason(&r), Some("hits 2 < expected_min 5"));
    }

    #[test]
    fn check_row_above_max_fails() {
        let r = check_row(&row(
            "{id: X, cross_check_command: 'echo 9', expected_max_hits: 4}",
        ));
        assert_eq!(fail_reason(&r), Some("hits 9 > expected_max 4"));
    }

    #[test]
    fn check_row_bounds_are_inclusive() {
        let lo = check_row(&row(
            "{cross_check_command: 'echo 5', expected_min_hits: 5, expected_max_hits: 7}",
        ));
        let hi = check_row(&row(
            "{cross_check_command: 'echo 7', expected_min_hits: 5, expected_max_hits: 7}",
        ));
        assert_eq!(pass_hits(&lo), Some(5));
        assert_eq!(pass_hits(&hi), Some(7));
    }

    #[test]
    fn check_row_falls_back_to_variant_count_bounds() {
        let low = check_row(&row(
            "{cross_check_command: 'echo 1', expected_variant_count_min: 2}",
        ));
        let high = check_row(&row(
            "{cross_check_command: 'echo 8', expected_variant_count_max: 3}",
        ));
        assert_eq!(fail_reason(&low), Some("hits 1 < expected_min 2"));
        assert_eq!(fail_reason(&high), Some("hits 8 > expected_max 3"));
    }

    #[test]
    fn check_row_hits_bound_wins_over_variant_bound() {
        let r = check_row(&row(
            "{cross_check_command: 'echo 4', expected_min_hits: 4, expected_variant_count_min: 9}",
        ));
        assert_eq!(pass_hits(&r), Some(4));
    }

    #[test]
    fn check_row_unbounded_row_passes_any_count() {
        let r = check_row(&row("{cross_check_command: '  echo 0  '}"));
        assert_eq!(pass_hits(&r), Some(0));
    }
}
