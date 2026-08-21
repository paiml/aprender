//! `pv parity-ledger` — evaluate a competitive-parity ledger AT CHECK TIME.
//!
//! `pv validate` answers "is this file well-formed?", which is a question about
//! the file. This answers "what does this ledger claim TODAY?", which is a
//! question about the world, and the two diverge the moment a row goes stale.
//! That divergence is the whole point: a `BETTER` row whose `valid_until` has
//! passed is reported as `UNMEASURED` here while still validating perfectly
//! there. `contracts/beat-ollama-decode-throughput-speed-v1.yaml` shipped a
//! `baseline_sourced_date` of `2026-06-15` against a baseline re-derived on
//! 2026-07-31 and nothing noticed, because nothing ever asked a contract what
//! it claimed *today*.
//!
//! Output is anchored `__KEY__=value` lines so a shell ratchet can read it with
//! `grep -E '^__MEASURED__=[0-9]+$'` instead of hand-rolling a YAML parser
//! (`scripts/check_no_hand_rolled_parsers.sh` bans the construct, and rightly).

use std::path::Path;

use provable_contracts::error::Severity;
use provable_contracts::schema::parity::today_utc;
use provable_contracts::schema::{
    parse_contract, validate_contract, ContractKind, ParityLedger, Verdict,
};

/// The anchored `__KEY__=value` block the shell ratchet greps.
///
/// The `__ROW__=` / `__MEASURED_ROW__=` / `__DOWNGRADE__=` lines carry SETS, not
/// counts, and that distinction is the whole of the ratchet's first fix. A count
/// ratchet is payable in the wrong currency: delete the StandardScaler row and
/// add a cheaper one, and `__MEASURED__` is unchanged while the only recorded
/// 0.69x loss has left the tree. That is PMAT-733 with the arithmetic balanced.
/// Keyed by entry_point, losing a SPECIFIC row is detectable no matter what is
/// added beside it.
///
/// Entry points contain `=` (`apr run --gpu (concurrency=1 ...)`), so a consumer
/// must strip the `^__KEY__=` PREFIX rather than cut on `=`.
fn emit_machine_readable(ledger: &ParityLedger, today: &str, expired: usize, errors: usize) {
    println!("__TODAY__={today}");
    println!("__ROWS__={}", ledger.rows.len());
    println!("__MEASURED__={}", ledger.measured_count(today));
    println!("__NON_WINS__={}", ledger.non_win_count(today));
    println!("__EXPIRED__={expired}");
    println!("__ERRORS__={errors}");
    println!("__DOWNGRADES__={}", ledger.downgrades.len());
    for row in &ledger.rows {
        println!("__ROW__={}", row.entry_point.trim());
        if row.effective_verdict(today).is_measured() {
            println!("__MEASURED_ROW__={}", row.entry_point.trim());
        }
    }
    for d in &ledger.downgrades {
        println!("__DOWNGRADE__={}", d.entry_point.trim());
    }
}

/// Evaluate `path` as of `today` (defaults to the UTC date now).
///
/// # Errors
/// Returns `Err` when the file will not parse, is not a `competitive-parity`
/// contract, fails schema validation, or contains an EXPIRED row. An expired
/// row is an error rather than a warning on purpose: staleness blocks, the
/// verdict value does not.
pub fn run(path: &Path, today: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let today = match today {
        Some(t) => t.to_string(),
        None => today_utc().ok_or("system clock is before the Unix epoch")?,
    };
    if provable_contracts::schema::parity::parse_iso_date(&today).is_none() {
        return Err(format!("--today must be an ISO date (YYYY-MM-DD), got {today:?}").into());
    }

    let contract = parse_contract(path)?;
    if contract.kind() != ContractKind::CompetitiveParity {
        return Err(format!(
            "{} is kind `{}`, not `competitive-parity`",
            path.display(),
            contract.kind()
        )
        .into());
    }

    let violations = validate_contract(&contract);
    let errors: Vec<_> = violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();
    for v in &errors {
        println!("{v}");
    }

    let empty = ParityLedger::default();
    let ledger = contract.parity.as_ref().unwrap_or(&empty);
    let expired = ledger.expired_rows(&today);

    for row in &ledger.rows {
        let declared = row.verdict.map_or("(none)".to_string(), |v| v.to_string());
        let effective = row.effective_verdict(&today).to_string();
        let flag = if row.is_expired(&today) {
            "EXPIRED"
        } else {
            "fresh"
        };
        println!(
            "ROW  {declared:<15} -> {effective:<15} {flag:<8} valid_until={} {}",
            if row.valid_until.trim().is_empty() {
                "(none)"
            } else {
                row.valid_until.trim()
            },
            row.entry_point.trim(),
        );
    }

    for d in &ledger.downgrades {
        println!(
            "DOWN {:<15} -> {:<15} recheck_by={} {}",
            d.reason.map_or("(none)".to_string(), |r| r.to_string()),
            d.owner.trim(),
            d.recheck_by.trim(),
            d.entry_point.trim(),
        );
    }

    // The anchored block the shell ratchet greps. Keep these keys stable; see
    // `emit_machine_readable` for why the row lines are SETS and not counts.
    emit_machine_readable(ledger, &today, expired.len(), errors.len());

    if !errors.is_empty() {
        return Err(format!("ledger has {} validation error(s)", errors.len()).into());
    }
    if !expired.is_empty() {
        for row in &expired {
            eprintln!(
                "EXPIRED: {} (valid_until {:?}, owner {:?}) — re-measure it or record it as \
                 UNMEASURED with a new bound; it no longer counts toward __MEASURED__",
                row.entry_point.trim(),
                row.valid_until.trim(),
                row.owner.trim(),
            );
        }
        return Err(format!(
            "{} row(s) are past valid_until as of {today}. Staleness blocks; the verdict VALUE \
             does not.",
            expired.len()
        )
        .into());
    }

    // A ledger of nothing but wins is untested in the direction that matters --
    // and this repo has already deleted its only two losing rows once
    // (PMAT-733). Refuse before that becomes the shape again.
    if ledger.non_win_count(&today) == 0 && !ledger.rows.is_empty() {
        return Err(
            "every row is BETTER. A ledger whose rows are all wins is untested in the \
             direction that matters: record the losses instead of deleting them."
                .into(),
        );
    }

    let wins = ledger
        .rows
        .iter()
        .filter(|r| r.effective_verdict(&today) == Verdict::Better)
        .count();
    println!(
        "Ledger OK as of {today}: {} row(s), {} measured, {} non-win(s), {wins} win(s), 0 expired.",
        ledger.rows.len(),
        ledger.measured_count(&today),
        ledger.non_win_count(&today),
    );
    Ok(())
}
