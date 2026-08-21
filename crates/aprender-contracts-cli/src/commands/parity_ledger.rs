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
///
/// # Keys are LENGTH-PREFIXED: `__ROW__=<bytes>:<key>`
///
/// The previous emitter printed `__ROW__={}` of `entry_point.trim()`, and both
/// halves of that were wrong.
///
/// * `.trim()` normalised on the way OUT, so the authored bytes and the
///   compared bytes could differ: a key could be perturbed in the file and
///   still match the baseline. Canonicality is now required of the AUTHOR
///   (PARITY-002), so the emitted key is the authored key, byte for byte.
/// * Far worse, nothing stopped a key from containing a NEWLINE. A block
///   scalar reading `apr qa\n__ROW__=<the deleted row>` printed several
///   well-formed key lines from ONE fabricated row, satisfying a deleted row's
///   baseline keys at constant totals — the set ratchet defeated by exactly
///   the move it exists to block.
///
/// The length prefix is the second, independent control: a consumer that
/// checks `bytes` against the length of what it actually read cannot be fooled
/// by a key that is not the whole line, whatever the character rules later
/// become. Two controls because either one alone is a single edit from
/// useless.
fn emit_key(tag: &str, key: &str) {
    println!("{tag}={}:{key}", key.len());
}

/// The separator inside a COMPOSITE key value (`VERDICT\tentry_point`).
///
/// TAB, and specifically not a space or a colon, because `PARITY-002` admits
/// only printable ASCII (`0x20..=0x7E`) in a ratchet key — so a TAB cannot
/// occur inside an `entry_point`, and a composite value splits unambiguously
/// no matter what the author writes. Entry points genuinely contain both
/// spaces and `=` (`apr run --gpu (concurrency=1 single-request decode)`), so
/// either of those would have been a parser that works until it does not.
const FIELD_SEP: char = '\t';

fn emit_machine_readable(ledger: &ParityLedger, today: &str, expired: usize, errors: usize) {
    println!("__TODAY__={today}");
    println!("__ROWS__={}", ledger.rows.len());
    println!("__MEASURED__={}", ledger.measured_count(today));
    println!("__NON_WINS__={}", ledger.non_win_count(today));
    println!("__EXPIRED__={expired}");
    println!("__ERRORS__={errors}");
    println!("__DOWNGRADES__={}", ledger.downgrades.len());
    println!(
        "__OVERDUE_DOWNGRADES__={}",
        ledger.overdue_downgrades(today).len()
    );
    println!(
        "__DECLARED_MEASURED__={}",
        ledger.declared_measured_rows().len()
    );
    println!("__COVERED__={}", ledger.covered_count());
    let (_, cov_floor) = ledger.coverage_status(today);
    println!("__COVERAGE_FLOOR__={cov_floor}");
    println!("__UPGRADES__={}", ledger.recorded_upgrades(today));
    let steps = ledger.coverage.as_ref().map_or(0, |c| c.steps.len());
    println!("__COVERAGE_STEPS__={steps}");
    for row in &ledger.rows {
        emit_key("__ROW__", row.entry_point.trim());
        if row.effective_verdict(today).is_measured() {
            emit_key("__MEASURED_ROW__", row.entry_point.trim());
        }
        // The DECLARED-measured set, which is what a LATER tree is ratcheted
        // against. Read `declared_measured_rows` for why the prior side must be
        // clock-independent: computed from effective verdicts, the bar the
        // current tree has to clear would fall on its own as `main`'s rows aged,
        // on a day nobody touched either file.
        if row.verdict.is_some_and(Verdict::is_measured) {
            emit_key("__DECLARED_MEASURED_ROW__", row.entry_point.trim());
        }
        // The DECLARED verdict, never the effective one.
        //
        // The ratchet compares this against the verdict in the committed
        // baseline and demands a record for any DIFFERENCE. Emitting the
        // EFFECTIVE verdict would make the mere passage of time look like an
        // author's relabelling — every row degrades to UNMEASURED on its expiry
        // date — so the gate would start demanding paperwork for something no
        // author did, on a day nobody touched the file. Expiry is already
        // handled, loudly, by `block_on_staleness`.
        emit_key(
            "__VERDICT_ROW__",
            &format!(
                "{}{FIELD_SEP}{}",
                row.verdict.unwrap_or(Verdict::Unmeasured),
                row.entry_point.trim()
            ),
        );
    }
    // Only downgrades that are still IN DATE are emitted. An overdue one stops
    // paying for the MEASURED_ROW drop it was justifying, so the ratchet names
    // that row again — which is the whole mechanism that makes the escape
    // valve COME DUE. `recheck_by` used to be checked against `recorded_on`
    // and never against today, so a downgrade was permanent the moment it was
    // written, and with the old MEASURED floor deleted, a series of them could
    // drain the ledger to zero while every gate stayed green.
    for d in ledger.in_date_records(today) {
        // A record excuses a MEASURED_ROW drop only when it is a drop TO
        // UNMEASURED. A `WORSE -> NOT_COMPARABLE` record describes a row that
        // is still measured; letting it also pay for a later, unrelated exit
        // from the measured set would make that second move free.
        if d.excuses_unmeasured() {
            emit_key("__DOWNGRADE__", d.entry_point.trim());
        }
        // The TRANSITION channel: `FROM<TAB>TO<TAB>entry_point`. The ratchet
        // will only spend one of these against a transition it exactly
        // describes, so a record cannot excuse a move it does not name.
        if let (Some(from), Some(to)) = (d.from_verdict, d.to_verdict) {
            emit_key(
                "__TRANSITION__",
                &format!("{from}{FIELD_SEP}{to}{FIELD_SEP}{}", d.entry_point.trim()),
            );
        }
    }
    // The coverage SCHEDULE, as a set. The ratchet reads it from the ledger on
    // protected `main` and refuses a step that has been deleted, lowered, or
    // dated further out — the schedule is the floor, so it needs the same
    // shrink-never treatment the row set gets.
    if let Some(cov) = ledger.coverage.as_ref() {
        for s in &cov.steps {
            emit_key(
                "__COVERAGE_STEP__",
                &format!("{}{FIELD_SEP}{}", s.by.trim(), s.covered_min),
            );
        }
    }
    let (spent, allowed) = ledger.excuse_budget(today);
    println!("__EXCUSES__={spent}");
    println!("__EXCUSE_BUDGET__={allowed}");
}

/// Evaluate `path` as of `today` (defaults to the UTC date now).
///
/// # Errors
/// Returns `Err` when the file will not parse, is not a `competitive-parity`
/// contract, fails schema validation, or contains an EXPIRED row. An expired
/// row is an error rather than a warning on purpose: staleness blocks, the
/// verdict value does not.
/// Resolve and BOUND the as-of date.
///
/// `--today` is the same defect one surface up: an author-supplied date that
/// nothing bounded against the real clock. Every date rule in the ledger is
/// evaluated against this value, so `--today 2020-01-01` un-expires the entire
/// ledger in one flag. Nothing in-tree passes it, which is exactly how such a
/// hole survives review.
///
/// Bounded MONOTONELY: it may move FORWARD (which only expires more rows, so it
/// is strictly stricter, and is what replay and "will this still be green next
/// month" want) and never BACKWARD. No invocation needs to ask what the ledger
/// claimed in the past; an obvious one would like to.
fn resolve_today(today: Option<&str>) -> Result<String, Box<dyn std::error::Error>> {
    use provable_contracts::schema::parity::{days_between, parse_iso_date};
    let today = match today {
        Some(t) => t.to_string(),
        None => today_utc().ok_or("system clock is before the Unix epoch")?,
    };
    if parse_iso_date(&today).is_none() {
        return Err(format!("--today must be an ISO date (YYYY-MM-DD), got {today:?}").into());
    }
    if let Some(real) = today_utc() {
        if days_between(&real, &today).is_some_and(|d| d < 0) {
            return Err(format!(
                "--today {today} is BEFORE the real UTC date ({real}). Freshness is evaluated \
                 against this value, so a past --today un-expires the whole ledger with one \
                 flag - the same unbounded-date defect the ledger's own fields had, one surface \
                 up. --today may only move FORWARD (which is strictly stricter)."
            )
            .into());
        }
    }
    Ok(today)
}

/// The human-readable ROW / DOWN block.
fn report(ledger: &ParityLedger, today: &str) {
    for row in &ledger.rows {
        println!(
            "ROW  {:<15} -> {:<15} {:<8} valid_until={} {}",
            row.verdict.map_or("(none)".to_string(), |v| v.to_string()),
            row.effective_verdict(today).to_string(),
            if row.is_expired(today) {
                "EXPIRED"
            } else {
                "fresh"
            },
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
            "DOWN {:<22} {:<15} -> {:<15} owner={} recheck_by={} {:<8} {}",
            d.reason.map_or("(none)".to_string(), |r| r.to_string()),
            d.from_verdict
                .map_or("(none)".to_string(), |v| v.to_string()),
            d.destination().to_string(),
            d.owner.trim(),
            d.recheck_by.trim(),
            if d.is_overdue(today) {
                "OVERDUE"
            } else {
                "in-date"
            },
            d.entry_point.trim(),
        );
    }
}

/// Staleness blocks. An OVERDUE downgrade blocks for the same reason an expired
/// row does: the record is a DEBT with a due date, and `recheck_by` was
/// validated only against `recorded_on` and never against today, so the debt
/// never came due. With the old MEASURED floor deleted, that combination could
/// drain the ledger to zero measured rows with every gate green.
fn block_on_staleness(
    ledger: &ParityLedger,
    today: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let expired = ledger.expired_rows(today);
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
    let overdue = ledger.overdue_downgrades(today);
    if !overdue.is_empty() {
        for d in &overdue {
            eprintln!(
                "OVERDUE DOWNGRADE: {} (recheck_by {:?}, owner {:?}, reason {}) — the debt is \
                 due. Re-measure the row and file it MEASURED again, or re-argue the downgrade \
                 with a fresh recheck_by. Until then it no longer pays for the row's absence \
                 from __MEASURED_ROW__, and the ratchet names that row.",
                d.entry_point.trim(),
                d.recheck_by.trim(),
                d.owner.trim(),
                d.reason.map_or("(none)".to_string(), |r| r.to_string()),
            );
        }
        return Err(format!(
            "{} downgrade(s) are past recheck_by as of {today}. A downgrade is a debt with a due \
             date, not a retirement.",
            overdue.len()
        )
        .into());
    }
    Ok(())
}

/// The EXCUSE BUDGET: a ledger may not owe more re-measurements than it holds
/// measurements.
///
/// `is_overdue` bounds how LONG one excuse lasts. Nothing bounded how MANY may
/// be outstanding at once, and the two are independent — a fresh, in-date,
/// correctly-owned, closed-vocabulary record for every row drives `__MEASURED__`
/// to zero with every individual rule satisfied and every gate green. Each
/// record impeccable, the aggregate a ledger that has stopped making claims.
///
/// A constant floor is the wrong shape here: `MEASURED_MIN=4` is exactly what
/// made the honest `apr code` correction mechanically forbidden, and a ratchet
/// that punishes increasing honesty produces dishonest ledgers. This bound
/// scales with the ledger and has give in it — paying one debt buys the
/// capacity to take on another — while making it impossible for more than half
/// the rows to be excused at once.
fn block_on_excuse_budget(
    ledger: &ParityLedger,
    today: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (spent, allowed) = ledger.excuse_budget(today);
    if spent > allowed {
        for d in ledger.in_date_records(today) {
            eprintln!(
                "EXCUSED: {} (reason {}, recheck_by {:?}, owner {:?})",
                d.entry_point.trim(),
                d.reason.map_or("(none)".to_string(), |r| r.to_string()),
                d.recheck_by.trim(),
                d.owner.trim(),
            );
        }
        return Err(format!(
            "{spent} row(s) are excused by an in-date record against {allowed} measured row(s). \
             A ledger may not owe more re-measurements than it holds measurements: every \
             individual record here can be impeccable - fresh, owned, closed-vocabulary reason - \
             while the ledger as a whole has quietly stopped measuring anything. Pay a debt \
             (re-measure a row) before taking on another."
        )
        .into());
    }
    Ok(())
}

pub fn run(path: &Path, today: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let today = resolve_today(today)?;

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

    report(ledger, &today);

    // The anchored block the shell ratchet greps. Keep these keys stable; see
    // `emit_machine_readable` for why the row lines are SETS and not counts.
    emit_machine_readable(ledger, &today, expired.len(), errors.len());

    if !errors.is_empty() {
        return Err(format!("ledger has {} validation error(s)", errors.len()).into());
    }
    block_on_staleness(ledger, &today)?;
    block_on_excuse_budget(ledger, &today)?;
    // The coverage ratchet is evaluated HERE and not in `validate_contract`,
    // for the same reason expiry is: `pv validate` answers a question about the
    // file, and "has this step come due?" is a question about the world.
    let debt = provable_contracts::schema::parity_coverage_debt(ledger, &today);
    if !debt.is_empty() {
        for d in &debt {
            eprintln!("{d}");
        }
        return Err(format!("{} coverage-ratchet violation(s) as of {today}", debt.len()).into());
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
    let (spent, allowed) = ledger.excuse_budget(&today);
    println!(
        "Ledger OK as of {today}: {} row(s), {} measured, {} non-win(s), {wins} win(s), 0 \
         expired, {spent}/{allowed} excuse budget spent.",
        ledger.rows.len(),
        ledger.measured_count(&today),
        ledger.non_win_count(&today),
    );
    Ok(())
}
