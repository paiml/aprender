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
    // The SCOPE floor that has come due, and how many steps declare one. The
    // scope file is a SIBLING of this contract, so the ratchet - not this
    // command - is the only thing that can compare the floor against the
    // actual scope. What is emitted here is the floor itself.
    let scope_floor = ledger
        .coverage
        .as_ref()
        .map_or(0, |c| c.scope_floor_as_of(today));
    println!("__SCOPE_FLOOR__={scope_floor}");
    let scope_steps = ledger.coverage.as_ref().map_or(0, |c| {
        c.steps.iter().filter(|s| s.scope_min.is_some()).count()
    });
    println!("__SCOPE_STEPS__={scope_steps}");
    println!("__REMOVALS__={}", ledger.removals.len());
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
            // The SCOPE schedule travels on its OWN channel rather than as a
            // third field of the coverage one. The ratchet compares each
            // schedule against protected `main` with the same shrink-never
            // rule, and a composite `by\tcovered\tscope` would make raising
            // ONE of the two floors read as deleting a step of the other.
            if let Some(scope_min) = s.scope_min {
                emit_key(
                    "__SCOPE_STEP__",
                    &format!("{}{FIELD_SEP}{scope_min}", s.by.trim()),
                );
            }
        }
    }
    // RECORDED REMOVALS. The ratchet spends one of these against a row or a
    // scope entry that has left, and only when the entry point is ALSO absent
    // from the live enumeration - the record makes the deletion owned, the
    // enumeration keeps it true.
    for r in &ledger.removals {
        emit_key("__REMOVAL__", r.entry_point.trim());
        if let Some(rep) = r.replacement.as_deref().map(str::trim) {
            if !rep.is_empty() {
                emit_key(
                    "__REMOVAL_REPLACEMENT__",
                    &format!("{}{FIELD_SEP}{rep}", r.entry_point.trim()),
                );
            }
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

/// DISCOVERY: which contracts under `root` are competitive-parity LEDGERS,
/// answered by PARSING them.
///
/// # The defect this replaces
///
/// The ratchet decides whether a prior ledger exists at the protected ref, and
/// "no prior ledger" is the BOOTSTRAP branch — the strongest possible pass,
/// because nothing is ratcheted against. That question was answered by a
/// `git grep` for the regular expression
///
/// ```text
/// ^[[:space:]]*kind:[[:space:]]*competitive-parity[[:space:]]*$
/// ```
///
/// over the protected tree. The INTENT was right, and was itself a fix: round
/// 4 keyed the window on a sibling file's PATH, so `git mv` manufactured a
/// fresh bootstrap silently, and keying on the KIND removes that. But a regex
/// over text is not the parsed kind, and every difference between the two is a
/// renewable bootstrap:
///
/// * `kind: "competitive-parity"` — quoted, identical meaning, regex misses;
/// * `kind: competitive-parity  # the ledger` — a trailing comment, misses;
/// * the mapping reflowed to flow style
///   (`metadata: {kind: competitive-parity, ...}`) — misses;
/// * `kind: competitive-parity` written inside a PROSE block scalar of some
///   unrelated contract — MATCHES, manufacturing an AMBIGUOUS COMPARAND out of
///   a sentence.
///
/// Every one of those is a semantically null edit that reopens the strongest
/// pass in the system. The fix is not a better regular expression; it is to
/// stop reading the text. The kind is `metadata.kind` after `serde` has read
/// the document, and that is what this reports.
///
/// # A file that will not parse is RED, never a bootstrap
///
/// The two failure directions are not symmetric. "Unreadable" collapsing to
/// "no prior ledger" hands the caller the bootstrap for a reason unrelated to
/// any ledger's contents — the same shape as the `fetch-depth: 1` collapse the
/// comparand-reachability check exists to refuse. So this exits non-zero
/// naming every offender, and the caller must treat that as fatal rather than
/// as an empty set.
///
/// # Errors
/// Returns `Err` when `root` cannot be read or any contract under it will not
/// parse.
pub fn discover(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !root.is_dir() {
        return Err(format!(
            "{} is not a directory. Discovery answers \"which contracts of this kind exist at \
             the protected ref?\", and an unreadable tree must not answer \"none\" - that is \
             the BOOTSTRAP branch, which is the strongest possible pass.",
            root.display()
        )
        .into());
    }

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    collect_yaml(root, &mut files)?;
    files.sort();

    let mut parity: Vec<String> = Vec::new();
    let mut unparseable: Vec<(String, String)> = Vec::new();
    for f in &files {
        let rel = f
            .strip_prefix(root)
            .unwrap_or(f)
            .to_string_lossy()
            .into_owned();
        match declared_kind(f) {
            Err(e) => unparseable.push((rel, e)),
            Ok(None) => {}
            Ok(Some(ContractKind::CompetitiveParity)) => {
                // A file that DECLARES this kind must also parse as a whole
                // contract. Discovery could stop at the kind, but then a
                // parity ledger with a broken `parity:` block would be
                // discovered and then fail one step later inside
                // `cp_prior_report`, which reports it as COMPARAND
                // UNEVALUABLE. Same verdict, worse diagnosis; catching it here
                // names the file at the ref that owns it.
                if let Err(e) = parse_contract(f) {
                    unparseable.push((rel, e.to_string()));
                } else {
                    parity.push(rel);
                }
            }
            Ok(Some(_)) => {}
        }
    }

    println!("__SCANNED__={}", files.len());
    println!("__UNPARSEABLE__={}", unparseable.len());
    for (rel, err) in &unparseable {
        // Newlines in a serde error would inject key lines, exactly as an
        // entry point containing one would; the length prefix is what the
        // consumer verifies, so the whole value travels or none of it does.
        emit_key("__UNPARSEABLE_FILE__", &format!("{rel}{FIELD_SEP}{err}"));
    }
    println!("__PARITY_CONTRACTS__={}", parity.len());
    for rel in &parity {
        emit_key("__PARITY_CONTRACT__", rel);
    }

    if !unparseable.is_empty() {
        for (rel, err) in &unparseable {
            eprintln!("UNPARSEABLE: {rel}: {err}");
        }
        return Err(format!(
            "{} contract(s) under {} will not parse. This is RED and never a bootstrap: the \
             caller is asking whether a prior competitive-parity ledger EXISTS at the protected \
             ref, and a file it cannot read is not evidence that one does not.",
            unparseable.len(),
            root.display()
        )
        .into());
    }
    Ok(())
}

/// The kind a document DECLARES at `metadata.kind`, read structurally.
///
/// Three outcomes, and the boundaries between them are the whole design:
///
/// * `Err(_)` — the file is not YAML at all, or `metadata.kind` holds a value
///   that is not a member of the closed [`ContractKind`] vocabulary. Both are
///   RED. "I cannot read this document" and "I cannot read this kind" must
///   never resolve to "it is not a parity ledger", because that answer opens
///   the bootstrap.
/// * `Ok(None)` — the document parses and demonstrably declares no kind:
///   either it has no `metadata` mapping, or that mapping has no `kind` key.
///   This is a PARSED conclusion about the document, not a guess from its
///   path, and it is the honest answer for the 47 files under `contracts/`
///   that are sidecars and ticket records rather than contracts. A file with
///   no `metadata:` cannot be declaring `metadata.kind: competitive-parity`.
/// * `Ok(Some(k))` — it declares `k`.
///
/// # Why a PROBE and not `parse_contract`
///
/// `Contract` requires `metadata.version` and `metadata.description`, so
/// parsing the whole type would make those 47 sidecars errors and the gate
/// would red on every run for a reason with nothing to do with parity. The
/// standing lesson here is that a gate which reds for a reason unrelated to
/// its property trains people to re-run it, and a red that gets re-run away is
/// how a REAL red gets re-run away too. The probe asks exactly the question
/// being asked and nothing else.
///
/// # Why this cannot be softened into "skip what does not parse"
///
/// That is what `contract_walk::collect_contracts` does, and it is the
/// difference between a discovered ledger and a bootstrap. Malformed YAML is
/// therefore an ERROR here even though the malformed file might have been a
/// sidecar: the caller is asking whether a parity ledger EXISTS, and an
/// unreadable file is not evidence that one does not.
fn declared_kind(path: &Path) -> Result<Option<ContractKind>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read: {e}"))?;
    let doc: serde_yaml::Value =
        serde_yaml::from_str(&text).map_err(|e| format!("not valid YAML: {e}"))?;
    let Some(kind) = doc.get("metadata").and_then(|m| m.get("kind")) else {
        return Ok(None);
    };
    // `serde_yaml::from_value` over the closed enum, so `competitive-parity`
    // is recognised however it was WRITTEN — quoted, followed by a comment, in
    // flow style — and an invented kind is an error rather than a miss. This is
    // the whole difference from the regex this replaces: the text is gone by
    // the time the comparison happens.
    serde_yaml::from_value::<ContractKind>(kind.clone())
        .map(Some)
        .map_err(|e| format!("metadata.kind {kind:?} is not a known contract kind: {e}"))
}

/// Every `.yaml` / `.yml` file under `dir`, recursively.
///
/// NOTHING IS SKIPPED BY NAME. `contract_walk::collect_contracts` drops
/// `binding.yaml`, drops any stem containing `playbook`, and silently drops
/// whatever fails to parse — all three are precisely the behaviours that must
/// not appear here. A path-keyed exclusion is a renewable bootstrap by
/// construction: name the ledger `parity-playbook-v1.yaml` and it stops being
/// discoverable, which is the `git mv` hole one rename further on.
///
/// A directory that cannot be read is an ERROR rather than an empty result,
/// for the same reason: silence here is the strongest pass in the system.
fn collect_yaml(
    dir: &Path,
    out: &mut Vec<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot read an entry of {}: {e}", dir.display()))?;
        let path = entry.path();
        // `symlink_metadata`, so a symlinked DIRECTORY is not descended into
        // and a cycle cannot hang the walk. A symlinked FILE still has its
        // extension read and is parsed like any other.
        let meta = std::fs::symlink_metadata(&path)
            .map_err(|e| format!("cannot stat {}: {e}", path.display()))?;
        if meta.is_dir() {
            collect_yaml(&path, out)?;
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("yaml" | "yml")
        ) {
            out.push(path);
        }
    }
    Ok(())
}
