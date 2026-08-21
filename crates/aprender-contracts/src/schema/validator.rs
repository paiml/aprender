use std::collections::HashSet;

use crate::error::{Severity, Violation};
use crate::schema::parity::{Verdict, MAX_VALIDITY_DAYS};
use crate::schema::types::{Contract, ContractKind};

/// Validate a parsed contract for completeness and consistency.
///
/// Returns a list of violations. If any violation has
/// [`Severity::Error`], the contract is considered invalid.
///
/// Validation is kind-aware: non-kernel contracts (registries, model-family
/// schemas, reference documents) are validated only for metadata consistency;
/// the provability invariant, equations, and proof/kani/falsification checks
/// only apply to `ContractKind::Kernel`.
pub fn validate_contract(contract: &Contract) -> Vec<Violation> {
    let mut violations = Vec::new();

    validate_metadata(contract, &mut violations);

    // Kernel-only checks: these enforce the provability invariant and
    // require equations + proof obligations + tests + Kani harnesses.
    if contract.kind() == ContractKind::Kernel && !contract.is_registry() {
        validate_equations(contract, &mut violations);
        validate_provability_invariant(contract, &mut violations);
        validate_proof_obligations(contract, &mut violations);
        validate_falsification_tests(contract, &mut violations);
        validate_kani_harnesses(contract, &mut violations);
        validate_qa_gate(contract, &mut violations);
    } else {
        // Non-kernel kinds (registry, model-family, schema): still validate
        // any proof obligations/falsification/kani data that IS present, so
        // mistakes are caught even on exempt contracts.
        validate_proof_obligations(contract, &mut violations);
        validate_falsification_tests(contract, &mut violations);
        validate_kani_harnesses(contract, &mut violations);
    }

    // BeatBenchmark-only checks (PMAT-741): the `beat:` block must pin a
    // falsifiable, four-pillar incumbent baseline. Independent of the
    // kernel/non-kernel split above.
    if contract.kind() == ContractKind::BeatBenchmark {
        validate_beat_benchmark(contract, &mut violations);
    }

    // CompetitiveParity-only checks. Deliberately OUTSIDE the kernel/non-kernel
    // split, and deliberately NOT guarded by `!contract.is_registry()`: the
    // whole defect this kind closes is that `registry: true` bought an
    // exemption from provability. Here it buys nothing -- PARITY-000 makes it a
    // hard error, and the provability invariant runs either way.
    if contract.kind() == ContractKind::CompetitiveParity {
        validate_provability_invariant(contract, &mut violations);
        validate_competitive_parity(contract, &mut violations);
    }

    violations
}

/// Tokens that are NOT a version pin, however confidently they are written.
/// Matched against `competitor_version` split on non-alphanumerics, so `main`
/// rejects `llama.cpp@main` without rejecting a version containing "domain".
const UNPINNED_VERSION_TOKENS: [&str; 12] = [
    "latest", "unpinned", "unknown", "tbd", "todo", "na", "any", "current", "head", "main",
    "master", "whatever",
];

/// Enforce the competitive-parity LEDGER shape (PARITY-000..010).
///
/// The rules exist to make a row falsifiable by someone who was not there: a
/// named competitor at a PINNED version, both invocations written out, a
/// verdict from the closed vocabulary, the date it was taken, the date it goes
/// stale, an owner, and a pointer to the receipt. Note what is NOT a rule:
/// nothing here requires the verdict to be `BETTER`. A rule that admitted only
/// wins would make deleting a losing row the cheapest way to comply, which is
/// how the StandardScaler 0.69x measurement left the tree (PMAT-733).
fn validate_competitive_parity(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut push = |rule: &str, message: String, location: String| {
        violations.push(Violation {
            severity: Severity::Error,
            rule: rule.to_string(),
            message,
            location: Some(location),
        });
    };

    // PARITY-000: the registry escape hatch is CLOSED on this kind. `kind()`
    // already refuses to rewrite anything but `Kernel`, so this cannot be
    // reached by accident -- it exists so that WRITING the intent fails loudly
    // rather than being silently ignored by a future reader who assumes, from
    // 481 other contracts, that `registry: true` means "exempt".
    if contract.metadata.registry {
        push(
            "PARITY-000",
            "metadata.registry: true is not permitted on a competitive-parity contract - \
             the registry flag exempts a contract from the provability invariant, and this \
             kind exists precisely because that exemption made 481 contracts prove nothing"
                .to_string(),
            "metadata.registry".to_string(),
        );
    }

    // PARITY-001: there must be a ledger, with rows.
    let Some(ledger) = contract.parity.as_ref() else {
        push(
            "PARITY-001",
            "competitive-parity contract must define a `parity:` block with `rows:`".to_string(),
            "parity".to_string(),
        );
        return;
    };
    if ledger.rows.is_empty() {
        push(
            "PARITY-001",
            "parity.rows must not be empty - an empty ledger is a 100% ratio over zero \
             entry points"
                .to_string(),
            "parity.rows".to_string(),
        );
        return;
    }

    let mut seen_entry_points: HashSet<&str> = HashSet::new();
    for (i, row) in ledger.rows.iter().enumerate() {
        validate_parity_row(row, i, &mut seen_entry_points, &mut push);
    }

    validate_parity_downgrades(ledger, &mut push);
    validate_parity_falsification_bindings(contract, &mut push);
    validate_parity_date_horizon(contract, &mut push);
}

/// One row: PARITY-002 (a named, unique, CANONICAL key), -004, -005, -006,
/// -007, -011 and -017.
fn validate_parity_row<'a>(
    row: &'a crate::schema::parity::ParityRow,
    i: usize,
    seen_entry_points: &mut HashSet<&'a str>,
    push: &mut impl FnMut(&str, String, String),
) {
    let at = |f: &str| format!("parity.rows[{i}].{f}");

    if row.entry_point.trim().is_empty() {
        push(
            "PARITY-002",
            format!("parity.rows[{i}].entry_point must name the apr entry point"),
            at("entry_point"),
        );
    } else if !seen_entry_points.insert(row.entry_point.trim()) {
        push(
            "PARITY-002",
            format!(
                "duplicate entry_point {:?} - two rows for one entry point let a \
                 favourable one mask an unfavourable one",
                row.entry_point.trim()
            ),
            at("entry_point"),
        );
    }
    validate_key_is_canonical(&row.entry_point, &at("entry_point"), "PARITY-002", push);

    validate_parity_version_pin(row, i, push);
    validate_parity_required_fields(row, i, push);

    // PARITY-006: a verdict from the closed vocabulary. An UNKNOWN string never
    // reaches here -- serde rejects the document -- so this fires only on an
    // absent verdict.
    if row.verdict.is_none() {
        push(
            "PARITY-006",
            format!(
                "parity.rows[{i}].verdict is required - one of \
                 BETTER / PARITY / WORSE / NOT_COMPARABLE / UNMEASURED"
            ),
            at("verdict"),
        );
    }

    validate_parity_row_dates(row, i, push);
}

/// PARITY-002 / PARITY-012: a ratchet KEY must be canonical.
///
/// See [`crate::schema::parity::is_canonical_key`] for the whole argument. The
/// short version: the ratchet's set membership travels over a text channel as
/// `__ROW__=<key>` lines, so an `entry_point` containing a NEWLINE prints
/// several well-formed key lines from one row and satisfies a DELETED row's
/// baseline keys — the set ratchet defeated at constant totals, which is what
/// it was built to prevent. Refusing the character at the source is one of the
/// two independent controls; the emitter's length prefix is the other.
fn validate_key_is_canonical(
    key: &str,
    location: &str,
    rule: &str,
    push: &mut impl FnMut(&str, String, String),
) {
    if key.is_empty() {
        return; // the "must be named" arm has already fired
    }
    if let Some((idx, ch)) = crate::schema::parity::bad_key_byte(key) {
        push(
            rule,
            format!(
                "{location} contains {ch:?} (U+{:04X}) at character {idx} - a ratchet key must \
                 be printable ASCII. The set ratchet carries membership over a TEXT channel \
                 (`__ROW__=<key>` lines that a shell reads back), so a key containing a newline \
                 prints SEVERAL well-formed key lines from ONE row and can satisfy a deleted \
                 row's baseline keys at constant totals - the set ratchet defeated by the same \
                 move it was built to block",
                u32::from(ch),
            ),
            location.to_string(),
        );
    } else if key.trim() != key {
        push(
            rule,
            format!(
                "{location} {key:?} has leading or trailing whitespace - write the key in \
                 canonical form. The emitted key and the baseline key must be byte-identical by \
                 CONSTRUCTION; normalising on the way out (the previous emitter called `.trim()`) \
                 means the authored bytes and the compared bytes can differ, so a key can be \
                 perturbed in the file and still match"
            ),
            location.to_string(),
        );
    }
}

/// PARITY-016: NO date anywhere in this document may be beyond the horizon.
///
/// # This rule, not the four field rules, is the class control
///
/// Every previous fix in this file bounded one author-supplied date against
/// ANOTHER author-supplied date, and each fix added a new unbounded field:
/// `valid_until` was capped from `measured_on` (unbounded), `recheck_by` from
/// `recorded_on` (unbounded). A difference bounds nothing when the author
/// writes both ends. [`crate::schema::parity::LedgerDate`] fixes that for the
/// four fields that exist TODAY — but a newtype only helps a field that USES
/// it, and the next author adding `certified_on: String` is the next
/// exemption.
///
/// So this sweep does not look at fields at all. It SERIALIZES the whole
/// contract and walks every scalar in it, at any depth, under any key. Any
/// string that is exactly a strict ISO date and lands past the horizon is an
/// error, whether it is a typed field, a field added next year, or a key serde
/// does not know (`parity.rows[].extra` captures those instead of dropping
/// them, precisely so this walk can see them). A field that does not exist yet
/// is bounded before it is written, which is the only version of this rule
/// that survives the next commit.
///
/// Prose is exempt because the whole scalar must BE the date: a `note:`
/// reading "measured 2099-01-01" is 25 characters, not 10, and never matches.
fn validate_parity_date_horizon(contract: &Contract, push: &mut impl FnMut(&str, String, String)) {
    let Some(today) = crate::schema::parity::today_utc() else {
        push(
            "PARITY-016",
            "the system clock is before the Unix epoch, so no date in this ledger can be \
             bounded against today. Refusing rather than passing: a bound that cannot be \
             EVALUATED must be red, not absent"
                .to_string(),
            "parity".to_string(),
        );
        return;
    };
    let Ok(doc) = serde_yaml::to_value(contract) else {
        push(
            "PARITY-016",
            "this contract could not be re-serialized, so its dates could not be swept for \
             the future horizon. A sweep that did not run must be red, not silent"
                .to_string(),
            "parity".to_string(),
        );
        return;
    };
    walk_dates(&doc, "", &today, push);
}

/// Recursive half of [`validate_parity_date_horizon`].
fn walk_dates(
    node: &serde_yaml::Value,
    path: &str,
    today: &str,
    push: &mut impl FnMut(&str, String, String),
) {
    use crate::schema::parity::{LedgerDate, MAX_FUTURE_DAYS};
    match node {
        serde_yaml::Value::String(s) => {
            if let Some(over) = LedgerDate::overshoot(s, today) {
                push(
                    "PARITY-016",
                    format!(
                        "{path} is the date {:?}, {over} day(s) past the {MAX_FUTURE_DAYS}-day \
                         horizon from today ({today}). EVERY date in this document is bounded \
                         against TODAY - not against another date the same author wrote. \
                         Bounding a DIFFERENCE leaves both ends free: `measured_on: 2099-01-01` \
                         with `valid_until: 2099-06-01` satisfied the 180-day window and made \
                         the row permanently fresh. This sweep is deliberately keyed on the \
                         VALUE and not on the field name, so a date field added later - under \
                         any name, of any type - is bounded before it is written",
                        s.trim(),
                    ),
                    if path.is_empty() { "<root>" } else { path }.to_string(),
                );
            }
        }
        serde_yaml::Value::Sequence(items) => {
            for (i, item) in items.iter().enumerate() {
                walk_dates(item, &format!("{path}[{i}]"), today, push);
            }
        }
        serde_yaml::Value::Mapping(map) => {
            for (k, v) in map {
                let key = k.as_str().map_or_else(|| format!("{k:?}"), str::to_string);
                let child = if path.is_empty() {
                    key
                } else {
                    format!("{path}.{key}")
                };
                walk_dates(v, &child, today, push);
            }
        }
        _ => {}
    }
}

/// PARITY-004: a PINNED competitor version.
///
/// Exactly one comparison in this repo pins one today (`bitsandbytes==0.49.2`);
/// every `uv run --with scikit-learn` beat re-resolves its oracle nightly, so
/// its "win" is against an unnamed build.
fn validate_parity_version_pin(
    row: &crate::schema::parity::ParityRow,
    i: usize,
    push: &mut impl FnMut(&str, String, String),
) {
    let at = format!("parity.rows[{i}].competitor_version");
    let ver = row.competitor_version.trim();
    if ver.is_empty() {
        push(
            "PARITY-004",
            format!(
                "parity.rows[{i}].competitor_version must pin an EXACT version (`0.49.2`, a \
                 git SHA, an image digest) - an unpinned oracle drifts under the claim"
            ),
            at,
        );
    } else if let Some(bad) = unpinned_token(ver) {
        push(
            "PARITY-004",
            format!(
                "parity.rows[{i}].competitor_version {ver:?} contains {bad:?}, which names a \
                 moving target rather than a version"
            ),
            at,
        );
    } else if !ver.chars().any(|c| c.is_ascii_digit()) {
        push(
            "PARITY-004",
            format!(
                "parity.rows[{i}].competitor_version {ver:?} has no digit - a version pin is a \
                 number, a SHA or a digest, not prose"
            ),
            at,
        );
    }
}

/// The plain "this field must not be empty" rules, as a TABLE rather than five
/// near-identical `if` blocks: PARITY-003, -005 (both invocations), -008, -009
/// and -010. Each exists so a row is falsifiable by someone who was not there —
/// a named competitor, both commands written out, an owner on the hook for the
/// re-measurement, a pointer to the receipt, and what was compared.
fn validate_parity_required_fields(
    row: &crate::schema::parity::ParityRow,
    i: usize,
    push: &mut impl FnMut(&str, String, String),
) {
    const REQUIRED: [(&str, &str, &str); 6] = [
        ("PARITY-003", "competitor", "must name the competing tool"),
        (
            "PARITY-005",
            "invocation_apr",
            "must give the exact apr-side command",
        ),
        (
            "PARITY-005",
            "invocation_competitor",
            "must give the exact competitor-side command",
        ),
        (
            "PARITY-008",
            "owner",
            "must name who re-measures this row when it expires",
        ),
        (
            "PARITY-009",
            "evidence",
            "must point at the receipt (a path, a commit, or a contract id)",
        ),
        (
            "PARITY-010",
            "dimension",
            "must name what was compared (decode_tok_s, wall_clock_ratio, accuracy, ...)",
        ),
    ];
    for (rule, field, why) in REQUIRED {
        let value = match field {
            "competitor" => &row.competitor,
            "invocation_apr" => &row.invocation_apr,
            "invocation_competitor" => &row.invocation_competitor,
            "owner" => &row.owner,
            "evidence" => &row.evidence,
            _ => &row.dimension,
        };
        if value.trim().is_empty() {
            push(
                rule,
                format!("parity.rows[{i}].{field} {why}"),
                format!("parity.rows[{i}].{field}"),
            );
        }
    }
}

/// PARITY-007 and PARITY-011: a row's two dates, on EVERY verdict class.
///
/// PARITY-007 requires both to be strict ISO and `valid_until` to follow
/// `measured_on`. Bounding only UNMEASURED rows was the original design and it
/// was backwards: MEASURED is where both withdrawn claims lived.
///
/// PARITY-011 additionally CAPS the window. Check-time freshness is only as
/// strong as the dates it reads, and an unbounded expiry field is an exemption
/// with a date on it: rewriting every `valid_until` to "2099-12-31" satisfied
/// every other rule here, kept `__MEASURED__` at full strength forever, and made
/// "staleness blocks" voluntary. The ceiling is anchored to `measured_on`, not
/// to today, so an OLD but honest measurement stays writable -- it is
/// `is_expired` that then degrades it.
fn validate_parity_row_dates(
    row: &crate::schema::parity::ParityRow,
    i: usize,
    push: &mut impl FnMut(&str, String, String),
) {
    let at = |field: &str| format!("parity.rows[{i}].{field}");
    let measured = crate::schema::parity::parse_iso_date(row.measured_on.trim());
    let until = crate::schema::parity::parse_iso_date(row.valid_until.trim());
    validate_not_in_the_future(
        row.measured_on.trim(),
        &at("measured_on"),
        "a measurement cannot have been taken tomorrow",
        push,
    );

    if measured.is_none() {
        push(
            "PARITY-007",
            format!(
                "parity.rows[{i}].measured_on must be a real ISO date (YYYY-MM-DD), got {:?}",
                row.measured_on
            ),
            at("measured_on"),
        );
    }
    let Some(until) = until else {
        push(
            "PARITY-007",
            format!(
                "parity.rows[{i}].valid_until must be a real ISO date (YYYY-MM-DD), got {:?} - \
                 every verdict class expires, including BETTER and PARITY",
                row.valid_until
            ),
            at("valid_until"),
        );
        return;
    };
    let Some(measured) = measured else { return };

    if until <= measured {
        push(
            "PARITY-007",
            format!(
                "parity.rows[{i}].valid_until ({}) must be AFTER measured_on ({}) - a row that \
                 expires on the day it is taken is permanently stale",
                row.valid_until.trim(),
                row.measured_on.trim()
            ),
            at("valid_until"),
        );
    }

    let span = crate::schema::parity::days_between(row.measured_on.trim(), row.valid_until.trim());
    if span.is_some_and(|d| d > MAX_VALIDITY_DAYS) {
        push(
            "PARITY-011",
            format!(
                "parity.rows[{i}].valid_until ({}) is {} days after measured_on ({}), over the \
                 {MAX_VALIDITY_DAYS}-day ceiling - a row that outlives review is an exemption \
                 with a date on it, which is exactly how 2099-12-31 would have disarmed this gate",
                row.valid_until.trim(),
                span.unwrap_or_default(),
                row.measured_on.trim()
            ),
            at("valid_until"),
        );
    }
}

/// PARITY-017: a date recording that something HAPPENED may not be in the
/// future.
///
/// The horizon (PARITY-016) is uniform — 180 days ahead for every date —
/// because a uniform rule is the one nobody has to remember. This is the
/// refinement that uniformity gives up: `measured_on` and `recorded_on` name
/// past events and belong strictly at or before today. Without it, a row could
/// be dated `today + 179` with a `valid_until` a day later, staying fresh for
/// the full horizon on a measurement that has not happened.
///
/// It is a per-field rule, and therefore the WEAK kind of control — a new field
/// of this shape must be added here by hand. That is acceptable only because
/// forgetting it is bounded: an unlisted field is still inside the horizon, so
/// the worst a forgotten entry costs is 180 days, never 2099.
fn validate_not_in_the_future(
    value: &str,
    location: &str,
    why: &str,
    push: &mut impl FnMut(&str, String, String),
) {
    let Some(today) = crate::schema::parity::today_utc() else {
        return; // PARITY-016 already reported the unusable clock.
    };
    let Some(ahead) = crate::schema::parity::days_between(&today, value) else {
        return; // Not an ISO date: PARITY-007 / PARITY-013 reports it.
    };
    if ahead > 0 {
        push(
            "PARITY-017",
            format!(
                "{location} ({value}) is {ahead} day(s) in the FUTURE as of {today} - {why}. \
                 The horizon (PARITY-016) bounds every date at {} days ahead so that no field \
                 is unbounded; this rule additionally pins the dates that record a PAST event \
                 to the past, so a row cannot be dated forward to buy itself a full horizon of \
                 freshness on a measurement nobody has taken",
                crate::schema::parity::MAX_FUTURE_DAYS,
            ),
            location.to_string(),
        );
    }
}

/// Enforce the DOWNGRADE record (PARITY-012..014).
///
/// A shrink-never floor on `__MEASURED__` mechanically forbids the honest
/// correction, and a ratchet that punishes increasing honesty produces
/// dishonest ledgers. So `MEASURED` may fall — but only against a record that a
/// machine can check. These rules are what make "recorded a reason" mean
/// something other than "wrote a sentence".
fn validate_parity_downgrades(
    ledger: &crate::schema::parity::ParityLedger,
    push: &mut impl FnMut(&str, String, String),
) {
    let mut seen: HashSet<&str> = HashSet::new();
    for (i, d) in ledger.downgrades.iter().enumerate() {
        validate_parity_downgrade_target(ledger, d, i, &mut seen, push);
        validate_parity_downgrade_record(d, i, push);
    }
}

/// PARITY-012 / PARITY-014: the TARGET half of a downgrade.
///
/// PARITY-012 is the hinge. Without it, "delete the row and record a downgrade"
/// would be a cheaper compliant move than measuring, and the whole mechanism
/// would be back where PMAT-733 left it. A downgrade says "this comparison
/// still exists and I currently cannot stand behind its number"; it can never
/// say "this comparison is gone".
///
/// PARITY-014 refuses a record beside a row that still claims a measurement —
/// that is a pre-authorisation for a correction nobody has made.
fn validate_parity_downgrade_target<'a>(
    ledger: &'a crate::schema::parity::ParityLedger,
    d: &'a crate::schema::parity::Downgrade,
    i: usize,
    seen: &mut HashSet<&'a str>,
    push: &mut impl FnMut(&str, String, String),
) {
    let at = |field: &str| format!("parity.downgrades[{i}].{field}");
    let key = d.entry_point.trim();

    if key.is_empty() {
        push(
            "PARITY-012",
            format!("parity.downgrades[{i}].entry_point is required"),
            at("entry_point"),
        );
    } else if !ledger.rows.iter().any(|r| r.entry_point.trim() == key) {
        push(
            "PARITY-012",
            format!(
                "parity.downgrades[{i}].entry_point {key:?} matches no row in this ledger - \
                 a downgrade justifies a row that is STILL PRESENT going UNMEASURED; it \
                 cannot justify a row's DELETION, which is the move this contract exists \
                 to make expensive"
            ),
            at("entry_point"),
        );
    } else if !seen.insert(key) {
        push(
            "PARITY-012",
            format!("parity.downgrades[{i}].entry_point {key:?} is recorded twice"),
            at("entry_point"),
        );
    }
    validate_key_is_canonical(&d.entry_point, &at("entry_point"), "PARITY-012", push);

    if let Some(row) = ledger.rows.iter().find(|r| r.entry_point.trim() == key) {
        if row.verdict.is_some_and(Verdict::is_measured) {
            push(
                "PARITY-014",
                format!(
                    "parity.downgrades[{i}] records a downgrade for {key:?}, but that row \
                     still declares verdict {} - a downgrade may only accompany a row \
                     declared UNMEASURED, or it is a pre-authorisation for a correction \
                     nobody has made",
                    row.verdict.unwrap_or(Verdict::Unmeasured)
                ),
                at("entry_point"),
            );
        }
    }
}

/// PARITY-013: the RECORD half of a downgrade — a reason from the CLOSED
/// vocabulary, an owner, and both dates within the same ceiling as a row's
/// `valid_until`.
///
/// `reason` is a serde enum, so an unknown value never reaches here: it fails
/// to PARSE. That is the point — "recorded a reason" must not be dischargeable
/// by writing a sentence. This arm fires only on an ABSENT reason.
fn validate_parity_downgrade_record(
    d: &crate::schema::parity::Downgrade,
    i: usize,
    push: &mut impl FnMut(&str, String, String),
) {
    let at = |field: &str| format!("parity.downgrades[{i}].{field}");
    // PARITY-013: a reason from the CLOSED vocabulary, an owner, and both
    // dates. `reason` is a serde enum, so an unknown value never reaches
    // here - it fails to parse. This arm fires only on an ABSENT reason.
    if d.reason.is_none() {
        push(
            "PARITY-013",
            format!(
                "parity.downgrades[{i}].reason is required - one of RECEIPT_MISSING / \
                 HARNESS_DELETED / MEASUREMENT_UNDATED / COMPETITOR_UNPINNABLE / \
                 HOST_UNAVAILABLE / SUPERSEDED. Prose is not a reason: the vocabulary is \
                 closed so that 'recorded a reason' cannot be discharged by writing a \
                 sentence"
            ),
            at("reason"),
        );
    }
    if d.owner.trim().is_empty() {
        push(
            "PARITY-013",
            format!(
                "parity.downgrades[{i}].owner must name who owes the re-measurement - an \
                 unowned downgrade is a permanent one"
            ),
            at("owner"),
        );
    }
    validate_not_in_the_future(
        d.recorded_on.trim(),
        &at("recorded_on"),
        "a downgrade cannot have been recorded tomorrow",
        push,
    );
    for (field, value) in [
        ("recorded_on", d.recorded_on.trim()),
        ("recheck_by", d.recheck_by.trim()),
    ] {
        if crate::schema::parity::parse_iso_date(value).is_none() {
            push(
                "PARITY-013",
                format!(
                    "parity.downgrades[{i}].{field} must be a real ISO date (YYYY-MM-DD), \
                     got {value:?}"
                ),
                at(field),
            );
        }
    }
    match crate::schema::parity::days_between(d.recorded_on.trim(), d.recheck_by.trim()) {
        None => {}
        Some(span) if span <= 0 => push(
            "PARITY-013",
            format!(
                "parity.downgrades[{i}].recheck_by ({}) must be AFTER recorded_on ({})",
                d.recheck_by.trim(),
                d.recorded_on.trim()
            ),
            at("recheck_by"),
        ),
        Some(span) if span > MAX_VALIDITY_DAYS => push(
            "PARITY-013",
            format!(
                "parity.downgrades[{i}].recheck_by ({}) is {span} days out, over the \
                 {MAX_VALIDITY_DAYS}-day ceiling - a downgrade dated far enough ahead is \
                 just a deletion that kept its paperwork",
                d.recheck_by.trim()
            ),
            at("recheck_by"),
        ),
        Some(_) => {}
    }
}

/// The CLOSED vocabulary of runners a falsification binding may start with.
///
/// A binding is a command a reader is meant to RUN, so its first token must be
/// something runnable. Without this, `test: "see the case table"` is a
/// "binding" and PARITY-015 degenerates into "the key is present".
const FALSIFICATION_RUNNERS: [&str; 6] = ["cargo", "bash", "sh", "make", "pv", "apr"];

/// PARITY-015: every falsification test must NAME an executable binding.
///
/// # What this rule proves, stated plainly, because it is less than it looks
///
/// It is a STRUCTURAL check and nothing more. It proves that a `test:` or
/// `test_harness:` string EXISTS and begins with a runner from a closed
/// vocabulary. **It does not run the command, does not check that the command
/// exists, and does not check that it passes.** A binding naming a test that
/// was deleted last week satisfies this rule.
///
/// So the provability invariant on this kind — `falsification_tests.len() >=
/// proof_obligations.len()` — is still satisfiable by writing YAML; what this
/// rule changes is only the PRICE. An entry can no longer be pure prose: it
/// must name something with the shape of a command, which a reader can copy
/// and run to discover the lie. That is worth having and it is not proof, and
/// the contract says so in the same words rather than letting the rule's name
/// imply otherwise.
///
/// Making it real means EXECUTING the bindings (`pv probar`-style) and failing
/// on a binding whose command does not exist or does not pass. That is a
/// separate, larger piece of work and it is recorded as such — not quietly
/// implied by this rule's existence. #2465 is this repo's standing lesson that
/// an unbound falsification entry reads as "neither bound nor broken"; an
/// unRUN one reads the same way, one level up.
fn validate_parity_falsification_bindings(
    contract: &Contract,
    push: &mut impl FnMut(&str, String, String),
) {
    for (i, t) in contract.falsification_tests.iter().enumerate() {
        let id = if t.id.trim().is_empty() {
            "<no id>"
        } else {
            t.id.trim()
        };
        let binding = [t.test.as_deref(), t.test_harness.as_deref()]
            .into_iter()
            .flatten()
            .map(str::trim)
            .find(|s| !s.is_empty());
        let Some(binding) = binding else {
            push(
                "PARITY-015",
                format!(
                    "falsification_tests[{i}] ({id}) has no executable binding - add `test:` or \
                     `test_harness:` naming a command a reader can RUN. Without one the \
                     provability invariant (falsification_tests >= proof_obligations) is \
                     discharged by writing more YAML. NOTE what this rule does NOT do: it never \
                     RUNS the command. It is a structural check on the shape of the entry"
                ),
                format!("falsification_tests[{i}].test"),
            );
            continue;
        };
        let runner = binding.split_whitespace().next().unwrap_or("");
        if !FALSIFICATION_RUNNERS.contains(&runner) && !runner.starts_with("./") {
            push(
                "PARITY-015",
                format!(
                    "falsification_tests[{i}] ({id}) binding {binding:?} does not start with a \
                     runner - one of {FALSIFICATION_RUNNERS:?} or a `./` path. A binding is a \
                     command a reader RUNS; if prose counts as a binding then the rule only \
                     proves the key is present, which is the prose problem with an extra step"
                ),
                format!("falsification_tests[{i}].test"),
            );
        }
    }
}

/// The first unpinned token in `version`, if any.
///
/// Tokenises on non-alphanumerics so `main` matches `llama.cpp@main` but not
/// `domain`. Known limitation: an all-alphabetic short git SHA would trip the
/// no-digit arm of PARITY-004; write it as a full 40-char SHA (which always
/// contains a digit in practice) or prefix it with the tag it belongs to.
fn unpinned_token(version: &str) -> Option<&'static str> {
    let lower = version.to_ascii_lowercase();
    for tok in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        if let Some(hit) = UNPINNED_VERSION_TOKENS.iter().find(|t| **t == tok) {
            return Some(hit);
        }
    }
    None
}

/// The four incumbents a BEAT may target (case-insensitive substring match, so
/// `ollama` and `llama.cpp` both satisfy Pillar 4).
const BEAT_INCUMBENTS: [&str; 5] = ["scikit-learn", "pytorch", "unsloth", "ollama", "llama.cpp"];

/// Enforce the BeatBenchmark shape (PMAT-741): a `beat-benchmark` contract MUST
/// carry a well-formed `beat:` block so the claim is a falsifiable CI gate, not
/// prose. Rules BEAT-001..007.
fn validate_beat_benchmark(contract: &Contract, violations: &mut Vec<Violation>) {
    let push = |violations: &mut Vec<Violation>, rule: &str, message: String, field: &str| {
        violations.push(Violation {
            severity: Severity::Error,
            rule: rule.to_string(),
            message,
            location: Some(format!("beat.{field}")),
        });
    };

    let Some(beat) = contract.beat.as_ref() else {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "BEAT-001".to_string(),
            message: "beat-benchmark contract must define a `beat:` block \
                      (incumbent, metric, direction, beat_threshold, ci_gate_name)"
                .to_string(),
            location: Some("beat".to_string()),
        });
        return;
    };

    // BEAT-002: incumbent must name one of the four pillars.
    let incumbent = beat.incumbent.trim().to_lowercase();
    if incumbent.is_empty() {
        push(
            violations,
            "BEAT-002",
            "beat.incumbent must not be empty".to_string(),
            "incumbent",
        );
    } else if !BEAT_INCUMBENTS.iter().any(|p| incumbent.contains(p)) {
        push(
            violations,
            "BEAT-002",
            format!(
                "beat.incumbent {:?} must name one of the four pillars ({})",
                beat.incumbent,
                BEAT_INCUMBENTS.join(", ")
            ),
            "incumbent",
        );
    }

    // BEAT-003: a measured metric is required.
    if beat.metric.trim().is_empty() {
        push(
            violations,
            "BEAT-003",
            "beat.metric must name the measured quantity (e.g. accuracy, wall_clock_ms, \
             tokens_per_sec)"
                .to_string(),
            "metric",
        );
    }

    // BEAT-004: direction fixes which way is a regression.
    match beat.direction.trim() {
        "higher_is_better" | "lower_is_better" => {}
        other => push(
            violations,
            "BEAT-004",
            format!(
                "beat.direction must be `higher_is_better` or `lower_is_better`, got {other:?}"
            ),
            "direction",
        ),
    }

    // BEAT-005: a finite, machine-pinned threshold is required (the gate value).
    match beat.beat_threshold {
        None => push(
            violations,
            "BEAT-005",
            "beat.beat_threshold is required — the pinned value CI fails below".to_string(),
            "beat_threshold",
        ),
        Some(t) if !t.is_finite() => push(
            violations,
            "BEAT-005",
            format!("beat.beat_threshold must be finite, got {t}"),
            "beat_threshold",
        ),
        Some(_) => {}
    }

    // BEAT-006: the enforcing CI gate must be named.
    if beat.ci_gate_name.trim().is_empty() {
        push(
            violations,
            "BEAT-006",
            "beat.ci_gate_name must name the CI test that enforces this gate".to_string(),
            "ci_gate_name",
        );
    }

    // BEAT-007: approved_compute is required and must be CPU or GPU (the
    // autonomous-vs-operator track distinction depends on it).
    match beat
        .approved_compute
        .as_deref()
        .map(|c| c.trim().to_uppercase())
    {
        None => push(
            violations,
            "BEAT-007",
            "beat.approved_compute is required — must be `CPU` or `GPU`".to_string(),
            "approved_compute",
        ),
        Some(ref c) if c != "CPU" && c != "GPU" => push(
            violations,
            "BEAT-007",
            format!(
                "beat.approved_compute must be `CPU` or `GPU`, got {:?}",
                beat.approved_compute
            ),
            "approved_compute",
        ),
        Some(_) => {}
    }
}

/// Enforce the provability invariant: kernel contracts (non-registry) MUST have
/// `proof_obligations`, `falsification_tests`, and `kani_harnesses`.
fn validate_provability_invariant(contract: &Contract, violations: &mut Vec<Violation>) {
    for v in contract.provability_violations() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "PROVABILITY-001".to_string(),
            message: v,
            location: None,
        });
    }
}

fn validate_metadata(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.metadata.references.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-001".to_string(),
            message: "metadata.references must not be empty — \
                      every contract must cite its source paper(s)"
                .to_string(),
            location: Some("metadata.references".to_string()),
        });
    }

    if contract.metadata.version.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-002".to_string(),
            message: "metadata.version must not be empty".to_string(),
            location: Some("metadata.version".to_string()),
        });
    }
}

fn validate_equations(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.equations.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-003".to_string(),
            message: "equations must contain at least one equation".to_string(),
            location: Some("equations".to_string()),
        });
    }

    for (name, eq) in &contract.equations {
        if eq.formula.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-004".to_string(),
                message: format!("equations.{name}.formula must not be empty"),
                location: Some(format!("equations.{name}.formula")),
            });
        }
    }
}

fn validate_proof_obligations(contract: &Contract, violations: &mut Vec<Violation>) {
    use crate::schema::types::ObligationType;

    let mut seen_ids = HashSet::new();
    for (i, ob) in contract.proof_obligations.iter().enumerate() {
        if ob.property.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-005".to_string(),
                message: format!("proof_obligations[{i}].property must not be empty"),
                location: Some(format!("proof_obligations[{i}].property")),
            });
        }
        if let Some(ref formal) = ob.formal {
            if !seen_ids.insert(formal.clone()) {
                violations.push(Violation {
                    severity: Severity::Warning,
                    rule: "SCHEMA-006".to_string(),
                    message: format!("Duplicate formal predicate: {formal}"),
                    location: Some(format!("proof_obligations[{i}].formal")),
                });
            }
        }

        // DbC field/type constraints
        if ob.requires.is_some() && ob.obligation_type != ObligationType::Postcondition {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-014".to_string(),
                message: format!(
                    "proof_obligations[{i}].requires is only valid on \
                     postcondition obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].requires")),
            });
        }

        if ob.applies_to_phase.is_some()
            && ob.obligation_type != ObligationType::LoopInvariant
            && ob.obligation_type != ObligationType::LoopVariant
        {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-015".to_string(),
                message: format!(
                    "proof_obligations[{i}].applies_to_phase is only valid on \
                     loop_invariant or loop_variant obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].applies_to_phase")),
            });
        }

        if ob.parent_contract.is_some() && ob.obligation_type != ObligationType::Subcontract {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-016".to_string(),
                message: format!(
                    "proof_obligations[{i}].parent_contract is only valid on \
                     subcontract obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].parent_contract")),
            });
        }

        // Subcontract parent_contract must be in depends_on
        if let Some(ref parent) = ob.parent_contract {
            if ob.obligation_type == ObligationType::Subcontract
                && !contract.metadata.depends_on.contains(parent)
            {
                violations.push(Violation {
                    severity: Severity::Error,
                    rule: "SCHEMA-017".to_string(),
                    message: format!(
                        "proof_obligations[{i}].parent_contract \"{parent}\" \
                         must be listed in metadata.depends_on"
                    ),
                    location: Some(format!("proof_obligations[{i}].parent_contract")),
                });
            }
        }
    }
}

fn validate_falsification_tests(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut ids = HashSet::new();
    for test in &contract.falsification_tests {
        if !ids.insert(&test.id) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-007".to_string(),
                message: format!("Duplicate falsification test ID: {}", test.id),
                location: Some(format!("falsification_tests.{}", test.id)),
            });
        }
        if test.prediction.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-008".to_string(),
                message: format!(
                    "falsification_tests.{}.prediction must not be empty — \
                     every test must make a falsifiable prediction",
                    test.id
                ),
                location: Some(format!("falsification_tests.{}.prediction", test.id)),
            });
        }
        if test.if_fails.is_empty() {
            violations.push(Violation {
                severity: Severity::Warning,
                rule: "SCHEMA-009".to_string(),
                message: format!(
                    "falsification_tests.{}.if_fails is empty — \
                     should describe root cause diagnosis",
                    test.id
                ),
                location: Some(format!("falsification_tests.{}.if_fails", test.id)),
            });
        }
    }
}

fn validate_kani_harnesses(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut ids = HashSet::new();
    for harness in &contract.kani_harnesses {
        if !ids.insert(&harness.id) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-010".to_string(),
                message: format!("Duplicate Kani harness ID: {}", harness.id),
                location: Some(format!("kani_harnesses.{}", harness.id)),
            });
        }
        if harness.obligation.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-011".to_string(),
                message: format!(
                    "kani_harnesses.{}.obligation must not be empty — \
                     every harness must reference a proof obligation",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.obligation", harness.id)),
            });
        }
        if harness.bound.is_none() {
            violations.push(Violation {
                severity: Severity::Warning,
                rule: "SCHEMA-012".to_string(),
                message: format!(
                    "kani_harnesses.{}.bound not specified — \
                     Kani requires an unwind bound",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.bound", harness.id)),
            });
        }
    }
}

fn validate_qa_gate(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.qa_gate.is_none() {
        violations.push(Violation {
            severity: Severity::Warning,
            rule: "SCHEMA-013".to_string(),
            message: "No qa_gate defined — contract should define a \
                      certeza quality gate"
                .to_string(),
            location: Some("qa_gate".to_string()),
        });
    }
}

#[cfg(test)]
mod tests {
    include!("validator_tests.rs");
}

#[cfg(test)]
mod parity_validator_tests {
    include!("validator_parity_tests.rs");
}
