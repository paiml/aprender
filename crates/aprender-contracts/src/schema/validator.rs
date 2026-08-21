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
    validate_parity_removals(ledger, &mut push);
    validate_parity_coverage(ledger, &mut push);
    validate_parity_falsification_bindings(contract, &mut push);
    validate_parity_date_horizon(contract, &mut push);
}

/// The COVERAGE RATCHET: PARITY-021 (it exists, and is argued), PARITY-022
/// (the schedule is a schedule), PARITY-023 (it still owes an increase) and
/// PARITY-024 (the step that has come due is MET).
///
/// # What this closes
///
/// Nothing in PARITY-000..020 requires an in-scope entry point to have a row.
/// Five rows over 41 scope entries over 111 live subcommands is "competitive
/// parity" asserted over ~4.5% of the surface, forever, with every gate green.
/// Every other rule makes the rows that EXIST honest; this is the only one
/// that makes rows exist.
///
/// # Why the floor is dated rather than constant
///
/// A constant floor set at what the ledger already has is inert on the day it
/// lands and inert forever after — which is how a coverage floor normally
/// dies. A dated schedule is landable at today's coverage AND owes an
/// increase, and `PARITY-023` refuses a schedule that has stopped owing one.
/// Because every date here is bounded at `MAX_FUTURE_DAYS` from today, the
/// furthest that debt can be pushed out is half a year, and pushing it out at
/// all is a visible edit to a reviewed file judged against protected `main`.
fn validate_parity_coverage<F>(ledger: &crate::schema::parity::ParityLedger, push: &mut F)
where
    F: FnMut(&str, String, String),
{
    use crate::schema::parity::days_between;

    let Some(cov) = ledger.coverage.as_ref() else {
        push(
            "PARITY-021",
            "parity.coverage is missing. Every other rule on this kind makes the rows that \
             EXIST honest; none of them makes rows exist, so a ledger of five impeccable rows \
             over 41 in-scope entry points satisfies all of them forever. Declare a coverage \
             ratchet: `steps:` (a dated, strictly increasing schedule of covered_min), plus \
             `rationale:` and `dissent:`"
                .to_string(),
            "parity.coverage".to_string(),
        );
        return;
    };

    // PARITY-021: the decision is ARGUED, and the argument against it is in the
    // same file. A floor with no recorded objection reads as unanimous, and the
    // next author cannot tell whether the obvious alternative was rejected or
    // never considered.
    for (field, value) in [
        ("rationale", cov.rationale.as_deref()),
        ("dissent", cov.dissent.as_deref()),
    ] {
        if value.is_none_or(|s| s.trim().is_empty()) {
            push(
                "PARITY-021",
                format!(
                    "parity.coverage.{field} must be written out. The ratio being ratcheted is \
                     a judgement call, not a measurement: record why this shape was chosen and \
                     what the case against it is, in the contract, where a reader meets it at \
                     the same time as the rule"
                ),
                format!("parity.coverage.{field}"),
            );
        }
    }

    if cov.steps.is_empty() {
        push(
            "PARITY-022",
            "parity.coverage.steps must not be empty - an empty schedule is a floor of zero \
             that can never come due"
                .to_string(),
            "parity.coverage.steps".to_string(),
        );
        return;
    }

    // PARITY-022: it must be a SCHEDULE — real dates, strictly increasing in
    // both coordinates. Two steps on the same date, or a later date with a
    // lower requirement, is not a ratchet; it is two floors of which the
    // reader has to guess the operative one.
    let mut prev: Option<(&str, usize)> = None;
    let mut prev_scope: Option<usize> = None;
    for (i, step) in cov.steps.iter().enumerate() {
        let at = format!("parity.coverage.steps[{i}]");
        let by = step.by.trim();
        // PARITY-022, the SECOND JOINT. `covered_min` bounds rows against
        // scope; nothing bounded scope against the world, and the measured
        // shape was 5 rows over 41 scope entries over 111 live subcommands.
        // Bounding one joint leaves the claim payable by never widening the
        // audited surface, so the schedule carries both floors and both
        // ratchet.
        //
        // Required on EVERY step rather than optional-with-a-default: a
        // default is a number nobody chose, and a step that silently inherits
        // one is how the second joint went unbounded in the first place. The
        // TYPE keeps it optional so an older ledger on protected `main` still
        // PARSES as a comparand; this rule is what refuses it in a tree under
        // test.
        validate_parity_scope_min(step, i, prev_scope, push);
        if let Some(s) = step.scope_min {
            prev_scope = Some(s);
        }
        if crate::schema::parity::parse_iso_date(by).is_none() {
            push(
                "PARITY-022",
                format!(
                    "coverage step `by` must be an ISO date (YYYY-MM-DD), got {by:?}. An \
                     unreadable date is treated as ALREADY DUE, so this does not buy a deferral \
                     - it only makes the schedule unreviewable"
                ),
                format!("{at}.by"),
            );
        }
        if step.covered_min == 0 {
            push(
                "PARITY-022",
                "coverage step covered_min must be at least 1 - a floor of zero is satisfied by \
                 an empty ledger"
                    .to_string(),
                format!("{at}.covered_min"),
            );
        }
        if let Some((pby, pmin)) = prev {
            let ordered = days_between(pby, by).is_some_and(|d| d > 0);
            if !ordered || step.covered_min <= pmin {
                push(
                    "PARITY-022",
                    format!(
                        "coverage steps must be STRICTLY increasing in both `by` and \
                         `covered_min`; step {i} ({by} -> {}) does not advance on step {} ({pby} \
                         -> {pmin}). A schedule that repeats or reverses is not a ratchet",
                        step.covered_min,
                        i - 1
                    ),
                    at,
                );
            }
        }
        prev = Some((by, step.covered_min));
    }
}

/// PARITY-022, THE SECOND JOINT: `scope_min` on one coverage step.
///
/// `covered_min` bounds ROWS against SCOPE. Nothing bounded SCOPE against the
/// world, and the measured shape was five rows over 41 scope entries over 111
/// live subcommands — so bounding one joint left the whole claim payable by
/// never widening the audited surface, with every gate green while the audited
/// fraction of a growing CLI fell.
///
/// Required on EVERY step rather than optional-with-a-default: a default is a
/// number nobody chose, and a step that silently inherits one is how the second
/// joint went unbounded in the first place. The TYPE keeps it `Option` so an
/// older ledger on protected `main` still PARSES as a comparand; this rule is
/// what refuses it in a tree under test.
fn validate_parity_scope_min<F>(
    step: &crate::schema::parity::CoverageStep,
    i: usize,
    prev_scope: Option<usize>,
    push: &mut F,
) where
    F: FnMut(&str, String, String),
{
    let at = format!("parity.coverage.steps[{i}].scope_min");
    let by = step.by.trim();
    let Some(s) = step.scope_min else {
        push(
            "PARITY-022",
            format!(
                "coverage step {i} declares no `scope_min`. `covered_min` bounds ROWS against \
                 SCOPE and nothing bounds SCOPE against the live surface - five rows over 41 \
                 scope entries over 111 live subcommands satisfies every other rule here \
                 forever. Declare the entry points that must BE in scope from {by} onward"
            ),
            at,
        );
        return;
    };
    if s == 0 {
        push(
            "PARITY-022",
            "coverage step scope_min must be at least 1 - a scope floor of zero is satisfied by \
             an empty scope file, which is a 100% ratio over nothing"
                .to_string(),
            at,
        );
        return;
    }
    if s < step.covered_min {
        push(
            "PARITY-022",
            format!(
                "coverage step {i} requires {} covered entry point(s) but only {s} in scope. \
                 Every ledger row must be IN scope, so covered can never exceed scope: this \
                 step is unsatisfiable as written",
                step.covered_min
            ),
            at.clone(),
        );
    }
    if let Some(ps) = prev_scope {
        if s <= ps {
            push(
                "PARITY-022",
                format!(
                    "coverage steps must be STRICTLY increasing in `scope_min` too; step {i} \
                     ({by} -> {s}) does not advance on step {} ({ps}). A surface floor that \
                     repeats has stopped widening, which is how the audited fraction of a \
                     growing CLI shrinks with every gate green",
                    i - 1
                ),
                at,
            );
        }
    }
}

/// PARITY-023 and PARITY-024 — evaluated against a DATE, so they live with the
/// other check-time rules rather than in `validate_contract`, which answers a
/// question about the file rather than about the world.
///
/// Returns the messages to report; empty means the ratchet is satisfied.
#[must_use]
pub fn parity_coverage_debt(
    ledger: &crate::schema::parity::ParityLedger,
    today: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    let Some(cov) = ledger.coverage.as_ref() else {
        return out; // PARITY-021 already reports the absent block.
    };
    let (achieved, floor) = ledger.coverage_status(today);
    if achieved < floor {
        out.push(format!(
            "PARITY-024: the coverage ratchet requires {floor} distinct in-scope entry \
             point(s) to carry a row as of {today}, and this ledger covers {achieved}. Add \
             rows - a verdict of UNMEASURED with an owner and a bound is a legitimate row and \
             costs no measurement. Lowering the step instead is refused against the schedule on \
             protected `main`."
        ));
    }
    if cov.next_step(today).is_none() {
        out.push(format!(
            "PARITY-023: the coverage schedule owes no future increase as of {today} - every \
             step has come due. A ratchet that owes nothing has stopped ratcheting, which is \
             exactly how a coverage floor dies: set once at the achievable value and never \
             moved. Add a step with a later `by` and a higher `covered_min`. Note that every \
             date here is bounded at {} days from today, so the next step can never be more \
             than half a year out, and moving one further out is refused against `main`.",
            crate::schema::parity::MAX_FUTURE_DAYS
        ));
    }
    out
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

/// Enforce the REMOVAL record (PARITY-025..027).
///
/// # What this is for
///
/// The shell ratchet used to excuse a deleted ROW, and a deleted SCOPE line,
/// on one condition: that the entry point was absent from the LIVE
/// ENUMERATION. That enumeration comes from `apr --help` of a binary built
/// FROM THE BRANCH, so the author of the deletion also authored its excuse.
/// Deleting a subcommand, its scope line and its row in a single commit
/// removed a losing comparison at rc=0 with nothing recorded anywhere — the
/// PMAT-733 move performed one level down, on the thing measured instead of on
/// the measurement.
///
/// Retirement stays possible, because a rule that forbids retiring a command
/// is a rule that gets deleted. It stops being free: the shell ratchet now
/// requires BOTH that the entry point is genuinely gone AND that a record here
/// names it. These rules are what make that record mean something other than a
/// line of text.
///
/// # The VALUE is not judged
///
/// As everywhere else on this kind: nothing here asks whether the removal was
/// a good idea, and nothing refuses the removal of a row whose verdict was
/// `WORSE`. A rule that admitted only convenient deletions would be the
/// fabrication engine wearing a fourth hat. What is refused is a deletion that
/// names no owner, no date and no reason a machine can check.
fn validate_parity_removals(
    ledger: &crate::schema::parity::ParityLedger,
    push: &mut impl FnMut(&str, String, String),
) {
    let mut seen: HashSet<&str> = HashSet::new();
    for (i, r) in ledger.removals.iter().enumerate() {
        validate_parity_removal_shape(r, i, &mut seen, push);
        validate_parity_removal_disjoint(ledger, r, i, push);
        validate_parity_removal_replacement(r, i, push);
    }
}

/// PARITY-025: the removal record's own shape — a named thing, an owner, a
/// date, and a reason from the closed vocabulary.
fn validate_parity_removal_shape<'a>(
    r: &'a crate::schema::parity::Removal,
    i: usize,
    seen: &mut HashSet<&'a str>,
    push: &mut impl FnMut(&str, String, String),
) {
    use crate::schema::parity::RemovalReason;

    let at = |field: &str| format!("parity.removals[{i}].{field}");
    let key = r.entry_point.trim();

    if key.is_empty() {
        push(
            "PARITY-025",
            format!("parity.removals[{i}].entry_point is required"),
            at("entry_point"),
        );
    } else if !seen.insert(key) {
        push(
            "PARITY-025",
            format!("parity.removals[{i}].entry_point {key:?} is recorded twice"),
            at("entry_point"),
        );
    }
    validate_key_is_canonical(&r.entry_point, &at("entry_point"), "PARITY-025", push);

    if r.reason.is_none() {
        push(
            "PARITY-025",
            format!(
                "parity.removals[{i}].reason is required, from the closed vocabulary {}. \
                 Prose is not accepted here for the same reason it is not accepted on a \
                 downgrade: `serde` refuses an unknown value, so \"recorded a reason\" cannot \
                 be discharged by writing a sentence",
                RemovalReason::vocabulary()
            ),
            at("reason"),
        );
    }
    if r.owner.trim().is_empty() {
        push(
            "PARITY-025",
            format!(
                "parity.removals[{i}].owner is required - a deletion with no owner is the \
                 deletion this contract exists to make expensive"
            ),
            at("owner"),
        );
    }
    if r.recorded_on.is_empty() {
        push(
            "PARITY-025",
            format!("parity.removals[{i}].recorded_on is required (ISO YYYY-MM-DD)"),
            at("recorded_on"),
        );
        return;
    }
    if crate::schema::parity::parse_iso_date(r.recorded_on.trim()).is_none() {
        push(
            "PARITY-025",
            format!(
                "parity.removals[{i}].recorded_on must be a real ISO date (YYYY-MM-DD), got {:?}",
                r.recorded_on.trim()
            ),
            at("recorded_on"),
        );
    }
    validate_not_in_the_future(
        r.recorded_on.trim(),
        &at("recorded_on"),
        "a removal cannot have been recorded tomorrow",
        push,
    );
}

/// PARITY-026: the removal set is DISJOINT from the row set and from the
/// downgrade set.
///
/// The mirror of PARITY-012, and load-bearing in the same way. PARITY-012
/// stops a DELETION being dressed as a downgrade by requiring the row to still
/// exist; this stops a LIVE row being pre-authorised for deletion by requiring
/// that it does not. Without it a removal record could sit in the ledger
/// beside the row it names, doing nothing visible, until the commit that
/// deletes the row spends it — a permission issued long before the change and
/// therefore unreviewable at the moment it matters. That is the
/// self-issued-permission shape round 5 removed from the verdict channel,
/// rebuilt one block over.
fn validate_parity_removal_disjoint(
    ledger: &crate::schema::parity::ParityLedger,
    r: &crate::schema::parity::Removal,
    i: usize,
    push: &mut impl FnMut(&str, String, String),
) {
    let at = format!("parity.removals[{i}].entry_point");
    let key = r.entry_point.trim();
    if key.is_empty() {
        return;
    }
    if ledger.rows.iter().any(|row| row.entry_point.trim() == key) {
        push(
            "PARITY-026",
            format!(
                "parity.removals[{i}].entry_point {key:?} still matches a row in this ledger. \
                 A removal record says \"this entry point is GONE\"; while the row is present \
                 that is false, and a record parked beside a live row is a pre-authorisation \
                 for a deletion nobody has made yet"
            ),
            at.clone(),
        );
    }
    if ledger
        .downgrades
        .iter()
        .any(|d| d.entry_point.trim() == key)
    {
        push(
            "PARITY-026",
            format!(
                "parity.removals[{i}].entry_point {key:?} is also recorded in \
                 parity.downgrades. The two blocks make opposite claims - a downgrade requires \
                 the row to be PRESENT (PARITY-012) and a removal requires it to be GONE - so \
                 one record must not be spendable on both sides"
            ),
            at,
        );
    }
}

/// PARITY-027: a `RENAMED` or `MERGED_INTO` removal must name where the
/// capability went, and may not name itself.
///
/// A rename that points at nothing is a retirement, and calling it a rename
/// hides that the capability left. Naming itself discharges the rule while
/// describing nothing — the shape of every "required" field that is
/// satisfiable by an echo. The shell ratchet additionally requires the
/// successor to be LIVE and IN SCOPE, which is a question about the world and
/// so cannot be answered here.
fn validate_parity_removal_replacement(
    r: &crate::schema::parity::Removal,
    i: usize,
    push: &mut impl FnMut(&str, String, String),
) {
    use crate::schema::parity::RemovalReason;

    let at = format!("parity.removals[{i}].replacement");
    let key = r.entry_point.trim();
    let replacement = r.replacement.as_deref().map(str::trim).unwrap_or("");
    if replacement.is_empty() {
        if r.reason.is_some_and(RemovalReason::requires_replacement) {
            push(
                "PARITY-027",
                format!(
                    "parity.removals[{i}].replacement is required for reason {} - a rename that \
                     points at nothing is a retirement, and calling it a rename hides that the \
                     capability left. Name the entry point that carries it now, or record the \
                     removal as RETIRED",
                    r.reason.unwrap_or(RemovalReason::Retired)
                ),
                at,
            );
        }
        return;
    }
    validate_key_is_canonical(replacement, &at, "PARITY-027", push);
    if replacement == key {
        push(
            "PARITY-027",
            format!(
                "parity.removals[{i}].replacement {replacement:?} is the entry point being \
                 removed. A record naming itself as its own successor discharges the rule \
                 while describing nothing"
            ),
            at,
        );
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
        validate_parity_transition(d, row, i, push);
    }
}

/// PARITY-014 / PARITY-019: the record must DESCRIBE the transition it excuses.
///
/// PARITY-014 refuses a record beside a row that does not declare the verdict
/// the record says it does — that is a pre-authorisation for a correction
/// nobody has made. In the legacy shape (`to_verdict` absent) the destination
/// is `UNMEASURED`, which is exactly the rule as it was first written.
///
/// PARITY-019 is the new half, and it is what makes a transition record cost
/// anything. The shell ratchet excuses a verdict change only against a record
/// whose `from_verdict` matches the verdict in the COMMITTED baseline and
/// whose `to_verdict` matches the verdict declared now. If either end could be
/// left blank, one record would launder every future relabelling of that row:
/// write `to_verdict` with no `from_verdict` and the record stops naming a
/// direction, which is precisely the property that made deleting the
/// StandardScaler row cheaper than keeping it.
///
/// The VALUES stay unconstrained. Nothing here says a verdict may not become
/// `WORSE`, or must become `BETTER`; a rule admitting only wins is the
/// fabrication engine this whole contract exists to disarm. What is refused is
/// a change that names no direction, no owner and no date.
fn validate_parity_transition(
    d: &crate::schema::parity::Downgrade,
    row: &crate::schema::parity::ParityRow,
    i: usize,
    push: &mut impl FnMut(&str, String, String),
) {
    let at = |field: &str| format!("parity.downgrades[{i}].{field}");
    let key = d.entry_point.trim();
    let declared = row.verdict.unwrap_or(Verdict::Unmeasured);
    let destination = d.destination();

    if declared != destination {
        push(
            "PARITY-014",
            format!(
                "parity.downgrades[{i}] records a transition for {key:?} ending at {destination}, \
                 but that row declares verdict {declared}. A record must describe the state the \
                 row is ACTUALLY in{} - otherwise it is a pre-authorisation for a correction \
                 nobody has made",
                if d.to_verdict.is_none() {
                    ", and a record with no `to_verdict` is the legacy downgrade shape, whose \
                     destination is UNMEASURED"
                } else {
                    ""
                }
            ),
            at("entry_point"),
        );
    }

    // PARITY-019: a transition names BOTH ends, and they differ.
    match (d.from_verdict, d.to_verdict) {
        (None, None) => {}
        (Some(_), None) => push(
            "PARITY-019",
            format!(
                "parity.downgrades[{i}] names from_verdict but no to_verdict. A transition record \
                 names BOTH ends: the ratchet matches from_verdict against the verdict in the \
                 COMMITTED baseline and to_verdict against the verdict declared now, so a record \
                 with one end open excuses every future relabelling of {key:?} rather than the \
                 one it was written for"
            ),
            at("to_verdict"),
        ),
        (None, Some(to)) => push(
            "PARITY-019",
            format!(
                "parity.downgrades[{i}] declares to_verdict {to} for {key:?} with no \
                 from_verdict. A record that names no direction is not a record of a change - it \
                 is a standing permission to change"
            ),
            at("from_verdict"),
        ),
        (Some(from), Some(to)) if from == to => push(
            "PARITY-019",
            format!(
                "parity.downgrades[{i}] records {key:?} moving from {from} to {to}, which is not \
                 a move. A record exists to own a CHANGE; one that describes none excuses none, \
                 and leaving it in the ledger spends budget that a real transition will need"
            ),
            at("from_verdict"),
        ),
        (Some(_), Some(_)) => {}
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
                "parity.downgrades[{i}].reason is required - one of {}. Prose is not a \
                 reason: the vocabulary is closed so that 'recorded a reason' cannot be \
                 discharged by writing a sentence",
                crate::schema::parity::DowngradeReason::vocabulary()
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
        validate_obligation_field_types(contract, ob, i, violations);
    }
}

/// SCHEMA-014..017: which DbC fields are legal on which obligation TYPE.
///
/// Split out of `validate_proof_obligations` as a pure extraction (no rule
/// changed) because that function sat at cognitive 30 against the repo's
/// threshold of 25 and blocked every commit touching this file — pre-existing
/// debt, charged to whoever next edits the file, which is the shape the
/// bin→lib lint-reattribution lesson warns about.
fn validate_obligation_field_types(
    contract: &Contract,
    ob: &crate::schema::types::ProofObligation,
    i: usize,
    violations: &mut Vec<Violation>,
) {
    use crate::schema::types::ObligationType;

    {
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
