// Unit tests for the competitive-parity ledger types.
// `include!`d from parity.rs so they live inside the crate (cargo test --lib).

use super::*;

fn row(verdict: Option<Verdict>, valid_until: &str) -> ParityRow {
    ParityRow {
        entry_point: "apr run".into(),
        competitor: "ollama".into(),
        competitor_version: "0.12.4".into(),
        invocation_apr: "apr run m.gguf".into(),
        invocation_competitor: "ollama run m".into(),
        dimension: "decode_tok_s".into(),
        verdict,
        // `unchecked` on purpose: these fixtures probe expiry arithmetic, and
        // several deliberately carry dates the horizon would refuse. There is
        // no `From<&str>` for LedgerDate precisely so that a construction which
        // skips the bound has to SAY it does, in the diff.
        measured_on: LedgerDate::unchecked("2026-07-31"),
        valid_until: LedgerDate::unchecked(valid_until),
        owner: "pillar-4".into(),
        evidence: "contracts/x.yaml".into(),
        note: None,
        extra: std::collections::BTreeMap::new(),
    }
}

/// An ISO date `days` from today, so a fixture that must sit inside or outside
/// the future horizon says so ARITHMETICALLY and keeps saying it in a year.
///
/// The alternative — a literal like `2099-12-31` — would be an out-of-horizon
/// date today and an in-horizon one eventually, so the test would silently
/// stop testing. Deliberately NOT a clock override: an override is one more
/// unbounded knob, which is the class this whole change closes.
fn days_from_today(days: i64) -> String {
    let today = today_utc().expect("system clock is after the epoch");
    let (y, m, d) = parse_iso_date(&today).expect("today_utc emits an ISO date");
    let (y, m, d) = civil_from_days(days_from_civil(i64::from(y), u32::from(m), u32::from(d)) + days);
    format!("{y:04}-{m:02}-{d:02}")
}

// ---------------------------------------------------------------------------
// LedgerDate — the class control, at the type level

/// The bound lives in `Deserialize`, so it cannot be skipped by a caller who
/// forgets a rule. Probed here directly on the TYPE rather than through a
/// contract fixture, so a future refactor that moves the check out of the type
/// and into a validator turns this red.
#[test]
fn ledger_date_refuses_a_value_past_the_future_horizon() {
    let inside = days_from_today(MAX_FUTURE_DAYS);
    let outside = days_from_today(MAX_FUTURE_DAYS + 1);
    assert_ne!(inside, outside, "the two fixtures must differ");

    let ok: LedgerDate =
        serde_yaml::from_str(&format!("{inside:?}")).expect("exactly the horizon is accepted");
    assert_eq!(ok.trim(), inside);

    let err = serde_yaml::from_str::<LedgerDate>(&format!("{outside:?}"))
        .expect_err("one day past the horizon is refused");
    assert!(
        err.to_string().contains("horizon"),
        "refused for the RIGHT reason -- a red for the wrong reason proves \
         nothing: {err}"
    );
}

/// A PAST date is always fine, however old. The bound is a horizon, not a
/// window: an honest record of an old measurement must stay writable, and it is
/// `is_expired` that then degrades it.
#[test]
fn ledger_date_accepts_any_past_date() {
    for d in ["1970-01-01", "2020-02-29", "2026-01-01"] {
        serde_yaml::from_str::<LedgerDate>(&format!("{d:?}"))
            .unwrap_or_else(|e| panic!("{d} must be accepted: {e}"));
    }
}

/// Non-ISO text is NOT a parse error — PARITY-007/013 report it, quoting the
/// value. An unparseable date cannot be a futurity exemption because
/// `is_expired` fails closed on it.
#[test]
fn ledger_date_leaves_non_iso_text_to_the_validator() {
    let d: LedgerDate = serde_yaml::from_str("\"whenever\"").expect("prose parses");
    assert_eq!(d.trim(), "whenever");
    assert!(parse_iso_date(d.trim()).is_none());
    assert!(LedgerDate::overshoot(d.trim(), "2026-08-21").is_none());
}

/// The horizon is MONOTONE: a date only ever becomes less future, so a document
/// accepted today is accepted forever after. Asserted rather than assumed,
/// because the alternative is a gate that reds by the calendar.
#[test]
fn the_future_horizon_never_turns_an_accepted_date_red_later() {
    let d = days_from_today(MAX_FUTURE_DAYS);
    assert!(LedgerDate::overshoot(&d, &days_from_today(0)).is_none());
    for later in [1, 30, 365, 10_000] {
        assert!(
            LedgerDate::overshoot(&d, &days_from_today(later)).is_none(),
            "{d} must stay acceptable {later} day(s) from now"
        );
    }
}

// ---------------------------------------------------------------------------
// Ratchet keys

#[test]
fn a_key_containing_a_control_character_is_not_canonical() {
    assert!(is_canonical_key("apr run --gpu"));
    assert!(is_canonical_key(
        "apr run --gpu (concurrency=1 single-request decode)"
    ));
    assert!(is_canonical_key("lib:aprender-core::Lasso::fit"));
    // THE INJECTION. One row, several well-formed `__ROW__=` lines.
    assert!(!is_canonical_key("apr qa\n__ROW__=lib:aprender-core::X::f"));
    assert!(!is_canonical_key("apr\tqa"));
    assert!(!is_canonical_key("apr code "));
    assert!(!is_canonical_key(" apr code"));
    assert!(!is_canonical_key(""));
    // Non-ASCII: keeps the emitter's BYTE length equal to a shell's character
    // count under any locale.
    assert!(!is_canonical_key("apr café"));
    assert_eq!(bad_key_byte("apr qa"), None);
    assert_eq!(bad_key_byte("a\nb"), Some((1, '\n')));
}

// ---------------------------------------------------------------------------
// The closed vocabulary

#[test]
fn verdict_round_trips_screaming_snake() {
    for (v, s) in [
        (Verdict::Better, "BETTER"),
        (Verdict::Parity, "PARITY"),
        (Verdict::Worse, "WORSE"),
        (Verdict::NotComparable, "NOT_COMPARABLE"),
        (Verdict::Unmeasured, "UNMEASURED"),
    ] {
        assert_eq!(v.to_string(), s);
        let parsed: Verdict = serde_yaml::from_str(s).expect("verdict parses");
        assert_eq!(parsed, v);
    }
}

/// A verdict outside the vocabulary is a PARSE error, not a lint. No ledger can
/// invent "MOSTLY_BETTER".
#[test]
fn unknown_verdict_is_a_parse_error() {
    let r: Result<Verdict, _> = serde_yaml::from_str("MOSTLY_BETTER");
    assert!(r.is_err(), "unknown verdict must not parse");
    let lower: Result<Verdict, _> = serde_yaml::from_str("better");
    assert!(lower.is_err(), "verdicts are SCREAMING_SNAKE_CASE only");
}

/// WORSE counts as MEASURED. This is the inversion in one assertion: recording
/// a loss is a measurement, so deleting it costs a point on the ratchet.
#[test]
fn worse_and_not_comparable_count_as_measured() {
    assert!(Verdict::Worse.is_measured());
    assert!(Verdict::NotComparable.is_measured());
    assert!(Verdict::Better.is_measured());
    assert!(Verdict::Parity.is_measured());
    assert!(!Verdict::Unmeasured.is_measured());
}

// ---------------------------------------------------------------------------
// Freshness, evaluated at check time, for EVERY verdict class

#[test]
fn a_live_row_keeps_its_verdict() {
    let r = row(Some(Verdict::Better), "2026-12-31");
    assert!(!r.is_expired("2026-08-21"));
    assert_eq!(r.effective_verdict("2026-08-21"), Verdict::Better);
}

/// The 1.371x mechanism. A BETTER row past its expiry is UNMEASURED, which
/// lowers `__MEASURED__`, which fails the ratchet. Bounding only UNMEASURED
/// rows -- the first design -- would have left this row claiming BETTER
/// forever.
#[test]
fn an_expired_better_row_degrades_to_unmeasured() {
    let r = row(Some(Verdict::Better), "2026-07-01");
    assert!(r.is_expired("2026-08-21"));
    assert_eq!(r.effective_verdict("2026-08-21"), Verdict::Unmeasured);
    assert!(!r.effective_verdict("2026-08-21").is_measured());
}

/// Every class expires, not just the convenient ones.
#[test]
fn expiry_applies_to_parity_worse_and_not_comparable_alike() {
    for v in [
        Verdict::Better,
        Verdict::Parity,
        Verdict::Worse,
        Verdict::NotComparable,
    ] {
        let r = row(Some(v), "2026-07-01");
        assert_eq!(
            r.effective_verdict("2026-08-21"),
            Verdict::Unmeasured,
            "{v} did not degrade on expiry"
        );
    }
}

/// The boundary: valid ON the expiry date, stale the day after.
#[test]
fn expiry_boundary_is_inclusive_of_the_named_day() {
    let r = row(Some(Verdict::Parity), "2026-08-21");
    assert!(!r.is_expired("2026-08-21"));
    assert!(r.is_expired("2026-08-22"));
}

/// Fail CLOSED: a missing or malformed bound expires the row rather than
/// blessing it forever. "No expiry" is exactly the state the withdrawn claims
/// lived in.
#[test]
fn a_missing_or_malformed_valid_until_is_expired() {
    for bad in ["", "soon", "2026-13-01", "2026-02-30", "26-08-21", "2026-8-21"] {
        let r = row(Some(Verdict::Better), bad);
        assert!(r.is_expired("2026-08-21"), "{bad:?} must read as expired");
        assert_eq!(r.effective_verdict("2026-08-21"), Verdict::Unmeasured);
    }
}

#[test]
fn a_row_with_no_verdict_is_unmeasured() {
    let r = row(None, "2026-12-31");
    assert_eq!(r.effective_verdict("2026-08-21"), Verdict::Unmeasured);
}

// ---------------------------------------------------------------------------
// Ledger aggregates

#[test]
fn measured_count_excludes_expired_and_unmeasured_rows() {
    let mut ledger = ParityLedger::default();
    let mut mk = |ep: &str, v: Verdict, until: &str| {
        let mut r = row(Some(v), until);
        r.entry_point = ep.into();
        ledger.rows.push(r);
    };
    mk("a", Verdict::Better, "2026-12-31"); // live   -> measured
    mk("b", Verdict::Worse, "2026-12-31"); // live   -> measured
    mk("c", Verdict::NotComparable, "2026-12-31"); // live   -> measured
    mk("d", Verdict::Better, "2026-01-01"); // EXPIRED -> not
    mk("e", Verdict::Unmeasured, "2026-12-31"); // live   -> not

    assert_eq!(ledger.measured_count("2026-08-21"), 3);
    assert_eq!(ledger.expired_rows("2026-08-21").len(), 1);
    // Non-wins: worse, not_comparable, the degraded row, and unmeasured.
    assert_eq!(ledger.non_win_count("2026-08-21"), 4);
}

/// Deleting a losing row LOWERS `__MEASURED__`. That is the PMAT-733
/// countermeasure stated as a property: the StandardScaler row was removed
/// because removal was cheaper than recording 0.69x, and under this ledger it
/// is strictly more expensive.
#[test]
fn deleting_a_worse_row_lowers_the_measured_count() {
    let mut ledger = ParityLedger::default();
    let mut a = row(Some(Verdict::Better), "2026-12-31");
    a.entry_point = "kept".into();
    let mut b = row(Some(Verdict::Worse), "2026-12-31");
    b.entry_point = "standard-scaler".into();
    ledger.rows.push(a);
    ledger.rows.push(b);
    let before = ledger.measured_count("2026-08-21");

    ledger.rows.retain(|r| r.entry_point != "standard-scaler");
    let after = ledger.measured_count("2026-08-21");

    assert_eq!(before, 2);
    assert_eq!(after, 1);
    assert!(after < before, "deletion must be visible as a LOSS of coverage");
}

// ---------------------------------------------------------------------------
// Date handling

#[test]
fn iso_dates_are_strict_and_leap_year_aware() {
    assert_eq!(parse_iso_date("2026-08-21"), Some((2026, 8, 21)));
    assert_eq!(parse_iso_date("2028-02-29"), Some((2028, 2, 29))); // leap
    assert_eq!(parse_iso_date("2027-02-29"), None); // not leap
    assert_eq!(parse_iso_date("2100-02-29"), None); // century, not leap
    assert_eq!(parse_iso_date("2000-02-29"), Some((2000, 2, 29))); // 400-year
    assert_eq!(parse_iso_date("2026-04-31"), None);
    assert_eq!(parse_iso_date("2026-00-10"), None);
    assert_eq!(parse_iso_date("2026-08-00"), None);
    assert_eq!(parse_iso_date("2026-8-21"), None);
    assert_eq!(parse_iso_date("2026/08/21"), None);
    assert_eq!(parse_iso_date(""), None);
    assert_eq!(parse_iso_date("20xx-08-21"), None);
}

/// Tuple ordering is calendar ordering, which is what `is_expired` relies on.
#[test]
fn parsed_dates_order_chronologically() {
    let a = parse_iso_date("2026-08-21").expect("date");
    let b = parse_iso_date("2026-09-01").expect("date");
    let c = parse_iso_date("2027-01-01").expect("date");
    assert!(a < b && b < c);
}

#[test]
fn civil_from_days_matches_known_epochs() {
    assert_eq!(civil_from_days(0), (1970, 1, 1));
    assert_eq!(civil_from_days(-1), (1969, 12, 31));
    assert_eq!(civil_from_days(19_723), (2024, 1, 1));
    assert_eq!(civil_from_days(20_686), (2026, 8, 21));
}

#[test]
fn today_utc_is_a_valid_iso_date() {
    let t = today_utc().expect("system clock is after the epoch");
    assert!(parse_iso_date(&t).is_some(), "today_utc gave {t:?}");
}

// ==========================================================================
// THE EXCUSE BUDGET.
//
// `Downgrade::is_overdue` bounds how LONG one excuse lasts. Nothing bounded how
// MANY may be outstanding at once, and the two are independent: a fresh,
// in-date, owned, closed-vocabulary record for EVERY row drives `__MEASURED__`
// to zero with every individual rule satisfied and every gate green. Each
// record impeccable; the aggregate a ledger that has stopped making claims.
// ==========================================================================

fn excuse(entry: &str, recheck_by: &str) -> Downgrade {
    Downgrade {
        entry_point: entry.into(),
        reason: Some(DowngradeReason::ReceiptMissing),
        from_verdict: Some(Verdict::Worse),
        to_verdict: Some(Verdict::Unmeasured),
        owner: "pillar-4".into(),
        recorded_on: LedgerDate::unchecked("2026-08-21"),
        recheck_by: LedgerDate::unchecked(recheck_by),
        detail: None,
        extra: std::collections::BTreeMap::new(),
    }
}

fn named_row(entry: &str, verdict: Option<Verdict>) -> ParityRow {
    let mut r = row(verdict, "2099-01-01");
    r.entry_point = entry.into();
    r
}

/// The ledger in the tree today: one excuse against three measured rows.
#[test]
fn excuse_budget_allows_a_minority_of_excused_rows() {
    let l = ParityLedger {
        rows: vec![
            named_row("a", Some(Verdict::Worse)),
            named_row("b", Some(Verdict::Parity)),
            named_row("c", Some(Verdict::NotComparable)),
            named_row("d", Some(Verdict::Unmeasured)),
        ],
        downgrades: vec![excuse("d", "2099-01-01")],
        ..ParityLedger::default()
    };
    assert_eq!(l.excuse_budget("2026-08-21"), (1, 3));
}

/// The drain-to-zero attack: an impeccable record for every row. Every
/// individual rule holds and the aggregate must not.
#[test]
fn excuse_budget_refuses_a_ledger_excused_into_silence() {
    let l = ParityLedger {
        rows: vec![
            named_row("a", Some(Verdict::Unmeasured)),
            named_row("b", Some(Verdict::Unmeasured)),
            named_row("c", Some(Verdict::Unmeasured)),
        ],
        downgrades: vec![
            excuse("a", "2099-01-01"),
            excuse("b", "2099-01-01"),
            excuse("c", "2099-01-01"),
        ],
        ..ParityLedger::default()
    };
    let (spent, allowed) = l.excuse_budget("2026-08-21");
    assert_eq!((spent, allowed), (3, 0));
    assert!(spent > allowed, "three debts against no measurement");
}

/// The budget has GIVE: paying a debt buys the capacity to take another. A
/// bound with no give is what made the honest `apr code` correction
/// mechanically forbidden, and a ratchet that punishes honesty produces
/// dishonest ledgers.
#[test]
fn excuse_budget_grows_as_debts_are_paid() {
    let mut l = ParityLedger {
        rows: vec![
            named_row("a", Some(Verdict::Unmeasured)),
            named_row("b", Some(Verdict::Unmeasured)),
        ],
        downgrades: vec![excuse("a", "2099-01-01"), excuse("b", "2099-01-01")],
        ..ParityLedger::default()
    };
    let (spent, allowed) = l.excuse_budget("2026-08-21");
    assert!(spent > allowed, "2 excused, 0 measured: over budget");

    // Re-measure `a`.
    l.rows[0].verdict = Some(Verdict::Worse);
    l.downgrades.remove(0);
    let (spent, allowed) = l.excuse_budget("2026-08-21");
    assert_eq!((spent, allowed), (1, 1));
    assert!(spent <= allowed, "paying one debt makes room for the other");
}

/// An OVERDUE record spends no budget -- it also excuses nothing, which is
/// what makes the debt come due rather than the ledger silently loosen.
#[test]
fn excuse_budget_ignores_overdue_records() {
    let l = ParityLedger {
        rows: vec![
            named_row("a", Some(Verdict::Unmeasured)),
            named_row("b", Some(Verdict::Worse)),
        ],
        downgrades: vec![excuse("a", "2020-01-01")],
        ..ParityLedger::default()
    };
    assert_eq!(l.excuse_budget("2026-08-21"), (0, 1));
}

/// Counted per unique entry point, so duplicate records cannot inflate the
/// numerator. (PARITY-012 refuses duplicates anyway; this must not depend on
/// that rule still holding.)
#[test]
fn excuse_budget_counts_rows_not_records() {
    let l = ParityLedger {
        rows: vec![
            named_row("a", Some(Verdict::Unmeasured)),
            named_row("b", Some(Verdict::Worse)),
        ],
        downgrades: vec![excuse("a", "2099-01-01"), excuse("a", "2099-01-01")],
        ..ParityLedger::default()
    };
    assert_eq!(l.excuse_budget("2026-08-21"), (1, 1));
}

/// A record whose destination is not UNMEASURED describes a row that is still
/// MEASURED. It must not also pay for a later, unrelated exit from the measured
/// set -- that would make the second move free, which is the whole class of
/// defect this file keeps closing.
#[test]
fn a_transition_record_does_not_double_as_an_unmeasured_excuse() {
    let mut d = excuse("a", "2099-01-01");
    d.to_verdict = Some(Verdict::NotComparable);
    assert!(!d.excuses_unmeasured());
    assert_eq!(d.destination(), Verdict::NotComparable);

    let legacy = Downgrade {
        to_verdict: None,
        ..excuse("a", "2099-01-01")
    };
    assert!(legacy.excuses_unmeasured(), "the legacy shape still excuses");
    assert_eq!(legacy.destination(), Verdict::Unmeasured);
}

/// The upward reasons are in the vocabulary, and the vocabulary renders every
/// one of them -- so the diagnostic that lists it cannot drift from the enum.
#[test]
fn downgrade_reason_vocabulary_lists_every_variant() {
    let v = DowngradeReason::vocabulary();
    for r in DowngradeReason::all() {
        assert!(v.contains(r.as_str()), "{} missing from {v}", r.as_str());
    }
    assert!(
        v.contains("REMEASURED"),
        "an honest re-measurement must be recordable without lying about why"
    );
}

/// A minimal transition record. `unchecked` on the dates for the same reason
/// `row` uses it: several fixtures deliberately probe expiry arithmetic.
fn downgrade(entry: &str, recheck_by: &str) -> Downgrade {
    Downgrade {
        entry_point: entry.into(),
        reason: Some(DowngradeReason::Remeasured),
        from_verdict: None,
        to_verdict: None,
        owner: "pillar-x".into(),
        recorded_on: LedgerDate::unchecked("2026-08-21"),
        recheck_by: LedgerDate::unchecked(recheck_by),
        detail: None,
        extra: std::collections::BTreeMap::new(),
    }
}

// -- COVERAGE: the scope-key collapse, and why it is load-bearing ------------

#[test]
fn parity_021_scope_key_collapses_a_qualified_entry_point() {
    // Two rows on the same subcommand are two legitimate comparison surfaces
    // and ONE covered entry point. Without the collapse the coverage ratchet
    // would be payable by splitting a row in half.
    assert_eq!(ParityLedger::scope_key("apr run --gpu"), "apr run");
    assert_eq!(
        ParityLedger::scope_key("apr run --gpu (concurrency=1 single-request decode)"),
        "apr run"
    );
    // `lib:` and `bin:` keys are already exact and must pass through intact --
    // truncating them would merge unrelated library surfaces into one.
    assert_eq!(
        ParityLedger::scope_key("lib:aprender-core::Lasso::fit"),
        "lib:aprender-core::Lasso::fit"
    );
    assert_eq!(ParityLedger::scope_key("bin:pv"), "bin:pv");
    // Surrounding whitespace is not a different entry point.
    assert_eq!(ParityLedger::scope_key("  apr qa  "), "apr qa");
}

#[test]
fn parity_024_covered_count_is_distinct_entry_points_not_rows() {
    let mut l = ParityLedger::default();
    let mut a = row(Some(Verdict::Worse), "2026-12-31");
    a.entry_point = "apr run --gpu".into();
    let mut b = row(Some(Verdict::Unmeasured), "2026-12-31");
    b.entry_point = "apr run --gpu (concurrency=1 single-request decode)".into();
    let mut c = row(Some(Verdict::Parity), "2026-12-31");
    c.entry_point = "lib:aprender-core::Lasso::fit".into();
    l.rows = vec![a, b, c];
    assert_eq!(l.rows.len(), 3, "three rows");
    assert_eq!(
        l.covered_count(),
        2,
        "two DISTINCT entry points -- splitting a row must not buy coverage"
    );
}

#[test]
fn parity_023_recorded_upgrades_are_the_give_in_the_non_win_floor() {
    // NON_WINS_MIN=5 over 5 rows was SATURATED: recording an honest BETTER was
    // mechanically forbidden. The floor now moves with the recorded upgrades,
    // and only IN-DATE ones count -- an expired excuse pays for nothing, the
    // same rule the downgrade path already uses.
    let mut l = ParityLedger::default();
    let mut up = downgrade("apr run", "2026-12-31");
    up.from_verdict = Some(Verdict::Worse);
    up.to_verdict = Some(Verdict::Better);
    l.downgrades = vec![up];
    assert_eq!(l.recorded_upgrades("2026-08-21"), 1);
    // OVERDUE -> pays for nothing.
    assert_eq!(l.recorded_upgrades("2027-01-01"), 0);
    // A record that is NOT an upgrade is not counted, so a downgrade cannot be
    // spent twice -- once to excuse leaving the measured set and once to lower
    // the non-win floor.
    let mut down = downgrade("apr qa", "2026-12-31");
    down.from_verdict = Some(Verdict::Worse);
    down.to_verdict = Some(Verdict::Unmeasured);
    l.downgrades.push(down);
    assert_eq!(l.recorded_upgrades("2026-08-21"), 1);
}

#[test]
fn parity_022_declared_measured_is_clock_independent() {
    // The PRIOR side of the ratchet is read from `main` days or weeks after it
    // landed. Computed from EFFECTIVE verdicts it would shed entries as those
    // rows aged, and the bar the current tree must clear would fall on a day
    // nobody touched either file.
    let mut l = ParityLedger::default();
    l.rows = vec![row(Some(Verdict::Worse), "2020-01-01")];
    assert_eq!(
        l.measured_count("2026-08-21"),
        0,
        "effective: the expired row is not measured today"
    );
    assert_eq!(
        l.declared_measured_rows().len(),
        1,
        "declared: what the protected state SAID, whatever the clock says now"
    );
}
