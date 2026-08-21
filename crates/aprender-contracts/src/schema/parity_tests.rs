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
