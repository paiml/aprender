//! Civil-date arithmetic (days since 1970-01-01), the same 60 lines every forecasting spike used.
//!
//! D-17: no calendar-library dependency anywhere in the Phase 6 crates. Dates are
//! `i64` days since the epoch; `future_days` covers `D`, `W` and `MS` and refuses
//! everything else by name.
//!
//! Ported verbatim from `sources/007-chronos-mcp-thin-server/src/dates.rs` (D-08),
//! with two mechanical changes: the fallible functions return [`ForecastError`]
//! instead of `String` (the spike-004 form), and [`parse_date`] carries ONE
//! recorded deviation — see its own docs (REVIEW-06-U2).

use crate::types::ForecastError;

/// Days since 1970-01-01 for a proleptic-Gregorian civil date (Howard Hinnant's algorithm).
#[must_use]
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`]: `(year, month, day)` for a day count.
#[must_use]
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Infallible `YYYY-MM-DD` parse for TRUSTED input (fixtures, generated series).
///
/// Untrusted input goes through [`parse_date`], which refuses instead of panicking.
///
/// # Panics
///
/// Panics if `s` is not three `-`-separated integers.
#[must_use]
pub fn parse_ymd(s: &str) -> i64 {
    let mut it = s.split('-');
    let y: i64 = it.next().expect("year").parse().expect("year int");
    let m: u32 = it.next().expect("month").parse().expect("month int");
    let d: u32 = it.next().expect("day").parse().expect("day int");
    days_from_civil(y, m, d)
}

/// `YYYY-MM-DD` rendering of a day count.
#[must_use]
pub fn format_ymd(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Strict `YYYY-MM-DD`; impossible dates are refused.
///
/// RECORDED D-08 DEVIATION (REVIEW-06-U2). The spike took a ten-byte PREFIX and never
/// looked at what followed, so `"2024-01-01garbage"` parsed as 2024-01-01. D-11 says
/// the tool boundary refuses rather than defaults, so the shape is now checked BEFORE
/// splitting: EXACTLY ten ASCII bytes, `-` at index 4 and 7, the other eight all ASCII
/// digits. The calendar round-trip check below is unchanged, so `"2008-02-30"` is still
/// refused with the same `not a calendar date` message. Every committed fixture and CSV
/// carries a bare `YYYY-MM-DD`, so no parity number can move.
///
/// # Errors
///
/// [`ForecastError::Validation`] naming `YYYY-MM-DD` when the shape is wrong, and a
/// `not a calendar date` refusal when the fields are in range but the date does not exist.
pub fn parse_date(s: &str) -> Result<i64, ForecastError> {
    let bad_shape = || ForecastError::Validation(format!("bad date {s:?}: want YYYY-MM-DD"));
    if s.len() != 10 || !s.is_ascii() {
        return Err(bad_shape());
    }
    let b = s.as_bytes();
    if b[4] != b'-' || b[7] != b'-' {
        return Err(bad_shape());
    }
    if !(b[..4].iter().all(u8::is_ascii_digit)
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[8..].iter().all(u8::is_ascii_digit))
    {
        return Err(bad_shape());
    }
    // The shape gate above has already proven 10 ASCII bytes, `-` at 4 and 7, and
    // digits everywhere else — so there is no parse that can fail here. The former
    // `cannot read {what}` error was unreachable and untestable; digits in, i64 out.
    let num =
        |r: std::ops::Range<usize>| b[r].iter().fold(0i64, |a, c| a * 10 + i64::from(*c - b'0'));
    let (y, m, d) = (num(0..4), num(5..7), num(8..10));
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(ForecastError::Validation(format!(
            "bad date {s:?}: month/day out of range"
        )));
    }
    let days = days_from_civil(y, m as u32, d as u32);
    let (yy, mm, dd) = civil_from_days(days);
    if (yy, i64::from(mm), i64::from(dd)) != (y, m, d) {
        return Err(ForecastError::Validation(format!(
            "bad date {s:?}: not a calendar date"
        )));
    }
    Ok(days)
}

/// `make_future_dataframe` for `D` / `W` / `MS`.
///
/// # Errors
///
/// [`ForecastError::Validation`] naming `D, W or MS` for any other frequency —
/// `H` in particular is refused (fractional days were never spiked).
pub fn future_days(last: i64, horizon: usize, freq: &str) -> Result<Vec<i64>, ForecastError> {
    Ok(match freq {
        "D" => (1..=horizon as i64).map(|i| last + i).collect(),
        "W" => (1..=horizon as i64).map(|i| last + 7 * i).collect(),
        "MS" => {
            let (y, m, _) = civil_from_days(last);
            (1..=horizon as i64)
                .map(|i| {
                    let total = (y * 12 + (i64::from(m) - 1)) + i;
                    days_from_civil(total.div_euclid(12), (total.rem_euclid(12) + 1) as u32, 1)
                })
                .collect()
        }
        other => {
            return Err(ForecastError::Validation(format!(
                "unsupported freq {other:?}: use D, W or MS"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{civil_from_days, days_from_civil, format_ymd, future_days, parse_date, parse_ymd};
    use crate::types::ForecastError;

    fn refusal(s: &str) -> String {
        match parse_date(s) {
            Err(ForecastError::Validation(m)) => m,
            Err(ForecastError::Internal(m)) => {
                panic!("{s:?} must be a Validation refusal, got internal: {m}")
            }
            Ok(d) => panic!("{s:?} must be refused, parsed as {d}"),
        }
    }

    #[test]
    fn a_bare_ymd_date_parses() {
        assert_eq!(
            parse_date("2008-02-01").expect("a bare YYYY-MM-DD parses"),
            days_from_civil(2008, 2, 1)
        );
        assert_eq!(parse_date("1970-01-01").expect("epoch parses"), 0);
    }

    #[test]
    fn trailing_content_after_ten_bytes_is_refused() {
        // REVIEW-06-U2: the spike's ten-byte prefix take accepted this.
        assert!(refusal("2008-02-01garbage").contains("YYYY-MM-DD"));
        assert!(refusal("2008-02-01T00:00:00").contains("YYYY-MM-DD"));
    }

    #[test]
    fn leading_or_trailing_whitespace_is_refused() {
        assert!(refusal(" 2008-02-01").contains("YYYY-MM-DD"));
        assert!(refusal("2008-02-01 ").contains("YYYY-MM-DD"));
    }

    #[test]
    fn an_empty_string_is_refused() {
        assert!(refusal("").contains("YYYY-MM-DD"));
    }

    #[test]
    fn a_non_ascii_prefix_is_refused() {
        // Ten CHARS but not ten BYTES, and not ASCII either way.
        assert!(refusal("é008-02-01").contains("YYYY-MM-DD"));
        assert!(refusal("日本-02-01").contains("YYYY-MM-DD"));
    }

    #[test]
    fn a_nine_byte_string_is_refused() {
        assert!(refusal("2008-2-01").contains("YYYY-MM-DD"));
        assert!(refusal("208-02-01").contains("YYYY-MM-DD"));
    }

    #[test]
    fn an_impossible_calendar_date_is_still_refused_as_calendar() {
        assert!(refusal("2008-02-30").contains("calendar"));
        assert!(refusal("2008-13-01").contains("month/day out of range"));
    }

    #[test]
    fn civil_days_round_trip_and_format() {
        for days in [-100_000_i64, -1, 0, 1, 12_345, 20_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
            assert_eq!(parse_ymd(&format_ymd(days)), days);
        }
    }

    #[test]
    fn future_days_covers_d_w_ms_and_refuses_h() {
        let last = days_from_civil(2016, 1, 20);
        assert_eq!(
            future_days(last, 3, "D").expect("D"),
            vec![last + 1, last + 2, last + 3]
        );
        assert_eq!(
            future_days(last, 2, "W").expect("W"),
            vec![last + 7, last + 14]
        );
        assert_eq!(
            future_days(days_from_civil(1960, 12, 1), 2, "MS").expect("MS"),
            vec![days_from_civil(1961, 1, 1), days_from_civil(1961, 2, 1)]
        );
        let ForecastError::Validation(msg) = future_days(last, 1, "H").expect_err("H is refused")
        else {
            panic!("H must be a Validation refusal")
        };
        assert!(msg.contains("use D, W or MS"), "{msg}");
    }
}
