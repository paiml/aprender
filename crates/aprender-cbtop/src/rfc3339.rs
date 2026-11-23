//! Std-only RFC 3339 UTC timestamps (PMAT-7191), byte-identical to the former
//! `chrono::Utc::now().to_rfc3339()`:
//! `YYYY-MM-DDTHH:MM:SS[.fff[fff[fff]]]+00:00`, fractional digits only as
//! needed, in groups of 3.

/// Format a UTC instant (seconds since the Unix epoch plus nanoseconds).
pub(crate) fn format_rfc3339_utc(secs: i64, nanos: u32) -> String {
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    // Civil-from-days (Howard Hinnant), proleptic Gregorian.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let frac = if nanos == 0 {
        String::new()
    } else if nanos.is_multiple_of(1_000_000) {
        format!(".{:03}", nanos / 1_000_000)
    } else if nanos.is_multiple_of(1_000) {
        format!(".{:06}", nanos / 1_000)
    } else {
        format!(".{nanos:09}")
    };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}{frac}+00:00",
        tod / 3_600,
        tod % 3_600 / 60,
        tod % 60
    )
}

/// Current UTC time as RFC 3339.
pub(crate) fn now_rfc3339() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format_rfc3339_utc(
        i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        d.subsec_nanos(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_rfc3339_utc_matches_chrono_output() {
        assert_eq!(format_rfc3339_utc(0, 0), "1970-01-01T00:00:00+00:00");
        assert_eq!(
            format_rfc3339_utc(951_782_400, 0),
            "2000-02-29T00:00:00+00:00"
        );
        assert_eq!(
            format_rfc3339_utc(1_700_000_000, 0),
            "2023-11-14T22:13:20+00:00"
        );
        assert_eq!(
            format_rfc3339_utc(4_102_444_800, 0),
            "2100-01-01T00:00:00+00:00"
        );
        assert_eq!(
            format_rfc3339_utc(4_107_542_400, 0),
            "2100-03-01T00:00:00+00:00"
        );
        assert_eq!(format_rfc3339_utc(-86_400, 0), "1969-12-31T00:00:00+00:00");
        assert_eq!(format_rfc3339_utc(-1, 0), "1969-12-31T23:59:59+00:00");
        assert_eq!(
            format_rfc3339_utc(1_700_000_000, 123_000_000),
            "2023-11-14T22:13:20.123+00:00"
        );
        assert_eq!(
            format_rfc3339_utc(1_700_000_000, 123_456_000),
            "2023-11-14T22:13:20.123456+00:00"
        );
        assert_eq!(
            format_rfc3339_utc(1_700_000_000, 123_456_789),
            "2023-11-14T22:13:20.123456789+00:00"
        );
    }

    #[test]
    fn now_rfc3339_shape() {
        let s = now_rfc3339();
        assert!(s.ends_with("+00:00"));
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[10..11], "T");
    }
}
