//! Civil-date arithmetic (days since 1970-01-01), the same 60 lines every forecasting spike used.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}
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
pub fn format_ymd(days: i64) -> String { let (y, m, d) = civil_from_days(days); format!("{y:04}-{m:02}-{d:02}") }

/// Strict YYYY-MM-DD (a time part is ignored); impossible dates are refused.
pub fn parse_date(s: &str) -> Result<i64, String> {
    let date = s.get(..10).ok_or_else(|| format!("bad date {s:?}: want YYYY-MM-DD"))?;
    let mut it = date.split('-');
    let mut next = |what: &str| -> Result<i64, String> { it.next().and_then(|p| p.parse::<i64>().ok()).ok_or_else(|| format!("bad date {s:?}: cannot read {what}")) };
    let (y, m, d) = (next("year")?, next("month")?, next("day")?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) { return Err(format!("bad date {s:?}: month/day out of range")); }
    let days = days_from_civil(y, m as u32, d as u32);
    let (yy, mm, dd) = civil_from_days(days);
    if (yy, mm as i64, dd as i64) != (y, m, d) { return Err(format!("bad date {s:?}: not a calendar date")); }
    Ok(days)
}

/// `make_future_dataframe` for D / W / MS.
pub fn future_days(last: i64, horizon: usize, freq: &str) -> Result<Vec<i64>, String> {
    Ok(match freq {
        "D" => (1..=horizon as i64).map(|i| last + i).collect(),
        "W" => (1..=horizon as i64).map(|i| last + 7 * i).collect(),
        "MS" => {
            let (y, m, _) = civil_from_days(last);
            (1..=horizon as i64).map(|i| { let total = (y * 12 + (m as i64 - 1)) + i; days_from_civil(total.div_euclid(12), (total.rem_euclid(12) + 1) as u32, 1) }).collect()
        }
        other => return Err(format!("unsupported freq {other:?}: use D, W or MS")),
    })
}
