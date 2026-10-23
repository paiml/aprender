//! Parsed-YAML comparison for the roadmap tools: the semantic half of the additive-diff rule.
//! Only entries whose BYTES differ are ever parsed.

use serde_json::Value;

/// Parse `text` as one YAML document into a JSON value. An empty document is `Null`, as
/// PyYAML's `safe_load("")` is `None`; the error is the first line of the parser's message.
///
/// # Errors
///
/// Returns the first line of the parser's message when `text` is not valid YAML.
pub fn parse(text: &str) -> Result<Value, String> {
    if text
        .lines()
        .all(|l| l.trim().is_empty() || l.trim_start().starts_with('#'))
    {
        return Ok(Value::Null);
    }
    let strict = |t: &str| {
        crate::yaml::from_str_merging::<Value>(t).map_err(|e| {
            e.to_string()
                .lines()
                .next()
                .unwrap_or("unknown error")
                .to_string()
        })
    };
    let mut msg = match strict(text) {
        Ok(v) => return Ok(v),
        Err(m) => m,
    };
    // PyYAML is laxer than the YAML spec in two ways the real roadmap relies on (see
    // `repair_position`). On exactly those errors the text is repaired at the reported position
    // and re-parsed. The repair is made on the one entry that holds the error first (cheap),
    // then the whole file is re-parsed to find the next one. Bounded: any other error is real.
    let mut lines: Vec<String> = text.split_inclusive('\n').map(str::to_string).collect();
    for _ in 0..MAX_COMMENT_REPAIRS {
        let Some((line, col, pad)) = repair_position(&msg) else {
            return Err(msg);
        };
        repair_entry(&mut lines, line, col, pad)?;
        match strict(&lines.concat()) {
            Ok(v) => return Ok(v),
            Err(m) => msg = m,
        }
    }
    Err("too many repairs".to_string())
}

const MAX_COMMENT_REPAIRS: usize = 4096;

fn is_entry_start(l: &str) -> bool {
    l.starts_with("- id:")
}

/// Apply the repair at 1-based (`line`, `col`) of `lines`, then keep repairing the SAME entry
/// block, parsed alone, for as long as it reports a repairable error.
fn repair_entry(lines: &mut [String], line: usize, col: usize, pad: &str) -> Result<(), String> {
    insert_in_line(lines, line, col, pad)?;
    let start = (0..line)
        .rev()
        .find(|&i| is_entry_start(&lines[i]))
        .unwrap_or(0);
    let end = (line..lines.len())
        .find(|&i| is_entry_start(&lines[i]))
        .unwrap_or(lines.len());
    for _ in 0..MAX_COMMENT_REPAIRS {
        let block = lines[start..end].concat();
        let msg = match crate::yaml::from_str_merging::<Value>(&block) {
            Ok(_) => return Ok(()),
            Err(e) => e.to_string().lines().next().unwrap_or("").to_string(),
        };
        match repair_position(&msg) {
            Some((l, c, p)) => insert_in_line(lines, start + l, c, p)?,
            None => return Ok(()),
        }
    }
    Ok(())
}

fn insert_in_line(lines: &mut [String], line: usize, col: usize, pad: &str) -> Result<(), String> {
    let l = lines
        .get_mut(line.wrapping_sub(1))
        .ok_or("repair position outside the text")?;
    let at = l
        .char_indices()
        .nth(col.saturating_sub(1))
        .map_or(l.len(), |(b, _)| b);
    l.insert_str(at, pad);
    Ok(())
}

/// `(line, column, text to insert)` for the two spec-strictness errors PyYAML does not raise:
/// a `#` glued to a closing quote (insert one space), and a continuation line of a multi-line
/// quoted scalar that is not indented (insert indentation, which folding strips again, so the
/// scalar's value is unchanged).
fn repair_position(msg: &str) -> Option<(usize, usize, &'static str)> {
    let pad = if msg.contains("comments must be separated from other tokens by whitespace") {
        " "
    } else if msg.contains("invalid indentation in multiline quoted scalar") {
        "    "
    } else {
        return None;
    };
    let after_line = msg.split("line ").nth(1)?;
    let line = after_line
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()?;
    let after_col = msg.split("column ").nth(1)?;
    let col = after_col
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()?;
    Some((line, col, pad))
}

fn timestamp_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        // YAML 1.1 timestamps, exactly as PyYAML's resolver recognises them.
        regex::Regex::new(
            r"^([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?:(?:[Tt]|[ \t]+)([0-9]{1,2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]*))?(?:[ \t]*(Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?)?$",
        )
        .expect("static timestamp regex")
    })
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// What PyYAML would turn this plain scalar into, as a comparable key: a date, a naive
/// datetime, or an aware datetime (by UTC instant). `None` when the string is not a timestamp.
/// Two timestamp-shaped strings that denote one instant compare equal, as Python's datetimes do.
fn timestamp_key(s: &str) -> Option<String> {
    let c = timestamp_re().captures(s)?;
    let n = |i: usize| c.get(i).and_then(|g| g.as_str().parse::<i64>().ok());
    let (y, mo, d) = (n(1)?, n(2)?, n(3)?);
    let Some(h) = n(4) else {
        return Some(format!("date {y}-{mo}-{d}"));
    };
    let (mi, se) = (n(5)?, n(6)?);
    let frac: String = c
        .get(7)
        .map_or("", |g| g.as_str())
        .chars()
        .chain("000000".chars())
        .take(6)
        .collect();
    let micro: i64 = frac.parse().ok()?;
    let local = ((days_from_civil(y, mo, d) * 24 + h) * 60 + mi) * 60 + se;
    let Some(tz) = c.get(8).map(|g| g.as_str()) else {
        return Some(format!("naive {local} {micro}"));
    };
    let off = if tz == "Z" {
        0
    } else {
        let sign = if tz.starts_with('-') { -1 } else { 1 };
        let body = &tz[1..];
        let (hh, mm) = body.split_once(':').map_or((body, "0"), |(a, b)| (a, b));
        sign * (hh.parse::<i64>().ok()? * 3600 + mm.parse::<i64>().ok()? * 60)
    };
    Some(format!("aware {} {micro}", local - off))
}

/// Python-style equality: numbers compare by value (`1 == 1.0`), mappings unordered.
#[must_use]
pub fn veq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(x), Value::String(y)) => {
            x == y || timestamp_key(x).is_some_and(|k| Some(k) == timestamp_key(y))
        }
        (Value::Number(x), Value::Number(y)) => x == y || x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| veq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| veq(v, w)))
        }
        _ => a == b,
    }
}

/// Object equality over two maps (the entry mappings).
#[must_use]
pub fn map_eq(a: &serde_json::Map<String, Value>, b: &serde_json::Map<String, Value>) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| veq(v, w)))
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    #[test]
    fn timestamps_denoting_one_instant_compare_equal() {
        use serde_json::json;
        let eq = |a: &str, b: &str| super::veq(&json!(a), &json!(b));
        assert!(eq("2026-09-21T00:00:00Z", "2026-09-21 00:00:00+00:00"));
        assert!(eq("2026-09-21T02:00:00+02:00", "2026-09-21T00:00:00Z"));
        assert!(!eq("2026-09-21T00:00:00Z", "2026-09-21T00:00:01Z"));
        assert!(!eq("2026-09-21", "2026-09-21T00:00:00Z"));
        assert!(!eq("2026-09-21T00:00:00", "2026-09-21T00:00:00Z"));
        assert!(!eq("not a date", "2026-09-21"));
    }

    #[test]
    fn hash_glued_to_a_closing_quote_reads_as_a_comment_like_pyyaml() {
        let v = super::parse("a: 'x'#note\nb: 2\n").unwrap();
        assert_eq!(v["a"], "x");
        assert_eq!(v["b"], 2);
        assert!(super::parse("a: [\n").is_err());
    }

    use super::*;

    #[test]
    fn numbers_and_maps_compare_like_python() {
        let a = parse("a: 1\nb: [1, 2]\n").expect("a");
        let b = parse("b: [1.0, 2]\na: 1\n").expect("b");
        assert!(veq(&a, &b));
        assert!(!veq(&a, &parse("a: 2\nb: [1, 2]\n").expect("c")));
    }

    #[test]
    fn empty_document_is_null_and_bad_yaml_is_an_error() {
        assert_eq!(parse("# only a comment\n"), Ok(Value::Null));
        assert!(parse("a: [1, 2\n").is_err());
    }

    #[test]
    fn aliases_resolve_across_entries() {
        let v = parse("r:\n- id: a\n  c: &x 5\n- id: b\n  c: *x\n").expect("alias doc");
        assert_eq!(v["r"][1]["c"], serde_json::json!(5));
    }

    /// YAML 1.1 (PyYAML `safe_load`, which the retired Python tools used) against the YAML 1.2
    /// parser the port uses, one row per plain scalar. `py` is what PyYAML returned, recorded
    /// from `yaml.safe_load` (PyYAML 6, 2026-10-23) and not re-derived here; `same` says the port
    /// parses the same value. A row that is not `same` is a KNOWN divergence, pinned so it can
    /// only change on purpose, and `roadmap_fields_use_no_diverging_scalar` below proves the
    /// real roadmap never writes one of them as a plain scalar.
    #[test]
    fn yaml_1_1_scalar_case_table() {
        use serde_json::json;
        // (document, PyYAML's value as JSON, the port's value, same?)
        let rows: &[(&str, serde_json::Value, serde_json::Value, bool)] = &[
            ("a: true", json!(true), json!(true), true),
            ("a: ~", json!(null), json!(null), true),
            ("a: null", json!(null), json!(null), true),
            ("a: 1_000", json!(1000), json!(1000), true),
            ("a: 0x1F", json!(31), json!(31), true),
            ("a: 0755", json!(493), json!(755.0), false),
            ("a: 1:30", json!(90), json!("1:30"), false),
            ("a: yes", json!(true), json!("yes"), false),
            ("a: on", json!(true), json!("on"), false),
            ("a: off", json!(false), json!("off"), false),
            ("a: No", json!(false), json!("No"), false),
            ("a: n", json!("n"), json!("n"), true),
            ("a: 0o17", json!("0o17"), json!(15), false),
            ("a: 1e3", json!("1e3"), json!(1000.0), false),
            ("a: \"yes\"", json!("yes"), json!("yes"), true),
            (
                "a: \"2026-01-01\"",
                json!("2026-01-01"),
                json!("2026-01-01"),
                true,
            ),
        ];
        let mut bad = Vec::new();
        for (doc, py, port, same) in rows {
            let got = parse(doc).expect(doc)["a"].clone();
            if got != *port || veq(&got, py) != *same {
                bad.push(format!("{doc:?}: port {got}, expected {port}, PyYAML {py}"));
            }
        }
        assert!(
            bad.is_empty(),
            "YAML 1.1 case table drifted:\n{}",
            bad.join("\n")
        );
    }

    /// The divergences above are harmless only while the roadmap never relies on them. Scan every
    /// plain `key: value` scalar of the real roadmap and its fragments for the YAML 1.1 shapes the
    /// port reads differently (bool words, legacy octal, base-60); `y`/`n` are plain strings in both.
    #[test]
    fn roadmap_fields_use_no_diverging_scalar() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/roadmaps");
        let mut files = vec![root.join("roadmap.yaml")];
        if let Ok(rd) = std::fs::read_dir(root.join("entries")) {
            files.extend(
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "yaml")),
            );
        }
        let re = regex::Regex::new(
            r"^\s*(?:- )?[A-Za-z_][\w-]*:\s+(?i:yes|no|on|off|[-+]?0[0-9]+|[0-9]+(?::[0-9]{2})+|[0-9]+(?:_[0-9]+)+)\s*(?:#.*)?$",
        )
        .expect("static regex");
        let mut hits = Vec::new();
        for f in files {
            let Ok(text) = std::fs::read_to_string(&f) else {
                continue;
            };
            for (i, l) in text.lines().enumerate() {
                if re.is_match(l) {
                    hits.push(format!("{}:{}: {}", f.display(), i + 1, l.trim()));
                }
            }
        }
        assert!(
            hits.is_empty(),
            "plain YAML 1.1-only scalars in the roadmap:\n{}",
            hits.join("\n")
        );
    }

    #[test]
    fn quoted_and_plain_timestamps_compare_equal_on_purpose() {
        // PyYAML read the plain one as a datetime and the quoted one as a str, so a pure
        // quoting change was a difference. serde_json has no datetime type, both are strings
        // here, and veq's timestamp_key makes them equal. Deliberate: re-quoting changes no
        // data, and the additive-diff rule is about data. Pinned so it only changes on purpose.
        assert!(veq(
            &serde_json::json!("2026-10-23T07:35:50Z"),
            &serde_json::json!("2026-10-23 07:35:50+00:00")
        ));
        assert!(veq(
            &parse("a: 2026-10-23T07:35:50Z\n").unwrap(),
            &parse("a: '2026-10-23T07:35:50Z'\n").unwrap()
        ));
    }
}
