//! The `[profile.ci].fail-fast` judge for a nextest config (was the inline Python judge in
//! `scripts/check_nextest_ci_profile_no_fail_fast.sh`, #5032). It prints one
//! `VERDICT <tag> <path>: <msg>` line: `ok` (exit 0) when `[profile.ci]` sets
//! `fail-fast = false`, `FAIL` (exit 1) for every other answer, and `ENV` (exit 2) when the
//! file cannot be read as TOML. ENV is never a pass and never a fail.
//!
//! Two readers, as in the original: the library reader (the `toml` crate, where the original
//! used `tomllib`/`tomli`) and a line-for-line port of its purpose-built reader, which
//! raises on anything outside its scope rather than guess. The original took the
//! purpose-built reader only on an interpreter without `tomllib`/`tomli`; here
//! `--reader fallback` picks it, so both stay testable on any host.

use crate::pystr::{is_py_space, py_repr, py_strip};
use regex::{Captures, Regex};
use std::borrow::Cow;
use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

/// Which reader judges the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reader {
    /// A full TOML parser (the `toml` crate).
    Library,
    /// The purpose-built line reader.
    Fallback,
}

impl Reader {
    /// `--reader` spelling: `library` or `fallback`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "library" => Some(Self::Library),
            "fallback" => Some(Self::Fallback),
            _ => None,
        }
    }

    /// The `reader=` text of a verdict.
    pub fn name(self) -> &'static str {
        match self {
            Self::Library => "toml crate",
            Self::Fallback => "purpose-built reader",
        }
    }
}

/// `fail-fast` as the judge sees it: whether it is the boolean `false`, and its Python
/// `repr()` for the FAIL line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailFast {
    pub is_false: bool,
    pub repr: String,
}

/// `None` when there is no `[profile.ci]` table, `Some(None)` when it does not set
/// `fail-fast`.
pub type Profile = Option<Option<FailFast>>;

/// An error as the original reported it: the Python exception class and `str(e)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyError {
    pub class: &'static str,
    pub msg: String,
}

impl PyError {
    fn new(class: &'static str, msg: impl Into<String>) -> Self {
        Self {
            class,
            msg: msg.into(),
        }
    }

    fn value(msg: String) -> Self {
        Self::new("ValueError", msg)
    }
}

/// The whole command: read `path` with `reader` and judge it. Returns the exit code and
/// stdout.
pub fn run(path: &Path, reader: Reader, runner: &str) -> (u8, String) {
    let shown = path.to_string_lossy();
    let loaded = read(path).and_then(|bytes| load(&bytes, reader));
    verdict(&shown, loaded, reader.name(), runner)
}

/// The profile as `reader` reads it.
pub fn load(bytes: &[u8], reader: Reader) -> Result<Profile, PyError> {
    match reader {
        Reader::Library => load_with_lib(bytes),
        Reader::Fallback => load_minimal(bytes),
    }
}

/// The verdict line for a loaded profile, and its exit code.
pub fn verdict(
    path: &str,
    loaded: Result<Profile, PyError>,
    how: &str,
    runner: &str,
) -> (u8, String) {
    let (code, tag, msg) = match loaded {
        Err(e) => (
            2,
            "ENV",
            format!(
                "cannot be parsed on runner={runner} ({}: {}) -- cannot judge, not a pass and not a fail",
                e.class, e.msg
            ),
        ),
        Ok(None) => (
            1,
            "FAIL",
            format!(
                "no [profile.ci] -- CI runs --profile ci, so the run would take nextest's default, \
                 which is fail-fast ON (reader={how})"
            ),
        ),
        Ok(Some(None)) => (
            1,
            "FAIL",
            format!(
                "[profile.ci] does not set fail-fast -- nextest's default is ON (measured: 3/6 tests \
                 run on a 6-test probe) (reader={how})"
            ),
        ),
        Ok(Some(Some(v))) if !v.is_false => (
            1,
            "FAIL",
            format!(
                "[profile.ci].fail-fast = {} -- a failing test cancels every test still queued and \
                 their verdicts are discarded (reader={how})",
                v.repr
            ),
        ),
        Ok(Some(Some(_))) => (
            0,
            "ok",
            format!(
                "[profile.ci].fail-fast = false -- a red run still measures everything else \
                 (reader={how}, runner={runner})"
            ),
        ),
    };
    (code, format!("VERDICT {tag} {path}: {msg}\n"))
}

/// `open(path, "rb").read()`, with the `OSError` text CPython gives for a failed open.
fn read(path: &Path) -> Result<Vec<u8>, PyError> {
    std::fs::read(path).map_err(|e| {
        let errno = e.raw_os_error().unwrap_or(0);
        let class = match errno {
            1 | 13 => "PermissionError",
            2 => "FileNotFoundError",
            20 => "NotADirectoryError",
            21 => "IsADirectoryError",
            _ => "OSError",
        };
        let text = e.to_string();
        let strerror = text
            .strip_suffix(&format!(" (os error {errno})"))
            .unwrap_or(&text);
        PyError::new(
            class,
            format!(
                "[Errno {errno}] {strerror}: {}",
                py_repr(&path.to_string_lossy())
            ),
        )
    })
}

/// `bytes.decode("utf-8")`, with CPython's `UnicodeDecodeError` text on failure.
fn decode(bytes: &[u8]) -> Result<&str, PyError> {
    std::str::from_utf8(bytes).map_err(|e| {
        let start = e.valid_up_to();
        let (len, reason) = match e.error_len() {
            None => (bytes.len() - start, "unexpected end of data"),
            Some(n) => {
                let first = bytes[start];
                let bad_start = matches!(first, 0x80..=0xc1 | 0xf5..=0xff);
                (
                    n,
                    if bad_start {
                        "invalid start byte"
                    } else {
                        "invalid continuation byte"
                    },
                )
            }
        };
        let at = if len == 1 {
            format!("byte 0x{:02x} in position {start}", bytes[start])
        } else {
            format!("bytes in position {start}-{}", start + len - 1)
        };
        PyError::new(
            "UnicodeDecodeError",
            format!("'utf-8' codec can't decode {at}: {reason}"),
        )
    })
}

// --- the library reader ---------------------------------------------------------------

/// `load_with_lib`: the whole file through a TOML parser, then `profile.ci`.
fn load_with_lib(bytes: &[u8]) -> Result<Profile, PyError> {
    let text = decode(bytes)?;
    if text.starts_with('\u{feff}') {
        // tomllib reads U+FEFF as a character no statement may start with.
        return Err(PyError::new(
            "TOMLDecodeError",
            "Invalid statement (at line 1, column 1)",
        ));
    }
    let doc: toml::Table = toml::from_str(text).map_err(|e| toml_error(text, &e))?;
    // d.get("profile", {}).get("ci"): a `profile` that is not a table has no `.get`.
    let ci = match doc.get("profile") {
        None => None,
        Some(toml::Value::Table(p)) => p.get("ci"),
        Some(other) => {
            return Err(PyError::new(
                "AttributeError",
                format!("'{}' object has no attribute 'get'", py_type(other, true)),
            ))
        }
    };
    match ci {
        None => Ok(None),
        Some(toml::Value::Table(t)) => Ok(Some(t.get("fail-fast").map(|v| FailFast {
            is_false: matches!(v, toml::Value::Boolean(false)),
            repr: py_value_repr(v),
        }))),
        Some(other) => Err(PyError::value(format!(
            "profile.ci is a {}, not a table",
            py_type(other, false)
        ))),
    }
}

/// A parse error on one line, with tomllib's `(at line L, column C)` position.
fn toml_error(text: &str, e: &toml::de::Error) -> PyError {
    let msg = e
        .message()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("; ");
    let at = e
        .span()
        .and_then(|s| text.get(..s.start))
        .map(|before| {
            let line = before.matches('\n').count() + 1;
            let col = before.rsplit('\n').next().map_or(0, |l| l.chars().count()) + 1;
            format!(" (at line {line}, column {col})")
        })
        .unwrap_or_default();
    PyError::new("TOMLDecodeError", format!("{msg}{at}"))
}

/// `type(v).__name__` of the value tomllib returns, or the qualified `tp_name` CPython puts in
/// an `AttributeError`.
fn py_type(v: &toml::Value, qualified: bool) -> &'static str {
    match v {
        toml::Value::String(_) => "str",
        toml::Value::Integer(_) => "int",
        toml::Value::Float(_) => "float",
        toml::Value::Boolean(_) => "bool",
        toml::Value::Array(_) => "list",
        toml::Value::Table(_) => "dict",
        toml::Value::Datetime(d) => match (d.date.is_some(), d.time.is_some(), qualified) {
            (true, true, false) => "datetime",
            (true, true, true) => "datetime.datetime",
            (true, false, false) => "date",
            (true, false, true) => "datetime.date",
            (false, _, false) => "time",
            (false, _, true) => "datetime.time",
        },
    }
}

/// `repr()` of the value tomllib returns for `v`. A datetime prints as its TOML text.
fn py_value_repr(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => py_repr(s),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => py_float_repr(*f),
        toml::Value::Boolean(true) => "True".to_owned(),
        toml::Value::Boolean(false) => "False".to_owned(),
        toml::Value::Array(a) => format!(
            "[{}]",
            a.iter().map(py_value_repr).collect::<Vec<_>>().join(", ")
        ),
        toml::Value::Table(t) => format!(
            "{{{}}}",
            t.iter()
                .map(|(k, v)| format!("{}: {}", py_repr(k), py_value_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        toml::Value::Datetime(d) => d.to_string(),
    }
}

/// `repr(float)`: the shortest round-trip digits, in exponent form when the decimal point
/// would sit more than 16 places right or 4 left of the first digit.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_owned();
    }
    if f.is_infinite() {
        return if f < 0.0 { "-inf" } else { "inf" }.to_owned();
    }
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if f.is_sign_negative() { "-" } else { "" };
    let decpt = exp + 1;
    let body = if decpt <= -4 || decpt > 16 {
        let (first, rest) = digits.split_at(1);
        let frac = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{first}{frac}e{esign}{:02}", exp.unsigned_abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat(decpt.unsigned_abs() as usize))
    } else {
        let point = decpt.unsigned_abs() as usize;
        if point >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point - digits.len()))
        } else {
            format!("{}.{}", &digits[..point], &digits[point..])
        }
    };
    format!("{sign}{body}")
}

// --- the purpose-built reader ---------------------------------------------------------

/// Python's `\s` in a `str` pattern: `str.isspace()`, which adds `\x1c`-`\x1f` to Unicode
/// `White_Space` (see `pystr::is_py_space`).
const PY_SPACE: &str = r"[\s\x1c-\x1f]";

static QUOTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""(?:[^"\\]|\\.)*"|'[^']*'"#).expect("static regex"));
static HEADER: LazyLock<Regex> = LazyLock::new(|| {
    let re = format!(r#"^\[(\[)?{PY_SPACE}*([A-Za-z0-9_.\-"' ]+?){PY_SPACE}*\](\])?$"#);
    Regex::new(&re).expect("static regex")
});
static KEY: LazyLock<Regex> = LazyLock::new(|| {
    let re = format!(r#"^([A-Za-z0-9_\-]+|"[^"]+"|'[^']+'){PY_SPACE}*={PY_SPACE}*(.*)$"#);
    Regex::new(&re).expect("static regex")
});
static INT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^-?\d+$").expect("static regex"));
static DIGIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d$").expect("static regex"));

/// A value the purpose-built reader typed.
enum Typed {
    Bool(bool),
    Int(String),
    Str(String),
    Raw,
}

impl Typed {
    /// `fail-fast` as the judge sees it; `v` is the value's text.
    fn into_fail_fast(self, v: &str) -> FailFast {
        match self {
            Self::Bool(b) => FailFast {
                is_false: !b,
                repr: if b { "True" } else { "False" }.to_owned(),
            },
            Self::Int(repr) => FailFast {
                is_false: false,
                repr,
            },
            Self::Str(s) => FailFast {
                is_false: false,
                repr: py_repr(&s),
            },
            // fail-fast never stays untyped: `type_value` raised.
            Self::Raw => FailFast {
                is_false: false,
                repr: py_repr(v),
            },
        }
    }
}

/// The purpose-built reader's state: the tables seen, the keys under `[profile.ci]`, the
/// judged value, and the current table (`None` before the first header).
#[derive(Default)]
struct Minimal {
    tables: HashSet<String>,
    ci_keys: HashSet<String>,
    fail_fast: Option<FailFast>,
    cur: Option<String>,
}

/// `load_minimal`: for ONE question -- under the table `[profile.ci]`, what is the value of
/// `fail-fast`? Table headers set the current table; every other line must be a single-line
/// `key = value` that is balanced and opens no multi-line construct, and is only SKIPPED
/// unless it is the judged key. Anything else raises, and the caller reports ENV.
fn load_minimal(bytes: &[u8]) -> Result<Profile, PyError> {
    // open(path, "r", encoding="utf-8"): universal newlines.
    let text = decode(bytes)?.replace("\r\n", "\n").replace('\r', "\n");
    let mut st = Minimal::default();
    for (i, raw) in text.split_inclusive('\n').enumerate() {
        st.line(i + 1, py_strip(raw))?;
    }
    Ok(st.tables.contains("profile.ci").then_some(st.fail_fast))
}

impl Minimal {
    /// Line `n`, stripped.
    fn line(&mut self, n: usize, line: &str) -> Result<(), PyError> {
        if line.is_empty() || line.starts_with('#') {
            return Ok(());
        }
        if let Some(m) = HEADER.captures(line) {
            return self.header(n, &m);
        }
        let Some(m) = KEY.captures(line) else {
            return Err(PyError::value(format!(
                "line {n}: not a table header or a `key = value` line"
            )));
        };
        let key = m[1].trim_matches(|c: char| c == '"' || c == '\'');
        let val = m.get(2).map_or("", |g| g.as_str());
        let bare = single_line_value(n, key, val)?;
        match self.cur.as_deref() {
            None => top_level_key(n, key),
            Some("profile.ci") => self.ci_key(n, key, val, &bare),
            Some(_) => Ok(()),
        }
    }

    /// A table header. `[[name]]` is an array of tables, which may repeat.
    fn header(&mut self, n: usize, m: &Captures<'_>) -> Result<(), PyError> {
        let (open, close) = (m.get(1).is_some(), m.get(3).is_some());
        let aot = open || close;
        if open != close {
            return Err(PyError::value(format!("line {n}: unbalanced table header")));
        }
        let name = m[2].replace(['"', '\''], "");
        if aot && name == "profile.ci" {
            return Err(PyError::value(format!(
                "line {n}: [[profile.ci]] is an array of tables, not the profile"
            )));
        }
        let table = if aot {
            format!("{name}[]")
        } else {
            name.clone()
        };
        if !aot && self.tables.contains(&table) {
            return Err(PyError::value(format!(
                "line {n}: duplicate table [{name}]"
            )));
        }
        self.tables.insert(table.clone());
        self.cur = Some(table);
        Ok(())
    }

    /// A `key = value` line under `[profile.ci]`; `bare` is the value with its strings cut.
    fn ci_key(&mut self, n: usize, key: &str, val: &str, bare: &str) -> Result<(), PyError> {
        let v = if val.trim_start_matches(is_py_space).starts_with(['"', '\'']) {
            py_strip(val)
        } else {
            py_strip(bare.split_once('#').map_or(bare, |(head, _)| head))
        };
        let typed = type_value(n, key, v)?;
        if !self.ci_keys.insert(key.to_owned()) {
            return Err(PyError::value(format!(
                "line {n}: duplicate key {}",
                py_repr(key)
            )));
        }
        if key == "fail-fast" {
            self.fail_fast = Some(typed.into_fail_fast(v));
        }
        Ok(())
    }
}

/// What every `key = value` line must pass, wherever it sits: a key that is not dotted,
/// and a value that ends on its line. Returns the value with its strings cut out.
fn single_line_value<'a>(n: usize, key: &str, val: &'a str) -> Result<Cow<'a, str>, PyError> {
    if key.contains('.') {
        return Err(PyError::value(format!(
            "line {n}: dotted key {} is outside this reader's scope",
            py_repr(key)
        )));
    }
    if val.starts_with("\"\"\"") || val.starts_with("'''") {
        return Err(PyError::value(format!(
            "line {n}: multi-line string is outside this reader's scope"
        )));
    }
    let bare = QUOTED.replace_all(val, "");
    let count = |c: char| bare.matches(c).count();
    if count('[') != count(']') || count('{') != count('}') {
        return Err(PyError::value(format!(
            "line {n}: value continues past the line -- outside this reader's scope"
        )));
    }
    if count('"') % 2 == 1 || count('\'') % 2 == 1 {
        return Err(PyError::value(format!("line {n}: unterminated string")));
    }
    Ok(bare)
}

/// A key before the first header. Only nextest's own top-level keys are skipped: neither
/// can reach [profile.ci]. Any other top-level key raises: `profile` can reach it as an
/// inline table, and skipping an unknown key would be a guess.
fn top_level_key(n: usize, key: &str) -> Result<(), PyError> {
    if key != "experimental" && key != "nextest-version" {
        return Err(PyError::value(format!(
            "line {n}: top-level key {} is outside this reader's scope",
            py_repr(key)
        )));
    }
    Ok(())
}

/// A value under `[profile.ci]`, typed: a boolean, an integer, a one-line string, or, for
/// any key but `fail-fast`, left untyped.
fn type_value(n: usize, key: &str, v: &str) -> Result<Typed, PyError> {
    let typed = if v == "true" {
        Typed::Bool(true)
    } else if v == "false" {
        Typed::Bool(false)
    } else if INT.is_match(v) {
        Typed::Int(py_int_repr(v)?)
    } else if v.len() >= 2 && v.starts_with(['"', '\'']) && v.ends_with(&v[..1]) {
        Typed::Str(v[1..v.len() - 1].to_owned())
    } else if key == "fail-fast" {
        return Err(PyError::value(format!(
            "line {n}: fail-fast = {} is not a value this reader can type",
            py_repr(v)
        )));
    } else {
        Typed::Raw
    };
    Ok(typed)
}

/// `sys.get_int_max_str_digits()`: the most digits `int()` converts from a string, on
/// CPython 3.11+ and the 3.10.7, 3.9.14 and 3.8.14 security releases.
const PY_INT_MAX_STR_DIGITS: usize = 4300;

/// `repr(int(s))` for an `s` matching `^-?\d+$`: any Unicode decimal digit counts, leading
/// zeros drop, and `-0` is `0`. Past 4300 digits, leading zeros included, `int()` raises.
fn py_int_repr(s: &str) -> Result<String, PyError> {
    let (neg, digits) = s.strip_prefix('-').map_or((false, s), |d| (true, d));
    let count = digits.chars().count();
    if count > PY_INT_MAX_STR_DIGITS {
        return Err(PyError::value(format!(
            "Exceeds the limit ({PY_INT_MAX_STR_DIGITS} digits) for integer string conversion: \
             value has {count} digits; use sys.set_int_max_str_digits() to increase the limit"
        )));
    }
    let value: String = digits.chars().map(digit_value).collect();
    let value = value.trim_start_matches('0');
    Ok(match (value.is_empty(), neg) {
        (true, _) => "0".to_owned(),
        (false, true) => format!("-{value}"),
        (false, false) => value.to_owned(),
    })
}

/// The ASCII digit for a Unicode decimal digit. Unicode encodes every decimal digit set as
/// ten contiguous code points, 0 to 9, so a digit's value is its distance from the start of
/// its run, modulo ten.
fn digit_value(c: char) -> char {
    if c.is_ascii_digit() {
        return c;
    }
    let is_digit =
        |u: u32| char::from_u32(u).is_some_and(|d| DIGIT.is_match(d.encode_utf8(&mut [0; 4])));
    let mut start = c as u32;
    while start > 0 && is_digit(start - 1) {
        start -= 1;
    }
    let offset = u8::try_from((c as u32 - start) % 10).unwrap_or(0);
    char::from(b'0' + offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (want rc, label, body): the guard's case table, run under BOTH readers there.
    const TABLE: &[(u8, &str, &str)] = &[
        (0, "false under [profile.ci]", "[profile.ci]\nretries = 2\nfail-fast = false\n"),
        (1, "true under [profile.ci]", "[profile.ci]\nretries = 2\nfail-fast = true\n"),
        (1, "key absent", "[profile.ci]\nretries = 2\n"),
        (
            1,
            "false under a different profile only",
            "[profile.default]\nfail-fast = false\n[profile.ci]\nretries = 2\n",
        ),
        (1, "no [profile.ci]", "[profile.default]\nfail-fast = false\n"),
        (
            1,
            "false in a comment, true in the key",
            "[profile.ci]\n# fail-fast = false\nfail-fast = true\n",
        ),
        (1, "the string \"false\"", "[profile.ci]\nfail-fast = \"false\"\n"),
        (
            1,
            "false under [[profile.ci.overrides]] only",
            "[profile.ci]\nretries = 2\n[[profile.ci.overrides]]\nfilter = \"test(/x/)\"\nfail-fast = false\n",
        ),
        (
            0,
            "the real file's shape",
            "[profile.ci]\nretries = 2\nfail-fast = false\nslow-timeout = { period = \"60s\", terminate-after = 20 }\nstatus-level = \"slow\"\n[profile.ci.junit]\npath = \"junit.xml\"\n",
        ),
        (2, "[[profile.ci]]", "[[profile.ci]]\nfail-fast = false\n"),
        (2, "unparseable TOML", "[profile.ci\nfail-fast = false\n"),
        (
            0,
            "a top-level nextest key",
            "experimental = [\"setup-scripts\"]\n[profile.ci]\nretries = 2\nfail-fast = false\n",
        ),
        (
            2,
            "an inline top-level profile",
            "profile = { ci = { \"fail-fast\" = true } }\n[profile.ci]\nfail-fast = false\n",
        ),
        (
            2,
            "the inline profile, key quoted",
            "\"profile\" = { ci = { \"fail-fast\" = true } }\n[profile.ci]\nfail-fast = false\n",
        ),
        (
            2,
            "a dotted top-level profile.ci.fail-fast",
            "profile.ci.fail-fast = true\n[profile.ci]\nfail-fast = false\n",
        ),
    ];

    /// What the purpose-built reader must refuse rather than guess.
    const FALLBACK_REFUSES: &[(&str, &str)] = &[
        (
            "a multi-line string hiding a [profile.ci] header",
            "[profile.default]\nnote = \"\"\"title\"\n[profile.ci]\nfail-fast = false\nx = \"\"\"\"\n",
        ),
        (
            "a value continuing past its line",
            "[profile.ci]\nfail-fast = false\nslow-timeout = { period = \"60s\",\n  terminate-after = 20 }\n",
        ),
        ("fail-fast = fals", "[profile.ci]\nfail-fast = fals\n"),
        (
            "an unknown top-level key",
            "experimentl = [\"setup-scripts\"]\n[profile.ci]\nfail-fast = false\n",
        ),
    ];

    fn judge(body: &str, reader: Reader) -> (u8, String) {
        verdict("c.toml", load(body.as_bytes(), reader), reader.name(), "r")
    }

    #[test]
    fn case_table_under_both_readers() {
        for reader in [Reader::Library, Reader::Fallback] {
            for (want, label, body) in TABLE {
                let (rc, out) = judge(body, reader);
                assert_eq!(rc, *want, "{reader:?} {label}: {out}");
            }
        }
    }

    #[test]
    fn fallback_refuses_rather_than_guess() {
        for (label, body) in FALLBACK_REFUSES {
            let (rc, out) = judge(body, Reader::Fallback);
            assert_eq!(rc, 2, "{label}: {out}");
            assert!(out.starts_with("VERDICT ENV c.toml: "), "{label}: {out}");
        }
    }

    #[test]
    fn the_real_config_is_green_under_both_readers() {
        let conf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.config/nextest.toml");
        for reader in [Reader::Library, Reader::Fallback] {
            let (rc, out) = run(&conf, reader, "r");
            assert_eq!(rc, 0, "{reader:?}: {out}");
        }
    }

    #[test]
    fn verdict_lines_are_the_originals() {
        let ok = judge("[profile.ci]\nfail-fast = false\n", Reader::Fallback);
        assert_eq!(
            ok.1,
            "VERDICT ok c.toml: [profile.ci].fail-fast = false -- a red run still measures \
             everything else (reader=purpose-built reader, runner=r)\n"
        );
        let s = judge("[profile.ci]\nfail-fast = \"false\"\n", Reader::Library);
        assert_eq!(
            s.1,
            "VERDICT FAIL c.toml: [profile.ci].fail-fast = 'false' -- a failing test cancels \
             every test still queued and their verdicts are discarded (reader=toml crate)\n"
        );
        let none = judge("[profile.default]\n", Reader::Library);
        assert_eq!(
            none.1,
            "VERDICT FAIL c.toml: no [profile.ci] -- CI runs --profile ci, so the run would \
             take nextest's default, which is fail-fast ON (reader=toml crate)\n"
        );
        let unset = judge("[profile.ci]\n", Reader::Fallback);
        assert_eq!(
            unset.1,
            "VERDICT FAIL c.toml: [profile.ci] does not set fail-fast -- nextest's default is \
             ON (measured: 3/6 tests run on a 6-test probe) (reader=purpose-built reader)\n"
        );
    }

    #[test]
    fn fallback_errors_are_the_originals() {
        let cases: &[(&str, &str)] = &[
            ("[[profile.ci]\n", "line 1: unbalanced table header"),
            (
                "[[profile.ci]]\n",
                "line 1: [[profile.ci]] is an array of tables, not the profile",
            ),
            ("[a]\n[a]\n", "line 2: duplicate table [a]"),
            (
                "[a]\n= 1\n",
                "line 2: not a table header or a `key = value` line",
            ),
            (
                "[a]\n\"a.b\" = 1\n",
                "line 2: dotted key 'a.b' is outside this reader's scope",
            ),
            (
                "[a]\nx = '''y'''\n",
                "line 2: multi-line string is outside this reader's scope",
            ),
            (
                "[a]\nx = [1\n",
                "line 2: value continues past the line -- outside this reader's scope",
            ),
            ("[a]\nx = \"y\n", "line 2: unterminated string"),
            (
                "profile = 1\n",
                "line 1: top-level key 'profile' is outside this reader's scope",
            ),
            (
                "[profile.ci]\nfail-fast = 1.5\n",
                "line 2: fail-fast = '1.5' is not a value this reader can type",
            ),
            ("[profile.ci]\nx = 1\nx = 2\n", "line 3: duplicate key 'x'"),
        ];
        for (body, want) in cases {
            let got = load(body.as_bytes(), Reader::Fallback).expect_err(body);
            assert_eq!(
                (got.class, got.msg.as_str()),
                ("ValueError", *want),
                "{body:?}"
            );
        }
    }

    #[test]
    fn fail_fast_reprs_are_pythons() {
        let lib: &[(&str, &str)] = &[
            ("true", "True"),
            ("0", "0"),
            ("-0", "0"),
            ("1.5", "1.5"),
            ("1e20", "1e+20"),
            ("1e-5", "1e-05"),
            ("1e15", "1000000000000000.0"),
            ("0.0001", "0.0001"),
            ("-0.0", "-0.0"),
            ("inf", "inf"),
            ("nan", "nan"),
            ("\"it's\"", "\"it's\""),
            ("[true, \"a\"]", "[True, 'a']"),
            ("{ a = 1 }", "{'a': 1}"),
        ];
        for (val, want) in lib {
            let body = format!("[profile.ci]\nfail-fast = {val}\n");
            let p = load(body.as_bytes(), Reader::Library).expect(val);
            assert_eq!(p.flatten().map(|v| v.repr).as_deref(), Some(*want), "{val}");
        }
        let min: &[(&str, &str)] = &[
            ("true", "True"),
            ("007", "7"),
            ("-0", "0"),
            ("\u{0663}", "3"),
            ("\"x\"", "'x'"),
            ("'y'", "'y'"),
        ];
        for (val, want) in min {
            let body = format!("[profile.ci]\nfail-fast = {val}\n");
            let p = load(body.as_bytes(), Reader::Fallback).expect(val);
            assert_eq!(p.flatten().map(|v| v.repr).as_deref(), Some(*want), "{val}");
        }
    }

    #[test]
    fn library_errors_are_the_originals() {
        let list = load(b"[[profile.ci]]\n", Reader::Library).expect_err("list");
        assert_eq!(
            (list.class, list.msg.as_str()),
            ("ValueError", "profile.ci is a list, not a table")
        );
        let int = load(b"profile = 1\n", Reader::Library).expect_err("int");
        assert_eq!(
            (int.class, int.msg.as_str()),
            ("AttributeError", "'int' object has no attribute 'get'")
        );
        let bom = load(
            b"\xef\xbb\xbf[profile.ci]\nfail-fast = false\n",
            Reader::Library,
        )
        .expect_err("bom");
        assert_eq!(bom.class, "TOMLDecodeError");
        let bad = load(b"[profile.ci\n", Reader::Library).expect_err("bad");
        assert_eq!(bad.class, "TOMLDecodeError");
        assert!(!bad.msg.contains('\n'), "one line: {}", bad.msg);
    }

    /// The README's rows where tomllib and the `toml` crate disagree, pinned on the crate's
    /// side, so an upgrade that moves one is seen here and not on a runner.
    #[test]
    fn library_reader_follows_the_toml_crate() {
        let ok = "[profile.ci]\nfail-fast = false\n";
        let nest = |n: usize| format!("x = {}1{}", "[".repeat(n), "]".repeat(n));
        let reads = [
            "t = 1979-05-27T07:32:60Z".to_owned(),
            "t = 07:32:60".to_owned(),
            "t = 0000-01-01".to_owned(),
            nest(79),
        ];
        for line in &reads {
            let (rc, out) = judge(&format!("{ok}{line}\n"), Reader::Library);
            assert_eq!(rc, 0, "{line}: {out}");
        }
        let refuses = [
            "x = 9223372036854775808".to_owned(),
            "x = -9223372036854775809".to_owned(),
            "x = 0xffffffffffffffff".to_owned(),
            "x = 1e400".to_owned(),
            nest(80),
        ];
        for line in &refuses {
            let (rc, out) = judge(&format!("{ok}{line}\n"), Reader::Library);
            assert_eq!(rc, 2, "{line}: {out}");
            assert!(out.contains("(TOMLDecodeError: "), "{line}: {out}");
        }
    }

    #[test]
    fn fallback_has_cpythons_int_digit_limit() {
        let d4300 = "1".repeat(4300);
        for val in [
            d4300.clone(),
            format!("-{d4300}"),
            format!("0{}", &d4300[1..]),
        ] {
            let body = format!("[profile.ci]\nfail-fast = false\nx = {val}\n");
            let (rc, out) = judge(&body, Reader::Fallback);
            assert_eq!(rc, 0, "{} digits: {out}", val.len());
        }
        for (val, n) in [(format!("1{d4300}"), 4301), (format!("-0{d4300}"), 4301)] {
            let body = format!("[profile.ci]\nfail-fast = {val}\n");
            let got = load(body.as_bytes(), Reader::Fallback).expect_err(&val[..8]);
            let want = format!(
                "Exceeds the limit (4300 digits) for integer string conversion: value has {n} \
                 digits; use sys.set_int_max_str_digits() to increase the limit"
            );
            assert_eq!((got.class, got.msg.as_str()), ("ValueError", want.as_str()));
        }
    }

    #[test]
    fn os_and_decode_errors_are_cpythons() {
        let (rc, out) = run(Path::new("/nonexistent/c.toml"), Reader::Library, "r");
        assert_eq!(rc, 2);
        assert!(
            out.contains(
                "(FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent/c.toml')"
            ),
            "{out}"
        );
        let cases: &[(&[u8], &str)] = &[
            (
                b"ab\xff",
                "'utf-8' codec can't decode byte 0xff in position 2: invalid start byte",
            ),
            (
                b"ab\xe2\x82x",
                "'utf-8' codec can't decode bytes in position 2-3: invalid continuation byte",
            ),
            (
                b"ab\xe2\x82",
                "'utf-8' codec can't decode bytes in position 2-3: unexpected end of data",
            ),
            (
                b"\xe0\x80",
                "'utf-8' codec can't decode byte 0xe0 in position 0: invalid continuation byte",
            ),
        ];
        for (bytes, want) in cases {
            for reader in [Reader::Library, Reader::Fallback] {
                let e = load(bytes, reader).expect_err(want);
                assert_eq!((e.class, e.msg.as_str()), ("UnicodeDecodeError", *want));
            }
        }
    }

    #[test]
    fn universal_newlines_and_python_whitespace() {
        // CRLF and a lone CR end a line in text mode; \x1c is whitespace to Python's \s.
        for body in [
            "[profile.ci]\r\nfail-fast = false\r\n",
            "[profile.ci]\rfail-fast = false\r",
            "[\x1cprofile.ci]\nfail-fast\x1c= false\n",
        ] {
            assert_eq!(judge(body, Reader::Fallback).0, 0, "{body:?}");
        }
    }
}
