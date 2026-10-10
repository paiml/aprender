//! `scripts/lib/llama_fit_verdict.py` ported (#4799): llama.cpp's fit verdict for one
//! (model, host) cell of the model ladder.
//!
//! A cell may be measured only if the pinned `llama-fit-params` says the model fits FULLY on
//! this host's GPU at a ctx no lower than min(4096, the model's trained context). The record
//! is printed as one JSON line, byte for byte as the original's `json.dumps` printed it.
//!
//! Python behaviour the original depends on, written out here:
//! - `int()`: surrounding ASCII or non-ASCII Unicode whitespace (but not `\x1c`–`\x1f`), one
//!   sign, single `_` between digits, any Unicode decimal digit, at most 4300 digits.
//! - `\s` in a `str` pattern also matches `\x1c`–`\x1f`; Rust's does not.
//! - Every failure inside the GGUF header walk, including Python's recursion limit on nested
//!   arrays, is "no trained context".
//!
//! Which characters are digits, and how deep the walk recurses, depend on the interpreter.
//! This port follows CPython 3.12 and 3.13, which agree on both and are what the ladder's
//! hosts run; under another interpreter the original differs from both on those edges.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::sync::LazyLock;

use regex::Regex;

use crate::pystr::{is_py_space, py_read_text, py_splitlines, py_strip};

const MIN_CTX: i128 = 4096;

/// CPython's `sys.get_int_max_str_digits()` default.
const INT_MAX_STR_DIGITS: usize = 4300;

/// The deepest array nesting the original's recursive `skip` walks before CPython raises
/// `RecursionError`, measured on the original run as its caller runs it: python3.12 and 3.13
/// alike parse 996 nested arrays and give up on 997.
const MAX_ARRAY_DEPTH: usize = 996;

/// Characters `str.isdecimal()` accepts (CPython 3.12 and 3.13: Unicode 15.0/15.1, which agree
/// here; the interpreters the model ladder's hosts run), as inclusive ranges
/// that each start at a zero, so a digit's value is its distance from the start, mod 10.
/// Rust's `\d` follows whatever Unicode the regex crate ships, not the interpreter's.
const DECIMAL: [(u32, u32); 64] = [
    (0x30, 0x39),
    (0x660, 0x669),
    (0x6F0, 0x6F9),
    (0x7C0, 0x7C9),
    (0x966, 0x96F),
    (0x9E6, 0x9EF),
    (0xA66, 0xA6F),
    (0xAE6, 0xAEF),
    (0xB66, 0xB6F),
    (0xBE6, 0xBEF),
    (0xC66, 0xC6F),
    (0xCE6, 0xCEF),
    (0xD66, 0xD6F),
    (0xDE6, 0xDEF),
    (0xE50, 0xE59),
    (0xED0, 0xED9),
    (0xF20, 0xF29),
    (0x1040, 0x1049),
    (0x1090, 0x1099),
    (0x17E0, 0x17E9),
    (0x1810, 0x1819),
    (0x1946, 0x194F),
    (0x19D0, 0x19D9),
    (0x1A80, 0x1A89),
    (0x1A90, 0x1A99),
    (0x1B50, 0x1B59),
    (0x1BB0, 0x1BB9),
    (0x1C40, 0x1C49),
    (0x1C50, 0x1C59),
    (0xA620, 0xA629),
    (0xA8D0, 0xA8D9),
    (0xA900, 0xA909),
    (0xA9D0, 0xA9D9),
    (0xA9F0, 0xA9F9),
    (0xAA50, 0xAA59),
    (0xABF0, 0xABF9),
    (0xFF10, 0xFF19),
    (0x104A0, 0x104A9),
    (0x10D30, 0x10D39),
    (0x11066, 0x1106F),
    (0x110F0, 0x110F9),
    (0x11136, 0x1113F),
    (0x111D0, 0x111D9),
    (0x112F0, 0x112F9),
    (0x11450, 0x11459),
    (0x114D0, 0x114D9),
    (0x11650, 0x11659),
    (0x116C0, 0x116C9),
    (0x11730, 0x11739),
    (0x118E0, 0x118E9),
    (0x11950, 0x11959),
    (0x11C50, 0x11C59),
    (0x11D50, 0x11D59),
    (0x11DA0, 0x11DA9),
    (0x11F50, 0x11F59),
    (0x16A60, 0x16A69),
    (0x16AC0, 0x16AC9),
    (0x16B50, 0x16B59),
    (0x1D7CE, 0x1D7FF),
    (0x1E140, 0x1E149),
    (0x1E2F0, 0x1E2F9),
    (0x1E4F0, 0x1E4F9),
    (0x1E950, 0x1E959),
    (0x1FBF0, 0x1FBF9),
];

/// Characters `str.isdigit()` accepts that `str.isdecimal()` does not (superscripts, circled
/// digits, ...): `free_mib` passes the original's `isdigit()` test with them, then `int()`
/// raises.
const DIGIT_ONLY: [(u32, u32); 20] = [
    (0xB2, 0xB3),
    (0xB9, 0xB9),
    (0x1369, 0x1371),
    (0x19DA, 0x19DA),
    (0x2070, 0x2070),
    (0x2074, 0x2079),
    (0x2080, 0x2089),
    (0x2460, 0x2468),
    (0x2474, 0x247C),
    (0x2488, 0x2490),
    (0x24EA, 0x24EA),
    (0x24F5, 0x24FD),
    (0x24FF, 0x24FF),
    (0x2776, 0x277E),
    (0x2780, 0x2788),
    (0x278A, 0x2792),
    (0x10A40, 0x10A43),
    (0x10E60, 0x10E68),
    (0x11052, 0x1105A),
    (0x1F100, 0x1F10A),
];

fn in_ranges(table: &[(u32, u32)], c: char) -> Option<u32> {
    let c = c as u32;
    table
        .iter()
        .find(|&&(lo, hi)| (lo..=hi).contains(&c))
        .map(|&(lo, _)| c - lo)
}

/// `str.isdecimal()` for one char: the digits `int()` and the original's `\d` accept.
fn is_decimal(c: char) -> bool {
    in_ranges(&DECIMAL, c).is_some()
}

/// The value of a decimal digit, as an ASCII digit.
fn nd_value(c: char) -> char {
    let v = in_ranges(&DECIMAL, c).expect("a decimal digit") % 10;
    char::from(b'0' + u8::try_from(v).expect("below ten"))
}

/// The original's `\d` and `\s` as Rust regex classes, from the tables above.
fn class(table: &[(u32, u32)]) -> String {
    let ranges: String = table
        .iter()
        .map(|&(lo, hi)| format!("\\x{{{lo:X}}}-\\x{{{hi:X}}}"))
        .collect();
    format!("[{ranges}]")
}

const PY_SPACE: &str = r"[\s\x1c-\x1f]";

static BUILT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(([0-9a-f]{7,40})\)|commit ([0-9a-f]{7,40})").expect("static regex")
});
static CTX: LazyLock<Regex> = LazyLock::new(|| {
    let d = class(&DECIMAL);
    let re = format!("(?:^|{PY_SPACE})-c ({d}+)(?:{PY_SPACE}|$)");
    Regex::new(&re).expect("static regex")
});
static NGL: LazyLock<Regex> = LazyLock::new(|| {
    let d = class(&DECIMAL);
    let re = format!("(?:^|{PY_SPACE})-ngl (-?{d}+)(?:{PY_SPACE}|$)");
    Regex::new(&re).expect("static regex")
});

/// A Python `int`, kept as its canonical decimal form so it prints exactly as `json.dumps`
/// prints an unbounded integer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyInt {
    neg: bool,
    digits: String,
}

impl PyInt {
    /// `int(<run of decimal digits>)`, or `None` past CPython's digit limit.
    fn from_digits(neg: bool, chars: impl Iterator<Item = char>) -> Option<Self> {
        let raw: String = chars.map(nd_value).collect();
        if raw.len() > INT_MAX_STR_DIGITS {
            return None;
        }
        let digits = match raw.trim_start_matches('0') {
            "" => "0".to_owned(),
            d => d.to_owned(),
        };
        Some(Self {
            neg: neg && digits != "0",
            digits,
        })
    }

    /// `int(s)` with base 10, or `None` where CPython raises `ValueError`.
    pub fn parse(s: &str) -> Option<Self> {
        let ws = |c: char| {
            if c.is_ascii() {
                " \t\n\x0b\x0c\r".contains(c)
            } else {
                is_py_space(c)
            }
        };
        let s = s.trim_matches(ws);
        let (neg, body) = match s.strip_prefix('-') {
            Some(b) => (true, b),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        let groups_ok = body
            .split('_')
            .all(|g| !g.is_empty() && g.chars().all(is_decimal));
        if !groups_ok {
            return None;
        }
        Self::from_digits(neg, body.chars().filter(|&c| c != '_'))
    }

    fn is_zero(&self) -> bool {
        self.digits == "0"
    }

    fn is_minus_one(&self) -> bool {
        self.neg && self.digits == "1"
    }

    /// A fitted ctx (never negative: it matched `\d+`) below `floor`. Past `i128` it is not.
    fn below(&self, floor: i128) -> bool {
        self.digits.parse::<i128>().is_ok_and(|v| v < floor)
    }
}

impl std::fmt::Display for PyInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.neg {
            f.write_str("-")?;
        }
        f.write_str(&self.digits)
    }
}

/// `free_mib`: an int when `free_mib.lstrip("-").isdigit()`, else `None`. `Err` where the
/// original's `int(free_mib)` then raises (two or more leading `-`, or too many digits).
fn parse_free(s: &str) -> Result<Option<PyInt>, String> {
    let body = s.trim_start_matches('-');
    let digit_only = |c| in_ranges(&DIGIT_ONLY, c).is_some();
    if body.is_empty() || !body.chars().all(|c| is_decimal(c) || digit_only(c)) {
        return Ok(None);
    }
    if s.len() - body.len() > 1 || body.chars().any(digit_only) {
        return Err(format!("invalid literal for int() with base 10: {s:?}"));
    }
    PyInt::from_digits(s.len() != body.len(), body.chars())
        .map(Some)
        .ok_or_else(|| format!("free_mib {s:?} exceeds {INT_MAX_STR_DIGITS} digits"))
}

/// One fit record, in the original's key order.
#[derive(Debug)]
pub struct Record {
    pin: String,
    llama_cpp: Option<String>,
    free_mib: Option<PyInt>,
    ctx: Option<PyInt>,
    ngl: Option<PyInt>,
    trained_ctx: Option<i128>,
    ctx_floor: i128,
    rc: PyInt,
    raw: String,
    pub verdict: &'static str,
    pub reason: String,
}

/// `json.dumps(str)` with the default `ensure_ascii=True`.
fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

fn json_opt<T: std::fmt::Display>(v: Option<&T>) -> String {
    v.map_or_else(|| "null".to_owned(), ToString::to_string)
}

impl Record {
    /// The line the original prints.
    pub fn to_json(&self) -> String {
        let fields = [
            ("tool", json_str("llama-fit-params")),
            ("pin", json_str(&self.pin)),
            (
                "llama_cpp",
                self.llama_cpp
                    .as_deref()
                    .map_or("null".to_owned(), json_str),
            ),
            ("free_mib", json_opt(self.free_mib.as_ref())),
            ("ctx", json_opt(self.ctx.as_ref())),
            ("ngl", json_opt(self.ngl.as_ref())),
            ("trained_ctx", json_opt(self.trained_ctx.as_ref())),
            ("ctx_floor", self.ctx_floor.to_string()),
            ("rc", self.rc.to_string()),
            ("raw", json_str(&self.raw)),
            ("verdict", json_str(self.verdict)),
            ("reason", json_str(&self.reason)),
        ];
        let body: Vec<String> = fields
            .iter()
            .map(|(k, v)| format!("{}: {v}", json_str(k)))
            .collect();
        format!("{{{}}}", body.join(", "))
    }

    fn set(mut self, verdict: &'static str, reason: String) -> Self {
        self.verdict = verdict;
        self.reason = reason;
        self
    }
}

/// The commit `llama-fit-params --version` names, if any.
fn built_from(version_out: &str) -> Option<String> {
    let m = BUILT.captures(version_out)?;
    m.get(1).or_else(|| m.get(2)).map(|g| g.as_str().to_owned())
}

/// The original's prefix test: the shorter of the two, and at least 7 characters, agrees.
fn pinned(built: Option<&str>, pin: &str) -> bool {
    let Some(built) = built else { return false };
    let k = built.chars().count().min(pin.chars().count());
    k >= 7 && built.chars().take(k).eq(pin.chars().take(k))
}

/// The cell's fit record. `verdict == "fits"` is the only value that admits it. `Err` where
/// the original raises (a fitted number past CPython's digit limit).
pub fn verdict(
    tool_found: bool,
    version_out: &str,
    pin: &str,
    rc: PyInt,
    stdout: &str,
    free_mib: Option<PyInt>,
    trained: Option<i128>,
) -> Result<Record, String> {
    let floor = match trained {
        Some(t) if t != 0 => t.min(MIN_CTX),
        _ => MIN_CTX,
    };
    let stripped = py_strip(stdout);
    let mut rec = Record {
        pin: pin.to_owned(),
        llama_cpp: None,
        free_mib,
        ctx: None,
        ngl: None,
        trained_ctx: trained,
        ctx_floor: floor,
        rc,
        raw: stripped.chars().take(300).collect(),
        verdict: "",
        reason: String::new(),
    };
    if !tool_found {
        let reason = "llama-fit-params is not installed on this host".to_owned();
        return Ok(rec.set("tool-absent", reason));
    }
    rec.llama_cpp = built_from(version_out);
    if !pinned(rec.llama_cpp.as_deref(), pin) {
        let built = rec.llama_cpp.as_deref().unwrap_or("an unknown commit");
        let reason = format!("llama-fit-params is built from {built}, not the pin {pin}");
        return Ok(rec.set("unpinned", reason));
    }
    if !rec.rc.is_zero() {
        let reason = format!(
            "llama-fit-params exit {}: the model does not fit this host's free memory",
            rec.rc
        );
        return Ok(rec.set("does-not-fit", reason));
    }
    let line = py_splitlines(stripped).last().copied().unwrap_or("");
    judge_line(rec, line)
}

/// The verdict on the last line `llama-fit-params` printed.
fn judge_line(mut rec: Record, line: &str) -> Result<Record, String> {
    let (Some(c), Some(n)) = (CTX.captures(line), NGL.captures(line)) else {
        let reason = "llama-fit-params printed no `-c N -ngl N` line: no verdict".to_owned();
        return Ok(rec.set("missing", reason));
    };
    let too_long = || format!("fitted `{line}` holds a number past {INT_MAX_STR_DIGITS} digits");
    let ctx = PyInt::from_digits(false, c[1].chars()).ok_or_else(too_long)?;
    let ngl_text = &n[1];
    let ngl_digits = ngl_text.strip_prefix('-');
    let ngl = PyInt::from_digits(ngl_digits.is_some(), ngl_digits.unwrap_or(ngl_text).chars())
        .ok_or_else(too_long)?;
    let padded = format!(" {line} ");
    let partial = !ngl.is_minus_one() || padded.contains(" -ot ") || padded.contains(" -ts ");
    let short = ctx.below(rec.ctx_floor);
    let ctx_text = ctx.to_string();
    rec.ctx = Some(ctx);
    rec.ngl = Some(ngl);
    if partial {
        let reason = format!("fitted `{line}` is not full GPU placement");
        return Ok(rec.set("partial-offload", reason));
    }
    if short {
        let trained = rec
            .trained_ctx
            .map_or_else(|| "None".to_owned(), |t| t.to_string());
        let reason = format!(
            "fitted ctx {ctx_text} < floor {} (min(4096, trained {trained}))",
            rec.ctx_floor
        );
        return Ok(rec.set("does-not-fit", reason));
    }
    Ok(rec.set("fits", format!("full GPU at ctx {ctx_text}")))
}

/// Byte size of a GGUF scalar value type.
fn scalar_size(t: u32) -> Option<u64> {
    match t {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

struct Gguf<R> {
    f: R,
}

impl<R: Read + Seek> Gguf<R> {
    fn bytes<const N: usize>(&mut self) -> Option<[u8; N]> {
        let mut b = [0; N];
        self.f.read_exact(&mut b).ok()?;
        Some(b)
    }

    fn u32(&mut self) -> Option<u32> {
        self.bytes().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.bytes().map(u64::from_le_bytes)
    }

    /// `f.seek(n, 1)`: past the end is allowed, past `i64::MAX` is not.
    fn seek(&mut self, n: u64) -> Option<()> {
        self.f
            .seek(SeekFrom::Current(i64::try_from(n).ok()?))
            .ok()
            .map(drop)
    }

    /// A length-prefixed string, as raw bytes. A short read ends the walk: the original read
    /// fewer bytes and then failed on the next field, at the end of the file.
    fn string(&mut self) -> Option<Vec<u8>> {
        let len = self.u64()?;
        let mut s = Vec::new();
        (&mut self.f).take(len).read_to_end(&mut s).ok()?;
        (s.len() as u64 == len).then_some(s)
    }

    fn skip(&mut self, t: u32, depth: usize) -> Option<()> {
        if depth > MAX_ARRAY_DEPTH {
            return None;
        }
        if let Some(size) = scalar_size(t) {
            return self.seek(size);
        }
        match t {
            8 => {
                let len = self.u64()?;
                self.seek(len)
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                if let Some(size) = scalar_size(et) {
                    return self.seek(size.checked_mul(n)?);
                }
                for _ in 0..n {
                    self.skip(et, depth + 1)?;
                }
                Some(())
            }
            _ => None,
        }
    }

    fn value(&mut self, t: u32) -> Option<i128> {
        match t {
            4 => self.bytes().map(|b| i128::from(u32::from_le_bytes(b))),
            5 => self.bytes().map(|b| i128::from(i32::from_le_bytes(b))),
            10 => self.bytes().map(|b| i128::from(u64::from_le_bytes(b))),
            _ => self.bytes().map(|b| i128::from(i64::from_le_bytes(b))),
        }
    }

    fn context_length(&mut self) -> Option<i128> {
        if &self.bytes::<4>()? != b"GGUF" || self.u32()? < 2 {
            return None;
        }
        let _n_tensors = self.u64()?;
        let n_kv = self.u64()?;
        for _ in 0..n_kv {
            let key = self.string()?;
            let t = self.u32()?;
            if key.ends_with(b".context_length") && matches!(t, 4 | 5 | 10 | 11) {
                return self.value(t);
            }
            self.skip(t, 0)?;
        }
        None
    }
}

/// `<arch>.context_length` from a GGUF header, or `None` when it cannot be read. Walks the
/// metadata KVs without loading tensors; arrays are skipped by length.
pub fn trained_ctx(path: &str) -> Option<i128> {
    let f = BufReader::new(File::open(path).ok()?);
    Gguf { f }.context_length()
}

/// The original's `read`: `""` and `-` name no file; otherwise the file in text mode.
fn read_text(p: &str) -> Result<String, String> {
    if p.is_empty() || p == "-" {
        return Ok(String::new());
    }
    std::fs::read(p)
        .map(|b| py_read_text(&b))
        .map_err(|e| format!("{p}: {e}"))
}

/// `tool_found(0/1) pin rc free_mib version_file stdout_file model_path` -> the JSON line.
/// `Err` (nothing printed, exit 1) wherever the original raised.
pub fn run(args: &[String]) -> Result<String, String> {
    let [tool_found, pin, rc, free_mib, vfile, sfile, model] = args else {
        return Err(format!(
            "llama-fit-verdict: expected 7 arguments (tool_found pin rc free_mib version_file \
             stdout_file model_path), got {}",
            args.len()
        ));
    };
    let free = parse_free(free_mib)?;
    let version_out = read_text(vfile)?;
    let rc = PyInt::parse(rc)
        .ok_or_else(|| format!("invalid literal for int() with base 10: {rc:?}"))?;
    let stdout = read_text(sfile)?;
    let rec = verdict(
        tool_found == "1",
        &version_out,
        pin,
        rc,
        &stdout,
        free,
        trained_ctx(model),
    )?;
    Ok(rec.to_json() + "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIN: &str = "a1b2c3d4e5f6";

    fn int(s: &str) -> PyInt {
        PyInt::parse(s).expect("an int")
    }

    fn judge(stdout: &str, trained: Option<i128>) -> Record {
        let version = format!("version: 1 ({PIN})");
        verdict(true, &version, PIN, int("0"), stdout, None, trained).expect("a record")
    }

    #[test]
    fn python_int_forms() {
        assert_eq!(int(" +0_07\u{a0}").to_string(), "7");
        assert_eq!(int("-0").to_string(), "0");
        assert_eq!(int("\u{664}\u{660}").to_string(), "40");
        assert_eq!(int("\u{1d7d9}").to_string(), "1"); // a second run of math digits
        for bad in ["", "-", "_1", "1_", "1__0", "+-1", "\x1c7", "1.0", "--1"] {
            assert_eq!(PyInt::parse(bad), None, "{bad:?}");
        }
        assert!(PyInt::parse(&"1".repeat(4300)).is_some());
        assert_eq!(PyInt::parse(&format!("0{}", "1".repeat(4300))), None);
        assert!(PyInt::parse(&format!("{}1", "1_".repeat(4299))).is_some());
        assert_eq!(int("\u{11F59}").to_string(), "9"); // Kawi, Unicode 15
        assert_eq!(PyInt::parse("\u{1CCF0}"), None); // outlined digit, Unicode 16
    }

    #[test]
    fn free_mib_rules() {
        assert_eq!(
            parse_free("-12").expect("test").expect("test").to_string(),
            "-12"
        );
        assert_eq!(parse_free("unknown").expect("test"), None);
        assert_eq!(parse_free("-").expect("test"), None);
        assert_eq!(parse_free("+5").expect("test"), None);
        assert!(parse_free("--5").is_err());
        assert!(parse_free("\u{b2}").is_err());
        assert_eq!(parse_free("1\u{2460}x").expect("test"), None);
    }

    #[test]
    fn json_escapes_like_python() {
        let want = r#""a\"\\\n\u007f\u00e9\ud83d\ude00\u0001""#;
        assert_eq!(json_str("a\"\\\n\u{7f}\u{e9}\u{1f600}\u{1}"), want);
    }

    #[test]
    fn every_verdict() {
        assert_eq!(judge("-c 8192 -ngl -1", None).verdict, "fits");
        assert_eq!(judge("-c 8192 -ngl 20", None).verdict, "partial-offload");
        assert_eq!(
            judge("-c 8192 -ngl -1 -ot \"x=CPU\"", None).verdict,
            "partial-offload"
        );
        assert_eq!(
            judge("-c 8192 -ngl -1 -ts 1,1", None).verdict,
            "partial-offload"
        );
        assert_eq!(judge("-c 2048 -ngl -1", None).verdict, "does-not-fit");
        assert_eq!(judge("-c 2048 -ngl -1", Some(2048)).verdict, "fits");
        assert_eq!(judge("-c 2047 -ngl -1", Some(2048)).verdict, "does-not-fit");
        assert_eq!(judge("-c 4096 -ngl -01", None).verdict, "fits");
        assert_eq!(judge("-c 4096 -ngl -1\x1f", None).verdict, "fits");
        assert_eq!(judge("-ngl -1", None).verdict, "missing");
        let no_tool = verdict(false, "", PIN, int("0"), "", None, None).expect("test");
        assert_eq!(no_tool.verdict, "tool-absent");
        let rc1 = verdict(
            true,
            &format!("commit {PIN}"),
            PIN,
            int("1"),
            "",
            None,
            None,
        );
        assert_eq!(rc1.expect("test").verdict, "does-not-fit");
        let other = verdict(true, "(0000000)", PIN, int("0"), "", None, None).expect("test");
        assert_eq!(
            (other.verdict, other.llama_cpp.as_deref()),
            ("unpinned", Some("0000000"))
        );
    }

    #[test]
    fn pin_prefix_needs_seven_agreeing_chars() {
        assert!(pinned(Some("a1b2c3d"), PIN));
        assert!(!pinned(Some("a1b2c3"), PIN));
        assert!(!pinned(Some("a1b2c3e"), PIN));
        assert!(!pinned(Some("a1b2c3d"), "a1b2c3"));
        assert!(!pinned(None, PIN));
    }

    fn gguf(kvs: &[u8], n_kv: u64) -> std::io::Cursor<Vec<u8>> {
        gguf_v(3, kvs, n_kv)
    }

    fn gguf_v(version: u32, kvs: &[u8], n_kv: u64) -> std::io::Cursor<Vec<u8>> {
        let mut b = b"GGUF".to_vec();
        b.extend(version.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(n_kv.to_le_bytes());
        b.extend(kvs);
        std::io::Cursor::new(b)
    }

    fn kv(key: &str, t: u32, value: &[u8]) -> Vec<u8> {
        let mut b = (key.len() as u64).to_le_bytes().to_vec();
        b.extend(key.as_bytes());
        b.extend(t.to_le_bytes());
        b.extend(value);
        b
    }

    #[test]
    fn gguf_header_walk() {
        let mut arr = 8u32.to_le_bytes().to_vec(); // array of 2 strings
        arr.extend(2u64.to_le_bytes());
        for s in ["ab", "c"] {
            arr.extend((s.len() as u64).to_le_bytes());
            arr.extend(s.as_bytes());
        }
        let mut kvs = kv("tokenizer.tokens", 9, &arr);
        kvs.extend(kv("general.name", 8, &[1, 0, 0, 0, 0, 0, 0, 0, b'x']));
        kvs.extend(kv("llama.context_length", 6, &[0; 4])); // f32: skipped
        kvs.extend(kv("llama.context_length", 5, &(-3i32).to_le_bytes()));
        let ctx = |kvs: &[u8], n| Gguf { f: gguf(kvs, n) }.context_length();
        assert_eq!(ctx(&kvs, 4), Some(-3));
        assert_eq!(ctx(&kvs, 3), None);
        assert_eq!(ctx(&kvs[..kvs.len() - 1], 4), None);
        assert_eq!(ctx(&kv("x", 13, &[]), 1), None);
    }

    // Goldens below are the original's own output for the same inputs.
    const GOLDEN_FITS: &str = r#"{"tool": "llama-fit-params", "pin": "a1b2c3d4e5f6", "llama_cpp": "a1b2c3d4e5f6", "free_mib": -12, "ctx": 8192, "ngl": -1, "trained_ctx": 2048, "ctx_floor": 2048, "rc": 0, "raw": "a\r\b\f\t\"\\\u00e9\ud83d\ude00\u0001\n-c 8192 -ngl -1", "verdict": "fits", "reason": "full GPU at ctx 8192"}"#;
    const GOLDEN_UNPINNED: &str = r#"{"tool": "llama-fit-params", "pin": "a1b2c3d4e5f6", "llama_cpp": "0000000", "free_mib": null, "ctx": null, "ngl": null, "trained_ctx": null, "ctx_floor": 4096, "rc": 0, "raw": "", "verdict": "unpinned", "reason": "llama-fit-params is built from 0000000, not the pin a1b2c3d4e5f6"}"#;
    const GOLDEN_TRAINED_0: &str = r#"{"tool": "llama-fit-params", "pin": "a1b2c3d4e5f6", "llama_cpp": "a1b2c3d4e5f6", "free_mib": null, "ctx": 2048, "ngl": -1, "trained_ctx": 0, "ctx_floor": 4096, "rc": 0, "raw": "-c 2048 -ngl -1", "verdict": "does-not-fit", "reason": "fitted ctx 2048 < floor 4096 (min(4096, trained 0))"}"#;
    const GOLDEN_RUN: &str = r#"{"tool": "llama-fit-params", "pin": "a1b2c3d4e5f6", "llama_cpp": "a1b2c3d", "free_mib": 24000, "ctx": 8192, "ngl": -1, "trained_ctx": 2048, "ctx_floor": 2048, "rc": 0, "raw": "x\n-c 8192 -ngl -1", "verdict": "fits", "reason": "full GPU at ctx 8192"}"#;
    const GOLDEN_ABSENT: &str = r#"{"tool": "llama-fit-params", "pin": "a1b2c3d4e5f6", "llama_cpp": null, "free_mib": 24000, "ctx": null, "ngl": null, "trained_ctx": null, "ctx_floor": 4096, "rc": 0, "raw": "", "verdict": "tool-absent", "reason": "llama-fit-params is not installed on this host"}"#;

    #[test]
    fn records_print_as_the_original_did() {
        let version = format!("version: 1 ({PIN})");
        let stdout = "a\r\u{8}\u{c}\t\"\\\u{e9}\u{1f600}\u{1}\n-c 8192 -ngl -1";
        let free = parse_free("-12").expect("test");
        let fits = verdict(true, &version, PIN, int("0"), stdout, free, Some(2048));
        assert_eq!(fits.expect("test").to_json(), GOLDEN_FITS);
        let other = verdict(true, "(0000000)", PIN, int("0"), "", None, None);
        assert_eq!(other.expect("test").to_json(), GOLDEN_UNPINNED);
        let zero = verdict(
            true,
            &version,
            PIN,
            int("0"),
            "-c 2048 -ngl -1",
            None,
            Some(0),
        );
        assert_eq!(zero.expect("test").to_json(), GOLDEN_TRAINED_0);
    }

    #[test]
    fn ngl_must_be_exactly_minus_one() {
        assert_eq!(judge("-c 8192 -ngl 1", None).verdict, "partial-offload");
        assert_eq!(judge("-c 8192 -ngl -2", None).verdict, "partial-offload");
        assert_eq!(judge("-c 8192 -ngl -0", None).verdict, "partial-offload");
    }

    #[test]
    fn free_mib_one_minus_is_negative() {
        assert_eq!(
            parse_free("-5").expect("test").expect("an int").to_string(),
            "-5"
        );
        assert_eq!(
            parse_free("5").expect("test").expect("an int").to_string(),
            "5"
        );
    }

    #[test]
    fn huge_ctx_is_not_below_the_floor() {
        let line = format!("-c {} -ngl -1", "9".repeat(60));
        assert_eq!(judge(&line, None).verdict, "fits");
        assert!(int("4095").below(4096));
        assert!(!int("4096").below(4096));
    }

    #[test]
    fn every_scalar_type_is_skipped_by_size() {
        let mut kvs = Vec::new();
        for (t, size) in [
            (0u32, 1usize),
            (1, 1),
            (7, 1),
            (2, 2),
            (3, 2),
            (4, 4),
            (5, 4),
            (6, 4),
        ] {
            kvs.extend(kv("s", t, &vec![0xAB; size]));
        }
        for t in [10u32, 11, 12] {
            kvs.extend(kv("s", t, &[0xAB; 8]));
        }
        kvs.extend(kv("x.context_length", 4, &4097u32.to_le_bytes()));
        assert_eq!(Gguf { f: gguf(&kvs, 12) }.context_length(), Some(4097));
    }

    #[test]
    fn unsigned_context_lengths_stay_unsigned() {
        let u32_max = kv("x.context_length", 4, &u32::MAX.to_le_bytes());
        assert_eq!(
            Gguf {
                f: gguf(&u32_max, 1)
            }
            .context_length(),
            Some(i128::from(u32::MAX))
        );
        let u64_max = kv("x.context_length", 10, &u64::MAX.to_le_bytes());
        assert_eq!(
            Gguf {
                f: gguf(&u64_max, 1)
            }
            .context_length(),
            Some(i128::from(u64::MAX))
        );
        let i64_neg = kv("x.context_length", 11, &(-7i64).to_le_bytes());
        assert_eq!(
            Gguf {
                f: gguf(&i64_neg, 1)
            }
            .context_length(),
            Some(-7)
        );
    }

    #[test]
    fn version_two_is_the_oldest_read() {
        let kvs = kv("x.context_length", 4, &9u32.to_le_bytes());
        assert_eq!(
            Gguf {
                f: gguf_v(1, &kvs, 1)
            }
            .context_length(),
            None
        );
        assert_eq!(
            Gguf {
                f: gguf_v(2, &kvs, 1)
            }
            .context_length(),
            Some(9)
        );
    }

    fn nested(depth: usize) -> Option<i128> {
        let mut v = Vec::new();
        for _ in 0..depth {
            v.extend(9u32.to_le_bytes());
            v.extend(1u64.to_le_bytes());
        }
        v.extend(4u32.to_le_bytes());
        v.extend(1u64.to_le_bytes());
        v.extend([0; 4]);
        let mut kvs = kv("deep", 9, &v);
        kvs.extend(kv("x.context_length", 4, &777u32.to_le_bytes()));
        Gguf { f: gguf(&kvs, 2) }.context_length()
    }

    #[test]
    fn nesting_stops_where_cpython_recursion_does() {
        assert_eq!(nested(1), Some(777));
        assert_eq!(nested(MAX_ARRAY_DEPTH), Some(777));
        assert_eq!(nested(MAX_ARRAY_DEPTH + 1), None);
    }

    fn scratch(name: &str, bytes: &[u8]) -> String {
        let dir = std::env::temp_dir().join(format!("lfv-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("test");
        let path = dir.join(name);
        std::fs::write(&path, bytes).expect("test");
        path.to_str().expect("test").to_owned()
    }

    #[test]
    fn run_reads_files_like_the_original() {
        let v = scratch("v", b"(a1b2c3d)\r\n");
        let s = scratch("s", b"x\r\n-c 8192 -ngl -1\r\n");
        let mut m = b"GGUF".to_vec();
        m.extend(3u32.to_le_bytes());
        m.extend(0u64.to_le_bytes());
        m.extend(1u64.to_le_bytes());
        m.extend(kv("llama.context_length", 4, &2048u32.to_le_bytes()));
        let m = scratch("m", &m);
        assert_eq!(trained_ctx(&m), Some(2048));
        assert_eq!(trained_ctx(&format!("{m}.missing")), None);
        let args = |found: &str, v: &str, s: &str, m: &str| {
            [found, PIN, "0", "24000", v, s, m].map(str::to_owned)
        };
        assert_eq!(run(&args("1", &v, &s, &m)), Ok(format!("{GOLDEN_RUN}\n")));
        assert_eq!(
            run(&args("0", "-", "", "-")),
            Ok(format!("{GOLDEN_ABSENT}\n"))
        );
        assert_eq!(read_text("").as_deref(), Ok(""));
        assert_eq!(read_text("-").as_deref(), Ok(""));
        assert_eq!(read_text(&v).as_deref(), Ok("(a1b2c3d)\n"));
        assert!(read_text(&format!("{v}.missing")).is_err());
        assert!(run(&args("1", &format!("{v}.missing"), &s, &m)).is_err());
        assert!(run(&args("1", &v, &s, &m)[..6]).is_err());
    }
}
