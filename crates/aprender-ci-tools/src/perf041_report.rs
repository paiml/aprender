//! PERF-041 report: decompose `serialization_index(c)` into its two factors (was
//! `scripts/perf041_report.py`).
//!
//! ```text
//! serialization_index(c) = wall(c) / wall_fast(1) = path_penalty * scaling_index(c)
//!   path_penalty  = latency_batched(1) / latency_fast(1)
//!   scaling_index = latency(c)         / latency_batched(1)
//! token_index(c)  = c * agg_tok_s(1) / agg_tok_s(c)
//! ```
//!
//! It decides nothing (PP-33, PP-12): every line is prefixed `REPORT`, and the verdict on
//! `serialization_index(c) < c` is `scripts/perf_gate.sh`'s, read from
//! `scripts/perf-matrix.yaml`.
//!
//! Input: the `*.json` records in OUT_DIR (default `/tmp/perf041`), one per band
//! replicate. Exit 0 with a report, 1 when there is no band or no `fast` c=1 band. Where
//! the original raised (a record that is not an object, a missing or non-numeric field, a
//! zero divisor, a file that is not UTF-8), the port stops with exit 1 after the lines the
//! original had already printed.

use crate::dag_status::{py_str, Py};
use crate::pystr::py_strip;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

/// A record: a JSON object, keys in input order (a repeated key keeps its last value).
type Rec = Vec<(String, Py)>;
/// `(mode, c)` -> the band's replicates in file order.
type Bands = BTreeMap<(String, i128), Vec<Rec>>;

/// The report: stdout, exit status, and the reason for a stop (empty unless one happened).
#[derive(Debug, PartialEq)]
pub struct Outcome {
    pub stdout: String,
    pub code: u8,
    pub stderr: String,
}

/// A number as Python holds it: `int` (a `bool` is one) or `float`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    fn of(v: &Py) -> Option<Num> {
        match v {
            Py::Bool(b) => Some(Num::Int(i128::from(*b))),
            Py::Int(i) => Some(Num::Int(*i)),
            Py::Float(f) => Some(Num::Float(*f)),
            _ => None,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn f(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
}

fn get<'a>(rec: &'a Rec, key: &str) -> Option<&'a Py> {
    rec.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn field(recs: &[Rec], key: &str) -> Result<Vec<Num>, String> {
    recs.iter()
        .map(|r| {
            let v =
                get(r, key).ok_or_else(|| format!("KeyError: {}", py_str(&Py::Str(key.into()))))?;
            Num::of(v).ok_or_else(|| format!("TypeError: {key} is {}", py_str(v)))
        })
        .collect()
}

/// `statistics.median(r[key] for r in recs)`: the middle of the sorted values, or the mean
/// of the two middle ones.
fn med(recs: &[Rec], key: &str) -> Result<f64, String> {
    let mut v = field(recs, key)?;
    // Stable, and -0.0 == 0.0 as Python compares them (no NaN parses).
    v.sort_by(|a, b| {
        a.f()
            .partial_cmp(&b.f())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = v.len();
    if n % 2 == 1 {
        return Ok(v[n / 2].f());
    }
    let (a, b) = (v[n / 2 - 1], v[n / 2]);
    #[allow(clippy::cast_precision_loss)]
    let sum = match (a, b) {
        (Num::Int(x), Num::Int(y)) => (x + y) as f64,
        _ => a.f() + b.f(),
    };
    Ok(sum / 2.0)
}

/// `min(...)` (`max` when `max`) of a field, which `:d` then requires to be an `int`. The
/// FIRST extreme wins, so `[2.0, 2]` yields the float and `[2, 2.0]` the int.
fn extreme(recs: &[Rec], key: &str, max: bool) -> Result<i128, String> {
    let v = field(recs, key)?;
    let mut best = v[0];
    for x in &v[1..] {
        if (max && x.f() > best.f()) || (!max && x.f() < best.f()) {
            best = *x;
        }
    }
    match best {
        Num::Int(i) => Ok(i),
        Num::Float(f) => Err(format!("ValueError: {key} {f} is not an int")),
    }
}

/// `format(x, f">{width}.{prec}f")`: Rust's fixed-point digits, but NaN prints `nan`.
fn fixed(x: f64, width: usize, prec: usize) -> String {
    if x.is_nan() {
        return format!("{:>width$}", "nan");
    }
    format!("{x:>width$.prec$}")
}

/// `a / b`, which raises `ZeroDivisionError` in Python where IEEE gives an infinity.
fn div(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        return Err("ZeroDivisionError: float division by zero".to_owned());
    }
    Ok(a / b)
}

/// `int(s)` for a `str`: surrounding whitespace, a sign, ASCII digits with single `_`
/// between them.
fn py_int_str(s: &str) -> Option<i128> {
    let t = py_strip(s);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let ok = !body.is_empty()
        && !body.starts_with('_')
        && !body.ends_with('_')
        && !body.contains("__")
        && body.bytes().all(|b| b.is_ascii_digit() || b == b'_');
    if !ok {
        return None;
    }
    let v: i128 = body.replace('_', "").parse().ok()?;
    Some(if neg { -v } else { v })
}

/// `int(rec["c"])`.
fn band_c(rec: &Rec) -> Result<i128, String> {
    let v = get(rec, "c").ok_or_else(|| "KeyError: 'c'".to_owned())?;
    // Python's int() of a float is exact at any size; the port stops beyond i128 rather
    // than let `as` saturate to a wrong band (README, divergences).
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let c = match v {
        Py::Str(s) => py_int_str(s),
        Py::Float(f) if f.abs() < i128::MAX as f64 => Some(f.trunc() as i128),
        other => match Num::of(other) {
            Some(Num::Int(i)) => Some(i),
            _ => None,
        },
    };
    c.ok_or_else(|| format!("ValueError: int({})", py_str(v)))
}

/// Python truthiness of a JSON value.
fn truthy(v: &Py) -> bool {
    match v {
        Py::None => false,
        Py::Bool(b) => *b,
        Py::Int(i) => *i != 0,
        Py::Float(f) => *f != 0.0,
        Py::Str(s) => !s.is_empty(),
        Py::List(a) => !a.is_empty(),
        Py::Dict(o) => !o.is_empty(),
    }
}

/// `label.split("-")[0] if label else "?"`, with `label = rec.get("label", "")`.
fn band_mode(rec: &Rec) -> Result<String, String> {
    match get(rec, "label") {
        Some(Py::Str(s)) if !s.is_empty() => Ok(s.split('-').next().unwrap_or_default().to_owned()),
        Some(v) if truthy(v) => Err(format!("AttributeError: label {} has no split", py_str(v))),
        _ => Ok("?".to_owned()),
    }
}

/// `sorted(glob.glob(os.path.join(out_dir, "*.json")))`, as file names: a hidden name never
/// matches `*`, and an unreadable or absent directory lists nothing.
fn json_names(out_dir: &str) -> Vec<String> {
    let dir = if out_dir.is_empty() { "." } else { out_dir };
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| !n.starts_with('.') && n.ends_with(".json"))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// One file: `Ok(None)` when it is not JSON (skipped silently) or carries `error` (skipped
/// with a line), `Ok(Some(rec))` for a record.
fn load_one(path: &Path, name: &str, out: &mut String) -> Result<Option<Rec>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
    let text = String::from_utf8(bytes).map_err(|e| format!("UnicodeDecodeError: {name}: {e}"))?;
    let Ok(rec) = serde_json::from_str::<Py>(&text) else {
        return Ok(None);
    };
    let Py::Dict(rec) = rec else {
        return Err(format!("TypeError: {name} is not an object"));
    };
    if let Some(e) = get(&rec, "error") {
        let _ = writeln!(out, "REPORT skip {name}: {}", py_str(e));
        return Ok(None);
    }
    Ok(Some(rec))
}

fn load(out_dir: &str, out: &mut String) -> Result<Bands, String> {
    let mut bands = Bands::new();
    for name in json_names(out_dir) {
        let path = Path::new(out_dir).join(&name);
        let Some(rec) = load_one(&path, &name, out)? else {
            continue;
        };
        let mode = band_mode(&rec)?;
        let c = band_c(&rec)?;
        bands.entry((mode, c)).or_default().push(rec);
    }
    Ok(bands)
}

fn print_bands(bands: &Bands, out: &mut String) -> Result<(), String> {
    let _ = writeln!(
        out,
        "\nREPORT === PERF-041 bands (median over replicates) ==="
    );
    let _ = writeln!(
        out,
        "REPORT {:<8}{:>3}{:>6}{:>12}{:>12}{:>10}{:>9}{:>9}",
        "mode", "c", "reps", "lat_p50_s", "agg_tok_s", "tok_p50", "tok_min", "tok_max"
    );
    for ((mode, c), recs) in bands {
        let row = format!(
            "REPORT {mode:<8}{c:>3}{:>6}{}{}{}{:>9}{:>9}",
            recs.len(),
            fixed(med(recs, "latency_p50_s")?, 12, 3),
            fixed(med(recs, "agg_tok_s")?, 12, 1),
            fixed(med(recs, "tokens_p50")?, 10, 0),
            extreme(recs, "tokens_min", false)?,
            extreme(recs, "tokens_max", true)?,
        );
        let _ = writeln!(out, "{row}");
    }
    Ok(())
}

/// `path_penalty = latency_batched(1) / latency_fast(1)`, or `None` if unmeasured.
fn print_penalty(bands: &Bands, lat_fast1: f64, out: &mut String) -> Result<Option<f64>, String> {
    let Some(forced1) = bands.get(&("forced".to_owned(), 1)) else {
        let _ = writeln!(
            out,
            "\nREPORT no forced c=1 band -- path_penalty UNMEASURED"
        );
        return Ok(None);
    };
    let lat_forced1 = med(forced1, "latency_p50_s")?;
    let penalty = div(lat_forced1, lat_fast1)?;
    let _ = writeln!(
        out,
        "\nREPORT === path_penalty (the term no recorded run contains) ==="
    );
    let _ = writeln!(
        out,
        "REPORT   latency_fast(1)    = {}s  CUDA-graph replay (generate_gpu_resident_streaming)",
        fixed(lat_fast1, 0, 3)
    );
    let _ = writeln!(
        out,
        "REPORT   latency_batched(1) = {}s  eager launch (batched_decode_step, APR_FORCE_BATCHED_PATH=1)",
        fixed(lat_forced1, 0, 3)
    );
    let _ = writeln!(
        out,
        "REPORT   path_penalty       = {}",
        fixed(penalty, 0, 3)
    );
    Ok(Some(penalty))
}

/// One row of the decomposition table. No comparison, no verdict.
#[allow(clippy::cast_precision_loss)]
fn index_row(
    c: i128,
    recs: &[Rec],
    lat_fast1: f64,
    agg_fast1: f64,
    penalty: Option<f64>,
) -> Result<String, String> {
    let ser = div(med(recs, "latency_p50_s")?, lat_fast1)?;
    let tok = div(c as f64 * agg_fast1, med(recs, "agg_tok_s")?)?;
    let scaling = match penalty {
        Some(p) if p != 0.0 => ser / p,
        _ => f64::NAN,
    };
    Ok(format!(
        "REPORT {c:>3}{}{}{}",
        fixed(ser, 12, 3),
        fixed(tok, 13, 3),
        fixed(scaling, 15, 3)
    ))
}

fn print_index(
    bands: &Bands,
    lat_fast1: f64,
    agg_fast1: f64,
    penalty: Option<f64>,
    out: &mut String,
) -> Result<(), String> {
    let _ = writeln!(out, "\nREPORT === index, decomposed ===");
    let _ = writeln!(
        out,
        "REPORT {:>3}{:>12}{:>13}{:>15}",
        "c", "ser_index", "token_index", "scaling_index"
    );
    for ((_, c), recs) in bands.iter().filter(|((mode, _), _)| mode == "fast") {
        let row = index_row(*c, recs, lat_fast1, agg_fast1, penalty)?;
        let _ = writeln!(out, "{row}");
    }
    Ok(())
}

/// `main()` short of the exit: the status it returns, or the reason it raised.
fn report(out_dir: &str, out: &mut String) -> Result<u8, String> {
    let bands = load(out_dir, out)?;
    if bands.is_empty() {
        out.push_str("REPORT no bands to report\n");
        return Ok(1);
    }
    print_bands(&bands, out)?;
    let Some(fast1) = bands.get(&("fast".to_owned(), 1)) else {
        out.push_str("\nREPORT no fast c=1 band -- cannot form the index\n");
        return Ok(1);
    };
    let lat_fast1 = med(fast1, "latency_p50_s")?;
    let penalty = print_penalty(&bands, lat_fast1, out)?;
    print_index(&bands, lat_fast1, med(fast1, "agg_tok_s")?, penalty, out)?;
    out.push_str(
        "\nREPORT the postcondition `serialization_index(c) < c` is NOT decided here.\n\
         REPORT it is an arm of scripts/perf_gate.sh, read from scripts/perf-matrix.yaml (PP-33).\n\
         REPORT this file decomposes the number; it does not discharge it.\n",
    );
    Ok(0)
}

/// The report for OUT_DIR (`None`: `/tmp/perf041`, as the original's default).
pub fn run(out_dir: Option<&str>) -> Outcome {
    let mut stdout = String::new();
    match report(out_dir.unwrap_or("/tmp/perf041"), &mut stdout) {
        Ok(code) => Outcome {
            stdout,
            code,
            stderr: String::new(),
        },
        Err(reason) => Outcome {
            stdout,
            code: 1,
            stderr: reason,
        },
    }
}

#[cfg(test)]
mod tests {
    //! The golden strings are what `scripts/perf041_report.py` printed on the same
    //! fixtures, which `scripts/tests/ci_tools_py_parity_test.sh` §6 builds identically.
    use super::*;
    use std::path::PathBuf;

    const TRAILER: &str =
        "\nREPORT the postcondition `serialization_index(c) < c` is NOT decided here.\n\
REPORT it is an arm of scripts/perf_gate.sh, read from scripts/perf-matrix.yaml (PP-33).\n\
REPORT this file decomposes the number; it does not discharge it.\n";
    const HEAD: &str = "\nREPORT === PERF-041 bands (median over replicates) ===\n\
REPORT mode      c  reps   lat_p50_s   agg_tok_s   tok_p50  tok_min  tok_max\n";
    const NO_FORCED: &str = "\nREPORT no forced c=1 band -- path_penalty UNMEASURED\n";
    const INDEX: &str =
        "\nREPORT === index, decomposed ===\nREPORT   c   ser_index  token_index  scaling_index\n";
    const FULL_BANDS: &str =
        "REPORT fast      1     2       0.515        40.2        64       60       62\n\
REPORT fast      2     2       0.715        71.2        64       58       66\n\
REPORT fast      4     2       1.315       110.2        63       50       70\n";
    const FULL_PENALTY: &str = "\nREPORT === path_penalty (the term no recorded run contains) ===\n\
REPORT   latency_fast(1)    = 0.515s  CUDA-graph replay (generate_gpu_resident_streaming)\n\
REPORT   latency_batched(1) = 0.615s  eager launch (batched_decode_step, APR_FORCE_BATCHED_PATH=1)\n\
REPORT   path_penalty       = 1.194\n";
    const SKIPS: &str = "REPORT skip b-err.json: connection refused\n\
REPORT skip c-err.json: {'code': 7, 'why': [1.5, None, True, 1e+16, 1e-05, \"it's\"]}\n\
REPORT skip d-err.json: None\n";

    fn band(label: &str, c: &str, lat: &str, agg: &str, p50: &str, mn: &str, mx: &str) -> String {
        format!(
            "{{\"label\": \"{label}\", \"c\": {c}, \"latency_p50_s\": {lat}, \"agg_tok_s\": {agg}, \
             \"tokens_p50\": {p50}, \"tokens_min\": {mn}, \"tokens_max\": {mx}}}\n"
        )
    }

    fn fixture(name: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let d = std::env::temp_dir().join(format!("perf041-report-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("fixture dir");
        for (f, body) in files {
            std::fs::write(d.join(f), body).expect("fixture file");
        }
        d
    }

    fn full_files() -> Vec<(String, String)> {
        let mut v = Vec::new();
        for r in 1..=2 {
            v.push((
                format!("fast-c1-r{r}.json"),
                band(
                    "fast-c1",
                    "1",
                    &format!("0.5{r}"),
                    &format!("40.{r}"),
                    "64",
                    "60",
                    &format!("6{r}"),
                ),
            ));
            v.push((
                format!("fast-c2-r{r}.json"),
                band(
                    "fast-c2",
                    "2",
                    &format!("0.7{r}"),
                    &format!("71.{r}"),
                    "64",
                    "58",
                    "66",
                ),
            ));
            v.push((
                format!("fast-c4-r{r}.json"),
                band(
                    "fast-c4",
                    "4",
                    &format!("1.3{r}"),
                    &format!("110.{r}"),
                    "63",
                    "50",
                    "70",
                ),
            ));
            v.push((
                format!("forced-c1-r{r}.json"),
                band(
                    "forced-c1",
                    "1",
                    &format!("0.6{r}"),
                    &format!("35.{r}"),
                    "64",
                    "61",
                    "64",
                ),
            ));
        }
        v
    }

    fn run_on(name: &str, files: &[(String, String)]) -> Outcome {
        let refs: Vec<(&str, &[u8])> = files
            .iter()
            .map(|(f, b)| (f.as_str(), b.as_bytes()))
            .collect();
        let d = fixture(name, &refs);
        let o = run(Some(d.to_str().expect("utf-8 temp dir")));
        let _ = std::fs::remove_dir_all(&d);
        o
    }

    fn ok(stdout: String) -> Outcome {
        Outcome {
            stdout,
            code: 0,
            stderr: String::new(),
        }
    }

    fn assert_stops(o: &Outcome, stdout: &str) {
        assert_eq!(o.stdout, stdout);
        assert_eq!(o.code, 1);
        assert!(!o.stderr.is_empty(), "a stop names its reason");
    }

    fn with(mut base: Vec<(String, String)>, extra: &[(&str, &str)]) -> Vec<(String, String)> {
        for (f, b) in extra {
            base.retain(|(g, _)| g != f);
            base.push(((*f).to_owned(), (*b).to_owned()));
        }
        base
    }

    fn skip_files() -> Vec<(String, String)> {
        with(
            full_files(),
            &[
                ("a-bad.json", "{\"label\": \n"),
                ("b-err.json", "{\"error\": \"connection refused\", \"c\": 1}\n"),
                ("c-err.json", "{\"error\": {\"code\": 7, \"why\": [1.5, null, true, 1e16, 1e-05, \"it's\"]}}\n"),
                ("d-err.json", "{\"error\": null}\n"),
                ("e-bom.json", "\u{feff}{\"c\": 1}\n"),
                (".hidden.json", "[]\n"),
                ("notes.txt", "[]\n"),
                ("x.JSON", "[]\n"),
            ],
        )
    }

    fn full_index() -> String {
        format!(
            "{INDEX}REPORT   1       1.000        1.000          0.837\n\
             REPORT   2       1.388        1.129          1.163\n\
             REPORT   4       2.553        1.458          2.138\n{TRAILER}"
        )
    }

    const FORCED_ROW: &str =
        "REPORT forced    1     2       0.615        35.2        64       61       64\n";

    #[test]
    fn full_sweep() {
        let want = format!(
            "{HEAD}{FULL_BANDS}{FORCED_ROW}{FULL_PENALTY}{}",
            full_index()
        );
        assert_eq!(run_on("full", &full_files()), ok(want));
    }

    #[test]
    fn skips_print_error_records_and_ignore_the_rest() {
        let want = format!(
            "{SKIPS}{HEAD}{FULL_BANDS}{FORCED_ROW}{FULL_PENALTY}{}",
            full_index()
        );
        assert_eq!(run_on("skips", &skip_files()), ok(want));
    }

    #[test]
    fn no_forced_band_leaves_scaling_nan() {
        let files: Vec<_> = full_files()
            .into_iter()
            .filter(|(f, _)| f.starts_with("fast-"))
            .collect();
        let want = format!(
            "{HEAD}{FULL_BANDS}{NO_FORCED}{INDEX}REPORT   1       1.000        1.000            nan\n\
             REPORT   2       1.388        1.129            nan\n\
             REPORT   4       2.553        1.458            nan\n{TRAILER}"
        );
        assert_eq!(run_on("noforced", &files), ok(want));
    }

    #[test]
    fn no_fast_c1_band_exits_1() {
        let files: Vec<_> = full_files()
            .into_iter()
            .filter(|(f, _)| !f.starts_with("fast-c1"))
            .collect();
        let want = format!(
            "{HEAD}REPORT fast      2     2       0.715        71.2        64       58       66\n\
             REPORT fast      4     2       1.315       110.2        63       50       70\n{FORCED_ROW}\
             \nREPORT no fast c=1 band -- cannot form the index\n"
        );
        assert_eq!(
            run_on("nofast1", &files),
            Outcome {
                stdout: want,
                code: 1,
                stderr: String::new()
            }
        );
    }

    #[test]
    fn no_band_exits_1() {
        let none = Outcome {
            stdout: "REPORT no bands to report\n".into(),
            code: 1,
            stderr: String::new(),
        };
        assert_eq!(run_on("empty", &[]), none);
        assert_eq!(run(Some("/nonexistent/perf041-report")), none);
        let only_skips = with(Vec::new(), &[("d-err.json", "{\"error\": null}\n")]);
        let want = "REPORT skip d-err.json: None\nREPORT no bands to report\n";
        assert_eq!(run_on("onlyskip", &only_skips).stdout, want);
    }

    #[test]
    fn integer_medians_and_bools() {
        let files = with(
            Vec::new(),
            &[
                ("m1.json", &band("fast-1", "1", "1", "true", "2", "2", "9")),
                ("m2.json", &band("fast-1", "1", "2", "2", "3", "2.0", "9")),
                ("m3.json", &band("fast-1", "1", "4", "3", "3", "5", "9")),
                ("m4.json", &band("fast-1", "1", "3", "4", "2", "3", "9")),
            ],
        );
        let want = format!(
            "{HEAD}REPORT fast      1     4       2.500         2.5         2        2        9\n\
             {NO_FORCED}{INDEX}REPORT   1       1.000        1.000            nan\n{TRAILER}"
        );
        assert_eq!(run_on("ints", &files), ok(want));
    }

    #[test]
    fn modes_and_their_order() {
        let other = |label: &str, c: &str| {
            let l = if label.is_empty() {
                String::new()
            } else {
                format!("\"label\": {label}, ")
            };
            format!("{{{l}\"c\": {c}, \"latency_p50_s\": 1, \"agg_tok_s\": 2, \"tokens_p50\": 3, \"tokens_min\": 1, \"tokens_max\": 1}}\n")
        };
        let files = with(
            Vec::new(),
            &[
                ("a.json", &band("fastest", "1", "0.5", "10", "1", "1", "1")),
                ("b.json", &band("", "3", "0.5", "10", "1", "1", "1")),
                ("c.json", &other("", "10")),
                ("d.json", &other("null", "9")),
                ("e.json", &other("false", "9")),
                ("f.json", &other("0", "9")),
                ("g.json", &band("-fast", "2", "0.5", "10", "1", "1", "1")),
                ("h.json", &band("Zeta-1", "1", "0.5", "10", "1", "1", "1")),
                (
                    "i.json",
                    &band("épsilon-1", "1", "0.5", "10", "1", "1", "1"),
                ),
                ("j.json", &band("fast-1", "1", "0.25", "20", "1", "1", "1")),
                ("k.json", &band("fast-10", "10", "2.5", "60", "1", "1", "1")),
                ("l.json", &band("fast-9", "9", "2", "50", "1", "1", "1")),
            ],
        );
        let want = format!(
            "{HEAD}REPORT           2     1       0.500        10.0         1        1        1\n\
             REPORT ?         3     1       0.500        10.0         1        1        1\n\
             REPORT ?         9     3       1.000         2.0         3        1        1\n\
             REPORT ?        10     1       1.000         2.0         3        1        1\n\
             REPORT Zeta      1     1       0.500        10.0         1        1        1\n\
             REPORT fast      1     1       0.250        20.0         1        1        1\n\
             REPORT fast      9     1       2.000        50.0         1        1        1\n\
             REPORT fast     10     1       2.500        60.0         1        1        1\n\
             REPORT fastest   1     1       0.500        10.0         1        1        1\n\
             REPORT épsilon   1     1       0.500        10.0         1        1        1\n\
             {NO_FORCED}{INDEX}REPORT   1       1.000        1.000            nan\n\
             REPORT   9       8.000        3.600            nan\n\
             REPORT  10      10.000        3.333            nan\n{TRAILER}"
        );
        assert_eq!(run_on("modes", &files), ok(want));
    }

    #[test]
    fn c_as_python_int_reads_it() {
        let files = with(
            Vec::new(),
            &[
                ("a.json", &band("fast-1", "\" 1 \"", "0.5", "10", "1", "1", "1")),
                ("b.json", &band("fast-10", "\"1_0\"", "1.5", "30", "1", "1", "1")),
                ("c.json", &band("fast-2", "2.9", "1", "18", "1", "1", "1")),
                ("d.json", &band("fast-1", "true", "0.4", "11", "1", "1", "1")),
                ("e.json", &band("fast-3", "\"+3\"", "1.2", "25", "1", "1", "1")),
                ("f.json", &band("fast-7", "\"-7\"", "1.2", "25", "1", "1", "1")),
                ("g.json", "{\"label\": \"fast-4\", \"c\": 1, \"c\": 4, \"latency_p50_s\": 1, \"agg_tok_s\": 2, \"tokens_p50\": 3, \"tokens_min\": 1, \"tokens_max\": 1}\n"),
                ("h.json", &band("fast-0", "-0.5", "1", "2", "1", "1", "1")),
            ],
        );
        let want = format!(
            "{HEAD}REPORT fast     -7     1       1.200        25.0         1        1        1\n\
             REPORT fast      0     1       1.000         2.0         1        1        1\n\
             REPORT fast      1     2       0.450        10.5         1        1        1\n\
             REPORT fast      2     1       1.000        18.0         1        1        1\n\
             REPORT fast      3     1       1.200        25.0         1        1        1\n\
             REPORT fast      4     1       1.000         2.0         3        1        1\n\
             REPORT fast     10     1       1.500        30.0         1        1        1\n\
             {NO_FORCED}{INDEX}REPORT  -7       2.667       -2.940            nan\n\
             REPORT   0       2.222        0.000            nan\n\
             REPORT   1       1.000        1.000            nan\n\
             REPORT   2       2.222        1.167            nan\n\
             REPORT   3       2.667        1.260            nan\n\
             REPORT   4       2.222       21.000            nan\n\
             REPORT  10       3.333        3.500            nan\n{TRAILER}"
        );
        assert_eq!(run_on("cforms", &files), ok(want));
    }

    #[test]
    fn the_first_extreme_decides_int_or_float() {
        let ints = with(
            Vec::new(),
            &[
                ("a.json", &band("fast-1", "1", "0.5", "10", "1", "2", "3")),
                (
                    "b.json",
                    &band("fast-1", "1", "0.5", "10", "1", "2.0", "3.0"),
                ),
            ],
        );
        let want = format!(
            "{HEAD}REPORT fast      1     2       0.500        10.0         1        2        3\n\
             {NO_FORCED}{INDEX}REPORT   1       1.000        1.000            nan\n{TRAILER}"
        );
        assert_eq!(run_on("tokint", &ints), ok(want));
        let float_min = with(
            Vec::new(),
            &[
                ("a.json", &band("fast-1", "1", "0.5", "10", "1", "2.0", "3")),
                ("b.json", &band("fast-1", "1", "0.5", "10", "1", "2", "3")),
            ],
        );
        assert_stops(&run_on("tokflt", &float_min), HEAD);
        let float_max = with(
            Vec::new(),
            &[
                ("a.json", &band("fast-1", "1", "0.5", "10", "1", "2", "3")),
                ("b.json", &band("fast-1", "1", "0.5", "10", "1", "2", "3.5")),
            ],
        );
        assert_stops(&run_on("tokmax", &float_max), HEAD);
    }

    #[test]
    fn inf_over_inf_prints_nan() {
        let files = with(
            Vec::new(),
            &[
                (
                    "a.json",
                    &band("fast-1", "1", "1.7e308", "5", "1", "1", "1"),
                ),
                (
                    "b.json",
                    &band("fast-1", "1", "1.7e308", "5", "1", "1", "1"),
                ),
                (
                    "c.json",
                    &band("fast-2", "2", "1.7e308", "5", "1", "1", "1"),
                ),
                (
                    "d.json",
                    &band("fast-2", "2", "1.7e308", "5", "1", "1", "1"),
                ),
            ],
        );
        let want = format!(
            "{HEAD}REPORT fast      1     2         inf         5.0         1        1        1\n\
             REPORT fast      2     2         inf         5.0         1        1        1\n\
             {NO_FORCED}{INDEX}REPORT   1         nan        1.000            nan\n\
             REPORT   2         nan        2.000            nan\n{TRAILER}"
        );
        assert_eq!(run_on("nan", &files), ok(want));
    }

    #[test]
    fn zero_penalty_is_not_divided_by() {
        let files = with(
            Vec::new(),
            &[
                ("a.json", &band("fast-1", "1", "0.5", "10", "1", "1", "1")),
                ("b.json", &band("forced-1", "1", "0", "10", "1", "1", "1")),
                ("c.json", &band("fast-2", "2", "0.9", "18", "1", "1", "1")),
            ],
        );
        let want = format!(
            "{HEAD}REPORT fast      1     1       0.500        10.0         1        1        1\n\
             REPORT fast      2     1       0.900        18.0         1        1        1\n\
             REPORT forced    1     1       0.000        10.0         1        1        1\n\
             \nREPORT === path_penalty (the term no recorded run contains) ===\n\
             REPORT   latency_fast(1)    = 0.500s  CUDA-graph replay (generate_gpu_resident_streaming)\n\
             REPORT   latency_batched(1) = 0.000s  eager launch (batched_decode_step, APR_FORCE_BATCHED_PATH=1)\n\
             REPORT   path_penalty       = 0.000\n\
             {INDEX}REPORT   1       1.000        1.000            nan\n\
             REPORT   2       1.800        1.111            nan\n{TRAILER}"
        );
        assert_eq!(run_on("z4", &files), ok(want));
    }

    #[test]
    fn zero_divisors_stop_where_python_raised() {
        let fast0 = band("fast-1", "1", "0", "10", "1", "1", "1");
        let row0 = "REPORT fast      1     1       0.000        10.0         1        1        1\n";
        let z1 = with(
            Vec::new(),
            &[
                ("a.json", &fast0),
                ("b.json", &band("forced-1", "1", "0.5", "10", "1", "1", "1")),
            ],
        );
        let forced =
            "REPORT forced    1     1       0.500        10.0         1        1        1\n";
        assert_stops(&run_on("z1", &z1), &format!("{HEAD}{row0}{forced}"));
        let z2 = with(Vec::new(), &[("a.json", &fast0)]);
        assert_stops(
            &run_on("z2", &z2),
            &format!("{HEAD}{row0}{NO_FORCED}{INDEX}"),
        );
        let z3 = with(
            Vec::new(),
            &[
                ("a.json", &band("fast-1", "1", "0.5", "10", "1", "1", "1")),
                ("b.json", &band("fast-2", "2", "0.5", "0", "1", "1", "1")),
            ],
        );
        let want = format!(
            "{HEAD}REPORT fast      1     1       0.500        10.0         1        1        1\n\
             REPORT fast      2     1       0.500         0.0         1        1        1\n\
             {NO_FORCED}{INDEX}REPORT   1       1.000        1.000            nan\n"
        );
        assert_stops(&run_on("z3", &z3), &want);
    }

    #[test]
    fn records_python_could_not_take_stop_after_what_it_printed() {
        assert_stops(
            &run_on("notobj", &with(skip_files(), &[("z.json", "[1, 2]\n")])),
            SKIPS,
        );
        assert_stops(
            &run_on("strobj", &with(skip_files(), &[("z.json", "\"error\"\n")])),
            SKIPS,
        );
        assert_stops(
            &run_on(
                "noc",
                &with(skip_files(), &[("z.json", "{\"label\": \"fast-1\"}\n")]),
            ),
            SKIPS,
        );
        let nolat = with(
            full_files(),
            &[("fast-c4-r3.json", "{\"label\": \"fast-4\", \"c\": 4}\n")],
        );
        let two = "REPORT fast      1     2       0.515        40.2        64       60       62\n\
                   REPORT fast      2     2       0.715        71.2        64       58       66\n";
        assert_stops(&run_on("nolat", &nolat), &format!("{HEAD}{two}"));
        let strlat = with(
            full_files(),
            &[(
                "fast-c4-r1.json",
                &band("fast-4", "4", "\"1.3\"", "1", "1", "1", "1"),
            )],
        );
        assert_stops(&run_on("strlat", &strlat), &format!("{HEAD}{two}"));
        let listtok = with(
            full_files(),
            &[(
                "fast-c2-r1.json",
                &band("fast-2", "2", "1", "1", "1", "[1]", "1"),
            )],
        );
        assert_stops(
            &run_on("listtok", &listtok),
            &format!("{HEAD}{}", &two[..=two.find('\n').expect("a row")]),
        );
        for (name, rec) in [
            (
                "badc",
                band("fast-1", "\"2.0\"", "0.5", "10", "1", "1", "1"),
            ),
            ("nullc", band("fast-1", "null", "0.5", "10", "1", "1", "1")),
            ("lblint", "{\"label\": 5, \"c\": 1}\n".to_owned()),
            ("lbltrue", "{\"label\": true, \"c\": 1}\n".to_owned()),
            ("lblfloat", "{\"label\": 0.5, \"c\": 1}\n".to_owned()),
            ("lbllist", "{\"label\": [0], \"c\": 1}\n".to_owned()),
            ("lbldict", "{\"label\": {\"a\": 0}, \"c\": 1}\n".to_owned()),
        ] {
            assert_stops(&run_on(name, &with(Vec::new(), &[("a.json", &rec)])), "");
        }
    }

    #[test]
    fn falsy_labels_are_unknown_mode() {
        for (name, label) in [
            ("lnull", "null"),
            ("lfalse", "false"),
            ("lzero", "0"),
            ("lzf", "0.0"),
            ("lemptylist", "[]"),
            ("lemptydict", "{}"),
            ("lstr", "\"\""),
        ] {
            let rec = format!("{{\"label\": {label}, \"c\": 1, \"latency_p50_s\": 1, \"agg_tok_s\": 2, \"tokens_p50\": 3, \"tokens_min\": 1, \"tokens_max\": 1}}\n");
            let o = run_on(name, &with(Vec::new(), &[("a.json", &rec)]));
            assert_eq!(o.code, 1, "{name}: no fast c=1 band");
            assert!(
                o.stdout.contains("REPORT ?         1     1"),
                "{name}: {}",
                o.stdout
            );
        }
    }

    #[test]
    fn unreadable_files_stop() {
        let d = fixture(
            "isdir",
            &[(
                "a.json",
                band("fast-1", "1", "0.5", "10", "1", "1", "1").as_bytes(),
            )],
        );
        std::fs::create_dir(d.join("x.json")).expect("dir");
        let o = run(Some(d.to_str().expect("utf-8")));
        let _ = std::fs::remove_dir_all(&d);
        assert_stops(&o, "");
        let d = fixture(
            "badutf8",
            &[
                (
                    "a.json",
                    band("fast-1", "1", "0.5", "10", "1", "1", "1").as_bytes(),
                ),
                ("b.json", b"{\"label\": \"\xff\"}\n"),
            ],
        );
        let o = run(Some(d.to_str().expect("utf-8")));
        let _ = std::fs::remove_dir_all(&d);
        assert_stops(&o, "");
    }

    #[test]
    fn medians() {
        let recs = |vals: &[&str]| -> Vec<Rec> {
            vals.iter()
                .map(|v| {
                    match serde_json::from_str::<Py>(&format!("{{\"k\": {v}}}")).expect("json") {
                        Py::Dict(d) => d,
                        _ => unreachable!(),
                    }
                })
                .collect()
        };
        assert_eq!(med(&recs(&["3", "1", "2"]), "k"), Ok(2.0));
        assert_eq!(med(&recs(&["1", "2"]), "k"), Ok(1.5));
        assert_eq!(med(&recs(&["4", "1", "3", "2"]), "k"), Ok(2.5));
        assert_eq!(med(&recs(&["1", "2.5"]), "k"), Ok(1.75));
        assert_eq!(med(&recs(&["1.5", "2"]), "k"), Ok(1.75));
        assert_eq!(med(&recs(&["3", "5"]), "k"), Ok(4.0));
        assert_eq!(
            med(&recs(&["9007199254740993", "9007199254740995"]), "k"),
            Ok(9_007_199_254_740_994.0)
        );
        // Two ints are summed exactly, then rounded once (Python: (x + y) / 2); summing
        // their f64 roundings gives ...992 here.
        assert_eq!(
            med(&recs(&["9007199254740993", "9007199254740994"]), "k"),
            Ok(9_007_199_254_740_994.0)
        );
        // A stable sort: -0.0 and 0.0 compare equal, so the middle one keeps its place.
        assert_eq!(
            med(&recs(&["0.0", "-0.0", "1"]), "k").map(f64::to_bits),
            Ok((-0.0f64).to_bits())
        );
        assert!(med(&recs(&["1", "\"x\""]), "k").is_err());
        assert!(med(&recs(&["1"]), "j").is_err());
        assert_eq!(extreme(&recs(&["3", "1", "2"]), "k", false), Ok(1));
        assert_eq!(extreme(&recs(&["3", "1", "2"]), "k", true), Ok(3));
        assert_eq!(extreme(&recs(&["1", "1.0"]), "k", false), Ok(1));
        assert_eq!(extreme(&recs(&["3", "3.0"]), "k", true), Ok(3));
        assert!(extreme(&recs(&["1.0", "1"]), "k", false).is_err());
        assert!(extreme(&recs(&["2", "2.5"]), "k", true).is_err());
        assert_eq!(extreme(&recs(&["true", "false"]), "k", true), Ok(1));
    }

    #[test]
    fn c_beyond_i128_stops_instead_of_saturating() {
        let at = |c: &str| {
            band_c(&vec![(
                "c".to_owned(),
                serde_json::from_str::<Py>(c).expect("json"),
            )])
        };
        assert_eq!(
            at("1.7e38"),
            Ok(169_999_999_999_999_998_061_923_293_023_115_935_744)
        );
        assert_eq!(
            at("-1.7e38"),
            Ok(-169_999_999_999_999_998_061_923_293_023_115_935_744)
        );
        assert!(
            at("1.7014118346046923e38").is_err(),
            "2^127 is not below i128::MAX as f64"
        );
        assert!(at("-1.7014118346046923e38").is_err());
        assert!(at("99999999999999999999999999999999999999999").is_err());
        assert!(at("\"99999999999999999999999999999999999999999\"").is_err());
    }

    #[test]
    fn python_int_of_a_str() {
        for (s, want) in [
            (" 12 ", Some(12)),
            ("+3", Some(3)),
            ("-7", Some(-7)),
            ("1_000", Some(1000)),
            ("007", Some(7)),
            ("\u{1f}5\t", Some(5)),
            ("", None),
            ("-", None),
            ("_1", None),
            ("1_", None),
            ("1__0", None),
            ("1a", None),
            ("2.0", None),
            ("+-1", None),
        ] {
            assert_eq!(py_int_str(s), want, "int({s:?})");
        }
    }

    #[test]
    fn fixed_point_as_python_formats_it() {
        assert_eq!(fixed(f64::NAN, 6, 3), "   nan");
        assert_eq!(fixed(-f64::NAN, 0, 3), "nan");
        assert_eq!(fixed(f64::INFINITY, 5, 3), "  inf");
        assert_eq!(fixed(f64::NEG_INFINITY, 0, 1), "-inf");
        assert_eq!(fixed(-0.0, 0, 3), "-0.000");
        assert_eq!(fixed(-0.0004, 0, 3), "-0.000");
        assert_eq!(fixed(2.5, 0, 0), "2");
        assert_eq!(fixed(3.5, 0, 0), "4");
        assert_eq!(fixed(0.125, 0, 2), "0.12");
        assert_eq!(fixed(0.0005, 0, 3), "0.001");
        assert_eq!(fixed(1e22, 0, 3), "10000000000000000000000.000");
        assert_eq!(
            div(1.0, 0.0),
            Err("ZeroDivisionError: float division by zero".to_owned())
        );
        assert!(div(1.0, -0.0).is_err());
        assert_eq!(div(1.0, 4.0), Ok(0.25));
    }
}
