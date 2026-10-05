//! The answer oracles of CRUX prompt-set v2 (the port of `crux_oracles.py`):
//! `evaluate` and `extract`, the only two functions the judge calls.

use crate::pyerr::{PyErr, PyResult};
use crate::pyio::{decode_strict, universal_newlines};
use crate::pyre::{answer_re, fence_re, findall1, int_full, think_close, think_open, think_re};
use crate::pyval::{
    casefold, char_len, concat_str, norm_ws, py_eq, py_int, py_repr, py_rstrip, py_splitlines,
    py_str, py_strip, slice_chars, PyInt, Val,
};
use crate::sandbox::{run_python_cell, sandbox, CellRun, Sandbox};

pub const UNCLOSED: &str = "unclosed think (budget exhausted)";

/// What the judge reads of `evaluate`'s verdict: `correct`, and `why`
/// (None exactly when correct).
#[derive(Debug, Clone)]
pub struct Verdict {
    pub correct: bool,
    pub why: Option<String>,
}

fn verdict(correct: bool, why: impl Into<String>) -> Verdict {
    Verdict {
        correct,
        why: if correct { None } else { Some(why.into()) },
    }
}

/// The text after every closed think block, or None when one never closed.
pub fn strip_think(text: &str) -> Option<String> {
    let rest = think_re().replace_all(text, "").into_owned();
    if think_open().is_match(&rest) {
        return None;
    }
    match think_close().find_iter(&rest).last() {
        Some(m) => Some(rest[m.end()..].to_string()),
        None => Some(rest),
    }
}

/// The LAST `<answer>…</answer>` body.
pub fn extract_answer(text: &str) -> Option<String> {
    findall1(answer_re(), text).pop()
}

/// `normalize(value, how)` -> Str | Int | None.
pub fn normalize(value: &str, how: &Val) -> PyResult<Val> {
    if py_eq(how, &Val::str("exact")) {
        return Ok(Val::str(value));
    }
    if py_eq(how, &Val::str("casefold_strip")) {
        let folded = casefold(&norm_ws(value));
        return Ok(Val::Str(folded.trim_end_matches('.').to_string()));
    }
    if py_eq(how, &Val::str("int")) {
        let s = py_strip(value).replace(',', "");
        let s = s.trim_end_matches('.');
        if int_full().is_match(s) {
            return Ok(Val::Int(PyInt::parse_digits(s)?));
        }
        return Ok(Val::None);
    }
    Err(PyErr::value(format!(
        "unknown normalize {} (one of ('int', 'casefold_strip', 'exact'))",
        py_repr(how)
    )))
}

fn judge_answer(text: &str, oracle: &Val) -> PyResult<Verdict> {
    let Some(got) = extract_answer(text) else {
        return Ok(verdict(false, "no_answer_tag"));
    };
    let how = oracle.get_or("normalize", Val::str("casefold_strip"))?;
    let have = normalize(&got, &how)?;
    let want = normalize(&py_str(&oracle.item("expect")?), &how)?;
    if have.is_none() {
        // only the "int" normalizer returns None, so `how` is a str here
        return Ok(verdict(false, format!("answer_not_{}", py_str(&how))));
    }
    let eq = py_eq(&have, &want);
    Ok(verdict(eq, if eq { "match" } else { "mismatch" }))
}

/// The stripped, non-blank lines of an `<answer>` body.
fn answer_lines(got: &str) -> Vec<String> {
    py_splitlines(py_strip(got))
        .into_iter()
        .map(|ln| py_strip(ln).to_string())
        .filter(|ln| !ln.is_empty())
        .collect()
}

fn judge_structure(text: &str, oracle: &Val) -> PyResult<Verdict> {
    let Some(got) = extract_answer(text) else {
        return Ok(verdict(false, "no_answer_tag"));
    };
    let lines = answer_lines(&got);
    let want: Vec<String> = oracle.item("lines")?.iter()?.iter().map(py_str).collect();
    if lines == want {
        return Ok(verdict(true, "match"));
    }
    let first = lines
        .iter()
        .zip(&want)
        .position(|(a, b)| a != b)
        .unwrap_or(lines.len().min(want.len()));
    Ok(verdict(
        false,
        format!(
            "lines_differ_at_{first}: got {} want {}",
            lines.len(),
            want.len()
        ),
    ))
}

/// `re.escape(x)` raises this for anything but str/bytes.
fn escape_arg(v: &Val) -> PyResult<&str> {
    v.as_str().ok_or_else(|| {
        PyErr::type_err(format!(
            "decoding to str: need a bytes-like object, {} found",
            v.type_name()
        ))
    })
}

/// The line the cell prints after the last assert: rc 0 alone would accept a
/// reply that calls `sys.exit(0)` before the asserts run.
pub const SENTINEL: &str = "CRUX_TESTS_PASSED";

/// `clip_head(s, n)`: `s` cut to its first `n` chars, saying how many were cut.
fn clip_head(s: &str, n: usize) -> String {
    let len = char_len(s);
    if len <= n {
        s.to_string()
    } else {
        format!("{} … and {} more chars", slice_chars(s, 0, n), len - n)
    }
}

/// `judge_code`: the last python fence, with the prompt's asserts, run in
/// the host's sandbox.
fn judge_code(text: &str, oracle: &Val) -> PyResult<Verdict> {
    judge_code_in(text, oracle, sandbox())
}

/// `judge_code` against a given sandbox (or the reason there is none).
fn judge_code_in(text: &str, oracle: &Val, sb: Result<&Sandbox, &str>) -> PyResult<Verdict> {
    if !py_eq(
        &oracle.get_or("lang", Val::str("python"))?,
        &Val::str("python"),
    ) {
        return Ok(verdict(false, "lang_unsupported"));
    }
    let blocks = findall1(fence_re(), text);
    let Some(code) = blocks.last() else {
        return Ok(verdict(false, "no_code_block"));
    };
    let entry = oracle.item("entry")?;
    let pat = format!(
        r"(?m)^[\s\x1c-\x1f]*def[\s\x1c-\x1f]+{}[\s\x1c-\x1f]*\(",
        regex::escape(escape_arg(&entry)?)
    );
    let re = regex::Regex::new(&pat).map_err(|e| PyErr::value(e.to_string()))?;
    if !re.is_match(code) {
        return Ok(verdict(false, "entry_not_defined"));
    }
    let Ok(sb) = sb else {
        return Ok(verdict(false, "sandbox_unavailable"));
    };
    // setrlimit reads -1 as RLIM_INFINITY and refuses any other negative or
    // anything past rlim_t, inside preexec_fn, which Popen reports as this.
    let preexec = || PyErr::new("SubprocessError", "Exception occurred in preexec_fn.");
    let timeout_s = py_int(&oracle.get_or("timeout_s", Val::int(10))?)?
        .to_i64()
        .ok_or_else(preexec)?;
    let (cpu_s, wall_s) = match u64::try_from(timeout_s) {
        Ok(t) => (Some(t), t.saturating_add(5)),
        Err(_) if timeout_s == -1 => (None, 4),
        Err(_) => return Err(preexec()),
    };
    let mut cell = concat_str(&format!("{code}\n\n"), &oracle.item("tests")?)?;
    cell.push_str(&format!("\nprint('{SENTINEL}')\n"));
    let (rc, stdout, stderr) = match run_python_cell(sb, &cell, cpu_s, wall_s) {
        CellRun::Unavailable(_) => return Ok(verdict(false, "sandbox_unavailable")),
        CellRun::Timeout => return Ok(verdict(false, "tests_timeout")),
        CellRun::Exited { rc, stdout, stderr } => (rc, stdout, stderr),
    };
    let stdout = universal_newlines(&decode_strict(&stdout)?);
    let stderr = universal_newlines(&decode_strict(&stderr)?);
    if rc == 0 && py_rstrip(&stdout).ends_with(SENTINEL) {
        return Ok(verdict(true, ""));
    }
    let tail = py_splitlines(py_strip(&stderr))
        .last()
        .map_or_else(|| format!("rc={rc}"), |l| (*l).to_string());
    Ok(verdict(
        false,
        clip_head(&format!("tests_failed: {tail}"), 200),
    ))
}

/// `evaluate(prompt, text, turns)`.
pub fn evaluate(prompt: &Val, text: &Val, turns: Option<&Val>) -> PyResult<Verdict> {
    let oracle = prompt.item("oracle")?;
    let kind = oracle.get("type")?;
    let mut text = text.clone();
    if let Some(turns) = turns {
        let mut n_user = 0usize;
        for m in prompt.item("messages")?.iter()? {
            if py_eq(&m.item("role")?, &Val::str("user")) {
                n_user += 1;
            }
        }
        let got = turns.len()?;
        if got != n_user {
            return Ok(verdict(
                false,
                format!("turns_missing: got {got} want {n_user}"),
            ));
        }
        if py_eq(&kind, &Val::str("state_recall")) {
            text = turns.last()?;
        }
    }
    let Val::Str(text) = text else {
        return Ok(verdict(false, "no_text"));
    };
    let Some(text) = strip_think(&text) else {
        return Ok(verdict(false, UNCLOSED));
    };
    if py_eq(&kind, &Val::str("answer")) || py_eq(&kind, &Val::str("state_recall")) {
        return judge_answer(&text, &oracle);
    }
    if py_eq(&kind, &Val::str("code_tests")) {
        return judge_code(&text, &oracle);
    }
    if py_eq(&kind, &Val::str("structure")) {
        return judge_structure(&text, &oracle);
    }
    Ok(verdict(false, format!("unknown_oracle {}", py_repr(&kind))))
}

/// `extract(prompt, text)`: what a same-representation comparison compares.
pub fn extract(prompt: &Val, text: &Val) -> PyResult<Val> {
    let Val::Str(text) = text else {
        return Ok(Val::None);
    };
    let Some(text) = strip_think(text) else {
        return Ok(Val::None);
    };
    let oracle = prompt.item("oracle")?;
    if py_eq(&oracle.get("type")?, &Val::str("code_tests")) {
        let blocks = findall1(fence_re(), &text);
        return Ok(blocks.last().map_or(Val::None, |b| Val::str(py_strip(b))));
    }
    let Some(got) = extract_answer(&text) else {
        return Ok(Val::None);
    };
    if py_eq(&oracle.get("type")?, &Val::str("structure")) {
        return Ok(Val::Str(answer_lines(&got).join("\n")));
    }
    let v = normalize(
        &got,
        &oracle.get_or("normalize", Val::str("casefold_strip"))?,
    )?;
    Ok(if v.is_none() {
        Val::None
    } else {
        Val::Str(py_str(&v))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::loads;
    use crate::pyval::dict;
    use crate::sandbox::tests::host_sandbox;

    fn p(json: &str) -> Val {
        loads(json).expect("test json")
    }

    #[test]
    fn strip_think_cases() {
        assert_eq!(strip_think("<think>a</think>b").as_deref(), Some("b"));
        assert_eq!(strip_think("<THINK>a").as_deref(), None);
        assert_eq!(strip_think("x</think>y</think>z").as_deref(), Some("z"));
        assert_eq!(strip_think("plain").as_deref(), Some("plain"));
    }

    #[test]
    fn normalizers() {
        let int = Val::str("int");
        assert!(py_eq(
            &normalize(" 1,234. ", &int).unwrap(),
            &Val::int(1234)
        ));
        assert!(normalize("12a", &int).unwrap().is_none());
        assert!(py_eq(
            &normalize("\u{661}\u{662}", &int).unwrap(),
            &Val::int(12)
        ));
        let cf = Val::str("casefold_strip");
        assert_eq!(
            py_str(&normalize("  Paris  Is..", &cf).unwrap()),
            "paris is"
        );
        assert_eq!(
            normalize("x", &Val::str("nope")).unwrap_err().msg,
            "unknown normalize 'nope' (one of ('int', 'casefold_strip', 'exact'))"
        );
    }

    #[test]
    fn evaluate_answer_and_structure() {
        let pr = p(
            r#"{"oracle": {"type": "answer", "expect": 4, "normalize": "int"}, "messages": [{"role": "user", "content": "x"}]}"#,
        );
        let v = evaluate(&pr, &Val::str("<think>no</think><answer>4</answer>"), None).unwrap();
        assert!(v.correct && v.why.is_none());
        let v = evaluate(&pr, &Val::str("4"), None).unwrap();
        assert_eq!(v.why.as_deref(), Some("no_answer_tag"));
        let turns = Val::List(vec![Val::str("a"), Val::str("b")]);
        let v = evaluate(&pr, &Val::str("x"), Some(&turns)).unwrap();
        assert_eq!(v.why.as_deref(), Some("turns_missing: got 2 want 1"));
        let st = p(r#"{"oracle": {"type": "structure", "lines": ["a", "b"]}}"#);
        let v = evaluate(&st, &Val::str("<answer>\n a\n\n c \n</answer>"), None).unwrap();
        assert_eq!(v.why.as_deref(), Some("lines_differ_at_1: got 2 want 2"));
        assert_eq!(
            py_str(&extract(&st, &Val::str("<answer> a \n b</answer>")).unwrap()),
            "a\nb"
        );
    }

    /// A `code_tests` oracle over `def f(): return 1` plus `extra`, judged in
    /// `sb`: "PASS", the RED reason, or "crash: <kind>: <msg>".
    fn judge_cell(extra: &str, tests: Val, timeout_s: Val, sb: Result<&Sandbox, &str>) -> String {
        let oracle = dict(vec![
            ("type", Val::str("code_tests")),
            ("entry", Val::str("f")),
            ("tests", tests),
            ("timeout_s", timeout_s),
        ]);
        let text = format!("reply\n```python\ndef f():\n    return 1\n{extra}```\n");
        match judge_code_in(&text, &oracle, sb) {
            Ok(v) if v.correct => "PASS".to_string(),
            Ok(v) => v.why.unwrap_or_default(),
            Err(e) => format!("crash: {}: {}", e.kind, e.msg),
        }
    }

    /// The sandbox case table: one row per property, each with the outcome
    /// measured on a host that has the sandbox. A positive row turns RED when
    /// its property is dropped from the seam; a planted row proves its assert
    /// can fail. Runs on the real sandbox: a host without one fails here.
    #[test]
    fn sandbox_case_table() {
        let clipped = format!("tests_failed: {} … and 114 more chars", "x".repeat(186));
        let null_stdin = "import os, stat\nst = os.fstat(0)\nnull = stat.S_ISCHR(st.st_mode) and st.st_rdev == os.stat('/dev/null').st_rdev\n";
        let mode = "import os, stat\nmode = stat.S_IMODE(os.stat('.').st_mode)\n";
        let (stdin_ok, stdin_planted) = (
            format!("{null_stdin}assert null, st"),
            format!("{null_stdin}assert not null"),
        );
        let (mode_ok, mode_planted) = (
            format!("{mode}assert mode == 0o700, oct(mode)"),
            format!("{mode}assert mode != 0o700"),
        );
        let rows: [(&str, &str, &str, i64, &str); 22] = [
            ("pass", "", "assert f() == 1", 10, "PASS"),
            ("assert fails", "", "assert f() == 2", 10, "tests_failed: AssertionError"),
            ("sentinel: exit 0 before the asserts", "import sys\nsys.exit(0)\n", "assert f() == 1", 10, "tests_failed: rc=0"),
            ("unshare -n: only lo", "", "import socket\nassert [n for _, n in socket.if_nameindex()] == ['lo']", 10, "PASS"),
            ("unshare -n: connect", "", "import socket\ns = socket.socket()\ns.settimeout(2)\ns.connect(('192.0.2.1', 53))", 10, "tests_failed: OSError: [Errno 101] Network is unreachable"),
            ("-I", "", "import sys\nassert sys.flags.isolated == 1", 10, "PASS"),
            ("-I planted", "", "import sys\nassert sys.flags.isolated == 0", 10, "tests_failed: AssertionError"),
            ("env scrubbed", "", "import os\nassert set(os.environ) <= {'PATH', 'LC_CTYPE'}, sorted(os.environ)\nassert os.environ['PATH'] == '/usr/bin:/bin'", 10, "PASS"),
            ("env planted", "", "import os\nassert 'HOME' in os.environ", 10, "tests_failed: AssertionError"),
            ("stdin is /dev/null", "", stdin_ok.as_str(), 10, "PASS"),
            ("stdin planted", "", stdin_planted.as_str(), 10, "tests_failed: AssertionError"),
            ("fresh tmpdir", "", "import os\nassert os.listdir('.') == ['cell.py'], os.listdir('.')", 10, "PASS"),
            ("fresh tmpdir planted", "", "import os\nassert os.listdir('.') != ['cell.py']", 10, "tests_failed: AssertionError"),
            ("tmpdir is 0700", "", mode_ok.as_str(), 10, "PASS"),
            ("tmpdir mode planted", "", mode_planted.as_str(), 10, "tests_failed: AssertionError"),
            ("RLIMIT_AS", "", "bytearray(2 << 30)", 10, "tests_failed: MemoryError"),
            ("RLIMIT_CPU (soft = hard: SIGKILL)", "", "while True:\n    pass", 1, "tests_failed: rc=-9"),
            ("RLIMIT_FSIZE", "", "with open('big', 'wb') as fh:\n    fh.write(b'x' * (32 << 20))", 10, "tests_failed: OSError: [Errno 27] File too large"),
            ("wall timeout = timeout_s + 5", "", "import time\ntime.sleep(60)", 1, "tests_timeout"),
            ("timeout_s -1 = no CPU limit", "", "assert f() == 1", -1, "PASS"),
            ("stderr tail clipped", "", "raise SystemExit('x' * 300)", 10, clipped.as_str()),
            // Inherited from the Python oracle, kept for parity: the sentinel
            // is printed by the cell, so a reply can print it, flush and leave
            // before the asserts run.
            ("inherited: sentinel then os._exit", "import os\nprint('CRUX_TESTS_PASSED', flush=True)\nos._exit(0)\n", "assert f() == 2", 10, "PASS"),
        ];
        let sb = host_sandbox();
        for (name, extra, tests, timeout_s, want) in rows {
            let got = judge_cell(extra, Val::str(tests), Val::int(timeout_s), Ok(sb));
            assert_eq!(got, want, "row {name:?}");
        }
    }

    /// What happens before and around the sandbox: no sandbox, a spawn error,
    /// and the TypeErrors / crashes the Python oracle raises.
    #[test]
    fn code_tests_outside_the_sandbox() {
        let ok = Val::str("assert f() == 1");
        assert_eq!(
            judge_cell("", ok.clone(), Val::int(10), Err("no python3 on PATH")),
            "sandbox_unavailable"
        );
        let broken = host_sandbox().with_prlimit("/nonexistent/prlimit");
        assert_eq!(
            judge_cell("", ok.clone(), Val::int(10), Ok(&broken)),
            "sandbox_unavailable"
        );
        let no_tmp = host_sandbox().with_tmp("/nonexistent/tmp");
        assert_eq!(
            judge_cell("", ok.clone(), Val::int(10), Ok(&no_tmp)),
            "sandbox_unavailable"
        );
        let sb = Ok(host_sandbox());
        assert_eq!(judge_cell("", ok.clone(), Val::str(" 1_0 "), sb), "PASS");
        assert_eq!(
            judge_cell("", ok.clone(), Val::str("1__0"), sb),
            "crash: ValueError: invalid literal for int() with base 10: '1__0'"
        );
        assert_eq!(
            judge_cell("", ok.clone(), Val::None, sb),
            "crash: TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"
        );
        assert_eq!(
            judge_cell("", ok.clone(), Val::int(-2), sb),
            "crash: SubprocessError: Exception occurred in preexec_fn."
        );
        assert_eq!(
            judge_cell("", Val::int(1), Val::int(10), sb),
            "crash: TypeError: can only concatenate str (not \"int\") to str"
        );
        assert_eq!(
            judge_cell("", Val::str("import sys\nsys.stdout.buffer.write(b'\\xff')"), Val::int(10), sb),
            "crash: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"
        );
        assert_eq!(
            judge_cell(
                "",
                Val::str(
                    "import sys\nsys.stdout.write('a\\r\\nCRUX_TESTS_PASSED\\r')\nsys.exit(0)"
                ),
                Val::int(10),
                sb
            ),
            "PASS"
        );
        let mut no_tests = dict(vec![
            ("type", Val::str("code_tests")),
            ("entry", Val::str("f")),
        ]);
        if let Val::Dict(d) = &mut no_tests {
            d.put("timeout_s", Val::int(10));
        }
        let text = "```python\ndef f():\n    return 1\n```";
        let err = judge_code_in(text, &no_tests, sb).unwrap_err();
        assert_eq!((err.kind, err.msg.as_str()), ("KeyError", "'tests'"));
        assert_eq!(clip_head("abc", 3), "abc");
        assert_eq!(clip_head("abcd", 3), "abc … and 1 more chars");
    }
}
