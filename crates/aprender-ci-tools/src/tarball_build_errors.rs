//! `tarball-build-errors LOG` (was `scripts/lib/tarball_build_errors.py`, #4114): attribute
//! a tarball-workspace build's errors to the crates that own them.
//!
//! Reads `cargo build --tests --keep-going --message-format short` output and prints one
//! block per failing crate. Exit codes, as the original: 1 an error is attributed to a
//! crate, 0 no error at all, 3 the only errors belong to no crate, 4 the BUILD HOST failed
//! (no line of the log is then evidence about any crate), 2 the log is missing or empty or
//! the argument count is wrong.

use regex::Regex;
use std::path::PathBuf;
use std::sync::LazyLock;

// CPython's `\S` excludes `\x1c`-`\x1f` (str.isspace); Rust's does not, so it is spelled out.
static DIAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:.*/)?pkgs/([^/]+)/([^\s\x1c-\x1f]+?):(\d+):(\d+): error(?:\[\w+\])?: (.*)$")
        .expect("DIAG regex")
});
static FAILED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^error: could not compile `([^`]+)` \(([^)]+)\)").expect("FAILED regex")
});
static HOST: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"No space left on device|os error 28|failed to write `|Disk quota exceeded|signal: 9, SIGKILL|\(signal: 9\)|Cannot allocate memory",
    )
    .expect("HOST regex")
});

/// What one run prints and how it exits.
#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub stdout: String,
    pub stderr: String,
    pub code: u8,
}

/// `open(path, encoding="utf-8", errors="replace").read().splitlines()`: universal newlines
/// first, then every break `str.splitlines` knows. No trailing empty item.
pub fn py_splitlines(bytes: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(bytes)
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let mut lines: Vec<String> = text
        .split([
            '\n', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\u{85}', '\u{2028}', '\u{2029}',
        ])
        .map(str::to_owned)
        .collect();
    if lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

/// The longest crate name cargo reported that prefixes `<name>-<version>` (a version may hold
/// `-`, e.g. `0.70.0-rc.1`). Python's `max` keeps the first of equal lengths and
/// `max_by_key` the last, but two distinct names that both prefix `dir` before a `-` never
/// have equal lengths, so the two agree.
fn owner<'a>(dir: &str, failed: &'a [(String, Vec<String>)]) -> Option<&'a str> {
    failed
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| {
            dir.strip_prefix(name)
                .is_some_and(|rest| rest.starts_with('-'))
        })
        .max_by_key(|name| name.len())
}

type Groups = Vec<(String, Vec<String>)>;

fn printed(stdout: String, code: u8) -> Outcome {
    Outcome {
        stdout,
        stderr: String::new(),
        code,
    }
}

/// The host-failure report, when any line says the build host itself failed.
fn host_failure(lines: &[String]) -> Option<String> {
    let host: Vec<&String> = lines.iter().filter(|ln| HOST.is_match(ln)).collect();
    if host.is_empty() {
        return None;
    }
    let mut out = format!(
        "HOST  the build host failed, so no crate verdict is possible ({} line(s)):\n",
        host.len()
    );
    for ln in host.iter().take(5) {
        out.push_str(&format!("        {ln}\n"));
    }
    Some(out)
}

/// Diagnostics by tarball dir, "could not compile" targets by crate, other `error` lines.
fn collect(lines: &[String]) -> (Groups, Groups, Vec<&String>) {
    let (mut by_dir, mut failed, mut other) = (Vec::new(), Vec::new(), Vec::new());
    for ln in lines {
        if let Some(m) = DIAG.captures(ln) {
            let err = format!("{}:{}:{}: error: {}", &m[2], &m[3], &m[4], &m[5]);
            push_to(&mut by_dir, &m[1], err);
        } else if let Some(f) = FAILED.captures(ln) {
            push_to(&mut failed, &f[1], f[2].to_owned());
        } else if ln.starts_with("error") {
            other.push(ln);
        }
    }
    (by_dir, failed, other)
}

/// One RED block per tarball dir; its owner's targets are taken out of `failed`.
fn render_dirs(by_dir: &Groups, failed: &mut Groups, out: &mut String) {
    for (dir, errs) in by_dir {
        let owned = owner(dir, failed)
            .and_then(|name| failed.iter().position(|(n, _)| n == name))
            .map(|i| failed.remove(i).1.join(", "));
        let targets = owned.unwrap_or_else(|| "?".to_owned());
        out.push_str(&format!(
            "RED   {dir} ({targets}): {} error(s) in the published tarball\n",
            errs.len()
        ));
        for e in errs.iter().take(20) {
            out.push_str(&format!("        {e}\n"));
        }
    }
}

/// Judge one build log's lines.
pub fn judge(lines: &[String]) -> Outcome {
    if let Some(out) = host_failure(lines) {
        return printed(out, 4);
    }
    let (by_dir, mut failed, other) = collect(lines);
    if by_dir.is_empty() && failed.is_empty() && other.is_empty() {
        return printed(String::new(), 0);
    }
    let attributed = !by_dir.is_empty() || !failed.is_empty();
    let mut out = String::new();
    render_dirs(&by_dir, &mut failed, &mut out);
    for (name, targets) in &failed {
        out.push_str(&format!(
            "RED   {name} ({}): could not compile (no diagnostic attributed to a tarball path)\n",
            targets.join(", ")
        ));
    }
    for ln in other.iter().take(10) {
        out.push_str(&format!("ERROR {ln}\n"));
    }
    printed(out, if attributed { 1 } else { 3 })
}

fn push_to(map: &mut Groups, key: &str, item: String) {
    match map.iter_mut().find(|(k, _)| k == key) {
        Some((_, items)) => items.push(item),
        None => map.push((key.to_owned(), vec![item])),
    }
}

fn refuse(stderr: String) -> Outcome {
    Outcome {
        stdout: String::new(),
        stderr,
        code: 2,
    }
}

/// The whole command: `args` is everything after the subcommand.
pub fn run(args: &[PathBuf]) -> Outcome {
    let [log] = args else {
        return refuse("usage: tarball_build_errors.py LOG".to_owned());
    };
    let bytes = match std::fs::read(log) {
        Ok(b) => b,
        Err(e) => {
            return refuse(format!(
                "tarball_build_errors: cannot read {}: {e}",
                log.display()
            ));
        }
    };
    let lines = py_splitlines(&bytes);
    if lines.is_empty() {
        return refuse("tarball_build_errors: empty build log".to_owned());
    }
    judge(&lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(s: &str) -> Vec<String> {
        py_splitlines(s.as_bytes())
    }

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(
            lines("a\r\nb\rc\x0bd\u{2028}e\n"),
            ["a", "b", "c", "d", "e"]
        );
        assert_eq!(lines("\n"), [""]);
        assert!(lines("").is_empty());
        assert_eq!(lines("a\n\n"), ["a", ""]);
    }

    #[test]
    fn clean_log_is_0_and_silent() {
        let o = judge(&lines("   Compiling a v1\nwarning: unused\n"));
        assert_eq!((o.code, o.stdout.as_str()), (0, ""));
    }

    #[test]
    fn host_failure_is_4_before_any_crate() {
        let o = judge(&lines(
            "/w/pkgs/a-1.0.0/src/lib.rs:1:2: error: x\nerror: No space left on device\n",
        ));
        assert_eq!(o.code, 4);
        assert!(o.stdout.starts_with("HOST  the build host failed"));
        assert!(o.stdout.contains("(1 line(s))"));
    }

    #[test]
    fn diagnostic_owned_by_longest_prefix_crate() {
        let o = judge(&lines(concat!(
            "/w/pkgs/a-b-0.70.0-rc.1/src/lib.rs:3:4: error[E0425]: nope\n",
            "error: could not compile `a` (lib)\n",
            "error: could not compile `a-b` (lib test)\n",
        )));
        assert_eq!(o.code, 1);
        assert_eq!(
            o.stdout,
            concat!(
                "RED   a-b-0.70.0-rc.1 (lib test): 1 error(s) in the published tarball\n",
                "        src/lib.rs:3:4: error: nope\n",
                "RED   a (lib): could not compile (no diagnostic attributed to a tarball path)\n",
            )
        );
    }

    #[test]
    fn longer_prefix_wins_whatever_the_report_order_and_non_prefix_never_owns() {
        let o = judge(&lines(concat!(
            "/w/pkgs/a-b-1.0.0/src/lib.rs:1:1: error: x\n",
            "error: could not compile `zzzzzz` (lib)\n",
            "error: could not compile `a-b` (lib)\n",
            "error: could not compile `a` (lib test)\n",
        )));
        assert_eq!(
            o.stdout,
            concat!(
                "RED   a-b-1.0.0 (lib): 1 error(s) in the published tarball\n",
                "        src/lib.rs:1:1: error: x\n",
                "RED   zzzzzz (lib): could not compile (no diagnostic attributed to a tarball path)\n",
                "RED   a (lib test): could not compile (no diagnostic attributed to a tarball path)\n",
            )
        );
        let o = judge(&lines(
            "/w/pkgs/a-1.0.0/src/lib.rs:1:1: error: x\nerror: could not compile `zzzzzz` (lib)\n",
        ));
        assert!(o.stdout.starts_with("RED   a-1.0.0 (?): 1 error(s)"));
    }

    #[test]
    fn either_kind_of_attribution_alone_is_1() {
        assert_eq!(
            judge(&lines("error: could not compile `a` (lib)\n")).code,
            1
        );
        assert_eq!(
            judge(&lines("/w/pkgs/a-1.0.0/src/lib.rs:1:1: error: x\n")).code,
            1
        );
    }

    #[test]
    fn unowned_errors_only_is_3() {
        let o = judge(&lines("error: unexpected argument '--bogus'\n"));
        assert_eq!(o.code, 3);
        assert_eq!(o.stdout, "ERROR error: unexpected argument '--bogus'\n");
    }

    #[test]
    fn wrong_arg_count_and_missing_file_are_2() {
        assert_eq!(run(&[]).code, 2);
        assert_eq!(run(&[PathBuf::from("a"), PathBuf::from("b")]).code, 2);
        assert_eq!(run(&[PathBuf::from("/nonexistent/tbe.log")]).code, 2);
    }
}
