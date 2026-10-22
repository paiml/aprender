//! `section-first-fail LOG...` (PMAT-5740, #5740): name the first failing line of each
//! section log the x86-main driver wrote, as one `::error::` line per log. The driver
//! prints only `section N: failure`; the cause sits in the section's own log, dumped in a
//! `::group` that a cut-off job log loses.

use std::path::Path;

/// The first `::error` line of the log, else the first panic or `error:` line, else "".
/// An unreadable log has none.
pub fn first_failing_line(path: &Path) -> String {
    let Ok(bytes) = std::fs::read(path) else {
        return String::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    if let Some(k) = killed_rustc(&lines) {
        return k;
    }
    let first = lines
        .iter()
        .find(|l| l.starts_with("::error"))
        .or_else(|| lines.iter().find(|l| is_panic_or_error(l)));
    first.map_or_else(String::new, |l| (*l).to_owned())
}

/// sccache exits 2 when its rustc is killed by a signal (OOM or host kill), so cargo prints
/// `could not compile` and `(exit status: 2)` with no rustc diagnostic (#5750). Name that,
/// not a compile error: the first `could not compile` line before an sccache `exit status: 2`. A real error
/// (`::error`, panic, `error:`) earlier in the log wins instead, so it is never masked.
fn killed_rustc(lines: &[&str]) -> Option<String> {
    let at = lines
        .iter()
        .position(|l| l.contains("sccache") && l.contains("(exit status: 2)"))?;
    let real_error_first = lines[..at].iter().any(|l| {
        l.starts_with("::error") || (is_panic_or_error(l) && !l.contains("could not compile"))
    });
    if real_error_first {
        return None;
    }
    let krate = lines[..at]
        .iter()
        .rev()
        .find(|l| l.contains("could not compile"))
        .map_or("", |l| l.trim_start_matches("error: ").trim());
    Some(format!(
        "rustc killed by signal (sccache exit status 2, not a compile error): {krate}"
    ))
}

fn is_panic_or_error(line: &str) -> bool {
    let t = line.trim_start();
    line.contains("panicked at")
        || t.starts_with("error:")
        || t.starts_with("error[")
        || t.starts_with("Error:")
}

/// `report` over the failed, blocking sections of a driver `results.json`; each section's
/// log is `<safe name>.log` beside it (the driver maps every char outside `[A-Za-z0-9_.-]`
/// to `_`). A section the driver counts as passing (success, skipped, or
/// `continue_on_error`) is not read.
pub fn report_results(results: &Path) -> Result<String, String> {
    let text =
        std::fs::read_to_string(results).map_err(|e| format!("{}: {e}", results.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", results.display()))?;
    let obj = v
        .as_object()
        .ok_or_else(|| format!("{}: not an object", results.display()))?;
    let dir = results.parent().unwrap_or_else(|| Path::new("."));
    let mut out = String::new();
    for (name, r) in obj {
        let result = r.get("result").and_then(|x| x.as_str()).unwrap_or("");
        let coe = r
            .get("continue_on_error")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        if matches!(result, "success" | "skipped") || coe {
            continue;
        }
        let safe: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || "_.-".contains(c) {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        out.push_str(&line_for(
            name,
            &first_failing_line(&dir.join(format!("{safe}.log"))),
        ));
    }
    Ok(out)
}

fn line_for(section: &str, line: &str) -> String {
    if line.is_empty() {
        return String::new();
    }
    let cut: String = line.chars().take(300).collect();
    format!("::error::section {section} first failing line: {cut}\n")
}

/// `::error::section <stem> first failing line: <line>` (line cut to 300 chars) for every
/// log that has one; logs with none print nothing.
pub fn report(logs: &[std::path::PathBuf]) -> String {
    let mut out = String::new();
    for log in logs {
        let line = first_failing_line(log);
        if line.is_empty() {
            continue;
        }
        let stem = log
            .file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        let cut: String = line.chars().take(300).collect();
        out.push_str(&format!(
            "::error::section {stem} first failing line: {cut}\n"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn log(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).expect("write log");
        p
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sff-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("mkdir");
        d
    }

    #[test]
    fn sccache_exit_2_is_named_a_killed_rustc_not_a_compile_error() {
        let d = tmp("k");
        let p = log(
            &d,
            "s.log",
            "warning: x\nerror: could not compile `aprender-core` (lib test); 3 warnings emitted\n\nCaused by:\n  process didn't exit successfully: `/usr/local/cargo/bin/sccache rustc --crate-name aprender` (exit status: 2)\n",
        );
        let got = first_failing_line(&p);
        assert!(got.starts_with("rustc killed by signal"), "{got}");
        assert!(got.contains("aprender-core"), "{got}");
    }

    #[test]
    fn a_real_error_before_the_exit_2_line_stays_the_first_failing_line() {
        let d = tmp("k2");
        let p = log(
            &d,
            "s.log",
            "error[E0425]: cannot find value `x`\nerror: could not compile `a` (lib test)\n\nCaused by:\n  process didn't exit successfully: `sccache rustc --crate-name a` (exit status: 2)\n",
        );
        assert_eq!(
            first_failing_line(&p),
            "error[E0425]: cannot find value `x`"
        );
        let q = log(
            &d,
            "t.log",
            "thread 'x' panicked at a.rs:1\nerror: could not compile `a` (lib)\n  process didn't exit successfully: `sccache rustc` (exit status: 2)\n",
        );
        assert_eq!(first_failing_line(&q), "thread 'x' panicked at a.rs:1");
    }

    #[test]
    fn plain_rustc_exit_1_stays_a_compile_error() {
        let d = tmp("k1");
        let p = log(
            &d,
            "s.log",
            "error: could not compile `a` (lib)\n\nCaused by:\n  process didn't exit successfully: `sccache rustc --crate-name a` (exit status: 1)\n",
        );
        assert_eq!(first_failing_line(&p), "error: could not compile `a` (lib)");
    }

    #[test]
    fn first_error_line_wins() {
        let d = tmp("a");
        let p = log(&d, "s.log", "ok\n::error::boom one\n::error::boom two\n");
        assert_eq!(first_failing_line(&p), "::error::boom one");
    }

    #[test]
    fn panic_without_error_annotation_counts() {
        let d = tmp("b");
        let p = log(&d, "s.log", "ok\nthread 'x' panicked at src/a.rs:1\n");
        assert_eq!(first_failing_line(&p), "thread 'x' panicked at src/a.rs:1");
    }

    #[test]
    fn error_annotation_beats_an_earlier_panic() {
        let d = tmp("c");
        let p = log(
            &d,
            "s.log",
            "thread 'x' panicked at a.rs:1\n::error::later\n",
        );
        assert_eq!(first_failing_line(&p), "::error::later");
    }

    #[test]
    fn results_mode_reads_only_blocking_failed_sections() {
        let d = tmp("f");
        log(&d, "bad.log", "::error::bad cause\n");
        log(&d, "good.log", "::error::noise in a green section\n");
        log(
            &d,
            "lab.log",
            "::error::noise in a continue-on-error section\n",
        );
        log(&d, "determinism_X64_.log", "error: link failed\n");
        let results = log(
            &d,
            "results.json",
            r#"{"bad":{"result":"failure"},"good":{"result":"success"},
               "lab":{"result":"failure","continue_on_error":true},
               "determinism[X64]":{"result":"cancelled"}}"#,
        );
        let out = report_results(&results).expect("results parse");
        assert_eq!(
            out,
            "::error::section bad first failing line: ::error::bad cause\n\
             ::error::section determinism[X64] first failing line: error: link failed\n"
        );
    }

    #[test]
    fn results_mode_rejects_unreadable_results() {
        let d = tmp("g");
        assert!(report_results(&d.join("absent.json")).is_err());
        assert!(report_results(&log(&d, "r.json", "not json")).is_err());
    }

    #[test]
    fn clean_and_missing_logs_have_none() {
        let d = tmp("d");
        assert_eq!(first_failing_line(&log(&d, "s.log", "ok\nfine\n")), "");
        assert_eq!(first_failing_line(&d.join("absent.log")), "");
    }

    #[test]
    fn report_names_the_section_by_log_stem_and_cuts_long_lines() {
        let d = tmp("e");
        let long = format!("::error::{}", "x".repeat(400));
        let a = log(&d, "guard-cargo.log", &format!("{long}\n"));
        let b = log(&d, "guard-tree.log", "ok\n");
        let out = report(&[a, b]);
        let want = format!(
            "::error::section guard-cargo first failing line: {}\n",
            &long[..300]
        );
        assert_eq!(out, want);
    }
}
