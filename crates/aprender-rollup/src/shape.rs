//! `aprender-rollup shape`: is this file a roadmap fragment the aggregate can take?
//!
//! Port of the Python `fragment_shape` (C301: the build is Rust only). A fragment is ONE mapping,
//! filed under its own id, that carries nothing the aggregate shares or the splitter misreads.
//! This part holds the text-level checks; the structure walk and the aggregate check follow.
//! Exit 0 ok, 1 refused (reason on stdout), 2 not measured (unreadable file).

use std::path::{Path, PathBuf};

/// `str.splitlines` breaks on these besides `\n` and `\r`; the splitter and the parser disagree on them.
const SEPARATORS: &str = "\x0b\x0c\x1c\x1d\x1e\u{85}\u{2028}\u{2029}";

/// Why the file is not a fragment, `Ok(None)` when it is. `Err` = not measured (exit 2).
/// `_roadmap` is read by the aggregate check in a later part.
pub fn judge(file: &Path, _roadmap: Option<&Path>) -> Result<Option<String>, String> {
    let name = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some(stem) = name.strip_suffix(".yaml") else {
        return Ok(Some("is not named <id>.yaml".into()));
    };
    let bytes = std::fs::read(file).map_err(|e| format!("cannot read {} ({e})", file.display()))?;
    let text = match String::from_utf8(bytes) {
        Ok(t) => t,
        Err(e) => {
            return Ok(Some(format!(
                "is not valid UTF-8 (byte {})",
                e.utf8_error().valid_up_to()
            )))
        }
    };
    Ok(text_problem(&text, stem))
}

fn is_marker(line: &str) -> bool {
    ["---", "..."].iter().any(|m| {
        line.strip_prefix(m)
            .is_some_and(|rest| rest.chars().next().is_none_or(char::is_whitespace))
    })
}

/// The first text-level defect, in the order the Python table listed them.
fn text_problem(text: &str, stem: &str) -> Option<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let cuts = lines.iter().filter(|l| l.starts_with("- id:")).count();
    let checks: [(bool, String); 6] = [
        (
            text.contains('\u{feff}'),
            "carries a byte order mark".into(),
        ),
        (
            text.contains('\r'),
            "carries a carriage return (the splitter and the parser read different lines)".into(),
        ),
        (
            text.chars().any(|c| SEPARATORS.contains(c)),
            "carries a Unicode line separator (the splitter and the parser read different lines)"
                .into(),
        ),
        (
            lines.iter().any(|l| is_marker(l)),
            "carries a '---' or '...' document marker line".into(),
        ),
        (
            cuts != 1,
            format!("has {cuts} lines opening '- id:', want exactly 1"),
        ),
        (
            lines[0] != format!("- id: {stem}"),
            format!("does not start with '- id: {stem}'"),
        ),
    ];
    checks.into_iter().find(|(bad, _)| *bad).map(|(_, why)| why)
}

/// `shape <fragment> [--roadmap <path>]`; anything else (a bare `--roadmap`, an extra word) is
/// not a measurement, so the caller exits 2.
pub fn parse_args(args: &[String]) -> Result<(PathBuf, Option<PathBuf>), String> {
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["shape", file] => Ok((PathBuf::from(file), None)),
        ["shape", file, "--roadmap", path] => Ok((PathBuf::from(file), Some(PathBuf::from(path)))),
        _ => Err("expected: shape <fragment> [--roadmap <path>]".into()),
    }
}

#[cfg(test)]
mod tests;
