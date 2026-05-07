//! REPL inline directives (Claude-Code parity Phase 2):
//! `!<cmd>` shell prefix and `@<path>` file expansion
//! (PMAT-CODE-REPL-PHASE2-001).
//!
//! Both directives mirror Claude Code's interactive shortcuts. They run
//! BEFORE slash-command parsing in `read_input`:
//!
//! * `!<cmd>` — anything after a leading `!` is treated as a shell
//!   command. Output is printed inline; the agent loop is **not**
//!   invoked. Useful for quick `!ls`, `!git status`, etc.
//!
//! * `@<path>` — a token of the form `@./README.md` (or any
//!   non-whitespace path after `@`) is expanded INLINE in the prompt
//!   text: the token is replaced with the file's contents wrapped in
//!   `<file path="...">...</file>` so the model sees both the path
//!   and the contents. Multiple `@` tokens per prompt expand left-to-
//!   right. Missing files are reported on stderr and the token is left
//!   verbatim (Poka-Yoke: no silent partial expansion).
//!
//! Pure-function design — terminal I/O lives in `repl.rs`, this module
//! only does string parsing and filesystem reads. That keeps the parser
//! headlessly testable (no TTY needed).

use std::io;
use std::path::Path;

/// If `input` starts with `!`, return the trimmed shell command after
/// the `!` prefix. Returns `None` for non-bang inputs (the regular
/// agent loop should handle them).
///
/// `!` alone (no command) returns `None` — there's nothing to run.
pub fn parse_bang_command(input: &str) -> Option<&str> {
    let trimmed = input.trim_start();
    let after_bang = trimmed.strip_prefix('!')?;
    let cmd = after_bang.trim();
    if cmd.is_empty() {
        None
    } else {
        Some(cmd)
    }
}

/// Run `cmd` via the system shell (`sh -c "<cmd>"`) and return
/// (exit_code, captured_output). `output` interleaves stdout+stderr
/// the way the user would see them in a terminal.
///
/// Failure to spawn (e.g. `sh` missing) returns `Err`. Non-zero shell
/// exit codes are NOT errors at this layer — they're returned in the
/// tuple so the REPL can echo them.
pub fn execute_bang_command(cmd: &str) -> io::Result<(i32, String)> {
    let output = std::process::Command::new("sh").arg("-c").arg(cmd).output()?;
    let mut buf = String::new();
    if !output.stdout.is_empty() {
        buf.push_str(&String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        if !buf.is_empty() && !buf.ends_with('\n') {
            buf.push('\n');
        }
        buf.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    let code = output.status.code().unwrap_or(-1);
    Ok((code, buf))
}

/// Find every `@<path>` token in `input`. A path token is `@` followed
/// by one or more non-whitespace characters; the `@` must be at the
/// start of input or preceded by whitespace (so `email@host` is NOT
/// matched).
///
/// Returns each (start_byte_offset, end_byte_offset, path_string)
/// triple in order so callers can splice without re-scanning.
pub fn find_at_path_tokens(input: &str) -> Vec<(usize, usize, String)> {
    let bytes = input.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'@' {
            // boundary: either start-of-input or previous byte is whitespace
            let at_boundary = i == 0 || bytes[i - 1].is_ascii_whitespace();
            if at_boundary {
                let start = i;
                let mut end = i + 1; // skip '@'
                while end < bytes.len() && !bytes[end].is_ascii_whitespace() {
                    end += 1;
                }
                let path = &input[start + 1..end];
                if !path.is_empty() {
                    out.push((start, end, path.to_owned()));
                }
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Replace every `@<path>` token in `input` with the file's contents
/// wrapped in `<file path="...">...</file>`. Missing or unreadable
/// files have their token left verbatim AND a warning is appended to
/// `warnings`. This way a typo'd `@/nope` makes it through to the
/// agent verbatim (the agent can complain) and the operator gets a
/// stderr warning.
///
/// Returns the expanded prompt text.
pub fn expand_at_paths(input: &str, warnings: &mut Vec<String>) -> String {
    let tokens = find_at_path_tokens(input);
    if tokens.is_empty() {
        return input.to_owned();
    }
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0;
    for (start, end, path) in &tokens {
        out.push_str(&input[cursor..*start]);
        let p = Path::new(path);
        match std::fs::read_to_string(p) {
            Ok(body) => {
                out.push_str(&format!("<file path=\"{path}\">\n{body}\n</file>"));
            }
            Err(e) => {
                warnings.push(format!("@{path}: {e}"));
                out.push_str(&input[*start..*end]);
            }
        }
        cursor = *end;
    }
    out.push_str(&input[cursor..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ── parse_bang_command ─────────────────────────────────────────

    #[test]
    fn bang_simple_command() {
        assert_eq!(parse_bang_command("!ls"), Some("ls"));
    }

    #[test]
    fn bang_with_args() {
        assert_eq!(parse_bang_command("!ls -la"), Some("ls -la"));
    }

    #[test]
    fn bang_strips_inner_padding() {
        assert_eq!(parse_bang_command("!   ls -la"), Some("ls -la"));
    }

    #[test]
    fn bang_strips_leading_whitespace_before_bang() {
        // Common typo — trailing newline from read_line is already stripped
        // by the caller, but leading spaces before `!` are charitable.
        assert_eq!(parse_bang_command("  !ls"), Some("ls"));
    }

    #[test]
    fn no_bang_returns_none() {
        assert_eq!(parse_bang_command("ls"), None);
        assert_eq!(parse_bang_command("hello !ls"), None, "bang must lead the line");
    }

    #[test]
    fn bare_bang_returns_none() {
        // `!` alone is not a valid command — no shell to run.
        assert_eq!(parse_bang_command("!"), None);
        assert_eq!(parse_bang_command("!  "), None);
    }

    // ── execute_bang_command (LIVE shell) ──────────────────────────

    #[test]
    fn exec_bang_echoes_string() {
        let (code, out) = execute_bang_command("echo hello-bang-test").expect("exec");
        assert_eq!(code, 0);
        assert!(out.contains("hello-bang-test"), "stdout missing: {out:?}");
    }

    #[test]
    fn exec_bang_returns_nonzero_on_false() {
        let (code, _out) = execute_bang_command("false").expect("exec");
        assert_ne!(code, 0, "false must return nonzero");
    }

    #[test]
    fn exec_bang_captures_stderr() {
        let (_code, out) =
            execute_bang_command("printf 'stdout-line\\n'; printf 'err-line\\n' 1>&2; true")
                .expect("exec");
        assert!(out.contains("stdout-line"));
        assert!(out.contains("err-line"), "stderr missing in {out:?}");
    }

    // ── find_at_path_tokens ────────────────────────────────────────

    #[test]
    fn at_at_start_of_line() {
        let tokens = find_at_path_tokens("@README.md");
        assert_eq!(tokens, vec![(0, 10, "README.md".to_string())]);
    }

    #[test]
    fn at_after_whitespace() {
        let tokens = find_at_path_tokens("look at @./foo.txt please");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].2, "./foo.txt");
    }

    #[test]
    fn email_not_matched() {
        // `noah@paiml.com` must NOT be parsed as `@paiml.com`.
        let tokens = find_at_path_tokens("ping noah@paiml.com");
        assert!(tokens.is_empty(), "email must not be at-path: {tokens:?}");
    }

    #[test]
    fn multiple_at_tokens() {
        let tokens = find_at_path_tokens("compare @./a.txt and @./b.txt");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].2, "./a.txt");
        assert_eq!(tokens[1].2, "./b.txt");
    }

    #[test]
    fn bare_at_yields_nothing() {
        // `@` followed by whitespace is not a path token.
        let tokens = find_at_path_tokens("hello @ world");
        assert!(tokens.is_empty());
    }

    // ── expand_at_paths ────────────────────────────────────────────

    #[test]
    fn expand_single_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("note.md");
        fs::write(&p, "Hello, world").expect("write");
        let input = format!("read @{}", p.display());
        let mut warns = Vec::new();
        let out = expand_at_paths(&input, &mut warns);
        assert!(out.contains("<file path="));
        assert!(out.contains("Hello, world"));
        assert!(out.contains("</file>"));
        assert!(warns.is_empty());
    }

    #[test]
    fn expand_missing_file_keeps_token_and_warns() {
        let mut warns = Vec::new();
        let out = expand_at_paths("look @/no/such/path here", &mut warns);
        // Token kept verbatim
        assert!(out.contains("@/no/such/path"));
        // Warning issued
        assert_eq!(warns.len(), 1);
        assert!(warns[0].contains("/no/such/path"));
    }

    #[test]
    fn expand_no_at_returns_input_unchanged() {
        let mut warns = Vec::new();
        let out = expand_at_paths("plain text no at-tokens", &mut warns);
        assert_eq!(out, "plain text no at-tokens");
        assert!(warns.is_empty());
    }

    #[test]
    fn expand_two_files_in_one_prompt() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a.md");
        let b = dir.path().join("b.md");
        fs::write(&a, "AAA").expect("write");
        fs::write(&b, "BBB").expect("write");
        let input = format!("@{} and @{}", a.display(), b.display());
        let mut warns = Vec::new();
        let out = expand_at_paths(&input, &mut warns);
        assert!(out.contains("AAA"));
        assert!(out.contains("BBB"));
        assert!(out.contains(" and "));
        assert!(warns.is_empty());
    }

    #[test]
    fn expand_email_unaffected() {
        let mut warns = Vec::new();
        let out = expand_at_paths("ping noah@paiml.com", &mut warns);
        assert_eq!(out, "ping noah@paiml.com");
        assert!(warns.is_empty());
    }
}
