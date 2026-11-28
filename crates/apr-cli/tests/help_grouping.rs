//! U-08 (paiml/aprender#7485): `apr --help` is grouped under headings and shows
//! no ticket ids. Written before the grouping code, so it fails on main first.
//!
//! Ticket-id rule (the issue's done_when): no line of `apr --help` matches
//! `GH-[0-9]+|PMAT-[0-9]+|CRUX-[A-Z]-[0-9]+|#[0-9]{3,}`.
//! Grouping: the first heading holds `run chat serve pull list rm`, and every
//! `*-lint` verb sits under one heading. No verb is renamed or moved.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use assert_cmd::Command;
use regex::Regex;

/// The `apr --help` text with colour off, as the user sees it in a non-TTY.
fn help_text() -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_apr"))
        .arg("--help")
        .env("NO_COLOR", "1")
        .output()
        .expect("run apr --help");
    assert!(out.status.success(), "apr --help exited {:?}", out.status);
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A heading inside the Commands block: a non-indented line ending in `:`.
fn heading(line: &str) -> Option<&str> {
    if line.starts_with(' ') {
        return None;
    }
    line.strip_suffix(':')
}

/// The command name on a command row: two spaces of indent, then the name.
/// A wrapped description continuation has four spaces and yields None.
fn command_row(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("  ")?;
    if rest.starts_with(' ') {
        return None;
    }
    rest.split_whitespace().next()
}

/// Headings inside the `Commands:` block, each with the command names listed
/// under it, in the order they print.
fn command_groups(text: &str) -> Vec<(String, Vec<String>)> {
    let body = text
        .split("\nCommands:\n")
        .nth(1)
        .unwrap_or("")
        .split("\nOptions:")
        .next()
        .unwrap_or("")
        .split("\nArguments:")
        .next()
        .unwrap_or("");
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    for line in body.lines().filter(|l| !l.is_empty()) {
        if let Some(h) = heading(line) {
            groups.push((h.to_string(), Vec::new()));
        } else if let (Some(name), Some((_, names))) = (command_row(line), groups.last_mut()) {
            names.push(name.to_string());
        }
    }
    groups
}

#[test]
fn help_shows_no_ticket_ids() {
    let ticket = Regex::new(r"GH-[0-9]+|PMAT-[0-9]+|CRUX-[A-Z]-[0-9]+|#[0-9]{3,}").unwrap();
    let text = help_text();
    let hits: Vec<&str> = text.lines().filter(|l| ticket.is_match(l)).collect();
    assert!(
        hits.is_empty(),
        "apr --help still shows ticket ids on {} line(s):\n{}",
        hits.len(),
        hits.join("\n")
    );
}

#[test]
fn help_opens_with_the_model_verbs_under_one_heading() {
    let groups = command_groups(&help_text());
    let (first, names) = groups
        .first()
        .expect("apr --help has no heading inside Commands:");
    assert_eq!(first, "Models", "the first heading must be Models");
    let head: Vec<&str> = names.iter().take(6).map(String::as_str).collect();
    assert_eq!(
        head,
        ["run", "chat", "serve", "pull", "list", "rm"],
        "the Models heading must open with run chat serve pull list rm"
    );
}

#[test]
fn every_lint_verb_sits_under_one_heading() {
    let groups = command_groups(&help_text());
    let lint_groups: Vec<&str> = groups
        .iter()
        .filter(|(_, names)| names.iter().any(|n| n.ends_with("-lint")))
        .map(|(h, _)| h.as_str())
        .collect();
    assert_eq!(
        lint_groups.len(),
        1,
        "*-lint verbs are spread over headings {lint_groups:?}, want one"
    );
    let lint_names: Vec<&str> = groups
        .iter()
        .find(|(h, _)| h == lint_groups[0])
        .map(|(_, n)| n.iter().map(String::as_str).collect())
        .unwrap();
    let all_lint = groups
        .iter()
        .flat_map(|(_, n)| n.iter())
        .filter(|n| n.ends_with("-lint"))
        .count();
    assert_eq!(
        lint_names.iter().filter(|n| n.ends_with("-lint")).count(),
        all_lint,
        "a *-lint verb is listed outside the lint heading"
    );
    assert!(
        all_lint > 30,
        "expected the full lint set, found {all_lint}"
    );
}

#[test]
fn no_verb_is_renamed_or_moved_rm_gc_lint_still_runs() {
    let out = Command::new(env!("CARGO_BIN_EXE_apr"))
        .args(["rm-gc-lint", "--help"])
        .env("NO_COLOR", "1")
        .output()
        .expect("run apr rm-gc-lint --help");
    assert!(
        out.status.success(),
        "apr rm-gc-lint --help exited {:?}",
        out.status
    );
}
