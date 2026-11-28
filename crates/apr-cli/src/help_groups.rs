//! Grouped `apr --help` (U-08, paiml/aprender#7485).
//!
//! The top-level help used to be one flat list of about 115 verbs, 32 of them
//! `*-lint`, with ticket ids (`GH-243`, `PMAT-741`, `CRUX-G-05`, `#3856`) in
//! the one-line summaries. This module renders the same root command with its
//! verbs under headings, and strips the ticket ids from the summaries shown in
//! that root list and its options. No verb is renamed or moved: the names and
//! the dispatch are the clap derive's. Per-verb `--help` text is unchanged.

use clap::{Command, CommandFactory};

use crate::Cli;

/// Headings in the order they print. `Models` opens the list.
const ORDER: &[&str] = &[
    "Models",
    "Lint checks",
    "Train and tune",
    "Inspect and debug",
    "Serve and tools",
    "Other",
];

/// Verbs placed under a named heading. A verb that is in none of these lists
/// goes under `Other`, and every `*-lint` verb goes under `Lint checks`.
const MEMBERS: &[(&str, &[&str])] = &[
    (
        "Models",
        &[
            "run",
            "chat",
            "serve",
            "pull",
            "list",
            "rm",
            "import",
            "export",
            "convert",
            "stamp",
            "compile",
            "merge",
            "quantize",
            "prune",
            "distill",
            "finetune",
            "shard",
            "unshard",
            "compare-hf",
            "registry",
            "manifest",
            "encrypt",
            "decrypt",
            "publish",
        ],
    ),
    (
        "Train and tune",
        &[
            "train",
            "pretrain",
            "tune",
            "tokenize",
            "data",
            "pipeline",
            "runs",
            "experiment",
            "monitor",
            "diagnose",
            "dataset",
            "kernel",
        ],
    ),
    (
        "Inspect and debug",
        &[
            "inspect",
            "debug",
            "trace",
            "explain",
            "check",
            "tensors",
            "validate",
            "diff",
            "tree",
            "flow",
            "hex",
            "qa",
            "qualify",
            "profile",
            "bench",
            "eval",
            "ppl",
            "parity",
            "cbtop",
            "gpu",
            "devices",
            "capability",
            "oracle",
            "ptx",
            "ptx-map",
            "ptx-debug",
            "cgp",
            "pv",
            "sim",
            "canary",
            "test",
            "validate-manifest",
        ],
    ),
    (
        "Serve and tools",
        &[
            "mcp",
            "tui",
            "code",
            "embed",
            "rerank",
            "rag",
            "zram",
            "showcase",
            "plot",
            "modelfile",
            "beat-run",
            "rosetta",
        ],
    ),
];

/// The heading a verb prints under.
fn heading_for(name: &str) -> &'static str {
    if name == "lint" || name.ends_with("-lint") {
        return "Lint checks";
    }
    for (heading, members) in MEMBERS {
        if members.contains(&name) {
            return heading;
        }
    }
    "Other"
}

/// Number of leading ASCII digits in `s`.
fn digits_len(s: &str) -> usize {
    s.bytes().take_while(u8::is_ascii_digit).count()
}

/// `PREFIX<digits>` at the start of `s`: its length, when there is at least one digit.
fn prefixed_digits(s: &str, prefix: &str) -> Option<usize> {
    let n = digits_len(s.strip_prefix(prefix)?);
    (n > 0).then_some(prefix.len() + n)
}

/// `CRUX-<A-Z>-<digits>` at the start of `s`.
fn crux_len(s: &str) -> Option<usize> {
    let rest = s.strip_prefix("CRUX-")?;
    let letter = *rest.as_bytes().first()?;
    if !letter.is_ascii_uppercase() || rest.as_bytes().get(1) != Some(&b'-') {
        return None;
    }
    let n = digits_len(&rest[2..]);
    (n > 0).then_some("CRUX-".len() + 2 + n)
}

/// `#<digits>` at the start of `s`, with at least three digits.
fn hash_len(s: &str) -> Option<usize> {
    let n = digits_len(s.strip_prefix('#')?);
    (n >= 3).then_some(1 + n)
}

/// Length of a ticket id at the start of `s`, or None. The ids are the
/// issue's own rule: `GH-[0-9]+`, `PMAT-[0-9]+`, `CRUX-[A-Z]-[0-9]+`, `#[0-9]{3,}`.
fn ticket_len(s: &str) -> Option<usize> {
    prefixed_digits(s, "GH-")
        .or_else(|| prefixed_digits(s, "PMAT-"))
        .or_else(|| crux_len(s))
        .or_else(|| hash_len(s))
}

/// Drops every `(...)` group whose content is only separators (spaces, commas,
/// colons), so `(GH-1, PMAT-2)` once its ids are gone leaves nothing behind.
fn drop_empty_groups(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find('(') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find(')') {
            Some(close)
                if after[..close]
                    .chars()
                    .all(|c| c == ' ' || c == ',' || c == ':') =>
            {
                rest = &after[close + 1..];
            }
            _ => {
                out.push('(');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// The summary with every ticket id removed, and the brackets they leave empty
/// (`(GH-243)` -> nothing, `(GH-243, ALB-011)` -> `(ALB-011)`) tidied away.
/// A `:` that followed a removed id (`(PMAT-237: use ...)`) goes with it.
pub fn strip_ticket_ids(summary: &str) -> String {
    let mut out = String::with_capacity(summary.len());
    let mut i = 0;
    while i < summary.len() {
        let rest = &summary[i..];
        if let Some(n) = ticket_len(rest) {
            i += n;
            let after = summary[i..].trim_start_matches(' ');
            if let Some(stripped) = after.strip_prefix(':') {
                i = summary.len() - stripped.trim_start_matches(' ').len();
            }
            continue;
        }
        let Some(c) = rest.chars().next() else { break };
        out.push(c);
        i += c.len_utf8();
    }
    let mut t = drop_empty_groups(&out)
        .replace("(, ", "(")
        .replace(", )", ")")
        .replace(" ,", ",");
    while t.contains("  ") {
        t = t.replace("  ", " ");
    }
    t.trim_end().to_string()
}

/// The root `apr` command with the grouped help template. The verbs, their
/// arguments and their dispatch come from the same clap derive as `Cli`.
pub fn grouped_command() -> Command {
    // Clap adds the built-in `help` subcommand and copies the global args into
    // every subcommand during build(). Build first, so each verb's `--help`
    // keeps the original text, and the rows below can list `help`.
    let mut root = Cli::command();
    root.build();
    // Only the root's own copies are stripped, so `apr --help` shows no ticket
    // ids while `apr <verb> --help` is byte-identical to before.
    let root = root.mut_args(|a| {
        let stripped = a.get_help().map(|h| strip_ticket_ids(&h.to_string()));
        match stripped {
            Some(text) => a.help(text),
            None => a,
        }
    });

    // One row per visible verb: (heading, name, summary without ticket ids).
    let mut rows: Vec<(&'static str, String, String)> = Vec::new();
    for sub in root.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }
        let name = sub.get_name().to_string();
        let about = sub.get_about().map(ToString::to_string).unwrap_or_default();
        let first = about.lines().next().unwrap_or("").trim();
        // Braces are template syntax to clap; a summary must not carry them.
        let summary = strip_ticket_ids(first).replace(['{', '}'], "");
        rows.push((heading_for(&name), name, summary));
    }

    // Inside a heading, curated verbs come first in MEMBERS order; the rest keep
    // clap's declaration order (sort_by_key is stable).
    rows.sort_by_key(|r| {
        MEMBERS
            .iter()
            .find(|(h, _)| *h == r.0)
            .and_then(|(_, m)| m.iter().position(|n| *n == r.1))
            .unwrap_or(usize::MAX)
    });

    let width = rows.iter().map(|r| r.1.len()).max().unwrap_or(0) + 2;
    let mut template = String::from("{about-with-newline}\n{usage-heading} {usage}\n\nCommands:\n");
    let mut first_heading = true;
    for heading in ORDER {
        let members: Vec<&(&'static str, String, String)> =
            rows.iter().filter(|r| r.0 == *heading).collect();
        if members.is_empty() {
            continue;
        }
        if !first_heading {
            template.push('\n');
        }
        first_heading = false;
        template.push_str(heading);
        template.push_str(":\n");
        for (_, name, summary) in members {
            template.push_str(&format!("  {name:<width$}{summary}\n"));
        }
    }
    // {all-args} would also print the subcommand list again. clap prints the option
    // rows without their heading, so the heading is written here. The root has no
    // positional arguments; `command` is the subcommand field.
    template.push_str("\nOptions:\n{options}");
    root.help_template(template)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_removes_each_ticket_form() {
        assert_eq!(
            strip_ticket_ids("Quantize model weights (GH-243)"),
            "Quantize model weights"
        );
        assert_eq!(
            strip_ticket_ids("ML tuning (GH-176, SPEC-TUNE-2026-001)"),
            "ML tuning (SPEC-TUNE-2026-001)"
        );
        assert_eq!(strip_ticket_ids("Tune (PMAT-741)"), "Tune");
        assert_eq!(strip_ticket_ids("Shard (CRUX-B-05)"), "Shard");
        assert_eq!(strip_ticket_ids("Pinned (#3856)"), "Pinned");
    }

    #[test]
    fn strip_leaves_no_orphan_colon_or_empty_parens() {
        assert_eq!(
            strip_ticket_ids(
                "Skip tensor contract validation (PMAT-237: use with diagnostic tooling)"
            ),
            "Skip tensor contract validation (use with diagnostic tooling)"
        );
        assert_eq!(strip_ticket_ids("Two ids (GH-1, PMAT-2)"), "Two ids");
        assert_eq!(
            strip_ticket_ids("Two ids (GH-1, PMAT-2) left"),
            "Two ids left"
        );
        assert!(!strip_ticket_ids("A (PMAT-237: x)").contains("(:"));
    }

    #[test]
    fn strip_keeps_text_that_is_not_a_ticket() {
        assert_eq!(
            strip_ticket_ids("Rosetta (PMAT-ROSETTA-001)"),
            "Rosetta (PMAT-ROSETTA-001)"
        );
        assert_eq!(
            strip_ticket_ids("Model size is 12 GB"),
            "Model size is 12 GB"
        );
        assert_eq!(
            strip_ticket_ids("Two-digit #12 stays"),
            "Two-digit #12 stays"
        );
    }

    #[test]
    fn lint_verbs_are_all_under_one_heading() {
        assert_eq!(heading_for("awq-lint"), "Lint checks");
        assert_eq!(heading_for("lint"), "Lint checks");
        assert_eq!(heading_for("brand-new-lint"), "Lint checks");
    }

    #[test]
    fn unlisted_verbs_fall_to_other() {
        assert_eq!(heading_for("help"), "Other");
        assert_eq!(heading_for("run"), "Models");
    }
}
