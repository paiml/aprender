//! "Would the SHIPPED `apr quantize` accept this argv?" — asked of the real
//! clap parser, never of a second implementation.
//!
//! # Why this module exists (aprender#2377 finding 2)
//!
//! `gptq_lint` and `awq_lint` each ran a "CLI flag accepted" gate whose verdict
//! came from a hand-rolled `while i < argv.len()` matcher living next to the
//! assertion (`parse_gptq_flags` / `parse_awq_flags` + `validate_*_flags`).
//! Those matchers understood `--method`, `--bits` and `--group-size`. The
//! shipped `apr quantize` (`Commands::Quantize` in `commands_enum.rs`) takes
//! `--scheme`/`-s`, `--output`/`-o`, `--format`, `--batch`, `--plan`, `--force`
//! over a required `<FILE>` positional. **Not one flag was shared**, and neither
//! `gptq` nor `awq` is an accepted `--scheme` value. The gate was green for
//! years while validating a CLI that does not exist.
//!
//! A gate may not own a parser. This module answers the question by handing the
//! argv to `Cli::command()` — the very `clap::Command` the binary parses with —
//! via `try_get_matches_from`, so the gate cannot drift from what ships: adding,
//! renaming or removing a `quantize` flag changes this verdict in the same
//! commit that changes the CLI.
//!
//! # Observation vocabulary
//!
//! `flags.expected_outcome` is now `accepted` | `rejected` (`ok` is kept as a
//! spelling of `accepted`, which is what the rest of the `*-lint` family uses).
//! The pre-fix labels (`missing_method`, `wrong_method`, `unknown_method`,
//! `invalid_bits`, `missing_bits`, `invalid_group_size`) name classifications
//! the real parser cannot produce, so they are refused as an unusable
//! observation (exit 4, "your capture step is broken") rather than quietly
//! folded into "rejected" — an old capture that passed for a brand-new reason
//! is the failure this whole change exists to remove.

use std::sync::OnceLock;

use clap::CommandFactory;
use serde_json::Value;

use crate::Cli;

/// `flags.expected_outcome` spelling for "the shipped parser takes this argv".
pub const EXPECTED_ACCEPTED: &str = "accepted";
/// Legacy spelling of [`EXPECTED_ACCEPTED`], used across the `*-lint` family.
pub const EXPECTED_ACCEPTED_ALIAS: &str = "ok";
/// `flags.expected_outcome` spelling for "the shipped parser refuses this argv".
pub const EXPECTED_REJECTED: &str = "rejected";

/// What the shipped `apr quantize` parser did with a recorded argv.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizeArgvVerdict {
    /// `apr quantize <argv>` parses.
    Accepted,
    /// `apr quantize <argv>` is refused; `reason` is the parser's own first
    /// diagnostic line (e.g. `unexpected argument '--method' found`).
    Rejected { reason: String },
}

impl QuantizeArgvVerdict {
    /// The word this verdict is reported and compared as.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Accepted => EXPECTED_ACCEPTED,
            Self::Rejected { .. } => EXPECTED_REJECTED,
        }
    }

    #[must_use]
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }
}

/// Clap's parser for `apr`'s ~103 subcommands needs more stack than a default
/// thread has in debug builds — `parsing.rs::parse_cli` spawns a 16 MB thread
/// for exactly this reason. Every entry into the shipped parser goes through
/// here so a lint gate is safe to call from a test thread too.
fn on_big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(f)
        .expect("spawn shipped-CLI parser thread")
        .join()
        .expect("shipped-CLI parser thread must not panic")
}

/// Prefix the recorded argv with the binary and subcommand the parser expects.
///
/// Observations record the argv *after* `apr quantize`; a capture that also
/// recorded the literal `quantize` is tolerated rather than double-prefixed.
fn normalized_argv(argv: &[&str]) -> Vec<String> {
    let mut full = Vec::with_capacity(argv.len() + 2);
    full.push("apr".to_string());
    if argv.first().copied() != Some("quantize") {
        full.push("quantize".to_string());
    }
    full.extend(argv.iter().map(|s| (*s).to_string()));
    full
}

/// The shipped parser's first diagnostic line, without clap's `error:` stamp.
fn clap_reason(e: &clap::Error) -> String {
    use clap::error::ErrorKind;
    match e.kind() {
        ErrorKind::DisplayHelp | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => {
            "argv is a help request, not an `apr quantize` invocation".to_string()
        }
        ErrorKind::DisplayVersion => {
            "argv is a version request, not an `apr quantize` invocation".to_string()
        }
        _ => first_diagnostic_line(&e.to_string()),
    }
}

fn first_diagnostic_line(rendered: &str) -> String {
    rendered
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(|l| l.trim_start_matches("error:").trim().to_string())
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| "refused by the shipped parser".to_string())
}

/// Ask the shipped `apr` clap parser whether `apr quantize <argv>` is accepted.
///
/// This is the only acceptance oracle the lint gates may use.
#[must_use]
pub fn shipped_quantize_verdict(argv: &[&str]) -> QuantizeArgvVerdict {
    let owned = normalized_argv(argv);
    on_big_stack(move || match Cli::command().try_get_matches_from(&owned) {
        Ok(_) => QuantizeArgvVerdict::Accepted,
        Err(e) => QuantizeArgvVerdict::Rejected {
            reason: clap_reason(&e),
        },
    })
}

fn arg_value_name(a: &clap::Arg) -> String {
    a.get_value_names()
        .and_then(<[clap::builder::Str]>::first)
        .map_or_else(|| a.get_id().as_str().to_uppercase(), ToString::to_string)
}

fn arg_takes_value(a: &clap::Arg) -> bool {
    use clap::ArgAction;
    !matches!(
        a.get_action(),
        ArgAction::SetTrue | ArgAction::SetFalse | ArgAction::Count | ArgAction::Help
    )
}

fn render_arg(a: &clap::Arg) -> String {
    if a.is_positional() {
        return format!("<{}>", arg_value_name(a));
    }
    let mut s = String::new();
    if let Some(l) = a.get_long() {
        s.push_str("--");
        s.push_str(l);
    }
    if let Some(c) = a.get_short() {
        s.push('/');
        s.push('-');
        s.push(c);
    }
    if arg_takes_value(a) {
        s.push_str(&format!(" <{}>", arg_value_name(a)));
    }
    s
}

fn build_quantize_summary() -> String {
    let root = Cli::command();
    let Some(q) = root.find_subcommand("quantize") else {
        return "(the shipped CLI has no `quantize` subcommand)".to_string();
    };
    let parts: Vec<String> = q
        .get_arguments()
        .filter(|a| !matches!(a.get_id().as_str(), "help" | "version"))
        .map(render_arg)
        .collect();
    if parts.is_empty() {
        return "(no arguments)".to_string();
    }
    parts.join(" ")
}

/// Every argument the shipped `apr quantize` actually accepts, read off the
/// same `clap::Command` the binary parses with — so an operator reading a
/// failure is told the truth and not a copy of it.
#[must_use]
pub fn shipped_quantize_accepts_summary() -> &'static str {
    static SUMMARY: OnceLock<String> = OnceLock::new();
    SUMMARY.get_or_init(|| on_big_stack(build_quantize_summary))
}

/// One flags-gate verdict, ready for a lint's `GateReport`.
#[derive(Debug, Clone)]
pub struct FlagParityGate {
    pub passed: bool,
    /// One-line summary for the gate report.
    pub outcome: String,
    /// Operator-facing explanation, present exactly when `!passed`.
    pub failure: Option<String>,
}

fn read_argv(v: &Value, falsify_id: &str) -> Result<Vec<String>, String> {
    let Some(items) = v.get("argv").and_then(Value::as_array) else {
        return Err(format!(
            "{falsify_id}: flags.argv is missing or is not an array — the gate has no argv to hand the shipped `apr quantize` parser"
        ));
    };
    items
        .iter()
        .map(|s| {
            s.as_str().map(ToString::to_string).ok_or_else(|| {
                format!("{falsify_id}: flags.argv contains a non-string element ({s})")
            })
        })
        .collect()
}

/// `true` = the observation claims the shipped parser accepts this argv.
fn read_expectation(v: &Value, falsify_id: &str) -> Result<bool, String> {
    let Some(raw) = v.get("expected_outcome").and_then(Value::as_str) else {
        return Err(format!(
            "{falsify_id}: flags.expected_outcome is missing — state `{EXPECTED_ACCEPTED}` or `{EXPECTED_REJECTED}`; a gate with an implied expectation asserts nothing"
        ));
    };
    match raw {
        EXPECTED_ACCEPTED | EXPECTED_ACCEPTED_ALIAS => Ok(true),
        EXPECTED_REJECTED => Ok(false),
        other => Err(format!(
            "{falsify_id}: flags.expected_outcome `{other}` is not a verdict of the shipped `apr quantize` parser. This gate now runs the real clap parser, whose only outcomes are `{EXPECTED_ACCEPTED}` (alias `{EXPECTED_ACCEPTED_ALIAS}`) and `{EXPECTED_REJECTED}`; the per-flag labels this observation was captured with described a parser that never shipped. Re-capture it."
        )),
    }
}

fn describe_argv(argv: &[String]) -> String {
    if argv.is_empty() {
        "(empty)".to_string()
    } else {
        argv.join(" ")
    }
}

fn failure_text(expected_accepted: bool, argv: &[String], verdict: &QuantizeArgvVerdict) -> String {
    let shown = describe_argv(argv);
    let accepts = shipped_quantize_accepts_summary();
    match verdict {
        QuantizeArgvVerdict::Rejected { reason } => format!(
            "the shipped `apr quantize` REJECTED this argv, but the observation expected it to be accepted.\n  \
             argv:                   {shown}\n  \
             shipped parser said:    {reason}\n  \
             `apr quantize` accepts: {accepts}"
        ),
        QuantizeArgvVerdict::Accepted => {
            debug_assert!(!expected_accepted, "accepted+expected-accepted is a pass");
            format!(
                "the shipped `apr quantize` ACCEPTED this argv, but the observation expected it to be rejected.\n  \
                 argv:                   {shown}\n  \
                 `apr quantize` accepts: {accepts}"
            )
        }
    }
}

/// Run the flags gate over an observation's `flags` object.
///
/// `Err` means the observation itself is unusable (the caller maps it to exit
/// 4); `Ok(gate)` with `passed == false` means the observation was usable and
/// the shipped parser disagreed with it (exit 5).
pub fn evaluate_flags_observation(v: &Value, falsify_id: &str) -> Result<FlagParityGate, String> {
    let argv = read_argv(v, falsify_id)?;
    let expected_accepted = read_expectation(v, falsify_id)?;

    let borrowed: Vec<&str> = argv.iter().map(String::as_str).collect();
    let verdict = shipped_quantize_verdict(&borrowed);
    let passed = verdict.is_accepted() == expected_accepted;

    let expected_label = if expected_accepted {
        EXPECTED_ACCEPTED
    } else {
        EXPECTED_REJECTED
    };
    let got = verdict.label();
    let outcome = match &verdict {
        QuantizeArgvVerdict::Accepted => {
            format!("expected={expected_label} got={got} (shipped `apr quantize` parser)")
        }
        QuantizeArgvVerdict::Rejected { reason } => {
            format!("expected={expected_label} got={got} (shipped `apr quantize` parser: {reason})")
        }
    };
    let failure = if passed {
        None
    } else {
        Some(failure_text(expected_accepted, &argv, &verdict))
    };
    Ok(FlagParityGate {
        passed,
        outcome,
        failure,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- FALSIFY-LINTFLAG-001: the oracle is the shipped parser ----

    #[test]
    fn shipped_parser_rejects_the_flags_the_hand_rolled_parser_accepted() {
        // The load-bearing case. `parse_gptq_flags` + `validate_gptq_flags`
        // called this argv `Ok { bits: 4, group_size: 128 }`. `apr quantize`
        // has no `--method`, no `--bits` and no `--group-size`.
        let v =
            shipped_quantize_verdict(&["--method", "gptq", "--bits", "4", "--group-size", "128"]);
        let QuantizeArgvVerdict::Rejected { reason } = v else {
            panic!("shipped `apr quantize` must REFUSE --method/--bits/--group-size, got: {v:?}");
        };
        assert!(
            reason.contains("--method"),
            "the refusal must name the flag that does not exist; got: {reason}"
        );
    }

    #[test]
    fn shipped_parser_rejects_the_awq_argv_the_hand_rolled_parser_accepted() {
        let v = shipped_quantize_verdict(&["--method=awq", "--bits=4"]);
        assert!(
            !v.is_accepted(),
            "shipped `apr quantize` must REFUSE --method=awq --bits=4, got: {v:?}"
        );
    }

    #[test]
    fn shipped_parser_accepts_a_real_quantize_invocation() {
        let v =
            shipped_quantize_verdict(&["model.safetensors", "--scheme", "int4", "-o", "out.apr"]);
        assert_eq!(
            v,
            QuantizeArgvVerdict::Accepted,
            "a real `apr quantize <FILE> --scheme int4 -o out.apr` must be accepted"
        );
    }

    #[test]
    fn shipped_parser_accepts_the_long_form_and_the_plan_flag() {
        assert_eq!(
            shipped_quantize_verdict(&["m.apr", "--scheme", "q4k", "--plan"]),
            QuantizeArgvVerdict::Accepted
        );
        assert_eq!(
            shipped_quantize_verdict(&[
                "m.apr", "--output", "o.apr", "--format", "gguf", "--force"
            ]),
            QuantizeArgvVerdict::Accepted
        );
    }

    #[test]
    fn a_leading_literal_quantize_is_not_double_prefixed() {
        assert_eq!(
            shipped_quantize_verdict(&["quantize", "m.apr", "--scheme", "int8"]),
            QuantizeArgvVerdict::Accepted
        );
    }

    #[test]
    fn shipped_parser_refuses_quantize_without_the_required_file() {
        let v = shipped_quantize_verdict(&["--scheme", "int4"]);
        assert!(
            !v.is_accepted(),
            "`apr quantize --scheme int4` omits the required <FILE>; got: {v:?}"
        );
    }

    #[test]
    fn shipped_parser_refuses_an_unknown_flag() {
        let v = shipped_quantize_verdict(&["m.apr", "--totally-made-up"]);
        assert!(!v.is_accepted(), "unknown flag must be refused; got: {v:?}");
    }

    #[test]
    fn verdict_labels_are_the_observation_vocabulary() {
        assert_eq!(QuantizeArgvVerdict::Accepted.label(), "accepted");
        assert_eq!(
            QuantizeArgvVerdict::Rejected { reason: "x".into() }.label(),
            "rejected"
        );
    }

    // ---- FALSIFY-LINTFLAG-002: the accepted-flag list is read off clap ----

    #[test]
    fn accepts_summary_names_the_real_quantize_surface() {
        let s = shipped_quantize_accepts_summary();
        for expected in [
            "<FILE>", "--scheme", "--output", "--format", "--batch", "--plan", "--force",
        ] {
            assert!(
                s.contains(expected),
                "summary must name {expected}; got: {s}"
            );
        }
    }

    #[test]
    fn accepts_summary_never_advertises_the_flags_that_do_not_exist() {
        let s = shipped_quantize_accepts_summary();
        for absent in ["--method", "--bits", "--group-size"] {
            assert!(
                !s.contains(absent),
                "summary must not advertise {absent}, which `apr quantize` does not take; got: {s}"
            );
        }
    }

    // ---- FALSIFY-LINTFLAG-003: gate verdict = shipped parser verdict ----

    #[test]
    fn gate_fails_when_observation_expects_the_nonexistent_flags_to_be_accepted() {
        let obs = json!({
            "argv": ["--method", "gptq", "--bits", "4", "--group-size", "128"],
            "expected_outcome": "ok"
        });
        let gate = evaluate_flags_observation(&obs, "FALSIFY-TEST").expect("observation is usable");
        assert!(
            !gate.passed,
            "gate must fail: the shipped parser refuses this argv; outcome={}",
            gate.outcome
        );
        let msg = gate.failure.expect("a failed gate must explain itself");
        assert!(msg.contains("REJECTED"), "got: {msg}");
        assert!(
            msg.contains("--scheme"),
            "the failure must name the flags `apr quantize` does accept; got: {msg}"
        );
    }

    #[test]
    fn gate_passes_when_observation_expects_a_real_invocation_to_be_accepted() {
        let obs = json!({
            "argv": ["model.safetensors", "--scheme", "int4", "-o", "out.apr"],
            "expected_outcome": "accepted"
        });
        let gate = evaluate_flags_observation(&obs, "FALSIFY-TEST").expect("observation is usable");
        assert!(gate.passed, "outcome={}", gate.outcome);
        assert!(gate.failure.is_none());
    }

    #[test]
    fn gate_passes_when_observation_expects_the_nonexistent_flags_to_be_rejected() {
        let obs = json!({
            "argv": ["--method", "awq", "--bits", "4"],
            "expected_outcome": "rejected"
        });
        let gate = evaluate_flags_observation(&obs, "FALSIFY-TEST").expect("observation is usable");
        assert!(gate.passed, "outcome={}", gate.outcome);
    }

    #[test]
    fn gate_fails_when_observation_expects_a_real_invocation_to_be_rejected() {
        let obs = json!({
            "argv": ["m.apr", "--scheme", "int4", "-o", "o.apr"],
            "expected_outcome": "rejected"
        });
        let gate = evaluate_flags_observation(&obs, "FALSIFY-TEST").expect("observation is usable");
        assert!(!gate.passed, "outcome={}", gate.outcome);
        let msg = gate.failure.expect("a failed gate must explain itself");
        assert!(msg.contains("ACCEPTED"), "got: {msg}");
    }

    // ---- unusable observations (exit 4), not gate failures (exit 5) ----

    #[test]
    fn a_stale_per_flag_label_is_refused_as_unusable_not_folded_into_rejected() {
        let obs = json!({ "argv": ["--method", "gptq"], "expected_outcome": "missing_bits" });
        let err = evaluate_flags_observation(&obs, "FALSIFY-TEST")
            .expect_err("a label the real parser cannot produce must be refused");
        assert!(err.contains("missing_bits"), "got: {err}");
        assert!(err.contains("Re-capture it"), "got: {err}");
    }

    #[test]
    fn a_missing_expectation_is_refused_rather_than_defaulted() {
        let obs = json!({ "argv": ["m.apr", "--scheme", "int4", "-o", "o.apr"] });
        let err = evaluate_flags_observation(&obs, "FALSIFY-TEST")
            .expect_err("an implied expectation asserts nothing");
        assert!(err.contains("expected_outcome is missing"), "got: {err}");
    }

    #[test]
    fn a_missing_argv_is_refused_rather_than_treated_as_empty() {
        let obs = json!({ "expected_outcome": "accepted" });
        let err = evaluate_flags_observation(&obs, "FALSIFY-TEST")
            .expect_err("no argv means nothing was captured");
        assert!(err.contains("flags.argv is missing"), "got: {err}");
    }

    #[test]
    fn a_non_string_argv_element_is_refused() {
        let obs = json!({ "argv": ["m.apr", 4], "expected_outcome": "accepted" });
        let err =
            evaluate_flags_observation(&obs, "FALSIFY-TEST").expect_err("argv must be strings");
        assert!(err.contains("non-string element"), "got: {err}");
    }

    #[test]
    fn the_gate_is_deterministic() {
        let obs = json!({
            "argv": ["--method", "gptq", "--bits", "4"],
            "expected_outcome": "ok"
        });
        let a = evaluate_flags_observation(&obs, "FALSIFY-TEST").expect("usable");
        let b = evaluate_flags_observation(&obs, "FALSIFY-TEST").expect("usable");
        assert_eq!(a.passed, b.passed);
        assert_eq!(a.outcome, b.outcome);
    }
}
