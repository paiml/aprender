//! Process-wide output level for `--quiet` / `--verbose` (dogfood-0.63.0, #2401).
//!
//! `-q, --quiet` and `-v, --verbose` are declared as clap **globals** on
//! [`crate::Cli`], so clap prints them in the Options block of all 104
//! subcommands' `--help`. Until this module existed nothing read them: in
//! v0.63.0 `apr inspect m.apr` and `apr inspect m.apr --quiet` produced
//! byte-identical stdout on 14 of 16 sampled commands, `apr hex m.apr
//! --quiet` still wrote 303 972 bytes, and `apr gbnf-lint ... -q` still
//! printed the full PASS report. Only `list` and `lint` — the two commands
//! that happened to receive `quiet` as a parameter — honoured it.
//!
//! Threading a `quiet` parameter into every command is the design that
//! already failed: it is the same forwarding bug `--offline` had (see
//! [`crate::commands::offline`]), where three commands forgot to pass the
//! flag along and the control was silently inert. So this is a **latch**,
//! set once in [`crate::execute_command`], plus a crate-wide shadow of
//! `println!`/`print!` that consults it. A command cannot disarm `--quiet`
//! by forgetting to plumb a parameter, because it never receives one.
//!
//! Semantics:
//!
//! * `--quiet` suppresses ordinary stdout. stderr is untouched, so the
//!   `error: ...` line printed by [`crate::cli_main`] and the process exit
//!   code both survive — "errors only", as the help text promises.
//! * `--quiet` does **not** suppress `--json`: the JSON document is the
//!   machine-readable payload a script asked for, and swallowing it would
//!   make `--json --quiet` useless. [`stdout_suppressed`] returns false
//!   whenever `--json` is in effect.
//! * A command that implements its own richer quiet semantics opts out of
//!   the blanket gate with [`emitln!`]/[`emit!`]. Two do: `apr list --quiet`
//!   must still print one model identifier per line (contract
//!   `apr-list-quiet-wiring-v1` F-LIST-QUIET-001), and `apr lint --quiet`
//!   filters its table down to errors rather than going silent.
//! * `--verbose` raises the level so commands can print detail they
//!   otherwise elide; see [`is_verbose`] and [`vprintln!`].
//! * `--quiet` wins over `--verbose` when both are given.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// How much ordinary stdout a run should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// `--quiet`: ordinary stdout suppressed.
    Quiet,
    /// Neither flag given.
    Normal,
    /// `--verbose`: commands may print elided detail.
    Verbose,
}

impl Level {
    const fn as_u8(self) -> u8 {
        match self {
            Level::Quiet => 0,
            Level::Normal => 1,
            Level::Verbose => 2,
        }
    }

    const fn from_u8(v: u8) -> Level {
        match v {
            0 => Level::Quiet,
            2 => Level::Verbose,
            _ => Level::Normal,
        }
    }
}

/// Resolve the two flags into one level.
///
/// `--quiet` wins over `--verbose`: the pair used to be accepted with no
/// error and no effect at all, so *some* rule is needed, and silencing is
/// the safer of the two for anything reading the stream.
#[must_use]
pub fn resolve(quiet: bool, verbose: bool) -> Level {
    if quiet {
        Level::Quiet
    } else if verbose {
        Level::Verbose
    } else {
        Level::Normal
    }
}

static PROCESS_LEVEL: AtomicU8 = AtomicU8::new(Level::Normal.as_u8());
static PROCESS_JSON: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Thread-scoped override used by [`scope`] so tests can drive the gate
    /// without mutating process-global state shared with ~6 600 siblings.
    static THREAD_LEVEL: Cell<Option<Level>> = const { Cell::new(None) };
    static THREAD_JSON: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Record the run's output level. Called once from `execute_command`.
///
/// Never resets to [`Level::Normal`]: an in-process test that runs a second
/// command must not be able to un-quiet a run the user asked to be quiet.
pub fn latch(quiet: bool, verbose: bool, json: bool) {
    match resolve(quiet, verbose) {
        Level::Normal => {}
        level => PROCESS_LEVEL.store(level.as_u8(), Ordering::SeqCst),
    }
    if json {
        PROCESS_JSON.store(true, Ordering::SeqCst);
    }
}

/// RAII guard returned by [`scope`]; restores the previous thread values.
pub struct VerbosityScope(Option<Level>, Option<bool>);

impl Drop for VerbosityScope {
    fn drop(&mut self) {
        THREAD_LEVEL.with(|c| c.set(self.0));
        THREAD_JSON.with(|c| c.set(self.1));
    }
}

/// Override the level for the current thread until the guard drops.
#[must_use]
pub fn scope(level: Level, json: bool) -> VerbosityScope {
    let prev_level = THREAD_LEVEL.with(|c| c.replace(Some(level)));
    let prev_json = THREAD_JSON.with(|c| c.replace(Some(json)));
    VerbosityScope(prev_level, prev_json)
}

/// The level in effect for this call.
#[must_use]
pub fn level() -> Level {
    if let Some(l) = THREAD_LEVEL.with(Cell::get) {
        return l;
    }
    Level::from_u8(PROCESS_LEVEL.load(Ordering::SeqCst))
}

/// True iff `--json` is in effect for this call.
#[must_use]
pub fn json_enabled() -> bool {
    if let Some(j) = THREAD_JSON.with(Cell::get) {
        return j;
    }
    PROCESS_JSON.load(Ordering::SeqCst)
}

/// True iff `--quiet` was given.
#[must_use]
pub fn is_quiet() -> bool {
    level() == Level::Quiet
}

/// True iff `--verbose` was given (and `--quiet` was not).
#[must_use]
pub fn is_verbose() -> bool {
    level() == Level::Verbose
}

/// The single decision the shadowed `println!`/`print!` consult.
///
/// Quiet suppresses ordinary stdout, except when `--json` is in effect —
/// the JSON document is the payload, not chatter.
#[must_use]
pub fn stdout_suppressed() -> bool {
    !json_enabled() && is_quiet()
}

/// The `--verbose` preamble `execute_command` prints before dispatching.
///
/// `--verbose` was the other half of #2401: byte-inert on 13 of 16 sampled
/// commands, because only `check`, `oracle` and `trace` ever received it as
/// a parameter. Rather than invent per-command chatter for 104 commands,
/// this reports what the *dispatcher* actually resolved and decided — facts
/// the run already computed and then threw away:
///
/// * which model paths `extract_model_paths` pulled out of the parsed
///   command, and their sizes on disk;
/// * whether the PMAT-237 contract gate ran over them, was disabled with
///   `--skip-contract`, or did not apply because the command is one of the
///   diagnostic ones the gate deliberately exempts. That last line also
///   explains the audit's separate observation that `--skip-contract` has
///   "zero effect" on `inspect`/`validate`/`tensors`: there is nothing for
///   it to skip there, and now the CLI says so instead of staying mute.
/// * whether `--offline` is latched.
///
/// Pure so it can be asserted directly; the caller prints the lines.
#[must_use]
pub fn preamble_lines(
    version: &str,
    offline: bool,
    skip_contract: bool,
    paths: &[std::path::PathBuf],
) -> Vec<String> {
    let mut out = vec![format!("verbose: apr {version}")];
    out.push(format!(
        "verbose: offline = {}",
        if offline { "on" } else { "off" }
    ));
    if skip_contract {
        out.push("verbose: contract gate = skipped (--skip-contract)".to_string());
    } else if paths.is_empty() {
        out.push(
            "verbose: contract gate = not applicable (no gated model path for this command)"
                .to_string(),
        );
    } else {
        out.push(format!(
            "verbose: contract gate = enforced over {} path(s)",
            paths.len()
        ));
    }
    for p in paths {
        let size = std::fs::metadata(p).map_or_else(
            |_| "unreadable".to_string(),
            |m| format!("{} bytes", m.len()),
        );
        out.push(format!("verbose: model = {} ({size})", p.display()));
    }
    out
}

/// Crate-wide shadow of `std::println!` that honours `--quiet`.
///
/// Declared with `#[macro_use] mod verbosity;` before `mod commands;` in
/// `lib.rs`, so every `println!` in the ~9 000 call sites below that point
/// resolves here instead of to the standard-library prelude. That is the
/// whole point: `--quiet` cannot be forgotten by a command author, because
/// there is nothing for a command author to remember.
macro_rules! println {
    () => {
        if !$crate::verbosity::stdout_suppressed() { ::std::println!() }
    };
    ($($arg:tt)*) => {
        if !$crate::verbosity::stdout_suppressed() { ::std::println!($($arg)*) }
    };
}

/// Crate-wide shadow of `std::print!` that honours `--quiet`.
macro_rules! print {
    ($($arg:tt)*) => {
        if !$crate::verbosity::stdout_suppressed() { ::std::print!($($arg)*) }
    };
}

/// Print only under `--verbose`.
macro_rules! vprintln {
    ($($arg:tt)*) => {
        if $crate::verbosity::is_verbose() { ::std::println!($($arg)*) }
    };
}

/// Print regardless of `--quiet` — the opt-out for commands that implement
/// their own quiet semantics (`apr list --quiet`, `apr lint --quiet`).
macro_rules! emitln {
    () => { ::std::println!() };
    ($($arg:tt)*) => { ::std::println!($($arg)*) };
}

/// `print!` counterpart of [`emitln!`].
macro_rules! emit {
    ($($arg:tt)*) => { ::std::print!($($arg)*) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_wins_over_verbose() {
        assert_eq!(resolve(true, true), Level::Quiet);
        assert_eq!(resolve(true, false), Level::Quiet);
        assert_eq!(resolve(false, true), Level::Verbose);
        assert_eq!(resolve(false, false), Level::Normal);
    }

    #[test]
    fn default_level_prints() {
        let _s = scope(Level::Normal, false);
        assert!(!stdout_suppressed());
        assert!(!is_quiet());
        assert!(!is_verbose());
    }

    #[test]
    fn quiet_suppresses_stdout() {
        let _s = scope(Level::Quiet, false);
        assert!(stdout_suppressed(), "--quiet must suppress ordinary stdout");
    }

    #[test]
    fn json_survives_quiet() {
        let _s = scope(Level::Quiet, true);
        assert!(
            !stdout_suppressed(),
            "--json --quiet must still emit the JSON document"
        );
    }

    #[test]
    fn verbose_does_not_suppress() {
        let _s = scope(Level::Verbose, false);
        assert!(!stdout_suppressed());
        assert!(is_verbose());
    }

    #[test]
    fn scope_restores_previous_level() {
        let baseline = level();
        {
            let _s = scope(Level::Quiet, false);
            assert_eq!(level(), Level::Quiet);
        }
        assert_eq!(level(), baseline, "scope must restore on drop");
    }

    #[test]
    fn preamble_reports_the_gate_decision_not_a_fixed_string() {
        let none: Vec<std::path::PathBuf> = vec![];
        let skipped = preamble_lines("9.9.9", false, true, &none);
        let inapplicable = preamble_lines("9.9.9", false, false, &none);
        let enforced = preamble_lines("9.9.9", true, false, &[std::path::PathBuf::from("/x.apr")]);

        assert!(
            skipped.iter().any(|l| l.contains("--skip-contract")),
            "--skip-contract must be visible under --verbose, got {skipped:?}"
        );
        assert!(
            inapplicable.iter().any(|l| l.contains("not applicable")),
            "a command the gate exempts must say so rather than stay mute, got {inapplicable:?}"
        );
        assert!(
            enforced.iter().any(|l| l.contains("enforced over 1 path")),
            "an enforced gate must report its paths, got {enforced:?}"
        );
        assert!(
            enforced.iter().any(|l| l.contains("/x.apr")),
            "the resolved model path must be reported, got {enforced:?}"
        );
        assert!(
            enforced.iter().any(|l| l.contains("offline = on")),
            "--offline must be visible under --verbose, got {enforced:?}"
        );
        assert_ne!(
            skipped, inapplicable,
            "the three gate outcomes must be distinguishable"
        );
    }

    // ---------------------------------------------------------------
    // Behavioural falsifiers for #2401.
    //
    // The unit tests above only prove the *decision function*. They cannot
    // see the thing that was actually broken in v0.63.0: `execute_command`
    // never recorded the flags, so the ~9 000 `println!` call sites below
    // `mod commands;` printed regardless. Proving that needs a real command
    // run end to end with real stdout.
    //
    // The level is a process-wide latch that deliberately cannot be un-set
    // (an in-process test must not be able to un-quiet a run the user asked
    // to be quiet), so each mode runs in its own child process — the same
    // pattern `commands::offline` uses for the same reason.
    //
    // `gbnf-lint` is the audit's own repro from finding 1 and needs nothing
    // but a small JSON file, so the falsifier is hermetic.
    // ---------------------------------------------------------------

    const CHILD_ENV: &str = "APR_VERBOSITY_LATCH_CHILD";
    const BEGIN: &str = "<<<APR-2401-BEGIN>>>";
    const END: &str = "<<<APR-2401-END>>>";
    const TEST_PATH: &str =
        "verbosity::tests::quiet_and_verbose_reach_a_command_that_never_receives_them";

    /// The parent creates this once and hands the SAME path to all three
    /// children. An earlier draft stamped the child's pid into the name, so
    /// `normal` and `verbose` differed by the path alone and the
    /// "--verbose is not a no-op" assertion passed while --verbose was
    /// disabled — a test that would have locked the defect in. One path for
    /// every mode makes the comparison a byte comparison, which is the
    /// methodology the audit used.
    const OBS_ENV: &str = "APR_VERBOSITY_OBS_FILE";

    fn parent_observation_file() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("apr-2401-obs-{}.json", std::process::id()));
        std::fs::write(&p, r#"{"output":"{\"a\":1}","finish_reason":"stop"}"#)
            .expect("write observation file");
        p
    }

    /// Run `apr gbnf-lint` in this child with whichever flag the parent asked
    /// for, going through the real `execute_command` so the latch is exercised.
    fn child_runs_gbnf_lint(mode: &str) {
        // `Commands` is a very large enum; parsing it on libtest's default
        // stack overflows, exactly as `commands::offline`'s child does.
        let mode = mode.to_string();
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;

                let obs = std::env::var(OBS_ENV).expect("parent must hand down the same obs file");
                let mut argv = vec!["apr", "gbnf-lint", "--observation-file", obs.as_str()];
                match mode.as_str() {
                    "quiet" => argv.push("--quiet"),
                    "verbose" => argv.push("--verbose"),
                    _ => {}
                }
                let cli = crate::Cli::parse_from(argv);
                // Markers use `::std::println!` so they survive `--quiet` and
                // give the parent an exact slice of the command's own stdout;
                // libtest with --nocapture interleaves its own text otherwise.
                ::std::println!("{BEGIN}");
                crate::execute_command(&cli)
                    .expect("gbnf-lint on a well-formed observation must succeed");
                ::std::println!("{END}");
            })
            .expect("spawn")
            .join()
            .expect("gbnf-lint verbosity falsifier panicked");
    }

    fn run_child(mode: &str, obs: &std::path::Path) -> String {
        let exe = std::env::current_exe().expect("current test binary");
        let out = std::process::Command::new(exe)
            .args(["--exact", TEST_PATH, "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, mode)
            .env(OBS_ENV, obs)
            .output()
            .expect("re-run this test binary as a child");
        assert!(
            out.status.success(),
            "child ({mode}) failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        // libtest's own progress lines share this stdout, so slice out the
        // marked region rather than trying to recognise them.
        let all = String::from_utf8_lossy(&out.stdout).into_owned();
        let start = all
            .find(BEGIN)
            .map(|i| i + BEGIN.len())
            .unwrap_or_else(|| panic!("child ({mode}) never reached the command; got:\n{all}"));
        let end = all[start..]
            .find(END)
            .unwrap_or_else(|| panic!("child ({mode}) never finished the command; got:\n{all}"));
        all[start..start + end].trim().to_string()
    }

    #[test]
    fn quiet_and_verbose_reach_a_command_that_never_receives_them() {
        if let Ok(mode) = std::env::var(CHILD_ENV) {
            child_runs_gbnf_lint(&mode);
            return;
        }

        let obs = parent_observation_file();
        let normal = run_child("normal", &obs);
        let quiet = run_child("quiet", &obs);
        let verbose = run_child("verbose", &obs);
        let _ = std::fs::remove_file(&obs);

        assert!(
            normal.contains("gbnf-lint report"),
            "control run must print the report, got:\n{normal}"
        );
        assert!(
            quiet.is_empty(),
            "--quiet must suppress the PASS report as its own help text promises \
             (`Quiet mode (errors only)`); `apr gbnf-lint -q` still printed:\n{quiet}"
        );
        assert_ne!(
            normal, verbose,
            "--verbose must not be a byte-for-byte no-op; it was on 13 of 16 \
             sampled commands in v0.63.0"
        );
        assert!(
            verbose.contains("verbose: contract gate ="),
            "--verbose must report the dispatcher's gate decision, got:\n{verbose}"
        );
    }
}
