//! Entry points for the interactive training shell.
//!
//! These functions were the body of the `aprender-train-shell` binary's
//! `main.rs`. That binary is gone (APR-MONO Rule 1: `apr` is the only
//! user-facing binary); the capability is reachable as `apr train shell`,
//! which calls exactly these entry points.

use crate::state::SessionState;
use entrenar_common::Result;
use std::path::Path;

/// Load a session from `path`, falling back to a fresh session on failure.
///
/// The pre-migration binary printed the load failure and carried on with an
/// empty session rather than aborting; that behaviour is preserved. The
/// returned `bool` is `true` when the session was loaded from disk, so callers
/// can report which of the two happened.
#[must_use]
pub fn load_session_or_default(path: &Path) -> (SessionState, bool) {
    // `SessionState::load` takes `&PathBuf`; this module's callers hold `&Path`.
    match SessionState::load(&path.to_path_buf()) {
        Ok(s) => {
            println!("Loaded session from {}", path.display());
            (s, true)
        }
        Err(e) => {
            eprintln!("Failed to load session: {e}");
            (SessionState::new(), false)
        }
    }
}

/// Execute one shell command against `state` and print its output.
///
/// This is `apr train shell --command "<cmd>"`: parse, execute, print, return.
/// Empty output is not printed, matching the pre-migration binary.
///
/// # Errors
///
/// Returns the parse error for an unrecognised command, or the execution error
/// the command produced.
pub fn run_single_command(command: &str, state: &mut SessionState) -> Result<()> {
    let parsed = crate::commands::parse(command)?;
    let output = crate::commands::execute(&parsed, state)?;
    if !output.is_empty() {
        println!("{output}");
    }
    Ok(())
}

/// Start the interactive REPL with `state` pre-loaded.
///
/// # Errors
///
/// Propagates terminal-editor construction failures and REPL errors.
pub fn run_interactive(state: SessionState) -> Result<()> {
    crate::start_with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_single_command_refuses_unparseable_input() {
        let mut state = SessionState::new();
        // Asserting is_ok() on a bogus command would lock the defect in.
        assert!(
            run_single_command("this-is-not-a-shell-command", &mut state).is_err(),
            "an unrecognised command must be refused, not silently ignored"
        );
    }

    #[test]
    fn load_session_or_default_reports_fallback_for_missing_file() {
        let missing = Path::new("/nonexistent/session-that-does-not-exist.json");
        let (state, loaded) = load_session_or_default(missing);
        assert!(
            !loaded,
            "a missing session file must report that it was NOT loaded"
        );
        assert!(
            state.loaded_models().is_empty(),
            "the fallback session must be empty"
        );
    }
}
