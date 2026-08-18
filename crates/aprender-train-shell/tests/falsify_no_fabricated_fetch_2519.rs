//! FALSIFY-SHELL-2519: `fetch` must not claim to have downloaded a model.
//!
//! `commands.rs:232` opened with "Simulate model fetching" and built a
//! `LoadedModel` out of the model ID string:
//!
//!     architecture: detect_architecture(model_id),
//!     parameters: estimate_params(model_id),   // "7b" in the NAME -> 7.0B
//!     layers: estimate_layers(model_id),       // "7b" in the NAME -> 32
//!     hidden_dim: 4096,                        // literal
//!
//! Measured before the fix, on an ID that cannot exist:
//!
//!     Warning: could not detect architecture from model ID '...', defaulting
//!     to 'unknown'
//!     ✓ Fetched does-not-exist/totally-fake-7b
//!       Architecture: unknown
//!       Parameters: 7.0B
//!       Layers: 32
//!
//! This crate is the least egregious of the three in #2519: the architecture
//! line does warn, and does say `unknown`. Exactly two things are wrong, and
//! only those two are asserted here -- (a) `✓ Fetched` for something that was
//! never fetched, and (b) a parameter/layer count read out of the ID string.

use entrenar_shell::commands::{execute, parse, Command};
use entrenar_shell::state::ModelRole;
use entrenar_shell::SessionState;

fn run(line: &str, state: &mut SessionState) -> entrenar_common::Result<String> {
    let cmd = parse(line).expect("these lines all parse");
    execute(&cmd, state)
}

#[test]
fn a_model_that_cannot_exist_is_not_reported_as_fetched() {
    let mut state = SessionState::new();

    let result = run("fetch does-not-exist/totally-fake-7b", &mut state);

    let err = result.expect_err(
        "fetch returned Ok for a model ID that cannot exist -- it is claiming a \
         download that never happened, which is the #2519 defect",
    );
    let text = format!("{err}");
    assert!(!text.contains("Fetched"), "got: {text}");
    // (b): the figures were string-matched out of the ID, so the refusal must
    // not restate them either.
    assert!(!text.contains("7.0B"), "got: {text}");
    assert!(!text.contains("Layers"), "got: {text}");

    assert!(
        state.loaded_models().is_empty(),
        "a model that was never downloaded was added to the session anyway"
    );
}

/// Discriminating test: the answer came from the ID STRING, so an ID that lies
/// about its size was believed. `tiny/model-70b` and a genuine 70B checkpoint
/// got the same 70.0B / 80 layers, because nothing was ever read from a file.
/// Whatever `fetch` does, it must not report a size for a name.
#[test]
fn size_is_not_read_out_of_the_model_name() {
    let mut state = SessionState::new();

    for (id, claimed) in [
        ("tiny/model-70b", "70.0B"),
        ("tiny/model-13b", "13.0B"),
        ("tiny/model-7b", "7.0B"),
    ] {
        let output =
            run(&format!("fetch {id}"), &mut state).unwrap_or_else(|e| format!("refused: {e}"));

        assert!(
            !output.contains(claimed),
            "`fetch {id}` still reports {claimed}, which is the substring of the \
             NAME and not a property of any file: {output}"
        );
    }

    assert!(state.loaded_models().is_empty());
}

/// Two IDs differing only in the digits of their name must not be the sole
/// reason two different answers are given -- the equal-size-different-files
/// analogue. Either both are refused, or the shell actually read two files and
/// can say which bytes it read.
#[test]
fn two_names_differing_only_in_digits_get_no_confident_answer() {
    let mut state = SessionState::new();

    let seven = run("fetch fake/model-7b", &mut state);
    let thirteen = run("fetch fake/model-13b", &mut state);

    if let (Ok(a), Ok(b)) = (&seven, &thirteen) {
        assert!(
            !a.contains("Parameters"),
            "fetch reported a parameter count derived from the name: {a}"
        );
        assert_ne!(
            a, b,
            "two nonexistent models produced identical descriptions: {a}"
        );
    }
}

/// Non-vacuity companion 1: `fetch` with no argument still fails at PARSE time
/// for its own distinct reason, so the refusal above is not a blanket "every
/// fetch errors for one cause".
#[test]
fn fetch_without_an_id_fails_for_its_own_reason() {
    let err = parse("fetch").expect_err("`fetch` with no model ID must not parse");
    let text = format!("{err}");

    assert!(text.contains("No model ID provided"), "got: {text}");
    assert!(!text.contains("HuggingFace client"), "got: {text}");
}

/// Non-vacuity companion 2: the shell still works. Commands that do their own
/// honest arithmetic or bookkeeping must still return Ok -- otherwise the tests
/// above would be observing a REPL that refuses everything.
#[test]
fn commands_that_do_real_work_still_succeed() {
    let mut state = SessionState::new();

    let set = run("set batch_size 64", &mut state).expect("set must still work");
    assert!(set.contains("64"));

    // Arithmetic on values the user supplied, not on invented model facts.
    let memory = run("memory --batch 8 --seq 512", &mut state).expect("memory must still work");
    assert!(memory.contains("batch=8"));
    assert!(memory.contains("seq=512"));

    let help = run("help fetch", &mut state).expect("help must still work");
    assert!(help.contains("fetch"));

    // And a genuinely unknown command is still diagnosed as one.
    let unknown = execute(
        &Command::Unknown {
            input: "frobnicate".to_string(),
        },
        &mut state,
    );
    assert!(unknown.is_err());
}

/// `distill` depended on fetched models. With nothing loadable it must say so
/// rather than report progress -- it used to end at "Training started...
/// (simulated)", which is only reachable once two models are in the session.
#[test]
fn distill_cannot_start_training_on_models_that_were_never_fetched() {
    let mut state = SessionState::new();

    assert!(run("fetch a/teacher-7b --teacher", &mut state).is_err());
    assert!(run("fetch b/student-1b --student", &mut state).is_err());

    let err = run("distill", &mut state).expect_err("distill must not start on nothing");
    assert!(format!("{err}").contains("teacher"));
}

/// The single-command CLI surface (`-c`) is the non-interactive form of the
/// reproduction in #2519, and it must exit non-zero.
#[test]
fn the_single_command_surface_exits_non_zero() {
    // Resolved at RUNTIME by asking cargo, not with
    // env!("CARGO_BIN_EXE_aprender-train-shell"). That macro is evaluated at
    // COMPILE time and failed the build in CI --
    //
    //     error: environment variable `CARGO_BIN_EXE_aprender-train-shell`
    //            not defined at compile time
    //
    // while compiling fine locally under the identical
    // `cargo test -p aprender-train-shell --test <name>` command. Rather than
    // keep guessing at the difference, this uses the pattern already proven for
    // aprender-mcp: ask cargo which executable it produced. Same doctrine as
    // scripts/apr_bin.sh -- never construct or assume a binary path.
    let exe = cargo_built_binary();
    let output = std::process::Command::new(&exe)
        .args(["-c", "fetch does-not-exist/totally-fake-7b"])
        .output()
        .expect("binary should run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "`-c 'fetch <nonexistent>'` exited 0:\n{stdout}"
    );
    assert!(
        !stdout.contains("Fetched"),
        "still claims a fetch:\n{stdout}"
    );
    assert!(!stdout.contains("7.0B"), "still reports a size:\n{stdout}");
}

/// Roles are still parsed even though nothing can be loaded into them -- the
/// parse-level behaviour was never the defect, and quietly dropping it would be
/// a second regression hiding behind the first fix.
#[test]
fn role_flags_are_still_parsed() {
    assert!(matches!(
        parse("fetch some/model --teacher").expect("parses"),
        Command::Fetch {
            role: ModelRole::Teacher,
            ..
        }
    ));
    assert!(matches!(
        parse("fetch some/model --student").expect("parses"),
        Command::Fetch {
            role: ModelRole::Student,
            ..
        }
    ));
}

/// Ask cargo for this package's binary, and fail loudly if it cannot say.
///
/// A test that silently skipped when the binary was unavailable would be the
/// skip-class escape this repo bans -- and would have hidden the very defect
/// #2519 is about.
fn cargo_built_binary() -> std::path::PathBuf {
    let out = std::process::Command::new(env!("CARGO"))
        .args([
            "build",
            "-p",
            "aprender-train-shell",
            "--bin",
            "aprender-train-shell",
            "--message-format=json-render-diagnostics",
        ])
        .output()
        .expect("cargo build must run");
    assert!(
        out.status.success(),
        "cargo build failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut found: Option<std::path::PathBuf> = None;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        // Deliberately a substring match rather than a JSON dependency: this
        // test crate must not grow one for a path lookup.
        if !line.contains("\"compiler-artifact\"") {
            continue;
        }
        if let Some(i) = line.find("\"executable\":\"") {
            let rest = &line[i + 14..];
            if let Some(j) = rest.find('"') {
                let p = std::path::PathBuf::from(&rest[..j]);
                if p.file_name().is_some_and(|n| n == "aprender-train-shell") {
                    found = Some(p);
                }
            }
        }
    }
    found.expect("cargo reported no executable for aprender-train-shell")
}
