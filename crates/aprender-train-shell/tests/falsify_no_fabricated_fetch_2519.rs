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

// ---------------------------------------------------------------------------
// FALSIFY-SHELL-SESSION-2519: the SECOND door.
//
// The eight tests above all drive state through `fetch`, so they can only ever
// observe the door `fetch` closed. `--session` is a second entrance into the
// same room: `LoadedModel` derives `Deserialize`, and `main.rs` fed
// `SessionState::load` straight from a user-supplied path. Measured on
// origin/main 5c08e771f, AFTER the fetch fix, with a hand-written sess.json
// naming /nonexistent:
//
//     $ aprender-train-shell --session sess.json -c distill
//     Loaded session from sess.json
//     Training started... (simulated)                              # exit 0
//     $ aprender-train-shell --session sess.json -c "distill --dry-run"
//     Teacher: does-not-exist/totally-fake-7b (7.0B)
//     Student: does-not-exist/totally-fake-1b (1.0B)
//     Ready to train                                               # exit 0
//     $ aprender-train-shell --session sess.json -c memory
//     Model: 16.0 GB / Total: 20.3 GB                              # exit 0
//
// Every figure there was typed into a JSON file. This is the project-memory
// lesson "a guard's UNIVERSE built from the wrong side": the falsifier
// enumerated the fetch path, and the defect simply was not in it.
// ---------------------------------------------------------------------------

use entrenar_shell::state::{LoadedModel, Preferences, SessionMetrics};
use std::path::{Path, PathBuf};

/// The exact session file used for the reproduction above.
fn crafted_session_json(teacher_path: &str, student_path: &str) -> String {
    format!(
        r#"{{
  "models": {{
    "teacher": {{"id":"does-not-exist/totally-fake-7b","path":"{teacher_path}",
      "architecture":"llama","parameters":7000000000,"layers":32,
      "hidden_dim":4096,"role":"Teacher"}},
    "student": {{"id":"does-not-exist/totally-fake-1b","path":"{student_path}",
      "architecture":"llama","parameters":1000000000,"layers":16,
      "hidden_dim":2048,"role":"Student"}}
  }},
  "history": [],
  "preferences": {{"output_format":"table","show_progress":true,
    "auto_save_history":true,"default_batch_size":32,"default_seq_len":512}},
  "metrics": {{"total_commands":0,"successful_commands":0,"total_duration_ms":0}}
}}"#
    )
}

fn write_session(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).expect("temp write");
    p
}

/// Rule 1: a model the session says is cached must actually be on disk.
#[test]
fn a_session_may_not_claim_a_model_that_is_not_on_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_session(
        dir.path(),
        "sess.json",
        &crafted_session_json("/nonexistent/teacher", "/nonexistent/student"),
    );

    let err = SessionState::load(&path).expect_err(
        "a session naming /nonexistent was accepted -- the shell adopted a model \
         nobody has, which is the #2519 defect reached through --session",
    );
    let text = format!("{err}");
    assert!(text.contains("nonexistent"), "got: {text}");
    assert!(text.contains("2519"), "got: {text}");
}

/// Rule 2, the discriminating case: paths that DO exist are still not enough,
/// because nothing in this crate can produce a `LoadedModel` at all. Rule 1
/// alone would be defeated by `touch`.
#[test]
fn existing_paths_do_not_make_hand_written_model_facts_true() {
    let dir = tempfile::tempdir().expect("tempdir");
    let t = dir.path().join("teacher.bin");
    let s = dir.path().join("student.bin");
    std::fs::write(&t, b"not a model").expect("write");
    std::fs::write(&s, b"not a model").expect("write");

    let path = write_session(
        dir.path(),
        "sess.json",
        &crafted_session_json(
            &t.display().to_string().replace('\\', "/"),
            &s.display().to_string().replace('\\', "/"),
        ),
    );

    let err = SessionState::load(&path).expect_err(
        "11 bytes of `not a model` at an existing path was accepted as a 7.0B \
         32-layer llama -- rule 1 (path exists) is defeated by `touch`, so rule 2 \
         must reject any model this shell cannot have loaded",
    );
    let text = format!("{err}");
    assert!(text.contains("2519"), "got: {text}");
    assert!(
        !text.contains("Ready to train"),
        "the refusal must not also report the configuration: {text}"
    );
}

/// Non-vacuity control 1: `--session` still works. A session with no models
/// round-trips through save/load with its preferences intact, so the two tests
/// above are not observing a loader that refuses every file.
#[test]
fn an_honest_session_still_loads_and_keeps_its_preferences() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("honest.json");

    let mut state = SessionState::new();
    state.preferences_mut().default_batch_size = 64;
    state.preferences_mut().default_seq_len = 128;
    state.save(&path).expect("save must succeed");

    let loaded = SessionState::load(&path).expect(
        "a session the shell wrote itself must load back -- if this fails the \
         provenance check is a blanket refusal, not a ratchet",
    );
    assert_eq!(loaded.preferences().default_batch_size, 64);
    assert_eq!(loaded.preferences().default_seq_len, 128);
    assert!(loaded.loaded_models().is_empty());
}

/// Non-vacuity control 2: the provenance rule is a property of the MODELS, not
/// of deserialization. A directly-constructed empty state passes it.
#[test]
fn provenance_check_passes_on_a_state_with_no_models() {
    let state = SessionState::new();
    assert!(state.validate_model_provenance().is_ok());
    assert_eq!(state.metrics(), &SessionMetrics::default());
    assert_eq!(state.preferences(), &Preferences::default());
}

/// The independent second rule, which holds even if a real loader lands one
/// day: `distill` must not report training it did not run. This drives state
/// through `add_model` directly, so it bypasses `load` entirely -- if the only
/// fix were in `load`, this test would still be RED.
#[test]
fn distill_never_reports_training_it_did_not_run() {
    let mut state = SessionState::new();
    for (name, id, role, params) in [
        (
            "teacher",
            "a/teacher-7b",
            ModelRole::Teacher,
            7_000_000_000u64,
        ),
        (
            "student",
            "b/student-1b",
            ModelRole::Student,
            1_000_000_000u64,
        ),
    ] {
        state.add_model(
            name.to_string(),
            LoadedModel {
                id: id.to_string(),
                path: PathBuf::from("/tmp"),
                architecture: "llama".to_string(),
                parameters: params,
                layers: 32,
                hidden_dim: 4096,
                role,
            },
        );
    }

    let err = run("distill", &mut state).expect_err(
        "distill returned Ok with two models in the session -- it has no training \
         loop, so any success string here is fabricated (#2519)",
    );
    let text = format!("{err}");
    assert!(!text.contains("Training started"), "got: {text}");
    assert!(!text.contains("simulated"), "got: {text}");

    // Discriminating companion: `--dry-run` only restates the session, so it
    // stays Ok. If BOTH refused, this test would prove nothing about the
    // difference between describing a plan and claiming to have executed it.
    let dry = run("distill --dry-run", &mut state).expect("--dry-run only describes");
    assert!(dry.contains("Ready to train"));
    assert!(!dry.contains("Training started"));
}

/// End-to-end through the binary, which is the form #2519 was reported in.
#[test]
fn the_session_flag_surface_exits_non_zero_and_prints_nothing_confident() {
    let exe = cargo_built_binary();
    let dir = tempfile::tempdir().expect("tempdir");
    let sess = write_session(
        dir.path(),
        "sess.json",
        &crafted_session_json("/nonexistent/teacher", "/nonexistent/student"),
    );

    for args in [
        vec!["distill"],
        vec!["distill --dry-run"],
        vec!["memory"],
        vec!["inspect"],
    ] {
        let output = std::process::Command::new(&exe)
            .arg("--session")
            .arg(&sess)
            .arg("-c")
            .arg(args[0])
            .output()
            .expect("binary should run");
        let stdout = String::from_utf8_lossy(&output.stdout);

        assert!(
            !output.status.success(),
            "`--session <crafted> -c {}` exited 0:\n{stdout}",
            args[0]
        );
        for forbidden in [
            "simulated",
            "Ready to train",
            "7.0B",
            "16.0 GB",
            "32 layers",
        ] {
            assert!(
                !stdout.contains(forbidden),
                "`-c {}` still prints `{forbidden}`:\n{stdout}",
                args[0]
            );
        }
    }
}

/// A session file that is not valid JSON must also fail CLOSED. It used to
/// print "Failed to load session: ..." and then run against a silently
/// different (empty) session, exiting 0 -- so `--session` could be a no-op
/// nobody noticed.
#[test]
fn an_unreadable_session_file_does_not_silently_become_an_empty_one() {
    let exe = cargo_built_binary();
    let dir = tempfile::tempdir().expect("tempdir");
    let bad = write_session(dir.path(), "bad.json", "not json");

    let output = std::process::Command::new(&exe)
        .arg("--session")
        .arg(&bad)
        .args(["-c", "help"])
        .output()
        .expect("binary should run");

    assert!(
        !output.status.success(),
        "a corrupt --session file exited 0:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );

    // Control: the SAME invocation with a session the shell wrote itself must
    // exit 0, so the assertion above is about the corrupt file and not about
    // `--session` being broken outright.
    let good = dir.path().join("good.json");
    SessionState::new().save(&good).expect("save");
    let ok = std::process::Command::new(&exe)
        .arg("--session")
        .arg(&good)
        .args(["-c", "memory --batch 8 --seq 512"])
        .output()
        .expect("binary should run");
    assert!(
        ok.status.success(),
        "a valid --session file must still work:\n{}",
        String::from_utf8_lossy(&ok.stderr)
    );
    assert!(String::from_utf8_lossy(&ok.stdout).contains("batch=8"));
}
