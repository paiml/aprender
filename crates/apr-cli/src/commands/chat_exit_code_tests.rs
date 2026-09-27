// #3367 — `echo hey | apr chat model.gguf` hit a generate error, printed it as an
// assistant turn (`Assistant: [Error: GGUF generate failed: ...]`) and exited **0**, so a
// gate keyed on the return code read a failed generation as a pass. Measured on v0.67.0
// (e45eaab47) and unchanged on main (eb262f8eb).
//
// Two halves, both pinned here:
//   1. `render_assistant_turn` is the ONE place a generation failure becomes a printed
//      turn, and the only place the session's failure flag is set. The error text still
//      reaches an interactive user unchanged.
//   2. `session_exit_result` is the exit-code decision at session end: any recorded
//      generation failure ends the command with `CliError::InferenceFailed` (exit 8).
//
// The mutation these are written against: drop `*had_generate_error = true;` from
// `render_assistant_turn` and `chat_exit_code_a_generate_error_marks_the_session_failed`
// plus `chat_exit_code_end_to_end_of_the_flag` go RED.

#[test]
fn chat_exit_code_a_generate_error_marks_the_session_failed() {
    let mut had_generate_error = false;
    let line = render_assistant_turn(
        Err("GGUF generate failed: no lm_head".to_string()),
        &mut had_generate_error,
    );
    // (a) the interactive user still sees the error turn, byte for byte as before.
    assert_eq!(line, "[Error: GGUF generate failed: no lm_head]");
    // (b) and the session now remembers that it failed.
    assert!(
        had_generate_error,
        "a generation error must set the session's failure flag (#3367)"
    );
}

#[test]
fn chat_exit_code_a_successful_turn_leaves_the_flag_clear() {
    let mut had_generate_error = false;
    let line = render_assistant_turn(Ok("4".to_string()), &mut had_generate_error);
    assert_eq!(line, "4");
    assert!(
        !had_generate_error,
        "a successful generation must not mark the session failed"
    );
}

#[test]
fn chat_exit_code_is_eight_after_a_failed_generation() {
    let err = session_exit_result(true).expect_err("a failed generation must not exit 0");
    // Exit 8 is `CliError::InferenceFailed` in error.rs — the code `main` maps a command
    // Err to today. Asserted as the numeric value because `ExitCode` has no accessor.
    assert_eq!(err.exit_code_value(), 8);
    assert!(
        format!("{err}").contains("3367") || format!("{err}").contains("generation"),
        "the exit must say why: {err}"
    );
}

#[test]
fn chat_exit_code_is_zero_when_every_turn_generated() {
    assert!(
        session_exit_result(false).is_ok(),
        "a clean session still exits 0"
    );
}

/// The two halves wired together: the flag a rendered error turn sets is the same flag
/// the session-end decision reads. Without this, half 1 could set a flag nothing consumes.
#[test]
fn chat_exit_code_end_to_end_of_the_flag() {
    let mut had_generate_error = false;
    let _ = render_assistant_turn(Ok("fine".to_string()), &mut had_generate_error);
    assert!(session_exit_result(had_generate_error).is_ok());
    let _ = render_assistant_turn(Err("boom".to_string()), &mut had_generate_error);
    assert_eq!(
        session_exit_result(had_generate_error)
            .expect_err("one failed turn fails the session")
            .exit_code_value(),
        8
    );
}

// #3881 — a missing companion file is not a malformed model.
//
// `apr chat` reported an absent `tokenizer.json` as `CliError::InvalidFormat`,
// whose Display hardcodes "Invalid APR format: " and whose exit code is 4 —
// the class that means "this input could not be parsed". The model parsed
// fine; a file beside it was absent. A gate keyed on 4 reads that as a corrupt
// artifact and a human reads it as "my model is broken".
//
// Both halves are asserted: the CODE (3, the FileNotFound class) and the
// WORDING (the message reaches the user unprefixed, because it already names
// every path searched).
#[test]
fn a_missing_companion_file_exits_3_and_does_not_blame_the_model() {
    let msg = "No Qwen tokenizer found for m.apr. Searched:\n1. …";
    let err = CliError::MissingCompanionFile(msg.to_string());

    assert_eq!(
        err.exit_code_value(),
        3,
        "a missing companion file is the FileNotFound class (3), not the \
         unparseable-input class (4)"
    );

    let shown = err.to_string();
    assert!(
        !shown.contains("Invalid APR format"),
        "the message must not blame the model: {shown}"
    );
    assert!(
        shown.starts_with("No Qwen tokenizer found"),
        "the message must reach the user unprefixed: {shown}"
    );
}

/// The contrast that gives the test above its meaning: `InvalidFormat` still
/// exits 4 and still names the APR format. If these two ever agree, the
/// distinction #3881 introduced has been lost.
#[test]
fn invalid_format_still_exits_4_and_still_names_the_format() {
    let err = CliError::InvalidFormat("truncated header".to_string());
    assert_eq!(err.exit_code_value(), 4);
    assert!(err.to_string().contains("Invalid APR format"));
}
