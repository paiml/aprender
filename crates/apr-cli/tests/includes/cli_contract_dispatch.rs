// ═══ Contract: cli-dispatch-v1 FALSIFY-CLI-005/006 (PMAT-188) ═══

/// FALSIFY-CLI-005: Code subcommand dispatches to batuta
#[test]
#[cfg(feature = "code")]
fn falsify_cli_005_code_dispatch_wired() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["code", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Sovereign AI"))
        .stdout(predicate::str::contains("--model"))
        .stdout(predicate::str::contains("--resume"))
        .stdout(predicate::str::contains("--project"));
}

/// FALSIFY-CLI-005b: Code subcommand accepts -p flag
#[test]
#[cfg(feature = "code")]
fn falsify_cli_005b_code_print_flag() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["code", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("-p"))
        .stdout(predicate::str::contains("--print"));
}

/// FALSIFY-CLI-005c: Code subcommand accepts --max-turns
#[test]
#[cfg(feature = "code")]
fn falsify_cli_005c_code_max_turns_flag() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["code", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--max-turns"));
}
