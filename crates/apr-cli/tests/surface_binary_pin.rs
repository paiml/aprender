//! #3745 S1: `apr surface --json` from the built binary is exactly the document
//! the library's `surface::emit()` returns.
//!
//! In-process gates (S3's hand-list guard, S2's cell derivation in lib tests)
//! read `emit()`. The release gate reads the release-candidate binary's
//! `apr surface --json`. If the two could differ, a gate would pass on one
//! surface and ship the other.

use std::process::Command;

fn apr(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_apr"))
        .args(args)
        .output()
        .expect("spawn the apr binary")
}

#[test]
fn the_binary_prints_the_library_surface() {
    let expected = format!("{}\n", apr_cli::surface::emit().to_json());
    for argv in [
        &["surface", "--json"][..],
        &["surface"][..],
        &["surface", "--quiet"][..],
    ] {
        let out = apr(argv);
        assert!(
            out.status.success(),
            "`apr {}` exited {:?}; stderr:\n{}",
            argv.join(" "),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8(out.stdout).expect("utf-8 JSON");
        assert!(
            stdout == expected,
            "`apr {}` stdout differs from surface::emit() ({} vs {} bytes)",
            argv.join(" "),
            stdout.len(),
            expected.len()
        );
        let v: serde_json::Value =
            serde_json::from_str(&stdout).expect("stdout is exactly one JSON document");
        assert_eq!(v["schema"], "apr-cli-surface/v1.1");
    }
}

/// Hidden means absent from `--help`, not absent from the binary.
#[test]
fn surface_is_hidden_from_help() {
    let out = apr(&["--help"]);
    assert!(out.status.success());
    let help = String::from_utf8_lossy(&out.stdout);
    assert!(
        !help.lines().any(|l| l.trim_start().starts_with("surface ")),
        "`surface` must not be listed in `apr --help`"
    );
}
