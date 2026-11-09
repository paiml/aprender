//! The probe `scripts/nightly_manifest.py` runs on every shipped bin: `<exe> --version`, rc 0, and
//! the build sha on line 1. aprender-rollup answered usage rc 2, so every nightly went red.
use std::process::Command;

fn run(arg: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aprender-rollup"))
        .arg(arg)
        .output()
        .expect("spawn aprender-rollup")
}

#[test]
fn the_built_bin_answers_version_rc_0_with_the_build_sha() {
    for flag in ["--version", "-V"] {
        let o = run(flag);
        assert!(o.status.success(), "{flag}: rc {:?}", o.status.code());
        let out = String::from_utf8_lossy(&o.stdout);
        let want = format!("aprender-rollup {} (", env!("CARGO_PKG_VERSION"));
        let line = out.lines().next().unwrap_or("");
        assert!(
            line.starts_with(&want) && line.ends_with(')'),
            "{flag}: {line}"
        );
        assert!(line.len() > want.len() + 1, "{flag}: empty sha in {line}");
    }
}

#[test]
fn an_unknown_arg_is_still_usage_rc_2() {
    assert_eq!(run("--bogus").status.code(), Some(2));
}

#[test]
fn help_prints_usage_to_stdout_rc_0() {
    for flag in ["--help", "-h"] {
        let o = run(flag);
        assert!(o.status.success(), "{flag}: rc {:?}", o.status.code());
        assert!(
            String::from_utf8_lossy(&o.stdout).starts_with("usage: aprender-rollup"),
            "{flag}: usage not on stdout"
        );
    }
}
