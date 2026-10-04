//! ONT-10 S14: `aprender-train-shell --version` names this binary, not the
//! pre-monorepo `entrenar-shell` (contracts/bin-aprender-train-shell--aprender-train-shell-v1.yaml).

use std::process::Command;

#[test]
fn version_names_the_binary() {
    let out = Command::new(env!("CARGO_BIN_EXE_aprender-train-shell"))
        .arg("--version")
        .output()
        .expect("spawn the binary under test");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let want = format!("aprender-train-shell {}", env!("CARGO_PKG_VERSION"));
    assert!(
        stdout.starts_with(&want),
        "--version printed {stdout:?}, want {want:?}"
    );
}
