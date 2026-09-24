//! PVL-001 EV-6b (#4199) — `pv discharge check --leanchecker`, end to end with a stub `lake` on PATH.
//!
//! The spec's RED, verbatim: a stub `lake` returning 1 for `leanchecker` → rc 1; no `lake` at all → rc 2. Plus
//! the case the rc alone cannot tell apart (measured on elan v4.15.0: `lake env leanchecker` exits 1 "does not
//! have the binary"): a toolchain WITHOUT `bin/leanchecker` declines, rc 2, and is never read as a failed check.
//! The real re-check needs the built tree and Mathlib; it is measured on lambda (see #4199).

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

struct Fx {
    dir: tempfile::TempDir,
}

impl Fx {
    /// A one-theorem tree with `Axioms.lean` generated, so everything before `--leanchecker` accepts.
    fn new() -> Self {
        let fx = Self {
            dir: tempfile::tempdir().expect("tempdir"),
        };
        fx.write(
            "lean/ProvableContracts.lean",
            "import ProvableContracts.Theorems.Gelu.Bound\n",
        );
        fx.write(
            "lean/ProvableContracts/Theorems/Gelu/Bound.lean",
            "namespace ProvableContracts.Gelu\ntheorem gelu_bound : True := trivial\nend ProvableContracts.Gelu\n",
        );
        fx.write(
            "contracts/gelu-v1.yaml",
            "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n",
        );
        let g = fx.pv(&["discharge", "gen-axioms", "lean"]);
        assert_eq!(g.0, 0, "gen-axioms: {}", g.1);
        fx
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir.path().join(rel)
    }

    fn write(&self, rel: &str, text: &str) {
        let p = self.path(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(p, text).expect("write");
    }

    /// `bin/lake` on the test's PATH: `printenv` names a sysroot (with `bin/leanchecker` when `has_checker`),
    /// `lean` passes, `leanchecker` prints a line and exits `rc`.
    fn stub_lake(&self, has_checker: bool, rc: i32) {
        let root = self.path("sysroot");
        std::fs::create_dir_all(root.join("bin")).expect("mkdir");
        if has_checker {
            std::fs::write(root.join("bin/leanchecker"), "").expect("w");
        }
        self.write(
            "bin/lake",
            &format!(
                "#!/bin/sh\ncase \"$2\" in\n  printenv) echo '{}' ;;\n  lean) exit 0 ;;\n  leanchecker) echo 'stub leanchecker says {rc}'; exit {rc} ;;\n  *) exit 99 ;;\nesac\n",
                root.display()
            ),
        );
        let p = self.path("bin/lake");
        let mut perm = std::fs::metadata(&p).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&p, perm).expect("chmod");
    }

    /// Runs pv with PATH = the fixture's `bin/` (where the stub lake is, if any), then the system dirs that hold
    /// `sh`, `timeout` and `env` — and no real `lake`.
    fn pv(&self, args: &[&str]) -> (i32, String) {
        let path = format!("{}:/usr/bin:/bin", self.path("bin").display());
        let out = Command::new(pv_bin())
            .current_dir(self.dir.path())
            .env("PATH", path)
            .args(args)
            .output()
            .expect("spawn pv");
        (
            out.status.code().unwrap_or(-1),
            format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ),
        )
    }

    fn check(&self) -> (i32, String) {
        self.pv(&[
            "discharge",
            "check",
            "lean",
            "--leanchecker",
            "--leanchecker-timeout",
            "60",
        ])
    }
}

fn assert_rc(r: &(i32, String), code: i32, needle: &str) {
    assert_eq!(r.0, code, "{}", r.1);
    assert!(r.1.contains(needle), "missing {needle:?}\n{}", r.1);
}

#[test]
fn a_passing_leanchecker_accepts() {
    let fx = Fx::new();
    fx.stub_lake(true, 0);
    assert_rc(
        &fx.check(),
        0,
        "ok    lake env leanchecker ProvableContracts",
    );
}

#[test]
fn a_failing_leanchecker_rejects_with_its_output() {
    let fx = Fx::new();
    fx.stub_lake(true, 1);
    let r = fx.check();
    assert_rc(&r, 1, "FAIL  lake env leanchecker ProvableContracts");
    assert!(r.1.contains("stub leanchecker says 1"), "{}", r.1);
}

#[test]
fn no_lake_declines() {
    let fx = Fx::new();
    assert_rc(&fx.check(), 2, "decline:");
}

#[test]
fn a_toolchain_without_leanchecker_declines_and_is_not_a_failure() {
    let fx = Fx::new();
    fx.stub_lake(false, 1);
    assert_rc(&fx.check(), 2, "leanchecker not in toolchain");
}

#[test]
fn leanchecker_and_no_lake_conflict() {
    let fx = Fx::new();
    let r = fx.pv(&["discharge", "check", "lean", "--no-lake", "--leanchecker"]);
    assert_eq!(r.0, 2, "clap refuses the pair: {}", r.1);
    assert!(r.1.contains("cannot be used with"), "{}", r.1);
}

/// The committed lean tree's `formalization.yaml` states the non-fresh limitation (PVL-001 EV-6b: "the
/// limitation is stated in formalization.yaml scope").
#[test]
fn the_real_trees_scope_states_the_non_fresh_limitation() {
    let f = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../aprender-contracts-staging/lean/formalization.yaml");
    let text = std::fs::read_to_string(&f).expect("formalization.yaml");
    let v: serde_yaml::Value = serde_yaml::from_str(&text).expect("yaml");
    let scope = v["scope"].as_str().expect("scope: is a string");
    for needle in ["--leanchecker", "--fresh", "F7"] {
        assert!(scope.contains(needle), "scope lacks {needle:?}: {scope}");
    }
    assert!(
        scope.to_lowercase().contains("non-fresh"),
        "scope lacks \"non-fresh\": {scope}"
    );
}
