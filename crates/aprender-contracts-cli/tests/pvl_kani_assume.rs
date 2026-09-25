//! PVL-001 EV-6c (#4197) — `pv discharge check --kani` and `pv discharge kani-ratchet`, end to end on a tempdir tree.
//!
//! The spec's mutation is a row: add one `kani::assume` → RED. The gate must never write the baseline, and the
//! ratchet must never launder a rise into it.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Fx {
    dir: tempfile::TempDir,
}

impl Fx {
    /// `src/a.rs` holds two assumes (one inside a macro body), `src/b.rs` one; comments and strings never count.
    fn new() -> Self {
        let fx = Self {
            dir: tempfile::tempdir().expect("tempdir"),
        };
        fx.write(
            "crates/src/a.rs",
            "fn p() { kani::assume(x > 0); m!(kani::assume(y)); }\n// kani::assume(z)\nconst S: &str = \"kani::assume\";\n",
        );
        fx.write("crates/src/b.rs", "fn q() { kani::assume(true); }\n");
        fx
    }

    fn write(&self, rel: &str, body: &str) {
        let p = self.dir.path().join(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(p, body).expect("write");
    }

    fn baseline(&self) -> PathBuf {
        self.dir.path().join("contracts/kani-assume-baseline.json")
    }

    fn read_baseline(&self) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(self.baseline()).expect("baseline"))
            .expect("json")
    }

    fn pv(&self, args: &[&str]) -> (i32, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_pv"))
            .args(args)
            .current_dir(self.dir.path())
            .output()
            .expect("spawn pv");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code().unwrap_or(-1), text)
    }

    fn check(&self) -> (i32, String) {
        self.pv(&[
            "discharge",
            "check",
            "--kani",
            "crates",
            "--baseline",
            "contracts/kani-assume-baseline.json",
        ])
    }

    fn ratchet(&self) -> (i32, String) {
        std::fs::create_dir_all(self.dir.path().join("contracts")).expect("mkdir");
        self.pv(&[
            "discharge",
            "kani-ratchet",
            "crates",
            "--baseline",
            "contracts/kani-assume-baseline.json",
        ])
    }
}

fn mtime_and_bytes(p: &Path) -> (std::time::SystemTime, Vec<u8>) {
    (
        std::fs::metadata(p)
            .expect("meta")
            .modified()
            .expect("mtime"),
        std::fs::read(p).expect("read"),
    )
}

#[test]
fn the_ratchet_seeds_the_baseline_with_the_command_recorded() {
    let fx = Fx::new();
    let (rc, out) = fx.ratchet();
    assert_eq!(rc, 0, "{out}");
    let b = fx.read_baseline();
    assert_eq!(b["command"], "make kani-ratchet");
    assert_eq!(b["total"], 3);
    assert_eq!(b["files"]["src/a.rs"], 2);
    assert_eq!(b["files"]["src/b.rs"], 1);
}

#[test]
fn check_accepts_at_the_baseline_and_never_writes_it() {
    let fx = Fx::new();
    assert_eq!(fx.ratchet().0, 0);
    let before = mtime_and_bytes(&fx.baseline());
    let (rc, out) = fx.check();
    assert_eq!(rc, 0, "{out}");
    assert!(out.contains("accept:"), "{out}");
    assert_eq!(
        mtime_and_bytes(&fx.baseline()),
        before,
        "the gate wrote the baseline"
    );
}

/// The spec's mutation: add one `kani::assume` → RED, naming the file, and the baseline is untouched.
#[test]
fn one_more_assume_is_red_and_names_the_file() {
    let fx = Fx::new();
    assert_eq!(fx.ratchet().0, 0);
    let before = mtime_and_bytes(&fx.baseline());
    fx.write(
        "crates/src/b.rs",
        "fn q() { kani::assume(true); kani::assume(false); }\n",
    );
    let (rc, out) = fx.check();
    assert_eq!(rc, 1, "{out}");
    assert!(out.contains("src/b.rs: kani::assume 1 -> 2"), "{out}");
    assert_eq!(
        mtime_and_bytes(&fx.baseline()),
        before,
        "the gate wrote the baseline"
    );
}

#[test]
fn an_assume_in_a_new_file_is_red() {
    let fx = Fx::new();
    assert_eq!(fx.ratchet().0, 0);
    fx.write("crates/src/c.rs", "fn r() { kani::assume(true); }\n");
    let (rc, out) = fx.check();
    assert_eq!(rc, 1, "{out}");
    assert!(out.contains("src/c.rs: kani::assume 0 -> 1"), "{out}");
}

#[test]
fn a_fall_is_green_and_only_the_ratchet_lowers_the_baseline() {
    let fx = Fx::new();
    assert_eq!(fx.ratchet().0, 0);
    fx.write("crates/src/a.rs", "fn p() { kani::assume(x > 0); }\n");
    assert_eq!(fx.check().0, 0);
    assert_eq!(
        fx.read_baseline()["files"]["src/a.rs"],
        2,
        "check lowered the baseline"
    );
    assert_eq!(fx.ratchet().0, 0);
    assert_eq!(fx.read_baseline()["files"]["src/a.rs"], 1);
    assert_eq!(fx.read_baseline()["total"], 2);
}

/// `make kani-ratchet` after a rise: rc 1, and the rewritten baseline still rejects it.
#[test]
fn the_ratchet_cannot_launder_a_rise() {
    let fx = Fx::new();
    assert_eq!(fx.ratchet().0, 0);
    fx.write(
        "crates/src/b.rs",
        "fn q() { kani::assume(true); kani::assume(false); }\n",
    );
    let (rc, out) = fx.ratchet();
    assert_eq!(rc, 1, "{out}");
    assert_eq!(fx.read_baseline()["files"]["src/b.rs"], 1);
    assert_eq!(fx.check().0, 1);
}

#[test]
fn a_missing_or_inconsistent_baseline_fails_closed() {
    let fx = Fx::new();
    let (rc, out) = fx.check();
    assert_eq!(rc, 1, "missing baseline: {out}");
    fx.write(
        "contracts/kani-assume-baseline.json",
        r#"{"command":"make kani-ratchet","total":9,"files":{"src/a.rs":2,"src/b.rs":1}}"#,
    );
    let (rc, out) = fx.check();
    assert_eq!(rc, 1, "inconsistent baseline: {out}");
    assert!(out.contains("not the sum"), "{out}");
}

#[test]
fn an_untokenizable_file_fails_closed_not_zero() {
    let fx = Fx::new();
    assert_eq!(fx.ratchet().0, 0);
    fx.write("crates/src/bad.rs", "fn f() { \"open");
    let (rc, out) = fx.check();
    assert_eq!(rc, 1, "{out}");
    assert!(out.contains("bad.rs"), "{out}");
}

#[test]
fn the_lean_arms_are_refused_with_kani() {
    let fx = Fx::new();
    assert_eq!(fx.ratchet().0, 0);
    let (rc, out) = fx.pv(&[
        "discharge",
        "check",
        "--kani",
        "crates",
        "--baseline",
        "contracts/kani-assume-baseline.json",
        "--strict",
    ]);
    assert_eq!(rc, 2, "clap usage error expected: {out}");
    for extra in [["--lake-timeout", "5"], ["--contracts", "contracts"]] {
        let mut args = vec![
            "discharge",
            "check",
            "--kani",
            "crates",
            "--baseline",
            "contracts/kani-assume-baseline.json",
        ];
        args.extend(extra);
        let (rc, out) = fx.pv(&args);
        assert_eq!(
            rc, 2,
            "{extra:?} with --kani must be refused, not ignored: {out}"
        );
    }
    let (rc, out) = fx.pv(&["discharge", "check", "--kani", "crates"]);
    assert_eq!(rc, 2, "--kani without --baseline: {out}");
}
