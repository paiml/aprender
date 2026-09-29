//! Which `apr` program this crate's subprocess runners execute (L25).
//!
//! Every runner here used to default to the bare name `"apr"`, which asks the
//! OS to search `$PATH`. Under `cargo test` that made 40 unit tests in the
//! required `workspace-test` job depend on whatever `apr` the runner happened
//! to have: they passed with the fleet 0.69 binary, with a 0.70 binary, and
//! with no `apr` at all. A test that cannot fail is not a test (L25).
//!
//! * Production builds keep the documented default, `"apr"` from `$PATH`.
//! * Unit-test builds never do. [`default_apr_binary`] returns either the
//!   operator-pinned real binary (`APR_BIN` + `APR_BIN_SHA256`, both checked),
//!   or a hermetic stub whose bytes are fixed in this file and re-hashed on
//!   every call. And [`guard_program`] panics at every spawn site when a test
//!   hands a runner a program that is not an absolute path, so a new test that
//!   writes `with_binary("apr")` fails loudly instead of passing vacuously.

/// The program a runner uses when the caller did not name one.
#[cfg(not(test))]
#[must_use]
pub fn default_apr_binary() -> String {
    "apr".to_string()
}

/// The program a runner uses when the caller did not name one. Test build:
/// the pinned real binary, or the hermetic stub. Never a `$PATH` lookup.
#[cfg(test)]
#[must_use]
pub fn default_apr_binary() -> String {
    test_bin::resolve().to_string_lossy().into_owned()
}

/// Spawn-site guard. A no-op in production builds; in unit-test builds, a
/// program that is not an absolute path is a `$PATH` lookup and panics.
#[inline]
pub(crate) fn guard_program(program: &str) -> &str {
    #[cfg(test)]
    assert!(
        std::path::Path::new(program).is_absolute(),
        "L25: a unit test spawned {program:?}, which resolves through $PATH. \
         Use crate::apr_bin::default_apr_binary() (pinned real apr or the \
         sha256-checked stub) or an explicit absolute path."
    );
    program
}

/// A program named exactly `apr` becomes [`default_apr_binary`]; any other
/// program (`ollama`, `curl`, …) is returned unchanged. For the call sites
/// that split a command string and spawn its first word.
#[must_use]
pub(crate) fn map_apr(program: &str) -> std::borrow::Cow<'_, str> {
    if program == "apr" {
        std::borrow::Cow::Owned(default_apr_binary())
    } else {
        std::borrow::Cow::Borrowed(program)
    }
}

#[cfg(test)]
pub(crate) mod test_bin {
    use sha2::{Digest, Sha256};
    use std::path::{Path, PathBuf};

    /// The stub `apr`: echoes its argv one per line on stdout, names itself on
    /// stderr, exits 3. A test that sees exit 3 knows the stub ran; a test
    /// that sees -1 knows nothing ran.
    pub(crate) const STUB: &[u8] =
        b"#!/bin/sh\n# apr-l25-stub\nprintf '%s\\n' \"$@\"\necho 'apr-l25-stub: not a real apr' >&2\nexit 3\n";

    /// Exit code the stub returns.
    pub(crate) const STUB_EXIT: i32 = 3;

    pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// `APR_BIN` (+ `APR_BIN_SHA256`) when set, else the stub.
    pub(crate) fn resolve() -> PathBuf {
        match std::env::var_os("APR_BIN").filter(|v| !v.is_empty()) {
            Some(bin) => pinned(Path::new(&bin), std::env::var("APR_BIN_SHA256").ok()),
            None => stub(),
        }
    }

    /// The operator-pinned real binary: absolute, and its bytes hash to the
    /// declared sha256. Anything else panics; there is no fallback.
    pub(crate) fn pinned(bin: &Path, want: Option<String>) -> PathBuf {
        assert!(
            bin.is_absolute(),
            "L25: APR_BIN={} is not an absolute path",
            bin.display()
        );
        let want = want.unwrap_or_else(|| {
            panic!(
                "L25: APR_BIN={} is set without APR_BIN_SHA256",
                bin.display()
            )
        });
        let bytes = std::fs::read(bin)
            .unwrap_or_else(|e| panic!("L25: APR_BIN={} unreadable: {e}", bin.display()));
        let got = sha256_hex(&bytes);
        assert_eq!(
            got,
            want.trim().to_ascii_lowercase(),
            "L25: APR_BIN={} sha256 differs from APR_BIN_SHA256",
            bin.display()
        );
        bin.to_path_buf()
    }

    /// The stub, materialised once per content hash under the temp dir, and
    /// re-verified byte-for-byte before every use.
    pub(crate) fn stub() -> PathBuf {
        let sha = sha256_hex(STUB);
        let dir = std::env::temp_dir().join(format!("apr-l25-stub-{}", &sha[..16]));
        let path = dir.join("apr");
        if std::fs::read(&path).map(|b| sha256_hex(&b)).ok().as_deref() != Some(sha.as_str()) {
            std::fs::create_dir_all(&dir).expect("L25: create stub dir");
            // Unique temp name + rename: parallel test processes never see a
            // half-written stub.
            let tmp = dir.join(format!(".apr.{}", std::process::id()));
            std::fs::write(&tmp, STUB).expect("L25: write stub");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))
                    .expect("L25: chmod stub");
            }
            std::fs::rename(&tmp, &path).expect("L25: install stub");
        }
        let got = sha256_hex(&std::fs::read(&path).expect("L25: read stub"));
        assert_eq!(got, sha, "L25: stub at {} was altered", path.display());
        path
    }
}

#[cfg(test)]
mod tests {
    use super::test_bin::{pinned, sha256_hex, stub, STUB, STUB_EXIT};
    use super::*;

    #[test]
    fn default_is_absolute_and_never_bare() {
        let bin = default_apr_binary();
        assert!(std::path::Path::new(&bin).is_absolute(), "{bin}");
        assert_ne!(bin, "apr");
    }

    #[test]
    fn stub_runs_echoes_argv_and_exits_3() {
        let out = std::process::Command::new(stub())
            .args(["qa", "m.gguf"])
            .output()
            .expect("spawn stub");
        assert_eq!(out.status.code(), Some(STUB_EXIT));
        assert_eq!(String::from_utf8_lossy(&out.stdout), "qa\nm.gguf\n");
    }

    #[test]
    #[should_panic(expected = "resolves through $PATH")]
    fn guard_panics_on_bare_apr() {
        let _ = guard_program("apr");
    }

    /// The planted bad call: a test that hands a runner the bare name must
    /// fail loudly at the spawn, not pass on whatever `apr` `$PATH` holds.
    #[test]
    #[should_panic(expected = "resolves through $PATH")]
    fn planted_bare_apr_runner_is_red() {
        use crate::command::{CommandRunner, RealCommandRunner};
        let _ = RealCommandRunner::with_binary("apr").inspect_model(std::path::Path::new("m.gguf"));
    }

    /// The default runner reaches the pinned binary or the stub, never `$PATH`.
    #[test]
    fn default_runner_reaches_the_resolved_binary() {
        use crate::command::{CommandRunner, RealCommandRunner};
        let out = RealCommandRunner::new().inspect_model(std::path::Path::new("m.gguf"));
        if std::env::var_os("APR_BIN").is_none_or(|v| v.is_empty()) {
            assert_eq!(out.exit_code, STUB_EXIT, "{out:?}");
            assert!(out.stderr.contains("apr-l25-stub"), "{out:?}");
        } else {
            assert!(!out.stderr.contains("apr-l25-stub"), "{out:?}");
        }
    }

    #[test]
    fn guard_passes_an_absolute_path() {
        assert_eq!(guard_program("/nonexistent/apr"), "/nonexistent/apr");
    }

    #[test]
    #[should_panic(expected = "not an absolute path")]
    fn pinned_rejects_relative() {
        let _ = pinned(std::path::Path::new("apr"), Some(sha256_hex(STUB)));
    }

    #[test]
    #[should_panic(expected = "without APR_BIN_SHA256")]
    fn pinned_rejects_missing_sha() {
        let _ = pinned(&stub(), None);
    }

    #[test]
    #[should_panic(expected = "sha256 differs")]
    fn pinned_rejects_wrong_sha() {
        let _ = pinned(&stub(), Some("0".repeat(64)));
    }

    #[test]
    fn pinned_accepts_matching_sha() {
        let s = stub();
        assert_eq!(pinned(&s, Some(sha256_hex(STUB).to_uppercase())), s);
    }
}
