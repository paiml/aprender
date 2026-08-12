//! Resolution of the `apr` binary this server delegates to.
//!
//! Every subprocess-backed MCP tool shells out to `apr <subcommand> --json`.
//! Until this module existed, that spawn was a literal `Command::new("apr")`,
//! which asks the operating system to search `$PATH`. That is the exact
//! anti-pattern `CLAUDE.md` opens with ("NEVER hardcode or PATH-resolve an
//! `apr` binary"), and it produced a wrong-answer channel in the field:
//!
//! * `apr mcp` launched from the freshly installed 0.63.0 artifact executed
//!   `/home/noah/.local/bin/apr`, which is 0.60.0, for all eight subprocess
//!   tools — while `apr.version` kept answering `0.63.0` from in-process
//!   state. The one tool a client uses to establish provenance reported a
//!   version that none of the other tools actually ran.
//! * A user who runs the binary by path without putting its install directory
//!   on `$PATH` gets `Failed to spawn ...: No such file or directory` from
//!   eight of nine tools.
//!
//! [`apr_binary`] fixes both: when the running executable *is* `apr`, the
//! server delegates to **itself**, so `apr mcp` from 0.63.0 runs 0.63.0.
//!
//! # Resolution order
//!
//! 1. `$APR_BIN`, if set and non-empty. Escape hatch for embedders and for
//!    tests that need to point the server at a mock.
//! 2. [`std::env::current_exe`], **if its file stem is exactly `apr`**. The
//!    stem check is what keeps the library usable outside the `apr` binary:
//!    under `cargo test` the current executable is
//!    `target/debug/deps/aprender_mcp-<hash>`, which must not be spawned with
//!    `validate model.gguf --json`.
//! 3. The bare name `apr`, resolved by the OS through `$PATH`. Reached only
//!    when the host process is not `apr` itself.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Environment variable that overrides binary resolution entirely.
pub const APR_BIN_ENV: &str = "APR_BIN";

/// The program the subprocess-backed tools should execute.
///
/// See the [module docs](self) for the resolution order.
#[must_use]
pub fn apr_binary() -> PathBuf {
    resolve(std::env::var_os(APR_BIN_ENV), std::env::current_exe().ok())
}

/// Pure core of [`apr_binary`], parameterised over the two pieces of process
/// state it reads so the resolution policy is testable without mutating the
/// environment of the running test process.
#[must_use]
pub fn resolve(override_var: Option<OsString>, current_exe: Option<PathBuf>) -> PathBuf {
    if let Some(explicit) = override_var {
        if !explicit.is_empty() {
            return PathBuf::from(explicit);
        }
    }
    if let Some(exe) = current_exe {
        if is_apr_binary(&exe) {
            return exe;
        }
    }
    PathBuf::from("apr")
}

/// True when `path` names the `apr` CLI itself (`apr`, or `apr.exe` on
/// Windows). Deliberately an exact stem match: `aprender_mcp-1a2b3c` and
/// `apr-cli` are *not* `apr`, and spawning them with `apr` subcommands would
/// be worse than falling back to `$PATH`.
fn is_apr_binary(path: &Path) -> bool {
    path.file_stem().is_some_and(|stem| stem == "apr")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write an executable shell script at `path` that prints `marker`.
    fn write_marker_bin(path: &Path, marker: &str) {
        let mut f = std::fs::File::create(path).expect("create marker bin");
        writeln!(f, "#!/bin/sh").expect("shebang");
        writeln!(f, "echo {marker}").expect("body");
        f.sync_all().expect("sync");
        drop(f);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(path).expect("stat").permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(path, perms).expect("chmod");
        }
    }

    /// Execute a just-written shim, retrying only on ETXTBSY.
    ///
    /// `write_marker_bin` already syncs and drops its own handle, so the fd
    /// that makes the file "busy" is not ours. Under `cargo test --workspace
    /// --lib` this module's neighbours in `tools::subprocess::tests` spawn
    /// subprocesses concurrently; if one of them forks in the window where our
    /// write fd to the shim is still open, the forked child inherits that fd
    /// and holds it until its own exec. Our exec of the shim then fails with
    /// ETXTBSY. `O_CLOEXEC` closes the fd at the child's exec but not before
    /// it, so the window is real — which is why this test passed standalone
    /// (`-p aprender-mcp --lib`) and failed under `--workspace --lib` with
    /// `spawn resolved program .../apr: Text file busy (os error 26)`.
    ///
    /// Retrying cannot mask the defect under test: only ETXTBSY is retried,
    /// the bound is one second, and a shim that is wrong or never becomes
    /// executable still fails. `exec_marker_bin_survives_a_transient_etxtbsy`
    /// holds a write fd open on purpose to prove both halves.
    #[cfg(unix)]
    fn exec_marker_bin(path: &Path) -> std::process::Output {
        const ETXTBSY: i32 = 26;
        let mut last = String::new();
        for _ in 0..100 {
            match std::process::Command::new(path).output() {
                Ok(out) => return out,
                Err(e) if e.raw_os_error() == Some(ETXTBSY) => {
                    last = e.to_string();
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => panic!("spawn {}: {e}", path.display()),
            }
        }
        panic!(
            "spawn {} still busy after 100 attempts: {last}",
            path.display()
        );
    }

    /// FALSIFIER: the ETXTBSY window is real, and `exec_marker_bin` rides it out.
    ///
    /// Holding a write handle open reproduces exactly the state a forked
    /// sibling leaves the shim in. A direct spawn must fail with ETXTBSY (if it
    /// does not, the premise of the retry is wrong and this test says so);
    /// the retrying helper must then succeed once the handle drops. Deleting
    /// the retry loop turns this RED deterministically.
    #[test]
    #[cfg(unix)]
    fn exec_marker_bin_survives_a_transient_etxtbsy() {
        let dir = scratch_dir("etxtbsy");
        let shim = dir.join("apr");
        write_marker_bin(&shim, "RETRY-MARKER");

        let held = std::fs::OpenOptions::new()
            .write(true)
            .open(&shim)
            .expect("hold a write fd open");
        let direct = std::process::Command::new(&shim).output();
        assert_eq!(
            direct.err().and_then(|e| e.raw_os_error()),
            Some(26),
            "an open write fd must make a direct spawn fail with ETXTBSY; without that \
             the retry loop is guarding nothing"
        );

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            drop(held);
        });

        let out = exec_marker_bin(&shim);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "RETRY-MARKER",
            "the helper must retry through ETXTBSY and then run the shim"
        );
    }

    /// Per-process, per-call scratch dir. A fixed path would let two
    /// concurrent runs of this test binary delete each other's shim.
    fn scratch_dir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "aprender-mcp-apr-bin-{name}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir scratch");
        dir
    }

    /// FALSIFIER (#2384): when the running executable *is* `apr`, resolution
    /// must yield that exact executable — not the bare name `apr`, which the
    /// OS would resolve through `$PATH` to whatever stale `apr` happens to be
    /// installed first.
    ///
    /// Behavioural, not shape-based: we execute the resolved program and
    /// assert it is the one we designated as "self". Before the fix, resolve
    /// returned `PathBuf::from("apr")`, which is not executable as written
    /// (no such relative file) and is not the self binary.
    #[test]
    #[cfg(unix)]
    fn resolution_executes_the_current_executable_not_a_path_lookup() {
        let dir = scratch_dir("self");
        let self_apr = dir.join("apr");
        write_marker_bin(&self_apr, "SELF-BINARY-UNDER-TEST");

        let resolved = resolve(None, Some(self_apr.clone()));
        assert_eq!(
            resolved,
            self_apr,
            "resolution must return the running executable, got {}",
            resolved.display()
        );

        let out = exec_marker_bin(&resolved);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "SELF-BINARY-UNDER-TEST",
            "the resolved program must be the current executable"
        );
    }

    /// The two `apr` binaries in the field bug differ only by directory, so a
    /// basename comparison would have passed while the defect was live. Assert
    /// on the full path: resolving from `/a/apr` must never yield `/b/apr`.
    #[test]
    fn resolution_keeps_the_directory_of_the_current_executable() {
        let a = PathBuf::from("/opt/release-0.63.0/bin/apr");
        let b = PathBuf::from("/home/user/.local/bin/apr");
        assert_eq!(resolve(None, Some(a.clone())), a);
        assert_eq!(resolve(None, Some(b.clone())), b);
        assert_ne!(resolve(None, Some(a)), b);
    }

    /// Library-embedded use (and every `cargo test` run) must still fall back
    /// to `$PATH`: the current executable is a test harness, and spawning it
    /// with `validate model.gguf --json` would be nonsense.
    #[test]
    fn non_apr_host_process_falls_back_to_the_path_name() {
        let harness = PathBuf::from("/w/target/debug/deps/aprender_mcp-1a2b3c4d");
        assert_eq!(resolve(None, Some(harness)), PathBuf::from("apr"));
        assert_eq!(resolve(None, None), PathBuf::from("apr"));
    }

    /// `apr-cli`, `aprender`, `apr_serve` are not `apr`. Exact stem only.
    #[test]
    fn similar_names_are_not_treated_as_apr() {
        for name in ["apr-cli", "aprender", "apr_serve", "aprx"] {
            let exe = PathBuf::from("/usr/bin").join(name);
            assert_eq!(
                resolve(None, Some(exe)),
                PathBuf::from("apr"),
                "{name} must not be mistaken for the apr binary"
            );
        }
    }

    /// An `.exe` suffix is stripped by `file_stem`, so `apr.exe` is `apr`.
    /// (Written with `/` separators so the assertion means the same thing on
    /// every host — `\` is not a separator on Unix.)
    #[test]
    fn exe_suffix_is_recognised() {
        let exe = PathBuf::from("/Program Files/apr/apr.exe");
        assert_eq!(resolve(None, Some(exe.clone())), exe);
    }

    /// `$APR_BIN` wins over self-resolution, and an empty value is ignored
    /// (an exported-but-empty variable must not spawn `""`).
    #[test]
    fn explicit_override_wins_and_empty_is_ignored() {
        let exe = PathBuf::from("/opt/bin/apr");
        assert_eq!(
            resolve(Some(OsString::from("/mock/apr")), Some(exe.clone())),
            PathBuf::from("/mock/apr")
        );
        assert_eq!(resolve(Some(OsString::new()), Some(exe.clone())), exe);
    }
}
