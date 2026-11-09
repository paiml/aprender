//! `run_python_cell`: the one place the judge starts a Python interpreter.
//!
//! A `code_tests` prompt asks the model for Python, so the reply can only be
//! tested by running it. This seam runs the model's reply plus the prompt's
//! asserts, and nothing else, in the sandbox the Python oracle used: a fresh
//! 0700 tmpdir holding `cell.py`, `prlimit` CPU/address-space/file-size
//! limits, `unshare -rn` (a network namespace with no interfaces), `python3
//! -I`, an environment of `PATH=/usr/bin:/bin` only, stdin from /dev/null,
//! and a sentinel printed after the last assert. It is the operator's
//! external-validation exception to "no Python"; a follow-up replaces it
//! with a Python-free executor.
//!
//! No interpreter, a failed probe, a tmpdir error or a spawn error is
//! `Unavailable`: the oracle reads it as `sandbox_unavailable`, never
//! correct. The probe runs once per process, so one failed probe makes every
//! `code_tests` row of that run `sandbox_unavailable`.

use std::io::Read;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{mpsc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The cell's whole environment, and where `prlimit`/`unshare` are looked up.
const CELL_PATH: &str = "/usr/bin:/bin";
/// RLIMIT_AS and RLIMIT_FSIZE, as in the Python oracle.
const CELL_AS: u64 = 1 << 30;
const CELL_FSIZE: u64 = 1 << 24;
/// The probe's own bound: `python3 -V` under the sandbox.
const PROBE_S: u64 = 10;

/// A probed sandbox: absolute paths of the three programs, the directory the
/// cell tmpdirs are made in, and what the interpreter says its version is
/// (`python3 -V`).
#[derive(Debug, Clone)]
pub struct Sandbox {
    prlimit: PathBuf,
    unshare: PathBuf,
    python: PathBuf,
    tmp: PathBuf,
    pub version: String,
}

impl Sandbox {
    /// This sandbox with `prlimit` swapped for `path`: a spawn-error row.
    #[cfg(test)]
    pub fn with_prlimit(&self, path: &str) -> Sandbox {
        Sandbox {
            prlimit: PathBuf::from(path),
            ..self.clone()
        }
    }

    /// This sandbox making its tmpdirs under `path`: a tmpdir-error row.
    #[cfg(test)]
    pub fn with_tmp(&self, path: &str) -> Sandbox {
        Sandbox {
            tmp: PathBuf::from(path),
            ..self.clone()
        }
    }
}

/// How one cell ended.
#[derive(Debug)]
pub enum CellRun {
    /// The sandbox could not start the cell (spawn error, tmpdir error).
    Unavailable(String),
    /// The wall-clock bound passed; the cell was killed.
    Timeout,
    /// The interpreter exited: Popen's returncode (minus the signal number
    /// when a signal ended it) and the raw output.
    Exited {
        rc: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
}

/// The first executable `name` in the `:`-separated `dirs`. Relative entries
/// are skipped: the program runs with the cell tmpdir as its cwd.
fn which_in(dirs: &str, name: &str) -> Option<PathBuf> {
    dirs.split(':')
        .map(Path::new)
        .filter(|d| d.is_absolute())
        .map(|d| d.join(name))
        .find(|p| {
            p.metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        })
}

/// The sandbox of this host, probed once per process.
pub fn sandbox() -> Result<&'static Sandbox, &'static str> {
    static PROBED: OnceLock<Result<Sandbox, String>> = OnceLock::new();
    PROBED
        .get_or_init(|| {
            let path = std::env::var("PATH").unwrap_or_default();
            let python = which_in(&path, "python3").ok_or("no python3 on PATH")?;
            let prlimit = which_in(CELL_PATH, "prlimit").ok_or("no prlimit in /usr/bin:/bin")?;
            let unshare = which_in(CELL_PATH, "unshare").ok_or("no unshare in /usr/bin:/bin")?;
            probe(prlimit, unshare, python)
        })
        .as_ref()
        .map_err(String::as_str)
}

/// Run `python3 -I -V` inside the full sandbox: the probe passes only when
/// every layer starts and the interpreter answers with its version.
pub fn probe(prlimit: PathBuf, unshare: PathBuf, python: PathBuf) -> Result<Sandbox, String> {
    let dir = std::env::temp_dir();
    let mut cmd = sandboxed(&prlimit, &unshare, Some(PROBE_S));
    cmd.arg(&python).args(["-I", "-V"]).current_dir(&dir);
    match wait_bounded(cmd, PROBE_S) {
        CellRun::Exited { rc: 0, stdout, .. } => {
            let version = String::from_utf8_lossy(&stdout).trim().to_string();
            if version.starts_with("Python ") {
                Ok(Sandbox {
                    prlimit,
                    unshare,
                    python,
                    tmp: dir,
                    version,
                })
            } else {
                Err(format!("probe printed {version:?}, not a Python version"))
            }
        }
        CellRun::Exited { rc, stderr, .. } => Err(format!(
            "probe exited {rc}: {}",
            String::from_utf8_lossy(&stderr).trim()
        )),
        CellRun::Timeout => Err(format!("probe took over {PROBE_S}s")),
        CellRun::Unavailable(why) => Err(why),
    }
}

/// `prlimit <limits> -- unshare -rn`, with the scrubbed environment; the
/// caller appends the interpreter and its arguments. `None` CPU seconds is
/// RLIM_INFINITY, which is what `setrlimit` reads a -1 as.
fn sandboxed(prlimit: &Path, unshare: &Path, cpu_s: Option<u64>) -> Command {
    let cpu = cpu_s.map_or_else(|| "unlimited".to_string(), |s| s.to_string());
    let mut cmd = Command::new(prlimit);
    cmd.arg(format!("--cpu={cpu}:{cpu}"))
        .arg(format!("--as={CELL_AS}:{CELL_AS}"))
        .arg(format!("--fsize={CELL_FSIZE}:{CELL_FSIZE}"))
        .arg("--")
        .arg(unshare)
        .arg("-rn")
        .env_clear()
        .env("PATH", CELL_PATH)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// A new 0700 directory `crux-code-*` under `base`.
fn fresh_dir(base: &Path) -> std::io::Result<PathBuf> {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut last = None;
    for n in 0..100u32 {
        let dir = base.join(format!("crux-code-{}-{seed:x}-{n}", std::process::id()));
        match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no fresh tmpdir")))
}

/// Read one pipe to its end on its own thread.
fn drain(mut pipe: impl Read + Send + 'static, tx: mpsc::Sender<(bool, Vec<u8>)>, is_err: bool) {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = pipe.read_to_end(&mut buf);
        let _ = tx.send((is_err, buf));
    });
}

/// Kill a cell that ran past its wall bound and reap it.
fn kill_timed_out(child: &mut Child) -> CellRun {
    let _ = child.kill();
    let _ = child.wait();
    CellRun::Timeout
}

/// Collect both pipes' ends from `rx` before `deadline`: `None` on timeout.
fn read_pipes(
    rx: &mpsc::Receiver<(bool, Vec<u8>)>,
    deadline: Instant,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
    for _ in 0..2 {
        let left = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok((true, b)) => stderr = b,
            Ok((false, b)) => stdout = b,
            Err(mpsc::RecvTimeoutError::Timeout) => return None,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Some((stdout, stderr))
}

/// Wait for `child` to exit before `deadline`; past it, kill it.
fn wait_until(child: &mut Child, deadline: Instant) -> Result<ExitStatus, CellRun> {
    loop {
        match child.try_wait() {
            Ok(Some(s)) => return Ok(s),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            Ok(None) => return Err(kill_timed_out(child)),
            Err(e) => return Err(CellRun::Unavailable(format!("wait: {e}"))),
        }
    }
}

/// Spawn `cmd`, read both pipes, and wait at most `bound_s` seconds for the
/// exit AND both pipes' end, as `subprocess.run(..., timeout=)` does.
fn wait_bounded(mut cmd: Command, bound_s: u64) -> CellRun {
    let now = Instant::now();
    let deadline = now
        .checked_add(Duration::from_secs(bound_s))
        .unwrap_or_else(|| now + Duration::from_secs(u64::from(u32::MAX)));
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return CellRun::Unavailable(format!("spawn: {e}")),
    };
    let (tx, rx) = mpsc::channel();
    if let Some(out) = child.stdout.take() {
        drain(out, tx.clone(), false);
    }
    if let Some(err) = child.stderr.take() {
        drain(err, tx, true);
    }
    let Some((stdout, stderr)) = read_pipes(&rx, deadline) else {
        return kill_timed_out(&mut child);
    };
    let status = match wait_until(&mut child, deadline) {
        Ok(s) => s,
        Err(run) => return run,
    };
    let rc = status
        .code()
        .unwrap_or_else(|| status.signal().map_or(-1, |s| -s));
    CellRun::Exited { rc, stdout, stderr }
}

/// THE seam: write `cell` to `cell.py` in a fresh tmpdir and run it there
/// under the sandbox, with `cpu_s` CPU seconds (None: unlimited) and `wall_s`
/// wall seconds. The tmpdir is removed before this returns.
pub fn run_python_cell(sb: &Sandbox, cell: &str, cpu_s: Option<u64>, wall_s: u64) -> CellRun {
    let dir = match fresh_dir(&sb.tmp) {
        Ok(d) => d,
        Err(e) => return CellRun::Unavailable(format!("tmpdir: {e}")),
    };
    let path = dir.join("cell.py");
    let run = match std::fs::write(&path, cell) {
        Ok(()) => {
            let mut cmd = sandboxed(&sb.prlimit, &sb.unshare, cpu_s);
            cmd.arg(&sb.python).arg("-I").arg(&path).current_dir(&dir);
            wait_bounded(cmd, wall_s)
        }
        Err(e) => CellRun::Unavailable(format!("write cell: {e}")),
    };
    let _ = std::fs::remove_dir_all(&dir);
    run
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// The host's sandbox. A test host without one FAILS here: a sandbox
    /// test that skips would pass without measuring anything.
    pub fn host_sandbox() -> &'static Sandbox {
        sandbox().expect("this host must have python3, prlimit and a working unshare -rn")
    }

    #[test]
    fn probe_reports_version_and_refuses_a_missing_interpreter() {
        assert!(host_sandbox().version.starts_with("Python 3."));
        let sb = host_sandbox();
        let err = probe(
            sb.prlimit.clone(),
            sb.unshare.clone(),
            PathBuf::from("/nonexistent/python3"),
        )
        .unwrap_err();
        assert!(err.starts_with("probe exited "), "{err}");
    }

    #[test]
    fn spawn_error_is_unavailable() {
        let mut sb = host_sandbox().clone();
        sb.prlimit = PathBuf::from("/nonexistent/prlimit");
        assert!(matches!(
            run_python_cell(&sb, "print(1)\n", Some(1), 6),
            CellRun::Unavailable(why) if why.starts_with("spawn: ")
        ));
    }

    #[test]
    fn tmpdir_error_is_unavailable() {
        let sb = host_sandbox().with_tmp("/nonexistent/tmp");
        assert!(matches!(
            run_python_cell(&sb, "print(1)\n", Some(1), 6),
            CellRun::Unavailable(why) if why.starts_with("tmpdir: ")
        ));
    }

    #[test]
    fn relative_path_entries_are_skipped() {
        // From any cwd this relative entry reaches /usr/bin, so only the
        // absolute-entry filter keeps `sh` out.
        let up = format!("{}usr/bin", "../".repeat(64));
        assert!(Path::new(&up).join("sh").exists());
        assert_eq!(which_in(&format!("{up}:.:bin:"), "sh"), None);
        assert!(which_in("/usr/bin:/bin", "sh").is_some_and(|p| p.is_absolute()));
    }

    #[test]
    fn the_tmpdir_is_fresh_and_removed() {
        let cell = "import os\nprint(os.getcwd())\nprint(sorted(os.listdir('.')))\n";
        let CellRun::Exited { rc: 0, stdout, .. } =
            run_python_cell(host_sandbox(), cell, Some(5), 10)
        else {
            panic!("cell did not exit 0");
        };
        let out = String::from_utf8(stdout).expect("utf-8");
        let mut lines = out.lines();
        let cwd = lines.next().expect("cwd line");
        assert!(Path::new(cwd)
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("crux-code-")));
        assert_eq!(lines.next(), Some("['cell.py']"));
        assert!(!Path::new(cwd).exists(), "{cwd} was not removed");
    }
}
