//! `pv discharge gen-axioms | check | run` (PVL-001 EV-6a, #4139; `--leanchecker` EV-6b, #4199; `--comparator`
//! EV-7b, #4201; `run` EV-8a, #4202). The judging lives in
//! [`provable_contracts::discharge`]; this module prints the report and runs Lean.
//!
//! Exit: 0 accept · 1 reject (`reject:`) · 2 decline (`decline:` — no root file, zero roots, no `lake`, no
//! `leanchecker` in the toolchain, no `Challenge/*.lean` or zero comparator rows).
//!
//! `--leanchecker` is NON-fresh: it re-checks the tree's own .olean files and trusts the Mathlib .oleans they
//! import. `--fresh` replays Mathlib and is the nightly's (PVL-F7). `formalization.yaml` `scope` says so.

use std::ffi::OsStr;
use std::fmt;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use provable_contracts::discharge::comparator::{self, CHALLENGE_DIR, COMPARATOR};
use provable_contracts::discharge::summary::{self, LOG_FILE};
use provable_contracts::discharge::{self, CheckOpts, Report, Tree, AXIOMS_FILE};

use crate::cli::DischargeAction;

/// Nothing could be judged. Exit 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DischargeDeclined(pub String);

impl fmt::Display for DischargeDeclined {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DischargeDeclined {}

/// Judged, and failed. Exit 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DischargeRejected(pub String);

impl fmt::Display for DischargeRejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DischargeRejected {}

type Res = Result<(), Box<dyn std::error::Error>>;

pub fn run(action: DischargeAction) -> Res {
    match action {
        DischargeAction::GenAxioms {
            lean_dir,
            contracts,
            check,
        } => gen_axioms(&lean_dir, &contracts, check),
        DischargeAction::Check {
            lean_dir,
            contracts,
            no_lake,
            strict,
            validate_formalization,
            leanchecker,
            leanchecker_timeout,
            leanchecker_ulimit_v,
            leanchecker_threads,
            leanchecker_memory_max_gib,
            leanchecker_cpu_quota_pct,
            leanchecker_unscoped,
            comparator,
            lake_timeout,
        } => {
            let opts = CheckOpts {
                strict,
                validate_formalization,
            };
            let r = discharge::check(&lean_dir, &contracts, opts);
            let lc = leanchecker.then_some(Leanchecker::from_flags(
                leanchecker_timeout,
                leanchecker_ulimit_v,
                leanchecker_threads,
                leanchecker_memory_max_gib,
                leanchecker_cpu_quota_pct,
                leanchecker_unscoped,
            ));
            finish_with(
                Lake::new(lake_timeout),
                r,
                &lean_dir,
                no_lake,
                comparator,
                lc,
            )
        }
        DischargeAction::Run {
            lean_dir,
            contracts,
            leanchecker_timeout,
            leanchecker_ulimit_v,
            leanchecker_threads,
            leanchecker_memory_max_gib,
            leanchecker_cpu_quota_pct,
            leanchecker_unscoped,
            lake_timeout,
        } => run_all(
            Lake::new(lake_timeout),
            "build.sh",
            &lean_dir,
            &contracts,
            Leanchecker::from_flags(
                leanchecker_timeout,
                leanchecker_ulimit_v,
                leanchecker_threads,
                leanchecker_memory_max_gib,
                leanchecker_cpu_quota_pct,
                leanchecker_unscoped,
            ),
        ),
        DischargeAction::LabelRatchet {
            lean_dir,
            contracts,
        } => finish_with(
            Lake::new(LAKE_TIMEOUT_S),
            discharge::ratchet_labels(&lean_dir, &contracts),
            &lean_dir,
            true,
            false,
            None,
        ),
    }
}

fn gen_axioms(lean_dir: &Path, contracts: &Path, check: bool) -> Res {
    let g = discharge::generate(lean_dir, contracts).map_err(DischargeDeclined)?;
    let (text, b) = (g.text, g.binding);
    let path = lean_dir.join(AXIOMS_FILE);
    if check {
        return match std::fs::read_to_string(&path) {
            Ok(on_disk) if on_disk == text => {
                println!(
                    "ok    {} is its regeneration ({} root(s) bound)",
                    path.display(),
                    b.roots.len()
                );
                Ok(())
            }
            Ok(_) => Err(DischargeRejected(format!(
                "{} differs from its regeneration",
                path.display()
            ))
            .into()),
            Err(e) => Err(DischargeRejected(format!("{}: {e}", path.display())).into()),
        };
    }
    std::fs::write(&path, &text)?;
    println!("wrote {} ({} root(s) bound)", path.display(), b.roots.len());
    Ok(())
}

/// How `--leanchecker` runs (PVL-001 EV-6b, #4199; the caps #4348).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leanchecker {
    pub timeout_s: u64,
    pub ulimit_v_kib: Option<u64>,
    /// `LEAN_NUM_THREADS`: leanchecker replays one full environment (Mathlib included) per concurrent module
    /// task, so its memory is threads x environment. Uncapped it took 51 threads and 58-67 GB on lambda (#4348).
    pub threads: u32,
    /// `None` = no systemd scope (`--leanchecker-unscoped`).
    pub scope: Option<Scope>,
}

impl Leanchecker {
    /// The `--leanchecker-*` flags, shared by `check` and `run`.
    fn from_flags(
        timeout_s: u64,
        ulimit_v_kib: Option<u64>,
        threads: u32,
        memory_max_gib: u32,
        cpu_quota_pct: u32,
        unscoped: bool,
    ) -> Self {
        Self {
            timeout_s,
            ulimit_v_kib,
            threads,
            scope: (!unscoped).then_some(Scope {
                memory_max_gib,
                cpu_quota_pct,
            }),
        }
    }
}

/// `systemd-run --user --scope -p MemoryMax=<G>G -p CPUQuota=<Q>%` around leanchecker (#4348, operator: "this
/// host must be able to do other work, so never let it get overloaded").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scope {
    pub memory_max_gib: u32,
    pub cpu_quota_pct: u32,
}

/// The defaults the cop ruled for lambda (#4348): <= 8 Lean threads, 24G, 800%.
pub const LEANCHECKER_THREADS: u32 = 8;
pub const LEANCHECKER_MEMORY_MAX_GIB: u32 = 24;
pub const LEANCHECKER_CPU_QUOTA_PCT: u32 = 800;

/// The slice the scope is created under. lambda's `agent-slice-sweep` adopts any lean/lake process OUTSIDE it into
/// it -- measured 2026-09-25: a leanchecker in a 24G `run-*.scope` in app.slice was moved to agent.slice (96G) and
/// grew to 71G. A scope nested in agent.slice is left alone and its own, tighter MemoryMax still binds. On a host
/// without that slice systemd creates it as a plain transient slice.
const AGENT_SLICE: &str = "agent.slice";

/// The script `sh -c` runs; `$1` ulimit, `$2` timeout, `$3` lake.
const RECHECK_SH: &str = r#"if [ -n "$1" ]; then ulimit -v "$1" || exit 125; fi; exec timeout -k 30 "$2" "$3" env leanchecker ProvableContracts"#;

/// The argv of the leanchecker arm: `[systemd-run --user --scope -q -p MemoryMax=.. -p CPUQuota=.. --] sh -c ..`.
/// `LEAN_NUM_THREADS` is set on the command, which a `--scope` unit inherits (it runs in the caller's process).
fn recheck_argv(lake_bin: &str, lc: Leanchecker) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    if let Some(s) = lc.scope {
        v.extend(
            [
                "systemd-run".to_string(),
                "--user".into(),
                "--scope".into(),
                format!("--slice={AGENT_SLICE}"),
                "-q".into(),
                "-p".into(),
                format!("MemoryMax={}G", s.memory_max_gib),
                "-p".into(),
                format!("CPUQuota={}%", s.cpu_quota_pct),
                "--".into(),
            ]
            .into_iter(),
        );
    }
    let limit = lc.ulimit_v_kib.map(|k| k.to_string()).unwrap_or_default();
    v.extend([
        "sh".to_string(),
        "-c".into(),
        RECHECK_SH.into(),
        "pv-leanchecker".into(),
        limit,
        lc.timeout_s.to_string(),
        lake_bin.into(),
    ]);
    v
}

/// The Lean steps run only on a tree nothing else has already failed or declined: a 60-minute re-check of a tree
/// that is already RED would only delay the verdict. They record their raw exits in `r.lake_exit` and
/// `r.leanchecker_exit`, and the comparator's closure in `r.challenges` (`None` = never ran), for
/// `discharge-summary.json` (EV-8a). The comparator runs before the (much slower) leanchecker.
pub(crate) fn lean_steps(
    lake: Lake<'_>,
    r: &mut Report,
    lean_dir: &Path,
    no_lake: bool,
    cmp: bool,
    lc: Option<Leanchecker>,
) {
    let open = |r: &Report| !r.reject && r.decline.is_none();
    if open(r) && !no_lake {
        elaborate(lake, lean_dir, r);
    }
    if cmp && open(r) {
        compare(lake, lean_dir, r);
    }
    if let Some(lc) = lc {
        if open(r) {
            recheck(lake, lean_dir, lc, r);
        }
    }
}

fn finish_with(
    lake: Lake<'_>,
    mut r: Report,
    lean_dir: &Path,
    no_lake: bool,
    cmp: bool,
    lc: Option<Leanchecker>,
) -> Res {
    lean_steps(lake, &mut r, lean_dir, no_lake, cmp, lc);
    verdict(r, lean_dir)
}

/// Print the report and turn it into the exit: reject (1) before decline (2) before accept (0).
fn verdict(r: Report, lean_dir: &Path) -> Res {
    for l in &r.lines {
        println!("{l}");
    }
    if r.reject {
        let n = r.lines.iter().filter(|l| l.starts_with("FAIL")).count();
        return Err(
            DischargeRejected(format!("{n} failure(s) under {}", lean_dir.display())).into(),
        );
    }
    if let Some(why) = r.decline {
        return Err(DischargeDeclined(why).into());
    }
    println!("ok    discharge {}", lean_dir.display());
    Ok(())
}

/// `pv discharge run` (PVL-001 EV-8a, #4202): `build` (run by bash in `<lean-dir>`), then `check --strict` and
/// every Lean arm, then the two files -- written whatever the verdict, so a RED run leaves a RED summary, never
/// none. `build.sh` rc 2 is its cache-miss decline and stays a decline; any other non-zero rejects; either way the
/// Lean steps do not run on a tree that did not build. A summary that cannot be written rejects: it is the output.
pub(crate) fn run_all(
    lake: Lake<'_>,
    build: &str,
    lean_dir: &Path,
    contracts: &Path,
    lc: Leanchecker,
) -> Res {
    let (build_exit, build_out) = match Command::new("bash")
        .arg(build)
        .current_dir(lean_dir)
        .output()
    {
        Ok(o) => (
            Some(raw_exit(o.status)),
            format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            ),
        ),
        Err(e) => (None, e.to_string()),
    };
    let mut r = discharge::check(
        lean_dir,
        contracts,
        CheckOpts {
            strict: true,
            validate_formalization: false,
        },
    );
    match build_exit {
        Some(0) => {
            r.lines.insert(0, format!("ok    {build}"));
            lean_steps(lake, &mut r, lean_dir, false, true, Some(lc));
        }
        Some(2) => {
            r.lines.push(format!("{build} declined (rc 2):"));
            r.lines.extend(tail(&build_out));
            r.decline.get_or_insert_with(|| {
                format!("{build} declined: the Lean steps did not run -- not a verdict")
            });
        }
        Some(rc) => {
            r.lines.push(format!("FAIL  {build} exited {rc}"));
            r.lines.extend(tail(&build_out));
            r.reject = true;
        }
        None => {
            r.decline
                .get_or_insert_with(|| format!("{build} could not be run ({build_out})"));
        }
    }
    let tree = Tree::load(lean_dir).ok();
    let s = summary::summarize(
        &r,
        tree.as_ref(),
        lean_dir,
        summary::current_tree_sha(lean_dir),
        build_exit,
    );
    let spath = summary::summary_path(lean_dir);
    // The summary first: the log is built after it, so a summary that cannot be written is in the log's verdict.
    write_or_reject(&mut r, &spath, Ok(s.render()));
    let log = RunLog {
        verdict: if r.reject {
            "reject"
        } else if r.decline.is_some() {
            "decline"
        } else {
            "accept"
        },
        decline: r.decline.as_deref(),
        lines: &r.lines,
        summary: &s,
    };
    let log_text = serde_json::to_string_pretty(&log)
        .map(|t| t + "\n")
        .map_err(|e| e.to_string());
    write_or_reject(&mut r, &lean_dir.join(LOG_FILE), log_text);
    verdict(r, lean_dir)
}

/// Write `text` to `p`, or record the failure and reject the run.
fn write_or_reject(r: &mut Report, p: &Path, text: Result<String, String>) {
    match text.and_then(|t| std::fs::write(p, t).map_err(|e| e.to_string())) {
        Ok(()) => r.lines.push(format!("wrote {}", p.display())),
        Err(e) => {
            r.lines
                .push(format!("FAIL  cannot write {}: {e}", p.display()));
            r.reject = true;
        }
    }
}

/// `<lean-dir>/discharge.json`, the untracked full log of one `run`: the verdict, every line, and the summary.
#[derive(serde::Serialize)]
struct RunLog<'a> {
    verdict: &'a str,
    decline: Option<&'a str>,
    lines: &'a [String],
    summary: &'a summary::Summary,
}

/// `lake env lean Axioms.lean`: the subset and capstone pins, elaborated against the BUILT tree (run `build.sh`
/// first; `lake env` builds nothing).
fn elaborate(lake: Lake<'_>, lean_dir: &Path, r: &mut Report) {
    let what = format!("lake env lean {AXIOMS_FILE}");
    match lake.run(&["env", "lean", AXIOMS_FILE], lean_dir) {
        Ok(Bounded::TimedOut(pgid)) => {
            r.lake_exit = Some(TIMED_OUT);
            r.lines.push(lake.timed_out(&what, pgid));
            r.reject = true;
        }
        Err(e) => {
            r.decline = Some(format!(
                "lake could not be run ({e}): Axioms.lean was not elaborated"
            ))
        }
        Ok(Bounded::Done(o)) if o.status.success() => {
            r.lake_exit = Some(raw_exit(o.status));
            r.lines.push(format!("ok    {what}"));
        }
        Ok(Bounded::Done(o)) => {
            r.lake_exit = Some(raw_exit(o.status));
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            r.lines.push(format!(
                "FAIL  lake env lean {AXIOMS_FILE} exited {:?}",
                o.status.code()
            ));
            r.lines.extend(
                text.lines()
                    .filter(|l| l.contains("error"))
                    .take(20)
                    .map(|l| format!("  {l}")),
            );
            r.reject = true;
        }
    }
}

/// `lake env lean --run scripts/Comparator.lean Challenge/*.lean` (PVL-001 EV-7b): the script MEASURES, one NDJSON
/// row per challenge, and [`comparator::judge_rows`] judges. No Challenge file, no script or no `lake` declines;
/// a Challenge file that does not elaborate, or output that is not rows, rejects: its rows were never judged.
fn compare(lake: Lake<'_>, lean_dir: &Path, r: &mut Report) {
    let files = comparator::challenge_files(lean_dir);
    if files.is_empty() {
        r.decline = Some(format!(
            "comparator: no {CHALLENGE_DIR}/*.lean under {} -- EV-7a writes them; nothing to compare",
            lean_dir.display()
        ));
        return;
    }
    if !lean_dir.join(COMPARATOR).is_file() {
        r.decline = Some(format!(
            "comparator: {} does not exist -- not a verdict",
            lean_dir.join(COMPARATOR).display()
        ));
        return;
    }
    if !self_test(lake, lean_dir, r) {
        return;
    }
    let what = format!("lake env lean --run {COMPARATOR} ({} file(s))", files.len());
    let mut args: Vec<&OsStr> = ["env", "lean", "--run", COMPARATOR]
        .map(OsStr::new)
        .to_vec();
    args.extend(files.iter().map(|f| f.as_os_str()));
    let out = match lake.run(&args, lean_dir) {
        Ok(Bounded::Done(o)) => o,
        Ok(Bounded::TimedOut(pgid)) => {
            r.lines.push(lake.timed_out(&what, pgid));
            r.reject = true;
            return;
        }
        Err(e) => {
            r.decline = Some(format!(
                "lake could not be run ({e}): the comparator did not run"
            ));
            return;
        }
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        r.lines.push(format!(
            "FAIL  {what} exited {} -- a Challenge file did not elaborate; its rows were withheld",
            raw_exit(out.status)
        ));
        r.lines.extend(tail(&String::from_utf8_lossy(&out.stderr)));
        r.reject = true;
        return;
    }
    match comparator::parse_rows(&stdout) {
        Ok(rows) => {
            r.lines.push(format!("ok    {what}"));
            let mut c = comparator::judge_rows(&rows, r);
            match comparator::expected_roots(lean_dir, &files) {
                Ok(roots) => comparator::cross_check(&rows, &roots, &mut c, r),
                Err(e) => {
                    r.lines.push(format!(
                        "FAIL  comparator: a Challenge file could not be read, its roots were never counted: {e}"
                    ));
                    r.reject = true;
                }
            }
            r.challenges = Some(c);
        }
        Err(e) => {
            r.lines.push(format!("FAIL  {what}: {e}"));
            r.reject = true;
        }
    }
}

/// The FIPS 180-4 vectors `Comparator.lean --self-test` checks: "", "abc" and the two-block 448-bit message.
const FIPS_VECTORS: usize = 3;

/// `lake env lean --run scripts/Comparator.lean --self-test` (#4238), before any row is trusted: the pure-Lean
/// sha256 against the FIPS vectors, then the defeq controls. A regressed sha256 still hashes consistently, so the
/// rows alone cannot catch it. rc != 0, a timeout, or fewer than [`FIPS_VECTORS`] `ok … sha256` lines rejects --
/// a self-test that checked nothing is not a pass.
fn self_test(lake: Lake<'_>, lean_dir: &Path, r: &mut Report) -> bool {
    let what = format!("lake env lean --run {COMPARATOR} --self-test");
    let o = match lake.run(
        &["env", "lean", "--run", COMPARATOR, "--self-test"],
        lean_dir,
    ) {
        Ok(Bounded::Done(o)) => o,
        Ok(Bounded::TimedOut(pgid)) => {
            r.lines.push(lake.timed_out(&what, pgid));
            r.reject = true;
            return false;
        }
        Err(e) => {
            r.decline = Some(format!(
                "lake could not be run ({e}): the comparator did not run (nor its self-test)"
            ));
            return false;
        }
    };
    let stdout = String::from_utf8_lossy(&o.stdout);
    let vectors = stdout
        .lines()
        .filter(|l| l.starts_with("ok") && l.contains(" sha256 "))
        .count();
    if o.status.success() && vectors >= FIPS_VECTORS {
        r.lines
            .push(format!("ok    {what} ({vectors} FIPS vectors)"));
        return true;
    }
    r.lines.push(format!(
        "FAIL  {what} exited {} with {vectors}/{FIPS_VECTORS} FIPS vectors ok -- the statement hash is not trusted",
        raw_exit(o.status)
    ));
    r.lines.extend(tail(&format!(
        "{stdout}{}",
        String::from_utf8_lossy(&o.stderr)
    )));
    r.reject = true;
    false
}

/// The last 20 lines of `text`, indented.
fn tail(text: &str) -> Vec<String> {
    let mut v: Vec<String> = text
        .lines()
        .rev()
        .take(20)
        .map(|l| format!("  {l}"))
        .collect();
    v.reverse();
    v
}

/// The process's exit code, or 128+signal when a signal ended it (the shell's convention).
fn raw_exit(s: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    s.code().unwrap_or_else(|| 128 + s.signal().unwrap_or(0))
}

/// The default bound on one `lake` call, seconds. `check`/`run` take `--lake-timeout` (#4239).
pub(crate) const LAKE_TIMEOUT_S: u64 = 1800;

/// The exit recorded for a step the bound killed: `timeout(1)`'s, as the leanchecker arm records.
const TIMED_OUT: i32 = 124;

/// A `lake` binary and the wall-clock bound on each call to it (#4239: a hang is RED, not a wait).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lake<'a> {
    pub bin: &'a str,
    pub timeout_s: u64,
}

/// How a bounded call ended. `TimedOut` carries the killed process group (the child's own PID).
enum Bounded {
    Done(std::process::Output),
    TimedOut(u32),
}

impl Lake<'static> {
    fn new(timeout_s: u64) -> Self {
        Lake {
            bin: "lake",
            timeout_s,
        }
    }
}

impl Lake<'_> {
    /// `lake <args>` in `dir`, in its own process group, waited on for at most `timeout_s`. At the deadline the
    /// group is SIGKILLed by the child's own PID (pgid == pid) -- never by pattern -- so a `lean` under `lake env`
    /// dies with it. The deadline also bounds the output drain: a grandchild holding a pipe cannot stall it.
    fn run<S: AsRef<OsStr>>(self, args: &[S], dir: &Path) -> std::io::Result<Bounded> {
        let mut child = self.spawn(args, dir)?;
        let pgid = child.id();
        let rx = drain_pipes(&mut child);
        let deadline = Instant::now() + Duration::from_secs(self.timeout_s);
        let Some(status) = wait_until(&mut child, deadline)? else {
            return Ok(kill_group(&mut child, pgid));
        };
        let Some([stdout, stderr]) = collect(&rx, deadline) else {
            return Ok(kill_group(&mut child, pgid));
        };
        Ok(Bounded::Done(std::process::Output {
            status,
            stdout,
            stderr,
        }))
    }

    /// `lake <args>` spawned in its own process group, piped. ETXTBSY (26): a just-written `lake` whose write fd a
    /// concurrent fork still holds until its exec. Transient, so it is retried.
    fn spawn<S: AsRef<OsStr>>(
        self,
        args: &[S],
        dir: &Path,
    ) -> std::io::Result<std::process::Child> {
        use std::os::unix::process::CommandExt;
        use std::process::Stdio;
        let spawn = || {
            Command::new(self.bin)
                .args(args)
                .current_dir(dir)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .process_group(0)
                .spawn()
        };
        let mut child = spawn();
        for _ in 0..20 {
            match &child {
                Err(e) if e.raw_os_error() == Some(26) => {
                    std::thread::sleep(Duration::from_millis(25));
                    child = spawn();
                }
                _ => break,
            }
        }
        child
    }

    fn timed_out(self, what: &str, pgid: u32) -> String {
        format!(
            "FAIL  {what} timed out after {}s -- killed its process group {pgid}; a hang is RED, not a wait",
            self.timeout_s
        )
    }
}

type Drained = std::sync::mpsc::Receiver<(usize, Vec<u8>)>;

/// Read the child's stdout (0) and stderr (1) on their own threads, so a full pipe cannot block its exit.
fn drain_pipes(child: &mut std::process::Child) -> Drained {
    use std::io::Read;
    let (tx, rx) = std::sync::mpsc::channel();
    let drain = |mut pipe: Box<dyn Read + Send>, which: usize| {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            let _ = tx.send((which, buf));
        });
    };
    if let Some(p) = child.stdout.take() {
        drain(Box::new(p), 0);
    }
    if let Some(p) = child.stderr.take() {
        drain(Box::new(p), 1);
    }
    rx
}

/// The child's exit status, or `None` once `deadline` passes with it still running.
fn wait_until(
    child: &mut std::process::Child,
    deadline: Instant,
) -> std::io::Result<Option<std::process::ExitStatus>> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Both drained pipes, or `None` if `deadline` passes first: a grandchild holding a pipe cannot stall the drain.
fn collect(rx: &Drained, deadline: Instant) -> Option<[Vec<u8>; 2]> {
    use std::sync::mpsc::RecvTimeoutError;
    let mut out = [Vec::new(), Vec::new()];
    for _ in 0..2 {
        let left = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok((which, buf)) => out[which] = buf,
            Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => return None,
        }
    }
    Some(out)
}

/// SIGKILL the whole process group by the child's own PID (pgid == pid) -- never by pattern -- then reap the child.
fn kill_group(child: &mut std::process::Child, pgid: u32) -> Bounded {
    let _ = Command::new("kill")
        .args(["-s", "KILL", "--", &format!("-{pgid}")])
        .status();
    let _ = child.kill();
    let _ = child.wait();
    Bounded::TimedOut(pgid)
}

/// The toolchain `lake env` puts on PATH for this tree: `lake env printenv LEAN_SYSROOT`. `Err` is the decline;
/// a timeout rejects in place and yields `Err(None)` (#4239).
fn sysroot(
    lake: Lake<'_>,
    lean_dir: &Path,
    r: &mut Report,
) -> Result<std::path::PathBuf, Option<String>> {
    let o = match lake.run(&["env", "printenv", "LEAN_SYSROOT"], lean_dir) {
        Ok(Bounded::Done(o)) => o,
        Ok(Bounded::TimedOut(pgid)) => {
            r.lines
                .push(lake.timed_out("lake env printenv LEAN_SYSROOT", pgid));
            r.reject = true;
            return Err(None);
        }
        Err(e) => {
            return Err(Some(format!(
                "lake could not be run ({e}): leanchecker did not run"
            )))
        }
    };
    let root = String::from_utf8_lossy(&o.stdout).trim().to_string();
    if !o.status.success() || root.is_empty() {
        return Err(Some(format!(
            "`lake env printenv LEAN_SYSROOT` exited {:?} and named no toolchain: leanchecker did not run",
            o.status.code()
        )));
    }
    Ok(root.into())
}

/// `timeout <T> lake env leanchecker ProvableContracts` (PVL-001 EV-6b): rc != 0 rejects, a timeout rejects, and a
/// toolchain without `leanchecker` declines -- measured: elan's `lake env leanchecker` on v4.15.0 exits 1 "does not
/// have the binary", which would otherwise read as a failed check. `timeout` signals the whole process group, so
/// the `leanchecker` under `lake` does not outlive it.
fn recheck(lake: Lake<'_>, lean_dir: &Path, lc: Leanchecker, r: &mut Report) {
    let root = match sysroot(lake, lean_dir, r) {
        Ok(root) => root,
        Err(why) => {
            r.decline = why;
            return;
        }
    };
    let bin = root.join("bin").join("leanchecker");
    if !bin.is_file() {
        r.decline = Some(format!(
            "leanchecker not in toolchain: {} does not exist",
            bin.display()
        ));
        return;
    }
    // A scope that cannot be created would otherwise read as leanchecker's own non-zero exit: probe it first.
    if lc.scope.is_some() {
        let probe = Command::new("systemd-run")
            .args([
                "--user",
                "--scope",
                &format!("--slice={AGENT_SLICE}"),
                "-q",
                "--",
                "true",
            ])
            .output();
        if !matches!(&probe, Ok(o) if o.status.success()) {
            r.decline = Some(
                "`systemd-run --user --scope` is unavailable here: leanchecker did not run (pass \
                 --leanchecker-unscoped only on a host where an uncapped run cannot starve other work, #4348)"
                    .to_string(),
            );
            return;
        }
    }
    let argv = recheck_argv(lake.bin, lc);
    let out = Command::new(&argv[0])
        .args(&argv[1..])
        .env("LEAN_NUM_THREADS", lc.threads.to_string())
        .current_dir(lean_dir)
        .output();
    let what = format!(
        "lake env leanchecker ProvableContracts (timeout {}s, {} threads, {})",
        lc.timeout_s,
        lc.threads,
        lc.scope.map_or("unscoped".to_string(), |s| format!(
            "scope MemoryMax={}G CPUQuota={}%",
            s.memory_max_gib, s.cpu_quota_pct
        ))
    );
    match out {
        Err(e) => {
            r.decline = Some(format!(
                "sh could not be run ({e}): leanchecker did not run"
            ))
        }
        Ok(o) if o.status.success() => {
            r.leanchecker_exit = Some(0);
            r.lines.push(format!("ok    {what}"));
        }
        Ok(o) => match o.status.code() {
            // timeout/ulimit/lake could not start leanchecker: it never ran, so no exit is recorded.
            Some(c @ 125..=127) => {
                r.decline = Some(format!(
                    "{what} could not be started (rc {c}: timeout/ulimit/lake): not a verdict"
                ));
            }
            code => {
                r.leanchecker_exit = Some(raw_exit(o.status));
                let why = if matches!(code, Some(124 | 137)) {
                    "timed out".to_string()
                } else {
                    format!("exited {code:?}")
                };
                r.lines.push(format!("FAIL  {what} {why}"));
                let text = format!(
                    "{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                );
                r.lines.extend(
                    text.lines()
                        .rev()
                        .take(20)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .map(|l| format!("  {l}")),
                );
                r.reject = true;
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract_walk::{exit_code_for, verdict_for};
    use std::path::PathBuf;

    /// A one-theorem tree bound by the label `Theorems.Gelu` (the integration fixture, in-process).
    fn tree() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let d = tempfile::tempdir().expect("tempdir");
        let lean = d.path().join("lean");
        let contracts = d.path().join("contracts");
        std::fs::create_dir_all(lean.join("ProvableContracts/Theorems/Gelu")).expect("mkdir");
        std::fs::create_dir_all(&contracts).expect("mkdir");
        std::fs::write(
            lean.join("ProvableContracts.lean"),
            "import ProvableContracts.Theorems.Gelu.Bound\n",
        )
        .expect("w");
        std::fs::write(
            lean.join("ProvableContracts/Theorems/Gelu/Bound.lean"),
            "namespace ProvableContracts.Gelu\ntheorem gelu_bound : True := trivial\nend ProvableContracts.Gelu\n",
        )
        .expect("w");
        std::fs::write(
            contracts.join("gelu-v1.yaml"),
            "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n",
        )
        .expect("w");
        (d, lean, contracts)
    }

    fn gen(lean: &Path, contracts: &Path, check: bool) -> Res {
        run(DischargeAction::GenAxioms {
            lean_dir: lean.into(),
            contracts: contracts.into(),
            check,
        })
    }

    fn check(lean: &Path, contracts: &Path, strict: bool) -> Res {
        run(DischargeAction::Check {
            lean_dir: lean.into(),
            contracts: contracts.into(),
            no_lake: true,
            strict,
            validate_formalization: false,
            leanchecker: false,
            leanchecker_timeout: 3600,
            leanchecker_ulimit_v: None,
            leanchecker_threads: LEANCHECKER_THREADS,
            leanchecker_memory_max_gib: LEANCHECKER_MEMORY_MAX_GIB,
            leanchecker_cpu_quota_pct: LEANCHECKER_CPU_QUOTA_PCT,
            leanchecker_unscoped: false,
            comparator: false,
            lake_timeout: LAKE_TIMEOUT_S,
        })
    }

    fn is_reject(r: &Res) -> bool {
        r.as_ref()
            .err()
            .is_some_and(|e| e.downcast_ref::<DischargeRejected>().is_some())
    }

    fn is_decline(r: &Res) -> bool {
        r.as_ref()
            .err()
            .is_some_and(|e| e.downcast_ref::<DischargeDeclined>().is_some())
    }

    #[test]
    fn the_verdict_words_and_exit_codes_are_pvl_1s() {
        let d: Box<dyn std::error::Error> = Box::new(DischargeDeclined("x".into()));
        let r: Box<dyn std::error::Error> = Box::new(DischargeRejected("y".into()));
        assert_eq!(
            (exit_code_for(d.as_ref()), verdict_for(d.as_ref())),
            (2, "decline")
        );
        assert_eq!(
            (exit_code_for(r.as_ref()), verdict_for(r.as_ref())),
            (1, "reject")
        );
        assert_eq!(
            (d.to_string(), r.to_string()),
            ("x".to_string(), "y".to_string())
        );
    }

    #[test]
    fn gen_axioms_writes_and_check_mode_judges_freshness() {
        let (_d, lean, contracts) = tree();
        assert!(
            is_reject(&gen(&lean, &contracts, true)),
            "no Axioms.lean yet must reject"
        );
        gen(&lean, &contracts, false).expect("write");
        let text = std::fs::read_to_string(lean.join(AXIOMS_FILE)).expect("written");
        assert!(
            text.contains("`ProvableContracts.Gelu.gelu_bound"),
            "{text}"
        );
        gen(&lean, &contracts, true).expect("fresh");
        std::fs::write(lean.join(AXIOMS_FILE), format!("{text}-- edit\n")).expect("w");
        assert!(is_reject(&gen(&lean, &contracts, true)));
        std::fs::remove_file(lean.join("ProvableContracts.lean")).expect("rm");
        assert!(is_decline(&gen(&lean, &contracts, false)));
    }

    #[test]
    fn check_accepts_rejects_and_declines() {
        let (_d, lean, contracts) = tree();
        gen(&lean, &contracts, false).expect("write");
        check(&lean, &contracts, false).expect("clean tree");
        let f = lean.join("ProvableContracts/Theorems/Gelu/Bound.lean");
        let src = std::fs::read_to_string(&f).expect("r");
        std::fs::write(
            &f,
            src.replace(
                "end ProvableContracts.Gelu",
                "axiom m : False\nend ProvableContracts.Gelu",
            ),
        )
        .expect("w");
        assert!(is_reject(&check(&lean, &contracts, false)));
        std::fs::write(
            lean.join("escape-allowlist.yaml"),
            "- file: ProvableContracts/Theorems/Gelu/Bound.lean\n  decl: ProvableContracts.Gelu.m\n  kind: axiom\n  reason: r\n  ticket: t\n  confirmed_by: pending\n",
        )
        .expect("w");
        gen(&lean, &contracts, false).expect("regen");
        check(&lean, &contracts, false).expect("pending is accepted");
        assert!(
            is_reject(&check(&lean, &contracts, true)),
            "--strict rejects pending"
        );
        std::fs::write(
            contracts.join("gelu-v1.yaml"),
            "equations:\n  e:\n    lean_theorem: none\n",
        )
        .expect("w");
        gen(&lean, &contracts, false).expect("regen");
        assert!(
            is_decline(&check(&lean, &contracts, false)),
            "zero roots declines"
        );
    }

    #[test]
    fn label_ratchet_writes_the_set() {
        let (_d, lean, contracts) = tree();
        run(DischargeAction::LabelRatchet {
            lean_dir: lean.clone(),
            contracts,
        })
        .expect("seed");
        assert!(lean.join(discharge::LABELS).is_file());
    }

    /// A `lake` under a 60 s bound: long enough for any fake, short enough that a hang fails the test.
    fn lk(bin: &str) -> Lake<'_> {
        Lake { bin, timeout_s: 60 }
    }

    /// A fake `lake`: exits `rc` after printing `out`.
    fn fake_lake(dir: &Path, rc: i32, out: &str) -> String {
        let p = dir.join(format!("lake-{rc}"));
        std::fs::write(&p, format!("#!/bin/sh\necho '{out}'\nexit {rc}\n")).expect("w");
        let mut perm = std::fs::metadata(&p).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&p, perm).expect("chmod");
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn the_lean_elaboration_decides_the_verdict() {
        let (d, lean, _) = tree();
        let ok = fake_lake(d.path(), 0, "fine");
        let bad = fake_lake(d.path(), 1, "Axioms.lean:3:0: error: AXIOMS x");
        finish_with(lk(&ok), Report::default(), &lean, false, false, None).expect("lake ok");
        assert!(
            is_reject(&finish_with(
                lk(&bad),
                Report::default(),
                &lean,
                false,
                false,
                None
            )),
            "a failing elaboration rejects"
        );
        assert!(
            is_decline(&finish_with(
                lk("/nonexistent/lake"),
                Report::default(),
                &lean,
                false,
                false,
                None
            )),
            "no lake declines"
        );
        finish_with(lk(&bad), Report::default(), &lean, true, false, None)
            .expect("--no-lake skips it");
        let rejected = Report {
            reject: true,
            ..Report::default()
        };
        assert!(
            is_reject(&finish_with(lk(&ok), rejected, &lean, false, false, None)),
            "a prior failure is not cleared by lake"
        );
    }

    /// A `lake` for `--leanchecker`: `lake env printenv LEAN_SYSROOT` names `<dir>/sysroot` (with `bin/leanchecker`
    /// when `has_checker`), `lake env lean …` passes, and `lake env leanchecker …` prints `out` and exits `rc`.
    fn checker_lake(dir: &Path, has_checker: bool, rc: i32, out: &str) -> String {
        let root = dir.join(format!("sysroot-{has_checker}"));
        std::fs::create_dir_all(root.join("bin")).expect("mkdir");
        if has_checker {
            std::fs::write(root.join("bin").join("leanchecker"), "").expect("w");
        }
        let p = dir.join(format!("checker-lake-{has_checker}-{rc}"));
        let script = format!(
            "#!/bin/sh\ncase \"$2\" in\n  printenv) echo '{}' ;;\n  lean) exit 0 ;;\n  leanchecker) echo '{out}'; exit {rc} ;;\n  *) exit 99 ;;\nesac\n",
            root.display()
        );
        std::fs::write(&p, script).expect("w");
        let mut perm = std::fs::metadata(&p).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&p, perm).expect("chmod");
        p.to_string_lossy().into_owned()
    }

    const LC: Option<Leanchecker> = Some(Leanchecker {
        timeout_s: 60,
        ulimit_v_kib: None,
        threads: 8,
        scope: None,
    });

    /// PVL-001 EV-6b: leanchecker's rc decides; an ABSENT leanchecker declines and is never read as a failure.
    #[test]
    fn leanchecker_rc_rejects_and_an_absent_checker_declines() {
        let (d, lean, _) = tree();
        let pass = checker_lake(d.path(), true, 0, "ok");
        let fail = checker_lake(
            d.path(),
            true,
            1,
            "error: kernel rejected Theorems.Gelu.bound",
        );
        let absent = checker_lake(d.path(), false, 0, "never reached");
        finish_with(lk(&pass), Report::default(), &lean, false, false, LC)
            .expect("leanchecker rc 0 accepts");
        assert!(
            is_reject(&finish_with(
                lk(&fail),
                Report::default(),
                &lean,
                false,
                false,
                LC
            )),
            "leanchecker rc 1 rejects"
        );
        assert!(
            is_decline(&finish_with(
                lk(&absent),
                Report::default(),
                &lean,
                false,
                false,
                LC
            )),
            "no leanchecker in the toolchain declines"
        );
        assert!(
            is_decline(&finish_with(
                lk("/nonexistent/lake"),
                Report::default(),
                &lean,
                true,
                false,
                LC
            )),
            "no lake declines under --leanchecker"
        );
        finish_with(lk(&fail), Report::default(), &lean, false, false, None)
            .expect("without --leanchecker it never runs");
    }

    #[test]
    fn a_timed_out_leanchecker_rejects_and_the_ulimit_reaches_the_checker() {
        let (d, lean, _) = tree();
        let slow = checker_lake(d.path(), true, 0, "x");
        let body = std::fs::read_to_string(&slow)
            .expect("r")
            .replace("echo 'x'; exit 0", "sleep 30");
        std::fs::write(&slow, body).expect("w");
        let t1 = Some(Leanchecker {
            timeout_s: 1,
            ulimit_v_kib: None,
            threads: 8,
            scope: None,
        });
        assert!(
            is_reject(&finish_with(
                lk(&slow),
                Report::default(),
                &lean,
                true,
                false,
                t1
            )),
            "a timeout rejects"
        );
        // The limit reaches the checker: a stub that prints its own `ulimit -v` and fails shows it in the reject.
        let shows = checker_lake(d.path(), true, 1, "x");
        let body = std::fs::read_to_string(&shows)
            .expect("r")
            .replace("echo 'x'", "echo \"vlimit=$(ulimit -v)\"");
        std::fs::write(&shows, body).expect("w");
        let mut r = Report::default();
        recheck(
            lk(&shows),
            &lean,
            Leanchecker {
                timeout_s: 60,
                ulimit_v_kib: Some(4_194_304),
                threads: 8,
                scope: None,
            },
            &mut r,
        );
        assert!(r.reject, "{:?}", r.lines);
        assert!(
            r.lines.iter().any(|l| l.contains("vlimit=4194304")),
            "{:?}",
            r.lines
        );
    }

    /// #4348: the thread cap reaches the checker -- a stub that prints its own `LEAN_NUM_THREADS` and fails shows it.
    #[test]
    fn the_thread_cap_reaches_the_checker() {
        let (d, lean, _) = tree();
        let shows = checker_lake(d.path(), true, 1, "x");
        let body = std::fs::read_to_string(&shows)
            .expect("r")
            .replace("echo 'x'", "echo \"threads=$LEAN_NUM_THREADS\"");
        std::fs::write(&shows, body).expect("w");
        let mut r = Report::default();
        let lc = Leanchecker {
            threads: 3,
            ..LC.expect("LC")
        };
        recheck(lk(&shows), &lean, lc, &mut r);
        assert!(r.reject, "{:?}", r.lines);
        assert!(
            r.lines.iter().any(|l| l.contains("threads=3")),
            "{:?}",
            r.lines
        );
    }

    /// #4348: scoped, the arm is `systemd-run --user --scope` with the memory and CPU caps around the same `sh -c`;
    /// unscoped it is the bare `sh -c`.
    #[test]
    fn the_scope_wraps_the_checker_with_both_caps() {
        let scoped = recheck_argv(
            "lake",
            Leanchecker {
                scope: Some(Scope {
                    memory_max_gib: 24,
                    cpu_quota_pct: 800,
                }),
                ..LC.expect("LC")
            },
        );
        assert_eq!(
            scoped[..10],
            [
                "systemd-run",
                "--user",
                "--scope",
                "--slice=agent.slice",
                "-q",
                "-p",
                "MemoryMax=24G",
                "-p",
                "CPUQuota=800%",
                "--"
            ]
        );
        let bare = recheck_argv("lake", LC.expect("LC"));
        assert_eq!(bare[..2], ["sh", "-c"]);
        assert_eq!(scoped[10..], bare[..]);
    }

    /// #4348: the DEFAULT `--leanchecker` is capped -- 8 threads inside a 24G/800% scope -- on `check` and `run`
    /// alike; only an explicit `--leanchecker-unscoped` drops the scope.
    #[test]
    fn the_default_leanchecker_is_capped() {
        #[derive(clap::Parser)]
        struct T {
            #[command(subcommand)]
            a: crate::cli::DischargeAction,
        }
        let parse = |args: &[&str]| {
            <T as clap::Parser>::try_parse_from(std::iter::once("t").chain(args.iter().copied()))
                .expect("parses")
                .a
        };
        for a in [
            parse(&["check", "L", "--leanchecker"]),
            parse(&["run", "L"]),
        ] {
            let (t, g, q, u) = match a {
                DischargeAction::Check {
                    leanchecker_threads: t,
                    leanchecker_memory_max_gib: g,
                    leanchecker_cpu_quota_pct: q,
                    leanchecker_unscoped: u,
                    ..
                }
                | DischargeAction::Run {
                    leanchecker_threads: t,
                    leanchecker_memory_max_gib: g,
                    leanchecker_cpu_quota_pct: q,
                    leanchecker_unscoped: u,
                    ..
                } => (t, g, q, u),
                _ => unreachable!(),
            };
            let lc = Leanchecker::from_flags(3600, None, t, g, q, u);
            assert_eq!(
                (lc.threads, lc.scope),
                (
                    8,
                    Some(Scope {
                        memory_max_gib: 24,
                        cpu_quota_pct: 800
                    })
                )
            );
        }
        match parse(&["check", "L", "--leanchecker", "--leanchecker-unscoped"]) {
            DischargeAction::Check {
                leanchecker_unscoped,
                ..
            } => assert!(leanchecker_unscoped),
            _ => unreachable!(),
        }
    }

    /// The defaults are also the ceilings: a caller may lower a cap, never raise it (51 threads took 58-67 GB).
    #[test]
    fn a_leanchecker_cap_can_be_lowered_never_raised() {
        #[derive(clap::Parser)]
        struct T {
            #[command(subcommand)]
            a: crate::cli::DischargeAction,
        }
        let parses = |args: &[&str]| {
            <T as clap::Parser>::try_parse_from(std::iter::once("t").chain(args.iter().copied()))
                .is_ok()
        };
        for (flag, max) in [
            ("--leanchecker-threads", LEANCHECKER_THREADS),
            ("--leanchecker-memory-max-gib", LEANCHECKER_MEMORY_MAX_GIB),
            ("--leanchecker-cpu-quota-pct", LEANCHECKER_CPU_QUOTA_PCT),
        ] {
            for base in [&["check", "L", "--leanchecker"][..], &["run", "L"][..]] {
                let with = |v: u32| {
                    let v = v.to_string();
                    let mut a = base.to_vec();
                    a.extend([flag, v.as_str()]);
                    parses(&a)
                };
                assert!(
                    with(1) && with(max),
                    "{flag} {base:?}: 1 and {max} must parse"
                );
                assert!(
                    !with(max + 1),
                    "{flag} {base:?}: {} must be refused",
                    max + 1
                );
                assert!(!with(0), "{flag} {base:?}: 0 must be refused");
            }
        }
    }

    /// EV-8a reads the raw exits: `Some(n)` for a step that ran (124 on a timeout), `None` for one that never ran.
    #[test]
    fn the_lean_steps_record_their_raw_exits_and_none_when_they_did_not_run() {
        let (d, lean, _) = tree();
        let exits = |lake: &str, no_lake: bool, lc: Option<Leanchecker>| {
            let mut r = Report::default();
            lean_steps(lk(lake), &mut r, &lean, no_lake, false, lc);
            (r.lake_exit, r.leanchecker_exit)
        };
        let pass = checker_lake(d.path(), true, 0, "ok");
        assert_eq!(exits(&pass, false, LC), (Some(0), Some(0)));
        assert_eq!(
            exits(&pass, false, None),
            (Some(0), None),
            "no --leanchecker"
        );
        assert_eq!(exits(&pass, true, None), (None, None), "--no-lake");
        let fail = checker_lake(d.path(), true, 3, "kernel error");
        assert_eq!(exits(&fail, true, LC), (None, Some(3)));
        let absent = checker_lake(d.path(), false, 0, "never reached");
        assert_eq!(
            exits(&absent, true, LC),
            (None, None),
            "absent checker never ran"
        );
        let slow = checker_lake(d.path(), true, 0, "x");
        let body = std::fs::read_to_string(&slow)
            .expect("r")
            .replace("echo 'x'; exit 0", "sleep 30");
        std::fs::write(&slow, body).expect("w");
        let t1 = Some(Leanchecker {
            timeout_s: 1,
            ulimit_v_kib: None,
            threads: 8,
            scope: None,
        });
        assert_eq!(exits(&slow, true, t1), (None, Some(124)), "timeout");
    }

    /// A passing `Comparator.lean --self-test`: the three FIPS vector lines, rc 0 (#4238).
    const SELF_TEST_OK: &str = "if [ \"$5\" = --self-test ]; then printf 'ok    sha256 \"\" = e3\\nok    sha256 \"abc\" = ba\\nok    sha256 \"abcdbcde\" = 24\\n'; exit 0; fi";

    /// The root EV-7a's `render` declares for the fixture's one solution. The cross-check (#4240) counts
    /// these lines, so an empty Challenge file would make every fixture row an UNEXPECTED-ROW.
    const GELU_CHALLENGE: &str =
        "theorem _root_.PvlChallenge.ProvableContracts.Gelu.gelu_bound : True := by\n  sorry\n";

    /// A `lake` for `--comparator`: `env lean --run …` writes `rows` to stdout, `stderr` to stderr, and exits
    /// `rc`; every other `env lean` passes. The tree gets `Challenge/gelu-v1.lean` and the comparator script.
    fn comparator_lake(dir: &Path, lean: &Path, rc: i32, rows: &str, stderr: &str) -> String {
        std::fs::create_dir_all(lean.join(CHALLENGE_DIR)).expect("mkdir");
        // The Challenge file declares exactly the roots the canned rows name (#4240 cross-checks the two);
        // output that is not rows declares nothing, so those cases still judge the output alone.
        let decls: String = comparator::parse_rows(rows)
            .map(|rs| {
                rs.iter()
                    .map(|r| {
                        format!(
                            "theorem _root_.PvlChallenge.{} : True := by\n  sorry\n",
                            r.name
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        std::fs::write(lean.join(CHALLENGE_DIR).join("gelu-v1.lean"), decls).expect("w");
        std::fs::create_dir_all(lean.join("scripts")).expect("mkdir");
        std::fs::write(lean.join(COMPARATOR), "").expect("w");
        std::fs::write(dir.join(format!("rows-{rc}")), rows).expect("w");
        let p = dir.join(format!("cmp-lake-{rc}"));
        let script = format!(
            "#!/bin/sh\n{SELF_TEST_OK}\nif [ \"$3\" = --run ]; then\n  [ \"$4 $5\" = \"{COMPARATOR} {CHALLENGE_DIR}/gelu-v1.lean\" ] || {{ echo \"bad args: $*\" >&2; exit 98; }}\n  cat '{}'; echo '{stderr}' >&2; exit {rc}\nfi\nexit 0\n",
            dir.join(format!("rows-{rc}")).display()
        );
        std::fs::write(&p, script).expect("w");
        let mut perm = std::fs::metadata(&p).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&p, perm).expect("chmod");
        p.to_string_lossy().into_owned()
    }

    fn cmp_row(name: &str, ch: &str, sol: &str, axioms: &str) -> String {
        format!(
            "{{\"name\": \"{name}\", \"challenge_type_hash\": \"{ch}\", \"solution_type_hash\": {sol}, \"axioms\": {axioms}}}\n"
        )
    }

    fn compared(lake: &str, lean: &Path) -> Report {
        let mut r = Report::default();
        lean_steps(lk(lake), &mut r, lean, false, true, None);
        r
    }

    /// PVL-001 EV-7b: a matching, sorry-free solution closes its challenge; the closure is recorded for EV-8a.
    #[test]
    fn the_comparator_closes_a_matching_challenge() {
        let (d, lean, _) = tree();
        let h = "ab".repeat(32);
        let lake = comparator_lake(
            d.path(),
            &lean,
            0,
            &cmp_row(
                "ProvableContracts.Gelu.gelu_bound",
                &h,
                &format!("\"{h}\""),
                "[]",
            ),
            "",
        );
        let r = compared(&lake, &lean);
        assert!(!r.reject && r.decline.is_none(), "{:?}", r.lines);
        assert_eq!(
            r.challenges,
            Some(comparator::Closure {
                closed: 1,
                total: 1
            })
        );
        assert!(
            r.lines
                .iter()
                .any(|l| l == "COMPARATOR 1/1 challenge(s) closed"),
            "{:?}",
            r.lines
        );
        finish_with(lk(&lake), Report::default(), &lean, false, true, None).expect("accepts");
    }

    /// The spec's RED: a solution of a WEAKER statement is a mismatch, and a sorry'd one closes nothing.
    #[test]
    fn a_mismatched_or_sorry_solution_rejects() {
        let (d, lean, _) = tree();
        let (c, s) = ("ab".repeat(32), "cd".repeat(32));
        let rows = format!(
            "{}{}",
            cmp_row("A.weak", &c, &format!("\"{s}\""), "[]"),
            cmp_row("A.sorried", &c, &format!("\"{c}\""), "[\"sorryAx\"]")
        );
        let lake = comparator_lake(d.path(), &lean, 0, &rows, "");
        let r = compared(&lake, &lean);
        assert!(r.reject, "{:?}", r.lines);
        assert!(r
            .lines
            .iter()
            .any(|l| l.starts_with("FAIL  MISMATCH A.weak")));
        assert!(r
            .lines
            .iter()
            .any(|l| l.starts_with("FAIL  SORRY A.sorried")));
        assert_eq!(r.challenges.map(|c| (c.closed, c.total)), Some((0, 2)));
    }

    #[test]
    fn a_challenge_that_does_not_elaborate_rejects_with_its_errors() {
        let (d, lean, _) = tree();
        let lake = comparator_lake(
            d.path(),
            &lean,
            1,
            "",
            "Challenge/gelu-v1.lean:5:0: error: unknown g",
        );
        let r = compared(&lake, &lean);
        assert!(r.reject, "{:?}", r.lines);
        assert!(
            r.lines.iter().any(|l| l.contains("exited 1")),
            "{:?}",
            r.lines
        );
        assert!(
            r.lines.iter().any(|l| l.contains("unknown g")),
            "{:?}",
            r.lines
        );
        assert_eq!(r.challenges, None, "rows withheld: nothing was judged");
    }

    #[test]
    fn unreadable_comparator_output_rejects() {
        let (d, lean, _) = tree();
        let lake = comparator_lake(d.path(), &lean, 0, "not a row\n", "");
        let r = compared(&lake, &lean);
        assert!(r.reject, "{:?}", r.lines);
        assert!(
            r.lines.iter().any(|l| l.contains("line 1")),
            "{:?}",
            r.lines
        );
    }

    /// Zero measured is a decline, never a pass: no Challenge file, no script, zero rows, no lake.
    #[test]
    fn the_comparator_declines_when_nothing_was_compared() {
        let (d, lean, _) = tree();
        let lake = comparator_lake(d.path(), &lean, 0, "", "");
        let r = compared(&lake, &lean);
        assert!(!r.reject, "{:?}", r.lines);
        assert!(
            r.decline
                .as_deref()
                .is_some_and(|w| w.contains("0 challenge rows")),
            "{r:?}"
        );
        assert_eq!(r.challenges.map(|c| c.total), Some(0));

        std::fs::remove_file(lean.join(COMPARATOR)).expect("rm");
        let r = compared(&lake, &lean);
        assert!(
            r.decline
                .as_deref()
                .is_some_and(|w| w.contains("does not exist")),
            "{r:?}"
        );

        std::fs::remove_dir_all(lean.join(CHALLENGE_DIR)).expect("rm");
        let r = compared(&lake, &lean);
        assert!(
            r.decline
                .as_deref()
                .is_some_and(|w| w.contains("no Challenge/*.lean")),
            "{r:?}"
        );
        assert!(is_decline(&finish_with(
            lk(&lake),
            Report::default(),
            &lean,
            false,
            true,
            None
        )));

        let (d2, lean2, _) = tree();
        comparator_lake(d2.path(), &lean2, 0, "", "");
        let mut r = Report::default();
        lean_steps(lk("/nonexistent/lake"), &mut r, &lean2, true, true, None);
        assert!(
            r.decline
                .as_deref()
                .is_some_and(|w| w.contains("comparator did not run")),
            "{r:?}"
        );
    }

    #[test]
    fn without_the_flag_the_comparator_never_runs() {
        let (d, lean, _) = tree();
        let lake = comparator_lake(d.path(), &lean, 1, "", "would reject");
        let mut r = Report::default();
        lean_steps(lk(&lake), &mut r, &lean, false, false, None);
        assert!(!r.reject && r.challenges.is_none(), "{:?}", r.lines);
    }

    /// PVL-001 EV-8a: a `lake` for `run` — every `env lean` passes, `--run` prints one closing comparator row,
    /// `printenv` names a sysroot with a leanchecker, and `env leanchecker` exits `lc_rc`.
    fn run_lake(dir: &Path, lean: &Path, lc_rc: i32) -> String {
        std::fs::create_dir_all(lean.join(CHALLENGE_DIR)).expect("mkdir");
        std::fs::write(
            lean.join(CHALLENGE_DIR).join("gelu-v1.lean"),
            GELU_CHALLENGE,
        )
        .expect("w");
        std::fs::create_dir_all(lean.join("scripts")).expect("mkdir");
        std::fs::write(lean.join(COMPARATOR), "").expect("w");
        let root = dir.join("run-sysroot");
        std::fs::create_dir_all(root.join("bin")).expect("mkdir");
        std::fs::write(root.join("bin").join("leanchecker"), "").expect("w");
        let h = "ab".repeat(32);
        let rows = dir.join("run-rows");
        std::fs::write(
            &rows,
            cmp_row(
                "ProvableContracts.Gelu.gelu_bound",
                &h,
                &format!("\"{h}\""),
                "[]",
            ),
        )
        .expect("w");
        let p = dir.join(format!("run-lake-{lc_rc}"));
        let script = format!(
            "#!/bin/sh\n{SELF_TEST_OK}\ncase \"$2 $3\" in\n  'lean --run') cat '{}' ;;\n  printenv*) echo '{}' ;;\n  leanchecker*) exit {lc_rc} ;;\nesac\nexit 0\n",
            rows.display(),
            root.display()
        );
        std::fs::write(&p, script).expect("w");
        let mut perm = std::fs::metadata(&p).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&p, perm).expect("chmod");
        p.to_string_lossy().into_owned()
    }

    fn build_sh(dir: &Path, rc: i32) -> String {
        let p = dir.join(format!("build-{rc}.sh"));
        std::fs::write(&p, format!("echo building\nexit {rc}\n")).expect("w");
        p.to_string_lossy().into_owned()
    }

    fn summary_of(lean: &Path) -> summary::Summary {
        summary::load(&summary::summary_path(lean)).expect("a summary was written")
    }

    fn lc() -> Leanchecker {
        LC.expect("LC")
    }

    /// The accept: every step green → a green summary deriving the tree's theorem, and the untracked log beside it.
    #[test]
    fn a_green_run_writes_a_green_summary_and_the_log() {
        let (d, lean, contracts) = tree();
        gen(&lean, &contracts, false).expect("gen-axioms");
        let lake = run_lake(d.path(), &lean, 0);
        let r = run_all(lk(&lake), &build_sh(d.path(), 0), &lean, &contracts, lc());
        let s = summary_of(&lean);
        assert!(r.is_ok(), "{r:?} {s:?}");
        assert!(s.is_green(), "{s:?}");
        assert_eq!(s.challenges_closed.as_deref(), Some("1/1"));
        assert!(s.derived().contains("ProvableContracts.Gelu.gelu_bound"));
        assert!(s.tree_sha.is_none(), "a tempdir is no git checkout");
        let log: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(lean.join(LOG_FILE)).expect("log"))
                .expect("json");
        assert_eq!(log["verdict"], "accept");
        assert_eq!(log["summary"]["leanchecker_exit"], 0);
    }

    /// The spec's mutation: a summary hand-edited toward pass differs from its regeneration, byte for byte.
    #[test]
    fn a_regeneration_reproduces_the_summary_and_catches_a_hand_edit() {
        let (d, lean, contracts) = tree();
        gen(&lean, &contracts, false).expect("gen-axioms");
        let lake = run_lake(d.path(), &lean, 1);
        let build = build_sh(d.path(), 0);
        assert!(is_reject(&run_all(
            lk(&lake),
            &build,
            &lean,
            &contracts,
            lc()
        )));
        let spath = summary::summary_path(&lean);
        let first = std::fs::read_to_string(&spath).expect("r");
        assert!(first.contains("\"leanchecker_exit\": 1"), "{first}");
        let edited = first.replace("\"leanchecker_exit\": 1", "\"leanchecker_exit\": 0");
        std::fs::write(&spath, &edited).expect("w");
        assert!(summary_of(&lean).is_green(), "the edit alone would derive");
        assert!(is_reject(&run_all(
            lk(&lake),
            &build,
            &lean,
            &contracts,
            lc()
        )));
        let again = std::fs::read_to_string(&spath).expect("r");
        assert_eq!(
            again, first,
            "the regeneration is byte-identical to the honest run"
        );
        assert_ne!(
            again, edited,
            "and so differs from the edit: git diff --exit-code fails"
        );
    }

    /// A tree that did not build runs no Lean step, and still leaves a summary saying so.
    #[test]
    fn a_failed_build_rejects_a_cache_miss_declines_and_both_leave_a_red_summary() {
        let (d, lean, contracts) = tree();
        gen(&lean, &contracts, false).expect("gen-axioms");
        let lake = run_lake(d.path(), &lean, 0);
        for (rc, want_reject) in [(1, true), (2, false)] {
            let r = run_all(lk(&lake), &build_sh(d.path(), rc), &lean, &contracts, lc());
            assert_eq!(is_reject(&r), want_reject, "build rc {rc}: {r:?}");
            assert_eq!(is_decline(&r), !want_reject, "build rc {rc}: {r:?}");
            let s = summary_of(&lean);
            assert_eq!(s.build_exit, Some(rc));
            assert_eq!(
                (s.lake_exit, s.leanchecker_exit),
                (None, None),
                "no Lean step ran"
            );
            assert!(!s.is_green() && s.derived().is_empty());
            assert!(lean.join(LOG_FILE).is_file());
        }
    }

    /// A summary that cannot be written is the run's output missing: reject.
    #[test]
    fn an_unwritable_summary_rejects() {
        let (d, lean, contracts) = tree();
        gen(&lean, &contracts, false).expect("gen-axioms");
        std::fs::create_dir_all(summary::summary_path(&lean)).expect("a dir where the file goes");
        let lake = run_lake(d.path(), &lean, 0);
        assert!(is_reject(&run_all(
            lk(&lake),
            &build_sh(d.path(), 0),
            &lean,
            &contracts,
            lc()
        )));
        // The log is written after the summary, so it carries the failed write and the reject (EV-8a quorum).
        let log = std::fs::read_to_string(lean.join(LOG_FILE)).expect("the log is still written");
        assert!(log.contains("\"verdict\": \"reject\""), "{log}");
        assert!(log.contains("cannot write"), "{log}");
    }

    /// A `lake` whose `env <step>` call runs `body` (every other call passes); `$PIDF` is a file for a pid.
    fn hang_lake(dir: &Path, step: &str, body: &str) -> (String, PathBuf) {
        let pidf = dir.join(format!("grandchild-{step}"));
        let p = dir.join(format!("hang-lake-{step}"));
        let script = format!(
            "#!/bin/sh\nPIDF='{}'\nif [ \"$2\" = {step} ]; then\n  {body}\nfi\nexit 0\n",
            pidf.display()
        );
        std::fs::write(&p, script).expect("w");
        let mut perm = std::fs::metadata(&p).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&p, perm).expect("chmod");
        (p.to_string_lossy().into_owned(), pidf)
    }

    /// Whether `pid` is gone (or a zombie awaiting its reaper), polled for up to 5 s.
    fn gone(pid: &str) -> bool {
        (0..250).any(|_| {
            let alive = std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|st| {
                st.rsplit(')')
                    .next()
                    .is_some_and(|t| !t.trim_start().starts_with('Z'))
            });
            if alive {
                std::thread::sleep(Duration::from_millis(20));
            }
            !alive
        })
    }

    /// #4239: a hung `lake env lean Axioms.lean` rejects with a named reason inside its bound, records 124, and
    /// takes the `lean` under it (a grandchild in its process group) down with it.
    #[test]
    fn a_hung_elaboration_rejects_within_its_bound_and_kills_its_process_group() {
        let (d, lean, _) = tree();
        let (lake, pidf) = hang_lake(d.path(), "lean", "sleep 60 & echo $! > \"$PIDF\"; wait");
        let mut r = Report::default();
        let t0 = Instant::now();
        lean_steps(
            Lake {
                bin: &lake,
                timeout_s: 1,
            },
            &mut r,
            &lean,
            false,
            false,
            None,
        );
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "the bound held: {:?}",
            t0.elapsed()
        );
        assert!(
            r.reject && r.decline.is_none(),
            "a hang is RED, not a decline: {:?}",
            r.lines
        );
        assert_eq!(r.lake_exit, Some(TIMED_OUT));
        assert!(
            r.lines
                .iter()
                .any(|l| l.contains("lake env lean Axioms.lean timed out after 1s")),
            "{:?}",
            r.lines
        );
        let pid = std::fs::read_to_string(&pidf).expect("the fake recorded its grandchild");
        assert!(
            gone(pid.trim()),
            "grandchild {} outlived the kill",
            pid.trim()
        );
    }

    /// #4239: the comparator and the leanchecker arm's `printenv LEAN_SYSROOT` are bounded too, and a child that
    /// exits while a grandchild holds its stdout cannot stall the drain past the bound.
    #[test]
    fn every_lake_call_is_bounded_and_a_held_pipe_cannot_stall_it() {
        let (d, lean, _) = tree();
        std::fs::create_dir_all(lean.join(CHALLENGE_DIR)).expect("mkdir");
        std::fs::write(
            lean.join(CHALLENGE_DIR).join("gelu-v1.lean"),
            GELU_CHALLENGE,
        )
        .expect("w");
        std::fs::create_dir_all(lean.join("scripts")).expect("mkdir");
        std::fs::write(lean.join(COMPARATOR), "").expect("w");
        let bounded = |lake: &str, cmp: bool, lc: Option<Leanchecker>| {
            let mut r = Report::default();
            let t0 = Instant::now();
            lean_steps(
                Lake {
                    bin: lake,
                    timeout_s: 1,
                },
                &mut r,
                &lean,
                lc.is_some(),
                cmp,
                lc,
            );
            assert!(t0.elapsed() < Duration::from_secs(20), "{:?}", t0.elapsed());
            r
        };
        // `env lean --run …`: $2 is `lean` for elaboration too, so hang only on `--run`.
        let (cmp, _) = hang_lake(
            d.path(),
            "lean",
            &format!("{SELF_TEST_OK}; [ \"$3\" = --run ] && sleep 60"),
        );
        let r = bounded(&cmp, true, None);
        assert!(r.reject && r.decline.is_none(), "{:?}", r.lines);
        assert!(
            r.lines
                .iter()
                .any(|l| l.contains("file(s)) timed out after 1s")),
            "{:?}",
            r.lines
        );
        let (sys, _) = hang_lake(d.path(), "printenv", "sleep 60");
        let r = bounded(&sys, false, LC);
        assert!(r.reject && r.decline.is_none(), "{:?}", r.lines);
        assert!(
            r.lines
                .iter()
                .any(|l| l.contains("printenv LEAN_SYSROOT timed out")),
            "{:?}",
            r.lines
        );
        assert_eq!(r.leanchecker_exit, None, "leanchecker never ran");
        // `lake` exits 0 at once, but a background child keeps stdout open: still bounded, still RED.
        let (held, pidf) = hang_lake(d.path(), "lean", "sleep 60 & echo $! > \"$PIDF\"; exit 0");
        let r = bounded(&held, false, None);
        assert!(r.reject, "{:?}", r.lines);
        let pid = std::fs::read_to_string(&pidf).expect("pid");
        assert!(
            gone(pid.trim()),
            "pipe holder {} outlived the kill",
            pid.trim()
        );
    }

    /// #4238: the comparator's sha256 is checked against the FIPS vectors before any row is judged. A failing
    /// self-test, a hung one, or one that exits 0 having checked fewer than three vectors rejects, and the rows
    /// never run; a passing one is recorded first.
    #[test]
    fn the_comparator_self_test_gates_the_rows() {
        let (d, lean, _) = tree();
        let h = "ab".repeat(32);
        let lake = comparator_lake(
            d.path(),
            &lean,
            0,
            &cmp_row(
                "ProvableContracts.Gelu.gelu_bound",
                &h,
                &format!("\"{h}\""),
                "[]",
            ),
            "",
        );
        let r = compared(&lake, &lean);
        let st = r
            .lines
            .iter()
            .position(|l| l.contains("--self-test (3 FIPS vectors)"));
        let rows = r.lines.iter().position(|l| l.contains("file(s))"));
        assert!(st.is_some() && st < rows, "self-test first: {:?}", r.lines);
        let body = std::fs::read_to_string(&lake).expect("r");
        for (name, stub) in [
            ("fails", "printf 'FAIL  sha256 \\\"abc\\\" = 00\\n'; exit 1"),
            (
                "two-vectors",
                "printf 'ok    sha256 a\\nok    sha256 b\\n'; exit 0",
            ),
            ("silent", "exit 0"),
            ("hangs", "sleep 60"),
        ] {
            let p = d.path().join(format!("self-test-{name}"));
            std::fs::write(
                &p,
                body.replace(
                    SELF_TEST_OK,
                    &format!("if [ \"$5\" = --self-test ]; then {stub}; fi"),
                ),
            )
            .expect("w");
            let mut perm = std::fs::metadata(&p).expect("meta").permissions();
            std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
            std::fs::set_permissions(&p, perm).expect("chmod");
            let mut r = Report::default();
            lean_steps(
                Lake {
                    bin: &p.to_string_lossy(),
                    timeout_s: 2,
                },
                &mut r,
                &lean,
                false,
                true,
                None,
            );
            assert!(r.reject && r.decline.is_none(), "{name}: {:?}", r.lines);
            assert!(r.challenges.is_none(), "{name}: rows judged: {:?}", r.lines);
            assert!(
                r.lines
                    .iter()
                    .any(|l| l.starts_with("FAIL") && l.contains("--self-test")),
                "{name}: {:?}",
                r.lines
            );
        }
    }
}
