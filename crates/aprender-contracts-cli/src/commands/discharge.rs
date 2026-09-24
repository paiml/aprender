//! `pv discharge gen-axioms | check | run` (PVL-001 EV-6a, #4139; `--leanchecker` EV-6b, #4199; `--comparator`
//! EV-7b, #4201; `run` EV-8a, #4202). The judging lives in
//! [`provable_contracts::discharge`]; this module prints the report and runs Lean.
//!
//! Exit: 0 accept · 1 reject (`reject:`) · 2 decline (`decline:` — no root file, zero roots, no `lake`, no
//! `leanchecker` in the toolchain, no `Challenge/*.lean` or zero comparator rows).
//!
//! `--leanchecker` is NON-fresh: it re-checks the tree's own .olean files and trusts the Mathlib .oleans they
//! import. `--fresh` replays Mathlib and is the nightly's (PVL-F7). `formalization.yaml` `scope` says so.

use std::fmt;
use std::path::Path;
use std::process::Command;

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
            leanchecker,
            leanchecker_timeout,
            leanchecker_ulimit_v,
            comparator,
            validate_formalization,
        } => {
            let mut r = discharge::check(&lean_dir, &contracts, CheckOpts { strict });
            if validate_formalization && r.decline.is_none() {
                judge_formalization(&lean_dir, &mut r);
            }
            let lc = leanchecker.then_some(Leanchecker {
                timeout_s: leanchecker_timeout,
                ulimit_v_kib: leanchecker_ulimit_v,
            });
            finish_with("lake", r, &lean_dir, no_lake, comparator, lc)
        }
        DischargeAction::Run {
            lean_dir,
            contracts,
            leanchecker_timeout,
            leanchecker_ulimit_v,
        } => run_all(
            "lake",
            "build.sh",
            &lean_dir,
            &contracts,
            Leanchecker {
                timeout_s: leanchecker_timeout,
                ulimit_v_kib: leanchecker_ulimit_v,
            },
        ),
        DischargeAction::LabelRatchet {
            lean_dir,
            contracts,
        } => finish_with(
            "lake",
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

/// How `--leanchecker` runs (PVL-001 EV-6b, #4199).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leanchecker {
    pub timeout_s: u64,
    pub ulimit_v_kib: Option<u64>,
}

/// The Lean steps run only on a tree nothing else has already failed or declined: a 60-minute re-check of a tree
/// that is already RED would only delay the verdict. They record their raw exits in `r.lake_exit` and
/// `r.leanchecker_exit`, and the comparator's closure in `r.challenges` (`None` = never ran), for
/// `discharge-summary.json` (EV-8a). The comparator runs before the (much slower) leanchecker.
pub(crate) fn lean_steps(
    lake: &str,
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

/// `--validate-formalization` (PVL-001 EV-8b): each inconsistency is a FAIL line and rejects.
fn judge_formalization(lean_dir: &Path, r: &mut Report) {
    let (tree, allow) = match (Tree::load(lean_dir), discharge::load_allowlist(lean_dir)) {
        (Ok(t), Ok(a)) => (t, a),
        (Err(e), _) | (_, Err(e)) => {
            r.decline = Some(e);
            return;
        }
    };
    let sum = summary::load(&summary::summary_path(lean_dir)).ok();
    let bad = discharge::validate_formalization(lean_dir, &tree, &allow, sum.as_ref());
    if bad.is_empty() {
        r.lines.push("FORMALIZATION ok".into());
    }
    for b in bad {
        r.lines.push(format!("FAIL formalization: {b}"));
        r.reject = true;
    }
}

fn finish_with(
    lake: &str,
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
    lake: &str,
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
    let mut r = discharge::check(lean_dir, contracts, CheckOpts { strict: true });
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
    let s = summary::summarize(&r, tree.as_ref(), lean_dir, tree_sha(lean_dir), build_exit);
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

/// `git rev-parse HEAD:<lean-dir>`, content-addressed: the tree the summary describes. `None` outside a git
/// checkout, or when the dir is not in HEAD.
fn tree_sha(lean_dir: &Path) -> Option<String> {
    let o = Command::new("git")
        .args(["rev-parse", "HEAD:./"])
        .current_dir(lean_dir)
        .output()
        .ok()?;
    let sha = String::from_utf8_lossy(&o.stdout).trim().to_string();
    (o.status.success() && !sha.is_empty()).then_some(sha)
}

/// `lake env lean Axioms.lean`: the subset and capstone pins, elaborated against the BUILT tree (run `build.sh`
/// first; `lake env` builds nothing).
fn elaborate(lake: &str, lean_dir: &Path, r: &mut Report) {
    match Command::new(lake)
        .args(["env", "lean", AXIOMS_FILE])
        .current_dir(lean_dir)
        .output()
    {
        Err(e) => {
            r.decline = Some(format!(
                "lake could not be run ({e}): Axioms.lean was not elaborated"
            ))
        }
        Ok(o) if o.status.success() => {
            r.lake_exit = Some(raw_exit(o.status));
            r.lines.push(format!("ok    lake env lean {AXIOMS_FILE}"));
        }
        Ok(o) => {
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
fn compare(lake: &str, lean_dir: &Path, r: &mut Report) {
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
    let what = format!("lake env lean --run {COMPARATOR} ({} file(s))", files.len());
    let out = match Command::new(lake)
        .args(["env", "lean", "--run", COMPARATOR])
        .args(&files)
        .current_dir(lean_dir)
        .output()
    {
        Ok(o) => o,
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
            r.challenges = Some(comparator::judge_rows(&rows, r));
        }
        Err(e) => {
            r.lines.push(format!("FAIL  {what}: {e}"));
            r.reject = true;
        }
    }
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

/// The toolchain `lake env` puts on PATH for this tree: `lake env printenv LEAN_SYSROOT`. `Err` is the decline.
fn sysroot(lake: &str, lean_dir: &Path) -> Result<std::path::PathBuf, String> {
    let o = Command::new(lake)
        .args(["env", "printenv", "LEAN_SYSROOT"])
        .current_dir(lean_dir)
        .output()
        .map_err(|e| format!("lake could not be run ({e}): leanchecker did not run"))?;
    let root = String::from_utf8_lossy(&o.stdout).trim().to_string();
    if !o.status.success() || root.is_empty() {
        return Err(format!(
            "`lake env printenv LEAN_SYSROOT` exited {:?} and named no toolchain: leanchecker did not run",
            o.status.code()
        ));
    }
    Ok(root.into())
}

/// `timeout <T> lake env leanchecker ProvableContracts` (PVL-001 EV-6b): rc != 0 rejects, a timeout rejects, and a
/// toolchain without `leanchecker` declines -- measured: elan's `lake env leanchecker` on v4.15.0 exits 1 "does not
/// have the binary", which would otherwise read as a failed check. `timeout` signals the whole process group, so
/// the `leanchecker` under `lake` does not outlive it.
fn recheck(lake: &str, lean_dir: &Path, lc: Leanchecker, r: &mut Report) {
    let root = match sysroot(lake, lean_dir) {
        Ok(root) => root,
        Err(why) => {
            r.decline = Some(why);
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
    let limit = lc.ulimit_v_kib.map(|k| k.to_string()).unwrap_or_default();
    let out = Command::new("sh")
        .args([
            "-c",
            r#"if [ -n "$1" ]; then ulimit -v "$1" || exit 125; fi; exec timeout -k 30 "$2" "$3" env leanchecker ProvableContracts"#,
            "pv-leanchecker",
            &limit,
            &lc.timeout_s.to_string(),
            lake,
        ])
        .current_dir(lean_dir)
        .output();
    let what = format!(
        "lake env leanchecker ProvableContracts (timeout {}s)",
        lc.timeout_s
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
            leanchecker: false,
            leanchecker_timeout: 3600,
            leanchecker_ulimit_v: None,
            comparator: false,
            validate_formalization: false,
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
        finish_with(&ok, Report::default(), &lean, false, false, None).expect("lake ok");
        assert!(
            is_reject(&finish_with(
                &bad,
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
                "/nonexistent/lake",
                Report::default(),
                &lean,
                false,
                false,
                None
            )),
            "no lake declines"
        );
        finish_with(&bad, Report::default(), &lean, true, false, None).expect("--no-lake skips it");
        let rejected = Report {
            reject: true,
            ..Report::default()
        };
        assert!(
            is_reject(&finish_with(&ok, rejected, &lean, false, false, None)),
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
        finish_with(&pass, Report::default(), &lean, false, false, LC)
            .expect("leanchecker rc 0 accepts");
        assert!(
            is_reject(&finish_with(
                &fail,
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
                &absent,
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
                "/nonexistent/lake",
                Report::default(),
                &lean,
                true,
                false,
                LC
            )),
            "no lake declines under --leanchecker"
        );
        finish_with(&fail, Report::default(), &lean, false, false, None)
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
        });
        assert!(
            is_reject(&finish_with(
                &slow,
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
            &shows,
            &lean,
            Leanchecker {
                timeout_s: 60,
                ulimit_v_kib: Some(4_194_304),
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

    /// EV-8a reads the raw exits: `Some(n)` for a step that ran (124 on a timeout), `None` for one that never ran.
    #[test]
    fn the_lean_steps_record_their_raw_exits_and_none_when_they_did_not_run() {
        let (d, lean, _) = tree();
        let exits = |lake: &str, no_lake: bool, lc: Option<Leanchecker>| {
            let mut r = Report::default();
            lean_steps(lake, &mut r, &lean, no_lake, false, lc);
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
        });
        assert_eq!(exits(&slow, true, t1), (None, Some(124)), "timeout");
    }

    /// A `lake` for `--comparator`: `env lean --run …` writes `rows` to stdout, `stderr` to stderr, and exits
    /// `rc`; every other `env lean` passes. The tree gets `Challenge/gelu-v1.lean` and the comparator script.
    fn comparator_lake(dir: &Path, lean: &Path, rc: i32, rows: &str, stderr: &str) -> String {
        std::fs::create_dir_all(lean.join(CHALLENGE_DIR)).expect("mkdir");
        std::fs::write(lean.join(CHALLENGE_DIR).join("gelu-v1.lean"), "").expect("w");
        std::fs::create_dir_all(lean.join("scripts")).expect("mkdir");
        std::fs::write(lean.join(COMPARATOR), "").expect("w");
        std::fs::write(dir.join(format!("rows-{rc}")), rows).expect("w");
        let p = dir.join(format!("cmp-lake-{rc}"));
        let script = format!(
            "#!/bin/sh\nif [ \"$3\" = --run ]; then\n  [ \"$4 $5\" = \"{COMPARATOR} {CHALLENGE_DIR}/gelu-v1.lean\" ] || {{ echo \"bad args: $*\" >&2; exit 98; }}\n  cat '{}'; echo '{stderr}' >&2; exit {rc}\nfi\nexit 0\n",
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
        lean_steps(lake, &mut r, lean, false, true, None);
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
        finish_with(&lake, Report::default(), &lean, false, true, None).expect("accepts");
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
            &lake,
            Report::default(),
            &lean,
            false,
            true,
            None
        )));

        let (d2, lean2, _) = tree();
        comparator_lake(d2.path(), &lean2, 0, "", "");
        let mut r = Report::default();
        lean_steps("/nonexistent/lake", &mut r, &lean2, true, true, None);
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
        lean_steps(&lake, &mut r, &lean, false, false, None);
        assert!(!r.reject && r.challenges.is_none(), "{:?}", r.lines);
    }

    /// PVL-001 EV-8a: a `lake` for `run` — every `env lean` passes, `--run` prints one closing comparator row,
    /// `printenv` names a sysroot with a leanchecker, and `env leanchecker` exits `lc_rc`.
    fn run_lake(dir: &Path, lean: &Path, lc_rc: i32) -> String {
        std::fs::create_dir_all(lean.join(CHALLENGE_DIR)).expect("mkdir");
        std::fs::write(lean.join(CHALLENGE_DIR).join("gelu-v1.lean"), "").expect("w");
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
            "#!/bin/sh\ncase \"$2 $3\" in\n  'lean --run') cat '{}' ;;\n  printenv*) echo '{}' ;;\n  leanchecker*) exit {lc_rc} ;;\nesac\nexit 0\n",
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
        let r = run_all(&lake, &build_sh(d.path(), 0), &lean, &contracts, lc());
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
        assert!(is_reject(&run_all(&lake, &build, &lean, &contracts, lc())));
        let spath = summary::summary_path(&lean);
        let first = std::fs::read_to_string(&spath).expect("r");
        assert!(first.contains("\"leanchecker_exit\": 1"), "{first}");
        let edited = first.replace("\"leanchecker_exit\": 1", "\"leanchecker_exit\": 0");
        std::fs::write(&spath, &edited).expect("w");
        assert!(summary_of(&lean).is_green(), "the edit alone would derive");
        assert!(is_reject(&run_all(&lake, &build, &lean, &contracts, lc())));
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
            let r = run_all(&lake, &build_sh(d.path(), rc), &lean, &contracts, lc());
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
            &lake,
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
}
