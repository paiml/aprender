//! `pv discharge gen-axioms | check` (PVL-001 EV-6a, #4139; `--leanchecker` EV-6b, #4199). The judging lives in
//! [`provable_contracts::discharge`]; this module prints the report and runs Lean.
//!
//! Exit: 0 accept · 1 reject (`reject:`) · 2 decline (`decline:` — no root file, zero roots, no `lake`, no
//! `leanchecker` in the toolchain).
//!
//! `--leanchecker` is NON-fresh: it re-checks the tree's own .olean files and trusts the Mathlib .oleans they
//! import. `--fresh` replays Mathlib and is the nightly's (PVL-F7). `formalization.yaml` `scope` says so.

use std::fmt;
use std::path::Path;
use std::process::Command;

use provable_contracts::discharge::{self, CheckOpts, Report, AXIOMS_FILE};

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
        } => {
            let r = discharge::check(&lean_dir, &contracts, CheckOpts { strict });
            let lc = leanchecker.then_some(Leanchecker {
                timeout_s: leanchecker_timeout,
                ulimit_v_kib: leanchecker_ulimit_v,
            });
            finish_with("lake", r, &lean_dir, no_lake, lc)
        }
        DischargeAction::LabelRatchet {
            lean_dir,
            contracts,
        } => finish_with(
            "lake",
            discharge::ratchet_labels(&lean_dir, &contracts),
            &lean_dir,
            true,
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
/// that is already RED would only delay the verdict.
fn finish_with(
    lake: &str,
    mut r: Report,
    lean_dir: &Path,
    no_lake: bool,
    lc: Option<Leanchecker>,
) -> Res {
    let open = |r: &Report| !r.reject && r.decline.is_none();
    if open(&r) && !no_lake {
        elaborate(lake, lean_dir, &mut r);
    }
    if let Some(lc) = lc {
        if open(&r) {
            recheck(lake, lean_dir, lc, &mut r);
        }
    }
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
        Ok(o) if o.status.success() => r.lines.push(format!("ok    lake env lean {AXIOMS_FILE}")),
        Ok(o) => {
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
        Ok(o) if o.status.success() => r.lines.push(format!("ok    {what}")),
        Ok(o) => match o.status.code() {
            Some(c @ 125..=127) => {
                r.decline = Some(format!(
                    "{what} could not be started (rc {c}: timeout/ulimit/lake): not a verdict"
                ));
            }
            code => {
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
        finish_with(&ok, Report::default(), &lean, false, None).expect("lake ok");
        assert!(
            is_reject(&finish_with(&bad, Report::default(), &lean, false, None)),
            "a failing elaboration rejects"
        );
        assert!(
            is_decline(&finish_with(
                "/nonexistent/lake",
                Report::default(),
                &lean,
                false,
                None
            )),
            "no lake declines"
        );
        finish_with(&bad, Report::default(), &lean, true, None).expect("--no-lake skips it");
        let rejected = Report {
            reject: true,
            ..Report::default()
        };
        assert!(
            is_reject(&finish_with(&ok, rejected, &lean, false, None)),
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
        finish_with(&pass, Report::default(), &lean, false, LC).expect("leanchecker rc 0 accepts");
        assert!(
            is_reject(&finish_with(&fail, Report::default(), &lean, false, LC)),
            "leanchecker rc 1 rejects"
        );
        assert!(
            is_decline(&finish_with(&absent, Report::default(), &lean, false, LC)),
            "no leanchecker in the toolchain declines"
        );
        assert!(
            is_decline(&finish_with(
                "/nonexistent/lake",
                Report::default(),
                &lean,
                true,
                LC
            )),
            "no lake declines under --leanchecker"
        );
        finish_with(&fail, Report::default(), &lean, false, None)
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
            is_reject(&finish_with(&slow, Report::default(), &lean, true, t1)),
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
}
