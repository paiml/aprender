//! `pv discharge gen-axioms | check` (PVL-001 EV-6a, #4139). The judging lives in
//! [`provable_contracts::discharge`]; this module prints the report and runs Lean.
//!
//! Exit: 0 accept · 1 reject (`reject:`) · 2 decline (`decline:` — no root file, zero roots, no `lake`).

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
        } => {
            let r = discharge::check(&lean_dir, &contracts, CheckOpts { strict });
            finish(r, &lean_dir, no_lake)
        }
        DischargeAction::LabelRatchet {
            lean_dir,
            contracts,
        } => finish(
            discharge::ratchet_labels(&lean_dir, &contracts),
            &lean_dir,
            true,
        ),
    }
}

fn gen_axioms(lean_dir: &Path, contracts: &Path, check: bool) -> Res {
    let (text, _, b) = discharge::generate(lean_dir, contracts).map_err(DischargeDeclined)?;
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

fn finish(r: Report, lean_dir: &Path, no_lake: bool) -> Res {
    finish_with("lake", r, lean_dir, no_lake)
}

fn finish_with(lake: &str, mut r: Report, lean_dir: &Path, no_lake: bool) -> Res {
    if !r.reject && r.decline.is_none() && !no_lake {
        elaborate(lake, lean_dir, &mut r);
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
        finish_with(&ok, Report::default(), &lean, false).expect("lake ok");
        assert!(
            is_reject(&finish_with(&bad, Report::default(), &lean, false)),
            "a failing elaboration rejects"
        );
        assert!(
            is_decline(&finish_with(
                "/nonexistent/lake",
                Report::default(),
                &lean,
                false
            )),
            "no lake declines"
        );
        finish_with(&bad, Report::default(), &lean, true).expect("--no-lake skips it");
        let rejected = Report {
            reject: true,
            ..Report::default()
        };
        assert!(
            is_reject(&finish_with(&ok, rejected, &lean, false)),
            "a prior failure is not cleared by lake"
        );
    }
}
