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
            update_baseline,
        } => {
            let r = discharge::check(
                &lean_dir,
                &contracts,
                CheckOpts {
                    strict,
                    update_baseline,
                },
            );
            finish(r, &lean_dir, no_lake)
        }
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

fn finish(mut r: Report, lean_dir: &Path, no_lake: bool) -> Res {
    if !r.reject && r.decline.is_none() && !no_lake {
        elaborate(lean_dir, &mut r);
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
    println!("ok    discharge check {}", lean_dir.display());
    Ok(())
}

/// `lake env lean Axioms.lean`: the subset and capstone pins, elaborated against the BUILT tree (run `build.sh`
/// first; `lake env` builds nothing).
fn elaborate(lean_dir: &Path, r: &mut Report) {
    match Command::new("lake")
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
